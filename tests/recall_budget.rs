//! `WITH RECALL` is the statement's target: several lossy `MEANS` filters
//! share it rather than each promising all of it.

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use semcast::model::MockModel;
use semcast::{IndexOptions, SemcastContextBuilder, create_semantic_index};

/// tickets(id, subject, body), with a semantic index on `body` only.
async fn context(scorer: bool) -> SessionContext {
    let model = Arc::new(MockModel::answering_yes_to(["refund"]));
    let mut builder = SemcastContextBuilder::new(model);
    if scorer {
        builder = builder.with_scorer(Arc::new(
            MockModel::answering_yes_to(["refund"]).with_id("scorer"),
        ));
    }
    let ctx = builder.build();
    ctx.sql(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES
             (1, 'refund please', 'i want a refund for my order'),
             (2, 'hello', 'just saying hi')
         ) AS t(id, subject, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "tickets", "body", IndexOptions::default())
        .await
        .unwrap();
    ctx.sql("CREATE TABLE notes AS SELECT * FROM (VALUES (1, 'order 7')) AS t(id, body)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    create_semantic_index(&ctx, "notes", "body", IndexOptions::default())
        .await
        .unwrap();
    ctx
}

async fn explain(ctx: &SessionContext, sql: &str) -> String {
    let batches = semcast::sql(ctx, &format!("EXPLAIN {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

#[tokio::test]
async fn two_indexed_filters_split_the_target() {
    let ctx = context(false).await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
         AND body MEANS 'is angry' WITH RECALL 0.9",
    )
    .await;
    assert_eq!(
        plan.matches(
            "recall ≥ 0.95 estimated (its share of the statement's 0.90 across 2 filters)"
        )
        .count(),
        2,
        "{plan}"
    );
    assert!(
        plan.contains("recall≥0.95"),
        "the scans calibrate to the share: {plan}"
    );
}

#[tokio::test]
async fn confidence_is_split_too() {
    let ctx = context(false).await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
         AND body MEANS 'is angry' WITH RECALL 0.9 WITH CONFIDENCE 0.9",
    )
    .await;
    assert_eq!(
        plan.matches("recall ≥ 0.95 certified at 0.95").count(),
        2,
        "{plan}"
    );
}

#[tokio::test]
async fn an_exact_filter_takes_no_share() {
    // `subject` has no index and there is no scorer: verified on full text,
    // it loses nothing, so the indexed filter keeps the whole target.
    let ctx = context(false).await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
         AND subject MEANS 'is about money' WITH RECALL 0.9",
    )
    .await;
    assert!(!plan.contains("its share"), "{plan}");
    assert!(plan.contains("recall≥0.90"), "{plan}");
}

#[tokio::test]
async fn a_scorer_makes_an_unindexed_filter_lossy() {
    let ctx = context(true).await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
         AND subject MEANS 'is about money' WITH RECALL 0.9",
    )
    .await;
    assert_eq!(plan.matches("across 2 filters").count(), 2, "{plan}");
}

#[tokio::test]
async fn one_lossy_filter_keeps_the_statement_target() {
    let ctx = context(false).await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' WITH RECALL 0.9",
    )
    .await;
    assert!(plan.contains("recall ≥ 0.90 estimated"), "{plan}");
    assert!(!plan.contains("its share"), "{plan}");
}

#[tokio::test]
async fn the_split_query_still_runs() {
    let ctx = context(false).await;
    let batches = semcast::sql(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
         AND body MEANS 'mentions an order' WITH RECALL 0.9",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let out = pretty_format_batches(&batches).unwrap().to_string();
    assert!(out.contains("| 1  |"), "{out}");
}

#[tokio::test]
async fn a_filter_in_a_subquery_takes_its_share() {
    let ctx = context(false).await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
         AND id IN (SELECT id FROM notes WHERE body MEANS 'mentions an order') \
         WITH RECALL 0.9",
    )
    .await;
    assert!(!plan.contains("best-effort"), "{plan}");
    assert_eq!(plan.matches("across 2 filters").count(), 2, "{plan}");
}

#[tokio::test]
async fn a_semantic_column_filter_takes_its_share() {
    let ctx = context(false).await;
    semcast::sql(
        &ctx,
        "ALTER TABLE tickets ADD SEMANTIC COLUMN angry AS (body MEANS 'is angry')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' AND angry WITH RECALL 0.9",
    )
    .await;
    assert!(!plan.contains("best-effort"), "{plan}");
    assert_eq!(plan.matches("across 2 filters").count(), 2, "{plan}");
}

#[tokio::test]
async fn the_physical_plan_prints_the_share_unrounded() {
    let ctx = context(false).await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
         AND body MEANS 'is angry' WITH RECALL 0.999 WITH CONFIDENCE 0.99",
    )
    .await;
    assert!(plan.contains("recall≥0.9995"), "{plan}");
    assert!(!plan.contains("recall≥1.00"), "{plan}");
}

#[tokio::test]
async fn a_view_cannot_carry_a_recall_target() {
    let ctx = context(true).await;
    let error = semcast::sql(
        &ctx,
        "CREATE VIEW orders AS SELECT * FROM tickets \
         WHERE body MEANS 'mentions an order' WITH RECALL 0.8",
    )
    .await
    .err()
    .expect("refused");
    assert!(
        error.to_string().contains("a view cannot carry it"),
        "{error}"
    );
}

#[tokio::test]
async fn a_view_read_without_a_target_stays_best_effort() {
    let ctx = context(true).await;
    semcast::sql(
        &ctx,
        "CREATE VIEW orders AS SELECT * FROM tickets WHERE body MEANS 'mentions an order'",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let plan = explain(
        &ctx,
        "SELECT id FROM orders WHERE subject MEANS 'about money'",
    )
    .await;
    assert!(!plan.contains("its share"), "{plan}");
    assert!(
        !plan.contains("ScoreScanExec"),
        "no target, no scorer: {plan}"
    );
}

#[tokio::test]
async fn a_view_read_under_a_target_takes_its_share() {
    let ctx = context(false).await;
    semcast::sql(
        &ctx,
        "CREATE VIEW orders AS SELECT * FROM tickets WHERE body MEANS 'mentions an order'",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let plan = explain(
        &ctx,
        "SELECT id FROM orders WHERE body MEANS 'asks for a refund' WITH RECALL 0.9",
    )
    .await;
    assert_eq!(plan.matches("across 2 filters").count(), 2, "{plan}");
}
