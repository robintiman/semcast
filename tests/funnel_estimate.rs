//! EXPLAIN prices the funnel before it runs: expected verify calls, labels
//! and list-price cost, from the plan's row estimate and the store's history.

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use semcast::model::MockModel;
use semcast::{IndexOptions, SemcastContextBuilder, create_semantic_index};

async fn context() -> SessionContext {
    let ctx = SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["refund"]))).build();
    let rows: Vec<String> = (1..=200)
        .map(|id| {
            let body = if id % 10 == 0 {
                "please refund my order"
            } else {
                "just saying hi"
            };
            format!("({id}, '{body} {id}')")
        })
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES {}) AS t(id, body)",
        rows.join(", ")
    ))
    .await
    .unwrap()
    .collect()
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
async fn verify_alone_expects_a_call_per_row() {
    let ctx = context().await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund'",
    )
    .await;
    assert!(
        plan.contains("expect, before stored answers, ~200 verify calls"),
        "{plan}"
    );
    // The mock has no price.
    assert!(plan.contains("cost unknown"), "{plan}");
}

#[tokio::test]
async fn a_calibrated_filter_is_priced_from_its_history() {
    let ctx = context().await;
    create_semantic_index(&ctx, "tickets", "body", IndexOptions::default())
        .await
        .unwrap();
    let query = "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' WITH RECALL 0.9";
    let before = explain(&ctx, query).await;
    assert!(before.contains("no pruning history: ceiling"), "{before}");
    assert!(before.contains("+ 64 labels"), "{before}");

    semcast::sql(&ctx, query)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let after = explain(&ctx, query).await;
    assert!(after.contains("reach verify, from history"), "{after}");
}

#[tokio::test]
async fn a_limit_is_noted() {
    let ctx = context().await;
    let plan = explain(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' LIMIT 3",
    )
    .await;
    assert!(plan.contains("fewer under LIMIT 3"), "{plan}");
}
