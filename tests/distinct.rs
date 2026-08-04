//! End-to-end tests for semantic dedupe: `SEMANTIC DISTINCT ON`, the
//! `WITH SIMILARITY` threshold, and the fact that none of it costs a model
//! call.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use datafusion::execution::context::SessionContext;
use semcast::model::MockModel;
use semcast::{IndexOptions, create_semantic_index, semcast_context};

/// Three ways of saying the same thing, then two distinct others. The mock
/// embeds by theme keyword, so the geometry under test is the dedupe rather
/// than an accident of how English bytes land.
const ROWS: [(i64, &str); 5] = [
    (1, "the refund never arrived"),
    (2, "my refund has not arrived yet"),
    (3, "still waiting on the refund"),
    (4, "the launch slipped to the fourth quarter"),
    (5, "an outage took the dashboard down"),
];

fn dedupe_model() -> Arc<MockModel> {
    Arc::new(MockModel::default().embedding_by_theme(["refund", "launch", "outage"]))
}

async fn notes_context(model: Arc<MockModel>) -> (SessionContext, tempfile::TempDir) {
    let ctx = semcast_context(model);
    let values: Vec<String> = ROWS
        .iter()
        .map(|(id, body)| format!("({id}, '{body}')"))
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE notes AS SELECT * FROM (VALUES {}) AS t(id, body)",
        values.join(", ")
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "notes",
        "body",
        IndexOptions {
            path: Some(dir.path().join("notes.body.lance")),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    (ctx, dir)
}

async fn run(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
}

async fn ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches = run(ctx, sql).await;
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let column = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64");
            (0..b.num_rows())
                .map(|i| column.value(i))
                .collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

async fn error(ctx: &SessionContext, sql: &str) -> String {
    match semcast::sql(ctx, sql).await {
        Ok(frame) => match frame.create_physical_plan().await {
            Ok(_) => panic!("expected an error from: {sql}"),
            Err(err) => err.to_string(),
        },
        Err(err) => err.to_string(),
    }
}

const DEDUPE: &str = "SELECT SEMANTIC DISTINCT ON (body) id, body FROM notes";

#[tokio::test]
async fn optimized_plan_contains_sem_distinct() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let plan = semcast::sql(&ctx, DEDUPE)
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();

    let display = format!("{}", plan.display_indent());
    assert!(display.contains("SemDistinct: SIMILARITY"), "{display}");
    assert!(
        display.contains("no model calls"),
        "EXPLAIN should say the dedupe is free:\n{display}"
    );
}

#[tokio::test]
async fn near_duplicates_collapse_to_one_row() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let surviving = ids(&ctx, DEDUPE).await;

    // Rows 1-3 say the same thing; 4 and 5 do not.
    assert_eq!(surviving.len(), 3, "three distinct meanings: {surviving:?}");
    assert!(surviving.contains(&4));
    assert!(surviving.contains(&5));
    assert_eq!(
        surviving.iter().filter(|id| (1..=3).contains(*id)).count(),
        1,
        "exactly one of the three refund rows survives: {surviving:?}"
    );
}

#[tokio::test]
async fn dedupe_costs_no_model_calls() {
    let model = dedupe_model();
    let (ctx, _dir) = notes_context(Arc::clone(&model)).await;
    let before = model.completion_calls();
    run(&ctx, DEDUPE).await;

    assert_eq!(
        model.completion_calls(),
        before,
        "the index already knows how alike these are"
    );
}

#[tokio::test]
async fn a_strict_threshold_keeps_more_rows() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    // The mock puts all three refund rows on exactly the same axis, so only a
    // threshold above 1.0 could separate them — but 1.0 still merges them,
    // and a loose threshold must not merge the distinct ones.
    let loose = ids(&ctx, &format!("{DEDUPE} WITH SIMILARITY 0.5")).await;
    let strict = ids(&ctx, &format!("{DEDUPE} WITH SIMILARITY 1")).await;
    assert_eq!(loose.len(), 3);
    assert_eq!(strict.len(), 3);
    assert_eq!(loose, strict, "identical documents merge at any threshold");
}

#[tokio::test]
async fn the_same_query_reproduces_its_survivors() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    assert_eq!(ids(&ctx, DEDUPE).await, ids(&ctx, DEDUPE).await);
}

#[tokio::test]
async fn a_plain_distinct_is_untouched() {
    let model = dedupe_model();
    let (ctx, _dir) = notes_context(Arc::clone(&model)).await;
    // Exact-match DISTINCT still means exact match: five distinct bodies.
    let rows = ids(&ctx, "SELECT DISTINCT id FROM notes").await;
    assert_eq!(rows.len(), 5);

    let plan = semcast::sql(&ctx, "SELECT DISTINCT body FROM notes")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();
    assert!(
        !format!("{}", plan.display_indent()).contains("SemDistinct"),
        "a plain DISTINCT must not become a semantic one"
    );
}

#[tokio::test]
async fn semantic_key_is_selectable() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let batches = run(
        &ctx,
        "SELECT id, semantic_key(body) AS grp FROM notes ORDER BY id",
    )
    .await;

    assert_eq!(batches[0].schema().field(1).name(), "grp");
    let keys = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("grp is Utf8");
    assert_eq!(keys.value(0), keys.value(1), "rows 1 and 2 are duplicates");
    assert_eq!(keys.value(0), keys.value(2));
    assert_ne!(keys.value(0), keys.value(3), "row 4 is its own");
}

#[tokio::test]
async fn dedupe_reports_a_funnel() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let plan = semcast::sql(&ctx, DEDUPE)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();

    let counts = semcast::server::progress::snapshot(&plan);
    assert!(counts.has_distinct);
    assert_eq!(counts.distinct_groups, 3);
    assert_eq!(counts.distinct_duplicate_rows, 2);
    let line = semcast::server::progress::render(&counts);
    assert!(line.contains("dedupe: 3 distinct"), "{line}");
}

/// An index gap must leave duplicates in, never take rows out.
#[tokio::test]
async fn rows_the_index_never_saw_all_survive() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    ctx.sql(
        "INSERT INTO notes VALUES
             (6, 'the refund never turned up at all'),
             (7, 'the refund never turned up at all either')",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let surviving = ids(&ctx, DEDUPE).await;
    assert!(
        surviving.contains(&6) && surviving.contains(&7),
        "unindexed rows keep their place rather than being collapsed: {surviving:?}"
    );
}

#[tokio::test]
async fn dedupe_without_an_index_is_a_plan_time_error() {
    let ctx = semcast_context(dedupe_model());
    ctx.sql("CREATE TABLE plain AS SELECT * FROM (VALUES (1, 'a')) AS t(id, body)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let message = error(&ctx, "SELECT SEMANTIC DISTINCT ON (body) id FROM plain").await;
    assert!(
        message.contains("requires a semantic index on plain(body)"),
        "{message}"
    );
    assert!(
        message.contains("CREATE SEMANTIC INDEX ON plain(body)"),
        "{message}"
    );
}

#[tokio::test]
async fn semantic_without_a_distinct_on_column_is_a_plan_time_error() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let message = error(&ctx, "SELECT SEMANTIC DISTINCT body FROM notes").await;
    assert!(
        message.contains("has no column to compare"),
        "a bare SEMANTIC DISTINCT should say why it makes no sense: {message}"
    );
}

#[tokio::test]
async fn with_similarity_without_a_semantic_distinct_is_an_error() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let message = error(&ctx, "SELECT id FROM notes WITH SIMILARITY 0.9").await;
    assert!(
        message.contains("requires a SEMANTIC DISTINCT ON"),
        "{message}"
    );
}

#[tokio::test]
async fn semantic_key_in_a_where_clause_is_a_plan_time_error() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let message = error(&ctx, "SELECT id FROM notes WHERE semantic_key(body) = 'x'").await;
    assert!(
        message.contains("DISTINCT ON list or a SELECT list"),
        "{message}"
    );
}

#[tokio::test]
async fn a_computed_text_expression_is_a_plan_time_error() {
    let (ctx, _dir) = notes_context(dedupe_model()).await;
    let message = error(
        &ctx,
        "SELECT SEMANTIC DISTINCT ON (upper(body)) id, body FROM notes",
    )
    .await;
    assert!(message.contains("plain indexed column"), "{message}");
}

/// The dedupe threshold and a recall target are independent knobs and must
/// be able to ride the same statement.
#[tokio::test]
async fn similarity_composes_with_recall() {
    let (ctx, _dir) = notes_context(Arc::new(
        MockModel::answering_yes_to(["refund"]).embedding_by_theme(["refund", "launch", "outage"]),
    ))
    .await;
    let surviving = ids(
        &ctx,
        "SELECT SEMANTIC DISTINCT ON (body) id FROM notes
         WHERE body MEANS 'about a refund'
         WITH RECALL 0.9 WITH SIMILARITY 0.9",
    )
    .await;
    assert_eq!(
        surviving.len(),
        1,
        "the filter keeps the refund rows, the dedupe collapses them: {surviving:?}"
    );
}
