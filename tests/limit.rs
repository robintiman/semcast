//! `LIMIT` stops model calls: a filter asks only until it has enough matches.

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use semcast::model::MockModel;
use semcast::semcast_context;

/// notes(id, body): 200 distinct texts, all in one partition (the exact call
/// counts below assume it: the fetch, and so each wave, is per partition); ids divisible by `every` mention
/// "offline sync", which the mock answers yes to.
async fn notes(model: Arc<MockModel>, every: i64) -> SessionContext {
    let ctx = semcast_context(model);
    ctx.sql(&format!(
        "CREATE TABLE notes AS
         SELECT value AS id,
                CASE WHEN value % {every} = 0 THEN concat('offline sync note ', value)
                     ELSE concat('quiet note ', value) END AS body
         FROM generate_series(1, 200)"
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

async fn run(ctx: &SessionContext, sql: &str) -> String {
    let batches = semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

fn yes_to_sync() -> Arc<MockModel> {
    Arc::new(MockModel::answering_yes_to(["offline sync"]))
}

#[tokio::test]
async fn limit_stops_asking_once_enough_rows_match() {
    let model = yes_to_sync();
    let ctx = notes(Arc::clone(&model), 1).await;
    let out = run(
        &ctx,
        "SELECT id FROM notes WHERE body MEANS 'it is about sync' LIMIT 5",
    )
    .await;
    assert_eq!(out.lines().count(), 4 + 5, "{out}");
    // The first wave, not the 200 rows.
    assert_eq!(
        model.completion_calls(),
        semcast::physical::verify::FIRST_LIMIT_WAVE
    );
}

#[tokio::test]
async fn sparse_matches_ask_in_growing_waves_until_found() {
    let model = yes_to_sync();
    // Matches at 50, 100, 150, 200.
    let ctx = notes(Arc::clone(&model), 50).await;
    let out = run(
        &ctx,
        "SELECT id FROM notes WHERE body MEANS 'it is about sync' LIMIT 1",
    )
    .await;
    assert!(out.contains(" 50 ") && !out.contains(" 100 "), "{out}");
    // A wave of 32 finds nothing, the next 64 reaches row 50: 96, not 200.
    assert_eq!(model.completion_calls(), 32 + 64);
}

#[tokio::test]
async fn a_limit_larger_than_the_matches_returns_them_all() {
    let model = yes_to_sync();
    let ctx = notes(Arc::clone(&model), 50).await;
    let out = run(
        &ctx,
        "SELECT id FROM notes WHERE body MEANS 'it is about sync' LIMIT 10",
    )
    .await;
    for id in [" 50 ", " 100 ", " 150 ", " 200 "] {
        assert!(out.contains(id), "{out}");
    }
    assert_eq!(model.completion_calls(), 200);
}

#[tokio::test]
async fn no_limit_asks_every_row() {
    let model = yes_to_sync();
    let ctx = notes(Arc::clone(&model), 1).await;
    run(
        &ctx,
        "SELECT id FROM notes WHERE body MEANS 'it is about sync'",
    )
    .await;
    assert_eq!(model.completion_calls(), 200);
}

#[tokio::test]
async fn order_by_with_limit_still_sees_every_match() {
    let model = yes_to_sync();
    let ctx = notes(Arc::clone(&model), 50).await;
    let out = run(
        &ctx,
        "SELECT id FROM notes WHERE body MEANS 'it is about sync' ORDER BY id DESC LIMIT 1",
    )
    .await;
    assert!(out.contains(" 200 "), "{out}");
    assert_eq!(model.completion_calls(), 200);
}

#[tokio::test]
async fn offset_counts_toward_what_is_asked() {
    let model = yes_to_sync();
    let ctx = notes(Arc::clone(&model), 50).await;
    let out = run(
        &ctx,
        "SELECT id FROM notes WHERE body MEANS 'it is about sync' LIMIT 1 OFFSET 2",
    )
    .await;
    assert_eq!(out.lines().count(), 4 + 1, "{out}");
    assert!(out.contains(" 150 "), "{out}");
}

#[tokio::test]
async fn a_label_under_limit_asks_only_the_rows_returned() {
    let model = yes_to_sync();
    let ctx = notes(Arc::clone(&model), 1).await;
    run(
        &ctx,
        "SELECT id, body MEANS 'it is about sync' FROM notes LIMIT 3",
    )
    .await;
    assert_eq!(model.completion_calls(), 3);
}

#[tokio::test]
async fn explain_shows_the_fetch() {
    let ctx = notes(yes_to_sync(), 1).await;
    let out = run(
        &ctx,
        "EXPLAIN SELECT id FROM notes WHERE body MEANS 'it is about sync' LIMIT 5",
    )
    .await;
    assert!(out.contains("fetch=5"), "{out}");
    assert!(
        out.contains("VerifyExec: MEANS('it is about sync') model=mock fetch=5"),
        "{out}"
    );
}
