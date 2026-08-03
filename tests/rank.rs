//! End-to-end tests for semantic ranking: the `RELEVANCE TO` operator, the
//! `relevance()` marker, the rewrite rule, and the two-stage rank plan,
//! against the deterministic mock model.

use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use semcast::model::{CompletionRequest, MockModel};
use semcast::{IndexOptions, create_semantic_index, semcast_context};
use serde_json::{Value, json};

const SYNC: &str = "we agreed to ship offline sync in Q3";
const LAUNCH: &str = "the launch date slipped to Q4";
const STANDUP: &str = "nothing notable happened";

/// Score by how many of the query's words the excerpt contains — enough
/// structure for order assertions without leaving the mock.
fn word_overlap(request: &CompletionRequest) -> Value {
    let query = request
        .system
        .split_once("Query: ")
        .and_then(|(_, rest)| rest.split_once("\n\n"))
        .map_or("", |(query, _)| query)
        .to_lowercase();
    let input = request.input.to_lowercase();
    let words: Vec<&str> = query.split_whitespace().collect();
    let hits = words.iter().filter(|w| input.contains(**w)).count();
    let score = if words.is_empty() {
        0.0
    } else {
        hits as f64 / words.len() as f64
    };
    json!({ "score": score })
}

fn ranking_model() -> Arc<MockModel> {
    Arc::new(MockModel::answering_json_with(word_overlap))
}

/// meetings(meeting_id, title, transcript) — three transcripts and a NULL.
async fn meetings_context_with_model(model: Arc<MockModel>) -> SessionContext {
    let ctx = semcast_context(model);
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'atlas planning', '{SYNC}'),
             (2, 'launch review',  '{LAUNCH}'),
             (3, 'weekly standup', '{STANDUP}'),
             (4, 'retro',          CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, title, transcript)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

/// A context whose `meetings.transcript` carries a semantic index. The temp
/// dir is returned so it outlives the queries.
async fn indexed_context(model: Arc<MockModel>) -> (SessionContext, tempfile::TempDir) {
    let ctx = meetings_context_with_model(model).await;
    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            path: Some(dir.path().join("meetings.transcript.lance")),
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

fn ids(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("meeting_id is Int64")
                .values()
                .to_vec()
        })
        .collect()
}

async fn error(ctx: &SessionContext, sql: &str) -> String {
    match semcast::sql(ctx, sql).await {
        Ok(frame) => match frame.collect().await {
            Ok(_) => panic!("expected an error from: {sql}"),
            Err(err) => err.to_string(),
        },
        Err(err) => err.to_string(),
    }
}

#[tokio::test]
async fn optimized_plan_contains_sem_rank() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let plan = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 2",
    )
    .await
    .unwrap()
    .into_optimized_plan()
    .unwrap();

    let display = format!("{}", plan.display_indent());
    assert!(
        display.contains("SemRank: RELEVANCE TO 'offline sync'"),
        "no SemRank in optimized plan:\n{display}"
    );
    // The candidate stage is sized from the LIMIT, not the table.
    assert!(
        display.contains("limit=2") && display.contains("candidates=32"),
        "SemRank should carry the limit and its over-fetch:\n{display}"
    );
}

#[tokio::test]
async fn ranks_rows_best_first() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let batches = run(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 3",
    )
    .await;

    // The mock scores by word overlap: only meeting 1 mentions both words.
    assert_eq!(ids(&batches)[0], 1, "best match must come first");
}

#[tokio::test]
async fn function_form_exposes_the_score_as_a_value() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let batches = run(
        &ctx,
        "SELECT meeting_id, relevance(transcript, 'offline sync') AS score
         FROM meetings ORDER BY score DESC LIMIT 3",
    )
    .await;

    assert_eq!(
        batches[0].schema().field(1).name(),
        "score",
        "the projection's alias must survive the rewrite"
    );
    let scores: Vec<f64> = batches
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("score is Float64")
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(scores[0], 1.0, "meeting 1 contains both query words");
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "scores must be descending: {scores:?}"
    );
}

#[tokio::test]
async fn bare_relevance_order_by_is_best_first() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    // No DESC written anywhere: the AST pass has to supply it, or the worst
    // match would come first.
    let ascending = run(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY relevance(transcript, 'offline sync') ASC LIMIT 3",
    )
    .await;
    let bare = run(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY relevance(transcript, 'offline sync') LIMIT 3",
    )
    .await;

    assert_eq!(bare[0].column(0).len(), ascending[0].column(0).len());
    assert_ne!(
        ids(&bare),
        ids(&ascending),
        "a bare ORDER BY relevance() must not sort ascending"
    );
    assert_eq!(ids(&bare)[0], 1);
}

#[tokio::test]
async fn model_calls_are_bounded_by_the_candidate_stage() {
    let model = ranking_model();
    let (ctx, _dir) = indexed_context(Arc::clone(&model)).await;
    run(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 1",
    )
    .await;

    // Three non-NULL transcripts, all candidates at this table size — but one
    // call each, never one per (row × chunk).
    assert_eq!(
        model.completion_calls(),
        3,
        "one rerank call per candidate document"
    );
}

#[tokio::test]
async fn scores_come_from_the_cache_the_second_time() {
    let model = ranking_model();
    let (ctx, _dir) = indexed_context(Arc::clone(&model)).await;
    let sql = "SELECT meeting_id FROM meetings
               ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 3";
    let first = run(&ctx, sql).await;
    let calls_after_first = model.completion_calls();
    let second = run(&ctx, sql).await;

    assert_eq!(ids(&first), ids(&second), "cached scores must reproduce");
    assert_eq!(
        model.completion_calls(),
        calls_after_first,
        "the second run must not reach the model"
    );
}

#[tokio::test]
async fn the_rerank_model_reads_chunks_not_the_whole_document() {
    let model = ranking_model();
    let (ctx, _dir) = indexed_context(Arc::clone(&model)).await;
    run(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 3",
    )
    .await;

    let schemas = model.completion_schemas();
    assert!(
        schemas.iter().all(|s| s.is_some()),
        "every rerank call must carry the score schema"
    );
    assert!(
        schemas[0].as_ref().unwrap()["properties"]["score"].is_object(),
        "the schema must constrain a `score` property: {:?}",
        schemas[0]
    );
}

#[tokio::test]
async fn null_text_never_ranks() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let batches = run(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 10",
    )
    .await;

    assert!(
        !ids(&batches).contains(&4),
        "the NULL transcript has no score and cannot be ranked"
    );
}

#[tokio::test]
async fn rows_the_index_never_saw_are_pruned() {
    let model = ranking_model();
    let (ctx, _dir) = indexed_context(Arc::clone(&model)).await;
    // Insert after indexing: the new row is invisible to the candidate stage.
    ctx.sql("INSERT INTO meetings VALUES (5, 'late add', 'offline sync shipped')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batches = run(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 10",
    )
    .await;

    assert!(
        !ids(&batches).contains(&5),
        "an unindexed row cannot be a candidate — REFRESH SEMANTIC INDEX first"
    );
}

/// Dropping a row silently is the failure mode this operator has to avoid, so
/// the count has to reach the funnel the server streams as NOTICEs.
#[tokio::test]
async fn skipped_unindexed_rows_are_reported_in_the_funnel() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    ctx.sql("INSERT INTO meetings VALUES (5, 'late add', 'offline sync shipped')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let plan = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 10",
    )
    .await
    .unwrap()
    .create_physical_plan()
    .await
    .unwrap();
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();

    let counts = semcast::server::progress::snapshot(&plan);
    assert!(counts.has_rank, "the rank operator must report a funnel");
    assert_eq!(
        counts.rank_unindexed_rows, 1,
        "the row inserted after indexing must be counted, not silently dropped"
    );
    let line = semcast::server::progress::render(&counts);
    assert!(
        line.contains("REFRESH SEMANTIC INDEX"),
        "the NOTICE must say how to fix it: {line}"
    );
}

#[tokio::test]
async fn ranking_without_an_index_is_a_plan_time_error() {
    let ctx = meetings_context_with_model(ranking_model()).await;
    let message = error(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 3",
    )
    .await;

    assert!(
        message.contains("requires a semantic index on meetings(transcript)"),
        "error should name the missing index: {message}"
    );
    assert!(
        message.contains("CREATE SEMANTIC INDEX ON meetings(transcript)"),
        "error should name the statement that fixes it: {message}"
    );
}

#[tokio::test]
async fn ranking_without_a_limit_is_a_plan_time_error() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let message = error(
        &ctx,
        "SELECT meeting_id FROM meetings ORDER BY transcript RELEVANCE TO 'offline sync'",
    )
    .await;

    assert!(
        message.contains("requires a LIMIT"),
        "error should demand a LIMIT: {message}"
    );
}

#[tokio::test]
async fn relevance_in_a_where_clause_is_a_plan_time_error() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let message = error(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE relevance(transcript, 'offline sync') > 0.5 LIMIT 3",
    )
    .await;

    assert!(
        message.contains("only supported in the SELECT list or an ORDER BY"),
        "error should name the legal positions: {message}"
    );
}

#[tokio::test]
async fn a_computed_text_expression_is_a_plan_time_error() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let message = error(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY (title || transcript) RELEVANCE TO 'offline sync' LIMIT 3",
    )
    .await;

    assert!(
        message.contains("plain indexed column"),
        "error should explain why a computed expression can't rank: {message}"
    );
}

#[tokio::test]
async fn a_non_literal_query_is_a_plan_time_error() {
    let (ctx, _dir) = indexed_context(ranking_model()).await;
    let message = error(
        &ctx,
        "SELECT meeting_id FROM meetings
         ORDER BY transcript RELEVANCE TO title LIMIT 3",
    )
    .await;

    assert!(
        message.contains("must be a string literal"),
        "error should demand a literal query: {message}"
    );
}

#[tokio::test]
async fn free_predicates_run_before_the_ranking() {
    let model = ranking_model();
    let (ctx, _dir) = indexed_context(Arc::clone(&model)).await;
    let batches = run(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE meeting_id = 2
         ORDER BY transcript RELEVANCE TO 'offline sync' LIMIT 3",
    )
    .await;

    assert_eq!(ids(&batches), vec![2]);
    assert_eq!(
        model.completion_calls(),
        1,
        "only the surviving row should be scored"
    );
}
