//! Tests for roadmap step 4: typed extraction. This file covers the logical
//! rewrite (`CAST(... AS T)[.field]` → `SemExtract` node) and the plan-time
//! errors; end-to-end execution against the mock model lands with the physical
//! `SemExtractExec`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, RecordBatch, StringArray};
use datafusion::execution::context::SessionContext;
use semcast::cache::InMemoryCache;
use semcast::model::MockModel;
use semcast::{semcast_context, semcast_context_with_cache};

/// A context with a `meetings` table and the README's `MeetingFacts` type.
async fn facts_context() -> SessionContext {
    facts_context_with(semcast_context(Arc::new(MockModel::default()))).await
}

/// A mock that fills the launch-stage fields from the transcript text.
fn json_mock() -> Arc<MockModel> {
    Arc::new(MockModel::answering_json_with(|req| {
        let stage = if req.input.contains("shipped") {
            "shipped"
        } else {
            "none"
        };
        serde_json::json!({
            "launch_stage": stage,
            "stage_quote": "the line",
            "products": [],
            "decisions": [],
        })
    }))
}

/// Load the `meetings` table (one shipped, one not, one NULL) and the type
/// into a caller-supplied context.
async fn facts_context_with(ctx: SessionContext) -> SessionContext {
    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'we shipped offline sync'),
             (2, 'nothing notable happened'),
             (3, CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    semcast::sql(
        &ctx,
        "CREATE SEMANTIC TYPE MeetingFacts AS (
             products  TEXT[]   'product names discussed in this meeting',
             decisions TEXT[]   'concrete decisions that were made',
             TOGETHER (
                 launch_stage ONEOF(none, idea, planned, scheduled, shipped)
                              'the furthest launch stage discussed',
                 stage_quote  TEXT 'the transcript line that shows that stage'
             )
         )",
    )
    .await
    .unwrap();
    ctx
}

async fn rows(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
}

/// The (meeting_id, launch_stage) pairs from a two-column result.
fn stage_pairs(batches: &[RecordBatch]) -> Vec<(i64, Option<String>)> {
    use datafusion::arrow::array::Int64Array;
    let mut out = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let stages = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let stage = stages.is_valid(row).then(|| stages.value(row).to_owned());
            out.push((ids.value(row), stage));
        }
    }
    out
}

async fn optimized(ctx: &SessionContext, sql: &str) -> datafusion::logical_expr::LogicalPlan {
    semcast::sql(ctx, sql)
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap()
}

#[tokio::test]
async fn field_access_plans_a_sem_extract_node() {
    let ctx = facts_context().await;
    let plan = optimized(
        &ctx,
        "SELECT meeting_id, CAST(transcript AS MeetingFacts).launch_stage FROM meetings",
    )
    .await;
    let display = format!("{}", plan.display_indent());
    assert!(
        display.contains("SemExtract: MeetingFacts"),
        "no SemExtract in optimized plan:\n{display}"
    );
}

#[tokio::test]
async fn field_pushdown_pulls_the_together_closure_and_nothing_else() {
    let ctx = facts_context().await;
    // Referencing launch_stage pulls in its TOGETHER sibling stage_quote, but
    // not the independent products/decisions fields.
    let plan = optimized(
        &ctx,
        "SELECT CAST(transcript AS MeetingFacts).launch_stage FROM meetings",
    )
    .await;
    let display = format!("{}", plan.display_indent());
    assert!(
        display.contains("SemExtract: MeetingFacts [2 field(s)]"),
        "expected exactly the 2 TOGETHER fields:\n{display}"
    );
}

#[tokio::test]
async fn marker_outside_select_list_is_a_plan_error() {
    let ctx = facts_context().await;
    let err = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE CAST(transcript AS MeetingFacts).launch_stage = 'shipped'",
    )
    .await
    .unwrap()
    .into_optimized_plan()
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("only supported in the SELECT list"),
        "got: {err}"
    );
}

#[tokio::test]
async fn subquery_lets_you_group_on_an_extracted_field() {
    let ctx = facts_context().await;
    // The inner projection is rewritten first (BottomUp), so grouping on the
    // extracted column in the outer query is legal.
    let plan = optimized(
        &ctx,
        "SELECT stage, count(*) FROM (
             SELECT CAST(transcript AS MeetingFacts).launch_stage AS stage FROM meetings
         ) GROUP BY stage",
    )
    .await;
    let display = format!("{}", plan.display_indent());
    assert!(
        display.contains("SemExtract: MeetingFacts"),
        "no SemExtract under the aggregate:\n{display}"
    );
}

#[tokio::test]
async fn unknown_type_is_a_clear_error() {
    let ctx = facts_context().await;
    // A cast to an unregistered type is left alone by the rewrite, so it fails
    // in DataFusion's own type resolution.
    let err = semcast::sql(&ctx, "SELECT CAST(transcript AS Nonesuch).x FROM meetings").await;
    assert!(err.is_err(), "unknown type should not plan");
}

#[tokio::test]
async fn extraction_fills_values_and_null_source_stays_null() {
    let ctx = facts_context_with(semcast_context(json_mock())).await;
    let batches = rows(
        &ctx,
        "SELECT meeting_id, CAST(transcript AS MeetingFacts).launch_stage AS stage
         FROM meetings ORDER BY meeting_id",
    )
    .await;
    assert_eq!(
        stage_pairs(&batches),
        vec![
            (1, Some("shipped".to_owned())),
            (2, Some("none".to_owned())),
            // NULL transcript never reaches the model → NULL, no call.
            (3, None),
        ],
    );
}

#[tokio::test]
async fn field_pushdown_sends_only_the_together_closure() {
    let mock = json_mock();
    let ctx = facts_context_with(semcast_context(Arc::clone(&mock) as Arc<_>)).await;
    rows(
        &ctx,
        "SELECT CAST(transcript AS MeetingFacts).launch_stage FROM meetings",
    )
    .await;
    // Every request's schema carries exactly the TOGETHER closure — never the
    // independent products/decisions fields.
    let schemas = mock.completion_schemas();
    assert!(!schemas.is_empty(), "extraction made model calls");
    for schema in schemas.into_iter().flatten() {
        let props = schema["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = props.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["launch_stage", "stage_quote"],
            "pushdown leaked fields"
        );
    }
}

#[tokio::test]
async fn rerun_is_served_from_cache() {
    let mock = json_mock();
    // A shared cache so the second run sees the first run's writes.
    let cache = Arc::new(InMemoryCache::default());
    let ctx = facts_context_with(semcast_context_with_cache(
        Arc::clone(&mock) as Arc<_>,
        cache,
    ))
    .await;
    let query =
        "SELECT CAST(transcript AS MeetingFacts).launch_stage FROM meetings ORDER BY meeting_id";
    rows(&ctx, query).await;
    let after_first = mock.completion_calls();
    assert!(after_first > 0, "first run calls the model");
    rows(&ctx, query).await;
    assert_eq!(
        mock.completion_calls(),
        after_first,
        "second run is fully cache-served",
    );
}

#[tokio::test]
async fn means_and_cast_compose_extraction_runs_on_survivors_only() {
    // One mock answers both the MEANS verdict (schemaless) and the extraction
    // (schema). MEANS passes only the "shipped" row, so extraction runs on
    // that row alone.
    let mock = Arc::new(
        MockModel::answering_json_with(
            |_| serde_json::json!({ "launch_stage": "shipped", "stage_quote": "q" }),
        )
        .also_answering_yes_to(["shipped"]),
    );
    let ctx = facts_context_with(semcast_context(Arc::clone(&mock) as Arc<_>)).await;
    let batches = rows(
        &ctx,
        "SELECT meeting_id, CAST(transcript AS MeetingFacts).launch_stage AS stage
         FROM meetings WHERE transcript MEANS 'shipped a product' ORDER BY meeting_id",
    )
    .await;
    assert_eq!(
        stage_pairs(&batches),
        vec![(1, Some("shipped".to_owned()))],
        "only the MEANS survivor is extracted",
    );
    // Exactly one extraction (schema) request — for the single survivor.
    let schema_calls = mock
        .completion_schemas()
        .into_iter()
        .filter(Option::is_some)
        .count();
    assert_eq!(schema_calls, 1, "extraction runs only on the survivor");
}
