//! End-to-end tests for roadmap step 1: the `means()` rewrite rule and the
//! verify-only physical plan, against the deterministic mock model.

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use semcast::model::MockModel;
use semcast::semcast_context;

/// meetings(meeting_id, title, transcript) — one matching transcript, one
/// non-matching, one NULL.
async fn meetings_context() -> SessionContext {
    let ctx = semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));
    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'atlas planning',  'we agreed to ship offline sync in Q3'),
             (2, 'weekly standup',  'nothing notable happened'),
             (3, 'retro',           CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, title, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

async fn matching_ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches: Vec<RecordBatch> = ctx.sql(sql).await.unwrap().collect().await.unwrap();
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

#[tokio::test]
async fn optimized_plan_contains_sem_filter_above_free_predicates() {
    let ctx = meetings_context().await;
    let plan = ctx
        .sql("SELECT meeting_id FROM meetings WHERE meeting_id < 3 AND means(transcript, 'discussed offline sync')")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();

    let display = format!("{}", plan.display_indent());
    assert!(
        display.contains("SemFilter: MEANS('discussed offline sync')"),
        "no SemFilter in optimized plan:\n{display}"
    );
    // The free predicate must sit below the SemFilter so it runs first.
    let sem_pos = display.find("SemFilter").unwrap();
    let filter_pos = display
        .find("Filter:")
        .expect("free predicate Filter survives");
    assert!(
        filter_pos > sem_pos,
        "free-predicate Filter should be below SemFilter:\n{display}"
    );
}

#[tokio::test]
async fn means_filter_executes_end_to_end() {
    let ctx = meetings_context().await;
    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE means(transcript, 'discussed offline sync')
         ORDER BY meeting_id",
    )
    .await;
    // Row 2 doesn't match; row 3's NULL transcript never reaches the model.
    assert_eq!(ids, vec![1]);
}

#[tokio::test]
async fn free_predicates_still_apply() {
    let ctx = meetings_context().await;
    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE meeting_id > 1 AND means(transcript, 'discussed offline sync')",
    )
    .await;
    assert!(ids.is_empty(), "meeting 1 matches means() but not the free predicate");
}

#[tokio::test]
async fn bare_means_with_no_free_predicates_works() {
    let ctx = meetings_context().await;
    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE means(transcript, 'discussed offline sync')",
    )
    .await;
    assert_eq!(ids, vec![1]);
}

#[tokio::test]
async fn means_under_or_is_a_plan_time_error() {
    let ctx = meetings_context().await;
    let err = ctx
        .sql("SELECT meeting_id FROM meetings WHERE meeting_id = 3 OR means(transcript, 'x')")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap_err();
    assert!(
        err.to_string().contains("top-level AND conjunct"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn means_under_not_is_a_plan_time_error() {
    let ctx = meetings_context().await;
    let err = ctx
        .sql("SELECT meeting_id FROM meetings WHERE NOT means(transcript, 'x')")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap_err();
    assert!(err.to_string().contains("top-level AND conjunct"), "{err}");
}

#[tokio::test]
async fn means_in_select_list_is_a_plan_time_error() {
    let ctx = meetings_context().await;
    let err = ctx
        .sql("SELECT means(transcript, 'x') FROM meetings")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap_err();
    assert!(err.to_string().contains("top-level AND conjunct"), "{err}");
}

#[tokio::test]
async fn non_literal_condition_is_a_plan_time_error() {
    let ctx = meetings_context().await;
    let err = ctx
        .sql("SELECT meeting_id FROM meetings WHERE means(transcript, title)")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap_err();
    assert!(err.to_string().contains("string literal"), "{err}");
}

#[tokio::test]
async fn two_means_conjuncts_both_apply() {
    let ctx = meetings_context().await;
    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE means(transcript, 'discussed offline sync')
           AND means(transcript, 'anything at all')",
    )
    .await;
    // The mock answers yes to both conditions for row 1 (same transcript
    // contains 'offline sync'), so stacking two SemFilters keeps it.
    assert_eq!(ids, vec![1]);
}
