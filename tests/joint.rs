//! An index and a scorer under one recall target: both floors are chosen
//! together on one sample, and the cheaper funnel that certifies wins.

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricsSet;
use semcast::model::MockModel;
use semcast::{IndexOptions, SemcastContextBuilder, create_semantic_index};

/// 200 short documents, the 40 with ids divisible by 5 mentioning a
/// deadline. With `weak_index`, every document mentions sync, so the index
/// (theme embedding on "sync") scores them all alike; otherwise only the
/// matches do, and the index separates them on its own.
async fn context(
    session: Arc<MockModel>,
    scorer: Arc<MockModel>,
    weak_index: bool,
) -> SessionContext {
    let rows: Vec<String> = (1..=200)
        .map(|id| {
            let text = match (id % 5 == 0, weak_index) {
                (true, _) => format!("sync deadline {id}"),
                (false, true) => format!("sync note {id}"),
                (false, false) => format!("note {id}"),
            };
            format!("({id}, '{text}')")
        })
        .collect();
    let ctx = SemcastContextBuilder::new(session as _)
        .with_scorer(scorer as _)
        .build();
    ctx.sql(&format!(
        "CREATE TABLE docs AS SELECT * FROM (VALUES {}) AS t(id, body)",
        rows.join(", ")
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "docs", "body", IndexOptions::default())
        .await
        .unwrap();
    ctx
}

fn models() -> (Arc<MockModel>, Arc<MockModel>) {
    (
        Arc::new(MockModel::answering_yes_to(["deadline"]).embedding_by_theme(["sync"])),
        Arc::new(MockModel::answering_yes_to(["deadline"]).with_id("scorer")),
    )
}

const QUERY: &str = "SELECT id FROM docs WHERE body MEANS 'sync deadline' WITH RECALL 0.9";

async fn run(ctx: &SessionContext) -> (Vec<i64>, MetricsSet) {
    let plan = semcast::sql(ctx, QUERY)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    ids.sort_unstable();
    (
        ids,
        find(&plan, "IndexScanExec").unwrap().metrics().unwrap(),
    )
}

fn find(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.name() == name {
        return Some(Arc::clone(plan));
    }
    plan.children().into_iter().find_map(|c| find(c, name))
}

fn count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map_or(0, |m| m.as_usize())
}

fn matches() -> Vec<i64> {
    (1..=200).filter(|id| id % 5 == 0).collect()
}

#[tokio::test]
async fn a_weak_index_hands_the_pruning_to_the_scorer() {
    let (session, scorer) = models();
    let ctx = context(Arc::clone(&session), Arc::clone(&scorer), true).await;
    let (ids, metrics) = run(&ctx).await;
    assert_eq!(ids, matches());
    assert!(count(&metrics, "scorer_rows_pruned") > 100, "{metrics}");
    // Labels and chunk checks for the sample, then verify for the unsampled
    // matches only: far under one call per row.
    assert!(
        session.completion_calls() < 130,
        "{} session calls",
        session.completion_calls()
    );
}

#[tokio::test]
async fn a_separating_index_leaves_the_scorer_to_the_sample() {
    let (session, scorer) = models();
    let ctx = context(Arc::clone(&session), Arc::clone(&scorer), false).await;
    let (ids, metrics) = run(&ctx).await;
    assert_eq!(ids, matches());
    assert_eq!(count(&metrics, "scorer_rows_pruned"), 0, "{metrics}");
    // The scorer read the sample, to price itself, and nothing more.
    assert!(
        scorer.completion_calls() <= 64,
        "{}",
        scorer.completion_calls()
    );
}

#[tokio::test]
async fn explain_shows_the_joint_funnel() {
    let (session, scorer) = models();
    let ctx = context(session, scorer, true).await;
    let batches = semcast::sql(&ctx, &format!("EXPLAIN {QUERY}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    assert!(
        plan.contains("then scorer=scorer (floors chosen jointly)"),
        "{plan}"
    );
    assert!(
        !plan.contains("ScoreScanExec"),
        "one funnel, not two stages: {plan}"
    );
}
