//! `WITH RECALL` certifies the whole funnel, not just the index floor: on an
//! indexed column verify reads chunks, and a match whose evidence sits
//! outside those chunks is lost however low the floor goes.

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricsSet;
use semcast::model::MockModel;
use semcast::{IndexOptions, create_semantic_index, semcast_context};

/// `matches` documents whose deciding word sits in their last chunk, plus
/// `negatives` without it. Under theme embedding against 'sync deadline',
/// every "sync" chunk scores 1.0 and the "other deadline" tail 0.0, so the
/// top three chunks verify reads never include the tail.
async fn tail_context(model: Arc<MockModel>, needle_in_tail: bool) -> SessionContext {
    let body = "sync ".repeat(1200);
    let (head, tail) = if needle_in_tail {
        (body.clone(), "other deadline ".repeat(300))
    } else {
        (format!("deadline {body}"), "other ".repeat(600))
    };
    let rows: Vec<String> = (1..=150)
        .map(|id| format!("({id}, 'id{id} {head}{tail}')"))
        .chain((151..=200).map(|id| format!("({id}, 'id{id} {body}{}')", "other ".repeat(600))))
        .collect();
    let ctx = semcast_context(model);
    ctx.sql(&format!(
        "CREATE TABLE docs AS SELECT * FROM (VALUES {}) AS t(id, body)",
        rows.join(", "),
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

fn model() -> Arc<MockModel> {
    Arc::new(MockModel::answering_yes_to(["deadline"]).embedding_by_theme(["sync", "other"]))
}

const QUERY: &str = "SELECT id FROM docs WHERE body MEANS 'sync deadline' WITH RECALL 0.9";

/// The matching ids and the index scan's metrics.
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
    let scan = find_node(&plan, "IndexScanExec").expect("the funnel planned a scan");
    (ids, scan.metrics().unwrap())
}

fn count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map_or(0, |m| m.as_usize())
}

fn find_node(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.name() == name {
        return Some(Arc::clone(plan));
    }
    plan.children()
        .into_iter()
        .find_map(|child| find_node(child, name))
}

#[tokio::test]
async fn chunked_verify_that_misses_matches_switches_to_full_text() {
    let model = model();
    let ctx = tail_context(Arc::clone(&model), true).await;
    let (ids, metrics) = run(&ctx).await;
    // Every match comes back: verify read the full text, where the needle is.
    assert_eq!(ids, (1..=150).collect::<Vec<_>>());
    assert_eq!(count(&metrics, "calibration_full_text"), 1);
    assert!(count(&metrics, "calibration_verify_misses") > 0);
    // No verify input was a chunk excerpt: each unsampled row read its text.
    let excerpts = model
        .completion_inputs()
        .iter()
        .filter(|input| input.contains("\n---\n"))
        .count();
    let calibration_calls = count(&metrics, "calibration_model_calls");
    assert!(
        model.completion_calls() > calibration_calls,
        "verify asked the unsampled rows"
    );
    assert!(
        excerpts <= calibration_calls,
        "only calibration asked chunks"
    );
}

#[tokio::test]
async fn chunked_verify_that_finds_the_matches_keeps_reading_chunks() {
    let model = model();
    let ctx = tail_context(Arc::clone(&model), false).await;
    let (ids, metrics) = run(&ctx).await;
    assert_eq!(ids, (1..=150).collect::<Vec<_>>());
    assert_eq!(count(&metrics, "calibration_full_text"), 0);
    assert_eq!(count(&metrics, "calibration_verify_misses"), 0);
}

#[tokio::test]
async fn a_failed_chunked_answer_counts_as_a_miss() {
    // Every chunked request carries the separator; failing on it fails every
    // chunked check, and none of the full-text labels.
    let model = Arc::new(
        MockModel::answering_yes_to(["deadline"])
            .embedding_by_theme(["sync", "other"])
            .failing_on(["\n---\n"]),
    );
    let ctx = tail_context(Arc::clone(&model), false).await;
    let (ids, metrics) = run(&ctx).await;
    assert_eq!(ids, (1..=150).collect::<Vec<_>>());
    assert_eq!(count(&metrics, "calibration_full_text"), 1);
}
