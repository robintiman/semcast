//! The certificate is exact and uncorrected across floors, and a calibration
//! that can only certify pruning nothing keeps drawing labels.

use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricsSet;
use semcast::model::MockModel;
use semcast::{IndexOptions, create_semantic_index, semcast_context};

/// `high` matches the index scores 1.0, `low` matches it scores 0.0 (no
/// "sync"), and `negatives` non-matches at 0.0.
async fn context(high: usize, low: usize, negatives: usize) -> SessionContext {
    let model = Arc::new(MockModel::answering_yes_to(["match"]).embedding_by_theme(["sync"]));
    let rows: Vec<String> = (0..high)
        .map(|i| format!("sync match {i}"))
        .chain((0..low).map(|i| format!("match {i}")))
        .chain((0..negatives).map(|i| format!("note {i}")))
        .enumerate()
        .map(|(id, body)| format!("({id}, '{body}')"))
        .collect();
    let ctx = semcast_context(model);
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

async fn scan_metrics(ctx: &SessionContext, sql: &str) -> (usize, MetricsSet) {
    let plan = semcast::sql(ctx, sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();
    let rows = batches.iter().map(|b| b.num_rows()).sum();
    (
        rows,
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

const CERTIFIED: &str =
    "SELECT id FROM docs WHERE body MEANS 'sync' WITH RECALL 0.9 WITH CONFIDENCE 0.9";

#[tokio::test]
async fn a_certificate_that_prunes_nothing_keeps_drawing() {
    // Matches are 95% high-scoring: pruning nothing certifies on the first
    // tranche, pruning the 0.0 floor needs more labels.
    let ctx = context(300, 15, 85).await;
    let (_, metrics) = scan_metrics(&ctx, CERTIFIED).await;
    println!("{metrics}");
    assert!(
        count(&metrics, "calibration_sampled_rows") > 64,
        "{metrics}"
    );
}

#[tokio::test]
async fn a_separating_index_certifies_and_prunes() {
    // 160 matches at 1.0, 240 non-matches at 0.0: an exact bound certifies
    // the 1.0 floor from the sample's matches, and the negatives are pruned.
    let ctx = context(160, 0, 240).await;
    let (rows, metrics) = scan_metrics(&ctx, CERTIFIED).await;
    assert_eq!(rows, 160);
    let pruned = count(&metrics, "rows_pruned") + count(&metrics, "store_pruned");
    assert_eq!(pruned, 240, "{metrics}");
    assert!(count(&metrics, "rows_pruned") > 0, "{metrics}");
}
