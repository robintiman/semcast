//! Funnel progress for NOTICE streaming: walk a physical plan for semcast
//! operators and turn their live metric counters into human lines. Reads
//! only `ExecutionPlan::metrics()` — no hooks inside the operators.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricsSet;

/// Live counter totals across all partitions of both semcast operators.
/// `has_*` distinguishes "operator absent" from "nothing counted yet".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FunnelCounts {
    pub has_index_scan: bool,
    pub index_hits: usize,
    pub rows_pruned: usize,
    pub calibration_sampled_rows: usize,
    pub calibration_model_calls: usize,
    pub has_verify: bool,
    pub model_calls: usize,
    pub cache_hits: usize,
    pub rows_dropped: usize,
    pub has_extract: bool,
    pub extract_model_calls: usize,
    pub extract_cache_hits: usize,
    pub extract_rows_failed: usize,
    pub extract_fields_failed: usize,
}

impl FunnelCounts {
    pub fn is_semantic(&self) -> bool {
        self.has_index_scan || self.has_verify || self.has_extract
    }
}

/// One `funnel:` line per semcast operator, top-down — the plan's own
/// `DisplayAs` text, which already states the model-call ceiling.
pub fn funnel_summary(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    semcast_nodes(plan)
        .iter()
        .map(|node| {
            let one_line = datafusion::physical_plan::displayable(node.as_ref())
                .one_line()
                .to_string();
            format!("funnel: {}", one_line.trim())
        })
        .collect()
}

pub fn snapshot(plan: &Arc<dyn ExecutionPlan>) -> FunnelCounts {
    let mut counts = FunnelCounts::default();
    for node in semcast_nodes(plan) {
        let metrics = node.metrics();
        match node.name() {
            "IndexScanExec" => {
                counts.has_index_scan = true;
                if let Some(metrics) = metrics {
                    counts.index_hits += counter_total(&metrics, "index_hits");
                    counts.rows_pruned += counter_total(&metrics, "rows_pruned");
                    counts.calibration_sampled_rows +=
                        counter_total(&metrics, "calibration_sampled_rows");
                    counts.calibration_model_calls +=
                        counter_total(&metrics, "calibration_model_calls");
                }
            }
            "VerifyExec" => {
                counts.has_verify = true;
                if let Some(metrics) = metrics {
                    counts.model_calls += counter_total(&metrics, "model_calls");
                    counts.cache_hits += counter_total(&metrics, "cache_hits");
                    counts.rows_dropped += counter_total(&metrics, "rows_dropped");
                }
            }
            "SemExtractExec" => {
                counts.has_extract = true;
                if let Some(metrics) = metrics {
                    counts.extract_model_calls += counter_total(&metrics, "model_calls");
                    counts.extract_cache_hits += counter_total(&metrics, "cache_hits");
                    counts.extract_rows_failed += counter_total(&metrics, "rows_failed");
                    counts.extract_fields_failed += counter_total(&metrics, "fields_failed");
                }
            }
            _ => unreachable!("semcast_nodes returns only semcast operators"),
        }
    }
    counts
}

/// A progress line if the counters moved since `last`, else `None`.
pub fn snapshot_if_changed(
    plan: &Arc<dyn ExecutionPlan>,
    last: &mut FunnelCounts,
) -> Option<String> {
    let now = snapshot(plan);
    if now == *last || !now.is_semantic() {
        return None;
    }
    let line = render(&now);
    *last = now;
    Some(line)
}

pub fn final_totals(plan: &Arc<dyn ExecutionPlan>) -> Option<String> {
    let counts = snapshot(plan);
    counts
        .is_semantic()
        .then(|| format!("funnel done — {}", render(&counts)))
}

fn render(counts: &FunnelCounts) -> String {
    let mut parts = Vec::new();
    if counts.has_index_scan {
        parts.push(format!(
            "index scan: {} hits, {} pruned",
            counts.index_hits, counts.rows_pruned,
        ));
        if counts.calibration_sampled_rows > 0 {
            parts.push(format!(
                "calibration: {} rows labeled, {} model calls",
                counts.calibration_sampled_rows, counts.calibration_model_calls,
            ));
        }
    }
    if counts.has_verify {
        parts.push(format!(
            "verify: {} model calls, {} cache hits, {} dropped",
            counts.model_calls, counts.cache_hits, counts.rows_dropped,
        ));
    }
    if counts.has_extract {
        parts.push(format!(
            "extract: {} model calls, {} cache hits, {} rows failed, {} fields failed",
            counts.extract_model_calls,
            counts.extract_cache_hits,
            counts.extract_rows_failed,
            counts.extract_fields_failed,
        ));
    }
    parts.join("; ")
}

fn semcast_nodes(plan: &Arc<dyn ExecutionPlan>) -> Vec<Arc<dyn ExecutionPlan>> {
    let mut nodes = Vec::new();
    collect(plan, &mut nodes);
    nodes
}

fn collect(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<Arc<dyn ExecutionPlan>>) {
    if matches!(
        plan.name(),
        "IndexScanExec" | "VerifyExec" | "SemExtractExec"
    ) {
        out.push(Arc::clone(plan));
    }
    for child in plan.children() {
        collect(child, out);
    }
}

fn counter_total(metrics: &MetricsSet, name: &str) -> usize {
    metrics
        .iter()
        .filter(|m| m.value().name() == name)
        .map(|m| m.value().as_usize())
        .sum()
}
