//! Funnel progress for NOTICE streaming: walk a physical plan for semcast
//! operators and turn their live metric counters into human lines. Reads
//! only `ExecutionPlan::metrics()` — no hooks inside the operators.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricsSet;
use serde::{Deserialize, Serialize};

/// What the engine emits while a statement runs. The counts travel
/// unrendered so a batch job can store them as columns; the pgwire handler
/// renders them back to the NOTICE text a connected client expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent {
    /// One line of the up-front plan summary.
    Plan(String),
    /// Counters moved since the last tick.
    Funnel(FunnelCounts),
    /// Final totals, once the stream is drained.
    Done(FunnelCounts),
}

impl ProgressEvent {
    /// The NOTICE line this event renders to.
    pub fn line(&self) -> String {
        match self {
            ProgressEvent::Plan(line) => line.clone(),
            ProgressEvent::Funnel(counts) => render(counts),
            ProgressEvent::Done(counts) => format!("funnel done — {}", render(counts)),
        }
    }
}

/// Live counter totals across all partitions of both semcast operators.
/// `has_*` distinguishes "operator absent" from "nothing counted yet".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
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
    pub has_cluster: bool,
    pub cluster_groups: usize,
    pub cluster_model_calls: usize,
    pub cluster_cache_hits: usize,
    /// Rows the index has never seen. Clustering gives them a NULL label
    /// rather than a group, so a non-zero count means a gap until refresh.
    pub cluster_unindexed_rows: usize,
    pub cluster_labels_failed: usize,
    pub has_classify: bool,
    pub classify_model_calls: usize,
    pub classify_cache_hits: usize,
    pub classify_rows_gated: usize,
    pub classify_rows_failed: usize,
    pub classify_branches_failed: usize,
    pub has_rank: bool,
    pub rank_candidate_rows: usize,
    pub rank_rows_pruned: usize,
    /// Rows the index has never seen. A rank drops them, so a non-zero count
    /// means results are missing until the index is refreshed.
    pub rank_unindexed_rows: usize,
    pub rank_model_calls: usize,
    pub rank_cache_hits: usize,
    pub rank_rows_failed: usize,
}

impl FunnelCounts {
    pub fn is_semantic(&self) -> bool {
        self.has_index_scan
            || self.has_verify
            || self.has_extract
            || self.has_rank
            || self.has_classify
            || self.has_cluster
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
            "SemClusterExec" => {
                counts.has_cluster = true;
                if let Some(metrics) = metrics {
                    counts.cluster_groups += counter_total(&metrics, "groups");
                    counts.cluster_model_calls += counter_total(&metrics, "model_calls");
                    counts.cluster_cache_hits += counter_total(&metrics, "cache_hits");
                    counts.cluster_unindexed_rows += counter_total(&metrics, "unindexed_rows");
                    counts.cluster_labels_failed += counter_total(&metrics, "labels_failed");
                }
            }
            "SemClassifyExec" => {
                counts.has_classify = true;
                if let Some(metrics) = metrics {
                    counts.classify_model_calls += counter_total(&metrics, "model_calls");
                    counts.classify_cache_hits += counter_total(&metrics, "cache_hits");
                    counts.classify_rows_gated += counter_total(&metrics, "rows_gated");
                    counts.classify_rows_failed += counter_total(&metrics, "rows_failed");
                    counts.classify_branches_failed += counter_total(&metrics, "branches_failed");
                }
            }
            "SemRankExec" => {
                counts.has_rank = true;
                if let Some(metrics) = metrics {
                    counts.rank_candidate_rows += counter_total(&metrics, "candidate_rows");
                    counts.rank_rows_pruned += counter_total(&metrics, "rows_pruned");
                    counts.rank_unindexed_rows += counter_total(&metrics, "unindexed_rows");
                    counts.rank_model_calls += counter_total(&metrics, "model_calls");
                    counts.rank_cache_hits += counter_total(&metrics, "cache_hits");
                    counts.rank_rows_failed += counter_total(&metrics, "rows_failed");
                }
            }
            _ => unreachable!("semcast_nodes returns only semcast operators"),
        }
    }
    counts
}

/// A progress event if the counters moved since `last`, else `None`.
pub fn snapshot_if_changed(
    plan: &Arc<dyn ExecutionPlan>,
    last: &mut FunnelCounts,
) -> Option<ProgressEvent> {
    let now = snapshot(plan);
    if now == *last || !now.is_semantic() {
        return None;
    }
    *last = now.clone();
    Some(ProgressEvent::Funnel(now))
}

pub fn final_totals(plan: &Arc<dyn ExecutionPlan>) -> Option<ProgressEvent> {
    let counts = snapshot(plan);
    counts.is_semantic().then_some(ProgressEvent::Done(counts))
}

pub fn render(counts: &FunnelCounts) -> String {
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
    if counts.has_cluster {
        parts.push(format!(
            "cluster: {} groups, {} model calls, {} cache hits, {} labels failed",
            counts.cluster_groups,
            counts.cluster_model_calls,
            counts.cluster_cache_hits,
            counts.cluster_labels_failed,
        ));
        // An unlabelled row is invisible in a GROUP BY result, so it gets its
        // own line rather than a number buried in the funnel.
        if counts.cluster_unindexed_rows > 0 {
            parts.push(format!(
                "cluster: {} rows are not in the semantic index and have no group — \
                 REFRESH SEMANTIC INDEX to include them",
                counts.cluster_unindexed_rows,
            ));
        }
    }
    if counts.has_classify {
        parts.push(format!(
            "classify: {} model calls, {} cache hits, {} rows gated, \
             {} rows failed, {} branches failed",
            counts.classify_model_calls,
            counts.classify_cache_hits,
            counts.classify_rows_gated,
            counts.classify_rows_failed,
            counts.classify_branches_failed,
        ));
    }
    if counts.has_rank {
        parts.push(format!(
            "rank: {} candidates, {} pruned, {} model calls, {} cache hits, {} rows failed",
            counts.rank_candidate_rows,
            counts.rank_rows_pruned,
            counts.rank_model_calls,
            counts.rank_cache_hits,
            counts.rank_rows_failed,
        ));
        // Dropped rows are invisible in the result, so they get their own
        // line rather than a number buried in the funnel.
        if counts.rank_unindexed_rows > 0 {
            parts.push(format!(
                "rank: {} rows are not in the semantic index and were skipped — \
                 REFRESH SEMANTIC INDEX to include them",
                counts.rank_unindexed_rows,
            ));
        }
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
        "IndexScanExec"
            | "VerifyExec"
            | "SemExtractExec"
            | "SemRankExec"
            | "SemClassifyExec"
            | "SemClusterExec"
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
