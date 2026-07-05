//! Funnel derivation — turning one `SemFilter` into the cheapest plan that
//! still answers the question (roadmap step 2).
//!
//! The funnel is derived, never hand-written: free predicates first, then a
//! semantic-index scan with a calibrated threshold, then model verification
//! of the survivors. `EXPLAIN` prices every stage before a token is spent.

use crate::logical::SemFilterNode;
use crate::model::ModelId;

#[derive(Debug, Clone, PartialEq)]
pub enum FunnelStage {
    /// Plain-SQL predicates, $0, run first.
    FreePredicate { description: String },
    /// Semantic index scan; threshold set by calibration (or best-effort).
    IndexScan { threshold: f32 },
    /// LLM verification of surviving rows, reading top-scoring chunks only.
    Verify { model: ModelId, chunks_per_row: usize },
}

/// A derived funnel plus the estimates `EXPLAIN` prints.
#[derive(Debug, Clone, PartialEq)]
pub struct Funnel {
    pub stages: Vec<FunnelStage>,
    /// Estimated model calls at the verify stage.
    pub estimated_calls: usize,
    pub estimated_cost_usd: f64,
}

/// Derive the cheap-then-verify funnel for one semantic filter. Without a
/// semantic index the funnel degenerates to verify-only — correct first,
/// cheap later — and the planner warns that the plan has no cheap stage.
pub fn derive_funnel(_filter: &SemFilterNode) -> crate::Result<Funnel> {
    todo!("funnel derivation (roadmap step 2: semantic index pre-filter stage)")
}
