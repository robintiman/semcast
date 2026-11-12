//! The statement's recall budget: `WITH RECALL r` is a promise about the
//! rows the whole statement returns, not about each `MEANS` filter in it.
//!
//! A filter loses matches only through a lossy stage: an index floor, or a
//! scorer's floor. One verified on full text alone is exact. With `m` lossy
//! filters, each gets recall `1 - (1 - r)/m` and confidence `1 - (1 - c)/m`.
//! By the union bound all `m` certificates hold together with confidence `c`,
//! and the statement loses at most the sum of what each filter loses, `1 - r`.
//!
//! That last step assumes a filter misses the statement's matches (rows that
//! meet every condition) no more often than its own matches. A filter's floor
//! scores its own condition, so nothing ties its losses to the other
//! conditions, but nothing rules it out either: certifying the statement
//! directly needs every condition's label on one sample, which the joint
//! calibration (phase 4 step 4) does. The additive split needs no
//! independence between filters, where a product split (`r^(1/m)` each)
//! would.
//!
//! The split is applied last, over the optimized plan, because only then is
//! it settled which filters have an index or a scorer: counting too early
//! (through a view, say) could miss a lossy filter and promise more than the
//! plan delivers. Each filter keeps the statement's own target, so running
//! the rule again recomputes the same split.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::index::registry::SemcastRuntime;
use crate::logical::SemFilterNode;

#[derive(Debug)]
pub struct RecallBudgetRule {
    runtime: Arc<SemcastRuntime>,
}

impl RecallBudgetRule {
    pub fn new(runtime: Arc<SemcastRuntime>) -> Self {
        Self { runtime }
    }

    /// Whether the physical planner will give this filter a lossy stage:
    /// the same test it applies (an index on the text, or a scorer with a
    /// recall target to calibrate).
    fn lossy(&self, filter: &SemFilterNode) -> bool {
        filter.statement_recall.is_some()
            && (self.runtime.covers(&filter.text, filter.input.schema())
                || self.runtime.has_scorer())
    }
}

/// This filter's share of a statement target split `shares` ways: an equal
/// part of the allowed loss, and of the allowed error.
pub fn share(recall: f64, confidence: Option<f64>, shares: usize) -> (f64, Option<f64>) {
    let n = shares as f64;
    (
        1.0 - (1.0 - recall) / n,
        confidence.map(|c| 1.0 - (1.0 - c) / n),
    )
}

fn as_filter(plan: &LogicalPlan) -> Option<&SemFilterNode> {
    match plan {
        LogicalPlan::Extension(Extension { node }) => node.as_any().downcast_ref::<SemFilterNode>(),
        _ => None,
    }
}

impl OptimizerRule for RecallBudgetRule {
    fn name(&self) -> &str {
        "semcast_recall_budget"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // The whole plan at once: the split needs every filter counted first.
        None
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut lossy = 0usize;
        plan.apply_with_subqueries(|node| {
            if as_filter(node).is_some_and(|filter| self.lossy(filter)) {
                lossy += 1;
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if lossy == 0 {
            return Ok(Transformed::no(plan));
        }
        plan.transform_down_with_subqueries(|node| {
            let Some(filter) = as_filter(&node) else {
                return Ok(Transformed::no(node));
            };
            if !self.lossy(filter) {
                return Ok(Transformed::no(node));
            }
            let statement = filter
                .statement_recall
                .expect("a lossy filter has a statement target");
            let (recall, confidence) = share(statement, filter.statement_confidence, lossy);
            if filter.recall == Some(recall)
                && filter.confidence == confidence
                && filter.shares == lossy
            {
                return Ok(Transformed::no(node));
            }
            let mut filter = filter.clone();
            filter.recall = Some(recall);
            filter.confidence = confidence;
            filter.shares = lossy;
            Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(filter),
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::share;

    #[test]
    fn shares_add_back_to_the_statement_target() {
        let (r, c) = share(0.9, Some(0.95), 3);
        assert!((3.0 * (1.0 - r) - 0.1).abs() < 1e-12);
        assert!((3.0 * (1.0 - c.unwrap()) - 0.05).abs() < 1e-12);
        assert_eq!(share(0.9, None, 1), (0.9, None));
    }
}
