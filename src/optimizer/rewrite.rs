//! Rewrites `Filter(means(text, 'condition'))` into a [`SemFilterNode`]
//! extension — the first half of roadmap step 1.
//!
//! [`SemFilterNode`]: crate::logical::SemFilterNode

use datafusion::common::Result;
use datafusion::common::tree_node::Transformed;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

/// Finds `means(..)` calls inside `Filter` predicates, splits them out of any
/// conjunction (the free predicates stay in the `Filter`, which the optimizer
/// will keep pushing down — predicate ordering is just predicate ordering),
/// and wraps the input in a `SemFilter` extension node.
#[derive(Debug, Default)]
pub struct MeansRewriteRule;

impl OptimizerRule for MeansRewriteRule {
    fn name(&self) -> &str {
        "semcast_means_rewrite"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Roadmap step 1: match LogicalPlan::Filter, walk the predicate for
        // ScalarFunction("means"), split the conjunction, emit
        // LogicalPlan::Extension(SemFilterNode). No-op until then so the
        // registered rule never breaks ordinary queries.
        Ok(Transformed::no(plan))
    }
}
