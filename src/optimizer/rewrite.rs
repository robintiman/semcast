//! Rewrites `Filter(means(text, 'condition'))` into a [`SemFilterNode`]
//! extension — the first half of roadmap step 1.
//!
//! [`SemFilterNode`]: crate::logical::SemFilterNode

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Result, ScalarValue, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::utils::{conjunction, split_conjunction_owned};
use datafusion::logical_expr::{Expr, Extension, Filter, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::logical::SemFilterNode;
use crate::sql::means_udf::MEANS_UDF_NAME;

/// Finds `means(..)` calls inside `Filter` predicates, splits them out of the
/// conjunction (the free predicates stay in a `Filter` below, which the
/// optimizer will keep pushing down — predicate ordering is just predicate
/// ordering), and stacks a `SemFilter` extension node on top.
///
/// MVP restriction: `means()` is only supported as a top-level `AND` conjunct
/// of a `WHERE` clause with a string-literal condition. Anywhere else —
/// under `OR`/`NOT`, in a `SELECT` list, a non-literal condition — is a
/// plan-time error rather than a silent model call per row.
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
        let LogicalPlan::Filter(filter) = plan else {
            for expr in plan.expressions() {
                if contains_means(&expr)? {
                    return plan_err!(
                        "means() is only supported as a top-level AND conjunct of a \
                         WHERE clause (found it in a {} node)",
                        plan.display()
                    );
                }
            }
            return Ok(Transformed::no(plan));
        };

        let (semantic, free): (Vec<Expr>, Vec<Expr>) =
            split_conjunction_owned(filter.predicate.clone())
                .into_iter()
                .partition(is_means_call);

        if semantic.is_empty() {
            if contains_means(&filter.predicate)? {
                return plan_err!(
                    "means() is only supported as a top-level AND conjunct of a WHERE \
                     clause; it cannot appear under OR, NOT, or inside another expression"
                );
            }
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        }
        for expr in &free {
            if contains_means(expr)? {
                return plan_err!(
                    "means() is only supported as a top-level AND conjunct of a WHERE \
                     clause; it cannot appear under OR, NOT, or inside another expression"
                );
            }
        }

        // Free predicates stay in a Filter below the SemFilter, so they run
        // first and DataFusion keeps optimizing them as usual.
        let mut rewritten = match conjunction(free) {
            Some(predicate) => {
                LogicalPlan::Filter(Filter::try_new(predicate, Arc::clone(&filter.input))?)
            }
            None => Arc::unwrap_or_clone(filter.input),
        };
        for call in semantic {
            let (text, condition, recall) = destructure_means(call)?;
            rewritten = LogicalPlan::Extension(Extension {
                node: Arc::new(SemFilterNode::new(rewritten, text, condition, recall)),
            });
        }
        Ok(Transformed::yes(rewritten))
    }
}

/// Attach a statement-level `WITH RECALL` target to every `means()` call in
/// the plan, as a third literal argument the rewrite reads back out. Zero
/// calls is a user mistake, not a no-op.
pub fn apply_recall(plan: LogicalPlan, recall: f64) -> Result<LogicalPlan> {
    let mut rewrites = 0usize;
    let transformed = plan.transform_up(|plan| {
        plan.map_expressions(|expr| {
            expr.transform_up(|expr| match expr {
                Expr::ScalarFunction(mut call)
                    if call.func.name() == MEANS_UDF_NAME && call.args.len() == 2 =>
                {
                    call.args
                        .push(Expr::Literal(ScalarValue::Float64(Some(recall)), None));
                    rewrites += 1;
                    Ok(Transformed::yes(Expr::ScalarFunction(call)))
                }
                other => Ok(Transformed::no(other)),
            })
        })
    })?;
    if rewrites == 0 {
        return plan_err!("WITH RECALL requires a MEANS predicate in the statement");
    }
    Ok(transformed.data)
}

fn is_means_call(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == MEANS_UDF_NAME)
}

fn contains_means(expr: &Expr) -> Result<bool> {
    expr.exists(|e| Ok(is_means_call(e)))
}

/// Pull `(text_expr, condition, recall)` out of a validated `means(..)` call.
fn destructure_means(expr: Expr) -> Result<(Expr, String, Option<f64>)> {
    let Expr::ScalarFunction(ScalarFunction { args, .. }) = expr else {
        unreachable!("caller checked is_means_call");
    };
    if !(2..=3).contains(&args.len()) {
        return plan_err!("means() takes 2 or 3 arguments, got {}", args.len());
    }
    let mut args = args.into_iter();
    let text = args.next().expect("length checked above");
    let condition = match args.next().expect("length checked above") {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => s,
        other => {
            return plan_err!(
                "the second argument of means() must be a string literal \
                 (the natural-language condition), got: {other}"
            );
        }
    };
    let recall = match args.next() {
        None => None,
        // Re-validated here because direct means() callers bypass the
        // WITH RECALL parser's range check.
        Some(Expr::Literal(ScalarValue::Float64(Some(r)), _)) if r > 0.0 && r <= 1.0 => Some(r),
        Some(other) => {
            return plan_err!(
                "the third argument of means() must be a recall target in (0, 1] \
                 as a float literal, got: {other}"
            );
        }
    };
    Ok((text, condition, recall))
}
