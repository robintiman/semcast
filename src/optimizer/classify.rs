//! Turns `means()` markers in a `SELECT` list into [`SemClassifyNode`]s — the
//! classify half of the classify/rank/cluster roadmap item.
//!
//! Called by [`MeansRewriteRule`] for the `Projection` case; the `Filter` case
//! stays in `rewrite.rs`, since a `MEANS` in a `WHERE` is a filter and a
//! `MEANS` in a `SELECT` list is a label.
//!
//! The node materializes one boolean column per condition and the `CASE`
//! stays where it was, referencing those columns. Conditions over the same
//! text are fused into a single model call per row, which is an optimization
//! only: every column still answers exactly the question the same `MEANS`
//! would ask in a `WHERE`.
//!
//! [`SemClassifyNode`]: crate::logical::SemClassifyNode
//! [`MeansRewriteRule`]: crate::optimizer::rewrite::MeansRewriteRule

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result, plan_err};
use datafusion::logical_expr::expr::Case;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, Projection};

use crate::logical::SemClassifyNode;
use crate::logical::sem_classify::branch_column_name;
use crate::optimizer::rewrite::{contains_means, destructure_means, is_means_call};

/// Rewrite a projection's `means()` markers into stacked classify nodes,
/// leaving a projection without markers untouched.
pub fn rewrite_projection(projection: Projection) -> Result<Transformed<LogicalPlan>> {
    let mut holds_marker = false;
    for expr in &projection.expr {
        if contains_means(expr)? {
            holds_marker = true;
            break;
        }
    }
    if !holds_marker {
        return Ok(Transformed::no(LogicalPlan::Projection(projection)));
    }

    // One pass: registering a marker yields the column that replaces it, so
    // the rewritten expressions and the node specs come out together.
    let mut builder = Builder::default();
    let preserver = NamePreserver::new_for_projection();
    let exprs = projection
        .expr
        .into_iter()
        .map(|expr| {
            let saved = preserver.save(&expr);
            let rewritten = rewrite_expr(expr, &None, &mut builder)?;
            Ok(saved.restore(rewritten))
        })
        .collect::<Result<Vec<_>>>()?;

    let input = builder.build((*projection.input).clone())?;
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(exprs, Arc::new(input))?,
    )))
}

/// Replace every `means()` in `expr` with the column its classify node will
/// produce, threading the guard that says which rows reach it.
fn rewrite_expr(expr: Expr, guard: &Option<Expr>, builder: &mut Builder) -> Result<Expr> {
    if is_means_call(&expr) {
        let (text, condition, recall) = destructure_means(expr)?;
        if recall.is_some() {
            return plan_err!(
                "WITH RECALL calibrates the index pre-filter for a MEANS in a WHERE \
                 clause; a MEANS in the SELECT list labels every row, so there is \
                 nothing to calibrate"
            );
        }
        return Ok(Expr::Column(builder.register(text, condition, guard)));
    }

    if let Expr::Case(case) = expr {
        return rewrite_case(case, guard, builder);
    }

    Ok(expr
        .map_children(|child| Ok(Transformed::yes(rewrite_expr(child, guard, builder)?)))?
        .data)
}

/// A `CASE` gates its `MEANS` branches by the ordinary conditions that precede
/// them: a row an earlier branch already claimed never reaches the model.
///
/// Only conditions before the *first* `MEANS` branch gate. Honouring the ones
/// after it would mean waiting for each branch's verdict before deciding
/// whether to ask the next — serializing exactly what the fusion exists to
/// batch. So the guard may occasionally pay for a call that lazy `CASE`
/// evaluation would have skipped, but it never skips one that is needed.
fn rewrite_case(case: Case, guard: &Option<Expr>, builder: &mut Builder) -> Result<Expr> {
    let branch_guard = case_guard(&case, guard)?;

    let expr = case
        .expr
        .map(|base| rewrite_expr(*base, guard, builder).map(Box::new))
        .transpose()?;
    let when_then_expr = case
        .when_then_expr
        .into_iter()
        .map(|(when, then)| {
            // Only a WHEN that actually asks the model is gated; the results
            // (THEN/ELSE) sit outside the branch conditions entirely.
            let when_guard = if contains_means(&when)? {
                &branch_guard
            } else {
                guard
            };
            Ok((
                Box::new(rewrite_expr(*when, when_guard, builder)?),
                Box::new(rewrite_expr(*then, guard, builder)?),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let else_expr = case
        .else_expr
        .map(|e| rewrite_expr(*e, guard, builder).map(Box::new))
        .transpose()?;

    Ok(Expr::Case(Case {
        expr,
        when_then_expr,
        else_expr,
    }))
}

/// The guard a `CASE`'s `MEANS` branches inherit: every ordinary condition
/// before the first of them must have failed to match.
///
/// `IS NOT TRUE` rather than `NOT`, because a NULL condition doesn't match a
/// `CASE` branch but `NOT NULL` is NULL — which, read as a guard, would
/// wrongly skip the call.
fn case_guard(case: &Case, outer: &Option<Expr>) -> Result<Option<Expr>> {
    // `CASE <base> WHEN <value> ...` compares each WHEN against the base, so
    // the condition is not the WHEN expression itself. Not worth modelling:
    // fall back to the outer guard.
    if case.expr.is_some() {
        return Ok(outer.clone());
    }
    let mut leading = Vec::new();
    for (when, _) in &case.when_then_expr {
        if contains_means(when)? {
            break;
        }
        leading.push((**when).clone().is_not_true());
    }
    Ok(match (outer.clone(), conjunction(leading)) {
        (Some(outer), Some(leading)) => Some(outer.and(leading)),
        (Some(outer), None) => Some(outer),
        (None, leading) => leading,
    })
}

/// Accumulates the `(text, guard)` groups a projection references. Registering
/// a marker returns its column immediately, so one traversal both rewrites the
/// expressions and describes the nodes to build.
#[derive(Default)]
struct Builder {
    groups: Vec<Group>,
}

struct Group {
    text: Expr,
    guard: Option<Expr>,
    /// Conditions in first-appearance order; the index is the branch index.
    conditions: Vec<String>,
}

impl Builder {
    /// The column that will carry `condition`'s verdict, creating the group
    /// and the branch if this is the first time either is seen.
    fn register(&mut self, text: Expr, condition: String, guard: &Option<Expr>) -> Column {
        let group = match self
            .groups
            .iter()
            .position(|g| g.text == text && &g.guard == guard)
        {
            Some(group) => group,
            None => {
                self.groups.push(Group {
                    text,
                    guard: guard.clone(),
                    conditions: Vec::new(),
                });
                self.groups.len() - 1
            }
        };
        let conditions = &mut self.groups[group].conditions;
        let branch = match conditions.iter().position(|c| c == &condition) {
            Some(branch) => branch,
            None => {
                conditions.push(condition);
                conditions.len() - 1
            }
        };
        Column::new_unqualified(branch_column_name(group, branch))
    }

    /// Stack one node per group above `input`. Group index is the node id, so
    /// the columns handed out during registration line up.
    fn build(self, input: LogicalPlan) -> Result<LogicalPlan> {
        let mut plan = input;
        for (id, group) in self.groups.into_iter().enumerate() {
            plan = LogicalPlan::Extension(Extension {
                node: Arc::new(SemClassifyNode::try_new(
                    plan,
                    group.text,
                    group.conditions,
                    group.guard,
                    id,
                )?),
            });
        }
        Ok(plan)
    }
}
