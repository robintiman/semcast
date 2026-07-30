//! Rewrites `relevance(text, 'query')` markers into [`SemRankNode`]s — the
//! ranking half of the classify/rank/cluster roadmap item.
//!
//! Ranking is not a filter, so the marker is legal exactly where a *value* is
//! legal on the way to an ordering: a `SELECT` list or an `ORDER BY`. The rule
//! inserts the node below whichever one holds the marker and swaps the marker
//! for a reference to the node's score column — the same shape
//! [`ExtractRewriteRule`] uses.
//!
//! A `LIMIT` is required. Without one, "rank these rows" means one model call
//! per row of the whole table, which is exactly the surprise semcast exists to
//! avoid; the limit also sizes the candidate stage.
//!
//! [`SemRankNode`]: crate::logical::SemRankNode
//! [`ExtractRewriteRule`]: crate::optimizer::extract::ExtractRewriteRule

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, ScalarValue, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{
    Expr, Extension, FetchType, LogicalPlan, Projection, SkipType, Sort,
};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::logical::SemRankNode;
use crate::sql::rank_udf::RELEVANCE_UDF_NAME;

const NOT_IN_SELECT_OR_ORDER_BY: &str = "relevance() is only supported in the SELECT list or an ORDER BY; \
     wrap it in a subquery to filter or group on a relevance score";

const NO_LIMIT: &str = "relevance() requires a LIMIT — ranking without one means \
     one model call per row of the whole table; add `LIMIT <k>` (the candidate \
     stage is sized from it)";

/// Turns `relevance()` markers into `SemRank` nodes below the projection or
/// sort that holds them.
#[derive(Debug, Default)]
pub struct RelevanceRewriteRule;

impl OptimizerRule for RelevanceRewriteRule {
    fn name(&self) -> &str {
        "semcast_relevance_rewrite"
    }

    /// Self-driven traversal: the `LIMIT` that sizes the candidate stage sits
    /// *above* the marker, so the rule needs the whole plan in hand rather
    /// than one node at a time.
    fn apply_order(&self) -> Option<ApplyOrder> {
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
        if !plan_contains_marker(&plan)? {
            return Ok(Transformed::no(plan));
        }
        let Some(limit) = find_fetch(&plan)? else {
            return plan_err!("{NO_LIMIT}");
        };

        let mut next_id = 0usize;
        plan.transform_up(|node| rewrite_node(node, limit, &mut next_id))
    }
}

fn rewrite_node(
    node: LogicalPlan,
    limit: usize,
    next_id: &mut usize,
) -> Result<Transformed<LogicalPlan>> {
    match node {
        LogicalPlan::Projection(projection) => {
            if !exprs_contain_marker(&projection.expr)? {
                return Ok(Transformed::no(LogicalPlan::Projection(projection)));
            }
            let mut builder = NodeBuilder::new(limit, next_id);
            for expr in &projection.expr {
                builder.observe(expr)?;
            }
            let input = builder.build((*projection.input).clone())?;

            // Preserve the projection's output names — `relevance(t,'q')`
            // keeps its display name even though it is now a column.
            let preserver = NamePreserver::new_for_projection();
            let exprs = projection
                .expr
                .into_iter()
                .map(|expr| {
                    let saved = preserver.save(&expr);
                    let rewritten = builder.replace(expr)?;
                    Ok(saved.restore(rewritten))
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(Transformed::yes(LogicalPlan::Projection(
                Projection::try_new(exprs, Arc::new(input))?,
            )))
        }
        LogicalPlan::Sort(sort) => {
            let sort_exprs: Vec<Expr> = sort.expr.iter().map(|s| s.expr.clone()).collect();
            if !exprs_contain_marker(&sort_exprs)? {
                return Ok(Transformed::no(LogicalPlan::Sort(sort)));
            }
            let mut builder = NodeBuilder::new(limit, next_id);
            for expr in &sort_exprs {
                builder.observe(expr)?;
            }
            let input = builder.build((*sort.input).clone())?;

            let expr = sort
                .expr
                .into_iter()
                .map(|mut item| {
                    item.expr = builder.replace(item.expr)?;
                    Ok(item)
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(Transformed::yes(LogicalPlan::Sort(Sort {
                expr,
                input: Arc::new(input),
                fetch: sort.fetch,
            })))
        }
        other => {
            // A marker anywhere else never gets a node, so catch it here
            // rather than letting the UDF's `not_impl` fire at run time.
            if exprs_contain_marker(&other.expressions())? {
                return plan_err!("{NOT_IN_SELECT_OR_ORDER_BY}");
            }
            Ok(Transformed::no(other))
        }
    }
}

/// Accumulates the distinct `(text, query)` pairs one plan node references,
/// then materializes them into stacked [`SemRankNode`]s.
struct NodeBuilder<'a> {
    limit: usize,
    next_id: &'a mut usize,
    /// One entry per distinct marker: its arguments and, after `build`, the
    /// score column that replaces it.
    groups: Vec<Group>,
}

struct Group {
    text: Expr,
    query: String,
    column: Option<datafusion::common::Column>,
}

impl<'a> NodeBuilder<'a> {
    fn new(limit: usize, next_id: &'a mut usize) -> Self {
        Self {
            limit,
            next_id,
            groups: Vec::new(),
        }
    }

    /// Record every marker in `expr`, deduplicating by `(text, query)` so a
    /// repeated `relevance(t,'q')` costs one node, not two.
    fn observe(&mut self, expr: &Expr) -> Result<()> {
        let mut found = Vec::new();
        collect_markers(expr, &mut found)?;
        for (text, query) in found {
            if !self
                .groups
                .iter()
                .any(|g| g.text == text && g.query == query)
            {
                self.groups.push(Group {
                    text,
                    query,
                    column: None,
                });
            }
        }
        Ok(())
    }

    /// Stack one node per group above `input`.
    fn build(&mut self, input: LogicalPlan) -> Result<LogicalPlan> {
        let mut plan = input;
        for group in &mut self.groups {
            let id = *self.next_id;
            *self.next_id += 1;
            let node = SemRankNode::try_new(
                plan,
                group.text.clone(),
                group.query.clone(),
                self.limit,
                id,
            )?;
            group.column = Some(node.score_column());
            plan = LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            });
        }
        Ok(plan)
    }

    /// Swap every marker in `expr` for its node's score column.
    fn replace(&self, expr: Expr) -> Result<Expr> {
        expr.transform(|e| {
            let Some((text, query)) = parse_marker(&e)? else {
                return Ok(Transformed::no(e));
            };
            let column = self
                .groups
                .iter()
                .find(|g| g.text == text && g.query == query)
                .and_then(|g| g.column.clone())
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(
                        "rank node missing for a collected relevance() marker".to_owned(),
                    )
                })?;
            Ok(Transformed::yes(Expr::Column(column)))
        })
        .map(|t| t.data)
    }
}

/// The first `LIMIT`/`Sort` fetch found walking down from the root, plus any
/// `OFFSET` — `LIMIT 10 OFFSET 5` needs 15 rows ranked, not 10.
///
/// `Ok(None)` means the query has no limit at all.
fn find_fetch(plan: &LogicalPlan) -> Result<Option<usize>> {
    let mut found = None;
    plan.apply(|node| {
        let fetch = match node {
            LogicalPlan::Limit(limit) => match limit.get_fetch_type()? {
                FetchType::Literal(Some(fetch)) => {
                    let skip = match limit.get_skip_type()? {
                        SkipType::Literal(skip) => skip,
                        SkipType::UnsupportedExpr => 0,
                    };
                    Some(fetch.saturating_add(skip))
                }
                // `LIMIT NULL` and computed limits carry no number to size
                // the candidate stage with.
                _ => None,
            },
            // The optimizer may already have folded the limit into the sort.
            LogicalPlan::Sort(sort) => sort.fetch,
            _ => None,
        };
        if let Some(fetch) = fetch {
            found = Some(fetch);
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

fn is_marker(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == RELEVANCE_UDF_NAME)
}

fn plan_contains_marker(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply(|node| {
        if exprs_contain_marker(&node.expressions())? {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

fn exprs_contain_marker(exprs: &[Expr]) -> Result<bool> {
    for expr in exprs {
        if expr.exists(|e| Ok(is_marker(e)))? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect_markers(expr: &Expr, out: &mut Vec<(Expr, String)>) -> Result<()> {
    expr.apply(|e| {
        if let Some(marker) = parse_marker(e)? {
            out.push(marker);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map(|_| ())
}

/// Pull `(text_expr, query)` out of a `relevance(..)` call.
fn parse_marker(expr: &Expr) -> Result<Option<(Expr, String)>> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return Ok(None);
    };
    if func.name() != RELEVANCE_UDF_NAME {
        return Ok(None);
    }
    if args.len() != 2 {
        return plan_err!("relevance() takes 2 arguments, got {}", args.len());
    }
    let query = match &args[1] {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => s.clone(),
        other => {
            return plan_err!(
                "the second argument of relevance() must be a string literal \
                 (the ranking query), got: {other}"
            );
        }
    };
    Ok(Some((args[0].clone(), query)))
}
