//! Turns `semantic_key()` markers into [`SemDistinctNode`]s — the dedupe half
//! of the classify/rank/cluster roadmap item.
//!
//! Whether a row is a duplicate is a fact about the rows around it, so the
//! marker cannot be evaluated where it is written. The rule inserts a node
//! that materializes the group key below whatever holds the marker, and swaps
//! the marker for a reference to that column. The deduplication itself stays
//! DataFusion's.
//!
//! Which node that is depends on how far the optimizer has already got:
//! `ReplaceDistinctWithAggregate` lowers `DISTINCT ON` into an `Aggregate`
//! over `first_value` before this rule runs, so both shapes are handled.
//! Rewriting the aggregate keeps its output field names by aliasing the
//! replacement back — the same care [`ClusterRewriteRule`] takes.
//!
//! [`SemDistinctNode`]: crate::logical::SemDistinctNode
//! [`ClusterRewriteRule`]: crate::optimizer::cluster::ClusterRewriteRule

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{
    Aggregate, Distinct, DistinctOn, Expr, Extension, LogicalPlan, Projection,
};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::index::dedupe::DEFAULT_SIMILARITY;
use crate::logical::SemDistinctNode;
use crate::sql::distinct_udf::SEMANTIC_KEY_UDF_NAME;

const NOT_A_DISTINCT_KEY: &str = "semantic_key() is supported in a DISTINCT ON list or a SELECT list; it \
     cannot appear in a WHERE clause, a JOIN condition, or an aggregate — wrap \
     it in a subquery to filter on a duplicate key";

/// Turns `semantic_key()` markers into `SemDistinct` nodes below whatever
/// holds them.
#[derive(Debug, Default)]
pub struct DistinctRewriteRule;

impl OptimizerRule for DistinctRewriteRule {
    fn name(&self) -> &str {
        "semcast_distinct_rewrite"
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
        match plan {
            LogicalPlan::Aggregate(aggregate) => rewrite_aggregate(aggregate),
            LogicalPlan::Distinct(Distinct::On(distinct)) => rewrite_distinct_on(distinct),
            LogicalPlan::Projection(projection) => rewrite_projection(projection),
            other => {
                if exprs_contain_marker(&other.expressions())? {
                    return plan_err!("{NOT_A_DISTINCT_KEY}");
                }
                Ok(Transformed::no(other))
            }
        }
    }
}

/// `DISTINCT ON` as DataFusion first plans it, before the lowering rule gets
/// to it.
fn rewrite_distinct_on(distinct: DistinctOn) -> Result<Transformed<LogicalPlan>> {
    if !exprs_contain_marker(&distinct.on_expr)? {
        return Ok(Transformed::no(LogicalPlan::Distinct(Distinct::On(
            distinct,
        ))));
    }
    let mut builder = Builder::default();
    for expr in &distinct.on_expr {
        builder.observe(expr)?;
    }
    let input = builder.build((*distinct.input).clone())?;

    let on_expr = distinct
        .on_expr
        .into_iter()
        .map(|expr| builder.replace(expr))
        .collect::<Result<Vec<_>>>()?;

    Ok(Transformed::yes(LogicalPlan::Distinct(Distinct::On(
        DistinctOn::try_new(
            on_expr,
            distinct.select_expr,
            distinct.sort_expr,
            Arc::new(input),
        )?,
    ))))
}

/// The lowered form: `DISTINCT ON` becomes an `Aggregate` grouping by the ON
/// expressions, with `first_value` picking a surviving row.
fn rewrite_aggregate(aggregate: Aggregate) -> Result<Transformed<LogicalPlan>> {
    if exprs_contain_marker(&aggregate.aggr_expr)? {
        return plan_err!("{NOT_A_DISTINCT_KEY}");
    }
    if !exprs_contain_marker(&aggregate.group_expr)? {
        return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
    }

    let mut builder = Builder::default();
    for expr in &aggregate.group_expr {
        builder.observe(expr)?;
    }
    let input = builder.build((*aggregate.input).clone())?;

    let group_expr = aggregate
        .group_expr
        .into_iter()
        .map(|expr| {
            if !contains_marker(&expr)? {
                return Ok(expr);
            }
            // Alias the replacement back to the marker's schema name so this
            // aggregate's output columns keep the names their consumers use.
            let name = expr.schema_name().to_string();
            Ok(builder.replace(expr)?.alias(name))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Transformed::yes(LogicalPlan::Aggregate(
        Aggregate::try_new(Arc::new(input), group_expr, aggregate.aggr_expr)?,
    )))
}

fn rewrite_projection(projection: Projection) -> Result<Transformed<LogicalPlan>> {
    if !exprs_contain_marker(&projection.expr)? {
        return Ok(Transformed::no(LogicalPlan::Projection(projection)));
    }

    let mut builder = Builder::default();
    for expr in &projection.expr {
        builder.observe(expr)?;
    }
    let input = builder.build((*projection.input).clone())?;

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

/// Attach a statement-level `WITH SIMILARITY` to every `semantic_key()` call
/// in the plan, as a second literal argument the rewrite reads back out.
/// Zero calls is a user mistake, not a no-op.
pub fn apply_similarity(plan: LogicalPlan, similarity: f64) -> Result<LogicalPlan> {
    let mut rewrites = 0usize;
    let transformed = plan.transform_up(|plan| {
        plan.map_expressions(|expr| {
            expr.transform_up(|expr| match expr {
                Expr::ScalarFunction(mut call)
                    if call.func.name() == SEMANTIC_KEY_UDF_NAME && call.args.len() == 1 =>
                {
                    call.args
                        .push(Expr::Literal(ScalarValue::Float64(Some(similarity)), None));
                    rewrites += 1;
                    Ok(Transformed::yes(Expr::ScalarFunction(call)))
                }
                other => Ok(Transformed::no(other)),
            })
        })
    })?;
    if rewrites == 0 {
        return plan_err!("WITH SIMILARITY requires a SEMANTIC DISTINCT ON in the statement");
    }
    Ok(transformed.data)
}

/// Accumulates the distinct `(text, similarity)` groups one plan node
/// references, then materializes them into stacked [`SemDistinctNode`]s.
#[derive(Default)]
struct Builder {
    groups: Vec<Group>,
}

struct Group {
    text: Expr,
    similarity: f32,
    column: Option<Column>,
}

impl Builder {
    fn observe(&mut self, expr: &Expr) -> Result<()> {
        let mut found = Vec::new();
        collect_markers(expr, &mut found)?;
        for (text, similarity) in found {
            if !self
                .groups
                .iter()
                .any(|g| g.text == text && g.similarity.to_bits() == similarity.to_bits())
            {
                self.groups.push(Group {
                    text,
                    similarity,
                    column: None,
                });
            }
        }
        Ok(())
    }

    fn build(&mut self, input: LogicalPlan) -> Result<LogicalPlan> {
        let mut plan = input;
        for (id, group) in self.groups.iter_mut().enumerate() {
            let node = SemDistinctNode::try_new(plan, group.text.clone(), group.similarity, id)?;
            group.column = Some(node.key_column());
            plan = LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            });
        }
        Ok(plan)
    }

    /// Swap every marker in `expr` for its node's key column.
    fn replace(&self, expr: Expr) -> Result<Expr> {
        expr.transform(|e| {
            let Some((text, similarity)) = parse_marker(&e)? else {
                return Ok(Transformed::no(e));
            };
            let column = self
                .groups
                .iter()
                .find(|g| g.text == text && g.similarity.to_bits() == similarity.to_bits())
                .and_then(|g| g.column.clone())
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(
                        "dedupe node missing for a collected semantic_key() marker".to_owned(),
                    )
                })?;
            Ok(Transformed::yes(Expr::Column(column)))
        })
        .map(|t| t.data)
    }
}

fn is_marker(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == SEMANTIC_KEY_UDF_NAME)
}

fn contains_marker(expr: &Expr) -> Result<bool> {
    expr.exists(|e| Ok(is_marker(e)))
}

fn exprs_contain_marker(exprs: &[Expr]) -> Result<bool> {
    for expr in exprs {
        if contains_marker(expr)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect_markers(expr: &Expr, out: &mut Vec<(Expr, f32)>) -> Result<()> {
    expr.apply(|e| {
        if let Some(marker) = parse_marker(e)? {
            out.push(marker);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map(|_| ())
}

/// Pull `(text_expr, similarity)` out of a `semantic_key(..)` call. Without a
/// `WITH SIMILARITY` the threshold is the default.
fn parse_marker(expr: &Expr) -> Result<Option<(Expr, f32)>> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return Ok(None);
    };
    if func.name() != SEMANTIC_KEY_UDF_NAME {
        return Ok(None);
    }
    if !(1..=2).contains(&args.len()) {
        return plan_err!("semantic_key() takes 1 or 2 arguments, got {}", args.len());
    }
    let similarity = match args.get(1) {
        None => DEFAULT_SIMILARITY,
        // Re-validated here because direct semantic_key() callers bypass the
        // WITH SIMILARITY parser's range check.
        Some(Expr::Literal(ScalarValue::Float64(Some(s)), _)) if *s > 0.0 && *s <= 1.0 => *s as f32,
        Some(other) => {
            return plan_err!(
                "the second argument of semantic_key() must be a similarity in (0, 1] \
                 as a float literal, got: {other}"
            );
        }
    };
    Ok(Some((args[0].clone(), similarity)))
}
