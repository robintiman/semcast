//! Turns `meaning_of()` markers into [`SemClusterNode`]s — the clustering
//! half of the classify/rank/cluster roadmap item.
//!
//! Which group a row belongs to is a fact about the whole relation, so the
//! marker cannot be evaluated where it is written. The rule inserts a node
//! that materializes the label below the `Aggregate` (or `Projection`) that
//! holds the marker, and swaps the marker for a reference to that column.
//! `GROUP BY` then groups by an ordinary column.
//!
//! The one subtlety is the alias. Replacing a group expression changes the
//! aggregate's output field name, which anything above it refers to by name.
//! So the replacement column is aliased back to the name the marker had, and
//! the schema above is left exactly as it was.
//!
//! [`SemClusterNode`]: crate::logical::SemClusterNode

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{Aggregate, Expr, Extension, LogicalPlan, Projection};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::logical::SemClusterNode;
use crate::sql::cluster_udf::MEANING_OF_UDF_NAME;

const NOT_A_GROUP_KEY: &str = "MEANING OF is supported as a GROUP BY key or in a SELECT list; \
     it cannot appear in an aggregate, a WHERE clause, or a JOIN condition — \
     wrap it in a subquery to filter on a group label";

/// Turns `meaning_of()` markers into `SemCluster` nodes below the aggregate
/// or projection that holds them.
#[derive(Debug, Default)]
pub struct ClusterRewriteRule;

impl OptimizerRule for ClusterRewriteRule {
    fn name(&self) -> &str {
        "semcast_cluster_rewrite"
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
            LogicalPlan::Projection(projection) => rewrite_projection(projection),
            other => {
                if exprs_contain_marker(&other.expressions())? {
                    return plan_err!("{NOT_A_GROUP_KEY}");
                }
                Ok(Transformed::no(other))
            }
        }
    }
}

fn rewrite_aggregate(aggregate: Aggregate) -> Result<Transformed<LogicalPlan>> {
    // An aggregate over a group label is a category error: the label is what
    // the rows are grouped *by*, not something to fold them with.
    if exprs_contain_marker(&aggregate.aggr_expr)? {
        return plan_err!("{NOT_A_GROUP_KEY}");
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

/// Accumulates the distinct `(text, k)` groups one plan node references, then
/// materializes them into stacked [`SemClusterNode`]s.
#[derive(Default)]
struct Builder {
    groups: Vec<Group>,
}

struct Group {
    text: Expr,
    k: Option<usize>,
    column: Option<Column>,
}

impl Builder {
    /// Record every marker in `expr`, deduplicating by `(text, k)` so the
    /// same clustering asked for twice costs one pass.
    fn observe(&mut self, expr: &Expr) -> Result<()> {
        let mut found = Vec::new();
        collect_markers(expr, &mut found)?;
        for (text, k) in found {
            if !self.groups.iter().any(|g| g.text == text && g.k == k) {
                self.groups.push(Group {
                    text,
                    k,
                    column: None,
                });
            }
        }
        Ok(())
    }

    fn build(&mut self, input: LogicalPlan) -> Result<LogicalPlan> {
        let mut plan = input;
        for (id, group) in self.groups.iter_mut().enumerate() {
            let node = SemClusterNode::try_new(plan, group.text.clone(), group.k, id)?;
            group.column = Some(node.label_column());
            plan = LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            });
        }
        Ok(plan)
    }

    /// Swap every marker in `expr` for its node's label column.
    fn replace(&self, expr: Expr) -> Result<Expr> {
        expr.transform(|e| {
            let Some((text, k)) = parse_marker(&e)? else {
                return Ok(Transformed::no(e));
            };
            let column = self
                .groups
                .iter()
                .find(|g| g.text == text && g.k == k)
                .and_then(|g| g.column.clone())
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(
                        "cluster node missing for a collected meaning_of() marker".to_owned(),
                    )
                })?;
            Ok(Transformed::yes(Expr::Column(column)))
        })
        .map(|t| t.data)
    }
}

fn is_marker(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == MEANING_OF_UDF_NAME)
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

fn collect_markers(expr: &Expr, out: &mut Vec<(Expr, Option<usize>)>) -> Result<()> {
    expr.apply(|e| {
        if let Some(marker) = parse_marker(e)? {
            out.push(marker);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map(|_| ())
}

/// Pull `(text_expr, k)` out of a `meaning_of(..)` call. A `k` of zero is the
/// sentinel the parser writes for an absent `INTO`, meaning "sweep for one".
fn parse_marker(expr: &Expr) -> Result<Option<(Expr, Option<usize>)>> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return Ok(None);
    };
    if func.name() != MEANING_OF_UDF_NAME {
        return Ok(None);
    }
    if !(1..=2).contains(&args.len()) {
        return plan_err!("meaning_of() takes 1 or 2 arguments, got {}", args.len());
    }
    let k = match args.get(1) {
        None => None,
        // The parser's "no explicit INTO" sentinel.
        Some(Expr::Literal(ScalarValue::Int64(Some(-1)), _)) => None,
        Some(Expr::Literal(ScalarValue::Int64(Some(k)), _)) if *k > 0 => Some(*k as usize),
        Some(other) => {
            return plan_err!("INTO takes a positive number of groups, got: {other}");
        }
    };
    Ok(Some((args[0].clone(), k)))
}
