//! Rewrites `means(text, 'condition')` into the extension node its position
//! calls for: a [`SemFilterNode`] in a `WHERE`, a [`SemClassifyNode`] in a
//! `SELECT` list.
//!
//! One rule owns both because one rule has to own *where `means()` may
//! appear* — split across two, they would have to agree on which one reports
//! the error for every other position. The classify machinery itself lives in
//! [`crate::optimizer::classify`].
//!
//! [`SemFilterNode`]: crate::logical::SemFilterNode
//! [`SemClassifyNode`]: crate::logical::SemClassifyNode

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::utils::{conjunction, split_conjunction, split_conjunction_owned};
use datafusion::logical_expr::{Expr, Extension, Filter, LogicalPlan, Projection};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::index::registry::SemcastRuntime;
use crate::logical::sem_classify::branch_column_name;
use crate::logical::{SemClassifyNode, SemFilterNode};
use crate::sql::confidence_udf::CONFIDENCE_UDF_NAME;
use crate::sql::means_udf::MEANS_UDF_NAME;

/// Finds `means(..)` calls inside `Filter` predicates, splits them out of the
/// conjunction (the free predicates stay in a `Filter` below, which the
/// optimizer will keep pushing down — predicate ordering is just predicate
/// ordering), and stacks a `SemFilter` extension node on top.
///
/// In a `Projection` the same marker becomes a `SemClassify` node instead:
/// filtering drops rows, labelling keeps them, so the two positions want
/// different operators.
///
/// Stacked semantic predicates are ordered before they are stacked: one that
/// a semantic index covers prunes rows for free before the model sees them,
/// so it belongs below one that pays full price for every row it is handed.
/// Source order breaks ties within a tier, keeping the rewrite deterministic.
///
/// Restriction: `means()` is supported as a top-level `AND` conjunct of a
/// `WHERE` clause, or anywhere in a `SELECT` list. Elsewhere — under
/// `OR`/`NOT` in a `WHERE`, in a `GROUP BY`, with a non-literal condition —
/// is a plan-time error rather than a silent model call per row.
///
/// `CONFIDENCE(text MEANS 'condition')` is a number, not a predicate, so it
/// goes in a `SELECT` list, anywhere in a `WHERE` (`CONFIDENCE(..) > 0.8`),
/// or in an `ORDER BY`. In a `WHERE` or `ORDER BY` it is computed by a
/// classify node below the filter or sort, and a projection drops the column
/// again.
#[derive(Debug)]
pub struct MeansRewriteRule {
    /// Holds the session runtime directly because `OptimizerConfig` exposes no
    /// route to a `SessionConfig` extension. The index map behind it is
    /// mutable and shared, so a `CREATE SEMANTIC INDEX` issued after the
    /// context was built is visible here.
    runtime: Arc<SemcastRuntime>,
}

impl MeansRewriteRule {
    pub fn new(runtime: Arc<SemcastRuntime>) -> Self {
        Self { runtime }
    }
}

/// Where a marker still cannot go, named in every rejection.
const LEGAL_POSITIONS: &str = "MEANS and CONFIDENCE() work anywhere an expression does in a \
     single-input step (WHERE, SELECT, GROUP BY, HAVING, ORDER BY, window and \
     aggregate arguments), but not where they would have to stay inside a join: \
     a condition on an outer join's preserved side, or one that shares an OR \
     with a correlated subquery's correlation. Compute it as a column in a \
     subquery first, then join or filter on that column";

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
        for expr in plan.expressions() {
            reject_nested_markers(&expr)?;
        }
        // A MEANS in a SELECT list labels rows rather than dropping them.
        if let LogicalPlan::Projection(projection) = plan {
            return crate::optimizer::classify::rewrite_projection(projection);
        }

        let LogicalPlan::Filter(filter) = plan else {
            let mut holds_marker = false;
            for expr in plan.expressions() {
                holds_marker |= contains_means(&expr)?;
            }
            if !holds_marker {
                return Ok(Transformed::no(plan));
            }
            return lift_below(plan);
        };
        let mut lifts = false;
        for conjunct in split_conjunction(&filter.predicate) {
            lifts |= !is_means_call(conjunct) && contains_means(conjunct)?;
        }
        if lifts {
            return self.rewrite_lifted_filter(filter);
        }
        self.rewrite_filter(filter)
    }
}

/// A step that uses a `MEANS` or `CONFIDENCE()` as a value — a `GROUP BY`
/// key, an aggregate or window argument, an `ORDER BY` — rather than as a
/// filter: compute each one as a column in a classify node below the step,
/// use the column, and drop it again above if the step passes its input's
/// columns through.
///
/// One model call per row reaching the step, shown by `EXPLAIN`'s
/// `SemClassify`. Names are preserved, so the steps above still find the
/// columns they reference.
fn lift_below(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let inputs = plan.inputs();
    let [input] = inputs.as_slice() else {
        return plan_err!("{LEGAL_POSITIONS} (found it in a {} node)", plan.display());
    };
    // A grouping set's keys sit inside one expression, so their names can't
    // be preserved one by one, and the projection above would lose them.
    if let LogicalPlan::Aggregate(aggregate) = &plan {
        for group in &aggregate.group_expr {
            if matches!(group, Expr::GroupingSet(_)) && contains_means(group)? {
                return plan_err!(
                    "MEANS and CONFIDENCE() can't be a ROLLUP, CUBE or GROUPING SETS key: \
                     compute the label as a column in a subquery, then group by it"
                );
            }
        }
    }
    let input = (*input).clone();
    let original = Arc::clone(plan.schema());
    let preserver = NamePreserver::new(&plan);
    let mut builder = crate::optimizer::classify::Builder::default();
    let exprs = plan
        .expressions()
        .into_iter()
        .map(|expr| {
            let saved = preserver.save(&expr);
            let lifted = crate::optimizer::classify::rewrite_expr(expr, &None, &mut builder)?;
            Ok(saved.restore(lifted))
        })
        .collect::<Result<Vec<_>>>()?;
    let lifted = plan.with_new_exprs(exprs, vec![builder.build(input)?])?;
    if lifted.schema().fields().len() == original.fields().len() {
        return Ok(Transformed::yes(lifted));
    }
    // The step passed the new columns through (a sort, a window): restore
    // the schema it had.
    let passthrough = original.columns().into_iter().map(Expr::Column).collect();
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(passthrough, Arc::new(lifted))?,
    )))
}

/// For a conjunct `d1 OR d2 OR …`, the rows still worth asking about: those
/// no marker-free disjunct already matches. `None` when there is nothing to
/// skip on.
fn or_guard(conjunct: &Expr) -> Result<Option<Expr>> {
    let mut disjuncts = Vec::new();
    collect_disjuncts(conjunct, &mut disjuncts);
    if disjuncts.len() < 2 {
        return Ok(None);
    }
    let mut unmatched = Vec::new();
    for disjunct in disjuncts {
        // A volatile disjunct is evaluated twice, in the guard and in the
        // filter, and could disagree: skip a row, then not match it either.
        if !contains_means(disjunct)? && !disjunct.is_volatile() {
            unmatched.push(disjunct.clone().is_not_true());
        }
    }
    Ok(conjunction(unmatched))
}

fn collect_disjuncts<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == datafusion::logical_expr::Operator::Or => {
            collect_disjuncts(&binary.left, out);
            collect_disjuncts(&binary.right, out);
        }
        other => out.push(other),
    }
}

fn passthrough_columns(plan: &LogicalPlan) -> Vec<Expr> {
    plan.schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect()
}

impl MeansRewriteRule {
    /// A `WHERE` with a `MEANS` used as a value rather than as a top-level
    /// `AND` condition: under `NOT` or `OR`, compared, inside a `CASE`, or
    /// read through `CONFIDENCE()`.
    ///
    /// The conjuncts that don't need lifting run first — ordinary predicates
    /// and plain `MEANS` conditions alike, through [`Self::rewrite_filter`],
    /// index and all — so the lifted ones are paid for only on rows that
    /// survive them. Each lifted `MEANS` becomes a column of a classify node,
    /// the rest of the predicate filters on those columns, and a projection
    /// restores the input schema on top.
    ///
    /// In `a OR x MEANS p`, a row an ordinary disjunct already matches skips
    /// the model. A row whose call fails reads NULL, so SQL's three-valued
    /// logic applies: it is dropped under `NOT`, and kept under `OR` only if
    /// another disjunct is true.
    fn rewrite_lifted_filter(&self, filter: Filter) -> Result<Transformed<LogicalPlan>> {
        let passthrough = passthrough_columns(&filter.input);
        let mut lifted = Vec::new();
        let mut first = Vec::new();
        for conjunct in split_conjunction_owned(filter.predicate) {
            if !is_means_call(&conjunct) && contains_means(&conjunct)? {
                lifted.push(conjunct);
            } else {
                first.push(conjunct);
            }
        }
        let mut input = Arc::unwrap_or_clone(filter.input);
        if let Some(predicate) = conjunction(first) {
            input = self
                .rewrite_filter(Filter::try_new(predicate, Arc::new(input))?)?
                .data;
        }
        let mut builder = crate::optimizer::classify::Builder::default();
        let lifted = lifted
            .into_iter()
            .map(|conjunct| {
                let guard = or_guard(&conjunct)?;
                crate::optimizer::classify::rewrite_expr(conjunct, &guard, &mut builder)
            })
            .collect::<Result<Vec<_>>>()?;
        let predicate = conjunction(lifted).expect("at least one conjunct lifts");
        let input = builder.build(input)?;
        let filtered = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?);
        Ok(Transformed::yes(LogicalPlan::Projection(
            Projection::try_new(passthrough, Arc::new(filtered))?,
        )))
    }

    fn rewrite_filter(&self, filter: Filter) -> Result<Transformed<LogicalPlan>> {
        let (semantic, free): (Vec<Expr>, Vec<Expr>) =
            split_conjunction_owned(filter.predicate.clone())
                .into_iter()
                .partition(is_means_call);

        if semantic.is_empty() {
            if contains_means(&filter.predicate)? {
                return plan_err!("{LEGAL_POSITIONS}");
            }
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        }
        for expr in &free {
            if contains_means(expr)? {
                return plan_err!("{LEGAL_POSITIONS}");
            }
        }

        let schema = Arc::clone(filter.input.schema());
        let mut specs = Vec::with_capacity(semantic.len());
        for call in semantic {
            specs.push(destructure_means(call)?);
        }
        let mut groups = group_by_text(specs);
        for group in &mut groups {
            group.indexed = self.runtime.covers(&group.text, &schema);
        }
        // An indexed group prunes before it pays, so it goes innermost.
        // `partition` preserves relative order, so equally-priced groups keep
        // the order they were written in.
        let (indexed, unindexed): (Vec<_>, Vec<_>) =
            groups.into_iter().partition(|group| group.indexed);

        // Free predicates stay in a Filter below the semantic stage, so they
        // run first and DataFusion keeps optimizing them as usual.
        let mut rewritten = match conjunction(free) {
            Some(predicate) => {
                LogicalPlan::Filter(Filter::try_new(predicate, Arc::clone(&filter.input))?)
            }
            None => Arc::unwrap_or_clone(filter.input),
        };
        // Each stage wraps the previous one, so the first built runs first.
        for (id, group) in indexed.into_iter().chain(unindexed).enumerate() {
            rewritten = if group.fusable() {
                fuse(rewritten, group, id)?
            } else {
                stack(rewritten, group)
            };
        }
        Ok(Transformed::yes(rewritten))
    }
}

/// Semantic predicates sharing one text expression — the unit the rewrite
/// orders, and the unit it can fuse into a single model call.
struct TextGroup {
    text: Expr,
    /// `(condition, recall, confidence)` in source order, duplicates kept:
    /// the unfused path stacks exactly what was written.
    specs: Vec<(String, Option<f64>, Option<f64>)>,
    /// Whether a semantic index covers `text`.
    indexed: bool,
}

impl TextGroup {
    /// Conditions in first-appearance order, asked once each. Writing the
    /// same condition twice is one question, not two.
    fn distinct_conditions(&self) -> Vec<String> {
        let mut conditions: Vec<String> = Vec::new();
        for (condition, _, _) in &self.specs {
            if !conditions.contains(condition) {
                conditions.push(condition.clone());
            }
        }
        conditions
    }

    /// Whether asking every condition at once beats asking them in sequence.
    ///
    /// Fused costs exactly one call per row with non-NULL text; stacked costs
    /// that plus one per surviving row per later predicate, so fusion never
    /// loses on call count — *except* against an index, which prunes rows
    /// before any call at all and so beats a floor of one-per-row. A classify
    /// has no index stage by design, so an indexed group keeps its stack.
    ///
    /// `WITH RECALL` calibrates that index stage, so a group carrying one has
    /// nothing to gain here and is left alone rather than having its target
    /// silently dropped.
    fn fusable(&self) -> bool {
        !self.indexed
            && self.specs.iter().all(|(_, recall, _)| recall.is_none())
            && self.distinct_conditions().len() > 1
    }
}

/// Collect predicates by the text they read, in first-appearance order.
fn group_by_text(specs: Vec<(Expr, String, Option<f64>, Option<f64>)>) -> Vec<TextGroup> {
    let mut groups: Vec<TextGroup> = Vec::new();
    for (text, condition, recall, confidence) in specs {
        match groups.iter_mut().find(|group| group.text == text) {
            Some(group) => group.specs.push((condition, recall, confidence)),
            None => groups.push(TextGroup {
                text,
                specs: vec![(condition, recall, confidence)],
                indexed: false,
            }),
        }
    }
    groups
}

/// One `SemFilter` per predicate, in the order they were written.
fn stack(input: LogicalPlan, group: TextGroup) -> LogicalPlan {
    let TextGroup { text, specs, .. } = group;
    let mut plan = input;
    for (condition, recall, confidence) in specs {
        plan = LogicalPlan::Extension(Extension {
            node: Arc::new(SemFilterNode::new(
                plan,
                text.clone(),
                condition,
                recall,
                confidence,
            )),
        });
    }
    plan
}

/// Every condition in one model call: a `SemClassify` materializes one boolean
/// per condition, a `Filter` demands all of them, and a projection drops the
/// booleans again.
///
/// The projection is not tidiness. It restores the input schema, so the rest
/// of the plan sees what it expects — and it frees the `__sem_class_*` names
/// again, which is what stops these columns colliding with a `SemClassify`
/// that a `MEANS` in the same query's `SELECT` list builds above us.
fn fuse(input: LogicalPlan, group: TextGroup, id: usize) -> Result<LogicalPlan> {
    let conditions = group.distinct_conditions();
    let passthrough: Vec<Expr> = input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    let classify = LogicalPlan::Extension(Extension {
        node: Arc::new(SemClassifyNode::try_new(
            input,
            group.text,
            conditions.clone(),
            None,
            id,
        )?),
    });
    let predicate = conjunction(
        (0..conditions.len())
            .map(|branch| Expr::Column(Column::new_unqualified(branch_column_name(id, branch)))),
    )
    .expect("a fusable group has more than one condition");
    let filtered = LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(classify))?);
    Ok(LogicalPlan::Projection(Projection::try_new(
        passthrough,
        Arc::new(filtered),
    )?))
}

/// Attach a statement-level `WITH RECALL` target to every `means()` call in
/// the plan, as a third literal argument the rewrite reads back out. Zero
/// calls is a user mistake, not a no-op.
pub fn apply_recall(
    plan: LogicalPlan,
    recall: f64,
    confidence: Option<f64>,
) -> Result<LogicalPlan> {
    // A confidence is a claim about a recall target; on its own there is
    // nothing for it to qualify.
    if let Some(confidence) = confidence {
        if recall >= 1.0 {
            return plan_err!(
                "WITH RECALL 1 cannot be certified: no finite sample proves that \
                 every match survives. Ask for a target below 1 (0.99, say), or \
                 drop WITH CONFIDENCE {confidence} for a best-effort estimate"
            );
        }
    }
    // Only a MEANS that filters — a top-level AND condition of a WHERE,
    // HAVING or inner join's ON — has a lossy pre-filter to calibrate. One used as a value
    // (a label, under NOT or OR, inside CONFIDENCE()) asks every row it sees
    // and is exact already.
    let mut rewrites = 0usize;
    let mut attach = |predicate: Expr| -> (Expr, bool) {
        let mut changed = false;
        let conjuncts = split_conjunction_owned(predicate)
            .into_iter()
            .map(|conjunct| match conjunct {
                Expr::ScalarFunction(mut call)
                    if call.func.name() == MEANS_UDF_NAME && call.args.len() == 2 =>
                {
                    call.args
                        .push(Expr::Literal(ScalarValue::Float64(Some(recall)), None));
                    // Positional: the fourth argument is the confidence, and
                    // `destructure_means` reads it back out in the same order.
                    call.args
                        .push(Expr::Literal(ScalarValue::Float64(confidence), None));
                    rewrites += 1;
                    changed = true;
                    Expr::ScalarFunction(call)
                }
                other => other,
            })
            .collect::<Vec<_>>();
        let predicate = conjunction(conjuncts).expect("a predicate has a conjunct");
        (predicate, changed)
    };
    let transformed = plan.transform_up(|plan| match plan {
        LogicalPlan::Filter(filter) => {
            let (predicate, changed) = attach(filter.predicate);
            let filter = LogicalPlan::Filter(Filter::try_new(predicate, filter.input)?);
            Ok(if changed {
                Transformed::yes(filter)
            } else {
                Transformed::no(filter)
            })
        }
        // An inner join's ON condition becomes a filter on its own; the
        // optimizer moves it there before the MEANS rewrite sees it.
        LogicalPlan::Join(mut join)
            if join.join_type == datafusion::logical_expr::JoinType::Inner
                && join.filter.is_some() =>
        {
            let (predicate, changed) = attach(join.filter.take().expect("checked above"));
            join.filter = Some(predicate);
            Ok(Transformed::new(
                LogicalPlan::Join(join),
                changed,
                TreeNodeRecursion::Continue,
            ))
        }
        other => Ok(Transformed::no(other)),
    })?;
    if rewrites == 0 {
        return plan_err!(
            "WITH RECALL requires a MEANS that filters: a top-level AND condition \
             of a WHERE (a label, or a MEANS under NOT or OR, asks every row and \
             has nothing to calibrate)"
        );
    }
    Ok(transformed.data)
}

pub(crate) fn is_means_call(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == MEANS_UDF_NAME)
}

pub(crate) fn is_confidence_call(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == CONFIDENCE_UDF_NAME)
}

/// Any semcast marker that must be rewritten before execution: a `means()`,
/// or a `confidence()` (which wraps one).
pub(crate) fn contains_means(expr: &Expr) -> Result<bool> {
    expr.exists(|e| Ok(is_means_call(e) || is_confidence_call(e)))
}

/// A `MEANS` or `CONFIDENCE()` must read text from the table, never another
/// marker's output: the model would be asked about a probability or a verdict.
fn reject_nested_markers(expr: &Expr) -> Result<()> {
    // A CONFIDENCE() wraps a means(), so checking every means()'s text covers
    // both markers.
    expr.apply(|e| {
        if let Expr::ScalarFunction(call) = e {
            if is_means_call(e) {
                if let Some(text) = call.args.first() {
                    if contains_means(text)? {
                        return plan_err!(
                            "a MEANS or CONFIDENCE() cannot read the output of \
                             another MEANS or CONFIDENCE(); give it a text column"
                        );
                    }
                }
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

/// Pull `(text_expr, condition)` out of a `confidence(means(..))` call.
pub(crate) fn destructure_confidence(expr: Expr) -> Result<(Expr, String)> {
    let Expr::ScalarFunction(ScalarFunction { mut args, .. }) = expr else {
        unreachable!("caller checked is_confidence_call");
    };
    if args.len() != 1 || !is_means_call(&args[0]) {
        return plan_err!("CONFIDENCE() takes one MEANS: CONFIDENCE(text MEANS 'condition')");
    }
    let (text, condition, recall, _confidence) = destructure_means(args.swap_remove(0))?;
    if recall.is_some() {
        return plan_err!(
            "WITH RECALL calibrates the index pre-filter for a MEANS in a WHERE \
             clause; CONFIDENCE() asks every row it sees, so there is nothing \
             to calibrate"
        );
    }
    Ok((text, condition))
}

/// Pull `(text_expr, condition, recall, confidence)` out of a validated
/// `means(..)` call.
pub(crate) fn destructure_means(expr: Expr) -> Result<(Expr, String, Option<f64>, Option<f64>)> {
    let Expr::ScalarFunction(ScalarFunction { args, .. }) = expr else {
        unreachable!("caller checked is_means_call");
    };
    if !(2..=4).contains(&args.len()) {
        return plan_err!("means() takes 2 to 4 arguments, got {}", args.len());
    }
    let mut args = args.into_iter();
    let text = args.next().expect("length checked above");
    // Normalized here, where every MEANS enters, so the filter, the label,
    // CONFIDENCE() and calibration all ask — and store — one question.
    let condition = match args.next().expect("length checked above") {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => {
            let condition = crate::semir::normalize_instruction(&s);
            if condition.is_empty() {
                return plan_err!(
                    "a MEANS condition cannot be empty: say what the text should mean"
                );
            }
            condition
        }
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
    let confidence = match args.next() {
        None | Some(Expr::Literal(ScalarValue::Float64(None), _)) => None,
        Some(Expr::Literal(ScalarValue::Float64(Some(c)), _)) if c > 0.0 && c < 1.0 => Some(c),
        Some(other) => {
            return plan_err!(
                "the fourth argument of means() must be a confidence in (0, 1) \
                 as a float literal, got: {other}"
            );
        }
    };
    Ok((text, condition, recall, confidence))
}
