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
use datafusion::logical_expr::utils::{conjunction, split_conjunction_owned};
use datafusion::logical_expr::{Expr, Extension, Filter, LogicalPlan, Projection, Sort, SortExpr};
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

/// Both legal positions, named in every rejection so the error says where the
/// marker *can* go rather than only where it cannot.
const LEGAL_POSITIONS: &str = "means() is supported as a top-level AND conjunct of a WHERE \
     clause (to filter) or in a SELECT list (to label); it cannot appear under \
     OR or NOT in a WHERE, nor in a GROUP BY, HAVING, or JOIN condition — wrap \
     it in a subquery to group or filter on a label. CONFIDENCE(text MEANS \
     'condition') goes in a SELECT list, a WHERE clause, or an ORDER BY, but \
     not inside an aggregate or window function — compute it in a subquery \
     and aggregate over that";

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

        if let LogicalPlan::Sort(sort) = &plan {
            let mut holds_confidence = false;
            for sort_expr in &sort.expr {
                holds_confidence |= contains_confidence(&sort_expr.expr)?;
            }
            if holds_confidence {
                let LogicalPlan::Sort(sort) = plan else {
                    unreachable!("matched above")
                };
                return rewrite_confidence_sort(sort);
            }
        }

        let LogicalPlan::Filter(filter) = plan else {
            for expr in plan.expressions() {
                if contains_means(&expr)? {
                    return plan_err!("{LEGAL_POSITIONS} (found it in a {} node)", plan.display());
                }
            }
            return Ok(Transformed::no(plan));
        };
        if contains_confidence(&filter.predicate)? {
            return self.rewrite_confidence_filter(filter);
        }
        self.rewrite_filter(filter)
    }
}

/// An `ORDER BY` that reads `CONFIDENCE()`: compute it below the sort, sort
/// on the column, and drop the column again above.
fn rewrite_confidence_sort(sort: Sort) -> Result<Transformed<LogicalPlan>> {
    let passthrough = passthrough_columns(&sort.input);
    let mut builder = crate::optimizer::classify::Builder::default();
    let expr = sort
        .expr
        .into_iter()
        .map(|sort_expr| {
            let expr = crate::optimizer::classify::lift_confidence(sort_expr.expr, &mut builder)?;
            Ok(SortExpr { expr, ..sort_expr })
        })
        .collect::<Result<Vec<_>>>()?;
    for sort_expr in &expr {
        if contains_means(&sort_expr.expr)? {
            return plan_err!("{LEGAL_POSITIONS} (found it in an ORDER BY)");
        }
    }
    let input = builder.build(Arc::unwrap_or_clone(sort.input))?;
    let sorted = LogicalPlan::Sort(Sort {
        expr,
        input: Arc::new(input),
        fetch: sort.fetch,
    });
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(passthrough, Arc::new(sorted))?,
    )))
}

fn passthrough_columns(plan: &LogicalPlan) -> Vec<Expr> {
    plan.schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect()
}

impl MeansRewriteRule {
    /// A `WHERE` that reads `CONFIDENCE()`. The conjuncts that don't read it
    /// run first — ordinary predicates and `MEANS` alike, through
    /// [`Self::rewrite_filter`] — so the probability is paid for only on rows
    /// that survive them. Then each probability is computed in a classify
    /// node, the rest of the predicate filters on its column, and a
    /// projection restores the input schema on top.
    fn rewrite_confidence_filter(&self, filter: Filter) -> Result<Transformed<LogicalPlan>> {
        let passthrough = passthrough_columns(&filter.input);
        let mut scored = Vec::new();
        let mut first = Vec::new();
        for conjunct in split_conjunction_owned(filter.predicate) {
            if contains_confidence(&conjunct)? {
                scored.push(conjunct);
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
        let predicate = crate::optimizer::classify::lift_confidence(
            conjunction(scored).expect("the predicate reads CONFIDENCE()"),
            &mut builder,
        )?;
        // What's left of a scored conjunct is plain SQL over the new columns;
        // a MEANS still in it sits under OR, NOT or a comparison.
        if contains_means(&predicate)? {
            return plan_err!("{LEGAL_POSITIONS}");
        }
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
    let mut rewrites = 0usize;
    let transformed = plan.transform_up(|plan| {
        plan.map_expressions(|expr| {
            // Top-down so a CONFIDENCE() can be skipped whole: its MEANS is
            // a question to score, not a filter to calibrate.
            expr.transform_down(|expr| match expr {
                confidence if is_confidence_call(&confidence) => {
                    Ok(Transformed::new(confidence, false, TreeNodeRecursion::Jump))
                }
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

fn contains_confidence(expr: &Expr) -> Result<bool> {
    expr.exists(|e| Ok(is_confidence_call(e)))
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
            crate::semir::normalize_instruction(&s)
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
