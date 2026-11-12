//! The `MEANS` predicate as a logical operator (roadmap step 1).
//!
//! Ground truth: *a model reading the full text would say yes*. Everything
//! the planner substitutes for that — index pre-filters, chunked reads — is
//! an approximation managed under [`SemFilterNode::recall`].

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::{DFSchemaRef, Result, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Logical node for `text MEANS 'condition' [WITH RECALL r]`.
///
/// Filter semantics: emits the subset of input rows for which the model,
/// reading `text`, affirms `condition`. The output schema is the input schema.
#[derive(Debug, Clone)]
pub struct SemFilterNode {
    pub input: LogicalPlan,
    /// Expression producing the text under scrutiny (usually a column).
    pub text: Expr,
    /// The natural-language condition, verbatim from the query.
    pub condition: String,
    /// Recall floor from `WITH RECALL`; `None` means best-effort thresholds,
    /// and `EXPLAIN` says so.
    pub recall: Option<f64>,
    /// Confidence from `WITH CONFIDENCE`, which turns `recall` from a point
    /// estimate on the calibration sample into a bound certified for the
    /// population. `None` keeps the estimate.
    pub confidence: Option<f64>,
    /// The statement's `WITH RECALL` / `WITH CONFIDENCE`, which `recall` and
    /// `confidence` are this filter's share of once the statement has
    /// several lossy filters (see [`crate::optimizer::budget`]).
    pub statement_recall: Option<f64>,
    pub statement_confidence: Option<f64>,
    /// How many lossy filters share the statement's target: 1 alone.
    pub shares: usize,
}

impl SemFilterNode {
    pub fn new(
        input: LogicalPlan,
        text: Expr,
        condition: impl Into<String>,
        recall: Option<f64>,
        confidence: Option<f64>,
    ) -> Self {
        Self {
            input,
            text,
            // Public constructor: normalize here too, so a caller outside
            // the rewrite still asks (and keys) the one question.
            condition: crate::semir::normalize_instruction(&condition.into()),
            recall,
            confidence,
            statement_recall: recall,
            statement_confidence: confidence,
            shares: 1,
        }
    }

    /// Every knob as bit patterns, so equality and hashing agree on floats.
    fn target_bits(&self) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>, usize) {
        (
            self.recall.map(f64::to_bits),
            self.confidence.map(f64::to_bits),
            self.statement_recall.map(f64::to_bits),
            self.statement_confidence.map(f64::to_bits),
            self.shares,
        )
    }
}

/// A target as EXPLAIN prints it: two decimals, more only when a share
/// needs them (0.9967 must not read as 1.00).
fn fraction(value: f64) -> String {
    let text = format!("{value:.4}");
    let text = text.trim_end_matches('0');
    match text.split_once('.') {
        Some((_, decimals)) if decimals.len() >= 2 => text.to_owned(),
        _ => format!("{value:.2}"),
    }
}

impl PartialEq for SemFilterNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.text == other.text
            && self.condition == other.condition
            && self.target_bits() == other.target_bits()
    }
}

impl Eq for SemFilterNode {}

impl PartialOrd for SemFilterNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.input, &self.text, &self.condition, self.target_bits()).partial_cmp(&(
            &other.input,
            &other.text,
            &other.condition,
            other.target_bits(),
        ))
    }
}

impl Hash for SemFilterNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.text.hash(state);
        self.condition.hash(state);
        self.target_bits().hash(state);
    }
}

impl UserDefinedLogicalNodeCore for SemFilterNode {
    fn name(&self) -> &str {
        "SemFilter"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.text.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemFilter: MEANS('{}')", self.condition)?;
        match (self.recall, self.confidence) {
            // A confidence turns the target into a claim about the population,
            // so EXPLAIN states which kind of promise this plan is making.
            (Some(r), Some(c)) => write!(
                f,
                "   recall ≥ {} certified at {}",
                fraction(r),
                fraction(c)
            )?,
            (Some(r), None) => write!(f, "   recall ≥ {} estimated", fraction(r))?,
            (None, _) => write!(f, "   recall best-effort")?,
        }
        if self.shares > 1
            && let Some(statement) = self.statement_recall
        {
            write!(
                f,
                " (its share of the statement's {} across {} filters)",
                fraction(statement),
                self.shares
            )?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 || inputs.len() != 1 {
            return internal_err!(
                "SemFilter expects exactly 1 expression and 1 input, got {} and {}",
                exprs.len(),
                inputs.len()
            );
        }
        Ok(Self {
            input: inputs.swap_remove(0),
            text: exprs.swap_remove(0),
            condition: self.condition.clone(),
            recall: self.recall,
            confidence: self.confidence,
            statement_recall: self.statement_recall,
            statement_confidence: self.statement_confidence,
            shares: self.shares,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::fraction;

    #[test]
    fn a_share_keeps_the_digits_it_needs() {
        assert_eq!(fraction(0.9), "0.90");
        assert_eq!(fraction(0.95), "0.95");
        assert_eq!(fraction(1.0 - 0.01 / 3.0), "0.9967");
        assert_eq!(fraction(0.975), "0.975");
    }
}
