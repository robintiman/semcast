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
}

impl SemFilterNode {
    pub fn new(
        input: LogicalPlan,
        text: Expr,
        condition: impl Into<String>,
        recall: Option<f64>,
    ) -> Self {
        Self {
            input,
            text,
            condition: condition.into(),
            recall,
        }
    }

    fn recall_bits(&self) -> Option<u64> {
        self.recall.map(f64::to_bits)
    }
}

impl PartialEq for SemFilterNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.text == other.text
            && self.condition == other.condition
            && self.recall_bits() == other.recall_bits()
    }
}

impl Eq for SemFilterNode {}

impl PartialOrd for SemFilterNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.input, &self.text, &self.condition, self.recall_bits()).partial_cmp(&(
            &other.input,
            &other.text,
            &other.condition,
            other.recall_bits(),
        ))
    }
}

impl Hash for SemFilterNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.text.hash(state);
        self.condition.hash(state);
        self.recall_bits().hash(state);
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
        match self.recall {
            Some(r) => write!(f, "   recall ≥ {r:.2}"),
            None => write!(f, "   recall best-effort"),
        }
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
        })
    }
}
