//! Typed extraction as a logical operator — `EXTRACT(field TYPE 'doc' FROM
//! text)` and `CAST(text AS SemanticType)`. Skeleton only; extraction runs
//! *after* the funnel, never before it.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::{DFSchemaRef, Result, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

use crate::types::SemanticType;

/// Logical node that extends the input with typed fields extracted from a
/// text expression by a model. Field independence (unless grouped by
/// `TOGETHER`) is what makes field pushdown legal.
#[derive(Debug, Clone)]
pub struct SemExtractNode {
    pub input: LogicalPlan,
    /// Expression producing the source text.
    pub source: Expr,
    /// The full extraction spec — field names, types, doc lines.
    pub target: SemanticType,
    /// Input schema plus one column per (surviving) extracted field.
    /// Not derived on the fly: field pushdown mutates it.
    pub output_schema: DFSchemaRef,
}

impl SemExtractNode {
    /// Compute `output_schema` from `input` and `target` and build the node.
    pub fn try_new(
        _input: LogicalPlan,
        _source: Expr,
        _target: SemanticType,
    ) -> crate::Result<Self> {
        todo!("derive Arrow output schema from the semantic type (after roadmap step 1)")
    }
}

// `output_schema` is derived from the other fields, so it is skipped in
// comparisons — standard practice for schema-carrying extension nodes.
impl PartialEq for SemExtractNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input && self.source == other.source && self.target == other.target
    }
}

impl Eq for SemExtractNode {}

impl PartialOrd for SemExtractNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.input, &self.source, &self.target).partial_cmp(&(
            &other.input,
            &other.source,
            &other.target,
        ))
    }
}

impl Hash for SemExtractNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.source.hash(state);
        self.target.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for SemExtractNode {
    fn name(&self) -> &str {
        "SemExtract"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.output_schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.source.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SemExtract: CAST(... AS {}) [{} fields]",
            self.target.name,
            self.target.fields.len()
        )
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 || inputs.len() != 1 {
            return internal_err!(
                "SemExtract expects exactly 1 expression and 1 input, got {} and {}",
                exprs.len(),
                inputs.len()
            );
        }
        Ok(Self {
            input: inputs.swap_remove(0),
            source: exprs.swap_remove(0),
            target: self.target.clone(),
            output_schema: self.output_schema.clone(),
        })
    }
}
