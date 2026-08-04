//! Semantic dedupe as a logical operator — the node `SEMANTIC DISTINCT ON`
//! desugars to, via [`DistinctRewriteRule`].
//!
//! It materializes a *key*: a short stable string shared by every document in
//! a near-duplicate group. `DISTINCT ON` then deduplicates by that key like
//! any other column, so which row survives, and how the result is ordered,
//! stay DataFusion's.
//!
//! Blocking, like clustering and for the same reason: whether a row is a
//! duplicate is a fact about the rows around it.
//!
//! [`DistinctRewriteRule`]: crate::optimizer::distinct::DistinctRewriteRule

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::{Column, DFSchema, DFSchemaRef, Result, TableReference, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Logical node that extends the input with the near-duplicate group key of
/// each row.
#[derive(Debug, Clone)]
pub struct SemDistinctNode {
    pub input: LogicalPlan,
    /// Expression producing the text being deduplicated.
    pub text: Expr,
    /// Similarity at or above which two documents are the same, from
    /// `WITH SIMILARITY`.
    pub similarity: f32,
    /// Disambiguates this node's column from a sibling dedupe node's.
    pub id: usize,
    /// Input schema plus the key column.
    pub output_schema: DFSchemaRef,
}

impl SemDistinctNode {
    /// Output = input columns + one nullable `Utf8` key column.
    pub fn try_new(input: LogicalPlan, text: Expr, similarity: f32, id: usize) -> Result<Self> {
        let mut qualified: Vec<(Option<TableReference>, Arc<Field>)> = input
            .schema()
            .iter()
            .map(|(q, f)| (q.cloned(), Arc::clone(f)))
            .collect();
        qualified.push((
            None,
            Arc::new(Field::new(key_column_name(id), DataType::Utf8, true)),
        ));
        let output_schema = Arc::new(DFSchema::new_with_metadata(qualified, HashMap::new())?);
        Ok(Self {
            input,
            text,
            similarity,
            id,
            output_schema,
        })
    }

    /// The key column, as an (unqualified) reference the rewrite substitutes
    /// for the marker call.
    pub fn key_column(&self) -> Column {
        Column::new_unqualified(key_column_name(self.id))
    }

    fn similarity_bits(&self) -> u32 {
        self.similarity.to_bits()
    }
}

/// Node-scoped column name — the `id` keeps two dedupe nodes in one plan from
/// colliding.
pub fn key_column_name(id: usize) -> String {
    format!("__sem_distinct_{id}_key")
}

// `output_schema` is derived from the other fields, and `similarity` is a
// float, so comparisons go through its bits — the same dance `SemFilterNode`
// does for its recall target.
impl PartialEq for SemDistinctNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.text == other.text
            && self.similarity_bits() == other.similarity_bits()
            && self.id == other.id
    }
}

impl Eq for SemDistinctNode {}

impl PartialOrd for SemDistinctNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.input, &self.text, self.similarity_bits(), self.id).partial_cmp(&(
            &other.input,
            &other.text,
            other.similarity_bits(),
            other.id,
        ))
    }
}

impl Hash for SemDistinctNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.text.hash(state);
        self.similarity_bits().hash(state);
        self.id.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for SemDistinctNode {
    fn name(&self) -> &str {
        "SemDistinct"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.output_schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.text.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SemDistinct: SIMILARITY ≥ {:.2}   no model calls, blocking",
            self.similarity
        )
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 || inputs.len() != 1 {
            return internal_err!(
                "SemDistinct expects exactly 1 expression and 1 input, got {} and {}",
                exprs.len(),
                inputs.len()
            );
        }
        // Recompute the schema — the input's columns may have changed.
        Self::try_new(
            inputs.swap_remove(0),
            exprs.swap_remove(0),
            self.similarity,
            self.id,
        )
    }
}
