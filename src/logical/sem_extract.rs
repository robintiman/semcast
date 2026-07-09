//! Typed extraction as a logical operator — the node `CAST(text AS T)[.field]`
//! and inline `EXTRACT(...)` desugar to, via [`ExtractRewriteRule`].
//! Extraction runs *after* the funnel, never before it: the rule inserts this
//! node between a `Projection` and its (already-filtered) input.
//!
//! [`ExtractRewriteRule`]: crate::optimizer::extract::ExtractRewriteRule

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::{Column, DFSchema, DFSchemaRef, Result, TableReference, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

use crate::types::SemanticType;

/// Logical node that extends the input with typed fields extracted from a
/// text expression by a model. Field independence (unless grouped by
/// `TOGETHER`) is what makes field pushdown legal — `target` is already pruned
/// to exactly the fields the query reaches.
#[derive(Debug, Clone)]
pub struct SemExtractNode {
    pub input: LogicalPlan,
    /// Expression producing the source text.
    pub source: Expr,
    /// The pruned extraction spec — field names, types, doc lines.
    pub target: SemanticType,
    /// Disambiguates this node's output columns from those of any sibling
    /// extraction node in the same plan (same type on a different source, say).
    pub id: usize,
    /// Input schema plus one column per extracted field.
    pub output_schema: DFSchemaRef,
}

impl SemExtractNode {
    /// Compute `output_schema` from `input` and the pruned `target`, and build
    /// the node. Output = input columns + one nullable column per target field.
    pub fn try_new(
        input: LogicalPlan,
        source: Expr,
        target: SemanticType,
        id: usize,
    ) -> crate::Result<Self> {
        let mut qualified: Vec<(Option<TableReference>, Arc<Field>)> = input
            .schema()
            .iter()
            .map(|(q, f)| (q.cloned(), Arc::clone(f)))
            .collect();
        for spec in &target.fields {
            let field = Field::new(
                output_column_name(id, &spec.name),
                spec.ty.arrow_type()?,
                true,
            );
            qualified.push((None, Arc::new(field)));
        }
        let output_schema = Arc::new(DFSchema::new_with_metadata(qualified, HashMap::new())?);
        Ok(Self {
            input,
            source,
            target,
            id,
            output_schema,
        })
    }

    /// The output column carrying `field`, as an (unqualified) column
    /// reference the rewrite substitutes for the marker call.
    pub fn output_column(&self, field: &str) -> Column {
        Column::new_unqualified(output_column_name(self.id, field))
    }
}

/// Node-scoped column name — the `id` keeps two extraction nodes over the same
/// type from colliding.
pub fn output_column_name(id: usize, field: &str) -> String {
    format!("__sem_{id}_{field}")
}

// `output_schema` and `id` are derived from the other fields plus construction
// order, so comparisons key on input/source/target — standard for
// schema-carrying extension nodes.
impl PartialEq for SemExtractNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.source == other.source
            && self.target == other.target
            && self.id == other.id
    }
}

impl Eq for SemExtractNode {}

impl PartialOrd for SemExtractNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.input, &self.source, &self.target, self.id).partial_cmp(&(
            &other.input,
            &other.source,
            &other.target,
            other.id,
        ))
    }
}

impl Hash for SemExtractNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.source.hash(state);
        self.target.hash(state);
        self.id.hash(state);
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
            "SemExtract: {} [{} field(s)]",
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
        // Recompute the schema — the input's columns may have changed.
        Self::try_new(
            inputs.swap_remove(0),
            exprs.swap_remove(0),
            self.target.clone(),
            self.id,
        )
        .map_err(Into::into)
    }
}
