//! Semantic classification as a logical operator — the node a `MEANS` in a
//! `SELECT` list desugars to, via [`MeansRewriteRule`].
//!
//! The node materializes one boolean column per condition and nothing else.
//! `CASE` stays in the projection above it, rewritten to reference those
//! columns, so first-match-wins, `ELSE`, and NULL handling remain
//! DataFusion's rather than something semcast reimplements.
//!
//! Each column means exactly what the same `MEANS` would mean in a `WHERE`:
//! *a model reading the text would say the condition holds*. Fusing several
//! conditions into one model call is an optimization, never a change of
//! question.
//!
//! [`MeansRewriteRule`]: crate::optimizer::rewrite::MeansRewriteRule

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::{Column, DFSchema, DFSchemaRef, Result, TableReference, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Logical node that extends the input with one model-assigned boolean per
/// natural-language condition.
#[derive(Debug, Clone)]
pub struct SemClassifyNode {
    pub input: LogicalPlan,
    /// Expression producing the text under scrutiny (usually a column).
    pub text: Expr,
    /// The natural-language conditions, verbatim from the query, in the order
    /// their columns appear.
    pub conditions: Vec<String>,
    /// Rows for which this is false (or NULL) skip the model entirely — the
    /// ordinary `CASE` branches that precede the first `MEANS` branch already
    /// decided them. `None` means every row is asked about.
    pub guard: Option<Expr>,
    /// Disambiguates this node's columns from a sibling classify node's.
    pub id: usize,
    /// Input schema plus one column per condition.
    pub output_schema: DFSchemaRef,
}

impl SemClassifyNode {
    /// Output = input columns + one nullable `Boolean` column per condition.
    pub fn try_new(
        input: LogicalPlan,
        text: Expr,
        conditions: Vec<String>,
        guard: Option<Expr>,
        id: usize,
    ) -> Result<Self> {
        if conditions.is_empty() {
            return internal_err!("SemClassify needs at least one condition");
        }
        let mut qualified: Vec<(Option<TableReference>, Arc<Field>)> = input
            .schema()
            .iter()
            .map(|(q, f)| (q.cloned(), Arc::clone(f)))
            .collect();
        for branch in 0..conditions.len() {
            qualified.push((
                None,
                Arc::new(Field::new(
                    branch_column_name(id, branch),
                    DataType::Boolean,
                    true,
                )),
            ));
        }
        let output_schema = Arc::new(DFSchema::new_with_metadata(qualified, HashMap::new())?);
        Ok(Self {
            input,
            text,
            conditions,
            guard,
            id,
            output_schema,
        })
    }

    /// The column carrying `condition`'s verdict, as an (unqualified)
    /// reference the rewrite substitutes for the marker call.
    pub fn branch_column(&self, condition: &str) -> Option<Column> {
        let branch = self.conditions.iter().position(|c| c == condition)?;
        Some(Column::new_unqualified(branch_column_name(self.id, branch)))
    }
}

/// Node-scoped column name — the `id` keeps two classify nodes in one plan
/// from colliding.
pub fn branch_column_name(id: usize, branch: usize) -> String {
    format!("__sem_class_{id}_{branch}")
}

// `output_schema` is derived from the other fields, so comparisons key on
// input/text/conditions/guard/id — standard for schema-carrying extension
// nodes.
impl PartialEq for SemClassifyNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.text == other.text
            && self.conditions == other.conditions
            && self.guard == other.guard
            && self.id == other.id
    }
}

impl Eq for SemClassifyNode {}

impl PartialOrd for SemClassifyNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (
            &self.input,
            &self.text,
            &self.conditions,
            &self.guard,
            self.id,
        )
            .partial_cmp(&(
                &other.input,
                &other.text,
                &other.conditions,
                &other.guard,
                other.id,
            ))
    }
}

impl Hash for SemClassifyNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.text.hash(state);
        self.conditions.hash(state);
        self.guard.hash(state);
        self.id.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for SemClassifyNode {
    fn name(&self) -> &str {
        "SemClassify"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.output_schema
    }

    /// Text first, then the guard when there is one — both reference input
    /// columns, so DataFusion has to see them to rewrite through this node.
    fn expressions(&self) -> Vec<Expr> {
        let mut exprs = vec![self.text.clone()];
        exprs.extend(self.guard.clone());
        exprs
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemClassify: MEANS(")?;
        for (i, condition) in self.conditions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "'{condition}'")?;
        }
        write!(f, ")   1 model call per row")?;
        if self.guard.is_some() {
            write!(f, ", gated")?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        let expected = 1 + usize::from(self.guard.is_some());
        if exprs.len() != expected || inputs.len() != 1 {
            return internal_err!(
                "SemClassify expects exactly {expected} expression(s) and 1 input, \
                 got {} and {}",
                exprs.len(),
                inputs.len()
            );
        }
        let guard = (exprs.len() > 1).then(|| exprs.swap_remove(1));
        // Recompute the schema — the input's columns may have changed.
        Self::try_new(
            inputs.swap_remove(0),
            exprs.swap_remove(0),
            self.conditions.clone(),
            guard,
            self.id,
        )
    }
}
