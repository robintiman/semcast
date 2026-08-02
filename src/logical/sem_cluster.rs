//! Semantic clustering as a logical operator — the node `GROUP BY MEANING OF`
//! desugars to, via [`ClusterRewriteRule`].
//!
//! Like the classify and rank nodes, this one only materializes a column: the
//! label of the group each row landed in. `GROUP BY` then groups by that
//! column like any other, so aggregation, `HAVING` and ordering stay
//! DataFusion's.
//!
//! Unlike them it is a *blocking* operator, and unavoidably so: which group a
//! row belongs to is a fact about the whole relation, not about the row. That
//! is the cost of clustering, and it is why `EXPLAIN` says so.
//!
//! [`ClusterRewriteRule`]: crate::optimizer::cluster::ClusterRewriteRule

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::{Column, DFSchema, DFSchemaRef, Result, TableReference, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// How many of a cluster's most central documents the model reads when
/// naming it. Enough to see the theme, few enough that labelling stays one
/// cheap call per cluster.
pub const REPRESENTATIVES_PER_CLUSTER: usize = 5;

/// Logical node that extends the input with the label of the semantic group
/// each row belongs to.
#[derive(Debug, Clone)]
pub struct SemClusterNode {
    pub input: LogicalPlan,
    /// Expression producing the text being clustered.
    pub text: Expr,
    /// Group count from `INTO k`; `None` sweeps for one.
    pub k: Option<usize>,
    /// Disambiguates this node's column from a sibling cluster node's.
    pub id: usize,
    /// Input schema plus the label column.
    pub output_schema: DFSchemaRef,
}

impl SemClusterNode {
    /// Output = input columns + one nullable `Utf8` label column.
    pub fn try_new(input: LogicalPlan, text: Expr, k: Option<usize>, id: usize) -> Result<Self> {
        let mut qualified: Vec<(Option<TableReference>, Arc<Field>)> = input
            .schema()
            .iter()
            .map(|(q, f)| (q.cloned(), Arc::clone(f)))
            .collect();
        qualified.push((
            None,
            Arc::new(Field::new(label_column_name(id), DataType::Utf8, true)),
        ));
        let output_schema = Arc::new(DFSchema::new_with_metadata(qualified, HashMap::new())?);
        Ok(Self {
            input,
            text,
            k,
            id,
            output_schema,
        })
    }

    /// The label column, as an (unqualified) reference the rewrite
    /// substitutes for the marker call.
    pub fn label_column(&self) -> Column {
        Column::new_unqualified(label_column_name(self.id))
    }
}

/// Node-scoped column name — the `id` keeps two cluster nodes in one plan
/// from colliding.
pub fn label_column_name(id: usize) -> String {
    format!("__sem_cluster_{id}_label")
}

// `output_schema` is derived from the other fields, so comparisons key on
// input/text/k/id — standard for schema-carrying extension nodes.
impl PartialEq for SemClusterNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.text == other.text
            && self.k == other.k
            && self.id == other.id
    }
}

impl Eq for SemClusterNode {}

impl PartialOrd for SemClusterNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.input, &self.text, self.k, self.id).partial_cmp(&(
            &other.input,
            &other.text,
            other.k,
            other.id,
        ))
    }
}

impl Hash for SemClusterNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.text.hash(state);
        self.k.hash(state);
        self.id.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for SemClusterNode {
    fn name(&self) -> &str {
        "SemCluster"
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
        write!(f, "SemCluster: MEANING OF ")?;
        match self.k {
            Some(k) => write!(f, "INTO {k}")?,
            None => write!(f, "INTO auto")?,
        }
        write!(f, "   1 model call per group, blocking")
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 || inputs.len() != 1 {
            return internal_err!(
                "SemCluster expects exactly 1 expression and 1 input, got {} and {}",
                exprs.len(),
                inputs.len()
            );
        }
        // Recompute the schema — the input's columns may have changed.
        Self::try_new(inputs.swap_remove(0), exprs.swap_remove(0), self.k, self.id)
    }
}
