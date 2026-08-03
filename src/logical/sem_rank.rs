//! Semantic ranking as a logical operator — the node `ORDER BY … RELEVANCE
//! TO …` and `relevance()` desugar to, via [`RelevanceRewriteRule`].
//!
//! The node is a *score materializer*, not a sort: it extends its input with
//! one score column and prunes to the index's candidates. The `Sort` and
//! `Limit` DataFusion already planned do the ordering and the truncation, so
//! semcast never reimplements TopK.
//!
//! [`RelevanceRewriteRule`]: crate::optimizer::rank::RelevanceRewriteRule

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::{Column, DFSchema, DFSchemaRef, Result, TableReference, internal_err};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// How many candidates the index stage fetches per row the query asked for.
/// The rerank pass is the expensive stage, so this multiplier *is* the
/// cost/recall dial: 4× means ~4k model calls, whatever the table's size.
pub const OVER_FETCH: usize = 4;

/// Never fetch fewer candidates than this, however small the `LIMIT` — a
/// `LIMIT 1` still deserves a field to choose from.
pub const MIN_CANDIDATES: usize = 32;

/// Logical node that extends the input with a model-assigned relevance score
/// for a text expression, keeping only the index's top candidates.
#[derive(Debug, Clone)]
pub struct SemRankNode {
    pub input: LogicalPlan,
    /// Expression producing the text being ranked.
    pub text: Expr,
    /// The natural-language query rows are ranked against.
    pub query: String,
    /// The `LIMIT` this ranking feeds — sizes the candidate stage.
    pub limit: usize,
    /// Disambiguates this node's score column from any sibling rank node's.
    pub id: usize,
    /// Input schema plus the score column.
    pub output_schema: DFSchemaRef,
}

impl SemRankNode {
    /// Output = input columns + one nullable `Float64` score column.
    pub fn try_new(
        input: LogicalPlan,
        text: Expr,
        query: impl Into<String>,
        limit: usize,
        id: usize,
    ) -> Result<Self> {
        let mut qualified: Vec<(Option<TableReference>, Arc<Field>)> = input
            .schema()
            .iter()
            .map(|(q, f)| (q.cloned(), Arc::clone(f)))
            .collect();
        qualified.push((
            None,
            Arc::new(Field::new(score_column_name(id), DataType::Float64, true)),
        ));
        let output_schema = Arc::new(DFSchema::new_with_metadata(qualified, HashMap::new())?);
        Ok(Self {
            input,
            text,
            query: query.into(),
            limit,
            id,
            output_schema,
        })
    }

    /// How many candidate documents the index stage should fetch.
    pub fn candidates(&self) -> usize {
        (self.limit * OVER_FETCH).max(MIN_CANDIDATES)
    }

    /// The score column, as an (unqualified) reference the rewrite
    /// substitutes for the marker call.
    pub fn score_column(&self) -> Column {
        Column::new_unqualified(score_column_name(self.id))
    }
}

/// Node-scoped column name — the `id` keeps two rank nodes in one plan from
/// colliding.
pub fn score_column_name(id: usize) -> String {
    format!("__sem_rank_{id}_score")
}

// `output_schema` is derived from the other fields, so comparisons key on
// input/text/query/limit/id — standard for schema-carrying extension nodes.
impl PartialEq for SemRankNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.text == other.text
            && self.query == other.query
            && self.limit == other.limit
            && self.id == other.id
    }
}

impl Eq for SemRankNode {}

impl PartialOrd for SemRankNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.input, &self.text, &self.query, self.limit, self.id).partial_cmp(&(
            &other.input,
            &other.text,
            &other.query,
            other.limit,
            other.id,
        ))
    }
}

impl Hash for SemRankNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.text.hash(state);
        self.query.hash(state);
        self.limit.hash(state);
        self.id.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for SemRankNode {
    fn name(&self) -> &str {
        "SemRank"
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
            "SemRank: RELEVANCE TO '{}' limit={} candidates={}",
            self.query,
            self.limit,
            self.candidates()
        )
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 || inputs.len() != 1 {
            return internal_err!(
                "SemRank expects exactly 1 expression and 1 input, got {} and {}",
                exprs.len(),
                inputs.len()
            );
        }
        // Recompute the schema — the input's columns may have changed.
        Self::try_new(
            inputs.swap_remove(0),
            exprs.swap_remove(0),
            self.query.clone(),
            self.limit,
            self.id,
        )
    }
}
