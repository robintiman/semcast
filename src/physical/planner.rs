//! Maps semcast logical extension nodes to physical operators.
//!
//! DataFusion hook: a `QueryPlanner` that wraps the default planner with a
//! semcast `ExtensionPlanner`, installed by [`semcast_context`].
//!
//! [`semcast_context`]: crate::semcast_context

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::DFSchema;
use datafusion::error::Result;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use crate::cache::SemanticCache;
use crate::index::SemanticIndex;
use crate::index::registry::SemcastRuntime;
use crate::logical::SemFilterNode;
use crate::model::ModelProvider;
use crate::physical::index_scan::{ChunkEvidence, IndexScanExec};
use crate::physical::VerifyExec;

/// The default DataFusion planner plus semcast extension planning.
#[derive(Debug)]
pub struct SemcastQueryPlanner {
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
}

impl SemcastQueryPlanner {
    pub fn new(model: Arc<dyn ModelProvider>, cache: Arc<dyn SemanticCache>) -> Self {
        Self { model, cache }
    }
}

#[async_trait]
impl QueryPlanner for SemcastQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            SemcastExtensionPlanner {
                model: Arc::clone(&self.model),
                cache: Arc::clone(&self.cache),
            },
        )]);
        planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// Plans `SemFilter` (and eventually `SemExtract`) extension nodes. Today
/// every `SemFilter` becomes a verify-only [`VerifyExec`] — correct first,
/// cheap later; funnel stages slot in with roadmap step 2.
struct SemcastExtensionPlanner {
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
}

#[async_trait]
impl ExtensionPlanner for SemcastExtensionPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(filter) = node.as_any().downcast_ref::<SemFilterNode>() {
            let text = planner.create_physical_expr(
                &filter.text,
                logical_inputs[0].schema(),
                session_state,
            )?;
            // Funnel when an index covers the filtered column: index scan
            // prunes, verify reads the surviving rows' top chunks. No index
            // (or an unresolvable text expr) → verify-only, correct but full
            // price.
            if let Some(index) = resolve_index(filter, logical_inputs[0].schema(), session_state) {
                let params = index.search_params();
                let evidence = Arc::new(ChunkEvidence::default());
                let scan = Arc::new(IndexScanExec::new(
                    Arc::clone(&physical_inputs[0]),
                    Arc::clone(&text),
                    filter.condition.clone(),
                    index,
                    params,
                    Arc::clone(&evidence),
                ));
                return Ok(Some(Arc::new(VerifyExec::new_with_evidence(
                    scan,
                    text,
                    filter.condition.clone(),
                    Arc::clone(&self.model),
                    Arc::clone(&self.cache),
                    evidence,
                    params.chunks_per_doc,
                ))));
            }
            return Ok(Some(Arc::new(VerifyExec::new(
                Arc::clone(&physical_inputs[0]),
                text,
                filter.condition.clone(),
                Arc::clone(&self.model),
                Arc::clone(&self.cache),
            ))));
        }
        Ok(None)
    }
}

/// The registered index for the filtered column, if the text expression
/// resolves to a plain qualified column. Resolution is deliberately exact:
/// computed expressions or alias-erased qualifiers plan verify-only rather
/// than borrow a possibly-wrong index.
fn resolve_index(
    filter: &SemFilterNode,
    schema: &DFSchema,
    session_state: &SessionState,
) -> Option<Arc<dyn SemanticIndex>> {
    let column = column_behind_casts(&filter.text)?;
    let (qualifier, field) = schema.qualified_field_from_column(column).ok()?;
    let table = qualifier?.table();
    let runtime = session_state.config().get_extension::<SemcastRuntime>()?;
    runtime.index_for(table, field.name())
}

/// The column a text expression reads, seen through casts and aliases —
/// type coercion wraps string columns in `CAST(... AS Utf8)` for the
/// `means` UDF signature, and a cast between string types doesn't change
/// which document the text is (both stages hash the evaluated text, so the
/// index keys still line up). Anything else is a computed expression: no
/// index.
fn column_behind_casts(expr: &Expr) -> Option<&datafusion::common::Column> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Cast(cast) => column_behind_casts(&cast.expr),
        Expr::TryCast(cast) => column_behind_casts(&cast.expr),
        Expr::Alias(alias) => column_behind_casts(&alias.expr),
        _ => None,
    }
}
