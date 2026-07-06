//! Maps semcast logical extension nodes to physical operators.
//!
//! DataFusion hook: a `QueryPlanner` that wraps the default planner with a
//! semcast `ExtensionPlanner`, installed by [`semcast_context`].
//!
//! [`semcast_context`]: crate::semcast_context

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::error::Result;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use crate::cache::SemanticCache;
use crate::logical::SemFilterNode;
use crate::model::ModelProvider;
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
