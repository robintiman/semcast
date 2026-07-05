//! The verify stage — the physical operator that spends model calls
//! (roadmap step 1: verify-only execution).

use std::fmt;
use std::sync::Arc;

use datafusion::common::not_impl_err;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};

use crate::model::ModelProvider;

/// Filters input batches by asking the model whether each row's text meets
/// the condition. Ground truth for `MEANS` — every cheaper stage upstream is
/// an approximation of what this operator computes.
#[derive(Debug)]
pub struct VerifyExec {
    input: Arc<dyn ExecutionPlan>,
    condition: String,
    model: Arc<dyn ModelProvider>,
    properties: Arc<PlanProperties>,
}

impl VerifyExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        condition: impl Into<String>,
        model: Arc<dyn ModelProvider>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            condition: condition.into(),
            model,
            properties,
        }
    }
}

impl DisplayAs for VerifyExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VerifyExec: MEANS('{}') model={}",
            self.condition,
            self.model.id()
        )
    }
}

impl ExecutionPlan for VerifyExec {
    fn name(&self) -> &str {
        "VerifyExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            self.condition.clone(),
            Arc::clone(&self.model),
        )))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Roadmap step 1: stream input batches, build one CompletionRequest
        // per row (later: top-k chunks from the index instead of full text),
        // batch through ModelProvider::complete, keep rows answered "yes".
        // Failed rows yield NULL + error column, not a failed query.
        not_impl_err!("VerifyExec::execute — roadmap step 1 (verify-only physical plan)")
    }
}
