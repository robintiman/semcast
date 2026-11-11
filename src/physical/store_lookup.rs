//! Store lookup — the first stage of a `MEANS` funnel (redesign phase 2).
//!
//! Before any pre-filter or model call, one batched store read per batch
//! learns every row whose `ASK` already has a full-text verdict for its
//! text. That verdict is the definition of `MEANS`, so it is final, and every
//! later stage honours it through [`StoreEvidence`]:
//!
//! - a stored "yes": the index scan and the scorer never prune the row, and
//!   verify keeps it without a call — not even a chunked one, which would be
//!   asking a weaker question than the one already answered;
//! - a stored "no": the pre-filters prune it without scoring it, and verify
//!   drops it without a call;
//! - an unknown row goes through the funnel as it would have.
//!
//! This stage drops nothing itself. A calibrating stage downstream samples
//! from the rows it sees with a fixed seed, so repeat queries draw the same
//! sample and find its labels in the store; removing the known rows first
//! would move the sample onto new, unlabeled rows and pay for them on every
//! run.
//!
//! Before this stage existed a stored "yes" could still be lost: the index
//! scan pruned on similarity before verify ever reached the cache.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use datafusion::arrow::array::{Array, StringArray};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Statistics;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::TryStreamExt;

use crate::cache::{CachedValue, SemanticCache, stable_digest};
use crate::model::ModelId;
use crate::physical::verify::{MEANS_PROMPT_VERSION, means_cache_key};
use crate::semir::{AskType, question_id};

/// The store's verdicts for a funnel's documents, shared from the lookup to
/// every later stage of the same funnel (and filled in by calibration as it
/// labels). Keyed by the text's 128-bit [`stable_digest`] — the digest the
/// store's own keys use — so it survives any repartitioning between stages
/// and two texts never share a verdict by a hash collision.
#[derive(Debug, Default)]
pub struct StoreEvidence {
    known: RwLock<HashMap<u128, bool>>,
}

impl StoreEvidence {
    /// The stored full-text verdict for this text, if there is one.
    pub fn verdict(&self, text: &str) -> Option<bool> {
        self.known
            .read()
            .expect("store evidence lock poisoned")
            .get(&text_digest(text))
            .copied()
    }

    /// Remember full-text verdicts learned during the query — calibration
    /// labels — so later stages read them as they read the store's.
    pub(crate) fn record<'a>(&self, verdicts: impl IntoIterator<Item = (&'a str, bool)>) {
        self.record_digests(
            verdicts
                .into_iter()
                .map(|(text, verdict)| (text_digest(text), verdict)),
        );
    }

    fn record_digests(&self, verdicts: impl IntoIterator<Item = (u128, bool)>) {
        self.known
            .write()
            .expect("store evidence lock poisoned")
            .extend(verdicts);
    }
}

fn text_digest(text: &str) -> u128 {
    stable_digest(&[text.as_bytes()])
}

/// Learns what the store already knows about a `MEANS` filter's rows.
#[derive(Debug)]
pub struct StoreLookupExec {
    input: Arc<dyn ExecutionPlan>,
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    model_id: ModelId,
    cache: Arc<dyn SemanticCache>,
    evidence: Arc<StoreEvidence>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl StoreLookupExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        condition: impl Into<String>,
        model_id: ModelId,
        cache: Arc<dyn SemanticCache>,
        evidence: Arc<StoreEvidence>,
    ) -> Self {
        // A filter: the input's properties hold, partitioning and order alike.
        let properties = Arc::clone(input.properties());
        Self {
            input,
            text,
            condition: crate::semir::normalize_instruction(&condition.into()),
            model_id,
            cache,
            evidence,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for StoreLookupExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "StoreLookupExec: ask={} MEANS('{}') model={}",
            question_id(
                &self.condition,
                AskType::Bool,
                &self.model_id,
                MEANS_PROMPT_VERSION
            ),
            self.condition,
            self.model_id,
        )
    }
}

impl ExecutionPlan for StoreLookupExec {
    fn name(&self) -> &str {
        "StoreLookupExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.text),
            self.condition.clone(),
            self.model_id.clone(),
            Arc::clone(&self.cache),
            Arc::clone(&self.evidence),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let metric =
            |name: &'static str| MetricBuilder::new(&self.metrics).counter(name, partition);
        let lookup = Arc::new(Lookup {
            text: Arc::clone(&self.text),
            condition: self.condition.clone(),
            model_id: self.model_id.clone(),
            cache: Arc::clone(&self.cache),
            evidence: Arc::clone(&self.evidence),
            rows_looked_up: metric("rows_looked_up"),
            store_matches: metric("store_matches"),
            store_non_matches: metric("store_non_matches"),
        });
        let stream = input.and_then(move |batch| {
            let lookup = Arc::clone(&lookup);
            // The store read can block on disk: keep it off the async workers.
            async move {
                tokio::task::spawn_blocking(move || lookup.lookup_batch(batch))
                    .await
                    .map_err(|error| {
                        datafusion::error::DataFusionError::External(Box::new(error))
                    })?
            }
        });
        let output = Box::pin(RecordBatchStreamAdapter::new(self.input.schema(), stream));
        Ok(crate::physical::trace::trace_stage(
            "StoreLookupExec",
            partition,
            output,
        ))
    }

    /// At most the input survives.
    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let input_rows = self.input.partition_statistics(partition)?.num_rows;
        let mut statistics = Statistics::new_unknown(&self.input.schema());
        statistics.num_rows = input_rows.to_inexact();
        Ok(Arc::new(statistics))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

struct Lookup {
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    model_id: ModelId,
    cache: Arc<dyn SemanticCache>,
    evidence: Arc<StoreEvidence>,
    /// Rows with text, each looked up once.
    rows_looked_up: Count,
    /// Rows the store answered "yes": kept, and never asked again.
    store_matches: Count,
    /// Rows the store answered "no": pruned downstream without a call.
    store_non_matches: Count,
}

impl Lookup {
    fn lookup_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(batch);
        }
        let texts = self.text.evaluate(&batch)?.into_array(rows)?;
        let texts = cast(&texts, &DataType::Utf8)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text cast to Utf8");

        let mut looked_up = Vec::new();
        let mut keys = Vec::new();
        for row in 0..rows {
            if texts.is_valid(row) {
                let text = texts.value(row);
                looked_up.push(row);
                keys.push(means_cache_key(
                    &self.condition,
                    text,
                    &self.model_id,
                    MEANS_PROMPT_VERSION,
                ));
            }
        }
        self.rows_looked_up.add(keys.len());
        let answers = self.cache.get_many(&keys);

        let mut verdicts = Vec::new();
        for (row, answer) in looked_up.into_iter().zip(answers) {
            let verdict = match answer {
                Some(CachedValue::Value(verdict)) if verdict == "yes" => true,
                Some(CachedValue::Value(verdict)) if verdict == "no" => false,
                _ => continue,
            };
            if verdict {
                self.store_matches.add(1);
            } else {
                self.store_non_matches.add(1);
            }
            verdicts.push((text_digest(texts.value(row)), verdict));
        }
        self.evidence.record_digests(verdicts);
        Ok(batch)
    }
}
