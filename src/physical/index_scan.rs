//! The index pre-filter stage — the funnel's cheap stage (roadmap step 2).
//!
//! Sits between the free predicates and [`VerifyExec`], pruning rows whose
//! indexed document scored below the floor against the embedded condition.
//! Costs one embed call per query, zero completion calls.
//!
//! [`VerifyExec`]: crate::physical::VerifyExec

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, StringArray};
use datafusion::arrow::compute::{cast, filter_record_batch};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Statistics;
use datafusion::common::stats::Precision;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::TryStreamExt;
use tokio::sync::OnceCell;

use crate::index::{SearchParams, SemanticIndex, doc_hash};

/// What one index search learned, shared from the scan to the verify stage.
#[derive(Debug)]
pub struct PrefilterResult {
    /// Surviving documents → their best chunks (verify's evidence), best
    /// first, at most `chunks_per_doc` each.
    pub chunks: HashMap<u64, Vec<String>>,
    /// Every document the index knows — how the scan tells "scored below
    /// the floor" (prune) from "never indexed" (pass through).
    pub indexed: HashSet<u64>,
}

/// Planning-time channel between [`IndexScanExec`] and [`VerifyExec`]: the
/// scan populates it before emitting a batch, verify reads it per row.
/// Keyed by [`doc_hash`], so it survives whatever repartitioning DataFusion
/// inserts between the two operators.
///
/// [`VerifyExec`]: crate::physical::VerifyExec
#[derive(Debug, Default)]
pub struct ChunkEvidence {
    cell: OnceCell<Arc<PrefilterResult>>,
}

impl ChunkEvidence {
    /// The chunks for a document, if the index scan ran and the document
    /// survived it.
    pub fn chunks_for(&self, doc_hash: u64) -> Option<&[String]> {
        self.cell
            .get()
            .and_then(|result| result.chunks.get(&doc_hash))
            .map(Vec::as_slice)
    }
}

/// Filters input batches through one nearest-vector search over the semantic
/// index. Rows the index never saw pass through — staleness must never
/// silently drop a row; it only costs a full-text verify call downstream.
#[derive(Debug)]
pub struct IndexScanExec {
    input: Arc<dyn ExecutionPlan>,
    /// Evaluates to the text under scrutiny, against input batches.
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    index: Arc<dyn SemanticIndex>,
    params: SearchParams,
    evidence: Arc<ChunkEvidence>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl IndexScanExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        condition: impl Into<String>,
        index: Arc<dyn SemanticIndex>,
        params: SearchParams,
        evidence: Arc<ChunkEvidence>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            text,
            condition: condition.into(),
            index,
            params,
            evidence,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for IndexScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IndexScanExec: MEANS('{}') embed_model={} floor={} top-{} chunks \
             (threshold best-effort — no WITH RECALL)",
            self.condition,
            self.index.embed_model_id(),
            self.params.score_floor,
            self.params.chunks_per_doc,
        )
    }
}

impl ExecutionPlan for IndexScanExec {
    fn name(&self) -> &str {
        "IndexScanExec"
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
            Arc::clone(&self.text),
            self.condition.clone(),
            Arc::clone(&self.index),
            self.params,
            Arc::clone(&self.evidence),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let scanner = Arc::new(Scanner {
            text: Arc::clone(&self.text),
            condition: self.condition.clone(),
            index: Arc::clone(&self.index),
            params: self.params,
            evidence: Arc::clone(&self.evidence),
            index_hits: MetricBuilder::new(&self.metrics).counter("index_hits", partition),
            rows_pruned: MetricBuilder::new(&self.metrics).counter("rows_pruned", partition),
            passthrough_rows: MetricBuilder::new(&self.metrics)
                .counter("passthrough_rows", partition),
        });
        let stream = input.and_then(move |batch| {
            let scanner = Arc::clone(&scanner);
            async move { scanner.scan_batch(batch).await }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.input.schema(),
            stream,
        )))
    }

    /// At most `fetch_k` distinct documents survive the vector scan; Inexact
    /// because unindexed rows pass through on top of that.
    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let input_rows = self.input.partition_statistics(partition)?.num_rows;
        let cap = self.params.fetch_k;
        let mut statistics = Statistics::new_unknown(&self.input.schema());
        statistics.num_rows = match input_rows {
            Precision::Exact(rows) | Precision::Inexact(rows) => {
                Precision::Inexact(rows.min(cap))
            }
            Precision::Absent => Precision::Inexact(cap),
        };
        Ok(Arc::new(statistics))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// Everything one partition's stream needs.
struct Scanner {
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    index: Arc<dyn SemanticIndex>,
    params: SearchParams,
    evidence: Arc<ChunkEvidence>,
    index_hits: Count,
    rows_pruned: Count,
    passthrough_rows: Count,
}

impl Scanner {
    async fn scan_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let result = self
            .evidence
            .cell
            .get_or_try_init(|| self.prefilter())
            .await
            .map_err(datafusion::error::DataFusionError::from)?;
        if batch.num_rows() == 0 {
            return Ok(batch);
        }
        let texts = self.text.evaluate(&batch)?.into_array(batch.num_rows())?;
        let texts = cast(&texts, &DataType::Utf8)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("array was just cast to Utf8");

        let mut keep = vec![false; batch.num_rows()];
        for (row, keep_row) in keep.iter_mut().enumerate() {
            // NULL text never matches — free to drop here.
            if !texts.is_valid(row) {
                self.rows_pruned.add(1);
                continue;
            }
            let hash = doc_hash(texts.value(row));
            if result.chunks.contains_key(&hash) {
                self.index_hits.add(1);
                *keep_row = true;
            } else if result.indexed.contains(&hash) {
                self.rows_pruned.add(1);
            } else {
                self.passthrough_rows.add(1);
                *keep_row = true;
            }
        }
        Ok(filter_record_batch(&batch, &BooleanArray::from(keep))?)
    }

    /// One embed call + one vector scan + one membership scan, shared by
    /// every partition through the evidence cell.
    async fn prefilter(&self) -> crate::Result<Arc<PrefilterResult>> {
        let hits = self.index.search(&self.condition, &self.params).await?;
        let mut chunks: HashMap<u64, Vec<String>> = HashMap::new();
        for hit in hits {
            let doc_chunks = chunks.entry(hit.doc_hash).or_default();
            if doc_chunks.len() < self.params.chunks_per_doc {
                doc_chunks.push(hit.text);
            }
        }
        let indexed = self.index.indexed_doc_hashes().await?;
        Ok(Arc::new(PrefilterResult { chunks, indexed }))
    }
}
