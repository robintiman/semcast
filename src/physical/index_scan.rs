//! The index pre-filter stage — the funnel's cheap stage (roadmap step 2),
//! with `WITH RECALL` threshold calibration (step 3).
//!
//! Sits between the free predicates and [`VerifyExec`], pruning rows whose
//! indexed document scored below the floor against the embedded condition.
//! Uncalibrated, it costs one embed call per query and zero completion
//! calls; under `WITH RECALL`, the first poll additionally labels a small
//! sample of input rows (full-text model calls, shared with the verdict
//! cache) to set the floor at the recall target.
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
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use tokio::sync::OnceCell;

use crate::cache::{CachedValue, SemanticCache};
use crate::index::{SearchParams, SemanticIndex, doc_hash};
use crate::model::{CompletionRequest, ModelProvider};
use crate::optimizer::calibrate::{SampledScores, calibrate_threshold};
use crate::physical::verify::{means_cache_key, parse_verdict, synthesize_means_prompt};

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

/// How a calibrated scan learns its threshold (`WITH RECALL`): label a
/// sample of input rows with the model and set the floor at the recall
/// target, instead of trusting `SearchParams::score_floor`.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    pub target_recall: f64,
    /// Documents to label, at most — the calibration cost ceiling.
    pub sample_size: usize,
    /// Labeling model — ground truth is this model reading the full text.
    pub model: Arc<dyn ModelProvider>,
    /// Verdict cache; labels are full-text verify verdicts and share keys.
    pub cache: Arc<dyn SemanticCache>,
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
    calibration: Option<CalibrationConfig>,
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
        calibration: Option<CalibrationConfig>,
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
            calibration,
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
            "IndexScanExec: MEANS('{}') embed_model={} ",
            self.condition,
            self.index.embed_model_id(),
        )?;
        // A calibrated floor doesn't exist until execution samples the
        // input — EXPLAIN shows the contract, not a number.
        match &self.calibration {
            Some(calibration) => write!(
                f,
                "floor=calibrated(recall≥{:.2}, sample≤{}) top-{} chunks",
                calibration.target_recall, calibration.sample_size, self.params.chunks_per_doc,
            ),
            None => write!(
                f,
                "floor={} top-{} chunks (threshold best-effort — no WITH RECALL)",
                self.params.score_floor, self.params.chunks_per_doc,
            ),
        }
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
            self.calibration.clone(),
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
            calibration: self.calibration.clone(),
            evidence: Arc::clone(&self.evidence),
            index_hits: MetricBuilder::new(&self.metrics).counter("index_hits", partition),
            rows_pruned: MetricBuilder::new(&self.metrics).counter("rows_pruned", partition),
            passthrough_rows: MetricBuilder::new(&self.metrics)
                .counter("passthrough_rows", partition),
            calibration_sampled_rows: MetricBuilder::new(&self.metrics)
                .counter("calibration_sampled_rows", partition),
            calibration_model_calls: MetricBuilder::new(&self.metrics)
                .counter("calibration_model_calls", partition),
        });
        let stream: BoxStream<'_, Result<RecordBatch>> = if scanner.calibration.is_some() {
            // Calibration needs sample rows before anything can be filtered:
            // buffer up to sample_size rows, calibrate once, then run the
            // buffered rows and the rest of the input through the filter.
            futures::stream::once(scanner.calibrate_then_scan(input))
                .try_flatten()
                .boxed()
        } else {
            input
                .and_then(move |batch| {
                    let scanner = Arc::clone(&scanner);
                    async move {
                        let result = scanner.prefilter_result().await?;
                        scanner.scan_batch(&result, batch)
                    }
                })
                .boxed()
        };
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
            Precision::Exact(rows) | Precision::Inexact(rows) => Precision::Inexact(rows.min(cap)),
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
    calibration: Option<CalibrationConfig>,
    evidence: Arc<ChunkEvidence>,
    index_hits: Count,
    rows_pruned: Count,
    passthrough_rows: Count,
    calibration_sampled_rows: Count,
    calibration_model_calls: Count,
}

impl Scanner {
    fn scan_batch(&self, result: &PrefilterResult, batch: RecordBatch) -> Result<RecordBatch> {
        if batch.num_rows() == 0 {
            return Ok(batch);
        }
        let texts = self.evaluate_texts(&batch)?;
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

    fn evaluate_texts(
        &self,
        batch: &RecordBatch,
    ) -> Result<Arc<dyn datafusion::arrow::array::Array>> {
        let texts = self.text.evaluate(batch)?.into_array(batch.num_rows())?;
        Ok(cast(&texts, &DataType::Utf8)?)
    }

    /// The shared prefilter, computed by whichever partition polls first.
    async fn prefilter_result(&self) -> Result<Arc<PrefilterResult>> {
        let result = self
            .evidence
            .cell
            .get_or_try_init(|| self.prefilter())
            .await
            .map_err(datafusion::error::DataFusionError::from)?;
        Ok(Arc::clone(result))
    }

    /// One embed call + one vector scan + one membership scan, shared by
    /// every partition through the evidence cell.
    async fn prefilter(&self) -> crate::Result<Arc<PrefilterResult>> {
        let hits = self.index.search(&self.condition, &self.params).await?;
        let indexed = self.index.indexed_doc_hashes().await?;
        Ok(Arc::new(PrefilterResult {
            chunks: bucket_chunks(hits, self.params.score_floor, self.params.chunks_per_doc),
            indexed,
        }))
    }

    /// The calibrated path (`WITH RECALL`): buffer up to `sample_size` input
    /// rows, calibrate the floor on them, then filter the buffered rows and
    /// the rest of the input as usual. One calibration globally — the sample
    /// comes from whichever partition's stream initializes the evidence
    /// cell first; later partitions just flow through the result.
    async fn calibrate_then_scan(
        self: Arc<Self>,
        mut input: SendableRecordBatchStream,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let sample_size = self
            .calibration
            .as_ref()
            .expect("calibrated path requires a config")
            .sample_size;
        let mut buffered: Vec<RecordBatch> = Vec::new();
        let mut buffered_rows = 0;
        while buffered_rows < sample_size {
            match input.try_next().await? {
                Some(batch) => {
                    buffered_rows += batch.num_rows();
                    buffered.push(batch);
                }
                None => break,
            }
        }
        let result = self
            .evidence
            .cell
            .get_or_try_init(|| self.calibrated_prefilter(&buffered))
            .await
            .map_err(datafusion::error::DataFusionError::from)?;
        let result = Arc::clone(result);
        let scanner = self;
        Ok(futures::stream::iter(buffered.into_iter().map(Ok))
            .chain(input)
            .and_then(move |batch| {
                let scanner = Arc::clone(&scanner);
                let result = Arc::clone(&result);
                async move { scanner.scan_batch(&result, batch) }
            })
            .boxed())
    }

    /// [`Scanner::prefilter`], with the floor calibrated on the buffered
    /// sample first. Same single embed call: searching with the floor
    /// dropped returns the identical `fetch_k`-nearest hit set unfiltered,
    /// so one search serves both the sample's score lookup and the final
    /// chunks map.
    async fn calibrated_prefilter(
        &self,
        buffered: &[RecordBatch],
    ) -> crate::Result<Arc<PrefilterResult>> {
        let calibration = self
            .calibration
            .as_ref()
            .expect("calibrated path requires a config");
        let params = SearchParams {
            score_floor: f32::NEG_INFINITY,
            ..self.params
        };
        let hits = self.index.search(&self.condition, &params).await?;
        let indexed = self.index.indexed_doc_hashes().await?;

        // Best chunk score per document — hits arrive best-first.
        let mut per_doc_best: HashMap<u64, f32> = HashMap::new();
        for hit in &hits {
            per_doc_best.entry(hit.doc_hash).or_insert(hit.score);
        }

        let sample = self.sample_documents(buffered, calibration.sample_size)?;
        let labels = self.label_sample(calibration, &sample).await;
        self.calibration_sampled_rows.add(labels.len());

        let mut scores = SampledScores {
            sampled: labels.len(),
            ..Default::default()
        };
        for (text, positive) in labels {
            if !positive {
                continue;
            }
            let hash = doc_hash(&text);
            match per_doc_best.get(&hash) {
                Some(&score) => scores.positive_scores.push(score),
                None if indexed.contains(&hash) => scores.positive_lost += 1,
                None => scores.positive_unindexed += 1,
            }
        }
        let calibrated =
            calibrate_threshold(calibration.target_recall, self.params.score_floor, &scores);

        Ok(Arc::new(PrefilterResult {
            chunks: bucket_chunks(hits, calibrated.threshold, self.params.chunks_per_doc),
            indexed,
        }))
    }

    /// The distinct documents of the buffered batches, at most `sample_size`.
    fn sample_documents(
        &self,
        buffered: &[RecordBatch],
        sample_size: usize,
    ) -> Result<Vec<String>> {
        let mut sample = Vec::new();
        let mut seen = HashSet::new();
        'batches: for batch in buffered {
            if batch.num_rows() == 0 {
                continue;
            }
            let texts = self.evaluate_texts(batch)?;
            let texts = texts
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("array was just cast to Utf8");
            for row in 0..texts.len() {
                if !texts.is_valid(row) {
                    continue;
                }
                let text = texts.value(row);
                if seen.insert(doc_hash(text)) {
                    sample.push(text.to_owned());
                    if sample.len() >= sample_size {
                        break 'batches;
                    }
                }
            }
        }
        Ok(sample)
    }

    /// Ground-truth labels for the sample: the model reads each document's
    /// full text — the definition of `MEANS`. Byte-identical requests and
    /// cache keys to the no-index verify path, so labels land in the verdict
    /// cache and repeat calibrations draw from it. A document whose call
    /// fails or answers unparseably drops out of the sample — rows fail,
    /// queries don't.
    async fn label_sample(
        &self,
        calibration: &CalibrationConfig,
        sample: &[String],
    ) -> Vec<(String, bool)> {
        let prompt = synthesize_means_prompt(&self.condition);
        let model_id = calibration.model.id();
        let mut labels = Vec::new();
        let mut misses = Vec::new();
        let mut requests = Vec::new();
        for text in sample {
            match calibration
                .cache
                .get(&means_cache_key(&self.condition, text, &model_id))
            {
                Some(CachedValue::Value(verdict)) => labels.push((text.clone(), verdict == "yes")),
                _ => {
                    misses.push(text);
                    requests.push(CompletionRequest {
                        system: prompt.clone(),
                        input: text.clone(),
                        max_tokens: 8,
                    });
                }
            }
        }
        self.calibration_model_calls.add(requests.len());

        let completions = calibration.model.complete(requests).await;
        debug_assert_eq!(completions.len(), misses.len());
        for (text, completion) in misses.into_iter().zip(&completions) {
            if let Ok(Some(matched)) = completion.as_ref().map(|c| parse_verdict(&c.text)) {
                calibration.cache.put(
                    means_cache_key(&self.condition, text, &model_id),
                    CachedValue::Value(if matched { "yes" } else { "no" }.to_owned()),
                );
                labels.push((text.clone(), matched));
            }
        }
        labels
    }
}

/// Bucket search hits into per-document evidence: chunks scoring at least
/// `floor`, best first, at most `chunks_per_doc` each.
fn bucket_chunks(
    hits: Vec<crate::index::ChunkHit>,
    floor: f32,
    chunks_per_doc: usize,
) -> HashMap<u64, Vec<String>> {
    let mut chunks: HashMap<u64, Vec<String>> = HashMap::new();
    for hit in hits {
        if hit.score < floor {
            continue;
        }
        let doc_chunks = chunks.entry(hit.doc_hash).or_default();
        if doc_chunks.len() < chunks_per_doc {
            doc_chunks.push(hit.text);
        }
    }
    chunks
}
