//! The index pre-filter stage — the funnel's cheap stage (roadmap step 2),
//! with `WITH RECALL` threshold calibration (step 3).
//!
//! Sits between the free predicates and [`VerifyExec`], pruning rows whose
//! indexed document scored below the floor against the embedded condition.
//! Uncalibrated, it costs one embed call per query and zero completion
//! calls; under `WITH RECALL`, it buffers its input and labels a random
//! sample of its documents (full-text model calls, shared with the verdict
//! cache) to set the floor at the recall target — thresholding either raw
//! similarity or a classifier fitted to those labels, whichever the sample
//! supports.
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
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::{Distribution, EquivalenceProperties, PhysicalExpr};
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
use crate::index::{FilterScan, SearchParams, SemanticIndex, doc_hash};
use crate::model::{CompletionRequest, ModelProvider};
use crate::optimizer::calibrate::{
    CALIBRATION_LABEL_BUDGET, Calibration, SampledScores, calibrate_threshold,
};
use crate::optimizer::proxy::LearnedProxy;
use crate::physical::verify::{
    MEANS_PROMPT_VERSION, means_cache_key, parse_verdict, synthesize_means_prompt,
};

/// A sampled document and its label: `Some(matched)`, or `None` when the
/// labeling call failed and its truth is unknown.
type Label = (String, Option<bool>);

/// Where each labeled document sits against the index: scored, never indexed
/// (passes through at any floor, so it helps recall for free), or indexed but
/// missing from the scores (lost at any floor — the scan is exhaustive and
/// reads scores and membership from one snapshot, so it produces none). A
/// document whose label failed counts as an unverified positive: the worst
/// case, since the scan cannot vouch for a match it could not check.
fn score_labels(
    labels: &[Label],
    per_doc_best: &HashMap<u64, f32>,
    indexed: &HashSet<u64>,
) -> SampledScores {
    let mut scores = SampledScores {
        sampled: labels.len(),
        ..Default::default()
    };
    for (text, label) in labels {
        match label {
            None => {
                scores.positive_unverified += 1;
                continue;
            }
            Some(false) => continue,
            Some(true) => {}
        }
        let hash = doc_hash(text);
        match per_doc_best.get(&hash) {
            Some(&score) => scores.positive_scores.push(score),
            None if indexed.contains(&hash) => scores.positive_lost += 1,
            None => scores.positive_unindexed += 1,
        }
    }
    scores
}

/// Seeds the calibration sample's document order. Fixed, so repeat queries
/// draw the same sample and their labels come from the verdict cache.
const SAMPLE_SEED: u64 = 0x5eed_ca1b_0000_0001;

/// A document's place in the calibration sample's draw order: SplitMix64 of
/// its [`doc_hash`] mixed with the seed. Taking the documents with the
/// smallest priorities is a uniform sample without replacement, and it
/// depends on nothing but the set of documents — not on input order, not on
/// partitioning.
fn sample_priority(doc_hash: u64) -> u64 {
    let mut z = (doc_hash ^ SAMPLE_SEED).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

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
    /// `WITH CONFIDENCE`: certify the target rather than estimate it. Turns
    /// the sample adaptive, since a bound needs however much evidence it
    /// needs, where an estimate is happy with whatever it is given.
    pub confidence: Option<f64>,
    /// Documents to label per tranche. Estimating buys exactly one.
    pub sample_size: usize,
    /// Labeling model — ground truth is this model reading the full text.
    pub model: Arc<dyn ModelProvider>,
    /// Verdict cache; labels are full-text verify verdicts and share keys.
    pub cache: Arc<dyn SemanticCache>,
}

impl CalibrationConfig {
    /// Labels this calibration may buy in total. Estimating stops after one
    /// tranche; certifying keeps drawing until the bound clears the target or
    /// the budget runs out.
    fn label_budget(&self) -> usize {
        match self.confidence {
            Some(_) => CALIBRATION_LABEL_BUDGET.max(self.sample_size),
            None => self.sample_size,
        }
    }
}

/// Filters input batches through one exhaustive similarity scan over the
/// semantic index: every indexed document is judged on its own score, never
/// on its rank, so the index changes the cost of a query and not its answer.
/// Rows the index never saw pass through — staleness must never silently
/// drop a row; it only costs a full-text verify call downstream.
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

    /// A calibrated scan samples its whole input, so it must see all of it
    /// in one stream; DataFusion merges the partitions below. Uncalibrated,
    /// each partition filters on its own.
    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.calibration.is_some() {
            vec![Distribution::SinglePartition]
        } else {
            vec![Distribution::UnspecifiedDistribution]
        }
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
        let input = self.input.execute(partition, Arc::clone(&context))?;
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
            calibration_label_errors: MetricBuilder::new(&self.metrics)
                .counter("calibration_label_errors", partition),
        });
        let stream: BoxStream<'_, Result<RecordBatch>> = if scanner.calibration.is_some() {
            // Calibration samples the whole input before anything can be
            // filtered: buffer it all, calibrate once, then filter the buffer.
            // The buffer is charged to the session's memory pool, so a query
            // too large for it fails cleanly instead of exhausting the process.
            let reservation = MemoryConsumer::new(format!("IndexScanExec[{partition}]"))
                .register(context.memory_pool());
            futures::stream::once(scanner.calibrate_then_scan(input, reservation))
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
        let output = Box::pin(RecordBatchStreamAdapter::new(self.input.schema(), stream));
        Ok(crate::physical::trace::trace_stage(
            "IndexScanExec",
            partition,
            output,
        ))
    }

    /// At most the input survives, and how much less depends on the floor —
    /// so the input's count, demoted to an estimate.
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
    calibration_label_errors: Count,
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
        // The search already applied `score_floor`, so every hit survives it
        // and bucketing needs no second filter.
        let scan = self
            .index
            .filter_scan(&self.condition, self.params.score_floor)
            .await?;
        Ok(Arc::new(PrefilterResult {
            chunks: bucket_chunks(scan.hits, self.params.chunks_per_doc),
            indexed: scan.indexed,
        }))
    }

    /// The calibrated path (`WITH RECALL`): buffer the whole input, draw a
    /// random sample of its documents, calibrate the floor on it, then filter
    /// the buffered rows. The input is a single partition (see
    /// `required_input_distribution`), so the sample is drawn from every row
    /// the predicate will judge — a sample from the head of the stream would
    /// see only one end of any input sorted by something the score tracks.
    ///
    /// Every buffered batch is charged to `reservation` before it is kept;
    /// the reservation lives as long as the output stream, which still holds
    /// the buffer.
    async fn calibrate_then_scan(
        self: Arc<Self>,
        mut input: SendableRecordBatchStream,
        reservation: MemoryReservation,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let mut buffered: Vec<RecordBatch> = Vec::new();
        while let Some(batch) = input.try_next().await? {
            reservation.try_grow(batch.get_array_memory_size())?;
            buffered.push(batch);
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
            .and_then(move |batch| {
                let scanner = Arc::clone(&scanner);
                let result = Arc::clone(&result);
                // Moved in so the charge is released only when the stream is.
                let _reservation = &reservation;
                async move { scanner.scan_batch(&result, batch) }
            })
            .boxed())
    }

    /// [`Scanner::prefilter`], with the floor calibrated on the buffered
    /// sample first. Same single embed call: searching with the floor
    /// dropped scores every chunk, so one search serves both the sample's
    /// score lookup and the final chunks map.
    async fn calibrated_prefilter(
        &self,
        buffered: &[RecordBatch],
    ) -> crate::Result<Arc<PrefilterResult>> {
        let calibration = self
            .calibration
            .as_ref()
            .expect("calibrated path requires a config");
        let FilterScan { hits, indexed } = self
            .index
            .filter_scan(&self.condition, f32::NEG_INFINITY)
            .await?;

        // Best chunk score per document — hits arrive best-first.
        let mut per_doc_best: HashMap<u64, f32> = HashMap::new();
        for hit in &hits {
            per_doc_best.entry(hit.doc_hash).or_insert(hit.score);
        }

        // Vectors the index already stores — the raw material for a proxy
        // fitted to this condition. An index that cannot supply them (or
        // fails to) simply leaves the cosine proxy in place.
        let vectors = self.index.doc_vectors().await.unwrap_or_default();

        // One tranche at a time. Estimating draws exactly one, because a
        // point estimate reads whatever the sample says. Certifying keeps
        // drawing until the bound clears the target, because a bound the
        // sample cannot support is not a bound — and stops at the budget,
        // reporting the shortfall rather than pruning on faith.
        let pool = self.sample_documents(buffered, calibration.label_budget())?;
        let mut labels: Vec<Label> = Vec::new();
        let mut labelled = 0;
        let mut attempt: Option<(Calibration, HashMap<u64, f32>)> = None;
        while labelled < pool.len() {
            let tranche = (labelled + calibration.sample_size).min(pool.len());
            labels.extend(
                self.label_sample(calibration, &pool[labelled..tranche])
                    .await,
            );
            labelled = tranche;

            let outcome = self.calibrate_over_best_proxy(
                calibration,
                &labels,
                &per_doc_best,
                &indexed,
                &vectors,
            );
            let settled =
                calibration.confidence.is_none() || outcome.0.meets(calibration.target_recall);
            attempt = Some(outcome);
            if settled {
                break;
            }
        }
        self.calibration_sampled_rows.add(labels.len());
        let (calibrated, scores) = match attempt {
            Some(outcome) => outcome,
            // Nothing to label: no rows, or every one of them NULL.
            None => (
                calibrate_threshold(
                    calibration.target_recall,
                    calibration.confidence,
                    self.params.score_floor,
                    &SampledScores::default(),
                ),
                per_doc_best.clone(),
            ),
        };

        if !calibrated.meets(calibration.target_recall) {
            tracing::warn!(
                target: "semcast::calibrate",
                condition = %self.condition,
                target_recall = calibration.target_recall,
                estimated_recall = calibrated.estimated_recall,
                certified_recall = ?calibrated.certified_recall,
                sampled_rows = calibrated.sampled_rows,
                "the sample cannot support the recall target; the index \
                 prunes nothing and every row is verified",
            );
        }

        // Two decisions, deliberately separate. Which documents survive is the
        // calibrated proxy's call — `scores` is cosine or a fitted
        // probability. Which *chunks* verify then reads is always the cosine
        // ranking, because that ranking is about where in the document the
        // condition is discussed, which a document-level probability has
        // nothing to say about.
        let mut chunks = bucket_chunks(hits, self.params.chunks_per_doc);
        chunks.retain(|hash, _| {
            scores
                .get(hash)
                .is_some_and(|score| *score >= calibrated.threshold)
        });
        Ok(Arc::new(PrefilterResult { chunks, indexed }))
    }

    /// Calibrate over the best proxy the labels support: a logistic
    /// regression fitted to *this* condition when there is enough evidence to
    /// fit one, raw cosine similarity otherwise.
    ///
    /// Returns the calibration and the per-document scores it was calibrated
    /// against, since the survivors must be decided on the same scale.
    ///
    /// The labels are split in half — the fit sees one half, the threshold is
    /// calibrated on the other. Calibrating on rows the model trained on would
    /// measure the fit rather than the population, and the bound would be
    /// certifying the wrong thing.
    fn calibrate_over_best_proxy(
        &self,
        calibration: &CalibrationConfig,
        labels: &[Label],
        per_doc_best: &HashMap<u64, f32>,
        indexed: &HashSet<u64>,
        vectors: &HashMap<u64, crate::model::Embedding>,
    ) -> (Calibration, HashMap<u64, f32>) {
        let calibrate_with = |scores: &HashMap<u64, f32>, labels: &[Label]| {
            calibrate_threshold(
                calibration.target_recall,
                calibration.confidence,
                self.params.score_floor,
                &score_labels(labels, scores, indexed),
            )
        };

        let (training, held_out) = labels.split_at(labels.len() / 2);
        let examples: Vec<(&crate::model::Embedding, bool)> = training
            .iter()
            .filter_map(|(text, label)| Some((vectors.get(&doc_hash(text))?, (*label)?)))
            .collect();

        // Every document the search returned needs a score on the new scale,
        // or it would be pruned for having no opinion rather than a low one.
        if let Some(proxy) = LearnedProxy::fit(&examples) {
            let learned: Option<HashMap<u64, f32>> = per_doc_best
                .keys()
                .map(|hash| Some((*hash, proxy.score(vectors.get(hash)?)?)))
                .collect();
            if let Some(learned) = learned {
                tracing::debug!(
                    target: "semcast::calibrate",
                    condition = %self.condition,
                    trained_on = examples.len(),
                    calibrated_on = held_out.len(),
                    "calibrating a learned proxy instead of raw similarity",
                );
                let calibrated = calibrate_with(&learned, held_out);
                return (calibrated, learned);
            }
        }
        // No fit: cosine over every label, since none of them trained anything.
        (calibrate_with(per_doc_best, labels), per_doc_best.clone())
    }

    /// A uniform random sample of the distinct documents in the buffered
    /// batches, at most `sample_size`, in draw order: the documents with the
    /// smallest [`sample_priority`]. Any prefix is itself a uniform sample,
    /// which is what lets certification label it a tranche at a time.
    ///
    /// Only hashes and row positions are kept while drawing; text is copied
    /// for the drawn documents alone, so the sample costs memory in its own
    /// size, not the input's.
    fn sample_documents(
        &self,
        buffered: &[RecordBatch],
        sample_size: usize,
    ) -> Result<Vec<String>> {
        let texts: Vec<Arc<dyn Array>> = buffered
            .iter()
            .map(|batch| self.evaluate_texts(batch))
            .collect::<Result<_>>()?;
        let mut first_seen: HashMap<u64, (usize, usize)> = HashMap::new();
        for (batch, array) in texts.iter().enumerate() {
            let array = as_strings(array);
            for row in 0..array.len() {
                if array.is_valid(row) {
                    first_seen
                        .entry(doc_hash(array.value(row)))
                        .or_insert((batch, row));
                }
            }
        }
        let mut drawn: Vec<(u64, (usize, usize))> = first_seen
            .into_iter()
            .map(|(hash, position)| (sample_priority(hash), position))
            .collect();
        drawn.sort_unstable_by_key(|(priority, _)| *priority);
        drawn.truncate(sample_size);
        Ok(drawn
            .into_iter()
            .map(|(_, (batch, row))| as_strings(&texts[batch]).value(row).to_owned())
            .collect())
    }

    /// Ground-truth labels for the sample: the model reads each document's
    /// full text — the definition of `MEANS`. Byte-identical requests and
    /// cache keys to the no-index verify path, so labels land in the verdict
    /// cache and repeat calibrations draw from it. A document whose call
    /// fails or answers unparseably stays in the sample, labelled `None`:
    /// rows fail, queries don't, but the certificate counts the failure as a
    /// possible miss rather than pretending the document was never drawn.
    /// Failures are not cached, so the next calibration retries them.
    async fn label_sample(&self, calibration: &CalibrationConfig, sample: &[String]) -> Vec<Label> {
        let prompt = synthesize_means_prompt(&self.condition);
        let model_id = calibration.model.id();
        let mut labels = Vec::new();
        let mut misses = Vec::new();
        let mut requests = Vec::new();
        for text in sample {
            match calibration.cache.get(&means_cache_key(
                &self.condition,
                text,
                &model_id,
                MEANS_PROMPT_VERSION,
            )) {
                Some(CachedValue::Value(verdict)) => {
                    labels.push((text.clone(), Some(verdict == "yes")));
                }
                _ => {
                    misses.push(text);
                    requests.push(CompletionRequest {
                        system: prompt.clone(),
                        input: text.clone(),
                        max_tokens: 8,
                        schema: None,
                    });
                }
            }
        }
        self.calibration_model_calls.add(requests.len());

        let completions = calibration.model.complete(requests).await;
        debug_assert_eq!(completions.len(), misses.len());
        for (text, completion) in misses.into_iter().zip(&completions) {
            match completion.as_ref().map(|c| parse_verdict(&c.text)) {
                Ok(Some(matched)) => {
                    calibration.cache.put(
                        means_cache_key(&self.condition, text, &model_id, MEANS_PROMPT_VERSION),
                        CachedValue::Value(if matched { "yes" } else { "no" }.to_owned()),
                    );
                    labels.push((text.clone(), Some(matched)));
                }
                Ok(None) | Err(_) => {
                    self.calibration_label_errors.add(1);
                    labels.push((text.clone(), None));
                }
            }
        }
        labels
    }
}

/// The Utf8 view of an array [`Scanner::evaluate_texts`] produced.
fn as_strings(array: &Arc<dyn Array>) -> &StringArray {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("array was cast to Utf8")
}

/// Bucket search hits into per-document evidence: chunks scoring at least
/// `floor`, best first, at most `chunks_per_doc` each.
fn bucket_chunks(
    hits: Vec<crate::index::ChunkHit>,
    chunks_per_doc: usize,
) -> HashMap<u64, Vec<String>> {
    let mut chunks: HashMap<u64, Vec<String>> = HashMap::new();
    for hit in hits {
        let doc_chunks = chunks.entry(hit.doc_hash).or_default();
        if doc_chunks.len() < chunks_per_doc {
            doc_chunks.push(hit.text);
        }
    }
    chunks
}
