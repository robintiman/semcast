//! The scorer pre-filter — a cheap model's P("yes") as the funnel's first
//! stage for a `MEANS` no semantic index covers.
//!
//! The scorer is a second, smaller chat model asked the verify question with
//! logprobs, so its score is the probability it would say yes. It reads
//! every row; the session model then verifies only those scoring at or above
//! a floor. The floor is not a guess: it is calibrated per query on a random
//! sample labeled by the session model, exactly like the index scan's
//! (`WITH RECALL`, certified under `WITH CONFIDENCE`). Without a recall
//! target the planner never builds this stage — a `MEANS` keeps meaning
//! what the session model says, row for row.
//!
//! A row the scorer gives no probability passes through to verify: missing
//! evidence costs a call, never a row. Scores are cached under the scorer's
//! own model id, so a repeat query costs no scorer calls either.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use datafusion::arrow::array::{Array, BooleanArray, StringArray};
use datafusion::arrow::compute::{cast, filter_record_batch};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Statistics;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::{Distribution, PhysicalExpr};
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

use crate::index::doc_hash;
use crate::model::{CompletionRequest, ModelProvider};
use crate::optimizer::calibrate::{Calibration, SampledScores, calibrate_threshold};
use crate::physical::index_scan::CalibrationConfig;
use crate::physical::sample::{Label, Labeler, draw_sample};
use crate::physical::store_lookup::StoreEvidence;
use crate::physical::verify::{
    MEANS_PROMPT_VERSION, cache_verdict, cached_score, means_cache_key, merged_calls,
    parse_verdict, synthesize_means_prompt,
};

/// The floor when the sample holds no known match to set one by: none. The
/// scorer's own yes/no is exactly the judgement the sample exists to check,
/// so with nothing to check it against, it prunes nothing.
pub const DEFAULT_SCORER_FLOOR: f32 = f32::NEG_INFINITY;

/// How many answers without a probability, and none ever with one, it
/// takes to conclude the scorer reports none.
const MIN_ANSWERS_TO_SWITCH_OFF: usize = 8;

/// What a session has learned about its scorer, shared by every scan that
/// uses it. In memory only, on purpose: see [`Scanner::score`].
#[derive(Debug, Default)]
pub struct ScorerHealth {
    /// Set once the scorer has given a probability, fresh or cached: from
    /// then on a missing one is a per-row miss, never a reason to stop.
    scored: AtomicBool,
    /// Answers without a probability while `scored` is unset.
    unscored_answers: AtomicUsize,
    no_logprobs: AtomicBool,
}

impl ScorerHealth {
    fn record(&self, with_probability: usize, without: usize) {
        if with_probability > 0 {
            // Also undoes a switch-off another query made meanwhile.
            self.scored.store(true, Ordering::Relaxed);
            self.no_logprobs.store(false, Ordering::Relaxed);
            return;
        }
        if self.scored.load(Ordering::Relaxed) {
            return;
        }
        let total = self.unscored_answers.fetch_add(without, Ordering::Relaxed) + without;
        if total >= MIN_ANSWERS_TO_SWITCH_OFF {
            self.no_logprobs.store(true, Ordering::Relaxed);
        }
    }
}

/// Prunes rows whose scorer P("yes") falls below a floor calibrated to the
/// recall target, ahead of the session model's verify.
#[derive(Debug)]
pub struct ScoreScanExec {
    input: Arc<dyn ExecutionPlan>,
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    scorer: Arc<dyn ModelProvider>,
    health: Arc<ScorerHealth>,
    calibration: CalibrationConfig,
    /// The store's verdicts: those documents are decided by them, and never
    /// scored.
    store: Option<Arc<StoreEvidence>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl ScoreScanExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        condition: impl Into<String>,
        scorer: Arc<dyn ModelProvider>,
        health: Arc<ScorerHealth>,
        calibration: CalibrationConfig,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            // A filter: the input's orderings and equivalences still hold.
            input.equivalence_properties().clone(),
            // One partition in (see `required_input_distribution`), one out.
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            input,
            text,
            condition: condition.into(),
            scorer,
            health,
            calibration,
            store: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Honour what an upstream [`StoreLookupExec`] learned: a document with
    /// a stored verdict is kept or pruned by it, and costs no scorer call.
    ///
    /// [`StoreLookupExec`]: crate::physical::StoreLookupExec
    pub fn with_store_evidence(mut self, store: Arc<StoreEvidence>) -> Self {
        self.store = Some(store);
        self
    }
}

impl DisplayAs for ScoreScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ScoreScanExec: MEANS('{}') scorer={} floor=calibrated(recall≥{:.2}, sample≤{})",
            self.condition,
            self.scorer.id(),
            self.calibration.target_recall,
            self.calibration.sample_size,
        )?;
        if let Some(confidence) = self.calibration.confidence {
            write!(f, " certified at {confidence:.2}")?;
        }
        match self
            .input
            .partition_statistics(None)
            .map(|stats| stats.num_rows)
        {
            Ok(datafusion::common::stats::Precision::Exact(rows)) => {
                write!(f, "   ≤{rows} scorer calls")
            }
            Ok(datafusion::common::stats::Precision::Inexact(rows)) => {
                write!(f, "   ~{rows} scorer calls")
            }
            _ => write!(f, "   scorer calls unknown"),
        }
    }
}

impl ExecutionPlan for ScoreScanExec {
    fn name(&self) -> &str {
        "ScoreScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// The sample is drawn from every row the predicate will judge, so this
    /// stage must see all of them in one stream.
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut scan = Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.text),
            self.condition.clone(),
            Arc::clone(&self.scorer),
            Arc::clone(&self.health),
            self.calibration.clone(),
        );
        scan.store = self.store.clone();
        Ok(Arc::new(scan))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let metric =
            |name: &'static str| MetricBuilder::new(&self.metrics).counter(name, partition);
        let scanner = Arc::new(Scanner {
            text: Arc::clone(&self.text),
            condition: self.condition.clone(),
            scorer: Arc::clone(&self.scorer),
            health: Arc::clone(&self.health),
            calibration: self.calibration.clone(),
            store: self.store.clone(),
            scorer_calls: metric("scorer_calls"),
            scorer_calls_merged: metric("scorer_calls_merged"),
            scorer_cache_hits: metric("scorer_cache_hits"),
            rows_pruned: metric("rows_pruned"),
            passthrough_rows: metric("passthrough_rows"),
            calibration_sampled_rows: metric("calibration_sampled_rows"),
            calibration_model_calls: metric("calibration_model_calls"),
            calibration_model_calls_merged: metric("calibration_model_calls_merged"),
            calibration_label_errors: metric("calibration_label_errors"),
        });
        // Calibration samples the whole input before anything can be
        // filtered: buffer it, charged to the session's memory pool.
        let reservation = MemoryConsumer::new(format!("ScoreScanExec[{partition}]"))
            .register(context.memory_pool());
        let stream: BoxStream<'static, Result<RecordBatch>> =
            futures::stream::once(scanner.score_then_scan(input, reservation))
                .try_flatten()
                .boxed();
        let output = Box::pin(RecordBatchStreamAdapter::new(self.input.schema(), stream));
        Ok(crate::physical::trace::trace_stage(
            "ScoreScanExec",
            partition,
            output,
        ))
    }

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

struct Scanner {
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    scorer: Arc<dyn ModelProvider>,
    health: Arc<ScorerHealth>,
    store: Option<Arc<StoreEvidence>>,
    calibration: CalibrationConfig,
    scorer_calls: Count,
    /// Of `scorer_calls`, those merged onto an identical call in flight.
    scorer_calls_merged: Count,
    scorer_cache_hits: Count,
    rows_pruned: Count,
    /// Rows the scorer gave no probability, sent on to verify.
    passthrough_rows: Count,
    calibration_sampled_rows: Count,
    calibration_model_calls: Count,
    calibration_model_calls_merged: Count,
    calibration_label_errors: Count,
}

impl Scanner {
    fn stored_verdict(&self, hash: u64) -> Option<bool> {
        self.store.as_deref().and_then(|store| store.verdict(hash))
    }

    async fn score_then_scan(
        self: Arc<Self>,
        mut input: SendableRecordBatchStream,
        reservation: MemoryReservation,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let mut buffered: Vec<RecordBatch> = Vec::new();
        while let Some(batch) = input.try_next().await? {
            reservation.try_grow(batch.get_array_memory_size())?;
            buffered.push(batch);
        }
        let texts: Vec<Arc<dyn Array>> = buffered
            .iter()
            .map(|batch| self.evaluate_texts(batch))
            .collect::<Result<_>>()?;
        // A plain Utf8 column evaluates to the batch's own array, already
        // charged; only a computed or cast text is new memory.
        for (array, batch) in texts.iter().zip(&buffered) {
            let shared = batch.columns().iter().any(|column| {
                column.to_data().buffers().first().map(|b| b.as_ptr())
                    == array.to_data().buffers().first().map(|b| b.as_ptr())
            });
            if !shared {
                reservation.try_grow(array.get_array_memory_size())?;
            }
        }
        let strings: Vec<&StringArray> = texts.iter().map(as_strings).collect();

        // The sample first: its documents need scores even when the store
        // already knows their verdict, or calibration has nothing to read.
        let pool = draw_sample(&strings, self.label_budget());
        let sampled: HashSet<u64> = pool.iter().map(|text| doc_hash(text)).collect();
        let scores = self.score(&strings, &sampled).await;
        let (calibrated, labels) = self.calibrate(&pool, &scores).await;
        // A sampled match is already paid for, and a failed label may be
        // one: never prune either, whatever the scorer said. Verify reads
        // their labels from the cache.
        let labelled: HashSet<u64> = labels
            .iter()
            .filter(|(_, label)| *label != Some(false))
            .map(|(text, _)| doc_hash(text))
            .collect();

        let mut output = Vec::with_capacity(buffered.len());
        for (batch, texts) in buffered.iter().zip(&strings) {
            let mut keep = vec![false; batch.num_rows()];
            for (row, keep_row) in keep.iter_mut().enumerate() {
                // NULL text never matches — free to drop here.
                if !texts.is_valid(row) {
                    self.rows_pruned.add(1);
                    continue;
                }
                let hash = doc_hash(texts.value(row));
                let stored = self.stored_verdict(hash);
                match scores.get(&hash) {
                    _ if stored == Some(false) => self.rows_pruned.add(1),
                    _ if labelled.contains(&hash) || stored == Some(true) => *keep_row = true,
                    Some(score) if *score >= calibrated.threshold => *keep_row = true,
                    Some(_) => self.rows_pruned.add(1),
                    None => {
                        self.passthrough_rows.add(1);
                        *keep_row = true;
                    }
                }
            }
            output.push(filter_record_batch(batch, &BooleanArray::from(keep))?);
        }
        drop(texts);
        drop(buffered);
        // Hold the charge for the filtered copies as long as the stream
        // holds them.
        reservation.resize(output.iter().map(RecordBatch::get_array_memory_size).sum());
        Ok(futures::stream::iter(output.into_iter().map(move |batch| {
            let _reservation = &reservation;
            Ok(batch)
        }))
        .boxed())
    }

    /// The scorer's P("yes") for every distinct document, from the cache
    /// where it has one. A document with no usable probability is absent.
    async fn score(&self, texts: &[&StringArray], sampled: &HashSet<u64>) -> HashMap<u64, f32> {
        let scorer_id = self.scorer.id();
        let prompt = synthesize_means_prompt(&self.condition);
        let mut scores = HashMap::new();
        let mut asked: HashSet<u64> = HashSet::new();
        let mut pending = Vec::new();
        let mut requests = Vec::new();
        for array in texts {
            for row in 0..array.len() {
                if !array.is_valid(row) {
                    continue;
                }
                let text = array.value(row);
                let hash = doc_hash(text);
                // A stored verdict decides the row whatever it scores: don't
                // pay for a score.
                if !asked.insert(hash)
                    || (self.stored_verdict(hash).is_some() && !sampled.contains(&hash))
                {
                    continue;
                }
                let key = means_cache_key(&self.condition, text, &scorer_id, MEANS_PROMPT_VERSION);
                match cached_score(self.calibration.cache.as_ref(), &key) {
                    Some(cached) => {
                        self.scorer_cache_hits.add(1);
                        if let Some(score) = cached {
                            scores.insert(hash, score);
                        }
                    }
                    None => {
                        requests.push(CompletionRequest {
                            system: prompt.clone(),
                            input: text.to_owned(),
                            max_tokens: 8,
                            schema: None,
                            logprobs: true,
                        });
                        pending.push((hash, key));
                    }
                }
            }
        }
        if !scores.is_empty() {
            self.health.record(scores.len(), 0);
        }
        if self.health.no_logprobs.load(Ordering::Relaxed) {
            // Every row passes through; asking would buy nothing.
            return scores;
        }
        self.scorer_calls.add(requests.len());
        let completions = self.scorer.complete(requests).await;
        let answered = completions.iter().filter(|c| c.is_ok()).count();
        let mut with_probability = 0;
        self.scorer_calls_merged.add(merged_calls(&completions));
        let cache = self.calibration.cache.as_ref();
        for ((hash, key), completion) in pending.into_iter().zip(&completions) {
            let Ok(completion) = completion else { continue };
            let verdict = parse_verdict(&completion.text);
            let Some(probability) = completion.yes_probability else {
                // Not cached: the next answer may carry one (see
                // `cached_score`). A scorer that never gives one is switched
                // off for the process below.
                continue;
            };
            with_probability += 1;
            if let Some(verdict) = verdict {
                cache_verdict(cache, key.clone(), verdict, Some(probability));
            }
            // What the cache now pairs, so this query scores as later ones
            // will; a store that keeps an older, different verdict has
            // settled this document as unscored.
            let score = match cached_score(cache, &key) {
                Some(settled) => settled,
                None => Some(probability),
            };
            if let Some(score) = score {
                scores.insert(hash, score);
            }
        }
        // A scorer that has never given a probability in this process, after
        // enough answers, reports none. Remembered in memory only, so one
        // that starts reporting them (another build, another server) is used
        // again after a restart, and nothing durable claims otherwise.
        let was_off = self.health.no_logprobs.load(Ordering::Relaxed);
        self.health
            .record(with_probability, answered - with_probability);
        if !was_off && self.health.no_logprobs.load(Ordering::Relaxed) {
            tracing::warn!(
                target: "semcast::scorer",
                scorer = %self.scorer.id(),
                "the scorer reports no probabilities (no logprobs); it is \
                 switched off until restart and every row is verified",
            );
        }
        scores
    }

    /// The floor: label a random sample with the session model and set it
    /// at the recall target, one tranche at a time when certifying.
    /// Returns the calibration and the labels it was drawn from.
    fn label_budget(&self) -> usize {
        match self.calibration.confidence {
            Some(_) => crate::optimizer::calibrate::CALIBRATION_LABEL_BUDGET
                .max(self.calibration.sample_size),
            None => self.calibration.sample_size,
        }
    }

    async fn calibrate(
        &self,
        pool: &[String],
        scores: &HashMap<u64, f32>,
    ) -> (Calibration, Vec<Label>) {
        let calibration = &self.calibration;
        let labeler = Labeler {
            condition: &self.condition,
            model: calibration.model.as_ref(),
            cache: calibration.cache.as_ref(),
            calls: &self.calibration_model_calls,
            calls_merged: &self.calibration_model_calls_merged,
            errors: &self.calibration_label_errors,
        };
        let mut labels: Vec<Label> = Vec::new();
        let mut labelled = 0;
        let mut calibrated = None;
        while labelled < pool.len() {
            let tranche = (labelled + calibration.sample_size).min(pool.len());
            labels.extend(labeler.label(&pool[labelled..tranche]).await);
            labelled = tranche;
            let attempt = calibrate_threshold(
                calibration.target_recall,
                calibration.confidence,
                DEFAULT_SCORER_FLOOR,
                &sampled_scores(&labels, scores),
            );
            let settled =
                calibration.confidence.is_none() || attempt.meets(calibration.target_recall);
            calibrated = Some(attempt);
            if settled {
                break;
            }
        }
        self.calibration_sampled_rows.add(labels.len());
        let calibrated = calibrated.unwrap_or_else(|| {
            calibrate_threshold(
                calibration.target_recall,
                calibration.confidence,
                DEFAULT_SCORER_FLOOR,
                &SampledScores::default(),
            )
        });
        if !calibrated.meets(calibration.target_recall) {
            tracing::warn!(
                target: "semcast::calibrate",
                condition = %self.condition,
                target_recall = calibration.target_recall,
                estimated_recall = calibrated.estimated_recall,
                certified_recall = ?calibrated.certified_recall,
                sampled_rows = calibrated.sampled_rows,
                "the sample cannot support the recall target; the scorer \
                 prunes nothing and every row is verified",
            );
        }
        (calibrated, labels)
    }

    fn evaluate_texts(&self, batch: &RecordBatch) -> Result<Arc<dyn Array>> {
        let texts = self.text.evaluate(batch)?.into_array(batch.num_rows())?;
        Ok(cast(&texts, &DataType::Utf8)?)
    }
}

/// Where each labeled document sits against the scores. A document without
/// a score passes through at any floor, like one the index never saw; a
/// failed label is recorded by its score, and the calibration assumes the
/// worst for it.
fn sampled_scores(labels: &[Label], scores: &HashMap<u64, f32>) -> SampledScores {
    let mut sampled = SampledScores {
        sampled: labels.len(),
        ..Default::default()
    };
    for (text, label) in labels {
        let score = scores.get(&doc_hash(text)).copied();
        match (label, score) {
            (Some(false), _) => {}
            (Some(true), Some(score)) => sampled.positive_scores.push(score),
            (Some(true), None) => sampled.positive_unindexed += 1,
            (None, Some(score)) => sampled.unverified_scores.push(score),
            (None, None) => {}
        }
    }
    sampled
}

fn as_strings(array: &Arc<dyn Array>) -> &StringArray {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("array was cast to Utf8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unscored_positives_pass_through_and_failures_count_against() {
        let scores = HashMap::from([(doc_hash("a"), 0.9), (doc_hash("c"), 0.2)]);
        let labels: Vec<Label> = vec![
            ("a".into(), Some(true)),
            ("b".into(), Some(true)),
            ("c".into(), None),
            ("d".into(), Some(false)),
        ];
        let sampled = sampled_scores(&labels, &scores);
        assert_eq!(sampled.positive_scores, [0.9]);
        assert_eq!(sampled.positive_unindexed, 1);
        assert_eq!(sampled.unverified_scores, [0.2]);
        assert_eq!(sampled.sampled, 4);
    }
}
