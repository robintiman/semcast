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
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};

use crate::index::doc_hash;
use crate::model::{CompletionRequest, ModelProvider};
use crate::optimizer::calibrate::{Calibration, SampledScores, calibrate_threshold};
use crate::physical::index_scan::CalibrationConfig;
use crate::physical::sample::{Label, Labeler, draw_sample};
use crate::physical::verify::{
    MEANS_PROMPT_VERSION, cache_verdict, cached_score, means_cache_key, merged_calls,
    parse_verdict, synthesize_means_prompt,
};

/// The floor when the sample holds no known match to set one by: the
/// scorer's own yes/no boundary.
pub const DEFAULT_SCORER_FLOOR: f32 = 0.5;

/// Prunes rows whose scorer P("yes") falls below a floor calibrated to the
/// recall target, ahead of the session model's verify.
#[derive(Debug)]
pub struct ScoreScanExec {
    input: Arc<dyn ExecutionPlan>,
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    scorer: Arc<dyn ModelProvider>,
    calibration: CalibrationConfig,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl ScoreScanExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        condition: impl Into<String>,
        scorer: Arc<dyn ModelProvider>,
        calibration: CalibrationConfig,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
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
            calibration,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
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

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.text),
            self.condition.clone(),
            Arc::clone(&self.scorer),
            self.calibration.clone(),
        )))
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
            calibration: self.calibration.clone(),
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
        let strings: Vec<&StringArray> = texts.iter().map(as_strings).collect();

        let scores = self.score(&strings).await;
        let threshold = self.calibrate(&strings, &scores).await.threshold;

        let mut output = Vec::with_capacity(buffered.len());
        for (batch, texts) in buffered.into_iter().zip(&strings) {
            let mut keep = vec![false; batch.num_rows()];
            for (row, keep_row) in keep.iter_mut().enumerate() {
                // NULL text never matches — free to drop here.
                if !texts.is_valid(row) {
                    self.rows_pruned.add(1);
                    continue;
                }
                match scores.get(&doc_hash(texts.value(row))) {
                    Some(score) if *score >= threshold => *keep_row = true,
                    Some(_) => self.rows_pruned.add(1),
                    None => {
                        self.passthrough_rows.add(1);
                        *keep_row = true;
                    }
                }
            }
            output.push(filter_record_batch(&batch, &BooleanArray::from(keep)));
        }
        drop(reservation);
        Ok(
            futures::stream::iter(output.into_iter().map(|batch| batch.map_err(Into::into)))
                .boxed(),
        )
    }

    /// The scorer's P("yes") for every distinct document, from the cache
    /// where it has one. A document with no usable probability is absent.
    async fn score(&self, texts: &[&StringArray]) -> HashMap<u64, f32> {
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
                if !asked.insert(hash) {
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
        self.scorer_calls.add(requests.len());
        let completions = self.scorer.complete(requests).await;
        self.scorer_calls_merged.add(merged_calls(&completions));
        let cache = self.calibration.cache.as_ref();
        for ((hash, key), completion) in pending.into_iter().zip(&completions) {
            let Ok(completion) = completion else { continue };
            let Some(probability) = completion.yes_probability else {
                continue;
            };
            if let Some(verdict) = parse_verdict(&completion.text) {
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
        scores
    }

    /// The floor: label a random sample with the session model and set it
    /// at the recall target, one tranche at a time when certifying.
    async fn calibrate(&self, texts: &[&StringArray], scores: &HashMap<u64, f32>) -> Calibration {
        let calibration = &self.calibration;
        let labeler = Labeler {
            condition: &self.condition,
            model: calibration.model.as_ref(),
            cache: calibration.cache.as_ref(),
            calls: &self.calibration_model_calls,
            calls_merged: &self.calibration_model_calls_merged,
            errors: &self.calibration_label_errors,
        };
        let budget = match calibration.confidence {
            Some(_) => {
                crate::optimizer::calibrate::CALIBRATION_LABEL_BUDGET.max(calibration.sample_size)
            }
            None => calibration.sample_size,
        };
        let pool = draw_sample(texts, budget);
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
        calibrated
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
