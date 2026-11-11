//! The classify stage — `MEANS` as a label rather than a filter.
//!
//! Extends each row with one boolean per condition. Several conditions over
//! the same text become a single model call, which is why a three-branch
//! `CASE` costs what a one-branch `CASE` costs. The fusion is an optimization
//! only: each boolean answers exactly the question `VerifyExec` would ask, and
//! a row that needs just one answer is sent the verify prompt byte-for-byte,
//! so it shares that stage's cache entries outright.
//!
//! `CONFIDENCE(text MEANS 'condition')` runs here too, as a one-condition
//! node whose column is the model's P("yes") rather than the verdict. It asks
//! exactly the verify request, so it reads and writes the same cache entries
//! a `WHERE ... MEANS` does, plus the probability stored beside them.
//!
//! No index stage. A classify labels every row, so there is nothing to prune;
//! and the conditions would each want their own chunk set while the fused call
//! sends one input. The model reads the full text.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Builder, StringArray,
};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
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
use futures::TryStreamExt;

use crate::cache::{CacheKey, CachedValue, SemanticCache};
use crate::logical::sem_classify::ClassifyOutput;
use crate::model::{CompletionRequest, ModelId, ModelProvider};
use crate::physical::extract::parse_json_object;
use crate::physical::verify::{
    MEANS_PROMPT_VERSION, cache_unscored, cache_verdict, cached_score, means_cache_key,
    parse_verdict, synthesize_means_prompt,
};

/// Version of the *fused* prompt — the one that asks about several conditions
/// at once. A row with a single pending condition is sent the verify prompt
/// instead and keyed with [`MEANS_PROMPT_VERSION`], so the two never share an
/// entry unless the request really is identical.
pub const CLASSIFY_PROMPT_VERSION: &str = "classify-v1";

/// Enough for a small JSON object of booleans.
const CLASSIFY_MAX_TOKENS: usize = 128;

/// The instruction the fused classify model sees. Users never write this.
pub fn synthesize_classify_prompt(conditions: &[&str]) -> String {
    let mut prompt = String::from(
        "You are evaluating several independent predicates over one document. \
         Judge each one on its own merits; they are not alternatives and any \
         number of them may hold.\n\n",
    );
    for (i, condition) in conditions.iter().enumerate() {
        prompt.push_str(&format!("c{i}: {condition}\n"));
    }
    prompt.push_str(
        "\nReply with a JSON object mapping each predicate's key to true or \
         false, e.g. {\"c0\": true, \"c1\": false}.",
    );
    prompt
}

/// The JSON schema the fused answer is decoded under — one boolean per
/// pending condition, nothing else allowed.
fn classify_schema(count: usize) -> serde_json::Value {
    let properties: serde_json::Map<String, serde_json::Value> = (0..count)
        .map(|i| (format!("c{i}"), serde_json::json!({ "type": "boolean" })))
        .collect();
    let required: Vec<String> = (0..count).map(|i| format!("c{i}")).collect();
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

/// Labels each input row against a set of natural-language conditions,
/// appending one boolean column per condition.
///
/// Row semantics ("rows fail, queries don't"): a row the guard excludes costs
/// no call and gets `false` throughout; a NULL text costs no call and gets
/// NULL, like `NULL LIKE 'x'`. A row whose call fails
/// or answers unparseably gets NULL — which no `CASE` branch matches, so the
/// row falls through to the next branch or `ELSE` instead of taking the query
/// down with it.
///
/// For `CONFIDENCE()` ([`ClassifyOutput::YesProbability`]) every one of those
/// rows is NULL instead — no probability was measured — and so is a row whose
/// provider reports no probability (counted in `rows_unscored`).
#[derive(Debug)]
pub struct SemClassifyExec {
    input: Arc<dyn ExecutionPlan>,
    /// Evaluates to the text under scrutiny, against input batches.
    text: Arc<dyn PhysicalExpr>,
    conditions: Vec<String>,
    /// Rows this evaluates false (or NULL) for skip the model — an earlier
    /// `CASE` branch already claimed them.
    guard: Option<Arc<dyn PhysicalExpr>>,
    output: ClassifyOutput,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl SemClassifyExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        conditions: Vec<String>,
        guard: Option<Arc<dyn PhysicalExpr>>,
        output: ClassifyOutput,
        column_names: Vec<String>,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
    ) -> Result<Self> {
        let data_type = match output {
            ClassifyOutput::Verdict => DataType::Boolean,
            ClassifyOutput::YesProbability => DataType::Float64,
        };
        let mut fields = input.schema().fields().to_vec();
        for name in &column_names {
            fields.push(Arc::new(Field::new(name, data_type.clone(), true)));
        }
        let output_schema = Arc::new(Schema::new(fields));
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            // Row-wise: same partitioning as the input.
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            text,
            conditions,
            guard,
            output,
            model,
            cache,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    fn column_names(&self) -> Vec<String> {
        let start = self.input.schema().fields().len();
        self.output_schema.fields()[start..]
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }
}

impl DisplayAs for SemClassifyExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match self.output {
            ClassifyOutput::Verdict => write!(
                f,
                "SemClassifyExec: {} condition(s) model={}",
                self.conditions.len(),
                self.model.id()
            )?,
            ClassifyOutput::YesProbability => write!(
                f,
                "SemClassifyExec: CONFIDENCE('{}') model={}",
                self.conditions[0],
                self.model.id()
            )?,
        }
        if self.guard.is_some() {
            write!(f, " gated")?;
        }
        // Know the bill before you run: the fusion is why this is one call per
        // row rather than one per condition.
        match self
            .input
            .partition_statistics(None)
            .map(|stats| stats.num_rows)
        {
            Ok(Precision::Exact(rows)) => write!(f, "   ≤{rows} model calls"),
            Ok(Precision::Inexact(rows)) => write!(f, "   ~{rows} model calls"),
            _ => write!(f, "   model calls unknown"),
        }
    }
}

impl ExecutionPlan for SemClassifyExec {
    fn name(&self) -> &str {
        "SemClassifyExec"
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
            self.conditions.clone(),
            self.guard.clone(),
            self.output,
            self.column_names(),
            Arc::clone(&self.model),
            Arc::clone(&self.cache),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let classifier = Arc::new(Classifier {
            text: Arc::clone(&self.text),
            conditions: self.conditions.clone(),
            guard: self.guard.clone(),
            model_id: self.model.id(),
            model: Arc::clone(&self.model),
            cache: Arc::clone(&self.cache),
            output_schema: Arc::clone(&self.output_schema),
            model_calls: MetricBuilder::new(&self.metrics).counter("model_calls", partition),
            model_calls_merged: MetricBuilder::new(&self.metrics)
                .counter("model_calls_merged", partition),
            cache_hits: MetricBuilder::new(&self.metrics).counter("cache_hits", partition),
            rows_gated: MetricBuilder::new(&self.metrics).counter("rows_gated", partition),
            rows_failed: MetricBuilder::new(&self.metrics).counter("rows_failed", partition),
            branches_failed: MetricBuilder::new(&self.metrics)
                .counter("branches_failed", partition),
            rows_unscored: MetricBuilder::new(&self.metrics).counter("rows_unscored", partition),
        });
        let output = self.output;
        let stream = input.and_then(move |batch| {
            let classifier = Arc::clone(&classifier);
            async move {
                match output {
                    ClassifyOutput::Verdict => classifier.classify_batch(batch).await,
                    ClassifyOutput::YesProbability => classifier.score_batch(batch).await,
                }
            }
        });
        let output = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        ));
        Ok(crate::physical::trace::trace_stage(
            "SemClassifyExec",
            partition,
            output,
        ))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// A row that needs a model call, and which conditions it still lacks.
struct PendingRow {
    row: usize,
    text: String,
    pending: Vec<usize>,
}

/// Everything one partition's stream needs.
struct Classifier {
    text: Arc<dyn PhysicalExpr>,
    conditions: Vec<String>,
    guard: Option<Arc<dyn PhysicalExpr>>,
    model_id: ModelId,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    output_schema: SchemaRef,
    model_calls: Count,
    /// Of `model_calls`, those the scheduler answered from an identical
    /// request already in flight: no call of their own.
    model_calls_merged: Count,
    cache_hits: Count,
    rows_gated: Count,
    rows_failed: Count,
    branches_failed: Count,
    /// `CONFIDENCE()` rows whose answer came back without a probability: the
    /// provider reports none, or the cache already holds a different verdict.
    rows_unscored: Count,
}

impl Classifier {
    /// `CONFIDENCE()`: one P("yes") per row, for the single condition.
    ///
    /// A cached probability is served only alongside the verdict it was
    /// measured with. A cached verdict *without* one doesn't count — the
    /// row is asked again, with logprobs, and both are stored. A row known
    /// to have no usable probability is NULL without a call.
    async fn score_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(RecordBatch::new_empty(Arc::clone(&self.output_schema)));
        }
        let texts = self.text.evaluate(&batch)?.into_array(rows)?;
        let texts = cast(&texts, &DataType::Utf8)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text cast to Utf8");
        let guard = self.evaluate_guard(&batch)?;
        let condition = &self.conditions[0];

        let mut scores: Vec<Option<f64>> = vec![None; rows];
        let mut requests = Vec::new();
        let mut pending: Vec<(usize, CacheKey)> = Vec::new();
        for (row, score) in scores.iter_mut().enumerate() {
            if !guard_allows(guard.as_ref(), row) {
                self.rows_gated.add(1);
                continue;
            }
            if !texts.is_valid(row) {
                continue;
            }
            let text = texts.value(row);
            let key = means_cache_key(condition, text, &self.model_id, MEANS_PROMPT_VERSION);
            if let Some(cached) = cached_score(self.cache.as_ref(), &key) {
                self.cache_hits.add(1);
                if cached.is_none() {
                    self.rows_unscored.add(1);
                }
                *score = cached.map(f64::from);
                continue;
            }
            requests.push(self.request(text, &[0]));
            pending.push((row, key));
        }
        self.model_calls.add(requests.len());
        let completions = self.model.complete(requests).await;
        self.model_calls_merged
            .add(crate::physical::verify::merged_calls(&completions));
        debug_assert_eq!(completions.len(), pending.len());
        for ((row, key), completion) in pending.into_iter().zip(&completions) {
            let Some((completion, verdict)) = completion
                .as_ref()
                .ok()
                .and_then(|c| parse_verdict(&c.text).map(|v| (c, v)))
            else {
                self.rows_failed.add(1);
                continue;
            };
            // The stored verdict wins; a probability is kept only beside the
            // verdict it was measured with. A stored verdict that differs can
            // never be paired, so the row is settled as unscored and later
            // queries don't pay for the same NULL.
            let stored = match self.cache.get(&key) {
                Some(CachedValue::Value(stored)) => Some(stored == "yes"),
                _ => None,
            };
            scores[row] = match (stored, completion.yes_probability) {
                (Some(stored), _) if stored != verdict => {
                    cache_unscored(self.cache.as_ref(), &key, stored);
                    None
                }
                (_, Some(probability)) => {
                    cache_verdict(self.cache.as_ref(), key, verdict, Some(probability));
                    Some(f64::from(probability))
                }
                // Not settled: the next answer may well carry a probability
                // (another provider under the same id, a one-off parse miss).
                (_, None) => {
                    cache_verdict(self.cache.as_ref(), key, verdict, None);
                    None
                }
            };
            if scores[row].is_none() {
                self.rows_unscored.add(1);
            }
        }

        let mut builder = Float64Builder::with_capacity(rows);
        for score in scores {
            builder.append_option(score);
        }
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        columns.push(Arc::new(builder.finish()));
        Ok(RecordBatch::try_new(
            Arc::clone(&self.output_schema),
            columns,
        )?)
    }

    async fn classify_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(RecordBatch::new_empty(Arc::clone(&self.output_schema)));
        }
        let texts = self.text.evaluate(&batch)?.into_array(rows)?;
        let texts = cast(&texts, &DataType::Utf8)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text cast to Utf8");
        let guard = self.evaluate_guard(&batch)?;

        // One verdict slot per (row, condition); None stays NULL.
        let mut verdicts: Vec<Vec<Option<bool>>> = vec![vec![None; rows]; self.conditions.len()];
        let mut requests = Vec::new();
        let mut pending_rows: Vec<PendingRow> = Vec::new();

        for row in 0..rows {
            // An earlier CASE branch already claimed this row — it never
            // reaches the model, and `false` keeps it out of every branch here.
            if !guard_allows(guard.as_ref(), row) {
                self.rows_gated.add(1);
                for verdict in verdicts.iter_mut() {
                    verdict[row] = Some(false);
                }
                continue;
            }
            // NULL text costs no call and has no verdict: NULL, as SQL reads
            // `NULL LIKE 'x'`. No CASE branch matches it, and NOT keeps it
            // out too, which `false` would not.
            if !texts.is_valid(row) {
                continue;
            }
            let text = texts.value(row);

            let mut pending = Vec::new();
            for (branch, condition) in self.conditions.iter().enumerate() {
                match self.cached(condition, text, self.conditions.len()) {
                    Some(verdict) => {
                        self.cache_hits.add(1);
                        verdicts[branch][row] = Some(verdict);
                    }
                    None => pending.push(branch),
                }
            }
            if pending.is_empty() {
                continue;
            }
            requests.push(self.request(text, &pending));
            pending_rows.push(PendingRow {
                row,
                text: text.to_owned(),
                pending,
            });
        }
        // Counted as sent, so progress shows calls while they run; merges
        // are known only once they answer.
        self.model_calls.add(requests.len());
        let completions = self.model.complete(requests).await;
        self.model_calls_merged
            .add(crate::physical::verify::merged_calls(&completions));
        debug_assert_eq!(completions.len(), pending_rows.len());
        for (pending_row, completion) in pending_rows.iter().zip(&completions) {
            let Ok(completion) = completion.as_ref() else {
                self.rows_failed.add(1);
                continue;
            };
            self.apply(pending_row, completion, &mut verdicts);
        }

        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for verdict in verdicts {
            let mut builder = BooleanBuilder::with_capacity(rows);
            for value in verdict {
                builder.append_option(value);
            }
            columns.push(Arc::new(builder.finish()));
        }
        Ok(RecordBatch::try_new(
            Arc::clone(&self.output_schema),
            columns,
        )?)
    }

    /// One request for a row's pending conditions.
    ///
    /// A single pending condition takes the verify prompt verbatim — same
    /// system text, same schemaless yes/no answer — which is what lets its
    /// verdict share a cache entry with a `WHERE ... MEANS` over the same
    /// text. Two or more take the fused prompt and a JSON schema.
    fn request(&self, text: &str, pending: &[usize]) -> CompletionRequest {
        if let [branch] = pending {
            return CompletionRequest {
                system: synthesize_means_prompt(&self.conditions[*branch]),
                input: text.to_owned(),
                max_tokens: 8,
                schema: None,
                logprobs: true,
            };
        }
        let conditions: Vec<&str> = pending
            .iter()
            .map(|&branch| self.conditions[branch].as_str())
            .collect();
        CompletionRequest {
            system: synthesize_classify_prompt(&conditions),
            input: text.to_owned(),
            max_tokens: CLASSIFY_MAX_TOKENS,
            schema: Some(classify_schema(conditions.len())),
            logprobs: false,
        }
    }

    /// Decode one row's answer and cache each verdict it yielded. Conditions
    /// are cached individually, so editing one branch of a `CASE` leaves the
    /// others warm.
    fn apply(
        &self,
        pending_row: &PendingRow,
        completion: &crate::model::Completion,
        verdicts: &mut [Vec<Option<bool>>],
    ) {
        let answer = completion.text.as_str();
        let decoded: Vec<Option<bool>> = if pending_row.pending.len() == 1 {
            vec![parse_verdict(answer)]
        } else {
            match parse_json_object(answer) {
                Some(object) => (0..pending_row.pending.len())
                    .map(|i| object.get(&format!("c{i}")).and_then(|v| v.as_bool()))
                    .collect(),
                None => vec![None; pending_row.pending.len()],
            }
        };
        if decoded.iter().all(Option::is_none) {
            self.rows_failed.add(1);
            return;
        }
        for (&branch, verdict) in pending_row.pending.iter().zip(decoded) {
            let Some(verdict) = verdict else {
                self.branches_failed.add(1);
                continue;
            };
            verdicts[branch][pending_row.row] = Some(verdict);
            // Only successful verdicts are cached: a transient model failure
            // must not permanently mislabel a row. A fused answer has no
            // single yes/no token, so only a lone condition has a probability.
            let yes_probability = if pending_row.pending.len() == 1 {
                completion.yes_probability
            } else {
                None
            };
            cache_verdict(
                self.cache.as_ref(),
                self.cache_key(
                    &self.conditions[branch],
                    &pending_row.text,
                    pending_row.pending.len(),
                ),
                verdict,
                yes_probability,
            );
        }
    }

    fn cached(&self, condition: &str, text: &str, asked: usize) -> Option<bool> {
        // Try the entry this request would write first, then the verify
        // stage's: a one-condition classify and a WHERE ... MEANS produce the
        // same key, so either can satisfy the other.
        for key in [
            self.cache_key(condition, text, asked),
            means_cache_key(condition, text, &self.model_id, MEANS_PROMPT_VERSION),
        ] {
            if let Some(CachedValue::Value(verdict)) = self.cache.get(&key) {
                return Some(verdict == "yes");
            }
        }
        None
    }

    /// `asked` is how many conditions the request carries, which decides the
    /// prompt and therefore the provenance the verdict is keyed under.
    fn cache_key(&self, condition: &str, text: &str, asked: usize) -> CacheKey {
        let prompt_version = if asked == 1 {
            MEANS_PROMPT_VERSION
        } else {
            CLASSIFY_PROMPT_VERSION
        };
        means_cache_key(condition, text, &self.model_id, prompt_version)
    }

    fn evaluate_guard(&self, batch: &RecordBatch) -> Result<Option<BooleanArray>> {
        let Some(guard) = &self.guard else {
            return Ok(None);
        };
        let values = guard.evaluate(batch)?.into_array(batch.num_rows())?;
        let values = cast(&values, &DataType::Boolean)?;
        Ok(Some(
            values
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("guard cast to Boolean")
                .clone(),
        ))
    }
}

/// A guard admits a row only when it is definitely true — NULL means an
/// earlier branch's condition was unknown, which no `CASE` treats as a match.
fn guard_allows(guard: Option<&BooleanArray>, row: usize) -> bool {
    match guard {
        None => true,
        Some(guard) => guard.is_valid(row) && guard.value(row),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_requires_one_boolean_per_pending_condition() {
        let schema = classify_schema(2);
        assert_eq!(schema["properties"]["c0"]["type"], "boolean");
        assert_eq!(schema["properties"]["c1"]["type"], "boolean");
        assert_eq!(schema["required"], serde_json::json!(["c0", "c1"]));
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }

    #[test]
    fn fused_prompt_numbers_every_condition() {
        let prompt = synthesize_classify_prompt(&["is angry", "asks about pricing"]);
        assert!(prompt.contains("c0: is angry"), "{prompt}");
        assert!(prompt.contains("c1: asks about pricing"), "{prompt}");
        // The predicates are independent, not a menu to pick one from.
        assert!(prompt.contains("any number of them may hold"), "{prompt}");
    }

    #[test]
    fn a_guard_admits_only_definite_truth() {
        let guard = BooleanArray::from(vec![Some(true), Some(false), None]);
        assert!(guard_allows(Some(&guard), 0));
        assert!(!guard_allows(Some(&guard), 1));
        assert!(!guard_allows(Some(&guard), 2), "NULL is not a match");
        assert!(guard_allows(None, 0), "no guard admits everything");
    }
}
