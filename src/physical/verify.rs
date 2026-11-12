//! The verify stage — the physical operator that spends model calls
//! (roadmap step 1: verify-only execution).

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, StringArray};
use datafusion::arrow::compute::{cast, concat_batches, filter_record_batch};
use datafusion::arrow::datatypes::DataType;
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
use crate::index::doc_hash;
use crate::model::{CompletionRequest, ModelId, ModelProvider};
use crate::physical::index_scan::ChunkEvidence;
use crate::physical::store_lookup::StoreEvidence;

/// Version of the synthesized `MEANS` verify prompt. Participates in cache
/// keys: bump it and every cached verdict is honestly invalidated.
pub const MEANS_PROMPT_VERSION: &str = "means-v2";

/// The prompt version of a verify that reads chunk evidence rather than the
/// full text. Its own constant because the prompt differs: a one-chunk
/// document sends the same input either way, and sharing a key would serve a
/// full-text verdict for a chunked question.
pub const MEANS_CHUNKED_PROMPT_VERSION: &str = "means-v2-chunked";

/// The output cap of a yes/no verify request — a word, with room to spare.
pub const VERIFY_MAX_TOKENS: usize = 8;

/// The first wave of a verify under `LIMIT`: rows asked before checking
/// whether enough matched. Each later wave doubles, so a limited query asks at
/// most about twice the rows it needed, plus this.
pub const FIRST_LIMIT_WAVE: usize = 32;

/// Separates chunks in a chunked verify input; part of the cache-keyed
/// input, so changing it invalidates exactly the chunked verdicts.
pub const CHUNK_SEPARATOR: &str = "\n---\n";

/// The instruction the verify model sees. Users never write this.
pub fn synthesize_means_prompt(condition: &str) -> String {
    format!(
        "You are evaluating a predicate over a document. \
         Answer with exactly one word: yes or no.\n\n\
         Predicate: {condition}\n\n\
         Does the document satisfy the predicate?"
    )
}

/// The chunked variant: the model sees the document's best excerpts from
/// the semantic index, not the whole text.
pub fn synthesize_means_prompt_chunked(condition: &str) -> String {
    format!(
        "You are evaluating a predicate over excerpts from a document, \
         separated by `---`. The excerpts may be partial. \
         Answer with exactly one word: yes or no.\n\n\
         Predicate: {condition}\n\n\
         Do the excerpts show the document satisfies the predicate?"
    )
}

/// Filters input batches by asking the model whether each row's text meets
/// the condition. Ground truth for `MEANS` — every cheaper stage upstream is
/// an approximation of what this operator computes.
///
/// Row semantics ("rows fail, queries don't"): a NULL text never matches and
/// costs no call; a row whose model call errors or answers unparseably is
/// excluded and counted in the `rows_dropped` metric.
#[derive(Debug)]
pub struct VerifyExec {
    input: Arc<dyn ExecutionPlan>,
    /// Evaluates to the text under scrutiny, against input batches.
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    /// Chunk evidence from an upstream [`IndexScanExec`]; rows it covers are
    /// verified on their best chunks instead of the full text.
    ///
    /// [`IndexScanExec`]: crate::physical::index_scan::IndexScanExec
    evidence: Option<Arc<ChunkEvidence>>,
    /// The store's full-text verdicts: those rows are decided without a call.
    store: Option<Arc<StoreEvidence>>,
    /// How many chunks the evidence holds per document — display only.
    chunks_per_doc: usize,
    /// A `LIMIT` pushed down from above: each partition stops asking once
    /// this many rows matched.
    fetch: Option<usize>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl VerifyExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        condition: impl Into<String>,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
    ) -> Self {
        Self::build(input, text, condition, model, cache, None, 0)
    }

    /// A verify stage fed by an index pre-filter: rows with evidence are
    /// verified on their top chunks, unindexed passthrough rows on full text.
    pub fn new_with_evidence(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        condition: impl Into<String>,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
        evidence: Arc<ChunkEvidence>,
        chunks_per_doc: usize,
    ) -> Self {
        Self::build(
            input,
            text,
            condition,
            model,
            cache,
            Some(evidence),
            chunks_per_doc,
        )
    }

    fn build(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        condition: impl Into<String>,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
        evidence: Option<Arc<ChunkEvidence>>,
        chunks_per_doc: usize,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            // Pass-through filter: same partitioning as the input, and every
            // partition must be executed.
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            text,
            // Public constructor: normalize here too, so a caller outside
            // the rewrite still asks (and keys) the one question.
            condition: crate::semir::normalize_instruction(&condition.into()),
            model,
            cache,
            evidence,
            store: None,
            chunks_per_doc,
            fetch: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Honour what an upstream [`StoreLookupExec`] learned: a document with a
    /// stored full-text verdict is decided by it without a call — not even a
    /// chunked one, which would ask a weaker question than the one already
    /// answered.
    ///
    /// [`StoreLookupExec`]: crate::physical::StoreLookupExec
    pub fn with_store_evidence(mut self, store: Arc<StoreEvidence>) -> Self {
        self.store = Some(store);
        self
    }
}

impl DisplayAs for VerifyExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "VerifyExec: MEANS('{}') model={}",
            self.condition,
            self.model.id()
        )?;
        if self.evidence.is_some() {
            write!(f, " reads top-{} chunks per doc", self.chunks_per_doc)?;
        }
        if let Some(fetch) = self.fetch {
            write!(f, " fetch={fetch}")?;
        }
        // Know the bill before you run: worst-case model calls from input
        // statistics (cache hits and NULLs are free, so this is a ceiling).
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
        let mut verify = Self::build(
            Arc::clone(&children[0]),
            Arc::clone(&self.text),
            self.condition.clone(),
            Arc::clone(&self.model),
            Arc::clone(&self.cache),
            self.evidence.clone(),
            self.chunks_per_doc,
        );
        verify.store = self.store.clone();
        verify.fetch = self.fetch;
        Ok(Arc::new(verify))
    }

    /// Take a `LIMIT` from above, so the stage stops asking once it has the
    /// rows; without it, a whole input batch is asked before the limit sees
    /// a single row.
    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let mut verify = Self::build(
            Arc::clone(&self.input),
            Arc::clone(&self.text),
            self.condition.clone(),
            Arc::clone(&self.model),
            Arc::clone(&self.cache),
            self.evidence.clone(),
            self.chunks_per_doc,
        );
        verify.store = self.store.clone();
        verify.fetch = limit;
        Some(Arc::new(verify))
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let verifier = Arc::new(Verifier {
            text: Arc::clone(&self.text),
            condition: self.condition.clone(),
            prompt: synthesize_means_prompt(&self.condition),
            chunked_prompt: synthesize_means_prompt_chunked(&self.condition),
            model_id: self.model.id(),
            model: Arc::clone(&self.model),
            cache: Arc::clone(&self.cache),
            evidence: self.evidence.clone(),
            store: self.store.clone(),
            model_calls: MetricBuilder::new(&self.metrics).counter("model_calls", partition),
            model_calls_merged: MetricBuilder::new(&self.metrics)
                .counter("model_calls_merged", partition),
            cache_hits: MetricBuilder::new(&self.metrics).counter("cache_hits", partition),
            rows_dropped: MetricBuilder::new(&self.metrics).counter("rows_dropped", partition),
        });
        let output: SendableRecordBatchStream = match self.fetch {
            None => {
                let stream = input.and_then(move |batch| {
                    let verifier = Arc::clone(&verifier);
                    async move { verifier.verify_batch(batch).await }
                });
                Box::pin(RecordBatchStreamAdapter::new(self.input.schema(), stream))
            }
            // Ends once `wanted` rows matched, without pulling more input.
            Some(fetch) => {
                let stream = futures::stream::try_unfold(
                    (input, verifier, fetch),
                    |(mut input, verifier, wanted)| async move {
                        if wanted == 0 {
                            return Ok(None);
                        }
                        let Some(batch) = input.try_next().await? else {
                            return Ok(None);
                        };
                        let out = verifier.verify_until(batch, wanted).await?;
                        let wanted = wanted - out.num_rows();
                        Ok(Some((out, (input, verifier, wanted))))
                    },
                );
                Box::pin(RecordBatchStreamAdapter::new(self.input.schema(), stream))
            }
        };
        Ok(crate::physical::trace::trace_stage(
            "VerifyExec",
            partition,
            output,
        ))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// Everything one partition's stream needs, bundled so the closure stays
/// cheap to clone.
struct Verifier {
    store: Option<Arc<StoreEvidence>>,
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    prompt: String,
    chunked_prompt: String,
    model_id: ModelId,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    evidence: Option<Arc<ChunkEvidence>>,
    model_calls: Count,
    /// Of `model_calls`, those the scheduler answered from an identical
    /// request already in flight: no call of their own.
    model_calls_merged: Count,
    cache_hits: Count,
    rows_dropped: Count,
}

impl Verifier {
    /// Verify `batch` in waves until `wanted` rows matched, returning at
    /// most that many. The first wave is a few times the in-flight cap and
    /// each next one doubles, so a sparse match costs few round trips and a
    /// dense one few wasted calls.
    async fn verify_until(&self, batch: RecordBatch, wanted: usize) -> Result<RecordBatch> {
        let mut kept = Vec::new();
        let (mut found, mut start) = (0, 0);
        let mut wave = (2 * wanted).max(FIRST_LIMIT_WAVE);
        while start < batch.num_rows() && found < wanted {
            let len = wave.min(batch.num_rows() - start);
            let out = self.verify_batch(batch.slice(start, len)).await?;
            found += out.num_rows();
            kept.push(out);
            start += len;
            wave = wave.saturating_mul(2);
        }
        let out = concat_batches(&batch.schema(), &kept)?;
        Ok(out.slice(0, out.num_rows().min(wanted)))
    }

    async fn verify_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if batch.num_rows() == 0 {
            return Ok(batch);
        }
        let texts = self.text.evaluate(&batch)?.into_array(batch.num_rows())?;
        let texts = cast(&texts, &DataType::Utf8)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("array was just cast to Utf8");

        // NULL text rows never match and never reach the model; cached rows
        // never reach it either — first evaluation wins.
        let mut keep = vec![false; batch.num_rows()];
        let mut requests = Vec::new();
        let mut pending: Vec<(usize, String, bool)> = Vec::new();
        // A text repeated in the batch is asked once (the cache only learns
        // the answer afterwards); later copies take the first one's verdict.
        let mut first_ask: std::collections::HashMap<(String, bool), usize> =
            std::collections::HashMap::new();
        let mut copies: Vec<(usize, usize)> = Vec::new();
        for (row, keep_row) in keep.iter_mut().enumerate() {
            if !texts.is_valid(row) {
                continue;
            }
            let text = texts.value(row);
            if let Some(verdict) = self.store.as_deref().and_then(|store| store.verdict(text)) {
                // Counted by the store lookup, not as a cache hit here.
                *keep_row = verdict;
                continue;
            }
            // With index evidence, the model reads the document's best
            // chunks; without (no index, or an unindexed passthrough row),
            // the full text. The cache keys the input and the prompt version
            // actually sent, so the two kinds of verdict never collide — not
            // even for a one-chunk document, whose input is the same.
            let (input, prompt, chunked) = match self
                .evidence
                .as_deref()
                .and_then(|evidence| evidence.chunks_for(doc_hash(text)))
            {
                Some(chunks) => (chunks.join(CHUNK_SEPARATOR), &self.chunked_prompt, true),
                None => (text.to_owned(), &self.prompt, false),
            };
            if let Some(CachedValue::Value(verdict)) =
                self.cache.get(&self.cache_key(&input, chunked))
            {
                self.cache_hits.add(1);
                *keep_row = verdict == "yes";
                continue;
            }
            if let Some(&asked) = first_ask.get(&(input.clone(), chunked)) {
                copies.push((row, asked));
                continue;
            }
            first_ask.insert((input.clone(), chunked), pending.len());
            requests.push(CompletionRequest {
                system: prompt.clone(),
                input: input.clone(),
                max_tokens: VERIFY_MAX_TOKENS,
                schema: None,
                logprobs: true,
            });
            pending.push((row, input, chunked));
        }
        // Counted as sent, so progress shows calls while they run; merges
        // are known only once they answer.
        self.model_calls.add(requests.len());
        let verdicts = self.model.complete(requests).await;
        self.model_calls_merged
            .add(crate::physical::verify::merged_calls(&verdicts));
        debug_assert_eq!(verdicts.len(), pending.len());
        for ((row, input, chunked), verdict) in pending.iter().zip(&verdicts) {
            match verdict
                .as_ref()
                .map(|c| (parse_verdict(&c.text), c.yes_probability))
            {
                Ok((Some(matched), yes_probability)) => {
                    keep[*row] = matched;
                    // Only successful verdicts are cached: a transient model
                    // failure must not permanently exclude a row.
                    cache_verdict(
                        self.cache.as_ref(),
                        self.cache_key(input, *chunked),
                        matched,
                        yes_probability,
                    );
                }
                // Row-level failure: exclude and count, don't fail the query.
                Ok((None, _)) | Err(_) => self.rows_dropped.add(1),
            }
        }
        for (row, asked) in copies {
            let (source, ..) = pending[asked];
            keep[row] = keep[source];
            let answered = matches!(
                verdicts[asked].as_ref().map(|c| parse_verdict(&c.text)),
                Ok(Some(_))
            );
            if !answered {
                self.rows_dropped.add(1);
            }
        }
        Ok(filter_record_batch(&batch, &BooleanArray::from(keep))?)
    }

    fn cache_key(&self, input: &str, chunked: bool) -> CacheKey {
        let prompt_version = if chunked {
            MEANS_CHUNKED_PROMPT_VERSION
        } else {
            MEANS_PROMPT_VERSION
        };
        means_cache_key(&self.condition, input, &self.model_id, prompt_version)
    }
}

/// Full provenance: same condition + sent input + model + prompt scheme
/// → same verdict, across every query that ever asks again. Shared with
/// `WITH RECALL` calibration, whose ground-truth labels are exactly
/// full-text verdicts — the two stages pay for each other's cache.
///
/// `prompt_version` is a parameter, not the constant, because classify asks
/// the same question through a different prompt when it fuses several
/// conditions into one call. Passing [`MEANS_PROMPT_VERSION`] is what declares
/// "this request is byte-identical to a verify call, so share the entry";
/// anything else keeps the two apart, which is the honest default.
///
/// It is the answer key of `ASK(input, condition, BOOL)`: see
/// [`crate::semir`].
pub(crate) fn means_cache_key(
    condition: &str,
    input: &str,
    model_id: &ModelId,
    prompt_version: &str,
) -> CacheKey {
    crate::semir::answer_key(
        condition,
        crate::semir::AskType::Bool,
        input,
        model_id,
        prompt_version,
    )
}

/// Where a verdict's P("yes") is kept: the verdict's own key, another field.
/// A separate entry rather than a richer value, so a build that predates it
/// reads the verdict exactly as before.
///
/// The entry names the verdict it was measured with (`"yes 0.97"`). Each
/// entry keeps its first value independently, so a verdict stored without a
/// probability (an older build, a provider without logprobs) can later be
/// joined by a probability from a different answer; naming the verdict
/// lets [`cached_score`] refuse such a pair.
pub(crate) fn yes_probability_key(verdict: &CacheKey) -> CacheKey {
    CacheKey {
        field: "means.yes_probability".to_owned(),
        ..verdict.clone()
    }
}

/// Cache a verdict, and the probability the model gave it when it reported
/// one, together.
pub(crate) fn cache_verdict(
    cache: &dyn SemanticCache,
    key: CacheKey,
    matched: bool,
    yes_probability: Option<f32>,
) {
    let mut entries = Vec::with_capacity(2);
    if let Some(probability) = yes_probability {
        let verdict = if matched { "yes" } else { "no" };
        entries.push((
            yes_probability_key(&key),
            CachedValue::Value(format!("{verdict} {probability}")),
        ));
    }
    entries.push((
        key,
        CachedValue::Value(if matched { "yes" } else { "no" }.to_owned()),
    ));
    cache.put_many(entries);
}

/// What the cache settles about P("yes") for the verdict under `key`:
/// `None` when nothing is known yet (ask the model), `Some(None)` when it is
/// known there is no usable probability, `Some(Some(p))` for a probability
/// measured with the stored verdict.
///
/// "No usable probability" is an `"<verdict> unscored"` entry — a store
/// that keeps its first value holds a different verdict than the answer
/// that came with a probability — or a probability measured with a
/// different verdict. Each entry keeps its first value, so asking again
/// could never fix either: settling it is what stops a query from paying
/// for the same NULL every time. A provider that simply gave no probability
/// is not settled; the next answer may carry one.
pub(crate) fn cached_score(cache: &dyn SemanticCache, key: &CacheKey) -> Option<Option<f32>> {
    let Some(CachedValue::Value(verdict)) = cache.get(key) else {
        return None;
    };
    let Some(CachedValue::Value(entry)) = cache.get(&yes_probability_key(key)) else {
        return None;
    };
    let Some((measured_with, probability)) = entry.split_once(' ') else {
        return Some(None);
    };
    if measured_with != verdict {
        return Some(None);
    }
    Some(
        probability
            .parse()
            .ok()
            .filter(|p: &f32| (0.0..=1.0).contains(p)),
    )
}

/// Record that the verdict under `key` has no usable probability; see
/// [`cached_score`].
pub(crate) fn cache_unscored(cache: &dyn SemanticCache, key: &CacheKey, verdict: bool) {
    let verdict = if verdict { "yes" } else { "no" };
    cache.put(
        yes_probability_key(key),
        CachedValue::Value(format!("{verdict} unscored")),
    );
}

/// The answers in a batch the scheduler merged onto an identical request
/// already in flight — counted as sent, but they cost no call. Progress
/// reports subtract them.
pub(crate) fn merged_calls(completions: &[crate::Result<crate::model::Completion>]) -> usize {
    completions
        .iter()
        .filter(|completion| match completion {
            Ok(completion) => completion.merged,
            Err(error) => matches!(error, crate::SemcastError::MergedModel(_)),
        })
        .count()
}

/// `None` means the model didn't give a usable yes/no.
pub(crate) fn parse_verdict(text: &str) -> Option<bool> {
    let normalized = text.trim().trim_matches(|c: char| c.is_ascii_punctuation());
    if normalized.eq_ignore_ascii_case("yes") {
        Some(true)
    } else if normalized.eq_ignore_ascii_case("no") {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod yes_probability_cache_tests {
    use super::*;
    use crate::cache::InMemoryCache;

    fn key() -> CacheKey {
        means_cache_key(
            "is angry",
            "I am furious",
            &ModelId("m".to_owned()),
            MEANS_PROMPT_VERSION,
        )
    }

    #[test]
    fn a_probability_is_read_back_with_its_verdict() {
        let cache = InMemoryCache::default();
        cache_verdict(&cache, key(), true, Some(0.8));
        assert_eq!(cached_score(&cache, &key()).flatten(), Some(0.8));
        assert_eq!(
            cache.get(&key()),
            Some(CachedValue::Value("yes".to_owned()))
        );
    }

    #[test]
    fn a_probability_from_another_answer_is_refused() {
        let cache = InMemoryCache::default();
        // The verdict came first, without a probability; a later answer's
        // probability landed next to it.
        cache_verdict(&cache, key(), true, None);
        cache.put(
            yes_probability_key(&key()),
            CachedValue::Value("no 0.08".to_owned()),
        );
        assert_eq!(cached_score(&cache, &key()).flatten(), None);
    }

    #[test]
    fn an_unscored_verdict_is_settled() {
        let cache = InMemoryCache::default();
        cache_verdict(&cache, key(), true, None);
        assert_eq!(cached_score(&cache, &key()), None, "not known yet");
        cache_unscored(&cache, &key(), true);
        assert_eq!(cached_score(&cache, &key()), Some(None));
        assert_eq!(cached_score(&cache, &key()).flatten(), None);
    }

    #[test]
    fn no_verdict_or_no_probability_reads_as_none() {
        let cache = InMemoryCache::default();
        assert_eq!(cached_score(&cache, &key()).flatten(), None);
        cache_verdict(&cache, key(), false, None);
        assert_eq!(cached_score(&cache, &key()).flatten(), None);
    }
}
