//! The rank stage — `x RELEVANCE TO 'q'` is `ASK(x, 'q', SCORE)`: the model
//! scores each row 0–1 against the query.
//!
//! With a semantic index it is a two-stage funnel. Stage one: one embed call
//! picks the candidates by cosine similarity, over-fetching a multiple of the
//! query's `LIMIT`. Stage two spends the model, scoring each candidate's best
//! chunks. Cost is a function of the `LIMIT`, not of the table's size, and the
//! result is an approximate top-k: an indexed row the index didn't surface as
//! a candidate cannot win, however well it would have scored.
//!
//! An index changes cost, never which rows can rank: a row the index has
//! never seen (or every row, with no index) is scored on its full text, one
//! call per distinct text. The two inputs are asked under different prompts
//! and prompt versions, as `MEANS` keeps chunked and full-text verdicts
//! apart.
//!
//! The operator only *materializes* the score column; the `Sort` and `Limit`
//! above it do the ordering and truncation.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, Float64Array, StringArray};
use datafusion::arrow::compute::{cast, filter_record_batch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Statistics;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{Distribution, EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::TryStreamExt;

use crate::cache::{CacheKey, CachedValue, SemanticCache};
use crate::index::{ChunkHit, SearchParams, SemanticIndex, doc_hash};
use crate::model::{CompletionRequest, ModelId, ModelProvider};
use crate::physical::extract::parse_json_object;
use crate::physical::verify::CHUNK_SEPARATOR;
use crate::semir::{AskType, answer_key, normalize_instruction};

/// Version of the synthesized rerank prompt over a document's best chunks.
/// Participates in cache keys: bump it and every cached score is honestly
/// invalidated.
pub const RELEVANCE_PROMPT_VERSION: &str = "relevance-v1";

/// Version of the prompt over a document's full text, for rows no index
/// covers. Its own constant: a one-chunk document sends the same input
/// either way, and sharing a key would serve one prompt's score for the
/// other's question.
pub const RELEVANCE_FULL_PROMPT_VERSION: &str = "relevance-v1-full";

/// The instruction the full-text scoring model sees.
pub fn synthesize_relevance_prompt_full(query: &str) -> String {
    format!(
        "You are scoring how well a document matches a search query.\n\n\
         Query: {query}\n\n\
         Reply with a JSON object {{\"score\": n}} where n is between 0 and 1: \
         1 means the document is exactly what the query asks for, 0 means it \
         is unrelated."
    )
}

/// The instruction the rerank model sees over a document's best chunks.
/// Users never write this.
pub fn synthesize_relevance_prompt(query: &str) -> String {
    format!(
        "You are scoring how well a document matches a search query. You see \
         the document's most relevant excerpts, separated by `---`; the \
         excerpts may be partial.\n\n\
         Query: {query}\n\n\
         Reply with a JSON object {{\"score\": n}} where n is between 0 and 1: \
         1 means the document is exactly what the query asks for, 0 means it \
         is unrelated."
    )
}

/// The JSON schema the score is decoded under — the same constrained-decoding
/// path `ONEOF` fields use.
fn score_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "score": { "type": "number" } },
        "required": ["score"],
        "additionalProperties": false
    })
}

/// Ranks input rows against a natural-language query, extending each row with
/// a relevance score.
///
/// Blocking and single-partition: an ordering is a global fact, so the
/// operator sees the whole (already candidate-pruned) input before it emits.
///
/// Row semantics ("rows fail, queries don't"): a row whose model call errors
/// or answers unparseably gets a NULL score and is counted in `rows_failed` —
/// NULLs sort last under `DESC`, so a failure costs a row its ranking, not the
/// query its results.
#[derive(Debug)]
pub struct SemRankExec {
    input: Arc<dyn ExecutionPlan>,
    /// Evaluates to the text being ranked, against input batches.
    text: Arc<dyn PhysicalExpr>,
    query: String,
    /// How many candidate documents the index stage keeps.
    candidates: usize,
    /// Name of the appended score column.
    score_column: String,
    /// Picks the candidates and their chunks; without one every row is
    /// scored on its full text.
    index: Option<Arc<dyn SemanticIndex>>,
    /// Chunks per document handed to the rerank model as evidence.
    chunks_per_doc: usize,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl SemRankExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        query: impl Into<String>,
        candidates: usize,
        score_column: impl Into<String>,
        index: Option<Arc<dyn SemanticIndex>>,
        chunks_per_doc: usize,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
    ) -> Result<Self> {
        let score_column = score_column.into();
        let mut fields = input.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(&score_column, DataType::Float64, true)));
        let output_schema = Arc::new(Schema::new(fields));
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            // One partition: the candidate set and the ordering it feeds are
            // global, not per-partition.
            Partitioning::UnknownPartitioning(1),
            // Nothing is emitted until every candidate has been scored.
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            text,
            // Normalized where it enters, so every key built from it agrees.
            query: normalize_instruction(&query.into()),
            candidates,
            score_column,
            index,
            chunks_per_doc,
            model,
            cache,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Search knobs for the candidate stage: no floor (the over-fetch count
    /// *is* the cut), and enough chunks to fill every candidate's evidence.
    fn search_params(&self) -> SearchParams {
        SearchParams {
            fetch_k: self.candidates.saturating_mul(self.chunks_per_doc.max(1)),
            score_floor: f32::NEG_INFINITY,
            chunks_per_doc: self.chunks_per_doc,
        }
    }
}

impl DisplayAs for SemRankExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.index {
            Some(index) => write!(
                f,
                "SemRankExec: RELEVANCE TO '{}' embed_model={} model={} \
                 top-{} chunks   ≤{} model calls, plus one per unindexed row",
                self.query,
                index.embed_model_id(),
                self.model.id(),
                self.chunks_per_doc,
                self.candidates,
            ),
            None => write!(
                f,
                "SemRankExec: RELEVANCE TO '{}' model={} full text, no index   \
                 one model call per distinct row",
                self.query,
                self.model.id(),
            ),
        }
    }
}

impl ExecutionPlan for SemRankExec {
    fn name(&self) -> &str {
        "SemRankExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// The candidate set spans the whole input, so DataFusion must coalesce
    /// before this operator rather than after it.
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
            self.query.clone(),
            self.candidates,
            self.score_column.clone(),
            self.index.clone(),
            self.chunks_per_doc,
            Arc::clone(&self.model),
            Arc::clone(&self.cache),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(0, context)?;
        let ranker = Ranker {
            text: Arc::clone(&self.text),
            query: self.query.clone(),
            prompt: synthesize_relevance_prompt(&self.query),
            full_prompt: synthesize_relevance_prompt_full(&self.query),
            candidates: self.candidates,
            params: self.search_params(),
            index: self.index.clone(),
            model_id: self.model.id(),
            model: Arc::clone(&self.model),
            cache: Arc::clone(&self.cache),
            output_schema: Arc::clone(&self.output_schema),
            candidate_rows: MetricBuilder::new(&self.metrics).counter("candidate_rows", partition),
            rows_pruned: MetricBuilder::new(&self.metrics).counter("rows_pruned", partition),
            unindexed_rows: MetricBuilder::new(&self.metrics).counter("unindexed_rows", partition),
            model_calls: MetricBuilder::new(&self.metrics).counter("model_calls", partition),
            model_calls_merged: MetricBuilder::new(&self.metrics)
                .counter("model_calls_merged", partition),
            cache_hits: MetricBuilder::new(&self.metrics).counter("cache_hits", partition),
            rows_failed: MetricBuilder::new(&self.metrics).counter("rows_failed", partition),
        };
        // One future produces every output batch: nothing can be emitted
        // before the whole candidate set is scored.
        let stream = futures::stream::once(async move { ranker.run(input).await })
            .map_ok(|batches| futures::stream::iter(batches.into_iter().map(Ok)))
            .try_flatten();
        let output = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        ));
        Ok(crate::physical::trace::trace_stage(
            "SemRankExec",
            partition,
            output,
        ))
    }

    /// With an index, about `candidates` distinct documents survive, plus
    /// any the index never saw; without one, every row. Inexact because rows
    /// sharing a text value all survive together.
    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let input_rows = self.input.partition_statistics(partition)?.num_rows;
        let mut statistics = Statistics::new_unknown(&self.output_schema);
        statistics.num_rows = match (input_rows, &self.index) {
            (Precision::Exact(rows) | Precision::Inexact(rows), Some(_)) => {
                Precision::Inexact(rows.min(self.candidates))
            }
            (rows, None) => rows.to_inexact(),
            (Precision::Absent, Some(_)) => Precision::Inexact(self.candidates),
        };
        Ok(Arc::new(statistics))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// What one index search learned: which documents can win, and which the
/// index has ever seen at all.
struct Candidates {
    /// Candidate documents → their best chunks, best first.
    evidence: HashMap<u64, Vec<String>>,
    /// Every document the index knows — how a losing document is told from an
    /// unindexed one.
    indexed: HashSet<u64>,
}

/// Everything the single output stream needs.
struct Ranker {
    text: Arc<dyn PhysicalExpr>,
    query: String,
    /// Over a document's best chunks.
    prompt: String,
    /// Over a document's full text.
    full_prompt: String,
    candidates: usize,
    params: SearchParams,
    index: Option<Arc<dyn SemanticIndex>>,
    model_id: ModelId,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    output_schema: SchemaRef,
    candidate_rows: Count,
    rows_pruned: Count,
    unindexed_rows: Count,
    model_calls: Count,
    /// Of `model_calls`, those the scheduler answered from an identical
    /// request already in flight: no call of their own.
    model_calls_merged: Count,
    cache_hits: Count,
    rows_failed: Count,
}

impl Ranker {
    async fn run(self, mut input: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
        let candidates = self.candidates().await?;

        // Keep candidates and rows no index covers. With an index this is
        // bounded by the over-fetch plus the index's staleness; without one
        // it is the whole input, as any sort would buffer.
        let mut kept = Vec::new();
        while let Some(batch) = input.try_next().await? {
            let batch = self.keep_candidates(&candidates, batch)?;
            if batch.num_rows() > 0 {
                kept.push(batch);
            }
        }

        let scores = self.score(&candidates.evidence, &kept).await?;
        kept.into_iter()
            .map(|batch| self.with_scores(&scores, batch))
            .collect()
    }

    /// Stage one: one embed call and one vector scan pick the candidates and
    /// the excerpts the model will read, plus one membership scan so a stale
    /// index can be told from a losing document.
    async fn candidates(&self) -> Result<Candidates> {
        let Some(index) = &self.index else {
            return Ok(Candidates {
                evidence: HashMap::new(),
                indexed: HashSet::new(),
            });
        };
        let hits = index.search(&self.query, &self.params).await?;
        Ok(Candidates {
            evidence: bucket_candidates(hits, self.candidates, self.params.chunks_per_doc),
            indexed: index.indexed_doc_hashes().await?,
        })
    }

    /// Prune a batch to the rows that will be scored: the index's candidates,
    /// and every row it has never seen, which is scored on its full text
    /// like the `MEANS` pre-filter passes such rows to full-text verify. An
    /// indexed row that isn't a candidate is pruned. Unindexed rows are
    /// counted in `unindexed_rows` (with an index), so index staleness shows
    /// up in the funnel as the calls it costs.
    fn keep_candidates(&self, candidates: &Candidates, batch: RecordBatch) -> Result<RecordBatch> {
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
            // NULL text has nothing to rank — free to drop here.
            if !texts.is_valid(row) {
                self.rows_pruned.add(1);
                continue;
            }
            let hash = doc_hash(texts.value(row));
            if candidates.evidence.contains_key(&hash) {
                self.candidate_rows.add(1);
                *keep_row = true;
            } else if candidates.indexed.contains(&hash) {
                self.rows_pruned.add(1);
            } else {
                if self.index.is_some() {
                    self.unindexed_rows.add(1);
                }
                *keep_row = true;
            }
        }
        Ok(filter_record_batch(&batch, &BooleanArray::from(keep))?)
    }

    /// Stage two: one store read, then one batched model call, score every
    /// distinct kept document — a candidate on its chunks, any other on its
    /// full text. Scores key on the document, so rows sharing a text value
    /// share one call.
    async fn score(
        &self,
        evidence: &HashMap<u64, Vec<String>>,
        kept: &[RecordBatch],
    ) -> Result<HashMap<u64, f64>> {
        // (document, input sent, whether it is chunks), once per document.
        let mut asks: Vec<(u64, String, bool)> = Vec::new();
        let mut seen = HashSet::new();
        for batch in kept {
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
                let hash = doc_hash(text);
                if !seen.insert(hash) {
                    continue;
                }
                asks.push(match evidence.get(&hash) {
                    Some(chunks) => (hash, chunks.join(CHUNK_SEPARATOR), true),
                    None => (hash, text.to_owned(), false),
                });
            }
        }

        let keys: Vec<CacheKey> = asks
            .iter()
            .map(|(_, input, chunked)| self.cache_key(input, *chunked))
            .collect();
        // The store read can block on disk: keep it off the async workers.
        let cache = Arc::clone(&self.cache);
        let expected = keys.len();
        let stored = tokio::task::spawn_blocking(move || cache.get_many(&keys))
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        // A store that breaks one answer per key can't be lined up with the
        // asks: ask everything rather than misread a score.
        let stored = if stored.len() == expected {
            stored
        } else {
            tracing::warn!(
                "store answered {} of {expected} relevance keys; asking the model",
                stored.len()
            );
            vec![None; expected]
        };

        let mut scores = HashMap::new();
        let mut requests = Vec::new();
        let mut pending = Vec::new();
        for ((hash, input, chunked), stored) in asks.into_iter().zip(stored) {
            if let Some(CachedValue::Value(score)) = stored {
                self.cache_hits.add(1);
                if let Ok(score) = score.parse::<f64>() {
                    scores.insert(hash, score);
                }
                continue;
            }
            requests.push(CompletionRequest {
                system: if chunked {
                    self.prompt.clone()
                } else {
                    self.full_prompt.clone()
                },
                input: input.clone(),
                max_tokens: 32,
                schema: Some(score_schema()),
                logprobs: false,
            });
            pending.push((hash, input, chunked));
        }
        // Counted as sent, so progress shows calls while they run; merges
        // are known only once they answer.
        self.model_calls.add(requests.len());
        let completions = self.model.complete(requests).await;
        self.model_calls_merged
            .add(crate::physical::verify::merged_calls(&completions));
        debug_assert_eq!(completions.len(), pending.len());
        for ((hash, input, chunked), completion) in pending.iter().zip(&completions) {
            match completion.as_ref().map(|c| parse_score(&c.text)) {
                Ok(Some(score)) => {
                    scores.insert(*hash, score);
                    // Only successful scores are cached: a transient model
                    // failure must not permanently sink a row's ranking.
                    self.cache.put(
                        self.cache_key(input, *chunked),
                        CachedValue::Value(score.to_string()),
                    );
                }
                // Row-level failure: NULL score, which sorts last under DESC.
                Ok(None) | Err(_) => self.rows_failed.add(1),
            }
        }
        Ok(scores)
    }

    /// Append the score column to a batch.
    fn with_scores(&self, scores: &HashMap<u64, f64>, batch: RecordBatch) -> Result<RecordBatch> {
        let texts = self.evaluate_texts(&batch)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("array was just cast to Utf8");
        let column: Float64Array = (0..batch.num_rows())
            .map(|row| {
                texts
                    .is_valid(row)
                    .then(|| scores.get(&doc_hash(texts.value(row))).copied())
                    .flatten()
            })
            .collect();

        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(column));
        Ok(RecordBatch::try_new(
            Arc::clone(&self.output_schema),
            columns,
        )?)
    }

    fn evaluate_texts(
        &self,
        batch: &RecordBatch,
    ) -> Result<Arc<dyn datafusion::arrow::array::Array>> {
        let texts = self.text.evaluate(batch)?.into_array(batch.num_rows())?;
        Ok(cast(&texts, &DataType::Utf8)?)
    }

    fn cache_key(&self, input: &str, chunked: bool) -> CacheKey {
        relevance_cache_key(&self.query, input, &self.model_id, chunked)
    }
}

/// The answer key of `ASK(input, query, SCORE)`: same query + sent input +
/// model + prompt scheme → same score, across every query that ever asks
/// again. The layout relevance scores have always been stored under.
pub(crate) fn relevance_cache_key(
    query: &str,
    input: &str,
    model_id: &ModelId,
    chunked: bool,
) -> CacheKey {
    let version = if chunked {
        RELEVANCE_PROMPT_VERSION
    } else {
        RELEVANCE_FULL_PROMPT_VERSION
    };
    answer_key(query, AskType::Score, input, model_id, version)
}

/// Bucket search hits into the top `candidates` documents' evidence, best
/// first, at most `chunks_per_doc` chunks each.
///
/// Hits arrive best-first, so the first `candidates` distinct documents *are*
/// the top ones by cosine similarity.
fn bucket_candidates(
    hits: Vec<ChunkHit>,
    candidates: usize,
    chunks_per_doc: usize,
) -> HashMap<u64, Vec<String>> {
    let mut chunks: HashMap<u64, Vec<String>> = HashMap::new();
    for hit in hits {
        if !chunks.contains_key(&hit.doc_hash) && chunks.len() >= candidates {
            continue;
        }
        let doc_chunks = chunks.entry(hit.doc_hash).or_default();
        if doc_chunks.len() < chunks_per_doc {
            doc_chunks.push(hit.text);
        }
    }
    chunks
}

/// `None` means the model didn't give a usable score. Out-of-range answers are
/// clamped rather than dropped — a model that says `1.2` still means "very
/// relevant".
fn parse_score(text: &str) -> Option<f64> {
    let score = parse_json_object(text)?.get("score")?.as_f64()?;
    if score.is_nan() {
        return None;
    }
    Some(score.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(doc: u64, chunk: usize, score: f32, text: &str) -> ChunkHit {
        ChunkHit {
            doc_hash: doc,
            chunk_index: chunk,
            score,
            text: text.to_owned(),
        }
    }

    #[test]
    fn buckets_keep_the_best_candidates_and_their_chunks() {
        let hits = vec![
            hit(1, 0, 0.9, "a1"),
            hit(2, 0, 0.8, "b1"),
            hit(1, 1, 0.7, "a2"),
            hit(3, 0, 0.6, "c1"),
        ];
        let buckets = bucket_candidates(hits, 2, 3);
        assert_eq!(buckets.len(), 2, "third document is over the candidate cap");
        assert_eq!(buckets[&1], vec!["a1", "a2"]);
        assert_eq!(buckets[&2], vec!["b1"]);
        assert!(!buckets.contains_key(&3));
    }

    #[test]
    fn buckets_cap_chunks_per_document() {
        let hits = vec![
            hit(1, 0, 0.9, "a1"),
            hit(1, 1, 0.8, "a2"),
            hit(1, 2, 0.7, "a3"),
        ];
        assert_eq!(bucket_candidates(hits, 4, 2)[&1], vec!["a1", "a2"]);
    }

    #[test]
    fn parses_a_score_object() {
        assert_eq!(parse_score(r#"{"score": 0.75}"#), Some(0.75));
        assert_eq!(parse_score("```json\n{\"score\": 1}\n```"), Some(1.0));
    }

    #[test]
    fn clamps_out_of_range_scores() {
        assert_eq!(parse_score(r#"{"score": 1.4}"#), Some(1.0));
        assert_eq!(parse_score(r#"{"score": -2}"#), Some(0.0));
    }

    #[test]
    fn unusable_answers_are_none() {
        assert_eq!(parse_score("very relevant"), None);
        assert_eq!(parse_score(r#"{"relevance": 0.5}"#), None);
        assert_eq!(parse_score(r#"{"score": "high"}"#), None);
    }
}
