//! The rerank stage — semantic ranking as a two-stage funnel.
//!
//! Stage one is the semantic index: one embed call picks the candidates by
//! cosine similarity, over-fetching a multiple of the query's `LIMIT`. Stage
//! two spends the model, scoring each candidate's best chunks on a 0–1 scale.
//! Cost is a function of the `LIMIT`, not of the table's size.
//!
//! The operator only *materializes* the score column; the `Sort` and `Limit`
//! above it do the ordering and truncation. That makes the result an
//! approximate top-k: a row the index never surfaced as a candidate cannot
//! win, however well it would have scored.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, Float64Array, StringArray};
use datafusion::arrow::compute::{cast, filter_record_batch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Statistics;
use datafusion::common::stats::Precision;
use datafusion::error::Result;
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

/// Version of the synthesized rerank prompt. Participates in cache keys: bump
/// it and every cached score is honestly invalidated.
pub const RELEVANCE_PROMPT_VERSION: &str = "relevance-v1";

/// The instruction the rerank model sees. Users never write this.
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
    index: Arc<dyn SemanticIndex>,
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
        index: Arc<dyn SemanticIndex>,
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
            query: query.into(),
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
        write!(
            f,
            "SemRankExec: RELEVANCE TO '{}' embed_model={} model={} \
             top-{} chunks   ≤{} model calls",
            self.query,
            self.index.embed_model_id(),
            self.model.id(),
            self.chunks_per_doc,
            self.candidates,
        )
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
            Arc::clone(&self.index),
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
            candidates: self.candidates,
            params: self.search_params(),
            index: Arc::clone(&self.index),
            model_id: self.model.id(),
            model: Arc::clone(&self.model),
            cache: Arc::clone(&self.cache),
            output_schema: Arc::clone(&self.output_schema),
            candidate_rows: MetricBuilder::new(&self.metrics).counter("candidate_rows", partition),
            rows_pruned: MetricBuilder::new(&self.metrics).counter("rows_pruned", partition),
            unindexed_rows: MetricBuilder::new(&self.metrics).counter("unindexed_rows", partition),
            model_calls: MetricBuilder::new(&self.metrics).counter("model_calls", partition),
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

    /// At most `candidates` distinct documents survive the index stage.
    /// Inexact because rows sharing a text value all survive together.
    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let input_rows = self.input.partition_statistics(partition)?.num_rows;
        let mut statistics = Statistics::new_unknown(&self.output_schema);
        statistics.num_rows = match input_rows {
            Precision::Exact(rows) | Precision::Inexact(rows) => {
                Precision::Inexact(rows.min(self.candidates))
            }
            Precision::Absent => Precision::Inexact(self.candidates),
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
    prompt: String,
    candidates: usize,
    params: SearchParams,
    index: Arc<dyn SemanticIndex>,
    model_id: ModelId,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    output_schema: SchemaRef,
    candidate_rows: Count,
    rows_pruned: Count,
    unindexed_rows: Count,
    model_calls: Count,
    cache_hits: Count,
    rows_failed: Count,
}

impl Ranker {
    async fn run(self, mut input: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
        let candidates = self.candidates().await?;

        // Keep only candidate rows. Bounded by the over-fetch, so buffering
        // the survivors is safe even over a large table.
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
        let hits = self.index.search(&self.query, &self.params).await?;
        Ok(Candidates {
            evidence: bucket_candidates(hits, self.candidates, self.params.chunks_per_doc),
            indexed: self.index.indexed_doc_hashes().await?,
        })
    }

    /// Prune a batch to its candidate rows.
    ///
    /// Three-way, and deliberately unlike the `MEANS` pre-filter, which passes
    /// unindexed rows through to full-text verification. A rank has nothing to
    /// pass them through *to*: an unscored row has no place in an ordering. So
    /// they are dropped — but counted separately in `unindexed_rows`, so index
    /// staleness shows up in the funnel instead of silently shrinking results.
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
                self.unindexed_rows.add(1);
            }
        }
        Ok(filter_record_batch(&batch, &BooleanArray::from(keep))?)
    }

    /// Stage two: one batched model call scores every distinct candidate
    /// document. Scores key on the document, so rows sharing a text value
    /// share one call.
    async fn score(
        &self,
        evidence: &HashMap<u64, Vec<String>>,
        kept: &[RecordBatch],
    ) -> Result<HashMap<u64, f64>> {
        let mut scores = HashMap::new();
        let mut seen = HashSet::new();
        let mut requests = Vec::new();
        let mut pending = Vec::new();

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
                let hash = doc_hash(texts.value(row));
                if !seen.insert(hash) {
                    continue;
                }
                let Some(chunks) = evidence.get(&hash) else {
                    // `keep_candidates` already dropped every non-candidate.
                    continue;
                };
                let input = chunks.join(CHUNK_SEPARATOR);
                if let Some(CachedValue::Value(score)) = self.cache.get(&self.cache_key(&input)) {
                    self.cache_hits.add(1);
                    if let Ok(score) = score.parse::<f64>() {
                        scores.insert(hash, score);
                    }
                    continue;
                }
                requests.push(CompletionRequest {
                    system: self.prompt.clone(),
                    input: input.clone(),
                    max_tokens: 32,
                    schema: Some(score_schema()),
                });
                pending.push((hash, input));
            }
        }
        self.model_calls.add(requests.len());

        let completions = self.model.complete(requests).await;
        debug_assert_eq!(completions.len(), pending.len());
        for ((hash, input), completion) in pending.iter().zip(&completions) {
            match completion.as_ref().map(|c| parse_score(&c.text)) {
                Ok(Some(score)) => {
                    scores.insert(*hash, score);
                    // Only successful scores are cached: a transient model
                    // failure must not permanently sink a row's ranking.
                    self.cache
                        .put(self.cache_key(input), CachedValue::Value(score.to_string()));
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

    fn cache_key(&self, input: &str) -> CacheKey {
        relevance_cache_key(&self.query, input, &self.model_id)
    }
}

/// Full provenance: same query + sent excerpts + model + prompt scheme → same
/// score, across every query that ever asks again.
pub(crate) fn relevance_cache_key(query: &str, input: &str, model_id: &ModelId) -> CacheKey {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    CacheKey {
        type_version: query.to_owned(),
        field: "relevance".to_owned(),
        input_hash: hasher.finish(),
        model_id: model_id.clone(),
        prompt_version: RELEVANCE_PROMPT_VERSION.to_owned(),
    }
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
