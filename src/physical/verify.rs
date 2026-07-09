//! The verify stage — the physical operator that spends model calls
//! (roadmap step 1: verify-only execution).

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, StringArray};
use datafusion::arrow::compute::{cast, filter_record_batch};
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

/// Version of the synthesized `MEANS` verify prompt. Participates in cache
/// keys: bump it and every cached verdict is honestly invalidated.
pub const MEANS_PROMPT_VERSION: &str = "means-v2";

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
    /// How many chunks the evidence holds per document — display only.
    chunks_per_doc: usize,
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
            condition: condition.into(),
            model,
            cache,
            evidence,
            chunks_per_doc,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
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
        Ok(Arc::new(Self::build(
            Arc::clone(&children[0]),
            Arc::clone(&self.text),
            self.condition.clone(),
            Arc::clone(&self.model),
            Arc::clone(&self.cache),
            self.evidence.clone(),
            self.chunks_per_doc,
        )))
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
            model_calls: MetricBuilder::new(&self.metrics).counter("model_calls", partition),
            cache_hits: MetricBuilder::new(&self.metrics).counter("cache_hits", partition),
            rows_dropped: MetricBuilder::new(&self.metrics).counter("rows_dropped", partition),
        });
        let stream = input.and_then(move |batch| {
            let verifier = Arc::clone(&verifier);
            async move { verifier.verify_batch(batch).await }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.input.schema(),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// Everything one partition's stream needs, bundled so the closure stays
/// cheap to clone.
struct Verifier {
    text: Arc<dyn PhysicalExpr>,
    condition: String,
    prompt: String,
    chunked_prompt: String,
    model_id: ModelId,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    evidence: Option<Arc<ChunkEvidence>>,
    model_calls: Count,
    cache_hits: Count,
    rows_dropped: Count,
}

impl Verifier {
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
        let mut pending: Vec<(usize, String)> = Vec::new();
        for (row, keep_row) in keep.iter_mut().enumerate() {
            if !texts.is_valid(row) {
                continue;
            }
            let text = texts.value(row);
            // With index evidence, the model reads the document's best
            // chunks; without (no index, or an unindexed passthrough row),
            // the full text. The cache keys the input actually sent, so the
            // two kinds of verdict never collide.
            let (input, prompt) = match self
                .evidence
                .as_deref()
                .and_then(|evidence| evidence.chunks_for(doc_hash(text)))
            {
                Some(chunks) => (chunks.join(CHUNK_SEPARATOR), &self.chunked_prompt),
                None => (text.to_owned(), &self.prompt),
            };
            if let Some(CachedValue::Value(verdict)) = self.cache.get(&self.cache_key(&input)) {
                self.cache_hits.add(1);
                *keep_row = verdict == "yes";
                continue;
            }
            requests.push(CompletionRequest {
                system: prompt.clone(),
                input: input.clone(),
                max_tokens: 8,
                schema: None,
            });
            pending.push((row, input));
        }
        self.model_calls.add(requests.len());

        let verdicts = self.model.complete(requests).await;
        debug_assert_eq!(verdicts.len(), pending.len());
        for ((row, input), verdict) in pending.iter().zip(&verdicts) {
            match verdict.as_ref().map(|c| parse_verdict(&c.text)) {
                Ok(Some(matched)) => {
                    keep[*row] = matched;
                    // Only successful verdicts are cached: a transient model
                    // failure must not permanently exclude a row.
                    self.cache.put(
                        self.cache_key(input),
                        CachedValue::Value(if matched { "yes" } else { "no" }.to_owned()),
                    );
                }
                // Row-level failure: exclude and count, don't fail the query.
                Ok(None) | Err(_) => self.rows_dropped.add(1),
            }
        }
        Ok(filter_record_batch(&batch, &BooleanArray::from(keep))?)
    }

    fn cache_key(&self, input: &str) -> CacheKey {
        means_cache_key(&self.condition, input, &self.model_id)
    }
}

/// Full provenance: same condition + sent input + model + prompt scheme
/// → same verdict, across every query that ever asks again. Shared with
/// `WITH RECALL` calibration, whose ground-truth labels are exactly
/// full-text verdicts — the two stages pay for each other's cache.
pub(crate) fn means_cache_key(condition: &str, input: &str, model_id: &ModelId) -> CacheKey {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    CacheKey {
        type_version: condition.to_owned(),
        field: "means".to_owned(),
        input_hash: hasher.finish(),
        model_id: model_id.clone(),
        prompt_version: MEANS_PROMPT_VERSION.to_owned(),
    }
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
