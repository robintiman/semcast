//! The typed-extraction physical operator — spends one model call per input
//! row to fill the extracted columns, then caches per generation unit so a
//! re-run (or a query sharing a field) is free.
//!
//! Modeled on [`VerifyExec`]: it reads the source text, batches one
//! `complete` call over the rows with pending fields, decodes and validates
//! the JSON, and passes the input columns through with the new field columns
//! appended.
//!
//! [`VerifyExec`]: crate::physical::verify::VerifyExec

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use tokio::sync::OnceCell;

use datafusion::arrow::array::{Array, ArrayRef, StringArray};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result};
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
use serde_json::Value;

use crate::cache::{CacheKey, CachedValue, SemanticCache};
use crate::index::{SearchParams, SemanticIndex, doc_hash};
use crate::logical::sem_extract::output_column_name;
use crate::model::{CompletionRequest, ModelId, ModelProvider};
use crate::physical::index_scan::bucket_chunks;
use crate::physical::verify::CHUNK_SEPARATOR;
use crate::types::{EXTRACT_PROMPT_VERSION, FieldSpec, FieldType, SemanticType, unit_hash};

/// Per-row `max_tokens` floor for an extraction request. Typed JSON is
/// compact, so a narrow extraction never approaches this.
const EXTRACT_MAX_TOKENS: usize = 2048;

/// Room allowed per field on top of the floor. A wide `TOGETHER` group emits
/// one JSON member per field, and list-valued fields quoting source text emit
/// many lines each — a flat ceiling truncates that mid-object, the truncated
/// JSON fails to decode, and the row is dropped as an error. Since the row
/// still costs a full model call, the failure is both silent and expensive, so
/// the ceiling scales with the number of fields the call actually fills.
const EXTRACT_TOKENS_PER_FIELD: usize = 384;

/// Absolute ceiling, so a pathological type cannot ask for an unbounded
/// generation.
const EXTRACT_MAX_TOKENS_CAP: usize = 16384;

/// The `max_tokens` for a call filling `fields` fields.
fn extract_max_tokens(fields: usize) -> usize {
    (fields * EXTRACT_TOKENS_PER_FIELD).clamp(EXTRACT_MAX_TOKENS, EXTRACT_MAX_TOKENS_CAP)
}

/// Per-document excerpts, keyed by [`doc_hash`] so they survive whatever
/// repartitioning DataFusion inserts under the operator.
type DocChunks = HashMap<u64, Vec<String>>;

/// Retrieved once per query and shared by every partition's stream.
type SharedChunks = Arc<OnceCell<Arc<DocChunks>>>;

/// Reduced-context extraction: feed the model the document's best excerpts
/// from the semantic index instead of its full text.
///
/// A full-text extraction pays for the whole document on every call, however
/// little of it bears on the fields being extracted. The index already knows
/// which chunks are near which text, and each field carries a natural-language
/// doc line — so the doc lines are the queries, and their top chunks are the
/// context the model actually needs.
///
/// This is a cost/recall trade, not a free win: a fact outside the retrieved
/// chunks is reported absent. How it lands depends on the corpus. Evidence
/// concentrated in one passage survives the reduction; evidence dispersed
/// through a long document does not. It is enabled by the presence of an
/// index on the source column, so `CREATE SEMANTIC INDEX` is the knob.
#[derive(Debug, Clone)]
pub struct ReducedContext {
    pub index: Arc<dyn SemanticIndex>,
    pub params: SearchParams,
}

/// Extends the input with one nullable column per extracted field, then fills
/// them with a model call per row (cached per generation unit).
#[derive(Debug)]
pub struct SemExtractExec {
    input: Arc<dyn ExecutionPlan>,
    /// Evaluates to the source text, against input batches.
    source: Arc<dyn PhysicalExpr>,
    /// The pruned extraction spec — exactly the fields this node produces.
    target: SemanticType,
    /// Disambiguates the output column names (`__sem_{id}_{field}`).
    id: usize,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    /// Present when a semantic index covers the source column: the model
    /// reads excerpts instead of full documents.
    context: Option<ReducedContext>,
    /// Retrieval runs once per query, not once per partition or per row —
    /// one search per field doc line covers the whole corpus.
    chunks: SharedChunks,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl SemExtractExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        source: Arc<dyn PhysicalExpr>,
        target: SemanticType,
        id: usize,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
    ) -> Result<Self> {
        Self::build(input, source, target, id, model, cache, None)
    }

    /// The reduced-context variant — see [`ReducedContext`].
    pub fn new_with_context(
        input: Arc<dyn ExecutionPlan>,
        source: Arc<dyn PhysicalExpr>,
        target: SemanticType,
        id: usize,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
        context: ReducedContext,
    ) -> Result<Self> {
        Self::build(input, source, target, id, model, cache, Some(context))
    }

    fn build(
        input: Arc<dyn ExecutionPlan>,
        source: Arc<dyn PhysicalExpr>,
        target: SemanticType,
        id: usize,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
        context: Option<ReducedContext>,
    ) -> Result<Self> {
        let output_schema = extend_schema(&input.schema(), &target, id)?;
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            source,
            target,
            id,
            model,
            cache,
            context,
            chunks: Arc::new(OnceCell::new()),
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

/// Input schema + one nullable column per extracted field.
fn extend_schema(input: &SchemaRef, target: &SemanticType, id: usize) -> Result<SchemaRef> {
    let mut fields: Vec<Field> = input.fields().iter().map(|f| f.as_ref().clone()).collect();
    for spec in &target.fields {
        let arrow = spec.ty.arrow_type().map_err(DataFusionError::from)?;
        fields.push(Field::new(output_column_name(id, &spec.name), arrow, true));
    }
    Ok(Arc::new(Schema::new(fields)))
}

impl DisplayAs for SemExtractExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SemExtractExec: {} [{} field(s)] model={}",
            self.target.name,
            self.target.fields.len(),
            self.model.id()
        )?;
        // Which context the plan will feed the model — the difference between
        // paying for whole documents and paying for excerpts.
        if let Some(context) = &self.context {
            write!(
                f,
                " reads top-{} chunks per doc",
                context.params.chunks_per_doc
            )?;
        }
        // The bill before you run: one call per surviving row, worst case.
        match self.input.partition_statistics(None).map(|s| s.num_rows) {
            Ok(Precision::Exact(rows)) => write!(f, "   ≤{rows} model calls"),
            Ok(Precision::Inexact(rows)) => write!(f, "   ~{rows} model calls"),
            _ => write!(f, "   model calls unknown"),
        }
    }
}

impl ExecutionPlan for SemExtractExec {
    fn name(&self) -> &str {
        "SemExtractExec"
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
            Arc::clone(&self.source),
            self.target.clone(),
            self.id,
            Arc::clone(&self.model),
            Arc::clone(&self.cache),
            self.context.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let extractor = Arc::new(Extractor {
            source: Arc::clone(&self.source),
            target: self.target.clone(),
            model_id: self.model.id(),
            model: Arc::clone(&self.model),
            cache: Arc::clone(&self.cache),
            output_schema: Arc::clone(&self.output_schema),
            context: self.context.clone(),
            chunks: Arc::clone(&self.chunks),
            model_calls: MetricBuilder::new(&self.metrics).counter("model_calls", partition),
            cache_hits: MetricBuilder::new(&self.metrics).counter("cache_hits", partition),
            rows_failed: MetricBuilder::new(&self.metrics).counter("rows_failed", partition),
            fields_failed: MetricBuilder::new(&self.metrics).counter("fields_failed", partition),
            reduced_rows: MetricBuilder::new(&self.metrics)
                .counter("reduced_context_rows", partition),
            full_text_rows: MetricBuilder::new(&self.metrics).counter("full_text_rows", partition),
            input_tokens: MetricBuilder::new(&self.metrics).counter("input_tokens", partition),
            output_tokens: MetricBuilder::new(&self.metrics).counter("output_tokens", partition),
        });
        let stream = input.and_then(move |batch| {
            let extractor = Arc::clone(&extractor);
            async move { extractor.extract_batch(batch).await }
        });
        let output = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        ));
        Ok(crate::physical::trace::trace_stage(
            "SemExtractExec",
            partition,
            output,
        ))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// Everything one partition's stream needs, bundled so the closure stays cheap
/// to clone.
struct Extractor {
    source: Arc<dyn PhysicalExpr>,
    target: SemanticType,
    model_id: ModelId,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    context: Option<ReducedContext>,
    /// Retrieved once per query, shared by every partition's stream.
    chunks: SharedChunks,
    output_schema: SchemaRef,
    model_calls: Count,
    cache_hits: Count,
    rows_failed: Count,
    fields_failed: Count,
    reduced_rows: Count,
    full_text_rows: Count,
    input_tokens: Count,
    output_tokens: Count,
}

/// A row that needs a model call, plus which units are still pending.
struct PendingRow {
    row: usize,
    /// What the model reads for this row — the document, or its excerpts under
    /// a [`ReducedContext`]. Cache keys hash this, so a reduced-context verdict
    /// is never served for a full-text one.
    text: String,
    pending_units: Vec<usize>,
}

impl Extractor {
    /// Retrieve the per-document excerpts, once per query.
    ///
    /// Each field's doc line is its own query — the fields are asking
    /// different questions of the document, and a single blended query would
    /// retrieve the passage that is vaguely near all of them rather than the
    /// passages that answer each. Hits are then unioned per document, so a
    /// document's excerpt set covers every field that found evidence in it.
    async fn retrieve_chunks(&self, context: &ReducedContext) -> crate::Result<Arc<DocChunks>> {
        let mut hits = Vec::new();
        for spec in &self.target.fields {
            hits.extend(context.index.search(&spec.doc, &context.params).await?);
        }
        // Best-scoring first, so the per-document cap keeps the strongest
        // evidence when more fields hit a document than it has room for.
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.dedup_by(|a, b| a.doc_hash == b.doc_hash && a.chunk_index == b.chunk_index);
        Ok(Arc::new(bucket_chunks(hits, context.params.chunks_per_doc)))
    }

    /// The text one row's model call should read: its excerpts when the index
    /// has them, the full document otherwise. An index gap degrades to full
    /// price rather than dropping the row.
    fn input_for(&self, chunks: Option<&DocChunks>, text: &str) -> String {
        let excerpts = chunks
            .and_then(|chunks| chunks.get(&doc_hash(text)))
            .filter(|excerpts| !excerpts.is_empty());
        match excerpts {
            Some(excerpts) => {
                self.reduced_rows.add(1);
                excerpts.join(CHUNK_SEPARATOR)
            }
            None => {
                self.full_text_rows.add(1);
                text.to_owned()
            }
        }
    }

    async fn extract_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(RecordBatch::new_empty(Arc::clone(&self.output_schema)));
        }
        let texts = self.source.evaluate(&batch)?.into_array(rows)?;
        let texts = cast(&texts, &DataType::Utf8)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("source cast to Utf8");

        // One column of decoded values per field, initialized to typed NULL.
        let mut values: HashMap<&str, Vec<ScalarValue>> = HashMap::new();
        for spec in &self.target.fields {
            let arrow = spec.ty.arrow_type().map_err(DataFusionError::from)?;
            values.insert(
                spec.name.as_str(),
                vec![ScalarValue::try_from(&arrow)?; rows],
            );
        }

        // Retrieval is per-query: the OnceCell means the first batch to get
        // here pays for it and every later batch and partition reads it.
        let chunks = match &self.context {
            Some(context) => Some(Arc::clone(
                self.chunks
                    .get_or_try_init(|| self.retrieve_chunks(context))
                    .await
                    .map_err(DataFusionError::from)?,
            )),
            None => None,
        };
        let chunks = chunks.as_deref();

        let units = self.target.generation_units();
        let mut requests = Vec::new();
        let mut pending: Vec<PendingRow> = Vec::new();

        for row in 0..rows {
            // NULL source never reaches the model — all fields stay NULL.
            if !texts.is_valid(row) {
                continue;
            }
            let text = texts.value(row);
            // Key the cache on what the model actually reads, not on the
            // document it came from: a full-text verdict and a reduced-context
            // verdict are different answers to different inputs, and keying
            // both on the document would let one be served for the other.
            let input = self.input_for(chunks, text);
            let mut pending_units = Vec::new();
            for (unit_idx, unit) in units.iter().enumerate() {
                match self.read_unit(unit, &input) {
                    Some(hits) => {
                        self.cache_hits.add(unit.len());
                        for (name, scalar) in hits {
                            values.get_mut(name).expect("field initialized")[row] = scalar;
                        }
                    }
                    None => pending_units.push(unit_idx),
                }
            }
            if pending_units.is_empty() {
                continue;
            }
            let pending_fields: Vec<&str> = pending_units
                .iter()
                .flat_map(|&i| units[i].iter().map(|s| s.name.as_str()))
                .collect();
            // The prompt has to match the input: telling a model it is reading
            // a document when it is reading excerpts invites it to report an
            // absent fact as present rather than null.
            let system = if self.context.is_some() {
                self.target.synthesize_prompt_chunked(&pending_fields)
            } else {
                self.target.synthesize_prompt(&pending_fields)
            }
            .map_err(DataFusionError::from)?;
            requests.push(CompletionRequest {
                system,
                input: input.clone(),
                max_tokens: extract_max_tokens(pending_fields.len()),
                schema: Some(
                    self.target
                        .json_schema(&pending_fields)
                        .map_err(DataFusionError::from)?,
                ),
            });
            pending.push(PendingRow {
                row,
                text: input,
                pending_units,
            });
        }
        self.model_calls.add(requests.len());

        let completions = self.model.complete(requests).await;
        debug_assert_eq!(completions.len(), pending.len());
        // Real spend, not an estimate from row counts: what a cost model has
        // to be calibrated against, and what makes a reduced context's saving
        // measurable rather than asserted.
        for completion in completions.iter().flatten() {
            self.input_tokens.add(completion.input_tokens);
            self.output_tokens.add(completion.output_tokens);
        }
        for (pending_row, completion) in pending.iter().zip(&completions) {
            let object = completion
                .as_ref()
                .ok()
                .and_then(|c| parse_json_object(&c.text));
            let Some(object) = object else {
                // Model error or unparseable response: all pending fields stay
                // NULL for this row.
                self.rows_failed.add(1);
                continue;
            };
            for &unit_idx in &pending_row.pending_units {
                self.apply_unit(&units[unit_idx], &object, pending_row, &mut values);
            }
        }

        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for spec in &self.target.fields {
            let column = values
                .remove(spec.name.as_str())
                .expect("field initialized");
            columns.push(ScalarValue::iter_to_array(column)?);
        }
        Ok(RecordBatch::try_new(
            Arc::clone(&self.output_schema),
            columns,
        )?)
    }

    /// Read a whole unit from the cache — all members must hit, or it is
    /// pending. Returns the decoded `(field, value)` pairs on a full hit.
    fn read_unit<'f>(
        &self,
        unit: &[&'f FieldSpec],
        text: &str,
    ) -> Option<Vec<(&'f str, ScalarValue)>> {
        let mut hits = Vec::with_capacity(unit.len());
        for spec in unit {
            match self.cache.get(&self.cache_key(unit, &spec.name, text)) {
                Some(CachedValue::Value(raw)) => {
                    let normalized: Value = serde_json::from_str(&raw).ok()?;
                    hits.push((
                        spec.name.as_str(),
                        scalar_from_normalized(&spec.ty, &normalized),
                    ));
                }
                _ => return None,
            }
        }
        Some(hits)
    }

    /// Decode and validate one unit's fields from the model's JSON object,
    /// writing the values and caching the unit only if every member validated.
    fn apply_unit(
        &self,
        unit: &[&FieldSpec],
        object: &serde_json::Map<String, Value>,
        pending_row: &PendingRow,
        values: &mut HashMap<&str, Vec<ScalarValue>>,
    ) {
        let mut decoded: Vec<(&FieldSpec, Value)> = Vec::with_capacity(unit.len());
        let mut all_valid = true;
        for spec in unit {
            match object
                .get(&spec.name)
                .and_then(|v| validate_field(&spec.ty, v))
            {
                Some(normalized) => decoded.push((spec, normalized)),
                None => {
                    all_valid = false;
                    self.fields_failed.add(1);
                }
            }
        }
        for (spec, normalized) in &decoded {
            values
                .get_mut(spec.name.as_str())
                .expect("field initialized")[pending_row.row] =
                scalar_from_normalized(&spec.ty, normalized);
        }
        // Cache the unit atomically: only when every member validated, so a
        // transient partial failure never permanently NULLs a sibling.
        if all_valid {
            for (spec, normalized) in &decoded {
                self.cache.put(
                    self.cache_key(unit, &spec.name, &pending_row.text),
                    CachedValue::Value(normalized.to_string()),
                );
            }
        }
    }

    /// Provenance: per-unit type hash + field + input + model + prompt scheme.
    fn cache_key(&self, unit: &[&FieldSpec], field: &str, text: &str) -> CacheKey {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        CacheKey {
            type_version: format!("{}@{:x}", self.target.name, unit_hash(unit)),
            field: field.to_owned(),
            input_hash: hasher.finish(),
            model_id: self.model_id.clone(),
            prompt_version: EXTRACT_PROMPT_VERSION.to_owned(),
        }
    }
}

/// Parse the model's response into a JSON object, tolerating a Markdown code
/// fence some local models wrap around JSON.
pub(crate) fn parse_json_object(text: &str) -> Option<serde_json::Map<String, Value>> {
    let trimmed = strip_code_fence(text.trim());
    serde_json::from_str::<Value>(trimmed)
        .ok()?
        .as_object()
        .cloned()
}

fn strip_code_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    // Drop the ```lang line and the trailing ```.
    let body = rest.split_once('\n').map_or(rest, |(_, body)| body);
    body.strip_suffix("```").unwrap_or(body).trim()
}

/// Validate a raw JSON value against a field type, returning the *normalized*
/// value (enum casing folded, bounds checked) or `None` if it doesn't fit.
fn validate_field(ty: &FieldType, value: &Value) -> Option<Value> {
    match ty {
        FieldType::Text => value.as_str().map(|s| Value::from(s.to_owned())),
        FieldType::Int => value.as_i64().map(Value::from),
        FieldType::Real => value.as_f64().map(Value::from),
        FieldType::RealBounded { min, max } => {
            let f = value.as_f64()?;
            (f >= min.0 && f <= max.0).then(|| Value::from(f))
        }
        FieldType::Bool => value.as_bool().map(Value::from),
        FieldType::OneOf(variants) | FieldType::Level(variants) => {
            let got = value.as_str()?;
            variants
                .iter()
                .find(|v| v.eq_ignore_ascii_case(got))
                .map(|v| Value::from(v.clone()))
        }
        FieldType::List(inner) => {
            let array = value.as_array()?;
            let normalized: Vec<Value> = array
                .iter()
                .filter_map(|v| validate_field(inner, v))
                .collect();
            Some(Value::Array(normalized))
        }
        FieldType::Nested(_) => None,
    }
}

/// Convert an already-normalized JSON value into the field's Arrow scalar.
fn scalar_from_normalized(ty: &FieldType, value: &Value) -> ScalarValue {
    match ty {
        FieldType::Text | FieldType::OneOf(_) | FieldType::Level(_) => {
            ScalarValue::Utf8(value.as_str().map(str::to_owned))
        }
        FieldType::Int => ScalarValue::Int64(value.as_i64()),
        FieldType::Real | FieldType::RealBounded { .. } => ScalarValue::Float64(value.as_f64()),
        FieldType::Bool => ScalarValue::Boolean(value.as_bool()),
        FieldType::List(inner) => match value.as_array() {
            Some(items) => {
                let inner_arrow = inner.arrow_type().unwrap_or(DataType::Utf8);
                let scalars: Vec<ScalarValue> = items
                    .iter()
                    .map(|v| scalar_from_normalized(inner, v))
                    .collect();
                ScalarValue::List(ScalarValue::new_list(&scalars, &inner_arrow, true))
            }
            None => ScalarValue::Null,
        },
        FieldType::Nested(_) => ScalarValue::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_tokens_scales_with_field_count() {
        // A narrow extraction keeps the historical floor.
        assert_eq!(extract_max_tokens(1), EXTRACT_MAX_TOKENS);
        assert_eq!(extract_max_tokens(5), EXTRACT_MAX_TOKENS);

        // A wide one gets room to finish the object. The 41-field CUAD type
        // is the case that motivated this: at a flat 2048 the JSON truncated
        // mid-object and every row was dropped as a decode error.
        assert!(extract_max_tokens(41) > EXTRACT_MAX_TOKENS);
        assert_eq!(extract_max_tokens(41), 41 * EXTRACT_TOKENS_PER_FIELD);

        // But never unbounded.
        assert_eq!(extract_max_tokens(10_000), EXTRACT_MAX_TOKENS_CAP);
    }
}
