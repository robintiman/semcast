//! The dedupe stage — collapsing near-duplicate rows.
//!
//! The cheapest semantic operator semcast has: **no model calls at all**.
//! Document vectors say how alike two documents are, and "same thing" is a
//! threshold on that, not a judgement a model needs to make. With a semantic
//! index, the vectors it paid for are read back; without one, the column is
//! vectored the same way ([`DocVectors`]).
//!
//! Blocking, like clustering and for the same reason: whether a row is a
//! duplicate is a fact about the rows around it. It materializes the group
//! key and nothing else — `DISTINCT ON` above decides which row of a group
//! survives.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, StringArray, StringBuilder};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Statistics;
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

use crate::index::dedupe::greedy_groups;
use crate::index::doc_hash;
use crate::model::Embedding;
use crate::physical::doc_vectors::{DocVectors, VectorCounts};

/// Groups near-duplicate input rows, appending the key of each row's group.
///
/// Row semantics ("rows fail, queries don't"): a document with no vector (an
/// index gap whose embedding failed, an empty text) is keyed by its own
/// identity, so it can only ever collapse with a byte-identical twin: that
/// degrades to exact-match dedupe — duplicates stay in, data never goes out.
///
/// NULL text keeps a NULL key, which SQL groups together exactly as a plain
/// `DISTINCT ON` would.
#[derive(Debug)]
pub struct SemDistinctExec {
    input: Arc<dyn ExecutionPlan>,
    /// Evaluates to the text being deduplicated, against input batches.
    text: Arc<dyn PhysicalExpr>,
    similarity: f32,
    /// Name of the appended key column.
    key_column: String,
    vectors: DocVectors,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl SemDistinctExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        similarity: f32,
        key_column: impl Into<String>,
        vectors: DocVectors,
    ) -> Result<Self> {
        let key_column = key_column.into();
        let mut fields = input.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(&key_column, DataType::Utf8, true)));
        let output_schema = Arc::new(Schema::new(fields));
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            // One partition: "is this a duplicate" spans the whole relation.
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            text,
            similarity,
            key_column,
            vectors,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for SemDistinctExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SemDistinctExec: SIMILARITY ≥ {:.2} embed_model={}   0 model calls",
            self.similarity,
            self.vectors.embed_model_id(),
        )?;
        if let DocVectors::Embed { .. } = self.vectors {
            write!(f, " no index: embeds the column")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for SemDistinctExec {
    fn name(&self) -> &str {
        "SemDistinctExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Duplicate detection spans the whole input, so DataFusion must coalesce
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
            self.similarity,
            self.key_column.clone(),
            self.vectors.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(0, context)?;
        let deduper = Deduper {
            text: Arc::clone(&self.text),
            similarity: self.similarity,
            vectors: self.vectors.clone(),
            output_schema: Arc::clone(&self.output_schema),
            groups: MetricBuilder::new(&self.metrics).counter("groups", partition),
            duplicate_rows: MetricBuilder::new(&self.metrics).counter("duplicate_rows", partition),
            counts: VectorCounts::new(&self.metrics, partition),
        };
        // One future produces every output batch: no row can be called a
        // duplicate before every row has been seen.
        let stream = futures::stream::once(async move { deduper.run(input).await })
            .map_ok(|batches| futures::stream::iter(batches.into_iter().map(Ok)))
            .try_flatten();
        let output = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        ));
        Ok(crate::physical::trace::trace_stage(
            "SemDistinctExec",
            partition,
            output,
        ))
    }

    /// Keying neither adds nor drops rows; the `DISTINCT ON` above does that.
    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let input_rows = self.input.partition_statistics(partition)?.num_rows;
        let mut statistics = Statistics::new_unknown(&self.output_schema);
        statistics.num_rows = input_rows;
        Ok(Arc::new(statistics))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

/// Everything the single output stream needs.
struct Deduper {
    text: Arc<dyn PhysicalExpr>,
    similarity: f32,
    vectors: DocVectors,
    output_schema: SchemaRef,
    groups: Count,
    duplicate_rows: Count,
    /// What vectoring the documents cost.
    counts: VectorCounts,
}

impl Deduper {
    async fn run(self, mut input: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        while let Some(batch) = input.try_next().await? {
            batches.push(batch);
        }

        // The distinct documents present, in first-seen order. Order is the
        // whole contract here: the first document of a group represents it,
        // so the grouping must not depend on how the scan batched rows.
        let mut hashes: Vec<u64> = Vec::new();
        let mut texts: HashMap<u64, String> = HashMap::new();
        let mut rows: HashMap<u64, usize> = HashMap::new();
        for batch in &batches {
            let values = self.evaluate_texts(batch)?;
            let values = values
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("text cast to Utf8");
            for row in 0..values.len() {
                if !values.is_valid(row) {
                    continue;
                }
                let text = values.value(row);
                let hash = doc_hash(text);
                *rows.entry(hash).or_default() += 1;
                if texts.insert(hash, text.to_owned()).is_none() {
                    hashes.push(hash);
                }
            }
        }
        let vectors = if hashes.is_empty() {
            HashMap::new()
        } else {
            self.vectors
                .for_documents(&hashes, &texts, &rows, &self.counts, "SEMANTIC DISTINCT ON")
                .await?
        };
        // A document with no vector is left out of the grouping rather than
        // given a NULL key: SQL groups NULLs *together*, so a NULL key would
        // collapse them all into one. `with_keys` falls back to the
        // document's own identity instead. Order is kept.
        hashes.retain(|hash| vectors.contains_key(hash));

        let keys = self.keys(&hashes, &vectors);
        batches
            .into_iter()
            .map(|batch| self.with_keys(&keys, batch))
            .collect()
    }

    /// Group the documents and give each one its representative's key.
    fn keys(&self, hashes: &[u64], vectors: &HashMap<u64, Embedding>) -> HashMap<u64, String> {
        if hashes.is_empty() {
            return HashMap::new();
        }
        let points: Vec<Embedding> = hashes.iter().map(|hash| vectors[hash].clone()).collect();
        let groups = greedy_groups(&points, self.similarity);

        let mut distinct = 0usize;
        let keys = hashes
            .iter()
            .zip(&groups)
            .map(|(&hash, &representative)| {
                let representative = hashes[representative];
                if representative == hash {
                    distinct += 1;
                } else {
                    self.duplicate_rows.add(1);
                }
                // The representative's hash *is* the key: short, stable, and
                // shared by exactly the documents that collapse together.
                (hash, format_key(representative))
            })
            .collect();
        self.groups.add(distinct);
        keys
    }

    /// Append the key column to a batch.
    fn with_keys(&self, keys: &HashMap<u64, String>, batch: RecordBatch) -> Result<RecordBatch> {
        let texts = self.evaluate_texts(&batch)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text cast to Utf8");
        let mut builder = StringBuilder::with_capacity(batch.num_rows(), 0);
        for row in 0..batch.num_rows() {
            if !texts.is_valid(row) {
                // NULL text has no identity to compare; SQL groups NULL keys
                // together, which is what a plain DISTINCT ON would do too.
                builder.append_null();
                continue;
            }
            let hash = doc_hash(texts.value(row));
            match keys.get(&hash) {
                Some(key) => builder.append_value(key),
                // No vector: be its own group, keyed by its own identity, so
                // only a byte-identical twin joins it.
                None => builder.append_value(format_key(hash)),
            }
        }
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        columns.push(Arc::new(builder.finish()));
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
}

/// A group key: the representative document's identity, rendered so it sorts
/// and compares as an ordinary string.
fn format_key(doc_hash: u64) -> String {
    format!("{doc_hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_fixed_width_hex() {
        assert_eq!(format_key(0), "0000000000000000");
        assert_eq!(format_key(u64::MAX), "ffffffffffffffff");
        // Fixed width so keys compare and sort as strings without surprises.
        assert_eq!(format_key(1).len(), format_key(u64::MAX).len());
    }
}
