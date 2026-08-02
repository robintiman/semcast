//! The clustering stage — grouping rows by what they are about.
//!
//! Three steps, none of them per-row. The index hands back the document
//! vectors it already paid to compute; k-means groups them (sweeping for `k`
//! when the query didn't say); then one model call per cluster reads its most
//! central documents and names it. Cost is a function of the number of
//! groups, not the number of rows.
//!
//! Blocking, and unavoidably so: which group a row belongs to is a fact about
//! the whole relation. That is the price of clustering, and the operator says
//! so in `EXPLAIN` rather than hiding it.

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

use crate::cache::{CacheKey, CachedValue, SemanticCache};
use crate::index::kmeans::{self, Clustering};
use crate::index::{SemanticIndex, doc_hash};
use crate::logical::sem_cluster::REPRESENTATIVES_PER_CLUSTER;
use crate::model::{CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::physical::extract::parse_json_object;

/// Version of the synthesized labelling prompt. Participates in cache keys:
/// bump it and every cached label is honestly invalidated.
pub const CLUSTER_PROMPT_VERSION: &str = "cluster-v1";

/// Enough for a short noun phrase and its JSON wrapper.
const LABEL_MAX_TOKENS: usize = 64;

/// Separates the excerpts one labelling call reads.
const MEMBER_SEPARATOR: &str = "\n---\n";

/// The instruction the labelling model sees. Users never write this.
pub fn synthesize_label_prompt() -> String {
    "You are naming a group of documents that were grouped together because \
     they are about similar things. You see several of them, separated by \
     `---`.\n\n\
     Reply with a JSON object {\"label\": \"...\"} whose label is a short noun \
     phrase — at most five words — naming what this group is about. Name what \
     these documents share, not what any one of them says."
        .to_owned()
}

fn label_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "label": { "type": "string" } },
        "required": ["label"],
        "additionalProperties": false
    })
}

/// Groups input rows by meaning, appending the label of each row's group.
///
/// Row semantics ("rows fail, queries don't"): a row whose text is NULL or
/// which the index has never seen gets a NULL label rather than being
/// dropped — it is still a row, it just has no group. A cluster whose
/// labelling call fails gets a positional fallback name, so the grouping
/// survives even when the naming does not.
#[derive(Debug)]
pub struct SemClusterExec {
    input: Arc<dyn ExecutionPlan>,
    /// Evaluates to the text being clustered, against input batches.
    text: Arc<dyn PhysicalExpr>,
    k: Option<usize>,
    /// Name of the appended label column.
    label_column: String,
    index: Arc<dyn SemanticIndex>,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl SemClusterExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        text: Arc<dyn PhysicalExpr>,
        k: Option<usize>,
        label_column: impl Into<String>,
        index: Arc<dyn SemanticIndex>,
        model: Arc<dyn ModelProvider>,
        cache: Arc<dyn SemanticCache>,
    ) -> Result<Self> {
        let label_column = label_column.into();
        let mut fields = input.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(&label_column, DataType::Utf8, true)));
        let output_schema = Arc::new(Schema::new(fields));
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            // One partition: a grouping is a fact about the whole relation.
            Partitioning::UnknownPartitioning(1),
            // Nothing is emitted until every group has been named.
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            text,
            k,
            label_column,
            index,
            model,
            cache,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for SemClusterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SemClusterExec: MEANING OF ")?;
        match self.k {
            Some(k) => write!(f, "INTO {k}   ≤{k} model calls", k = k)?,
            None => write!(
                f,
                "INTO auto (sweeping {:?})   ≤{} model calls",
                kmeans::AUTO_K,
                kmeans::AUTO_K.iter().max().copied().unwrap_or(0)
            )?,
        }
        write!(
            f,
            " embed_model={} model={}",
            self.index.embed_model_id(),
            self.model.id()
        )
    }
}

impl ExecutionPlan for SemClusterExec {
    fn name(&self) -> &str {
        "SemClusterExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// The grouping spans the whole input, so DataFusion must coalesce before
    /// this operator rather than after it.
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
            self.k,
            self.label_column.clone(),
            Arc::clone(&self.index),
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
        let grouper = Grouper {
            text: Arc::clone(&self.text),
            k: self.k,
            index: Arc::clone(&self.index),
            model_id: self.model.id(),
            model: Arc::clone(&self.model),
            cache: Arc::clone(&self.cache),
            output_schema: Arc::clone(&self.output_schema),
            groups: MetricBuilder::new(&self.metrics).counter("groups", partition),
            model_calls: MetricBuilder::new(&self.metrics).counter("model_calls", partition),
            cache_hits: MetricBuilder::new(&self.metrics).counter("cache_hits", partition),
            unindexed_rows: MetricBuilder::new(&self.metrics).counter("unindexed_rows", partition),
            labels_failed: MetricBuilder::new(&self.metrics).counter("labels_failed", partition),
        };
        // One future produces every output batch: no row can be labelled
        // before every row has been seen.
        let stream = futures::stream::once(async move { grouper.run(input).await })
            .map_ok(|batches| futures::stream::iter(batches.into_iter().map(Ok)))
            .try_flatten();
        let output = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        ));
        Ok(crate::physical::trace::trace_stage(
            "SemClusterExec",
            partition,
            output,
        ))
    }

    /// Clustering neither adds nor drops rows.
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
struct Grouper {
    text: Arc<dyn PhysicalExpr>,
    k: Option<usize>,
    index: Arc<dyn SemanticIndex>,
    model_id: ModelId,
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    output_schema: SchemaRef,
    groups: Count,
    model_calls: Count,
    cache_hits: Count,
    unindexed_rows: Count,
    labels_failed: Count,
}

impl Grouper {
    async fn run(self, mut input: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
        let vectors = self.index.doc_vectors().await?;

        let mut batches = Vec::new();
        while let Some(batch) = input.try_next().await? {
            batches.push(batch);
        }

        // The distinct documents present in the input, in a stable order:
        // clustering must not depend on how the scan happened to batch rows.
        let mut hashes: Vec<u64> = Vec::new();
        let mut texts: HashMap<u64, String> = HashMap::new();
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
                if !vectors.contains_key(&hash) {
                    self.unindexed_rows.add(1);
                    continue;
                }
                if texts.insert(hash, text.to_owned()).is_none() {
                    hashes.push(hash);
                }
            }
        }
        hashes.sort_unstable();

        let labels = self.label(&hashes, &texts, &vectors).await?;
        batches
            .into_iter()
            .map(|batch| self.with_labels(&labels, batch))
            .collect()
    }

    /// Cluster the documents and name each group, returning the label per
    /// document.
    async fn label(
        &self,
        hashes: &[u64],
        texts: &HashMap<u64, String>,
        vectors: &HashMap<u64, Embedding>,
    ) -> Result<HashMap<u64, String>> {
        if hashes.is_empty() {
            return Ok(HashMap::new());
        }
        let points: Vec<Embedding> = hashes.iter().map(|hash| vectors[hash].clone()).collect();
        let clustering = kmeans::cluster(&points, self.k);
        self.groups.add(clustering.k);

        let names = self
            .name_clusters(hashes, texts, &points, &clustering)
            .await;
        Ok(hashes
            .iter()
            .zip(&clustering.assignments)
            .map(|(&hash, &cluster)| (hash, names[cluster].clone()))
            .collect())
    }

    /// One model call per cluster, reading its most central documents.
    async fn name_clusters(
        &self,
        hashes: &[u64],
        texts: &HashMap<u64, String>,
        points: &[Embedding],
        clustering: &Clustering,
    ) -> Vec<String> {
        let representatives =
            kmeans::representatives(points, clustering, REPRESENTATIVES_PER_CLUSTER);
        let members = clustering.members();

        let mut names = vec![String::new(); clustering.k];
        let mut requests = Vec::new();
        let mut pending = Vec::new();
        for cluster in 0..clustering.k {
            let Some(chosen) = representatives.get(&cluster) else {
                names[cluster] = fallback_name(cluster);
                continue;
            };
            let excerpts: Vec<&str> = chosen
                .iter()
                .map(|&point| texts[&hashes[point]].as_str())
                .collect();
            let input = excerpts.join(MEMBER_SEPARATOR);
            // Keyed on the cluster's whole membership: the same grouping
            // always gets the same name, and a different grouping never
            // inherits one.
            let key = self.cache_key(&members[cluster], hashes);
            if let Some(CachedValue::Value(label)) = self.cache.get(&key) {
                self.cache_hits.add(1);
                names[cluster] = label;
                continue;
            }
            requests.push(CompletionRequest {
                system: synthesize_label_prompt(),
                input,
                max_tokens: LABEL_MAX_TOKENS,
                schema: Some(label_schema()),
            });
            pending.push((cluster, key));
        }
        self.model_calls.add(requests.len());

        let completions = self.model.complete(requests).await;
        debug_assert_eq!(completions.len(), pending.len());
        for ((cluster, key), completion) in pending.into_iter().zip(&completions) {
            match completion.as_ref().ok().and_then(|c| parse_label(&c.text)) {
                Some(label) => {
                    names[cluster] = label.clone();
                    self.cache.put(key, CachedValue::Value(label));
                }
                // A group that could not be named is still a group: fall back
                // to a positional name rather than collapsing the grouping.
                None => {
                    self.labels_failed.add(1);
                    names[cluster] = fallback_name(cluster);
                }
            }
        }
        deduplicate(names)
    }

    /// Append the label column to a batch.
    fn with_labels(
        &self,
        labels: &HashMap<u64, String>,
        batch: RecordBatch,
    ) -> Result<RecordBatch> {
        let texts = self.evaluate_texts(&batch)?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text cast to Utf8");
        let mut builder = StringBuilder::with_capacity(batch.num_rows(), 0);
        for row in 0..batch.num_rows() {
            // NULL text, or a document the index never saw: no group.
            let label = texts
                .is_valid(row)
                .then(|| labels.get(&doc_hash(texts.value(row))))
                .flatten();
            builder.append_option(label);
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

    /// Provenance: the exact set of documents grouped together, plus the
    /// model and prompt scheme that named them.
    fn cache_key(&self, members: &[usize], hashes: &[u64]) -> CacheKey {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        let mut member_hashes: Vec<u64> = members.iter().map(|&point| hashes[point]).collect();
        member_hashes.sort_unstable();
        member_hashes.hash(&mut hasher);
        CacheKey {
            type_version: "cluster".to_owned(),
            field: "label".to_owned(),
            input_hash: hasher.finish(),
            model_id: self.model_id.clone(),
            prompt_version: CLUSTER_PROMPT_VERSION.to_owned(),
        }
    }
}

fn fallback_name(cluster: usize) -> String {
    format!("group {}", cluster + 1)
}

/// Group labels are keys, so two groups must never share one. A model that
/// names two clusters identically gets them numbered apart rather than
/// silently merged by the `GROUP BY` above.
fn deduplicate(names: Vec<String>) -> Vec<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    names
        .into_iter()
        .map(|name| {
            let count = seen.entry(name.clone()).or_insert(0);
            *count += 1;
            if *count == 1 {
                name
            } else {
                format!("{name} ({count})")
            }
        })
        .collect()
}

/// `None` means the model didn't give a usable label.
fn parse_label(text: &str) -> Option<String> {
    let label = parse_json_object(text)?
        .get("label")?
        .as_str()?
        .trim()
        .to_owned();
    (!label.is_empty()).then_some(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_label_object() {
        assert_eq!(
            parse_label(r#"{"label": "billing complaints"}"#),
            Some("billing complaints".to_owned())
        );
        assert_eq!(
            parse_label("```json\n{\"label\": \"launch planning\"}\n```"),
            Some("launch planning".to_owned())
        );
    }

    #[test]
    fn unusable_labels_are_none() {
        assert_eq!(parse_label("billing complaints"), None);
        assert_eq!(parse_label(r#"{"name": "x"}"#), None);
        assert_eq!(
            parse_label(r#"{"label": "   "}"#),
            None,
            "blank is unusable"
        );
    }

    #[test]
    fn duplicate_labels_are_numbered_apart() {
        assert_eq!(
            deduplicate(vec![
                "billing".to_owned(),
                "launch".to_owned(),
                "billing".to_owned(),
                "billing".to_owned(),
            ]),
            vec!["billing", "launch", "billing (2)", "billing (3)"]
        );
    }

    #[test]
    fn a_group_that_cannot_be_named_still_has_one() {
        assert_eq!(fallback_name(0), "group 1");
        assert_eq!(fallback_name(3), "group 4");
    }
}
