//! Fusing several `MEANS` over one column in a `WHERE` into a single model
//! call — the filter-side counterpart of the `CASE` branch fusion in
//! `tests/classify.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use datafusion::arrow::array::Int64Array;
use datafusion::execution::context::SessionContext;
use semcast::Result;
use semcast::model::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use semcast::{IndexOptions, create_semantic_index, semcast_context};
use serde_json::Value;

/// Answers per *condition*, which `MockModel` cannot: its needles match the
/// document, so every predicate over one document would agree. Serves both
/// request shapes — the fused JSON object and the schemaless yes/no.
#[derive(Debug, Default)]
struct KeywordModel {
    calls: AtomicUsize,
}

impl KeywordModel {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

/// A condition holds when the document contains it as a substring.
#[async_trait]
impl ModelProvider for KeywordModel {
    fn id(&self) -> ModelId {
        ModelId("keyword".to_owned())
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        self.calls.fetch_add(requests.len(), Ordering::Relaxed);
        requests
            .into_iter()
            .map(|req| {
                let text = match &req.schema {
                    // Fused: "c0: <condition>" lines, one boolean each.
                    Some(_) => {
                        let mut object = serde_json::Map::new();
                        for line in req.system.lines() {
                            if let Some((key, condition)) = line.split_once(": ") {
                                if key.starts_with('c') && key.len() > 1 {
                                    object.insert(
                                        key.to_owned(),
                                        Value::Bool(req.input.contains(condition.trim())),
                                    );
                                }
                            }
                        }
                        Value::Object(object).to_string()
                    }
                    // Verify: "Predicate: <condition>", answered yes or no.
                    None => {
                        let condition = req
                            .system
                            .lines()
                            .find_map(|l| l.strip_prefix("Predicate: "))
                            .unwrap_or_default()
                            .trim();
                        if !condition.is_empty() && req.input.contains(condition) {
                            "yes".to_owned()
                        } else {
                            "no".to_owned()
                        }
                    }
                };
                Ok(Completion {
                    output_tokens: 1,
                    input_tokens: req.input.len() / 4,
                    text,
                })
            })
            .collect()
    }

    /// One axis per keyword so an indexed search can actually separate the
    /// documents; anything else lands on its own axis.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0_f32; 8];
                for (i, needle) in ["rare", "common"].iter().enumerate() {
                    if text.contains(needle) {
                        v[i] = 1.0;
                    }
                }
                if v.iter().all(|x| *x == 0.0) {
                    v[7] = 1.0;
                }
                v
            })
            .collect())
    }
}

/// 6 docs: "rare" is in 1, "common" is in 5, one row has neither.
async fn docs(model: Arc<KeywordModel>) -> SessionContext {
    let ctx = semcast_context(model);
    ctx.sql(
        "CREATE TABLE docs AS SELECT * FROM (VALUES
             (1,'rare common'),(2,'common'),(3,'common'),
             (4,'common'),(5,'common'),(6,'nothing')
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

const SELECTIVE_FIRST: &str = "SELECT id FROM docs WHERE body MEANS 'rare' AND body MEANS 'common'";
const SELECTIVE_LAST: &str = "SELECT id FROM docs WHERE body MEANS 'common' AND body MEANS 'rare'";

async fn ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches = semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64")
                .values()
                .to_vec()
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// The teardown's measurement, inverted. Stacked, this query cost 7 calls one
/// way round and 11 the other; fused it costs one per row either way.
#[tokio::test]
async fn conjunct_order_no_longer_changes_the_cost() {
    let mut costs = Vec::new();
    for query in [SELECTIVE_FIRST, SELECTIVE_LAST] {
        let model = KeywordModel::new();
        let ctx = docs(Arc::clone(&model)).await;
        assert_eq!(ids(&ctx, query).await, vec![1]);
        costs.push(model.calls());
    }
    assert_eq!(costs[0], costs[1], "order must not change the price");
    assert_eq!(
        costs[0], 6,
        "one fused call per row, beating the best stacked ordering",
    );
}

#[tokio::test]
async fn fusing_asks_every_condition_in_one_call() {
    let model = KeywordModel::new();
    let ctx = docs(Arc::clone(&model)).await;
    let plan = semcast::sql(&ctx, SELECTIVE_FIRST)
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();
    let display = plan.display_indent().to_string();

    assert!(display.contains("SemClassify"), "plan:\n{display}");
    assert!(
        display.contains("1 model call per row"),
        "the node should advertise the fusion:\n{display}",
    );
    assert!(
        !display.contains("SemFilter"),
        "a fused group leaves no SemFilter behind:\n{display}",
    );
}

/// The added booleans are an implementation detail of the filter, not part of
/// the answer.
#[tokio::test]
async fn fusion_leaves_the_output_schema_alone() {
    let ctx = docs(KeywordModel::new()).await;
    let batches = semcast::sql(
        &ctx,
        "SELECT * FROM docs WHERE body MEANS 'rare' AND body MEANS 'common'",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let schema = batches[0].schema();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["id", "body"],
        "no __sem_class_* column may escape the filter",
    );
}

/// A single predicate has nothing to fuse with and keeps the funnel-capable
/// `SemFilter` it always had.
#[tokio::test]
async fn one_condition_is_not_fused() {
    let ctx = docs(KeywordModel::new()).await;
    let plan = semcast::sql(&ctx, "SELECT id FROM docs WHERE body MEANS 'rare'")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();
    let display = plan.display_indent().to_string();
    assert!(display.contains("SemFilter"), "plan:\n{display}");
    assert!(!display.contains("SemClassify"), "plan:\n{display}");
}

/// Predicates over different columns cannot share a call: the model would
/// have to read two documents at once.
#[tokio::test]
async fn different_columns_are_not_fused() {
    let model = KeywordModel::new();
    let ctx = semcast_context(Arc::clone(&model) as Arc<dyn ModelProvider>);
    ctx.sql(
        "CREATE TABLE pairs AS SELECT * FROM (VALUES
             (1,'rare','common'),(2,'nothing','common')
         ) AS t(id, left_text, right_text)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let plan = semcast::sql(
        &ctx,
        "SELECT id FROM pairs WHERE left_text MEANS 'rare' AND right_text MEANS 'common'",
    )
    .await
    .unwrap()
    .into_optimized_plan()
    .unwrap();
    let display = plan.display_indent().to_string();
    assert!(!display.contains("SemClassify"), "plan:\n{display}");
    assert_eq!(display.matches("SemFilter").count(), 2, "plan:\n{display}");
}

/// An index prunes rows before any call at all, which beats a floor of one
/// call per row — so an indexed column keeps its stack rather than fusing.
#[tokio::test]
async fn an_indexed_column_keeps_its_funnel() {
    let model = KeywordModel::new();
    let ctx = docs(Arc::clone(&model)).await;
    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "docs",
        "body",
        IndexOptions {
            path: Some(dir.path().join("docs.body.lance")),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let plan = semcast::sql(&ctx, SELECTIVE_FIRST)
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();
    let display = plan.display_indent().to_string();
    assert!(
        !display.contains("SemClassify"),
        "an indexed group must not fuse away its pre-filter:\n{display}",
    );
    assert_eq!(display.matches("SemFilter").count(), 2, "plan:\n{display}");
}

/// `WITH RECALL` calibrates the index stage, which a classify does not have.
/// Rather than drop the target silently, the group stays stacked.
#[tokio::test]
async fn with_recall_opts_a_group_out_of_fusion() {
    let ctx = docs(KeywordModel::new()).await;
    let plan = semcast::sql(
        &ctx,
        "SELECT id FROM docs WHERE body MEANS 'rare' AND body MEANS 'common' WITH RECALL 0.9",
    )
    .await
    .unwrap()
    .into_optimized_plan()
    .unwrap();
    let display = plan.display_indent().to_string();
    assert!(!display.contains("SemClassify"), "plan:\n{display}");
    assert_eq!(display.matches("SemFilter").count(), 2, "plan:\n{display}");
}

/// A row the model fails on is dropped, exactly as the unfused filter drops
/// it — fusion must not turn a failure into a match.
#[tokio::test]
async fn a_null_text_row_costs_nothing_and_matches_nothing() {
    let model = KeywordModel::new();
    let ctx = semcast_context(Arc::clone(&model) as Arc<dyn ModelProvider>);
    ctx.sql(
        "CREATE TABLE maybe AS SELECT * FROM (VALUES
             (1,'rare common'),(2, CAST(NULL AS VARCHAR))
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    assert_eq!(
        ids(
            &ctx,
            "SELECT id FROM maybe WHERE body MEANS 'rare' AND body MEANS 'common'"
        )
        .await,
        vec![1],
    );
    assert_eq!(model.calls(), 1, "the NULL row never reaches the model");
}

/// Both halves of the query build a `SemClassify`, and both number their
/// columns from zero. The filter's projection drops its booleans before the
/// select list's node adds its own, so the names never collide.
#[tokio::test]
async fn fusing_in_a_where_and_labelling_in_a_select_coexist() {
    let model = KeywordModel::new();
    let ctx = docs(Arc::clone(&model)).await;
    let batches = semcast::sql(
        &ctx,
        "SELECT id, body MEANS 'common' AS flag
         FROM docs WHERE body MEANS 'rare' AND body MEANS 'common'",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let schema = batches[0].schema();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["id", "flag"],
    );
    let flags = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::BooleanArray>()
        .expect("flag is Boolean");
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert!(flags.value(0), "doc 1 is 'rare common'");
}
