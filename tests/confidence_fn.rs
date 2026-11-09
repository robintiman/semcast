//! `CONFIDENCE(text MEANS 'condition')`: the model's P("yes") for exactly the
//! question the `MEANS` asks, sharing that question's cache entries.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use semcast::cache::{CacheKey, CachedValue, InMemoryCache, SemanticCache};
use semcast::model::{Completion, CompletionRequest, Embedding, MockModel, ModelId, ModelProvider};
use semcast::{Result, SemcastContextBuilder};

fn furious_model() -> MockModel {
    MockModel::answering_yes_to(["furious"])
        .with_yes_probability(|input| if input.contains("furious") { 0.8 } else { 0.3 })
}

async fn tickets(model: Arc<dyn ModelProvider>, cache: Arc<dyn SemanticCache>) -> SessionContext {
    let ctx = SemcastContextBuilder::new(model).with_cache(cache).build();
    ctx.sql(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES
             (1, 'I am furious about the outage'),
             (2, 'thanks, all good now'),
             (3, NULL)
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

async fn run(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    semcast::sql(ctx, sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn ids(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            ids.iter().map(|v| v.unwrap()).collect::<Vec<_>>()
        })
        .collect()
}

fn floats(batches: &[RecordBatch], column: usize) -> Vec<Option<f64>> {
    batches
        .iter()
        .flat_map(|b| {
            let values = b
                .column(column)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("CONFIDENCE() is a DOUBLE");
            values.iter().collect::<Vec<_>>()
        })
        .map(|v| v.map(|v| (v * 1000.0).round() / 1000.0))
        .collect()
}

#[tokio::test]
async fn a_select_list_confidence_is_p_yes_per_row() {
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    let sql = "SELECT id, CONFIDENCE(body MEANS 'the customer is angry') AS angry \
               FROM tickets ORDER BY id";
    let batches = run(&ctx, sql).await;
    assert_eq!(ids(&batches), [1, 2, 3]);
    assert_eq!(
        floats(&batches, 1),
        [Some(0.8), Some(0.3), None],
        "NULL text has no probability"
    );
    assert_eq!(batches[0].schema().field(1).name(), "angry");
    assert_eq!(model.completion_calls(), 2, "NULL text costs no call");

    run(&ctx, sql).await;
    assert_eq!(model.completion_calls(), 2, "the probability is cached");
}

#[tokio::test]
async fn confidence_and_where_means_share_one_answer() {
    // WHERE first: the verdict and its probability are cached together, so
    // CONFIDENCE() costs nothing.
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    run(&ctx, "SELECT id FROM tickets WHERE body MEANS 'is angry'").await;
    assert_eq!(model.completion_calls(), 2);
    let batches = run(
        &ctx,
        "SELECT id, CONFIDENCE(body MEANS 'is angry') FROM tickets ORDER BY id",
    )
    .await;
    assert_eq!(floats(&batches, 1), [Some(0.8), Some(0.3), None]);
    assert_eq!(model.completion_calls(), 2);

    // And the other way round.
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    run(
        &ctx,
        "SELECT CONFIDENCE(body MEANS 'is angry') FROM tickets",
    )
    .await;
    assert_eq!(model.completion_calls(), 2);
    let batches = run(&ctx, "SELECT id FROM tickets WHERE body MEANS 'is angry'").await;
    assert_eq!(ids(&batches), [1]);
    assert_eq!(model.completion_calls(), 2);
}

#[tokio::test]
async fn a_cached_verdict_without_a_probability_is_asked_again() {
    let cache: Arc<dyn SemanticCache> = Arc::new(InMemoryCache::default());
    // A verdict stored by a provider without logprobs.
    let ctx = tickets(Arc::new(Unscored(furious_model())), Arc::clone(&cache)).await;
    run(&ctx, "SELECT id FROM tickets WHERE body MEANS 'is angry'").await;

    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::clone(&cache)).await;
    run(&ctx, "SELECT id FROM tickets WHERE body MEANS 'is angry'").await;
    assert_eq!(
        model.completion_calls(),
        0,
        "the verdict alone serves a WHERE"
    );
    let batches = run(
        &ctx,
        "SELECT id, CONFIDENCE(body MEANS 'is angry') FROM tickets ORDER BY id",
    )
    .await;
    assert_eq!(floats(&batches, 1), [Some(0.8), Some(0.3), None]);
    assert_eq!(model.completion_calls(), 2, "but not a CONFIDENCE()");
}

#[tokio::test]
async fn a_provider_without_probabilities_gives_null() {
    let ctx = tickets(
        Arc::new(Unscored(furious_model())),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(
        &ctx,
        "SELECT id, CONFIDENCE(body MEANS 'is angry') FROM tickets ORDER BY id",
    )
    .await;
    assert_eq!(floats(&batches, 1), [None, None, None]);
}

#[tokio::test]
async fn a_missing_probability_is_asked_again() {
    // Not settled: another provider under the same model id may report one.
    let cache: Arc<dyn SemanticCache> = Arc::new(InMemoryCache::default());
    let unscored = Arc::new(Counting::new(Unscored(furious_model())));
    let ctx = tickets(Arc::clone(&unscored) as _, Arc::clone(&cache)).await;
    let sql = "SELECT id, CONFIDENCE(body MEANS 'is angry') FROM tickets ORDER BY id";
    let batches = run(&ctx, sql).await;
    assert_eq!(floats(&batches, 1), [None, None, None]);
    assert_eq!(unscored.calls(), 2);

    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, cache).await;
    let batches = run(&ctx, sql).await;
    assert_eq!(floats(&batches, 1), [Some(0.8), Some(0.3), None]);
    assert_eq!(model.completion_calls(), 2);
}

#[tokio::test]
async fn a_probability_is_not_paired_with_a_different_stored_verdict() {
    // A store that keeps its first value, as the SQLite store does, already
    // holds "no" for row 1. The fresh answer says yes, so its probability
    // describes a verdict the store doesn't hold: NULL, not a mismatch.
    let model = furious_model();
    let cache = Arc::new(FirstWins::default());
    // Seed the opposite verdicts through a model that disagrees.
    let contrary = tickets(
        Arc::new(Unscored(MockModel::answering_yes_to(["thanks"]))),
        Arc::clone(&cache) as _,
    )
    .await;
    run(
        &contrary,
        "SELECT id FROM tickets WHERE body MEANS 'is angry'",
    )
    .await;

    let model = Arc::new(model);
    let ctx = tickets(Arc::clone(&model) as _, Arc::clone(&cache) as _).await;
    let sql = "SELECT id, CONFIDENCE(body MEANS 'is angry') FROM tickets ORDER BY id";
    let batches = run(&ctx, sql).await;
    assert_eq!(floats(&batches, 1), [None, None, None]);
    assert_eq!(model.completion_calls(), 2);
    run(&ctx, sql).await;
    assert_eq!(
        model.completion_calls(),
        2,
        "the mismatch is settled, not re-asked"
    );
    let batches = run(&ctx, "SELECT id FROM tickets WHERE body MEANS 'is angry'").await;
    assert_eq!(ids(&batches), [2], "the stored verdict still wins");
}

#[tokio::test]
async fn confidence_filters_in_a_where() {
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    let batches = run(
        &ctx,
        "SELECT * FROM tickets WHERE CONFIDENCE(body MEANS 'is angry') > 0.5",
    )
    .await;
    assert_eq!(ids(&batches), [1]);
    assert_eq!(
        batches[0].schema().fields().len(),
        2,
        "no helper column leaks into SELECT *"
    );

    // Alongside a MEANS on the same question, the MEANS reads the cache.
    let batches = run(
        &ctx,
        "SELECT id FROM tickets \
         WHERE CONFIDENCE(body MEANS 'is angry') < 0.5 AND body MEANS 'is angry'",
    )
    .await;
    assert!(ids(&batches).is_empty());
    assert_eq!(model.completion_calls(), 2);
}

#[tokio::test]
async fn cheaper_conjuncts_run_before_confidence() {
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    let batches = run(
        &ctx,
        "SELECT id FROM tickets WHERE id = 1 AND CONFIDENCE(body MEANS 'is angry') > 0.5",
    )
    .await;
    assert_eq!(ids(&batches), [1]);
    assert_eq!(model.completion_calls(), 1, "row 2 was filtered out first");

    // A MEANS conjunct prunes before CONFIDENCE() pays, too. The mock says
    // yes to anything about the furious row, so only row 1 survives it.
    let batches = run(
        &ctx,
        "SELECT id FROM tickets \
         WHERE body MEANS 'is upset' AND CONFIDENCE(body MEANS 'is happy') > 0.5",
    )
    .await;
    assert_eq!(ids(&batches), [1]);
    assert_eq!(
        model.completion_calls(),
        4,
        "1 earlier + 2 MEANS calls + 1 CONFIDENCE call, for the survivor only"
    );
}

#[tokio::test]
async fn delete_and_update_refuse_semantic_predicates() {
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    for sql in [
        "DELETE FROM tickets WHERE body MEANS 'is angry'",
        "DELETE FROM tickets WHERE CONFIDENCE(body MEANS 'is angry') > 0.5",
        "DELETE FROM tickets WHERE id IN (SELECT id FROM tickets WHERE body MEANS 'x')",
        "UPDATE tickets SET body = 'x' WHERE body MEANS 'is angry'",
        "EXPLAIN ANALYZE DELETE FROM tickets WHERE body MEANS 'is angry'",
        // DataFusion drops a subquery predicate from DELETE, semantic or not.
        "DELETE FROM tickets WHERE id IN (SELECT id FROM tickets WHERE id = 1)",
    ] {
        let error = match semcast::sql(&ctx, sql).await {
            Err(e) => e.to_string(),
            Ok(df) => df.collect().await.expect_err(sql).to_string(),
        };
        assert!(error.contains("affect every row"), "{sql}: {error}");
    }
    let batches = run(&ctx, "SELECT id FROM tickets ORDER BY id").await;
    assert_eq!(ids(&batches), [1, 2, 3], "nothing was deleted");
    assert_eq!(model.completion_calls(), 0);
}

#[tokio::test]
async fn the_documented_delete_workaround_keeps_the_other_rows() {
    let ctx = tickets(
        Arc::new(furious_model()),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    run(
        &ctx,
        "CREATE TABLE calm AS SELECT * FROM tickets \
         WHERE id NOT IN (SELECT id FROM tickets WHERE body MEANS 'is angry')",
    )
    .await;
    let batches = run(&ctx, "SELECT id FROM calm ORDER BY id").await;
    assert_eq!(ids(&batches), [2, 3]);
}

#[tokio::test]
async fn with_recall_leaves_confidence_alone() {
    let ctx = tickets(
        Arc::new(furious_model()),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(
        &ctx,
        "SELECT id, CONFIDENCE(body MEANS 'is angry') FROM tickets \
         WHERE body MEANS 'is upset' WITH RECALL 0.9",
    )
    .await;
    assert_eq!(ids(&batches), [1]);
    assert_eq!(floats(&batches, 1), [Some(0.8)]);
}

#[tokio::test]
async fn confidence_orders_rows() {
    let ctx = tickets(
        Arc::new(furious_model()),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(
        &ctx,
        "SELECT id FROM tickets WHERE body IS NOT NULL \
         ORDER BY CONFIDENCE(body MEANS 'is angry') ASC",
    )
    .await;
    assert_eq!(ids(&batches), [2, 1]);
}

#[tokio::test]
async fn a_repeated_confidence_is_asked_once() {
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    let batches = run(
        &ctx,
        "SELECT id, CONFIDENCE(body MEANS 'is angry') AS a, \
                1 - CONFIDENCE(body MEANS 'is angry') AS b FROM tickets ORDER BY id",
    )
    .await;
    assert_eq!(floats(&batches, 1), [Some(0.8), Some(0.3), None]);
    assert_eq!(floats(&batches, 2), [Some(0.2), Some(0.7), None]);
    assert_eq!(model.completion_calls(), 2);
}

#[tokio::test]
async fn confidence_in_a_case_is_gated() {
    let model = Arc::new(furious_model());
    let ctx = tickets(Arc::clone(&model) as _, Arc::new(InMemoryCache::default())).await;
    let batches = run(
        &ctx,
        "SELECT id, CASE WHEN id = 2 THEN 'skip' \
                         WHEN CONFIDENCE(body MEANS 'is angry') > 0.5 THEN 'angry' END \
         FROM tickets ORDER BY id",
    )
    .await;
    assert_eq!(ids(&batches), [1, 2, 3]);
    assert_eq!(model.completion_calls(), 1, "row 2 was claimed first");
}

#[tokio::test]
async fn confidence_misuse_is_a_plan_error() {
    let ctx = tickets(
        Arc::new(furious_model()),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    for (sql, expected) in [
        ("SELECT CONFIDENCE(id = 1) FROM tickets", "takes one MEANS"),
        (
            "SELECT count(*) FROM tickets GROUP BY CONFIDENCE(body MEANS 'x')",
            "CONFIDENCE(text MEANS",
        ),
        (
            "SELECT id FROM tickets \
             WHERE CONFIDENCE(CONFIDENCE(body MEANS 'a') MEANS 'b') > 0",
            "cannot read the output",
        ),
        (
            "SELECT id FROM tickets \
             ORDER BY CONFIDENCE(CONFIDENCE(body MEANS 'a') MEANS 'b')",
            "cannot read the output",
        ),
        (
            "SELECT id FROM tickets WHERE CONFIDENCE(body MEANS 'a') MEANS 'b'",
            "cannot read the output",
        ),
        (
            "SELECT id FROM tickets \
             WHERE CAST(CONFIDENCE(body MEANS 'a') AS VARCHAR) MEANS 'b'",
            "cannot read the output",
        ),
        (
            "SELECT id FROM tickets \
             WHERE CONFIDENCE(body MEANS 'a') > 0.5 OR body MEANS 'b'",
            "top-level AND conjunct",
        ),
        (
            "SELECT avg(CONFIDENCE(body MEANS 'x')) FROM tickets",
            "aggregate or window",
        ),
    ] {
        let error = match semcast::sql(&ctx, sql).await {
            Err(e) => e.to_string(),
            Ok(df) => df.collect().await.expect_err(sql).to_string(),
        };
        assert!(error.contains(expected), "{sql}: {error}");
    }
}

#[tokio::test]
async fn explain_names_the_confidence_stage() {
    let ctx = tickets(
        Arc::new(furious_model()),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(
        &ctx,
        "EXPLAIN SELECT CONFIDENCE(body MEANS 'is angry') FROM tickets",
    )
    .await;
    let text = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    assert!(
        text.contains("SemClassify: CONFIDENCE('is angry')"),
        "{text}"
    );
    assert!(
        text.contains("SemClassifyExec: CONFIDENCE('is angry')"),
        "{text}"
    );
}

/// Counts completion requests sent through it.
#[derive(Debug)]
struct Counting<M> {
    inner: M,
    calls: std::sync::atomic::AtomicUsize,
}

impl<M> Counting<M> {
    fn new(inner: M) -> Self {
        Self {
            inner,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait]
impl<M: ModelProvider> ModelProvider for Counting<M> {
    fn id(&self) -> ModelId {
        self.inner.id()
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        self.calls
            .fetch_add(requests.len(), std::sync::atomic::Ordering::SeqCst);
        self.inner.complete(requests).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        self.inner.embed(texts).await
    }
}

/// A provider that reports no probabilities, like one without logprobs.
#[derive(Debug)]
struct Unscored(MockModel);

#[async_trait]
impl ModelProvider for Unscored {
    fn id(&self) -> ModelId {
        self.0.id()
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        self.0
            .complete(requests)
            .await
            .into_iter()
            .map(|c| {
                c.map(|c| Completion {
                    yes_probability: None,
                    ..c
                })
            })
            .collect()
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        self.0.embed(texts).await
    }
}

/// Keeps the first value written under each key, as the SQLite store does.
#[derive(Debug, Default)]
struct FirstWins(Mutex<InMemoryCache>);

impl SemanticCache for FirstWins {
    fn get(&self, key: &CacheKey) -> Option<CachedValue> {
        self.0.lock().unwrap().get(key)
    }

    fn put(&self, key: CacheKey, value: CachedValue) {
        let inner = self.0.lock().unwrap();
        if inner.get(&key).is_none() {
            inner.put(key, value);
        }
    }
}
