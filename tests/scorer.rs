//! The scorer role: a cheap model's P("yes") pre-filters an unindexed `MEANS`
//! under `WITH RECALL`, with the floor calibrated on a sample the session
//! model labels.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use datafusion::arrow::array::Int64Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use semcast::cache::{InMemoryCache, SemanticCache};
use semcast::model::{Completion, CompletionRequest, Embedding, MockModel, ModelId, ModelProvider};
use semcast::{Result, SemcastContextBuilder};

/// 200 tickets; every tenth is furious. The session model (a mock) says yes
/// to exactly those.
const MATCHES: [i64; 20] = [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
];

/// A model under its own id that counts the requests it is sent.
#[derive(Debug)]
struct Named {
    id: &'static str,
    inner: MockModel,
    calls: AtomicUsize,
    /// Drop every probability, like a provider without logprobs.
    unscored: bool,
    /// Drop the probability for inputs containing this.
    unscored_on: Option<&'static str>,
}

impl Named {
    fn new(id: &'static str, inner: MockModel) -> Arc<Self> {
        Arc::new(Self {
            id,
            inner,
            calls: AtomicUsize::new(0),
            unscored: false,
            unscored_on: None,
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ModelProvider for Named {
    fn id(&self) -> ModelId {
        ModelId(self.id.to_owned())
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        self.calls.fetch_add(requests.len(), Ordering::SeqCst);
        let drop: Vec<bool> = requests
            .iter()
            .map(|r| self.unscored || self.unscored_on.is_some_and(|n| r.input.contains(n)))
            .collect();
        self.inner
            .complete(requests)
            .await
            .into_iter()
            .zip(drop)
            .map(|(c, drop)| {
                c.map(|c| Completion {
                    yes_probability: if drop { None } else { c.yes_probability },
                    ..c
                })
            })
            .collect()
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        self.inner.embed(texts).await
    }
}

fn session_model() -> Arc<Named> {
    Named::new("session", MockModel::answering_yes_to(["furious"]))
}

/// A scorer that ranks the furious tickets high and everything else low.
fn good_scorer() -> Arc<Named> {
    Named::new(
        "scorer",
        MockModel::answering_yes_to(["furious"])
            .with_yes_probability(|input| if input.contains("furious") { 0.9 } else { 0.05 }),
    )
}

async fn tickets(
    model: Arc<Named>,
    scorer: Option<Arc<Named>>,
    cache: Arc<dyn SemanticCache>,
) -> SessionContext {
    let mut builder = SemcastContextBuilder::new(model).with_cache(cache);
    if let Some(scorer) = scorer {
        builder = builder.with_scorer(scorer);
    }
    let ctx = builder.build();
    ctx.sql(
        "CREATE TABLE tickets AS SELECT value AS id, \
             CASE WHEN value % 10 = 0 THEN 'ticket ' || value || ': I am furious' \
                  ELSE 'ticket ' || value || ': all good' END AS body \
         FROM generate_series(1, 200)",
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
    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            ids.iter().map(|v| v.unwrap()).collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

const RECALL_QUERY: &str = "SELECT id FROM tickets WHERE body MEANS 'is angry' WITH RECALL 0.9";

#[tokio::test]
async fn a_scorer_spares_the_session_model_under_recall() {
    let model = session_model();
    let scorer = good_scorer();
    let ctx = tickets(
        Arc::clone(&model),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(&ctx, RECALL_QUERY).await;
    assert_eq!(ids(&batches), MATCHES);
    assert_eq!(scorer.calls(), 200, "the scorer reads every row");
    // 64 sampled labels, then verify on the survivors the sample missed.
    assert!(
        model.calls() <= 64 + MATCHES.len(),
        "session model calls: {}",
        model.calls()
    );

    // Everything is cached now: scores under the scorer's id, verdicts
    // under the session model's.
    let session_calls = model.calls();
    run(&ctx, RECALL_QUERY).await;
    assert_eq!(scorer.calls(), 200);
    assert_eq!(model.calls(), session_calls);
}

#[tokio::test]
async fn without_recall_the_scorer_is_never_asked() {
    let model = session_model();
    let scorer = good_scorer();
    let ctx = tickets(
        Arc::clone(&model),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(&ctx, "SELECT id FROM tickets WHERE body MEANS 'is angry'").await;
    assert_eq!(ids(&batches), MATCHES);
    assert_eq!(scorer.calls(), 0);
    assert_eq!(model.calls(), 200);
}

#[tokio::test]
async fn without_a_scorer_recall_changes_nothing_unindexed() {
    let model = session_model();
    let ctx = tickets(Arc::clone(&model), None, Arc::new(InMemoryCache::default())).await;
    let batches = run(&ctx, RECALL_QUERY).await;
    assert_eq!(ids(&batches), MATCHES);
    assert_eq!(model.calls(), 200);
}

#[tokio::test]
async fn a_scorer_without_probabilities_prunes_nothing() {
    let model = session_model();
    let scorer = Arc::new(Named {
        id: "scorer",
        inner: MockModel::answering_yes_to(["furious"]),
        calls: AtomicUsize::new(0),
        unscored: true,
        unscored_on: None,
    });
    let ctx = tickets(
        Arc::clone(&model),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(&ctx, RECALL_QUERY).await;
    assert_eq!(ids(&batches), MATCHES);
    assert_eq!(
        model.calls(),
        200,
        "every row verified; the sample's labels are reused, not re-asked"
    );
}

#[tokio::test]
async fn a_misleading_scorer_lowers_the_floor_instead_of_losing_matches() {
    // The scorer misses half the furious tickets, scoring them like calm
    // ones. Keeping 90% of the matches means keeping those too.
    let model = session_model();
    let scorer = Named::new(
        "scorer",
        MockModel::answering_yes_to(["furious"]).with_yes_probability(|input| {
            let furious = input.contains("furious");
            let odd_tens = input.contains("10:")
                || input.contains("30:")
                || input.contains("50:")
                || input.contains("70:")
                || input.contains("90:");
            if furious && !odd_tens { 0.9 } else { 0.05 }
        }),
    );
    let ctx = tickets(
        Arc::clone(&model),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(&ctx, RECALL_QUERY).await;
    assert_eq!(ids(&batches), MATCHES);
}

#[tokio::test]
async fn a_certified_target_holds() {
    let model = session_model();
    let scorer = good_scorer();
    let ctx = tickets(
        Arc::clone(&model),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'is angry' \
         WITH RECALL 0.8 WITH CONFIDENCE 0.9",
    )
    .await;
    assert_eq!(ids(&batches), MATCHES);
    assert!(model.calls() <= 200);
}

#[tokio::test]
async fn explain_shows_the_scorer_stage() {
    let ctx = tickets(
        session_model(),
        Some(good_scorer()),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(&ctx, &format!("EXPLAIN {RECALL_QUERY}")).await;
    let text = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    assert!(
        text.contains("ScoreScanExec: MEANS('is angry') scorer=scorer"),
        "{text}"
    );
}

#[tokio::test]
async fn a_sample_without_matches_prunes_nothing() {
    // 2000 rows, 8 matches: a 64-row sample likely holds none, and the
    // scorer underrates every match. Its own yes/no must not decide alone.
    for sql in [
        "SELECT id FROM rare WHERE body MEANS 'is angry' WITH RECALL 0.9",
        "SELECT id FROM rare WHERE body MEANS 'is angry' WITH RECALL 0.9 WITH CONFIDENCE 0.95",
    ] {
        let scorer = Named::new(
            "scorer",
            MockModel::answering_yes_to(["furious"])
                .with_yes_probability(|input| if input.contains("furious") { 0.3 } else { 0.05 }),
        );
        let ctx = SemcastContextBuilder::new(session_model())
            .with_scorer(scorer)
            .build();
        ctx.sql(
            "CREATE TABLE rare AS SELECT value AS id, \
                 CASE WHEN value % 250 = 0 THEN 'ticket ' || value || ': I am furious' \
                      ELSE 'ticket ' || value || ': all good' END AS body \
             FROM generate_series(1, 2000)",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
        let batches = run(&ctx, sql).await;
        assert_eq!(
            ids(&batches),
            [250, 500, 750, 1000, 1250, 1500, 1750, 2000],
            "{sql}"
        );
    }
}

#[tokio::test]
async fn a_scorer_without_probabilities_is_asked_once() {
    let scorer = Arc::new(Named {
        id: "scorer",
        inner: MockModel::answering_yes_to(["furious"]),
        calls: AtomicUsize::new(0),
        unscored: true,
        unscored_on: None,
    });
    let ctx = tickets(
        session_model(),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    run(&ctx, RECALL_QUERY).await;
    assert_eq!(scorer.calls(), 200);
    run(&ctx, RECALL_QUERY).await;
    assert_eq!(scorer.calls(), 200, "switched off, not re-asked");
}

#[tokio::test]
async fn an_unscored_scorer_leaves_nothing_durable_behind() {
    // The scorer gave no probabilities here; the same model, later serving
    // CONFIDENCE() with logprobs over the same cache, must still be asked.
    let cache: Arc<dyn SemanticCache> = Arc::new(InMemoryCache::default());
    let scorer = Arc::new(Named {
        id: "small",
        inner: MockModel::answering_yes_to(["furious"]),
        calls: AtomicUsize::new(0),
        unscored: true,
        unscored_on: None,
    });
    let ctx = tickets(session_model(), Some(scorer), Arc::clone(&cache)).await;
    run(&ctx, RECALL_QUERY).await;

    let small = Named::new("small", MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&small), None, cache).await;
    let batches = run(
        &ctx,
        "SELECT id, CONFIDENCE(body MEANS 'is angry') AS c FROM tickets WHERE id = 10",
    )
    .await;
    let scores = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Float64Array>()
        .unwrap();
    assert_eq!(scores.value(0), 0.9_f32 as f64);
    assert_eq!(small.calls(), 1);
}

#[tokio::test]
async fn a_scorer_that_is_the_session_model_is_ignored() {
    let model = session_model();
    let scorer = Named::new("session", MockModel::answering_yes_to(["nothing"]));
    let ctx = tickets(
        Arc::clone(&model),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    let batches = run(&ctx, RECALL_QUERY).await;
    assert_eq!(ids(&batches), MATCHES);
    assert_eq!(scorer.calls(), 0);
}

#[tokio::test]
async fn a_few_unscored_rows_never_switch_a_working_scorer_off() {
    let scorer = Arc::new(Named {
        id: "scorer",
        inner: MockModel::answering_yes_to(["furious"])
            .with_yes_probability(|input| if input.contains("furious") { 0.9 } else { 0.05 }),
        calls: AtomicUsize::new(0),
        unscored: false,
        unscored_on: Some("odd"),
    });
    let ctx = tickets(
        session_model(),
        Some(Arc::clone(&scorer)),
        Arc::new(InMemoryCache::default()),
    )
    .await;
    run(&ctx, RECALL_QUERY).await;
    run(
        &ctx,
        "INSERT INTO tickets SELECT value + 1000, 'ticket ' || value || ': odd' \
         FROM generate_series(1, 10)",
    )
    .await;
    // Ten new rows the scorer can't score: one batch without a probability.
    run(&ctx, RECALL_QUERY).await;
    assert_eq!(scorer.calls(), 210);
    let before = scorer.calls();
    run(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'is furious' WITH RECALL 0.9",
    )
    .await;
    assert_eq!(
        scorer.calls() - before,
        210,
        "still scoring a new condition"
    );
}
