//! A session with a decider: every `MEANS` question goes to it, everything
//! generative stays with the session model. Asserted on call counts, since
//! both models would return the right rows either way.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use semcast::cache::{InMemoryCache, SemanticCache};
use semcast::model::{
    Completion, CompletionRequest, Embedding, MockModel, ModelId, ModelProvider, YesNoAsk,
};
use semcast::{Result, SemcastContextBuilder, SemcastError};

const ANGRY: &str = "this is completely unacceptable, I want a refund now";
const PRICING: &str = "could you tell me what the enterprise plan costs";
const DULL: &str = "thanks, that answers my question";

/// A decision model: answers a [`YesNoAsk`] by the condition's keyword, with
/// a probability, and refuses anything else — as `SystemOneProvider` does.
#[derive(Debug, Default)]
struct Decider {
    asks: Mutex<Vec<YesNoAsk>>,
}

impl Decider {
    fn calls(&self) -> usize {
        self.asks.lock().unwrap().len()
    }

    fn conditions(&self) -> Vec<String> {
        let mut conditions: Vec<String> = self
            .asks
            .lock()
            .unwrap()
            .iter()
            .map(|ask| ask.condition.clone())
            .collect();
        conditions.sort();
        conditions.dedup();
        conditions
    }
}

fn holds(condition: &str, input: &str) -> bool {
    match condition {
        c if c.contains("angry") => input.contains("unacceptable"),
        c if c.contains("pricing") => input.contains("costs"),
        c if c.contains("sync") => input.contains("sync"),
        _ => false,
    }
}

#[async_trait]
impl ModelProvider for Decider {
    fn id(&self) -> ModelId {
        ModelId("decider".to_owned())
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        requests
            .into_iter()
            .map(|request| {
                let Some(ask) = request.ask else {
                    return Err(SemcastError::Model(
                        "a decision model answers yes/no questions only".to_owned(),
                    ));
                };
                let yes = holds(&ask.condition, &request.input);
                self.asks.lock().unwrap().push(ask);
                Ok(Completion {
                    text: if yes { "yes" } else { "no" }.to_owned(),
                    input_tokens: 0,
                    output_tokens: 0,
                    yes_probability: Some(if yes { 0.8 } else { 0.2 }),
                    merged: false,
                })
            })
            .collect()
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Embedding>> {
        Err(SemcastError::Model(
            "a decision model cannot embed".to_owned(),
        ))
    }
}

/// The session model: serves `CAST`, and would answer "yes" to every `MEANS`
/// about the angry ticket if one ever reached it.
fn session_model() -> Arc<MockModel> {
    Arc::new(
        MockModel::answering_json_with(|request| {
            let mood = if request.input.contains("unacceptable") {
                "angry"
            } else {
                "calm"
            };
            serde_json::json!({ "mood": mood })
        })
        .also_answering_yes_to(["unacceptable"]),
    )
}

struct Session {
    ctx: SessionContext,
    decider: Arc<Decider>,
    model: Arc<MockModel>,
}

/// tickets(id, body) — one angry, one pricing, one dull, one NULL.
async fn load(ctx: SessionContext) -> SessionContext {
    ctx.sql(&format!(
        "CREATE TABLE tickets AS
         SELECT * FROM (VALUES
             (1, '{ANGRY}'),
             (2, '{PRICING}'),
             (3, '{DULL}'),
             (4, CAST(NULL AS VARCHAR))
         ) AS t(id, body)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    semcast::sql(
        &ctx,
        "CREATE SEMANTIC TYPE Ticket AS (mood ONEOF(calm, angry) 'how the customer feels')",
    )
    .await
    .unwrap();
    ctx
}

async fn session_with(cache: Arc<dyn SemanticCache>) -> Session {
    let decider = Arc::new(Decider::default());
    let model = session_model();
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as Arc<dyn ModelProvider>)
        .with_decider(Arc::clone(&decider) as Arc<dyn ModelProvider>)
        .with_cache(cache)
        .build();
    Session {
        ctx: load(ctx).await,
        decider,
        model,
    }
}

async fn session() -> Session {
    session_with(Arc::new(InMemoryCache::default())).await
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
            let column = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            column.values().to_vec()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn means_is_answered_by_the_decider() {
    let s = session().await;
    let batches = run(
        &s.ctx,
        "SELECT id FROM tickets WHERE body MEANS 'the customer is angry'",
    )
    .await;
    assert_eq!(ids(&batches), vec![1]);
    // One question per non-NULL row, none to the session model.
    assert_eq!(s.decider.calls(), 3);
    assert_eq!(s.model.completion_calls(), 0);
    let asks = s.decider.asks.lock().unwrap();
    assert!(asks.iter().all(|ask| !ask.excerpts), "full text, no index");
}

#[tokio::test]
async fn several_conditions_are_asked_one_at_a_time() {
    let s = session().await;
    let batches = run(
        &s.ctx,
        "SELECT id,
                body MEANS 'the customer is angry' AS angry,
                body MEANS 'the customer asks about pricing' AS pricing
         FROM tickets ORDER BY id",
    )
    .await;
    let mut rows = Vec::new();
    for batch in &batches {
        let id = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let angry = batch
            .column(1)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let pricing = batch
            .column(2)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            rows.push((
                id.value(row),
                angry.is_valid(row).then(|| angry.value(row)),
                pricing.is_valid(row).then(|| pricing.value(row)),
            ));
        }
    }
    assert_eq!(
        rows,
        vec![
            (1, Some(true), Some(false)),
            (2, Some(false), Some(true)),
            (3, Some(false), Some(false)),
            // NULL text: no call, NULL labels.
            (4, None, None),
        ],
    );
    // Three texts, two conditions, each its own question: a fused call would
    // be three, and would need the JSON answer a decider cannot write.
    assert_eq!(s.decider.calls(), 6);
    assert_eq!(s.decider.conditions().len(), 2);
    assert_eq!(s.model.completion_calls(), 0);
}

#[tokio::test]
async fn a_label_and_a_filter_share_one_verdict() {
    let s = session().await;
    run(
        &s.ctx,
        "SELECT id,
                body MEANS 'the customer is angry' AS angry,
                body MEANS 'the customer asks about pricing' AS pricing
         FROM tickets",
    )
    .await;
    let asked = s.decider.calls();
    // Unfused labels are verify questions, so the filter finds them stored.
    let batches = run(
        &s.ctx,
        "SELECT id FROM tickets WHERE body MEANS 'the customer asks about pricing'",
    )
    .await;
    assert_eq!(ids(&batches), vec![2]);
    assert_eq!(s.decider.calls(), asked);
}

#[tokio::test]
async fn confidence_is_the_deciders_probability() {
    let s = session().await;
    let batches = run(
        &s.ctx,
        "SELECT id, CONFIDENCE(body MEANS 'the customer is angry') AS p
         FROM tickets ORDER BY id",
    )
    .await;
    let mut scores = Vec::new();
    for batch in &batches {
        let p = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            scores.push(
                p.is_valid(row)
                    .then(|| (p.value(row) * 10.0).round() / 10.0),
            );
        }
    }
    assert_eq!(scores, vec![Some(0.8), Some(0.2), Some(0.2), None]);
    assert_eq!(s.model.completion_calls(), 0);
}

#[tokio::test]
async fn cast_stays_with_the_session_model() {
    let s = session().await;
    let batches = run(
        &s.ctx,
        "SELECT id, CAST(body AS Ticket).mood AS mood
         FROM tickets
         WHERE body MEANS 'the customer is angry'",
    )
    .await;
    assert_eq!(ids(&batches), vec![1]);
    let mood = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(mood.value(0), "angry");
    // The filter cost three decisions; extraction ran on the one survivor,
    // on the session model, with a schema.
    assert_eq!(s.decider.calls(), 3);
    assert_eq!(s.model.completion_calls(), 1);
    assert!(s.model.completion_schemas()[0].is_some());
}

#[tokio::test]
async fn a_deciders_verdicts_are_its_own() {
    let cache: Arc<dyn SemanticCache> = Arc::new(InMemoryCache::default());
    let query = "SELECT id FROM tickets WHERE body MEANS 'the customer is angry'";

    let with_decider = session_with(Arc::clone(&cache)).await;
    run(&with_decider.ctx, query).await;
    assert_eq!(with_decider.decider.calls(), 3);

    // The same store, no decider: the session model is asked afresh rather
    // than served another model's verdicts.
    let model = session_model();
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as Arc<dyn ModelProvider>)
        .with_cache(Arc::clone(&cache))
        .build();
    let ctx = load(ctx).await;
    run(&ctx, query).await;
    assert_eq!(model.completion_calls(), 3);

    // And the decider's are still there for a session that has one.
    let again = session_with(cache).await;
    run(&again.ctx, query).await;
    assert_eq!(again.decider.calls(), 0);
}

#[tokio::test]
async fn explain_names_the_decider() {
    let s = session().await;
    let batches = run(
        &s.ctx,
        "EXPLAIN SELECT id FROM tickets WHERE body MEANS 'the customer is angry'",
    )
    .await;
    let plan: String = batches
        .iter()
        .flat_map(|b| {
            let column = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            (0..b.num_rows())
                .map(|row| column.value(row).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(plan.contains("decider"), "{plan}");
}

#[tokio::test]
async fn explain_bills_one_call_per_row_per_condition() {
    let s = session().await;
    let batches = run(
        &s.ctx,
        "EXPLAIN SELECT id,
                body MEANS 'the customer is angry' AS angry,
                body MEANS 'the customer asks about pricing' AS pricing
         FROM tickets",
    )
    .await;
    let plan: String = batches
        .iter()
        .flat_map(|b| {
            let column = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            (0..b.num_rows())
                .map(|row| column.value(row).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    // Four rows, two conditions, nothing fused.
    assert!(plan.contains("unfused"), "{plan}");
    assert!(plan.contains("≤8 model calls"), "{plan}");
}

/// meetings(id, transcript) with a semantic index: the mock embeds by byte
/// histogram, so the document identical to the condition scores 1.0.
async fn indexed_session() -> Session {
    let decider = Arc::new(Decider::default());
    let model = session_model();
    let root = tempfile::tempdir().unwrap();
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as Arc<dyn ModelProvider>)
        .with_decider(Arc::clone(&decider) as Arc<dyn ModelProvider>)
        .with_index_root(root.path())
        .build();
    // Leak the tempdir so the Lance dataset outlives this helper.
    std::mem::forget(root);
    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES (1, 'sync'), (2, 'abcdefghijkl')) AS t(id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    run(&ctx, "CREATE SEMANTIC INDEX ON meetings(transcript)").await;
    Session {
        ctx,
        decider,
        model,
    }
}

#[tokio::test]
async fn an_indexed_filter_asks_the_decider_about_excerpts() {
    let s = indexed_session().await;
    let batches = run(
        &s.ctx,
        "SELECT id FROM meetings WHERE transcript MEANS 'sync'",
    )
    .await;
    assert_eq!(ids(&batches), vec![1]);
    assert_eq!(s.model.completion_calls(), 0);
    let asks = s.decider.asks.lock().unwrap();
    assert!(!asks.is_empty());
    assert!(asks.iter().all(|ask| ask.excerpts), "{asks:?}");
}

#[tokio::test]
async fn calibration_labels_come_from_the_decider() {
    let s = indexed_session().await;
    let batches = run(
        &s.ctx,
        "SELECT id FROM meetings WHERE transcript MEANS 'sync' WITH RECALL 0.9",
    )
    .await;
    assert_eq!(ids(&batches), vec![1]);
    // The sample is labeled on full text, by the decider; the session model
    // is never asked.
    assert_eq!(s.model.completion_calls(), 0);
    let asks = s.decider.asks.lock().unwrap();
    assert!(asks.iter().any(|ask| !ask.excerpts), "{asks:?}");
}
