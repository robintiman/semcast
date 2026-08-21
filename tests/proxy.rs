//! The learned proxy: calibration fits a classifier to the labels it already
//! bought, and thresholds that instead of raw similarity.
//!
//! The corpus here is built so cosine similarity carries *no* signal — the
//! condition embeds onto an axis no document occupies, so every document
//! scores the same against it. That is the honest version of a weak embedder
//! or a condition phrased unlike the corpus, and it is where a proxy fitted to
//! the labels has something to add that similarity cannot.

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use semcast::model::MockModel;
use semcast::{IndexOptions, create_semantic_index, semcast_context};

/// 200 documents, half of them matching. `alpha`/`beta` are the embedding
/// themes; the mock's verdict follows `alpha`.
const CORPUS: usize = 200;

async fn corpus_context(model: Arc<MockModel>, dir: &tempfile::TempDir) -> SessionContext {
    let ctx = semcast_context(model);
    let rows: Vec<String> = (0..CORPUS)
        .map(|i| {
            let theme = if i % 2 == 0 { "alpha" } else { "beta" };
            format!("({i}, '{theme} document number {i}')")
        })
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE docs AS SELECT * FROM (VALUES {}) AS t(id, body)",
        rows.join(", "),
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
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
    ctx
}

/// The mock embeds by theme, so `alpha` documents sit on one axis and `beta`
/// on another. The condition mentions neither, so it lands on the catch-all
/// axis and is equidistant from everything.
fn themed_model() -> Arc<MockModel> {
    Arc::new(MockModel::answering_yes_to(["alpha"]).embedding_by_theme(["alpha", "beta"]))
}

async fn matching_ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches: Vec<RecordBatch> = semcast::sql(ctx, sql)
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

const QUERY: &str = "SELECT id FROM docs
                     WHERE body MEANS 'the customer escalated' WITH RECALL 0.9";

/// Recall first: whatever the proxy does, every true match must come back.
/// A cheaper funnel that loses rows is not cheaper, it is wrong.
#[tokio::test]
async fn the_learned_proxy_keeps_every_match() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = corpus_context(themed_model(), &dir).await;

    let ids = matching_ids(&ctx, QUERY).await;
    let expected: Vec<i64> = (0..CORPUS as i64).filter(|i| i % 2 == 0).collect();
    assert_eq!(ids, expected, "every alpha document must survive");
}

/// And it must actually prune. Similarity alone cannot separate this corpus —
/// every document scores identically against the condition — so a funnel that
/// only knew about similarity would send all 200 documents to verify. Coming
/// in under that baseline is the learned proxy doing the work.
#[tokio::test]
async fn the_learned_proxy_prunes_what_similarity_cannot() {
    let dir = tempfile::tempdir().unwrap();
    let model = themed_model();
    let ctx = corpus_context(Arc::clone(&model), &dir).await;

    matching_ids(&ctx, QUERY).await;
    let calls = model.completion_calls();

    // The arithmetic when the proxy prunes every non-matching document: 64
    // calibration labels, plus the matching half minus those the calibration
    // already labelled and cached. A funnel that could not separate them
    // would spend one call per document instead.
    let floor = 64 + CORPUS / 2 - 32;
    assert!(
        (floor..CORPUS).contains(&calls),
        "expected about {floor} calls — 64 labels plus the uncached matches — \
         against a similarity-only baseline of {CORPUS}; spent {calls}",
    );
}

/// A corpus the labels cannot separate has nothing to learn from, and the
/// fallback to similarity must keep the query correct rather than fail it.
#[tokio::test]
async fn a_one_class_sample_falls_back_and_stays_correct() {
    let dir = tempfile::tempdir().unwrap();
    // Every document matches, so the fit has one class and is refused.
    let model =
        Arc::new(MockModel::answering_yes_to(["document"]).embedding_by_theme(["alpha", "beta"]));
    let ctx = corpus_context(Arc::clone(&model), &dir).await;

    let ids = matching_ids(&ctx, QUERY).await;
    assert_eq!(ids.len(), CORPUS, "every document matches, so all return");
}
