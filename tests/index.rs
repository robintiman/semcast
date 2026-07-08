//! Tests for roadmap step 2: the Lance-backed semantic index and its
//! programmatic API, against the deterministic mock model.

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::displayable;
use semcast::index::lance::LanceIndex;
use semcast::index::{SearchParams, doc_hash};
use semcast::model::{MockModel, ModelId, ModelProvider};
use semcast::{IndexOptions, create_semantic_index, refresh_semantic_index, semcast_context};

const MATCHING: &str = "we agreed to ship offline sync in Q3";
const OTHER: &str = "nothing notable happened";

/// meetings(meeting_id, title, transcript) — one matching transcript, one
/// non-matching, one NULL.
async fn meetings_context() -> SessionContext {
    let ctx = semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'atlas planning',  '{MATCHING}'),
             (2, 'weekly standup',  '{OTHER}'),
             (3, 'retro',           CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, title, transcript)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

fn index_options(dir: &tempfile::TempDir) -> IndexOptions {
    IndexOptions {
        path: Some(dir.path().join("meetings.transcript.lance")),
        ..Default::default()
    }
}

#[tokio::test]
async fn index_skips_nulls_and_knows_its_documents() {
    let ctx = meetings_context().await;
    let dir = tempfile::tempdir().unwrap();
    let index = create_semantic_index(&ctx, "meetings", "transcript", index_options(&dir))
        .await
        .unwrap();

    let hashes = index.indexed_doc_hashes().await.unwrap();
    assert_eq!(hashes.len(), 2, "two non-null transcripts");
    assert!(hashes.contains(&doc_hash(MATCHING)));
    assert!(hashes.contains(&doc_hash(OTHER)));
}

#[tokio::test]
async fn search_ranks_the_identical_document_first() {
    let ctx = meetings_context().await;
    let dir = tempfile::tempdir().unwrap();
    let index = create_semantic_index(&ctx, "meetings", "transcript", index_options(&dir))
        .await
        .unwrap();

    // The mock embeds by byte histogram, so a query identical to a stored
    // chunk has cosine similarity exactly 1.
    let hits = index
        .search(MATCHING, &SearchParams::default())
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].doc_hash, doc_hash(MATCHING));
    assert!(
        hits[0].score > 0.999,
        "identical text scores ~1: {}",
        hits[0].score
    );
    assert_eq!(hits[0].text, MATCHING, "chunk text is the verify evidence");
}

#[tokio::test]
async fn refresh_indexes_only_new_documents() {
    let ctx = meetings_context().await;
    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(&ctx, "meetings", "transcript", index_options(&dir))
        .await
        .unwrap();

    assert_eq!(
        refresh_semantic_index(&ctx, "meetings", "transcript")
            .await
            .unwrap(),
        0,
        "nothing changed, nothing re-indexed",
    );

    ctx.sql("INSERT INTO meetings VALUES (4, 'sync sync', 'offline sync ships next week')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        refresh_semantic_index(&ctx, "meetings", "transcript")
            .await
            .unwrap(),
        1,
        "exactly the inserted row",
    );
}

#[tokio::test]
async fn opening_with_a_different_embedder_errors() {
    let ctx = meetings_context().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meetings.transcript.lance");
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            path: Some(path.clone()),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    #[derive(Debug)]
    struct OtherEmbedder(MockModel);

    #[async_trait::async_trait]
    impl ModelProvider for OtherEmbedder {
        fn id(&self) -> ModelId {
            ModelId("other-model".to_owned())
        }
        async fn complete(
            &self,
            requests: Vec<semcast::model::CompletionRequest>,
        ) -> Vec<semcast::Result<semcast::model::Completion>> {
            self.0.complete(requests).await
        }
        async fn embed(
            &self,
            texts: Vec<String>,
        ) -> semcast::Result<Vec<semcast::model::Embedding>> {
            self.0.embed(texts).await
        }
    }

    let err = LanceIndex::open(
        path.to_str().unwrap(),
        Arc::new(OtherEmbedder(MockModel::default())),
        SearchParams::default(),
    )
    .await
    .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("built with embed model"), "got: {message}");
    assert!(
        message.contains("mock"),
        "names the recorded model: {message}"
    );
    assert!(
        message.contains("other-model"),
        "names the session model: {message}"
    );
}

// ---------------------------------------------------------------------------
// The funnel: IndexScanExec pruning + chunk-based verify.
//
// The mock embeds by byte histogram over position mod 16, which only
// discriminates between short strings: a stored chunk identical to the
// query scores 1.0, while a short string with different characters scores
// well below 0.9. Long texts all look uniform (score ≈ 0.87 against short
// queries) — the chunk-verify test exploits that with the default floor.
// ---------------------------------------------------------------------------

/// Fixture with byte-histogram-separable transcripts: doc 1 is exactly the
/// query, doc 2 scores ~0.58 against it, doc 3 is NULL.
async fn separable_context(model: Arc<MockModel>) -> SessionContext {
    let ctx = semcast_context(model);
    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'a', 'sync'),
             (2, 'b', 'abcdefghijkl'),
             (3, 'c', CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, title, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
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
                .expect("meeting_id is Int64")
                .values()
                .to_vec()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn indexed_query_plans_the_funnel_and_prunes() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = separable_context(Arc::clone(&model)).await;
    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            path: Some(dir.path().join("idx.lance")),
            search: SearchParams {
                score_floor: 0.9,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let query = "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'";
    let plan = semcast::sql(&ctx, query)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let display = displayable(plan.as_ref()).indent(true).to_string();
    assert!(display.contains("IndexScanExec"), "plan:\n{display}");
    assert!(display.contains("best-effort"), "plan:\n{display}");
    assert!(
        display.contains("reads top-3 chunks per doc"),
        "plan:\n{display}",
    );

    let embeds_before = model.embed_calls();
    assert_eq!(matching_ids(&ctx, query).await, vec![1]);
    assert_eq!(
        model.completion_calls(),
        1,
        "doc 2 pruned by the index, NULL free — only doc 1 is verified",
    );
    assert_eq!(
        model.embed_calls() - embeds_before,
        1,
        "the condition is embedded exactly once per query",
    );
}

#[tokio::test]
async fn unindexed_rows_pass_through_to_full_text_verify() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = separable_context(Arc::clone(&model)).await;
    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            path: Some(dir.path().join("idx.lance")),
            search: SearchParams {
                score_floor: 0.9,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Inserted after the index was built: the index has never seen it, so
    // it must reach the model on full text — never be silently dropped.
    ctx.sql("INSERT INTO meetings VALUES (4, 'd', 'big sync news')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'",
    )
    .await;
    assert_eq!(ids, vec![1, 4]);
    assert_eq!(
        model.completion_calls(),
        2,
        "doc 1 verified via the index, doc 4 via passthrough",
    );
}

#[tokio::test]
async fn verify_reads_chunks_not_the_full_document() {
    let long_doc = "offline sync ".repeat(2000); // 4000 words ≫ one 384-word chunk
    let model = Arc::new(MockModel::answering_yes_to(["offline sync"]));
    let ctx = semcast_context(model.clone());
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES (1, 'a', '{long_doc}')) AS t(meeting_id, title, transcript)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            path: Some(dir.path().join("idx.lance")),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let query = "SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'";
    assert_eq!(matching_ids(&ctx, query).await, vec![1]);

    let inputs = model.completion_inputs();
    let input = inputs.last().expect("one verify call");
    assert!(
        input.len() < long_doc.len() / 2,
        "verify read {} bytes of a {}-byte document — should be top-3 chunks",
        input.len(),
        long_doc.len(),
    );
    assert!(input.contains("offline sync"));

    // The chunked verdict is cached: re-running costs zero new calls.
    let calls = model.completion_calls();
    assert_eq!(matching_ids(&ctx, query).await, vec![1]);
    assert_eq!(model.completion_calls(), calls);
}

#[tokio::test]
async fn create_semantic_index_ddl_plans_the_funnel() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = separable_context(model).await;

    let batches = semcast::sql(&ctx, "CREATE SEMANTIC INDEX ON meetings(transcript)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(
        batches.iter().all(|b| b.num_rows() == 0),
        "DDL yields an empty result",
    );

    let plan = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'",
    )
    .await
    .unwrap()
    .create_physical_plan()
    .await
    .unwrap();
    let display = displayable(plan.as_ref()).indent(true).to_string();
    assert!(display.contains("IndexScanExec"), "plan:\n{display}");
}

#[tokio::test]
async fn create_on_plain_datafusion_context_is_a_clear_error() {
    let ctx = SessionContext::new();
    ctx.sql("CREATE TABLE t AS SELECT 'text' AS c")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let err = create_semantic_index(&ctx, "t", "c", IndexOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("semcast_context"), "got: {err}");
}

// ---------------------------------------------------------------------------
// WITH RECALL (roadmap step 3): calibration with the index at the *default*
// floor (0.35) so a calibrated floor visibly changes what survives. Against
// the query 'sync', doc 1 scores 1.0 and the long doc 2 ~0.87 — the default
// floor keeps both, a calibrated one only doc 1.
//
// Doc sizes are load-bearing for the call counts. Doc 1 is a single chunk,
// so its chunked-verify input equals its full text — the same cache key its
// calibration label wrote, making its verify free. Doc 2 spans multiple
// chunks, so if it survives to verify, that's a separately-keyed (and
// therefore countable) model call.
// ---------------------------------------------------------------------------

async fn calibration_context(model: Arc<MockModel>) -> SessionContext {
    let long_doc = "abcdefghijkl ".repeat(800); // ≫ one 384-word chunk
    let ctx = semcast_context(model);
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'a', 'sync'),
             (2, 'b', '{long_doc}'),
             (3, 'c', CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, title, transcript)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(&ctx, "meetings", "transcript", index_options(&dir))
        .await
        .unwrap();
    // Leak the tempdir so the Lance dataset outlives this helper.
    std::mem::forget(dir);
    ctx
}

const CALIBRATED_QUERY: &str =
    "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync' WITH RECALL 0.9";

#[tokio::test]
async fn calibration_raises_the_floor_and_prunes_the_near_miss() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = calibration_context(Arc::clone(&model)).await;

    let embeds_before = model.embed_calls();
    assert_eq!(matching_ids(&ctx, CALIBRATED_QUERY).await, vec![1]);
    // 2 label calls (docs 1 and 2, full text); doc 1's verify hits the
    // cache entry its label wrote. Doc 2 scores ~0.87 — above the default
    // floor, below the calibrated one (1.0, doc 1's score) — so it never
    // reaches verify; surviving would cost a countable third call.
    assert_eq!(model.completion_calls(), 2);
    assert_eq!(
        model.embed_calls() - embeds_before,
        1,
        "calibration reuses the query's single wide search",
    );
}

#[tokio::test]
async fn repeat_calibrated_query_costs_zero_new_completions() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = calibration_context(Arc::clone(&model)).await;

    assert_eq!(matching_ids(&ctx, CALIBRATED_QUERY).await, vec![1]);
    let calls = model.completion_calls();
    assert_eq!(matching_ids(&ctx, CALIBRATED_QUERY).await, vec![1]);
    assert_eq!(
        model.completion_calls(),
        calls,
        "labels and verdicts are both served from the cache",
    );
}

#[tokio::test]
async fn unindexed_rows_still_pass_through_under_recall() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = calibration_context(Arc::clone(&model)).await;

    // Inserted after the index was built — never indexed.
    ctx.sql("INSERT INTO meetings VALUES (4, 'd', 'big sync news')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(matching_ids(&ctx, CALIBRATED_QUERY).await, vec![1, 4]);
    // 3 labels (docs 1, 2, 4); every verify is then free — doc 1's chunk
    // equals its full text, and doc 4's passthrough verify is full text,
    // the same cache key its label wrote. Doc 2 is pruned.
    assert_eq!(model.completion_calls(), 3);
}

#[tokio::test]
async fn all_negative_sample_falls_back_to_the_default_floor() {
    let model = Arc::new(MockModel::answering_yes_to(["nothing matches this"]));
    let ctx = calibration_context(Arc::clone(&model)).await;

    assert_eq!(
        matching_ids(&ctx, CALIBRATED_QUERY).await,
        Vec::<i64>::new()
    );
    // With no positives the floor stays at the default 0.35, which both
    // docs clear: 2 labels + doc 2's multi-chunk verify (doc 1's verify is
    // a cache hit). A wrongly raised floor would prune doc 2 → 2 calls.
    assert_eq!(model.completion_calls(), 3);
}

#[tokio::test]
async fn calibrated_plan_explains_the_contract_not_a_number() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = calibration_context(model).await;

    let plan = semcast::sql(&ctx, CALIBRATED_QUERY)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let display = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        display.contains("floor=calibrated(recall≥0.90, sample≤64)"),
        "plan:\n{display}",
    );
    assert!(!display.contains("best-effort"), "plan:\n{display}");
}

#[tokio::test]
async fn builder_index_root_hosts_ddl_created_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = semcast::SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["sync"])))
        .with_index_root(dir.path())
        .build();
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES (1, '{MATCHING}')) AS t(meeting_id, transcript)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    semcast::sql(&ctx, "CREATE SEMANTIC INDEX ON meetings(transcript)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert!(
        dir.path().join("meetings.transcript.lance").is_dir(),
        "Lance dataset lands under the builder's index root",
    );
}
