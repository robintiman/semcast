//! Tests for roadmap step 2: the stored semantic index and its
//! programmatic API, against the deterministic mock model.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::displayable;
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

#[tokio::test]
async fn index_skips_nulls_and_knows_its_documents() {
    let ctx = meetings_context().await;
    let index = create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
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
    let index = create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
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
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
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
async fn reattaching_with_a_different_embedder_errors() {
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

    // What the catalog saves for an index built with the mock embedder.
    let definition = r#"{"table":"meetings","column":"transcript","embed_model":"mock",
        "chunk_max_tokens":512,"chunk_overlap_tokens":64,
        "fetch_k":256,"score_floor":0.35,"chunks_per_doc":3}"#;
    let ctx = semcast_context(Arc::new(OtherEmbedder(MockModel::default())));
    let err = semcast::reattach_semantic_index(&ctx, definition)
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("built with embedder mock"),
        "got: {message}"
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
    let ctx = semcast::SemcastContextBuilder::new(model).build();
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
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
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
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
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
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
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

/// Issue #17: a leading comment made the DDL fail to parse, which is the first
/// thing you hit once the SQL lives in a file instead of at a `psql` prompt.
/// Asserting the index is actually *built*, not just that the call returned.
#[tokio::test]
async fn a_commented_create_semantic_index_still_builds_the_index() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = separable_context(model).await;

    semcast::sql(
        &ctx,
        "-- the transcripts are long, so index them\n\
         CREATE SEMANTIC INDEX ON meetings(transcript);",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    // The funnel only plans an IndexScanExec if the index exists.
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
// Doc sizes are load-bearing for the call counts. Doc 1 is a single chunk:
// its chunked-verify input equals its full text, but the chunked prompt
// differs, so it is keyed apart from its calibration label and costs its own
// call. Doc 2 spans multiple chunks, so if it survives to verify, that's a
// separately-keyed (and therefore countable) model call too.
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
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
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
    // 2 label calls (docs 1 and 2, full text), and nothing else: doc 1's
    // label already answers verify, which no longer asks the chunked
    // question too. Doc 2 scores ~0.87 — above the default floor, below the
    // calibrated one — and its label says no, so it never reaches verify.
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
    // 3 labels (docs 1, 2, 4), and nothing else: each label decides its
    // document for verify too. Doc 2 is pruned.
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
    // docs clear; their labels (2 calls) decide them, so no chunked verify
    // follows.
    assert_eq!(model.completion_calls(), 2);
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

// ---------------------------------------------------------------------------
// The index changes cost, never the answer: a corpus with more matching
// chunks than one search's `fetch_k` must still return every match. Before
// the pre-filter scored every chunk, it kept the global top `fetch_k` and
// pruned every other indexed document — a silent row drop that grew with
// the corpus and was invisible on small eval sets.
// ---------------------------------------------------------------------------

const MANY_MATCHES: i64 = 40;
const MANY_NEGATIVES: i64 = 20;

/// `MANY_MATCHES` "sync N" transcripts plus `MANY_NEGATIVES` digit-only
/// ones, indexed with a `fetch_k` far smaller than the number of matches.
/// Embeds by theme, so every match scores 1.0 against 'sync' and every
/// negative 0.0 — the index separates them perfectly, and any match lost is
/// the cap's doing, not the embedding's.
async fn oversubscribed_context(model: Arc<MockModel>) -> SessionContext {
    let ctx = semcast_context(model);
    let rows: Vec<String> = (1..=MANY_MATCHES)
        .map(|id| format!("({id}, 'sync {id}')"))
        .chain((1..=MANY_NEGATIVES).map(|n| format!("({}, '{:04}')", MANY_MATCHES + n, n * 7)))
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES {}) AS t(meeting_id, transcript)",
        rows.join(", "),
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            search: SearchParams {
                fetch_k: 16,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await
    .unwrap();
    ctx
}

#[tokio::test]
async fn filter_scan_scores_every_document_it_lists() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]).embedding_by_theme(["sync"]));
    let ctx = oversubscribed_context(model).await;
    let runtime = ctx
        .state()
        .config()
        .get_extension::<semcast::index::registry::SemcastRuntime>()
        .unwrap();
    let index = runtime.index_for("meetings", "transcript").unwrap();

    // Far more chunks than the index's fetch_k of 16, and no floor: one
    // consistent read must score every document it calls indexed, or the
    // pre-filter would prune the unscored ones.
    let scan = index.filter_scan("sync", f32::NEG_INFINITY).await.unwrap();
    let scored: std::collections::HashSet<u64> = scan.hits.iter().map(|h| h.doc_hash).collect();
    assert_eq!(scan.indexed.len(), (MANY_MATCHES + MANY_NEGATIVES) as usize);
    assert_eq!(scored, scan.indexed);
}

#[tokio::test]
async fn index_returns_every_match_beyond_fetch_k() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]).embedding_by_theme(["sync"]));
    let ctx = oversubscribed_context(Arc::clone(&model)).await;

    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'",
    )
    .await;
    assert_eq!(ids, (1..=MANY_MATCHES).collect::<Vec<_>>());
    assert_eq!(
        model.completion_calls(),
        MANY_MATCHES as usize,
        "every match verified, every negative still pruned by the index",
    );
}

#[tokio::test]
async fn calibrated_index_returns_matches_beyond_fetch_k() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]).embedding_by_theme(["sync"]));
    let ctx = oversubscribed_context(Arc::clone(&model)).await;

    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync' WITH RECALL 0.95",
    )
    .await;
    let floor = (0.95 * MANY_MATCHES as f64).ceil() as usize;
    assert!(
        ids.len() >= floor,
        "recall 0.95 of {MANY_MATCHES} matches keeps at least {floor}, got {}",
        ids.len(),
    );
}

#[tokio::test]
async fn uncertifiable_recall_target_prunes_nothing() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]).embedding_by_theme(["sync"]));
    let ctx = oversubscribed_context(Arc::clone(&model)).await;

    // 40 positives cannot certify 99.9% recall at 99% confidence. Pruning on
    // an uncertified floor would drop unsampled matches below it, so the
    // pre-filter stands down: every row reaches verify, negatives included.
    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync' \
         WITH RECALL 0.999 WITH CONFIDENCE 0.99",
    )
    .await;
    assert_eq!(ids, (1..=MANY_MATCHES).collect::<Vec<_>>());
    let verified_negatives = model
        .completion_inputs()
        .iter()
        .filter(|input| !input.contains("sync"))
        .count();
    assert_eq!(
        verified_negatives, MANY_NEGATIVES as usize,
        "no floor: each negative is judged on full text once — its label — \
         and that verdict decides it, rather than the score",
    );
}

// ---------------------------------------------------------------------------
// The calibration sample must be random over the whole input, not its first
// rows: on input sorted by anything that correlates with the score, a
// first-N sample sees one end of the distribution and sets the floor for
// the other end blind.
// ---------------------------------------------------------------------------

const EASY_MATCHES: i64 = 80;
const HARD_MATCHES: i64 = 40;

/// Every row matches 'sync'. The first `EASY_MATCHES` are short ("sync07",
/// cosine ~0.95 against 'sync' under the mock's positional byte embedding);
/// the last `HARD_MATCHES` are padded to 16 bytes ("sync zzzzzzzzz07",
/// ~0.5). Sorted easy-first, so the default 64-row sample, if taken from
/// the front, contains no hard match at all.
async fn sorted_matches_context(model: Arc<MockModel>) -> SessionContext {
    let ctx = semcast_context(model);
    let rows: Vec<String> = (1..=EASY_MATCHES)
        .map(|id| format!("({id}, 'sync{id:02}')"))
        .chain((1..=HARD_MATCHES).map(|n| {
            let id = EASY_MATCHES + n;
            format!("({id}, 'sync zzzzzzzzz{n:02}')")
        }))
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES {}) AS t(meeting_id, transcript)",
        rows.join(", "),
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    ctx
}

#[tokio::test]
async fn calibration_samples_the_whole_input_not_its_head() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = sorted_matches_context(model).await;

    let ids = matching_ids(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync' WITH RECALL 0.9",
    )
    .await;
    let total = (EASY_MATCHES + HARD_MATCHES) as usize;
    // A head sample sees only easy matches, sets the floor among them and
    // prunes every hard one: recall ~0.6. A random sample sees both kinds
    // and lands near the 0.9 target; 0.85 leaves room for sampling error.
    assert!(
        ids.len() * 100 >= total * 85,
        "recall target 0.9, kept {} of {total}",
        ids.len(),
    );
    let hard_kept = ids.iter().filter(|id| **id > EASY_MATCHES).count();
    assert!(hard_kept > 0, "hard matches must not all be pruned");
}

/// A calibrated scan buffers its whole input to sample it. That buffer is
/// charged to the session's memory pool: over the limit, the query fails
/// with a clear error instead of exhausting the process. The same query
/// without `WITH RECALL` streams and still succeeds.
#[tokio::test]
async fn calibration_buffer_respects_the_memory_limit() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = semcast::SemcastContextBuilder::new(model)
        .with_memory_limit(16 * 1024)
        .build();
    let padding = "x".repeat(1000);
    let rows: Vec<String> = (1..=100)
        .map(|id| format!("({id}, 'sync {id} {padding}')"))
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES {}) AS t(meeting_id, transcript)",
        rows.join(", "),
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();

    let streaming = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'",
    )
    .await
    .unwrap()
    .collect()
    .await;
    assert!(
        streaming.is_ok(),
        "uncalibrated scan streams: {streaming:?}"
    );

    let error = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync' WITH RECALL 0.9",
    )
    .await
    .unwrap()
    .collect()
    .await
    .expect_err("100 KB of input cannot fit a 16 KB pool");
    assert!(
        error.to_string().contains("Resources exhausted"),
        "error: {error}",
    );
}

/// 40 "sync N" matches and 10 digit-only negatives, plus `extra` rows, under
/// theme embedding: "sync" docs score 1.0 against 'sync', everything else
/// 0.0. Small enough that the 64-document calibration sample is all of it.
async fn failure_context(model: Arc<MockModel>, extra: &[&str]) -> SessionContext {
    let ctx = semcast_context(model);
    let rows: Vec<String> = (1..=40)
        .map(|id| format!("({id}, 'sync {id}')"))
        .chain((41..=50).map(|id| format!("({id}, '{:04}')", id * 7)))
        .chain(
            extra
                .iter()
                .enumerate()
                .map(|(i, text)| format!("({}, '{text}')", 51 + i)),
        )
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES {}) AS t(meeting_id, transcript)",
        rows.join(", "),
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    ctx
}

/// Runs the calibrated query and returns the scan's metrics.
async fn calibrated_scan_metrics(
    ctx: &SessionContext,
) -> datafusion::physical_plan::metrics::MetricsSet {
    let plan = semcast::sql(
        ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync' WITH RECALL 0.9",
    )
    .await
    .unwrap()
    .create_physical_plan()
    .await
    .unwrap();
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();
    find_node(&plan, "IndexScanExec")
        .expect("the funnel planned a scan")
        .metrics()
        .unwrap()
}

/// A failed label the floor keeps anyway is no miss. The model fails on 11
/// matches that all score 1.0: the floor keeps them, so the worst case for
/// them is a kept non-match, and the index still prunes every negative.
#[tokio::test]
async fn failed_labels_the_floor_keeps_do_not_stop_pruning() {
    let model = Arc::new(
        MockModel::answering_yes_to(["sync"])
            .embedding_by_theme(["sync"])
            .failing_on(["sync 1"]),
    );
    let ctx = failure_context(model, &[]).await;
    let metrics = calibrated_scan_metrics(&ctx).await;
    assert_eq!(
        metrics
            .sum_by_name("calibration_label_errors")
            .unwrap()
            .as_usize(),
        11,
    );
    // Every negative is dropped: by the floor, or by the full-text label
    // calibration just paid for, which decides the row first.
    let pruned = metrics.sum_by_name("rows_pruned").unwrap().as_usize()
        + metrics.sum_by_name("store_pruned").unwrap().as_usize();
    assert_eq!(pruned, 10);
}

/// A failed label below the floor is a possible lost match. Six documents
/// the model fails on score 0.0, beside 40 known matches at 1.0. Dropped from
/// the sample (the old behaviour), they were invisible: the floor sat at 1.0
/// and pruned them with the negatives. Counted at worst as lost matches,
/// they cap recall at 40/46 under a 1.0 floor, so the floor drops to keep
/// them and nothing is pruned.
#[tokio::test]
async fn failed_labels_below_the_floor_stop_pruning() {
    let model = Arc::new(
        MockModel::answering_yes_to(["sync"])
            .embedding_by_theme(["sync"])
            .failing_on(["SYNC"]),
    );
    let extra = ["SYNC a", "SYNC b", "SYNC c", "SYNC d", "SYNC e", "SYNC f"];
    let ctx = failure_context(model, &extra).await;
    let metrics = calibrated_scan_metrics(&ctx).await;
    assert_eq!(
        metrics
            .sum_by_name("calibration_label_errors")
            .unwrap()
            .as_usize(),
        6,
    );
    assert_eq!(
        metrics.sum_by_name("rows_pruned").unwrap().as_usize(),
        0,
        "the floor drops below the failed labels, keeping every row",
    );
}

/// The first node named `name` in `plan`, depth first.
fn find_node(
    plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    name: &str,
) -> Option<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
    if plan.name() == name {
        return Some(Arc::clone(plan));
    }
    plan.children()
        .into_iter()
        .find_map(|child| find_node(child, name))
}

#[tokio::test]
async fn a_chunk_store_keeps_ddl_created_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.sqlite");
    let store = semcast::index::stored::SqliteChunkStore::open(&path).unwrap();
    let ctx = semcast::SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["sync"])))
        .with_chunk_store(Arc::new(store))
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

    // Another connection to the same file sees the index's documents.
    use semcast::index::stored::ChunkStore;
    let reopened = semcast::index::stored::SqliteChunkStore::open(&path).unwrap();
    let docs = reopened.doc_hashes("meetings(transcript)").unwrap();
    assert_eq!(docs, HashSet::from([doc_hash(MATCHING)]));
}

#[tokio::test]
async fn builder_embedder_serves_ddl_created_indexes() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let embedder = Arc::new(MockModel::default());
    let ctx = semcast::SemcastContextBuilder::new(Arc::clone(&model) as Arc<dyn ModelProvider>)
        .with_embedder(Arc::clone(&embedder) as Arc<dyn ModelProvider>)
        .build();
    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES (1, 'sync')) AS t(meeting_id, transcript)",
    )
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
    assert_eq!(
        matching_ids(
            &ctx,
            "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'"
        )
        .await,
        vec![1]
    );

    assert!(
        embedder.embed_calls() > 0,
        "indexing and search embed through the builder's embedder",
    );
    assert_eq!(
        model.embed_calls(),
        0,
        "the session model only verifies, it never embeds",
    );
    assert!(
        model.completion_calls() > 0,
        "verify calls stay on the session model",
    );
}

/// Depth of the first line containing `needle` in an indented plan display —
/// deeper means closer to the scan, which means it runs earlier.
fn depth_of(display: &str, needle: &str) -> usize {
    display
        .lines()
        .find(|line| line.contains(needle))
        .map(|line| line.len() - line.trim_start().len())
        .unwrap_or_else(|| panic!("no line matching {needle:?} in:\n{display}"))
}

/// Both a title and a transcript can match, so a two-predicate query returns
/// a row rather than trivially nothing.
async fn indexed_transcript_context() -> SessionContext {
    let ctx = semcast_context(Arc::new(MockModel::answering_yes_to([
        "offline sync",
        "atlas planning",
    ])));
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
    // Only `transcript` gets an index; `title` pays full price.
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    ctx
}

const INDEXED_LAST: &str = "SELECT meeting_id FROM meetings
     WHERE title MEANS 'planning' AND transcript MEANS 'sync'";
const INDEXED_FIRST: &str = "SELECT meeting_id FROM meetings
     WHERE transcript MEANS 'sync' AND title MEANS 'planning'";

/// An indexed predicate prunes for free before the model sees a row, so it
/// belongs below one that pays full price — whichever order it was written in.
#[tokio::test]
async fn an_indexed_predicate_is_stacked_below_an_unindexed_one() {
    for (label, query) in [
        ("indexed written last", INDEXED_LAST),
        ("indexed written first", INDEXED_FIRST),
    ] {
        let ctx = indexed_transcript_context().await;
        let plan = semcast::sql(&ctx, query)
            .await
            .unwrap()
            .into_optimized_plan()
            .unwrap();
        let display = plan.display_indent().to_string();

        assert!(
            depth_of(&display, "MEANS('sync')") > depth_of(&display, "MEANS('planning')"),
            "{label}: the indexed predicate must run first:\n{display}",
        );
    }
}

/// Ordering is a cost decision, not a semantic one: same rows either way.
#[tokio::test]
async fn conjunct_ordering_does_not_change_the_rows() {
    let mut answers = Vec::new();
    for query in [INDEXED_LAST, INDEXED_FIRST] {
        let ctx = indexed_transcript_context().await;
        answers.push(matching_ids(&ctx, query).await);
    }
    assert_eq!(answers[0], vec![1], "the one meeting matching both");
    assert_eq!(answers[0], answers[1]);
}

// ---------------------------------------------------------------------------
// WITH CONFIDENCE: the recall target becomes a certified lower bound rather
// than a point estimate on the calibration sample.

async fn plan_error(ctx: &SessionContext, sql: &str) -> String {
    match semcast::sql(ctx, sql).await {
        Ok(frame) => match frame.into_optimized_plan() {
            Ok(plan) => panic!("expected an error, got a plan:\n{}", plan.display_indent()),
            Err(err) => err.to_string(),
        },
        Err(err) => err.to_string(),
    }
}

/// A confidence is a claim *about* a recall target; on its own there is
/// nothing for it to qualify.
#[tokio::test]
async fn confidence_without_recall_is_a_plan_time_error() {
    let ctx = meetings_context().await;
    let message = plan_error(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE transcript MEANS 'sync' WITH CONFIDENCE 0.95",
    )
    .await;
    assert!(
        message.contains("certifies a WITH RECALL target"),
        "got: {message}",
    );
}

/// No finite sample proves that *every* match survives, so asking to certify
/// 1.0 is refused rather than quietly certified.
#[tokio::test]
async fn certifying_full_recall_is_refused() {
    let ctx = meetings_context().await;
    let message = plan_error(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE transcript MEANS 'sync' WITH RECALL 1 WITH CONFIDENCE 0.95",
    )
    .await;
    assert!(message.contains("cannot be certified"), "got: {message}");
    // Without the confidence it is still a legal, best-effort request.
    assert!(
        semcast::sql(
            &ctx,
            "SELECT meeting_id FROM meetings
             WHERE transcript MEANS 'sync' WITH RECALL 1",
        )
        .await
        .unwrap()
        .into_optimized_plan()
        .is_ok(),
    );
}

/// EXPLAIN says which kind of promise the plan makes, because "recall ≥ 0.9"
/// means something different with and without a confidence behind it.
#[tokio::test]
async fn explain_distinguishes_an_estimate_from_a_certificate() {
    let ctx = meetings_context().await;
    for (query, expected) in [
        (
            "SELECT meeting_id FROM meetings
             WHERE transcript MEANS 'sync' WITH RECALL 0.9",
            "recall ≥ 0.90 estimated",
        ),
        (
            "SELECT meeting_id FROM meetings
             WHERE transcript MEANS 'sync' WITH RECALL 0.9 WITH CONFIDENCE 0.95",
            "recall ≥ 0.90 certified at 0.95",
        ),
    ] {
        let plan = semcast::sql(&ctx, query)
            .await
            .unwrap()
            .into_optimized_plan()
            .unwrap();
        let display = plan.display_indent().to_string();
        assert!(
            display.contains(expected),
            "want {expected:?} in:\n{display}"
        );
    }
}

/// Certifying must never drop a row an estimate would have kept: the bound is
/// more conservative, so the funnel is wider, never narrower.
#[tokio::test]
async fn certifying_keeps_at_least_what_estimating_keeps() {
    let query = "SELECT meeting_id FROM meetings
                 WHERE transcript MEANS 'offline sync' WITH RECALL 0.9";
    let certified_query = "SELECT meeting_id FROM meetings
                           WHERE transcript MEANS 'offline sync'
                           WITH RECALL 0.9 WITH CONFIDENCE 0.95";

    let mut kept = Vec::new();
    for sql in [query, certified_query] {
        let ctx = meetings_context().await;
        create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
            .await
            .unwrap();
        kept.push(matching_ids(&ctx, sql).await);
    }
    assert_eq!(kept[0], vec![1]);
    assert_eq!(kept[1], vec![1], "the certified funnel is at least as wide");
}

/// The calibrated floor itself, on rows the sample never labels: 200
/// documents against a 64-document sample, so most rows are decided by the
/// floor alone. Matches embed at 1.0 against 'sync'; near-misses (which the
/// model rejects) at ~0.71, above the default 0.35 cosine floor; plain
/// negatives at 0.0. The scan must prune the unsampled near-misses as well
/// as the negatives, without labels for them.
///
/// With vectors to fit, calibration thresholds a learned proxy's
/// probability, which separates near-misses on its own, so this does not
/// isolate the floor being *raised*; `tests/proxy.rs`
/// (`a_one_class_sample_falls_back_and_stays_correct`) pins that, on cosine.
#[tokio::test]
async fn the_calibrated_floor_prunes_rows_outside_the_sample() {
    const MATCHES: i64 = 30;
    const NEAR_MISSES: i64 = 100;
    const NEGATIVES: i64 = 70;
    let model =
        Arc::new(MockModel::answering_yes_to(["match"]).embedding_by_theme(["sync", "noise"]));
    let ctx = semcast_context(Arc::clone(&model) as _);
    let rows: Vec<String> = (1..=MATCHES)
        .map(|id| format!("({id}, 'sync match {id}')"))
        .chain((1..=NEAR_MISSES).map(|n| format!("({}, 'sync noise {n}')", MATCHES + n)))
        .chain((1..=NEGATIVES).map(|n| format!("({}, '{:04}')", MATCHES + NEAR_MISSES + n, n * 7)))
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE meetings AS SELECT * FROM (VALUES {}) AS t(meeting_id, transcript)",
        rows.join(", "),
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();

    let metrics = calibrated_scan_metrics(&ctx).await;
    let by_floor = metrics.sum_by_name("rows_pruned").unwrap().as_usize();
    let by_label = metrics.sum_by_name("store_pruned").unwrap().as_usize();
    assert_eq!(
        by_floor + by_label,
        (NEAR_MISSES + NEGATIVES) as usize,
        "every non-match is dropped"
    );
    // The 0.0 negatives alone are at most 70: pruning more means the scan
    // also dropped unsampled near-misses, which score above the default floor.
    assert!(
        by_floor > NEGATIVES as usize,
        "the calibrated scan must prune unsampled near-misses: {by_floor} by \
         the scan, {by_label} by their labels"
    );
    // Only the sample is labelled and only unsampled matches are verified:
    // nowhere near one call per row.
    assert!(model.completion_calls() < 64 + MATCHES as usize);
    assert_eq!(
        matching_ids(&ctx, CALIBRATED_QUERY).await,
        (1..=MATCHES).collect::<Vec<_>>()
    );
}

/// Vectors are `EMBED` answers, kept by embedder and chunk text: re-creating
/// an index over the same text reads them back instead of embedding again.
#[tokio::test]
async fn recreating_an_index_reuses_stored_vectors() {
    let model = Arc::new(MockModel::answering_yes_to(["offline sync"]));
    let ctx = semcast_context(Arc::clone(&model) as _);
    ctx.sql(&format!(
        "CREATE TABLE meetings AS SELECT * FROM (VALUES (1, '{MATCHING}'), (2, '{OTHER}')) \
         AS t(meeting_id, transcript)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    let first = model.embed_calls();
    assert!(first > 0);
    let index = create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    assert_eq!(
        model.embed_calls(),
        first + 1,
        "only the dimension probe: no chunk embedded twice"
    );
    assert_eq!(index.indexed_doc_hashes().await.unwrap().len(), 2);
}

/// An index saved by a build that kept it in Lance has a dataset path and no
/// chunking fields. Restoring it rebuilds it from its column.
#[tokio::test]
async fn an_index_saved_by_a_lance_build_is_rebuilt() {
    let model = Arc::new(MockModel::answering_yes_to(["offline sync"]));
    let ctx = semcast_context(Arc::clone(&model) as _);
    ctx.sql(&format!(
        "CREATE TABLE meetings AS SELECT * FROM (VALUES (1, '{MATCHING}'), (2, '{OTHER}')) \
         AS t(meeting_id, transcript)",
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let definition = r#"{"table":"meetings","column":"transcript",
        "path":"/gone/meetings.transcript.lance",
        "fetch_k":256,"score_floor":0.35,"chunks_per_doc":3}"#;
    let index = semcast::reattach_semantic_index(&ctx, definition)
        .await
        .unwrap();
    assert!(model.embed_calls() > 0, "rebuilt by embedding its column");
    assert_eq!(index.indexed_doc_hashes().await.unwrap().len(), 2);
    let plan = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'",
    )
    .await
    .unwrap()
    .create_physical_plan()
    .await
    .unwrap();
    let display = displayable(plan.as_ref()).indent(true).to_string();
    assert!(display.contains("IndexScanExec"), "{display}");
}

/// An embedder whose vector size can change under the same model id, as a
/// re-pulled tag or a fallback chain can, and that can be made to fail.
#[derive(Debug)]
struct Shifting {
    dim: std::sync::atomic::AtomicUsize,
    failing: std::sync::atomic::AtomicBool,
    calls: std::sync::atomic::AtomicUsize,
}

impl Shifting {
    fn new(dim: usize) -> Arc<Self> {
        Arc::new(Self {
            dim: std::sync::atomic::AtomicUsize::new(dim),
            failing: std::sync::atomic::AtomicBool::new(false),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Shifting {
    fn id(&self) -> ModelId {
        ModelId("shifting".to_owned())
    }
    async fn complete(
        &self,
        requests: Vec<semcast::model::CompletionRequest>,
    ) -> Vec<semcast::Result<semcast::model::Completion>> {
        MockModel::answering_yes_to(["sync"])
            .complete(requests)
            .await
    }
    async fn embed(&self, texts: Vec<String>) -> semcast::Result<Vec<semcast::model::Embedding>> {
        use std::sync::atomic::Ordering;
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Fails only on the new text, so a re-create gets past its probe and
        // fails mid-build.
        if self.failing.load(Ordering::SeqCst) && texts.iter().any(|t| t.contains("again")) {
            return Err(semcast::SemcastError::Index("embedder down".to_owned()));
        }
        let dim = self.dim.load(Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.1; dim];
                v[0] = if text.contains("sync") { 1.0 } else { 0.0 };
                v
            })
            .collect())
    }
}

async fn shifting_context(embedder: Arc<Shifting>) -> SessionContext {
    let ctx = semcast_context(embedder as _);
    ctx.sql(
        "CREATE TABLE meetings AS SELECT * FROM (VALUES (1, 'sync now'), (2, 'sync later'), \
         (3, 'lunch')) AS t(meeting_id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

/// A vector size change under the same model id is loud, never a silently
/// empty result, and re-creating the index embeds afresh rather than reading
/// back vectors of the old size.
#[tokio::test]
async fn a_dimension_change_fails_loudly_and_a_recreate_heals_it() {
    use std::sync::atomic::Ordering;
    let embedder = Shifting::new(4);
    let ctx = shifting_context(Arc::clone(&embedder)).await;
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    let query = "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'";
    assert_eq!(matching_ids(&ctx, query).await, [1, 2]);

    embedder.dim.store(5, Ordering::SeqCst);
    let error = semcast::sql(&ctx, query)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("dimensional query vector"), "{error}");

    let before = embedder.calls.load(Ordering::SeqCst);
    create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    assert!(
        embedder.calls.load(Ordering::SeqCst) > before + 1,
        "the chunks are embedded again at the new size"
    );
    assert_eq!(matching_ids(&ctx, query).await, [1, 2]);
}

/// A re-create that fails keeps the index it was replacing.
#[tokio::test]
async fn a_failed_recreate_keeps_the_old_index() {
    use std::sync::atomic::Ordering;
    let embedder = Shifting::new(4);
    let ctx = shifting_context(Arc::clone(&embedder)).await;
    let index = create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
        .await
        .unwrap();
    assert_eq!(index.indexed_doc_hashes().await.unwrap().len(), 3);

    // New text the embedder can't reach.
    ctx.sql("INSERT INTO meetings VALUES (4, 'sync again')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    embedder.failing.store(true, Ordering::SeqCst);
    assert!(
        create_semantic_index(&ctx, "meetings", "transcript", IndexOptions::default())
            .await
            .is_err()
    );
    assert_eq!(
        index.indexed_doc_hashes().await.unwrap().len(),
        3,
        "the old chunks still serve"
    );
}
