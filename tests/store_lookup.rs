//! Store lookup: the first stage of every `MEANS` filter settles each row
//! whose question the store has already answered.

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::displayable;
use semcast::cache::sqlite::SqliteCache;
use semcast::cache::{CachedValue, SemanticCache};
use semcast::index::SearchParams;
use semcast::model::MockModel;
use semcast::{IndexOptions, SemcastContextBuilder, create_semantic_index};

const QUERY: &str = "SELECT meeting_id FROM meetings WHERE transcript MEANS 'sync'";

/// Meeting 2 never mentions "sync", so its embedding is far from the
/// condition, but the model says it matches.
async fn meetings(model: Arc<MockModel>, cache: Option<Arc<dyn SemanticCache>>) -> SessionContext {
    let mut builder = SemcastContextBuilder::new(model);
    if let Some(cache) = cache {
        builder = builder.with_cache(cache);
    }
    let ctx = builder.build();
    ctx.sql(
        "CREATE TABLE meetings AS SELECT * FROM (VALUES
             (1, 'sync'),
             (2, 'abcdefghijkl'),
             (3, CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

async fn ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
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
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    ids.sort_unstable();
    ids
}

async fn index_with_floor(ctx: &SessionContext, floor: f32) {
    create_semantic_index(
        ctx,
        "meetings",
        "transcript",
        IndexOptions {
            search: SearchParams {
                score_floor: floor,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn a_stored_match_survives_an_index_that_would_prune_it() {
    let model = Arc::new(MockModel::answering_yes_to(["sync", "abcdefghijkl"]));
    let ctx = meetings(Arc::clone(&model), None).await;
    // Answered once without an index: both meetings match.
    assert_eq!(ids(&ctx, QUERY).await, [1, 2]);
    assert_eq!(model.completion_calls(), 2);

    // An index whose floor prunes meeting 2 on similarity alone.
    index_with_floor(&ctx, 0.9).await;
    assert_eq!(
        ids(&ctx, QUERY).await,
        [1, 2],
        "the stored answer beats the similarity score"
    );
    assert_eq!(model.completion_calls(), 2, "no call for a known row");

    // Without the stored answer, the same index does prune meeting 2.
    let fresh = Arc::new(MockModel::answering_yes_to(["sync", "abcdefghijkl"]));
    let ctx = meetings(Arc::clone(&fresh), None).await;
    index_with_floor(&ctx, 0.9).await;
    assert_eq!(ids(&ctx, QUERY).await, [1]);
}

#[tokio::test]
async fn a_stored_non_match_costs_nothing_downstream() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = meetings(Arc::clone(&model), None).await;
    assert_eq!(ids(&ctx, QUERY).await, [1]);
    assert_eq!(model.completion_calls(), 2);
    // An index that keeps everything: verify would read meeting 2's chunks
    // and ask again under the chunked prompt, but its full-text "no" is known.
    index_with_floor(&ctx, f32::NEG_INFINITY).await;
    assert_eq!(ids(&ctx, QUERY).await, [1]);
    assert_eq!(model.completion_calls(), 2);
}

#[tokio::test]
async fn explain_names_the_question_and_progress_counts_the_store() {
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let ctx = meetings(Arc::clone(&model), None).await;
    let df = semcast::sql(&ctx, QUERY).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let text = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        text.contains("StoreLookupExec: ask=") && text.contains("MEANS('sync')"),
        "{text}"
    );

    ids(&ctx, QUERY).await;
    let plan = semcast::sql(&ctx, QUERY)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();
    let counts = semcast::server::progress::snapshot(&plan);
    assert_eq!(counts.store_rows, 2);
    assert_eq!(counts.store_matches, 1);
    assert_eq!(counts.store_non_matches, 1);
    assert_eq!(model.completion_calls(), 2, "the second run asked nothing");
}

#[tokio::test]
async fn the_store_answers_across_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.sqlite");
    {
        let model = Arc::new(MockModel::answering_yes_to(["sync"]));
        let store: Arc<dyn SemanticCache> = Arc::new(SqliteCache::open(&path).unwrap());
        let ctx = meetings(Arc::clone(&model), Some(store)).await;
        assert_eq!(ids(&ctx, QUERY).await, [1]);
    }
    let model = Arc::new(MockModel::answering_yes_to(["sync"]));
    let store: Arc<dyn SemanticCache> = Arc::new(SqliteCache::open(&path).unwrap());
    let ctx = meetings(Arc::clone(&model), Some(store)).await;
    assert_eq!(ids(&ctx, QUERY).await, [1]);
    assert_eq!(model.completion_calls(), 0);
}

/// `get_many` answers in the order asked, misses included, from what this
/// instance holds and from disk alike.
#[tokio::test]
async fn get_many_reads_in_order_across_memory_and_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.sqlite");
    let key = |input: &str| semcast::cache::CacheKey {
        type_version: "is angry".to_owned(),
        field: "means".to_owned(),
        input_digest: semcast::cache::stable_digest(&[input.as_bytes()]),
        model_id: semcast::model::ModelId("m".to_owned()),
        prompt_version: "means-v2".to_owned(),
    };
    {
        let store = SqliteCache::open(&path).unwrap();
        store.put(key("a"), CachedValue::Value("yes".to_owned()));
        store.put(key("b"), CachedValue::Value("no".to_owned()));
        store.flush();
    }
    let store = SqliteCache::open(&path).unwrap();
    store.put(key("c"), CachedValue::Value("yes".to_owned()));
    let answers = store.get_many(&[key("b"), key("missing"), key("c"), key("a")]);
    assert_eq!(
        answers,
        [
            Some(CachedValue::Value("no".to_owned())),
            None,
            Some(CachedValue::Value("yes".to_owned())),
            Some(CachedValue::Value("yes".to_owned())),
        ]
    );
}

/// A calibration label is a full-text verdict bought during the query; verify
/// must not ask the weaker chunked question for the same document, and a
/// repeat query must return the same rows.
#[tokio::test]
async fn calibration_labels_decide_their_rows_in_the_same_query() {
    let model = Arc::new(MockModel::answering_yes_to(["qqqq"]));
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
    ctx.sql(
        "CREATE TABLE docs AS SELECT * FROM (VALUES
             (1, 'sync sync sync qqqq qqqx qqqy'),
             (2, 'filler words here')
         ) AS t(id, body)",
    )
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
            chunk: semcast::index::chunk::ChunkConfig {
                max_tokens: 4,
                overlap_tokens: 0,
            },
            search: SearchParams {
                score_floor: f32::NEG_INFINITY,
                chunks_per_doc: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let query = "SELECT id FROM docs WHERE body MEANS 'sync' WITH RECALL 0.9";
    assert_eq!(ids(&ctx, query).await, [1], "the full-text label wins");
    let calls = model.completion_calls();
    assert_eq!(calls, 2, "two labels, no chunked call");
    assert_eq!(ids(&ctx, query).await, [1], "the same rows again");
    assert_eq!(model.completion_calls(), calls);
}
