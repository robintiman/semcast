//! The replay provider reruns a recorded funnel exactly: same rows, same
//! requests, none missing from the recording.

use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use semcast::cache::{InMemoryCache, SemanticCache};
use semcast::model::{MockModel, ModelProvider, ReplayProvider};
use semcast::{IndexOptions, SemcastContextBuilder, create_semantic_index};

/// Long and short documents, a third matching, so calibration labels a
/// sample, checks chunks and prunes, and verify reads what survives.
async fn contracts(ctx: &SessionContext) {
    let rows: Vec<String> = (1..=60)
        .map(|id| {
            let filler = "the parties agree to the terms herein ".repeat(id * 7);
            let body = if id % 3 == 0 {
                format!("{filler} we agreed to ship offline sync {filler}")
            } else {
                format!("{filler} nothing notable")
            };
            format!("({id}, '{body}')")
        })
        .collect();
    let sql = format!(
        "CREATE TABLE contracts AS SELECT * FROM (VALUES {}) AS t(id, body)",
        rows.join(", ")
    );
    ctx.sql(&sql).await.unwrap().collect().await.unwrap();
    create_semantic_index(ctx, "contracts", "body", IndexOptions::default())
        .await
        .unwrap();
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

const QUERY: &str = "SELECT id FROM contracts WHERE body MEANS 'offline sync' \
                     WITH RECALL 0.9 WITH CONFIDENCE 0.8";

#[tokio::test]
async fn a_replay_reruns_the_recorded_funnel_exactly() {
    let model = Arc::new(MockModel::answering_yes_to(["offline sync"]));
    let recording = Arc::new(InMemoryCache::default());
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _)
        .with_cache(Arc::clone(&recording) as Arc<dyn SemanticCache>)
        .build();
    contracts(&ctx).await;
    let recorded = ids(&ctx, QUERY).await;
    let asked = model.completion_calls();

    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("replay.jsonl");
    let replay = Arc::new(
        ReplayProvider::new(recording, model.id().0)
            .with_log(&log)
            .unwrap(),
    );
    // A fresh session store, so the funnel runs as it did cold.
    let embedder = Arc::new(MockModel::answering_yes_to(["offline sync"]));
    let ctx = SemcastContextBuilder::new(Arc::clone(&replay) as _)
        .with_embedder(embedder)
        .build();
    contracts(&ctx).await;
    assert_eq!(ids(&ctx, QUERY).await, recorded);
    assert_eq!(replay.misses(), 0);
    let log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        log.lines().count(),
        asked,
        "the replay asked what the recording paid for"
    );
    assert!(
        log.contains("\"kind\":\"chunked\""),
        "chunk checks were replayed"
    );
    assert!(
        log.contains("\"kind\":\"full\""),
        "full-text labels were replayed"
    );
}
