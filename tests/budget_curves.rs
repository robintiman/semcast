//! The recall budget follows the filters' pruning curves: the loss goes to
//! the filter where giving up recall prunes the most rows.

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use semcast::cache::{InMemoryCache, SemanticCache};
use semcast::model::{MockModel, ModelId};
use semcast::physical::verify::stats_key;
use semcast::{IndexOptions, SemcastContextBuilder, create_semantic_index};

async fn context(cache: Arc<InMemoryCache>) -> SessionContext {
    let ctx = SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["refund"])))
        .with_cache(cache as _)
        .build();
    let rows: Vec<String> = (1..=40)
        .map(|id| {
            let body = if id % 4 == 0 {
                "please refund my order"
            } else {
                "just saying hi"
            };
            format!("({id}, '{body} {id}')")
        })
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES {}) AS t(id, body)",
        rows.join(", ")
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    create_semantic_index(&ctx, "tickets", "body", IndexOptions::default())
        .await
        .unwrap();
    ctx
}

fn key(condition: &str) -> u128 {
    stats_key(condition, &ModelId("mock".to_owned()), "tickets.body")
}

async fn explain(ctx: &SessionContext, sql: &str) -> String {
    let batches = semcast::sql(ctx, &format!("EXPLAIN {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

const QUERY: &str = "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' \
     AND body MEANS 'is angry' WITH RECALL 0.9";

#[tokio::test]
async fn the_loss_goes_to_the_filter_it_prunes() {
    let cache = Arc::new(InMemoryCache::default());
    // Giving up recall on 'asks for a refund' prunes nothing; on 'is angry'
    // it halves what is kept.
    cache.record_curve(
        key("asks for a refund"),
        vec![(0.8, 0.9), (0.9, 0.9), (1.0, 0.9)],
    );
    cache.record_curve(key("is angry"), vec![(0.8, 0.1), (0.9, 0.3), (1.0, 1.0)]);
    let ctx = context(cache).await;
    let plan = explain(&ctx, QUERY).await;
    assert!(
        plan.contains("SemFilter: MEANS('asks for a refund')   recall ≥ 1.00 estimated"),
        "{plan}"
    );
    assert!(
        plan.contains("SemFilter: MEANS('is angry')   recall ≥ 0.90 estimated"),
        "{plan}"
    );
}

#[tokio::test]
async fn a_filter_without_a_curve_keeps_equal_shares() {
    let cache = Arc::new(InMemoryCache::default());
    cache.record_curve(key("is angry"), vec![(0.8, 0.1), (0.9, 0.3), (1.0, 1.0)]);
    let ctx = context(cache).await;
    let plan = explain(&ctx, QUERY).await;
    assert_eq!(plan.matches("recall ≥ 0.95 estimated").count(), 2, "{plan}");
}

#[tokio::test]
async fn a_calibration_records_its_curve() {
    let cache = Arc::new(InMemoryCache::default());
    let ctx = context(Arc::clone(&cache)).await;
    semcast::sql(
        &ctx,
        "SELECT id FROM tickets WHERE body MEANS 'asks for a refund' WITH RECALL 0.9",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let curve = cache.curve(key("asks for a refund")).expect("recorded");
    assert_eq!(curve.len(), 7);
    // Asking for more recall never keeps fewer rows.
    assert!(
        curve.windows(2).all(|w| w[0].1 <= w[1].1 + 1e-12),
        "{curve:?}"
    );
    assert!(
        curve.iter().all(|(_, kept)| (0.0..=1.0).contains(kept)),
        "{curve:?}"
    );
}
