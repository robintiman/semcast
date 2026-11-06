//! The durable verdict store: a query answered once costs no model calls
//! after a restart, because the verdicts outlive the process.

use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use semcast::SemcastContextBuilder;
use semcast::cache::sqlite::SqliteCache;
use semcast::model::MockModel;

const QUERIES: [&str; 2] = [
    "SELECT id FROM tickets WHERE body MEANS 'the customer is angry' ORDER BY id",
    "SELECT id, CASE WHEN body MEANS 'the customer is angry' THEN 'angry' ELSE 'calm' END \
     AS mood FROM tickets ORDER BY id",
];

/// A fresh process, as far as semcast can tell: new model, new context, new
/// cache handle — only the store file carries over.
async fn session(model: Arc<MockModel>, store: &Path) -> SessionContext {
    let builder =
        SemcastContextBuilder::new(model).with_cache(Arc::new(SqliteCache::open(store).unwrap()));
    with_tickets(builder.build()).await
}

async fn with_tickets(ctx: SessionContext) -> SessionContext {
    ctx.sql(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES
             (1, 'I am furious, this is the third outage'),
             (2, 'thanks, all good now'),
             (3, 'furious again: refund me')
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx
}

async fn run(ctx: &SessionContext, sql: &str) -> String {
    let batches: Vec<RecordBatch> = semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

#[tokio::test]
async fn a_restart_costs_no_model_calls() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("verdicts.db");

    let first_model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let first = session(Arc::clone(&first_model), &store).await;
    let mut answers = Vec::new();
    for query in QUERIES {
        answers.push(run(&first, query).await);
    }
    assert!(
        first_model.completion_calls() > 0,
        "the first run asks the model"
    );
    drop(first);

    let second_model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let second = session(Arc::clone(&second_model), &store).await;
    for (query, answer) in QUERIES.iter().zip(&answers) {
        assert_eq!(&run(&second, query).await, answer, "{query}");
    }
    assert_eq!(
        second_model.completion_calls(),
        0,
        "every verdict comes from the store",
    );
}

#[tokio::test]
async fn a_different_model_does_not_reuse_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("verdicts.db");

    let first = session(Arc::new(MockModel::answering_yes_to(["furious"])), &store).await;
    run(&first, QUERIES[0]).await;
    drop(first);

    // Same store, same rows, different model chain: the chain's id names
    // every model in it, so the first session's verdicts are not its own.
    let other = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = with_tickets(
        SemcastContextBuilder::new(Arc::clone(&other) as Arc<dyn semcast::model::ModelProvider>)
            .with_fallback_models(vec![Arc::new(MockModel::default())])
            .with_cache(Arc::new(SqliteCache::open(&store).unwrap()))
            .build(),
    )
    .await;
    run(&ctx, QUERIES[0]).await;
    assert_eq!(other.completion_calls(), 3, "a new model chain asks again");
}
