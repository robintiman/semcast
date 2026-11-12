//! Semantic columns: `ALTER TABLE t ADD SEMANTIC COLUMN name AS (text MEANS 'c')`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use semcast::SemcastContextBuilder;
use semcast::model::MockModel;

async fn tickets(model: Arc<MockModel>) -> SessionContext {
    let ctx = SemcastContextBuilder::new(model).build();
    ctx.sql(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES
             (1, 'I am furious about the outage'),
             (2, 'thanks, all good now'),
             (3, CAST(NULL AS VARCHAR))
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

async fn error(ctx: &SessionContext, sql: &str) -> String {
    match semcast::sql(ctx, sql).await {
        Err(e) => e.to_string(),
        Ok(df) => df.collect().await.expect_err(sql).to_string(),
    }
}

fn ids(batches: &[RecordBatch]) -> Vec<i64> {
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

fn flags(batches: &[RecordBatch], column: usize) -> Vec<Option<bool>> {
    batches
        .iter()
        .flat_map(|b| {
            let column = b
                .column(column)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("a semantic column is a boolean");
            column.iter().collect::<Vec<_>>()
        })
        .collect()
}

const ADD: &str =
    "ALTER TABLE tickets ADD SEMANTIC COLUMN angry AS (body MEANS 'the customer is angry')";

#[tokio::test]
async fn a_semantic_column_reads_like_a_column_and_pays_once() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(&ctx, ADD).await;
    assert_eq!(model.completion_calls(), 0, "adding a column asks nothing");

    let batches = run(&ctx, "SELECT id, angry FROM tickets ORDER BY id").await;
    assert_eq!(flags(&batches, 1), [Some(true), Some(false), None]);
    assert_eq!(batches[0].schema().field(1).name(), "angry");
    assert_eq!(model.completion_calls(), 2, "NULL text costs nothing");

    run(&ctx, "SELECT * FROM tickets").await;
    assert_eq!(model.completion_calls(), 2, "stored answers are free");
}

#[tokio::test]
async fn a_semantic_column_filters_and_shares_answers_with_means() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(&ctx, ADD).await;
    assert_eq!(
        ids(&run(&ctx, "SELECT id FROM tickets WHERE angry").await),
        [1]
    );
    assert_eq!(
        ids(&run(&ctx, "SELECT id FROM tickets WHERE NOT angry").await),
        [2]
    );
    assert_eq!(
        ids(&run(
            &ctx,
            "SELECT id FROM tickets WHERE body MEANS 'the customer is angry'"
        )
        .await),
        [1]
    );
    assert_eq!(
        model.completion_calls(),
        2,
        "one question, asked once per row"
    );
}

#[tokio::test]
async fn refresh_pays_ahead_of_time() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(&ctx, ADD).await;
    run(&ctx, "REFRESH SEMANTIC COLUMN tickets(angry)").await;
    assert_eq!(model.completion_calls(), 2);
    run(&ctx, "SELECT id FROM tickets WHERE angry").await;
    assert_eq!(model.completion_calls(), 2);
}

#[tokio::test]
async fn dropping_a_semantic_column_removes_it() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(&ctx, ADD).await;
    run(&ctx, "ALTER TABLE tickets DROP SEMANTIC COLUMN angry").await;
    let message = error(&ctx, "SELECT angry FROM tickets").await;
    assert!(message.contains("angry"), "{message}");
    assert_eq!(
        ids(&run(&ctx, "SELECT id FROM tickets").await),
        [1, 2, 3],
        "the table is as it was"
    );
    run(
        &ctx,
        "ALTER TABLE tickets DROP SEMANTIC COLUMN IF EXISTS angry",
    )
    .await;
    let message = error(&ctx, "ALTER TABLE tickets DROP SEMANTIC COLUMN angry").await;
    assert!(message.contains("no semantic column"), "{message}");
}

#[tokio::test]
async fn several_semantic_columns_on_one_table() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(&ctx, ADD).await;
    run(
        &ctx,
        "ALTER TABLE tickets ADD SEMANTIC COLUMN refund AS (body MEANS 'asks for a refund')",
    )
    .await;
    let batches = run(&ctx, "SELECT id, angry, refund FROM tickets ORDER BY id").await;
    assert_eq!(flags(&batches, 1), [Some(true), Some(false), None]);
    // The mock says yes to any condition on the furious ticket.
    assert_eq!(flags(&batches, 2), [Some(true), Some(false), None]);
    run(&ctx, "ALTER TABLE tickets DROP SEMANTIC COLUMN angry").await;
    let batches = run(&ctx, "SELECT id, refund FROM tickets ORDER BY id").await;
    assert_eq!(flags(&batches, 1), [Some(true), Some(false), None]);
}

#[tokio::test]
async fn a_semantic_column_must_read_text_and_must_not_shadow() {
    let ctx = tickets(Arc::new(MockModel::default())).await;
    let message = error(
        &ctx,
        "ALTER TABLE tickets ADD SEMANTIC COLUMN body AS (body MEANS 'x')",
    )
    .await;
    assert!(
        message.contains("already has a column named body"),
        "{message}"
    );
    let message = error(
        &ctx,
        "ALTER TABLE tickets ADD SEMANTIC COLUMN x AS (id MEANS 'x')",
    )
    .await;
    assert!(message.contains("reads text"), "{message}");
    let message = error(
        &ctx,
        "ALTER TABLE tickets ADD SEMANTIC COLUMN x AS (missing MEANS 'x')",
    )
    .await;
    assert!(message.contains("no column named missing"), "{message}");
}

#[tokio::test]
async fn other_alter_table_statements_are_still_sql() {
    let ctx = tickets(Arc::new(MockModel::default())).await;
    // DataFusion doesn't support this one; the point is that the error is
    // DataFusion's, not the semantic column grammar's.
    let message = error(&ctx, "ALTER TABLE tickets ADD COLUMN x INT").await;
    assert!(!message.contains("semantic column statement"), "{message}");
}

#[tokio::test]
async fn a_semantic_column_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("tickets.csv");
    std::fs::write(&csv, "id,body\n1,I am furious\n2,all good\n").unwrap();
    let store = dir.path().join("store.db");
    let session = |model: Arc<MockModel>| {
        let store = store.clone();
        async move {
            let cache: Arc<dyn semcast::cache::SemanticCache> =
                Arc::new(semcast::cache::sqlite::SqliteCache::open(&store).unwrap());
            let ctx = SemcastContextBuilder::new(model)
                .with_cache(cache)
                .with_catalog(semcast::catalog::Catalog::open(&store).unwrap())
                .build();
            let restored = semcast::restore_catalog(&ctx).await.unwrap();
            assert!(restored.failed.is_empty(), "{:?}", restored.failed);
            ctx
        }
    };
    {
        let model = Arc::new(MockModel::answering_yes_to(["furious"]));
        let ctx = session(Arc::clone(&model)).await;
        run(
            &ctx,
            &format!(
                "CREATE EXTERNAL TABLE tickets STORED AS CSV LOCATION '{}' \
                 OPTIONS ('format.has_header' 'true')",
                csv.display()
            ),
        )
        .await;
        run(&ctx, ADD).await;
        assert_eq!(
            ids(&run(&ctx, "SELECT id FROM tickets WHERE angry").await),
            [1]
        );
    }
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = session(Arc::clone(&model)).await;
    assert_eq!(
        ids(&run(&ctx, "SELECT id FROM tickets WHERE angry").await),
        [1]
    );
    assert_eq!(
        model.completion_calls(),
        0,
        "the column and its answers came back"
    );

    // Dropping the table forgets the column too.
    run(&ctx, "DROP TABLE tickets").await;
    let ctx = session(Arc::new(MockModel::default())).await;
    let message = error(&ctx, "SELECT * FROM tickets").await;
    assert!(message.contains("tickets"), "{message}");
}

/// `WHERE angry` is the same `MEANS` filter `WHERE body MEANS '...'` is —
/// store lookup, and the index when there is one — not a label computed for
/// every row and filtered afterwards.
#[tokio::test]
async fn filtering_on_a_semantic_column_plans_a_means_filter() {
    let ctx = tickets(Arc::new(MockModel::answering_yes_to(["furious"]))).await;
    run(&ctx, ADD).await;
    let plan = semcast::sql(&ctx, "SELECT id FROM tickets WHERE angry AND id > 0")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap()
        .display_indent()
        .to_string();
    assert!(
        plan.contains("SemFilter: MEANS('the customer is angry')"),
        "{plan}"
    );
    assert!(!plan.contains("SemClassify"), "{plan}");
}
