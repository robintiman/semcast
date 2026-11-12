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

#[tokio::test]
async fn writes_reach_the_table_underneath() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(&ctx, ADD).await;
    run(
        &ctx,
        "INSERT INTO tickets (id, body) VALUES (4, 'furious again')",
    )
    .await;
    run(&ctx, "DELETE FROM tickets WHERE id = 2").await;
    let calls = model.completion_calls();
    run(&ctx, "UPDATE tickets SET body = 'calm now' WHERE id = 1").await;
    assert_eq!(model.completion_calls(), calls, "writing asks nothing");
    let batches = run(&ctx, "SELECT id, angry FROM tickets ORDER BY id").await;
    assert_eq!(ids(&batches), [1, 3, 4]);
    assert_eq!(flags(&batches, 1), [Some(false), None, Some(true)]);

    for (sql, expected) in [
        (
            "DELETE FROM tickets WHERE angry",
            "cannot filter on a semantic column",
        ),
        (
            "UPDATE tickets SET body = 'x' WHERE NOT angry",
            "cannot filter on a semantic column",
        ),
        (
            "UPDATE tickets SET angry = true WHERE id = 1",
            "computed, not stored",
        ),
    ] {
        let message = error(&ctx, sql).await;
        assert!(message.contains(expected), "{sql}: {message}");
    }
    assert_eq!(ids(&run(&ctx, "SELECT id FROM tickets").await), [1, 3, 4]);
}

#[tokio::test]
async fn an_aliased_filter_keeps_the_means_funnel() {
    let ctx = tickets(Arc::new(MockModel::answering_yes_to(["furious"]))).await;
    run(&ctx, ADD).await;
    let sql = "SELECT t.id FROM tickets t WHERE t.angry";
    let plan = semcast::sql(&ctx, sql)
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap()
        .display_indent()
        .to_string();
    assert!(plan.contains("SemFilter"), "{plan}");
    assert!(!plan.contains("SemClassify"), "{plan}");
    assert_eq!(ids(&run(&ctx, sql).await), [1]);
}

#[tokio::test]
async fn a_correlated_subquery_can_read_a_semantic_column() {
    let ctx = tickets(Arc::new(MockModel::answering_yes_to(["furious"]))).await;
    run(&ctx, ADD).await;
    let batches = run(
        &ctx,
        "SELECT id FROM tickets WHERE EXISTS \
         (SELECT 1 FROM (VALUES (true)) v(x) WHERE v.x = tickets.angry)",
    )
    .await;
    assert_eq!(ids(&batches), [1]);
}

#[tokio::test]
async fn alter_table_if_exists() {
    let ctx = tickets(Arc::new(MockModel::answering_yes_to(["furious"]))).await;
    run(
        &ctx,
        "ALTER TABLE IF EXISTS nowhere ADD SEMANTIC COLUMN x AS (body MEANS 'x')",
    )
    .await;
    run(
        &ctx,
        "ALTER TABLE IF EXISTS tickets ADD SEMANTIC COLUMN angry AS (body MEANS 'the customer is angry')",
    )
    .await;
    assert_eq!(
        ids(&run(&ctx, "SELECT id FROM tickets WHERE angry").await),
        [1]
    );
    run(&ctx, "ALTER TABLE IF EXISTS nowhere DROP SEMANTIC COLUMN x").await;
}

/// The catalog keeps only what the session has: a replaced table comes back
/// without its old columns, and a column named through a qualified table
/// name is the same entry as through the bare one.
#[tokio::test]
async fn the_catalog_forgets_columns_the_session_no_longer_has() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("tickets.csv");
    std::fs::write(&csv, "id,body\n1,I am furious\n2,all good\n").unwrap();
    let store = dir.path().join("store.db");
    let session = || {
        let store = store.clone();
        async move {
            let ctx =
                SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["furious"])))
                    .with_catalog(semcast::catalog::Catalog::open(&store).unwrap())
                    .build();
            let restored = semcast::restore_catalog(&ctx).await.unwrap();
            assert!(restored.failed.is_empty(), "{:?}", restored.failed);
            ctx
        }
    };
    let create = |replace: bool| {
        format!(
            "CREATE {}EXTERNAL TABLE tickets STORED AS CSV LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            if replace { "OR REPLACE " } else { "" },
            csv.display()
        )
    };

    let ctx = session().await;
    run(&ctx, &create(false)).await;
    run(&ctx, ADD).await;
    run(&ctx, &create(true)).await;
    let ctx = session().await;
    let message = error(&ctx, "SELECT angry FROM tickets").await;
    assert!(message.contains("angry"), "replaced away: {message}");

    run(
        &ctx,
        "ALTER TABLE public.tickets ADD SEMANTIC COLUMN angry AS (body MEANS 'x')",
    )
    .await;
    run(&ctx, "ALTER TABLE tickets DROP SEMANTIC COLUMN angry").await;
    let ctx = session().await;
    let message = error(&ctx, "SELECT angry FROM tickets").await;
    assert!(
        message.contains("angry"),
        "dropped by its bare name: {message}"
    );
}

/// A view keeps the scan it was planned with; columns added since must not
/// break it. A column it read that is dropped is a clear error.
#[tokio::test]
async fn views_survive_changes_to_the_semantic_columns() {
    let ctx = tickets(Arc::new(MockModel::answering_yes_to(["furious"]))).await;
    run(&ctx, ADD).await;
    run(&ctx, "CREATE VIEW v AS SELECT * FROM tickets").await;
    run(&ctx, "CREATE VIEW w AS SELECT id FROM tickets WHERE angry").await;
    run(
        &ctx,
        "ALTER TABLE tickets ADD SEMANTIC COLUMN refund AS (body MEANS 'asks for a refund')",
    )
    .await;
    assert_eq!(ids(&run(&ctx, "SELECT id FROM v WHERE angry").await), [1]);
    assert_eq!(ids(&run(&ctx, "SELECT id FROM w").await), [1]);
    run(&ctx, "ALTER TABLE tickets DROP SEMANTIC COLUMN refund").await;
    assert_eq!(ids(&run(&ctx, "SELECT id FROM w").await), [1]);
    run(&ctx, "ALTER TABLE tickets DROP SEMANTIC COLUMN angry").await;
    let message = error(&ctx, "SELECT id FROM w").await;
    assert!(message.contains("no longer exists"), "{message}");
}

#[tokio::test]
async fn update_cannot_set_or_read_a_semantic_column() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(&ctx, ADD).await;
    run(
        &ctx,
        "ALTER TABLE tickets ADD SEMANTIC COLUMN refund AS (body MEANS 'asks for a refund')",
    )
    .await;
    for (sql, expected) in [
        (
            "UPDATE tickets SET angry = refund WHERE id = 1",
            "cannot set it",
        ),
        (
            "UPDATE tickets SET angry = false WHERE id = 1",
            "cannot set it",
        ),
        (
            "UPDATE tickets SET body = CASE WHEN angry THEN 'x' ELSE body END WHERE id = 1",
            "from a semantic column",
        ),
    ] {
        let message = error(&ctx, sql).await;
        assert!(message.contains(expected), "{sql}: {message}");
    }
    assert_eq!(model.completion_calls(), 0);
}

/// Same-named tables in two schemas keep their own semantic columns.
#[tokio::test]
async fn replacing_a_table_in_another_schema_keeps_this_ones_columns() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("tickets.csv");
    std::fs::write(&csv, "id,body\n1,I am furious\n2,all good\n").unwrap();
    let store = dir.path().join("store.db");
    let session = || {
        let store = store.clone();
        async move {
            let ctx =
                SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["furious"])))
                    .with_catalog(semcast::catalog::Catalog::open(&store).unwrap())
                    .build();
            let restored = semcast::restore_catalog(&ctx).await.unwrap();
            assert!(restored.failed.is_empty(), "{:?}", restored.failed);
            ctx
        }
    };
    let external = |name: &str, replace: bool| {
        format!(
            "CREATE {}EXTERNAL TABLE {name} STORED AS CSV LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            if replace { "OR REPLACE " } else { "" },
            csv.display()
        )
    };
    let ctx = session().await;
    run(&ctx, &external("tickets", false)).await;
    run(&ctx, ADD).await;
    run(&ctx, "CREATE SCHEMA s").await;
    run(&ctx, &external("s.tickets", false)).await;
    run(&ctx, &external("s.tickets", true)).await;
    let ctx = session().await;
    assert_eq!(
        ids(&run(&ctx, "SELECT id FROM tickets WHERE angry").await),
        [1]
    );
}

/// A model under another id, so an estimate can find a list price.
#[derive(Debug)]
struct Priced(MockModel, &'static str);

#[async_trait::async_trait]
impl semcast::model::ModelProvider for Priced {
    fn id(&self) -> semcast::model::ModelId {
        semcast::model::ModelId(self.1.to_owned())
    }
    async fn complete(
        &self,
        requests: Vec<semcast::model::CompletionRequest>,
    ) -> Vec<semcast::Result<semcast::model::Completion>> {
        self.0.complete(requests).await
    }
    async fn embed(&self, texts: Vec<String>) -> semcast::Result<Vec<semcast::model::Embedding>> {
        self.0.embed(texts).await
    }
}

fn cell_u64(batches: &[RecordBatch], name: &str) -> u64 {
    let column = batches[0].column_by_name(name).unwrap();
    column
        .as_any()
        .downcast_ref::<datafusion::arrow::array::UInt64Array>()
        .unwrap()
        .value(0)
}

fn cell_cost(batches: &[RecordBatch]) -> Option<f64> {
    let column = batches[0].column_by_name("est_cost_usd").unwrap();
    let column = column
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Float64Array>()
        .unwrap();
    column.is_valid(0).then(|| column.value(0))
}

#[tokio::test]
async fn explain_refresh_shows_the_bill_without_paying_it() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    // A duplicate text: one question, asked once.
    run(
        &ctx,
        "INSERT INTO tickets VALUES (4, 'I am furious about the outage')",
    )
    .await;
    run(&ctx, ADD).await;

    let before = run(&ctx, "EXPLAIN REFRESH SEMANTIC COLUMN tickets(angry)").await;
    assert_eq!(cell_u64(&before, "rows"), 4);
    assert_eq!(
        cell_u64(&before, "texts"),
        2,
        "NULL and the duplicate don't count"
    );
    assert_eq!(cell_u64(&before, "stored"), 0);
    assert_eq!(cell_u64(&before, "to_ask"), 2);
    assert!(cell_u64(&before, "est_input_tokens") > 0);
    assert_eq!(
        cell_u64(&before, "est_output_tokens"),
        16,
        "each answer at its 8-token cap"
    );
    assert_eq!(cell_cost(&before), None, "the mock has no list price");
    assert_eq!(model.completion_calls(), 0, "explaining asks nothing");

    run(&ctx, "REFRESH SEMANTIC COLUMN tickets(angry)").await;
    assert_eq!(model.completion_calls(), 2);
    let after = run(&ctx, "EXPLAIN REFRESH SEMANTIC COLUMN tickets(angry)").await;
    assert_eq!(cell_u64(&after, "stored"), 2);
    assert_eq!(cell_u64(&after, "to_ask"), 0);
    assert_eq!(cell_u64(&after, "est_input_tokens"), 0);
}

#[tokio::test]
async fn explain_refresh_prices_a_known_model() {
    let model = Arc::new(Priced(
        MockModel::answering_yes_to(["furious"]),
        "anthropic/claude-haiku-4-5",
    ));
    let ctx = SemcastContextBuilder::new(model).build();
    ctx.sql(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'I am furious'), (2, 'fine')) AS v(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    run(
        &ctx,
        "ALTER TABLE t ADD SEMANTIC COLUMN angry AS (body MEANS 'is angry')",
    )
    .await;
    let batches = run(&ctx, "EXPLAIN REFRESH SEMANTIC COLUMN t(angry)").await;
    let input = cell_u64(&batches, "est_input_tokens") as f64;
    let output = cell_u64(&batches, "est_output_tokens") as f64;
    let expected = (input * 1.0 + output * 5.0) / 1_000_000.0;
    let cost = cell_cost(&batches).expect("Haiku 4.5 has a list price");
    assert!((cost - expected).abs() < 1e-12, "{cost} vs {expected}");
}

/// A text repeated within one batch is asked once, whether it is read as a
/// label or filtered on — and so the estimate's "to ask" is what is paid.
#[tokio::test]
async fn a_repeated_text_is_asked_once_per_batch() {
    for sql in [
        "SELECT id, angry FROM many",
        "SELECT id FROM many WHERE angry",
        "SELECT id FROM many WHERE body MEANS 'the customer is angry'",
    ] {
        let model = Arc::new(MockModel::answering_yes_to(["furious"]));
        let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
        // 100 rows, two texts, one batch.
        ctx.sql(
            "CREATE TABLE many AS SELECT value AS id, \
             CASE WHEN value % 2 = 0 THEN 'I am furious' ELSE 'all good' END AS body \
             FROM generate_series(1, 100)",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
        run(
            &ctx,
            "ALTER TABLE many ADD SEMANTIC COLUMN angry AS (body MEANS 'the customer is angry')",
        )
        .await;
        let explained = run(&ctx, "EXPLAIN REFRESH SEMANTIC COLUMN many(angry)").await;
        assert_eq!(cell_u64(&explained, "to_ask"), 2);
        run(&ctx, sql).await;
        assert!(
            model.completion_calls() <= 2 * 16,
            "{sql}: {} calls for two texts (at most two per partition batch; without the dedupe it is one per row)",
            model.completion_calls()
        );
    }
}
