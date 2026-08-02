//! End-to-end tests for semantic clustering: `GROUP BY MEANING OF`, the
//! auto-`k` sweep, and the one-call-per-group labelling, against the
//! deterministic mock model.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use datafusion::execution::context::SessionContext;
use semcast::model::{CompletionRequest, MockModel};
use semcast::{IndexOptions, create_semantic_index, semcast_context};
use serde_json::{Value, json};

/// Two obvious themes, three documents each. The mock embeds by theme
/// keyword (see `MockModel::embedding_by_theme`), so the geometry under test
/// is the clustering, not an accident of how English bytes happen to land.
const BILLING: [&str; 3] = [
    "the invoice was wrong and I want a refund for the billing error",
    "billing charged my card twice, please refund the duplicate invoice",
    "refund the incorrect invoice, the billing amount is wrong",
];
const LAUNCH: [&str; 3] = [
    "we agreed to ship offline sync in the third quarter launch",
    "the launch of offline sync is scheduled for the third quarter",
    "offline sync ships at the quarterly launch as agreed",
];

/// Name a group after whichever theme its excerpts are drawn from.
fn theme_label(request: &CompletionRequest) -> Value {
    let label = if request.input.contains("refund") {
        "billing problems"
    } else if request.input.contains("launch") {
        "launch planning"
    } else {
        "other"
    };
    json!({ "label": label })
}

fn clustering_model() -> Arc<MockModel> {
    Arc::new(MockModel::answering_json_with(theme_label).embedding_by_theme(["refund", "launch"]))
}

/// notes(id, body) — three billing, three launch.
async fn notes_context(model: Arc<MockModel>) -> (SessionContext, tempfile::TempDir) {
    let ctx = semcast_context(model);
    let values: Vec<String> = BILLING
        .iter()
        .chain(LAUNCH.iter())
        .enumerate()
        .map(|(i, body)| format!("({}, '{body}')", i + 1))
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE notes AS SELECT * FROM (VALUES {}) AS t(id, body)",
        values.join(", ")
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "notes",
        "body",
        IndexOptions {
            path: Some(dir.path().join("notes.body.lance")),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    (ctx, dir)
}

async fn run(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
}

/// `(label, count)` pairs from a two-column aggregate, sorted for stability.
async fn groups(ctx: &SessionContext, sql: &str) -> Vec<(String, i64)> {
    let batches = run(ctx, sql).await;
    let mut rows: Vec<(String, i64)> = batches
        .iter()
        .flat_map(|b| {
            let labels = b
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("label is Utf8");
            let counts = b
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("count is Int64");
            (0..b.num_rows())
                .map(|i| {
                    (
                        if labels.is_valid(i) {
                            labels.value(i).to_owned()
                        } else {
                            String::new()
                        },
                        counts.value(i),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    rows.sort();
    rows
}

async fn error(ctx: &SessionContext, sql: &str) -> String {
    match semcast::sql(ctx, sql).await {
        Ok(frame) => match frame.create_physical_plan().await {
            Ok(_) => panic!("expected an error from: {sql}"),
            Err(err) => err.to_string(),
        },
        Err(err) => err.to_string(),
    }
}

const BY_THEME: &str =
    "SELECT topic, count(*) AS n FROM notes GROUP BY MEANING OF body INTO 2 AS topic";

#[tokio::test]
async fn optimized_plan_puts_cluster_below_the_aggregate() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let plan = semcast::sql(&ctx, BY_THEME)
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();

    let display = format!("{}", plan.display_indent());
    assert!(
        display.contains("SemCluster: MEANING OF INTO 2"),
        "{display}"
    );
    assert!(
        display.contains("blocking"),
        "EXPLAIN should admit the operator blocks:\n{display}"
    );
    let aggregate = display.find("Aggregate").expect("aggregate survives");
    let cluster = display.find("SemCluster").unwrap();
    assert!(cluster > aggregate, "cluster belongs below:\n{display}");
}

#[tokio::test]
async fn groups_rows_by_what_they_are_about() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    assert_eq!(
        groups(&ctx, BY_THEME).await,
        vec![
            ("billing problems".to_owned(), 3),
            ("launch planning".to_owned(), 3),
        ]
    );
}

#[tokio::test]
async fn one_model_call_per_group_not_per_row() {
    let model = clustering_model();
    let (ctx, _dir) = notes_context(Arc::clone(&model)).await;
    run(&ctx, BY_THEME).await;

    assert_eq!(
        model.completion_calls(),
        2,
        "six rows, two groups: two labelling calls"
    );
}

#[tokio::test]
async fn auto_k_needs_no_into() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let rows = groups(
        &ctx,
        "SELECT topic, count(*) AS n FROM notes GROUP BY MEANING OF body AS topic",
    )
    .await;

    assert!(!rows.is_empty(), "auto-k must produce groups");
    let total: i64 = rows.iter().map(|(_, n)| n).sum();
    assert_eq!(total, 6, "every row lands in exactly one group: {rows:?}");
}

/// `INTO` overrides what the data would suggest — asking for one group over
/// two obvious themes has to collapse them, not quietly keep both.
#[tokio::test]
async fn explicit_into_fixes_the_group_count() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let rows = groups(
        &ctx,
        "SELECT topic, count(*) AS n FROM notes GROUP BY MEANING OF body INTO 1 AS topic",
    )
    .await;
    assert_eq!(rows.len(), 1, "INTO 1 must yield one group: {rows:?}");
    assert_eq!(rows[0].1, 6, "and it holds every row");
}

/// Asking for more groups than the corpus has distinct positions cannot
/// invent them: empty clusters carry no rows, so the result is what the data
/// supports rather than a padded-out `k`.
#[tokio::test]
async fn asking_for_more_groups_than_exist_yields_what_the_data_supports() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let rows = groups(
        &ctx,
        "SELECT topic, count(*) AS n FROM notes GROUP BY MEANING OF body INTO 5 AS topic",
    )
    .await;
    assert_eq!(
        rows.len(),
        2,
        "two themes, however many groups asked: {rows:?}"
    );
}

#[tokio::test]
async fn the_same_query_reproduces_its_grouping() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    // k-means is seeded, so the grouping is a function of the corpus alone.
    assert_eq!(groups(&ctx, BY_THEME).await, groups(&ctx, BY_THEME).await);
}

#[tokio::test]
async fn labels_come_from_the_cache_the_second_time() {
    let model = clustering_model();
    let (ctx, _dir) = notes_context(Arc::clone(&model)).await;
    run(&ctx, BY_THEME).await;
    let after_first = model.completion_calls();
    run(&ctx, BY_THEME).await;

    assert_eq!(
        model.completion_calls(),
        after_first,
        "an unchanged grouping must not be re-labelled"
    );
}

#[tokio::test]
async fn labelling_reads_the_documents_not_the_whole_corpus() {
    let model = clustering_model();
    let (ctx, _dir) = notes_context(Arc::clone(&model)).await;
    run(&ctx, BY_THEME).await;

    let schemas = model.completion_schemas();
    assert!(
        schemas.iter().all(|s| s.is_some()),
        "every labelling call is schema-constrained"
    );
    for input in model.completion_inputs() {
        assert!(
            input.matches("---").count() <= 4,
            "at most five representatives per group: {input}"
        );
    }
}

#[tokio::test]
async fn a_group_that_cannot_be_named_still_groups() {
    // Answers nothing usable, so every label falls back.
    let model = Arc::new(MockModel::answering_json_with(|_| json!({"nope": 1})));
    let (ctx, _dir) = notes_context(Arc::clone(&model)).await;
    let rows = groups(&ctx, BY_THEME).await;

    assert_eq!(rows.len(), 2, "the grouping survives the naming: {rows:?}");
    assert!(
        rows.iter().all(|(label, _)| label.starts_with("group ")),
        "unnamed groups get positional names: {rows:?}"
    );
}

#[tokio::test]
async fn meaning_of_works_as_a_plain_select_column() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let batches = run(
        &ctx,
        "SELECT id, meaning_of(body, 2) AS topic FROM notes ORDER BY id",
    )
    .await;

    assert_eq!(batches[0].schema().field(1).name(), "topic");
    let labels = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("topic is Utf8");
    // The first three notes are billing, the last three launch.
    assert_eq!(labels.value(0), labels.value(1));
    assert_ne!(labels.value(0), labels.value(5));
}

#[tokio::test]
async fn cluster_reports_a_funnel() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let plan = semcast::sql(&ctx, BY_THEME)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();

    let counts = semcast::server::progress::snapshot(&plan);
    assert!(counts.has_cluster);
    assert_eq!(counts.cluster_groups, 2);
    assert_eq!(counts.cluster_model_calls, 2);
    let line = semcast::server::progress::render(&counts);
    assert!(line.contains("cluster: 2 groups"), "{line}");
}

#[tokio::test]
async fn rows_the_index_never_saw_have_no_group() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    ctx.sql("INSERT INTO notes VALUES (99, 'a late arrival nobody indexed')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let batches = run(
        &ctx,
        "SELECT id, meaning_of(body, 2) AS topic FROM notes WHERE id = 99",
    )
    .await;
    let labels = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("topic is Utf8");
    assert!(
        labels.is_null(0),
        "an unindexed row keeps its place with no group, rather than vanishing"
    );
}

#[tokio::test]
async fn clustering_without_an_index_is_a_plan_time_error() {
    let ctx = semcast_context(clustering_model());
    ctx.sql("CREATE TABLE plain AS SELECT * FROM (VALUES (1, 'a')) AS t(id, body)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let message = error(
        &ctx,
        "SELECT topic, count(*) FROM plain GROUP BY MEANING OF body AS topic",
    )
    .await;
    assert!(
        message.contains("requires a semantic index on plain(body)"),
        "{message}"
    );
    assert!(
        message.contains("CREATE SEMANTIC INDEX ON plain(body)"),
        "{message}"
    );
}

#[tokio::test]
async fn meaning_of_in_a_where_clause_is_a_plan_time_error() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let message = error(
        &ctx,
        "SELECT id FROM notes WHERE meaning_of(body, 2) = 'billing problems'",
    )
    .await;
    assert!(
        message.contains("GROUP BY key or in a SELECT list"),
        "{message}"
    );
}

/// DataFusion lifts an aggregate's arguments into a projection below it, so
/// this is a legal "how many groups are there" — the label is materialized
/// once and counted like any other column.
#[tokio::test]
async fn counting_distinct_groups_works() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let batches = run(
        &ctx,
        "SELECT count(DISTINCT meaning_of(body, 2)) AS n FROM notes",
    )
    .await;
    let counts = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count is Int64");
    assert_eq!(counts.value(0), 2);
}

#[tokio::test]
async fn meaning_of_in_a_join_condition_is_a_plan_time_error() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let message = error(
        &ctx,
        "SELECT a.id FROM notes a JOIN notes b ON meaning_of(a.body, 2) = b.body",
    )
    .await;
    assert!(
        message.contains("GROUP BY key or in a SELECT list"),
        "{message}"
    );
}

#[tokio::test]
async fn into_zero_is_a_parse_error() {
    let (ctx, _dir) = notes_context(clustering_model()).await;
    let message = error(
        &ctx,
        "SELECT topic, count(*) FROM notes GROUP BY MEANING OF body INTO 0 AS topic",
    )
    .await;
    assert!(
        message.contains("positive number of groups"),
        "INTO 0 must be rejected on its own terms, not as a parse failure: {message}"
    );
}
