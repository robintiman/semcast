//! End-to-end tests for semantic classification: `MEANS` in a `SELECT` list,
//! branch fusion, and the cache it shares with the verify stage, against the
//! deterministic mock model.

use std::sync::Arc;

use datafusion::arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray};
use datafusion::execution::context::SessionContext;
use semcast::model::{CompletionRequest, MockModel};
use semcast::semcast_context;
use serde_json::{Value, json};

const ANGRY: &str = "this is completely unacceptable, I want a refund now";
const PRICING: &str = "could you tell me what the enterprise plan costs";
const DULL: &str = "thanks, that answers my question";

/// Answer every predicate the request asks about by keyword. Handles both
/// shapes: the fused JSON object, and the schemaless yes/no a single pending
/// condition gets.
fn keyword_answers(request: &CompletionRequest) -> Value {
    let mut object = serde_json::Map::new();
    for line in request.system.lines() {
        let Some((key, condition)) = line.split_once(": ") else {
            continue;
        };
        if !key.starts_with('c') || key.len() < 2 {
            continue;
        }
        object.insert(key.to_owned(), json!(holds(condition, &request.input)));
    }
    Value::Object(object)
}

/// The mock's ground truth: a condition holds when the document contains the
/// condition's distinctive keyword.
fn holds(condition: &str, input: &str) -> bool {
    match condition {
        c if c.contains("angry") => input.contains("unacceptable"),
        c if c.contains("pricing") => input.contains("costs"),
        c if c.contains("refund") => input.contains("refund"),
        _ => false,
    }
}

/// A mock that serves both request shapes: schema'd requests get the fused
/// JSON, schemaless ones the yes/no the verify stage speaks.
fn classify_model() -> Arc<MockModel> {
    Arc::new(
        MockModel::answering_json_with(keyword_answers).also_answering_yes_to(["unacceptable"]),
    )
}

/// tickets(id, status, body) — one angry, one pricing, one dull, one NULL.
async fn tickets_context(model: Arc<MockModel>) -> SessionContext {
    let ctx = semcast_context(model);
    ctx.sql(&format!(
        "CREATE TABLE tickets AS
         SELECT * FROM (VALUES
             (1, 'open',   '{ANGRY}'),
             (2, 'open',   '{PRICING}'),
             (3, 'open',   '{DULL}'),
             (4, 'closed', CAST(NULL AS VARCHAR))
         ) AS t(id, status, body)",
    ))
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
        .unwrap()
        .collect()
        .await
        .unwrap()
}

/// The (id, label) pairs of a two-column result, ordered by id.
async fn labels(ctx: &SessionContext, sql: &str) -> Vec<(i64, Option<String>)> {
    let batches = run(ctx, sql).await;
    let mut rows: Vec<(i64, Option<String>)> = batches
        .iter()
        .flat_map(|b| {
            let ids = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64");
            let labels = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("label is Utf8");
            (0..b.num_rows())
                .map(|i| {
                    (
                        ids.value(i),
                        labels.is_valid(i).then(|| labels.value(i).to_owned()),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    rows.sort_by_key(|(id, _)| *id);
    rows
}

async fn error(ctx: &SessionContext, sql: &str) -> String {
    match semcast::sql(ctx, sql).await {
        Ok(frame) => match frame.into_optimized_plan() {
            Ok(_) => panic!("expected an error from: {sql}"),
            Err(err) => err.to_string(),
        },
        Err(err) => err.to_string(),
    }
}

const ROUTE: &str = "SELECT id, CASE WHEN body MEANS 'the customer is angry'      THEN 'escalate'
                                     WHEN body MEANS 'asks about pricing'         THEN 'sales'
                                     ELSE 'other' END AS route
                     FROM tickets";

#[tokio::test]
async fn optimized_plan_puts_classify_below_the_case() {
    let ctx = tickets_context(classify_model()).await;
    let plan = semcast::sql(&ctx, ROUTE)
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();

    let display = format!("{}", plan.display_indent());
    assert!(display.contains("SemClassify"), "{display}");
    assert!(
        display.contains("1 model call per row"),
        "the node should advertise the fusion:\n{display}"
    );
    // CASE stays in the projection — first-match-wins is DataFusion's.
    assert!(display.contains("CASE"), "{display}");
    let projection = display.find("Projection").unwrap();
    let classify = display.find("SemClassify").unwrap();
    assert!(classify > projection, "classify belongs below:\n{display}");
}

#[tokio::test]
async fn each_row_gets_its_first_matching_branch() {
    let ctx = tickets_context(classify_model()).await;
    assert_eq!(
        labels(&ctx, ROUTE).await,
        vec![
            (1, Some("escalate".to_owned())),
            (2, Some("sales".to_owned())),
            (3, Some("other".to_owned())),
            (4, Some("other".to_owned())),
        ]
    );
}

#[tokio::test]
async fn earlier_branches_win() {
    let ctx = tickets_context(classify_model()).await;
    // The angry ticket also mentions a refund; the first branch must claim it.
    let rows = labels(
        &ctx,
        "SELECT id, CASE WHEN body MEANS 'the customer is angry' THEN 'escalate'
                         WHEN body MEANS 'mentions a refund'     THEN 'billing'
                         ELSE 'other' END AS route
         FROM tickets",
    )
    .await;
    assert_eq!(rows[0], (1, Some("escalate".to_owned())));
}

#[tokio::test]
async fn one_model_call_per_row_not_per_branch() {
    let model = classify_model();
    let ctx = tickets_context(Arc::clone(&model)).await;
    run(
        &ctx,
        "SELECT id, CASE WHEN body MEANS 'the customer is angry' THEN 'escalate'
                         WHEN body MEANS 'asks about pricing'    THEN 'sales'
                         WHEN body MEANS 'mentions a refund'     THEN 'billing'
                         ELSE 'other' END AS route
         FROM tickets",
    )
    .await;

    // Three branches over three non-NULL bodies: three calls, not nine.
    assert_eq!(
        model.completion_calls(),
        3,
        "branches must be fused into one call per row"
    );
}

#[tokio::test]
async fn bare_means_in_the_select_list_is_a_boolean_column() {
    let ctx = tickets_context(classify_model()).await;
    let batches = run(
        &ctx,
        "SELECT id, body MEANS 'the customer is angry' AS is_angry FROM tickets",
    )
    .await;

    assert_eq!(batches[0].schema().field(1).name(), "is_angry");
    let flags = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("is_angry is Boolean");
    assert!(flags.value(0), "the angry ticket is angry");
    assert!(!flags.value(1));
    assert!(!flags.value(3), "NULL body satisfies nothing");
}

/// A one-condition classify sends the verify prompt byte-for-byte, so the two
/// stages must land on the same cache entry.
#[tokio::test]
async fn a_single_condition_classify_shares_the_filter_cache() {
    let model = classify_model();
    let ctx = tickets_context(Arc::clone(&model)).await;
    let condition = "the customer is angry";

    run(
        &ctx,
        &format!("SELECT id FROM tickets WHERE body MEANS '{condition}'"),
    )
    .await;
    let after_filter = model.completion_calls();
    assert!(after_filter > 0, "the filter must have asked the model");

    run(
        &ctx,
        &format!("SELECT id, body MEANS '{condition}' AS flag FROM tickets"),
    )
    .await;
    assert_eq!(
        model.completion_calls(),
        after_filter,
        "the classify must reuse the filter's verdicts"
    );
}

#[tokio::test]
async fn editing_one_branch_only_re_asks_that_branch() {
    let model = classify_model();
    let ctx = tickets_context(Arc::clone(&model)).await;
    run(&ctx, ROUTE).await;
    let after_first = model.completion_calls();

    // Same first branch, reworded second one.
    run(
        &ctx,
        "SELECT id, CASE WHEN body MEANS 'the customer is angry'   THEN 'escalate'
                         WHEN body MEANS 'asks about pricing tiers' THEN 'sales'
                         ELSE 'other' END AS route
         FROM tickets",
    )
    .await;

    let asked = model.completion_calls() - after_first;
    assert!(asked > 0, "the reworded branch must be asked");
    let inputs = &model.completion_inputs()[after_first..];
    for input in inputs {
        assert!(
            !input.is_empty(),
            "each re-ask carries the document it is about"
        );
    }
    // The unchanged branch is cached, so the re-ask is a single-condition
    // request and takes the verify prompt.
    let schemas = &model.completion_schemas()[after_first..];
    assert!(
        schemas.iter().all(Option::is_none),
        "only the reworded branch is pending, so no fused schema: {schemas:?}"
    );
}

#[tokio::test]
async fn rows_claimed_by_an_earlier_plain_branch_cost_no_call() {
    let model = classify_model();
    let ctx = tickets_context(Arc::clone(&model)).await;
    let rows = labels(
        &ctx,
        "SELECT id, CASE WHEN status = 'closed'                  THEN 'archived'
                         WHEN body MEANS 'the customer is angry' THEN 'escalate'
                         ELSE 'other' END AS route
         FROM tickets",
    )
    .await;

    assert_eq!(rows[3], (4, Some("archived".to_owned())));
    assert_eq!(
        model.completion_calls(),
        3,
        "the closed ticket never reaches the model"
    );
}

#[tokio::test]
async fn a_failing_model_falls_through_to_else() {
    // Answers nothing usable, so every branch verdict is NULL.
    let model = Arc::new(MockModel::answering_json_with(|_| json!({"nonsense": 1})));
    let ctx = tickets_context(Arc::clone(&model)).await;
    let rows = labels(&ctx, ROUTE).await;

    assert!(
        rows.iter()
            .all(|(_, label)| label.as_deref() == Some("other")),
        "an unusable answer must fall through, not fail the query: {rows:?}"
    );
}

#[tokio::test]
async fn two_case_expressions_over_the_same_text_share_one_node() {
    let model = classify_model();
    let ctx = tickets_context(Arc::clone(&model)).await;
    run(
        &ctx,
        "SELECT id,
                CASE WHEN body MEANS 'the customer is angry' THEN 'escalate' ELSE 'other' END AS a,
                CASE WHEN body MEANS 'asks about pricing'    THEN 'sales'    ELSE 'other' END AS b
         FROM tickets",
    )
    .await;

    // Same text, same (absent) guard: one group, so one fused call per row.
    assert_eq!(model.completion_calls(), 3);
}

#[tokio::test]
async fn classify_reports_a_funnel() {
    let ctx = tickets_context(classify_model()).await;
    let plan = semcast::sql(&ctx, ROUTE)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap();

    let counts = semcast::server::progress::snapshot(&plan);
    assert!(counts.has_classify);
    assert_eq!(counts.classify_model_calls, 3);
    let line = semcast::server::progress::render(&counts);
    assert!(line.contains("classify:"), "{line}");
}

#[tokio::test]
async fn with_recall_on_a_classify_is_a_plan_time_error() {
    let ctx = tickets_context(classify_model()).await;
    let message = error(
        &ctx,
        "SELECT id, body MEANS 'the customer is angry' AS flag FROM tickets WITH RECALL 0.9",
    )
    .await;

    assert!(
        message.contains("nothing to calibrate"),
        "error should explain why recall doesn't apply: {message}"
    );
}

#[tokio::test]
async fn means_in_a_having_clause_is_still_a_plan_time_error() {
    let ctx = tickets_context(classify_model()).await;
    let message = error(
        &ctx,
        "SELECT status, count(*) FROM tickets GROUP BY status
         HAVING bool_or(body MEANS 'the customer is angry')",
    )
    .await;

    assert!(
        message.contains("WHERE") && message.contains("SELECT list"),
        "error should name the legal positions: {message}"
    );
}

#[tokio::test]
async fn a_non_literal_condition_is_still_a_plan_time_error() {
    let ctx = tickets_context(classify_model()).await;
    let message = error(&ctx, "SELECT id, body MEANS status AS flag FROM tickets").await;
    assert!(message.contains("string literal"), "{message}");
}
