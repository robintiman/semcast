//! `[EXPLAIN] REFRESH SEMANTIC VIEW`: paying ahead for a view's questions.

use std::sync::Arc;

use datafusion::arrow::array::{Array, RecordBatch, StringArray, UInt64Array};
use datafusion::execution::context::SessionContext;
use semcast::SemcastContextBuilder;
use semcast::model::MockModel;

async fn tickets(model: Arc<MockModel>) -> SessionContext {
    let ctx = SemcastContextBuilder::new(model).build();
    ctx.sql(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES
             (1, 'I am furious about the outage'),
             (2, 'thanks, all good now'),
             (3, 'furious, and the refund is late'),
             (4, CAST(NULL AS VARCHAR))
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

/// Column `name` of every row of a quote, as `u64` (`None` for NULL).
fn column(batches: &[RecordBatch], name: &str) -> Vec<Option<u64>> {
    batches
        .iter()
        .flat_map(|b| {
            let values = b
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            (0..values.len())
                .map(|i| values.is_valid(i).then(|| values.value(i)))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn text(batches: &[RecordBatch], name: &str) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|b| {
            let values = b
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..values.len())
                .map(|i| values.is_valid(i).then(|| values.value(i).to_owned()))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn a_refresh_pays_ahead_and_the_next_read_is_free() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(
        &ctx,
        "CREATE VIEW angry AS SELECT id FROM tickets WHERE body MEANS 'the customer is angry'",
    )
    .await;

    let quote = run(&ctx, "EXPLAIN REFRESH SEMANTIC VIEW angry").await;
    assert_eq!(text(&quote, "question"), vec![Some("MEANS".to_owned())]);
    assert_eq!(text(&quote, "reads"), vec![Some("tickets.body".to_owned())]);
    assert_eq!(column(&quote, "texts"), vec![Some(3)], "NULL doesn't count");
    assert_eq!(column(&quote, "to_ask"), vec![Some(3)]);
    assert_eq!(model.completion_calls(), 0, "explaining asks nothing");

    run(&ctx, "REFRESH SEMANTIC VIEW angry").await;
    let paid = model.completion_calls();
    assert_eq!(paid, 3);
    assert_eq!(
        run(&ctx, "SELECT * FROM angry")
            .await
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>(),
        2
    );
    assert_eq!(
        model.completion_calls(),
        paid,
        "the read is served from the store"
    );

    let quote = run(&ctx, "EXPLAIN REFRESH SEMANTIC VIEW angry").await;
    assert_eq!(column(&quote, "stored"), vec![Some(3)]);
    assert_eq!(column(&quote, "to_ask"), vec![Some(0)]);
}

#[tokio::test]
async fn a_view_quotes_each_question_it_asks() {
    let model = Arc::new(MockModel::answering_yes_to(["furious", "refund"]));
    let ctx = tickets(Arc::clone(&model)).await;
    run(
        &ctx,
        "CREATE VIEW triage AS SELECT id,
             CASE WHEN body MEANS 'angry' THEN 'urgent'
                  WHEN body MEANS 'about a refund' THEN 'billing'
                  ELSE 'other' END AS queue,
             CONFIDENCE(upper(body) MEANS 'angry') AS sure
         FROM tickets",
    )
    .await;

    let quote = run(&ctx, "EXPLAIN REFRESH SEMANTIC VIEW triage").await;
    let mut questions = text(&quote, "question");
    questions.sort();
    assert_eq!(
        questions,
        vec![Some("CONFIDENCE".to_owned()), Some("label".to_owned())]
    );
    let reads = text(&quote, "reads");
    let to_ask = column(&quote, "to_ask");
    for (read, to_ask) in reads.iter().zip(&to_ask) {
        match read {
            // The CASE: two conditions, one fused call per text.
            Some(read) => {
                assert_eq!(read, "tickets.body");
                assert_eq!(*to_ask, Some(3));
            }
            // CONFIDENCE over a computed expression: no column to count.
            None => assert_eq!(*to_ask, None),
        }
    }
    assert_eq!(model.completion_calls(), 0);

    run(&ctx, "REFRESH SEMANTIC VIEW triage").await;
    let paid = model.completion_calls();
    run(&ctx, "SELECT * FROM triage").await;
    assert_eq!(
        model.completion_calls(),
        paid,
        "the read is served from the store"
    );
    let quote = run(&ctx, "EXPLAIN REFRESH SEMANTIC VIEW triage").await;
    for (read, to_ask) in text(&quote, "reads").iter().zip(column(&quote, "to_ask")) {
        if read.is_some() {
            assert_eq!(to_ask, Some(0), "the fused answers count as stored");
        }
    }
}

#[tokio::test]
async fn only_a_view_that_asks_something_refreshes() {
    let ctx = tickets(Arc::new(MockModel::default())).await;
    run(&ctx, "CREATE VIEW plain AS SELECT id FROM tickets").await;
    let message = error(&ctx, "REFRESH SEMANTIC VIEW plain").await;
    assert!(message.contains("asks no semantic question"), "{message}");
    let message = error(&ctx, "EXPLAIN REFRESH SEMANTIC VIEW tickets").await;
    assert!(message.contains("is not a view"), "{message}");
    let message = error(&ctx, "REFRESH SEMANTIC VIEW").await;
    assert!(message.contains("REFRESH SEMANTIC VIEW name"), "{message}");
}
