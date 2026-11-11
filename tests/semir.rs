//! SemIR: one question, however it is written, is one stored answer per text.

use std::sync::Arc;

use semcast::SemcastContextBuilder;
use semcast::model::MockModel;

#[tokio::test]
async fn whitespace_in_a_condition_does_not_make_a_new_question() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
    ctx.sql(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES
             (1, 'I am furious about the outage'),
             (2, 'thanks, all good now')
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    for sql in [
        "SELECT id FROM tickets WHERE body MEANS 'the customer is angry'",
        "SELECT id FROM tickets WHERE body MEANS '  the customer\n   is angry '",
        "SELECT id, CONFIDENCE(body MEANS 'the  customer is angry') FROM tickets",
        "SELECT id, body MEANS 'the customer\tis angry' FROM tickets",
    ] {
        semcast::sql(&ctx, sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    assert_eq!(
        model.completion_calls(),
        2,
        "every spelling asks the one question, once per ticket"
    );
}

#[tokio::test]
async fn the_model_is_asked_the_normalized_condition() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
    ctx.sql("CREATE TABLE t AS VALUES ('I am furious')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batches = semcast::sql(
        &ctx,
        "EXPLAIN SELECT * FROM t WHERE column1 MEANS '  is\n angry  '",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let text = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    assert!(text.contains("MEANS('is angry')"), "{text}");
}

#[tokio::test]
async fn an_empty_condition_is_a_plan_error() {
    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
    ctx.sql("CREATE TABLE t AS VALUES ('I am furious')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    for sql in [
        "SELECT * FROM t WHERE column1 MEANS ' \t '",
        "SELECT * FROM t WHERE column1 MEANS ''",
    ] {
        let error = match semcast::sql(&ctx, sql).await {
            Err(e) => e.to_string(),
            Ok(df) => df.collect().await.expect_err(sql).to_string(),
        };
        assert!(error.contains("cannot be empty"), "{sql}: {error}");
    }
    assert_eq!(model.completion_calls(), 0);
}
