//! The model scheduler: one cap on requests in flight for the whole
//! session, and identical requests in flight sent once.

use std::sync::Arc;
use std::time::Duration;

use datafusion::execution::context::SessionContext;
use semcast::SemcastContextBuilder;
use semcast::model::MockModel;

const ANGRY: &str = "SELECT id FROM tickets WHERE body MEANS 'the customer is angry' ORDER BY id";

async fn tickets(ctx: &SessionContext) {
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
}

#[tokio::test]
async fn the_same_query_twice_at_once_pays_once() {
    let model = Arc::new(
        MockModel::answering_yes_to(["furious"]).with_latency(Duration::from_millis(200)),
    );
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
    tickets(&ctx).await;

    let run = |ctx: SessionContext| async move {
        semcast::sql(&ctx, ANGRY)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
    };
    let (first, second) = tokio::join!(run(ctx.clone()), run(ctx.clone()));
    assert_eq!(first, second);
    assert_eq!(
        model.completion_calls(),
        3,
        "both queries missed the cache, but each verdict was asked once"
    );
}
