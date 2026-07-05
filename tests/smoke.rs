//! Skeleton smoke tests: everything registers, plain SQL still works, and
//! the pieces that are implemented behave. No test touches a `todo!()` path.

use std::sync::Arc;

use datafusion::common::DFSchema;
use datafusion::logical_expr::{Extension, LogicalPlan, lit};
use semcast::logical::SemFilterNode;
use semcast::model::MockModel;
use semcast::semcast_context;

fn test_context() -> datafusion::execution::context::SessionContext {
    semcast_context(Arc::new(MockModel::default()))
}

#[tokio::test]
async fn context_runs_plain_sql() {
    let ctx = test_context();
    let batches = ctx
        .sql("SELECT 1 + 1 AS two")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].num_rows(), 1);
}

#[tokio::test]
async fn means_udf_is_registered_and_plans() {
    let ctx = test_context();
    assert!(ctx.state().scalar_functions().contains_key("means"));

    // A query using means() must plan cleanly. (Executing it is roadmap
    // step 1 — the optimizer rewrite and VerifyExec::execute.)
    ctx.sql("SELECT means('some transcript text', 'discussed a launch') AS hit")
        .await
        .unwrap();
}

#[test]
fn sem_filter_node_displays_like_the_readme() {
    let empty = LogicalPlan::EmptyRelation(datafusion::logical_expr::EmptyRelation {
        produce_one_row: false,
        schema: Arc::new(DFSchema::empty()),
    });
    let node = SemFilterNode::new(
        empty,
        lit("transcript"),
        "discussed the launch of offline sync in Atlas",
        Some(0.9),
    );
    let plan = LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    });

    let display = format!("{}", plan.display_indent());
    assert!(
        display.contains("SemFilter: MEANS('discussed the launch of offline sync in Atlas')"),
        "unexpected explain output: {display}"
    );
    assert!(display.contains("recall ≥ 0.90"), "{display}");
}

#[tokio::test]
async fn mock_model_is_deterministic() {
    use semcast::model::{CompletionRequest, ModelProvider};

    let model = MockModel::answering_yes_to(["offline sync"]);
    let request = |input: &str| CompletionRequest {
        system: "does this meeting discuss the condition?".into(),
        input: input.into(),
        max_tokens: 4,
    };

    let answers = model
        .complete(vec![
            request("we agreed to ship offline sync in Q3"),
            request("weekly standup, nothing notable"),
        ])
        .await;

    assert_eq!(answers[0].as_ref().unwrap().text, "yes");
    assert_eq!(answers[1].as_ref().unwrap().text, "no");
}
