//! The phase 2 binder gate: the share of the `MEANS` corpus
//! (`tests/corpus/means.sql`) that plans with every `MEANS` turned into an
//! `ASK` node. Run with `--nocapture` to see the report:
//!
//! ```sh
//! cargo test --test binder_coverage -- --nocapture
//! ```

use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::Expr;
use semcast::SemcastContextBuilder;
use semcast::model::MockModel;

/// The gate. Below it, the redesign falls back to "A plus store".
const GATE: f64 = 0.90;

struct Entry {
    source: String,
    sql: String,
}

fn corpus() -> Vec<Entry> {
    let text = include_str!("corpus/means.sql");
    let mut entries: Vec<Entry> = Vec::new();
    for line in text.lines() {
        if let Some(source) = line.strip_prefix("--- ") {
            entries.push(Entry {
                source: source.to_owned(),
                sql: String::new(),
            });
        } else if let Some(entry) = entries.last_mut() {
            entry.sql.push_str(line);
            entry.sql.push('\n');
        }
    }
    entries
}

async fn fixture() -> SessionContext {
    let ctx =
        SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["furious"]))).build();
    for ddl in [
        "CREATE TABLE meetings (meeting_id BIGINT, title VARCHAR, held_at TIMESTAMP, transcript VARCHAR)",
        "CREATE TABLE tickets (id BIGINT, customer_id BIGINT, status VARCHAR, priority VARCHAR, \
         subject VARCHAR, body VARCHAR, created_at TIMESTAMP)",
        "CREATE TABLE customers (id BIGINT, name VARCHAR, notes VARCHAR)",
    ] {
        ctx.sql(ddl).await.unwrap().collect().await.unwrap();
    }
    for ddl in [
        "CREATE SEMANTIC TYPE MeetingFacts AS ( \
           launch_stage ONEOF(none, idea, planned, scheduled, shipped) \
             'the furthest launch stage discussed' )",
        "CREATE VIEW angry_tickets AS SELECT * FROM tickets WHERE body MEANS 'is angry'",
    ] {
        semcast::sql(&ctx, ddl)
            .await
            .unwrap_or_else(|e| panic!("{ddl}: {e}"))
            .collect()
            .await
            .unwrap();
    }
    ctx
}

/// `Ok(())` when the statement plans and no marker survives into the plan;
/// the reason otherwise.
async fn bind(ctx: &SessionContext, sql: &str) -> Result<(), String> {
    let df = semcast::sql(ctx, sql)
        .await
        .map_err(|e| first_line(&e.to_string()))?;
    let plan = df
        .clone()
        .into_optimized_plan()
        .map_err(|e| first_line(&e.to_string()))?;
    // A marker call left in any expression would run as a UDF and fail; the
    // output column *names* may still spell `means(...)`, which is fine.
    let mut survived = false;
    plan.apply_with_subqueries(|node| {
        for expr in node.expressions() {
            if expr.exists(|e| {
                Ok(matches!(e, Expr::ScalarFunction(f)
                    if f.func.name() == "means" || f.func.name() == "confidence"))
            })? {
                survived = true;
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map_err(|e| e.to_string())?;
    if survived {
        return Err("a MEANS marker survived planning".to_owned());
    }
    df.create_physical_plan()
        .await
        .map_err(|e| first_line(&e.to_string()))?;
    Ok(())
}

fn first_line(message: &str) -> String {
    let message = message
        .rsplit("Error during planning: ")
        .next()
        .unwrap_or(message);
    message
        .lines()
        .next()
        .unwrap_or(message)
        .chars()
        .take(110)
        .collect()
}

#[tokio::test]
async fn binder_coverage_meets_the_gate() {
    let ctx = fixture().await;
    let entries = corpus();
    let mut bound = 0;
    let mut report = Vec::new();
    for entry in &entries {
        match bind(&ctx, &entry.sql).await {
            Ok(()) => bound += 1,
            Err(reason) => report.push(format!("  ✗ {}: {reason}", entry.source)),
        }
    }
    let coverage = bound as f64 / entries.len() as f64;
    println!(
        "binder coverage: {bound}/{} = {:.1}% (gate {:.0}%)",
        entries.len(),
        coverage * 100.0,
        GATE * 100.0
    );
    for line in &report {
        println!("{line}");
    }
    assert!(
        coverage >= GATE,
        "binder coverage {coverage:.3} is below the gate"
    );
}

/// DataFusion moves an inner join's ON condition into a filter, and an outer
/// join's condition on the joined side into that side's input, so a MEANS
/// there binds. A condition on an outer join's preserved side stays in the
/// join, and is refused at plan time.
#[tokio::test]
async fn only_an_outer_joins_preserved_side_condition_is_refused() {
    let ctx = fixture().await;
    let inner = "SELECT t.id FROM tickets t JOIN customers c \
                 ON t.customer_id = c.id AND t.body MEANS 'mentions the account manager'";
    assert_eq!(bind(&ctx, inner).await, Ok(()));
    // The joined side's condition moves into that side's input before the join.
    let pushed = "SELECT t.id FROM tickets t LEFT JOIN customers c \
                  ON t.customer_id = c.id AND c.notes MEANS 'enterprise account'";
    assert_eq!(bind(&ctx, pushed).await, Ok(()));
    // The preserved side's can't: it decides which rows get a match, not
    // which rows exist.
    let outer = "SELECT t.id FROM tickets t LEFT JOIN customers c \
                 ON t.customer_id = c.id AND t.body MEANS 'mentions the account manager'";
    let error = bind(&ctx, outer).await.unwrap_err();
    assert!(error.contains("single-input step"), "{error}");
}
