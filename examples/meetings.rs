//! The README walk-through, as far as the skeleton takes it: build a semcast
//! context, create the meetings table, and *plan* a `means()` query.
//!
//! Run with: `cargo run --example meetings`

use std::sync::Arc;

use semcast::model::MockModel;
use semcast::semcast_context;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));

    ctx.sql(
        "CREATE TABLE meetings (
             meeting_id INT,
             title      TEXT,
             held_at    TIMESTAMP,
             attendees  TEXT,
             transcript TEXT
         )",
    )
    .await?
    .collect()
    .await?;

    // Eventually: transcript MEANS '...' WITH RECALL 0.9 — the means() UDF
    // stands in for the operator until the parser extension lands.
    let df = ctx
        .sql(
            "SELECT meeting_id, title, held_at
             FROM meetings
             WHERE held_at >= now() - INTERVAL '6 months'
               AND means(transcript, 'discussed the launch of offline sync in Atlas')",
        )
        .await?;

    println!("Logical plan:\n{}\n", df.logical_plan().display_indent());
    println!(
        "Execution stops here for now: the optimizer rewrite to SemFilter and \
         VerifyExec::execute are roadmap step 1."
    );

    Ok(())
}
