//! The README walk-through, as far as the MVP takes it: a semantic filter
//! that plans through the rewrite rule and executes through VerifyExec —
//! against the deterministic mock model, so it runs without any setup.
//!
//! Run with: `cargo run --example meetings`

use std::sync::Arc;

use semcast::model::MockModel;
use semcast::semcast_context;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));

    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'atlas planning',   CAST('2026-05-12 10:00:00' AS TIMESTAMP),
              'we agreed to ship offline sync in Q3, pending the sync-engine work'),
             (2, 'weekly standup',   CAST('2026-06-02 09:30:00' AS TIMESTAMP),
              'status round, nothing notable happened'),
             (3, 'beacon retro',     CAST('2025-11-20 15:00:00' AS TIMESTAMP),
              'retro on the beacon launch; offline sync came up as a future idea'),
             (4, 'incident review',  CAST('2026-06-20 11:00:00' AS TIMESTAMP),
              CAST(NULL AS VARCHAR))
         ) AS t(meeting_id, title, held_at, transcript)",
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
             WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
               AND means(transcript, 'discussed the launch of offline sync in Atlas')
             ORDER BY held_at",
        )
        .await?;

    println!("Optimized plan:\n{}\n", df.clone().into_optimized_plan()?.display_indent());

    let physical = df.clone().create_physical_plan().await?;
    println!(
        "Physical plan (with the verify-stage call estimate):\n{}",
        datafusion::physical_plan::displayable(physical.as_ref()).indent(false)
    );

    println!("Results:");
    df.show().await?;

    Ok(())
}
