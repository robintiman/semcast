//! The README walk-through, as far as the roadmap takes it: a semantic
//! index built with `CREATE SEMANTIC INDEX`, a `MEANS` filter that plans
//! the derived funnel — free predicates, index pre-filter, chunk-fed
//! verify — against the deterministic mock model, so it runs without any
//! setup.
//!
//! Run with: `cargo run --example meetings`

use std::sync::Arc;

use semcast::model::MockModel;
use semcast::semcast_context;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));

    // Transcript 1 runs long enough (>384 words) to be split into several
    // chunks, so the verify stage demonstrably reads excerpts, not the
    // whole document.
    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'atlas planning',   CAST('2026-05-12 10:00:00' AS TIMESTAMP),
              'we agreed to ship offline sync in Q3, pending the sync-engine work. '
              || repeat('the team walked through rollout details, conflict handling, and the storage budget. ', 60)),
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

    semcast::sql(&ctx, "CREATE SEMANTIC INDEX ON meetings(transcript)")
        .await?
        .collect()
        .await?;

    // Eventually: transcript MEANS '...' WITH RECALL 0.9 — recall bounds
    // arrive with calibration.
    let df = semcast::sql(
        &ctx,
        "SELECT meeting_id, title, held_at
         FROM meetings
         WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
           AND transcript MEANS 'discussed the launch of offline sync in Atlas'
         ORDER BY held_at",
    )
    .await?;

    println!(
        "Optimized plan:\n{}\n",
        df.clone().into_optimized_plan()?.display_indent()
    );

    let physical = df.clone().create_physical_plan().await?;
    println!(
        "Physical plan (the derived funnel, with the verify-stage call estimate):\n{}",
        datafusion::physical_plan::displayable(physical.as_ref()).indent(false)
    );

    println!("Results:");
    df.show().await?;

    Ok(())
}
