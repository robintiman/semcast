//! The README walk-through, as far as the roadmap takes it: a semantic
//! index built with `CREATE SEMANTIC INDEX`, a `MEANS` filter with a
//! `WITH RECALL` target that plans the derived funnel — free predicates,
//! calibrated index pre-filter, chunk-fed verify — then `CREATE SEMANTIC
//! TYPE` + `CAST(... AS MeetingFacts)` typed extraction stacked above the
//! funnel. All against the deterministic mock, so it runs without setup.
//!
//! Run with: `cargo run --example meetings`

use std::sync::Arc;

use semcast::model::MockModel;
use semcast::semcast_context;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    // One mock answers both surfaces: the yes/no `MEANS` verdict (via the
    // "offline sync" needle) and typed extraction (a JSON object keyed on the
    // transcript's content).
    let model = Arc::new(
        MockModel::answering_json_with(|req| {
            let stage = if req.input.contains("agreed to ship") {
                "shipped"
            } else if req.input.contains("future idea") {
                "idea"
            } else {
                "none"
            };
            serde_json::json!({
                "launch_stage": stage,
                "stage_quote": "the transcript line that shows the stage",
                "products": ["offline sync"],
                "decisions": [],
            })
        })
        .also_answering_yes_to(["offline sync"]),
    );
    let ctx = semcast_context(model);

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

    // WITH RECALL: the index threshold is calibrated at execution time by
    // labeling a sample of the date-surviving rows, instead of trusting a
    // fixed floor.
    let df = semcast::sql(
        &ctx,
        "SELECT meeting_id, title, held_at
         FROM meetings
         WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
           AND transcript MEANS 'discussed the launch of offline sync in Atlas'
         ORDER BY held_at
         WITH RECALL 0.9",
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

    // ── Typed extraction ────────────────────────────────────────────────
    // A semantic type names a recurring extraction; the planner synthesizes
    // the prompt, constrains decoding to the schema, and turns the extracted
    // columns into ordinary SQL. `TOGETHER` co-generates the launch stage and
    // its evidence.
    semcast::sql(
        &ctx,
        "CREATE SEMANTIC TYPE MeetingFacts AS (
             products  TEXT[]   'product names discussed in this meeting',
             decisions TEXT[]   'concrete decisions that were made',
             TOGETHER (
                 launch_stage ONEOF(none, idea, planned, scheduled, shipped)
                              'the furthest launch stage discussed',
                 stage_quote  TEXT 'the transcript line that shows that stage'
             )
         )",
    )
    .await?
    .collect()
    .await?;

    // Extraction runs *after* the funnel: `SemExtract` sits above the
    // `SemFilter`, so only the rows surviving the date predicate and the
    // `MEANS` filter are ever extracted. Only `launch_stage` is referenced, so
    // field pushdown generates just it and its `TOGETHER` sibling.
    let typed = semcast::sql(
        &ctx,
        "SELECT meeting_id, CAST(transcript AS MeetingFacts).launch_stage AS launch_stage
         FROM meetings
         WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
           AND transcript MEANS 'offline sync'
         ORDER BY meeting_id",
    )
    .await?;

    println!(
        "\nTyped extraction — optimized plan (SemExtract above the funnel):\n{}\n",
        typed.clone().into_optimized_plan()?.display_indent()
    );
    println!("Extracted results:");
    typed.show().await?;

    Ok(())
}
