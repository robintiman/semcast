//! Dogfooding semcast on real unstructured data: 5,000 U.S. consumer-finance
//! complaints from the CFPB Consumer Complaint Database, each a free-text
//! `narrative` beside structured columns (`product`, `company`, `state`,
//! `date_received`). It's the README's "support-ticket mining" case on real
//! prose — a cheap structured pre-filter runs first, then `MEANS` verifies the
//! survivors.
//!
//! Data: `data/cfpb_complaints.csv` (pulled from the CFPB search API).
//!
//! Run with: `cargo run --example complaints`
//!
//! Needs a running Ollama with both models pulled: gemma4:31b answers `MEANS`
//! and constrained extraction, nomic-embed-text builds the semantic index. The
//! full funnel runs: cheap filter → a calibrated index pre-filter under
//! `WITH RECALL 0.9` → chunk-fed verify.
//!
//! A `CREATE SEMANTIC TYPE` + `CAST(narrative AS ComplaintFacts)` block then
//! extracts structured columns from the same survivors — constrained decoding,
//! field pushdown, and `TOGETHER` co-generation — stacked above the funnel.

use std::sync::Arc;

use semcast::model::{ModelProvider, OllamaProvider};
use semcast::semcast_context;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    semcast::telemetry::init();
    let model: Arc<dyn ModelProvider> = Arc::new(OllamaProvider::new("gemma4:e4b")); // embeds with nomic-embed-text
    let ctx = semcast_context(model);

    // Mount the CSV as an external table; paths resolve from the repo root.
    ctx.sql(
        "CREATE EXTERNAL TABLE complaints
         STORED AS CSV LOCATION 'data/cfpb_complaints.csv'",
    )
    .await?
    .collect()
    .await?;

    // Chunk + embed every narrative once, giving the funnel its cheap stage.
    semcast::sql(&ctx, "CREATE SEMANTIC INDEX ON complaints(narrative)")
        .await?
        .collect()
        .await?;

    // The cheap structured predicates (product, recent date) run first and cut
    // 5,000 rows to ~570 for free; only those reach verify. The index then lets
    // survivors pass a calibrated similarity floor (`WITH RECALL`), so the model
    // reads far fewer — and only each survivor's top-3 chunks.
    let df = semcast::sql(
        &ctx,
        "SELECT complaint_id, company, state, date_received
         FROM complaints
         WHERE product = 'Debt collection'
           AND date_received >= '2026-06-01'
           AND narrative MEANS 'the consumer reports identity theft or an account they never opened'
         ORDER BY date_received
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

    println!("Matches:");
    df.show().await?;

    // ── Typed extraction ────────────────────────────────────────────────
    // A semantic type names a recurring extraction; the planner synthesizes the
    // prompt, constrains decoding to the schema, and hands back ordinary SQL
    // columns. `TOGETHER` co-generates the dispute stage and the sentence that
    // evidences it.
    semcast::sql(
        &ctx,
        "CREATE SEMANTIC TYPE ComplaintFacts AS (
             companies TEXT[] 'companies or creditors named in the complaint',
             relief    ONEOF(none, validation, correction, removal, refund, other)
                       'the primary resolution the consumer is asking for',
             TOGETHER (
                 dispute_stage LEVEL(none, disputed, escalated, unresolved)
                               'how far the consumer has taken the dispute',
                 stage_quote   TEXT 'the narrative sentence that shows that stage'
             )
         )",
    )
    .await?
    .collect()
    .await?;

    // `SemExtract` stacks *above* the same funnel, so only the survivors of the
    // cheap predicates and `MEANS` are ever extracted. The query references
    // `relief` and `dispute_stage`, so field pushdown generates just those (plus
    // `dispute_stage`'s `TOGETHER` sibling) and never touches `companies`.
    let typed = semcast::sql(
        &ctx,
        "SELECT complaint_id,
                CAST(narrative AS ComplaintFacts).relief        AS relief,
                CAST(narrative AS ComplaintFacts).dispute_stage AS dispute_stage
         FROM complaints
         WHERE product = 'Debt collection'
           AND date_received >= '2026-06-01'
           AND narrative MEANS 'the consumer reports identity theft or an account they never opened'
         ORDER BY complaint_id
         WITH RECALL 0.9",
    )
    .await?;

    println!(
        "\nTyped extraction — optimized plan (SemExtract above the funnel):\n{}\n",
        typed.clone().into_optimized_plan()?.display_indent()
    );
    println!("Extracted facts:");
    typed.show().await?;

    Ok(())
}
