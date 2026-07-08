//! Live end-to-end tests against a local Ollama server. Ignored by default:
//!
//! ```sh
//! ollama pull gemma4:31b        # or export SEMCAST_OLLAMA_MODEL=<model>
//! ollama pull nomic-embed-text  # embeddings, for the index test
//! cargo test --test live_ollama -- --ignored --nocapture
//! ```

use std::sync::Arc;

use semcast::model::OllamaProvider;
use semcast::{IndexOptions, create_semantic_index, semcast_context};

#[tokio::test]
#[ignore = "requires a running Ollama server with a pulled model"]
async fn means_filter_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_owned());
    let ctx = semcast_context(Arc::new(OllamaProvider::new(model)));

    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'we agreed to ship offline sync in the third quarter'),
             (2, 'status round about the cafeteria menu, nothing else')
         ) AS t(meeting_id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE transcript MEANS 'discussed shipping an offline sync feature'
         ORDER BY meeting_id",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("live ollama verify kept {rows} of 2 rows");
    // Meeting 1 unambiguously matches; any reasonable model keeps it and
    // drops the cafeteria meeting.
    assert_eq!(
        rows, 1,
        "expected exactly the offline-sync meeting to survive"
    );
}

/// Real embeddings end-to-end: build a Lance index with nomic-embed-text,
/// plan the funnel, and verify the right row survives it.
#[tokio::test]
#[ignore = "requires a running Ollama server with gemma4:31b and nomic-embed-text pulled"]
async fn semantic_index_funnel_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_owned());
    let ctx = semcast_context(Arc::new(OllamaProvider::new(model)));

    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'we agreed to ship offline sync in the third quarter'),
             (2, 'status round about the cafeteria menu, nothing else'),
             (3, 'the quarterly budget review ran long, no engineering topics')
         ) AS t(meeting_id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            path: Some(dir.path().join("meetings.transcript.lance")),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let df = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE transcript MEANS 'discussed shipping an offline sync feature'",
    )
    .await
    .unwrap();

    let physical = df.clone().create_physical_plan().await.unwrap();
    let display = datafusion::physical_plan::displayable(physical.as_ref())
        .indent(true)
        .to_string();
    println!("physical plan:\n{display}");
    assert!(display.contains("IndexScanExec"), "plan:\n{display}");

    let batches = df.collect().await.unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("live ollama funnel kept {rows} of 3 rows");
    assert_eq!(
        rows, 1,
        "expected exactly the offline-sync meeting to survive the funnel"
    );
}

/// `WITH RECALL` end-to-end: the scan labels a sample with the live model
/// and calibrates its floor before pruning.
#[tokio::test]
#[ignore = "requires a running Ollama server with gemma4:31b and nomic-embed-text pulled"]
async fn calibrated_funnel_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_owned());
    let ctx = semcast_context(Arc::new(OllamaProvider::new(model)));

    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'we agreed to ship offline sync in the third quarter'),
             (2, 'status round about the cafeteria menu, nothing else'),
             (3, 'the quarterly budget review ran long, no engineering topics')
         ) AS t(meeting_id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "meetings",
        "transcript",
        IndexOptions {
            path: Some(dir.path().join("meetings.transcript.lance")),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let df = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE transcript MEANS 'discussed shipping an offline sync feature'
         WITH RECALL 0.9",
    )
    .await
    .unwrap();

    let physical = df.clone().create_physical_plan().await.unwrap();
    let display = datafusion::physical_plan::displayable(physical.as_ref())
        .indent(true)
        .to_string();
    println!("physical plan:\n{display}");
    assert!(
        display.contains("floor=calibrated(recall≥0.90"),
        "plan:\n{display}"
    );

    let batches = df.collect().await.unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("live ollama calibrated funnel kept {rows} of 3 rows");
    assert_eq!(
        rows, 1,
        "expected exactly the offline-sync meeting to survive the calibrated funnel"
    );
}
