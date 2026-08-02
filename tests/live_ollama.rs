//! Live end-to-end tests against a local Ollama server. Ignored by default:
//!
//! ```sh
//! ollama pull gemma4:e4b        # or export SEMCAST_OLLAMA_MODEL=<model>
//! ollama pull nomic-embed-text  # embeddings, for the index test
//! cargo test --test live_ollama -- --ignored --nocapture
//! ```

use std::sync::Arc;

use semcast::model::OllamaProvider;
use semcast::{IndexOptions, create_semantic_index, semcast_context};

#[tokio::test]
#[ignore = "requires a running Ollama server with a pulled model"]
async fn means_filter_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
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
#[ignore = "requires a running Ollama server with gemma4:e4b and nomic-embed-text pulled"]
async fn semantic_index_funnel_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
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
#[ignore = "requires a running Ollama server with gemma4:e4b and nomic-embed-text pulled"]
async fn calibrated_funnel_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
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

/// `RELEVANCE TO` end-to-end: real embeddings pick the candidates, a real
/// model scores them, and the best match sorts first.
#[tokio::test]
#[ignore = "requires a running Ollama server with gemma4:e4b and nomic-embed-text pulled"]
async fn rank_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
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
        "SELECT meeting_id, relevance(transcript, 'shipping an offline sync feature') AS score
         FROM meetings
         ORDER BY score DESC LIMIT 3",
    )
    .await
    .unwrap();

    let physical = df.clone().create_physical_plan().await.unwrap();
    let display = datafusion::physical_plan::displayable(physical.as_ref())
        .indent(true)
        .to_string();
    println!("physical plan:\n{display}");
    assert!(display.contains("SemRankExec"), "plan:\n{display}");

    let batches = df.collect().await.unwrap();
    println!(
        "live ollama ranking:\n{}",
        datafusion::arrow::util::pretty::pretty_format_batches(&batches).unwrap()
    );
    let ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("meeting_id is Int64")
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(
        ids.first(),
        Some(&1),
        "the offline-sync meeting must rank first"
    );
}

/// Classify end-to-end: a three-branch `CASE` costs one real model call per
/// row, and each row lands in the branch it belongs to.
#[tokio::test]
#[ignore = "requires a running Ollama server with gemma4:e4b and nomic-embed-text pulled"]
async fn classify_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
    let ctx = semcast_context(Arc::new(OllamaProvider::new(model)));

    ctx.sql(
        "CREATE TABLE tickets AS
         SELECT * FROM (VALUES
             (1, 'this is completely unacceptable, I demand a refund immediately'),
             (2, 'what does the enterprise plan cost per seat per month?'),
             (3, 'thanks for your help, that answered my question')
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let df = semcast::sql(
        &ctx,
        "SELECT id, CASE WHEN body MEANS 'the customer is angry or upset' THEN 'escalate'
                         WHEN body MEANS 'asks about pricing or cost'     THEN 'sales'
                         ELSE 'other' END AS route
         FROM tickets",
    )
    .await
    .unwrap();

    let physical = df.clone().create_physical_plan().await.unwrap();
    let display = datafusion::physical_plan::displayable(physical.as_ref())
        .indent(true)
        .to_string();
    println!("physical plan:\n{display}");
    assert!(display.contains("SemClassifyExec"), "plan:\n{display}");

    let batches = df.collect().await.unwrap();
    println!(
        "live ollama routing:\n{}",
        datafusion::arrow::util::pretty::pretty_format_batches(&batches).unwrap()
    );

    let routes: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let labels = b
                .column(1)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::StringArray>()
                .expect("route is Utf8");
            (0..b.num_rows())
                .map(|i| labels.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(routes[0], "escalate");
    assert_eq!(routes[1], "sales");
    assert_eq!(routes[2], "other");
}

/// Clustering end-to-end: real embeddings group two obvious themes apart, and
/// a real model names each group.
#[tokio::test]
#[ignore = "requires a running Ollama server with gemma4:e4b and nomic-embed-text pulled"]
async fn cluster_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
    let ctx = semcast_context(Arc::new(OllamaProvider::new(model)));

    ctx.sql(
        "CREATE TABLE notes AS
         SELECT * FROM (VALUES
             (1, 'the invoice was wrong and I want a refund for the billing error'),
             (2, 'billing charged my card twice, please refund the duplicate invoice'),
             (3, 'the refund for the incorrect invoice has not arrived yet'),
             (4, 'we agreed to ship offline sync in the third quarter launch'),
             (5, 'the launch of offline sync is scheduled for the third quarter'),
             (6, 'offline sync ships at the quarterly product launch as agreed')
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    create_semantic_index(
        &ctx,
        "notes",
        "body",
        IndexOptions {
            path: Some(dir.path().join("notes.body.lance")),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let df = semcast::sql(
        &ctx,
        "SELECT topic, count(*) AS n FROM notes GROUP BY MEANING OF body INTO 2 AS topic",
    )
    .await
    .unwrap();

    let physical = df.clone().create_physical_plan().await.unwrap();
    let display = datafusion::physical_plan::displayable(physical.as_ref())
        .indent(true)
        .to_string();
    println!("physical plan:\n{display}");
    assert!(display.contains("SemClusterExec"), "plan:\n{display}");

    let batches = df.collect().await.unwrap();
    println!(
        "live ollama grouping:\n{}",
        datafusion::arrow::util::pretty::pretty_format_batches(&batches).unwrap()
    );

    let counts: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("count is Int64")
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(counts.len(), 2, "two themes, two groups");
    assert_eq!(
        counts.iter().sum::<i64>(),
        6,
        "every row lands in exactly one group"
    );
    assert!(
        counts.iter().all(|&n| n == 3),
        "the two themes should split evenly: {counts:?}"
    );
}
