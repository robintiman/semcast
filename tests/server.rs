//! End-to-end tests for the pgwire server: a real TCP round trip with
//! tokio-postgres as the client, against the deterministic mock model.
//! The live variant at the bottom follows the `live_ollama.rs` convention.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use semcast::SemcastContextBuilder;
use semcast::model::{MockModel, ModelProvider};
use semcast::server::{QueryEngine, serve};
use tokio_postgres::{AsyncMessage, NoTls, SimpleQueryMessage};

/// Serve a fresh context on an ephemeral port; return a connected client
/// and the notices its connection receives.
async fn connect(
    model: Arc<dyn ModelProvider>,
) -> (tokio_postgres::Client, Arc<Mutex<Vec<String>>>) {
    let index_root = tempfile::tempdir().unwrap();
    let ctx = SemcastContextBuilder::new(model)
        .with_index_root(index_root.path())
        .with_information_schema(true)
        .build();
    std::mem::forget(index_root); // keep Lance datasets alive for the test
    let engine = Arc::new(QueryEngine::new(Arc::new(ctx)));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, engine));

    let (client, mut connection) = tokio_postgres::connect(
        &format!(
            "host=127.0.0.1 port={} user=test dbname=semcast",
            addr.port()
        ),
        NoTls,
    )
    .await
    .unwrap();

    let notices = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&notices);
    tokio::spawn(async move {
        let mut messages = futures::stream::poll_fn(move |cx| connection.poll_message(cx));
        while let Some(message) = messages.next().await {
            match message {
                Ok(AsyncMessage::Notice(notice)) => {
                    sink.lock().unwrap().push(notice.message().to_owned());
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });
    (client, notices)
}

fn single_column(messages: &[SimpleQueryMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(row) => Some(row.get(0).unwrap().to_owned()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn calibrated_funnel_round_trips_with_progress_notices() {
    let (client, notices) = connect(Arc::new(MockModel::answering_yes_to(["sync"]))).await;

    // Corpus in the mock-embedding regime: one short match, one long
    // multi-chunk miss (same shape as tests/index.rs calibration tests).
    let long_doc = "abcdefghijkl ".repeat(800);
    client
        .simple_query(&format!(
            "CREATE TABLE meetings AS
             SELECT * FROM (VALUES
                 (1, 'a', 'sync'),
                 (2, 'b', '{long_doc}'),
                 (3, 'c', CAST(NULL AS VARCHAR))
             ) AS t(meeting_id, title, transcript)",
        ))
        .await
        .unwrap();
    client
        .simple_query("CREATE SEMANTIC INDEX ON meetings(transcript)")
        .await
        .unwrap();

    let rows = client
        .simple_query(
            "SELECT meeting_id FROM meetings
             WHERE transcript MEANS 'sync'
             WITH RECALL 0.9",
        )
        .await
        .unwrap();
    assert_eq!(single_column(&rows), vec!["1"]);

    let notices = notices.lock().unwrap().clone();
    assert!(
        notices
            .iter()
            .any(|n| n.starts_with("funnel: IndexScanExec")),
        "index stage announced, got: {notices:?}",
    );
    assert!(
        notices.iter().any(|n| n.starts_with("funnel: VerifyExec")),
        "verify stage announced, got: {notices:?}",
    );
    assert!(
        notices
            .iter()
            .any(|n| n.starts_with("funnel done") && n.contains("model calls")),
        "final totals reported, got: {notices:?}",
    );
}

#[tokio::test]
async fn multi_statement_strings_split_and_chatter_is_tolerated() {
    let (client, _) = connect(Arc::new(MockModel::default())).await;

    let messages = client
        .simple_query("SET application_name = 'psql'; BEGIN; SELECT 1; SELECT 2; COMMIT")
        .await
        .unwrap();
    assert_eq!(single_column(&messages), vec!["1", "2"]);
    let completions = messages
        .iter()
        .filter(|m| matches!(m, SimpleQueryMessage::CommandComplete(_)))
        .count();
    assert_eq!(completions, 5, "every statement completes: {messages:?}");
}

#[tokio::test]
async fn pg_catalog_queries_fail_politely() {
    let (client, _) = connect(Arc::new(MockModel::default())).await;

    let error = client
        .simple_query("SELECT * FROM pg_catalog.pg_tables")
        .await
        .unwrap_err();
    let db_error = error.as_db_error().expect("server-side error");
    assert!(
        db_error.message().contains("pg_catalog introspection"),
        "friendly message, got: {}",
        db_error.message(),
    );
}

#[tokio::test]
async fn errors_abort_the_rest_of_a_multi_statement_string() {
    let (client, _) = connect(Arc::new(MockModel::default())).await;

    let error = client
        .simple_query("SELECT 1; SELECT * FROM no_such_table; SELECT 2")
        .await
        .unwrap_err();
    assert!(error.as_db_error().is_some(), "statement error surfaced");
}

#[tokio::test]
#[ignore = "requires a running Ollama server with a pulled model"]
async fn live_ollama_means_query_over_the_wire() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_owned());
    let (client, notices) = connect(Arc::new(semcast::model::OllamaProvider::new(model))).await;

    client
        .simple_query(
            "CREATE TABLE meetings AS
             SELECT * FROM (VALUES
                 (1, 'we agreed to ship offline sync in the third quarter'),
                 (2, 'status round about the cafeteria menu, nothing else')
             ) AS t(meeting_id, transcript)",
        )
        .await
        .unwrap();
    let rows = client
        .simple_query(
            "SELECT meeting_id FROM meetings
             WHERE transcript MEANS 'discussed shipping an offline sync feature'
             ORDER BY meeting_id",
        )
        .await
        .unwrap();
    assert_eq!(single_column(&rows), vec!["1"]);
    assert!(
        notices
            .lock()
            .unwrap()
            .iter()
            .any(|n| n.contains("model calls")),
        "progress notices arrive from a live model",
    );
}
