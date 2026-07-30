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
    let ctx = Arc::new(ctx);
    // The served surface includes the shim, so tests get it too.
    semcast::server::pg_catalog::install(&ctx).await.unwrap();
    let engine = Arc::new(QueryEngine::new(ctx));

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
async fn csv_ingestion_round_trips_over_the_wire() {
    let (client, _) = connect(Arc::new(MockModel::default())).await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meetings.csv");
    std::fs::write(&path, "meeting_id,transcript\n1,planning\n2,standup\n").unwrap();

    let messages = client
        .simple_query(&format!(
            "CREATE EXTERNAL TABLE meetings STORED AS CSV LOCATION '{}'",
            path.display(),
        ))
        .await
        .unwrap();
    assert!(
        messages
            .iter()
            .any(|m| matches!(m, SimpleQueryMessage::CommandComplete(_))),
        "external table DDL completes: {messages:?}",
    );

    let rows = client
        .simple_query("SELECT meeting_id FROM meetings ORDER BY meeting_id")
        .await
        .unwrap();
    assert_eq!(single_column(&rows), vec!["1", "2"]);

    // Path literal, no DDL: the DuckDB-style direct file query.
    let rows = client
        .simple_query(&format!(
            "SELECT transcript FROM '{}' ORDER BY meeting_id DESC",
            path.display(),
        ))
        .await
        .unwrap();
    assert_eq!(single_column(&rows), vec!["standup", "planning"]);
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
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
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

/// Replay of dbt's table materialization, statement for statement: stage the
/// model under a temp name, park the live relation as a backup, swap the
/// staged one in, drop the backup. dbt then re-reads `pg_tables` to refresh
/// its relation cache.
#[tokio::test]
async fn a_dbt_table_materialization_round_trips() {
    let (client, _) = connect(Arc::new(MockModel::default())).await;

    client
        .simple_query("CREATE SCHEMA IF NOT EXISTS analytics")
        .await
        .unwrap();
    client
        .simple_query(
            r#"CREATE TABLE "datafusion"."analytics"."model" AS (SELECT 1 AS id, 'first' AS name)"#,
        )
        .await
        .unwrap();

    // Second run: the relation already exists, so dbt takes the rename path.
    for statement in [
        r#"CREATE TABLE "datafusion"."analytics"."model__dbt_tmp" AS (SELECT 2 AS id, 'second' AS name)"#,
        r#"ALTER TABLE "datafusion"."analytics"."model" RENAME TO "model__dbt_backup""#,
        r#"ALTER TABLE "datafusion"."analytics"."model__dbt_tmp" RENAME TO "model""#,
        r#"DROP TABLE IF EXISTS "datafusion"."analytics"."model__dbt_backup" CASCADE"#,
    ] {
        client.simple_query(statement).await.unwrap();
    }

    let rows = client
        .simple_query(r#"SELECT name FROM "datafusion"."analytics"."model""#)
        .await
        .unwrap();
    assert_eq!(single_column(&rows), vec!["second"]);

    let tables = client
        .simple_query("SELECT tablename FROM pg_tables WHERE schemaname ILIKE 'analytics'")
        .await
        .unwrap();
    assert_eq!(
        single_column(&tables),
        vec!["model"],
        "the staging and backup relations are gone from the catalog",
    );
}

/// dbt's `delete+insert` incremental strategy emits a subquery predicate that
/// DataFusion plans without the predicate — every row goes. The server must
/// refuse it rather than empty the model.
#[tokio::test]
async fn a_delete_that_would_empty_the_table_is_refused_over_the_wire() {
    let (client, _) = connect(Arc::new(MockModel::default())).await;

    for setup in [
        "CREATE TABLE target AS SELECT * FROM (VALUES (1),(2),(3)) AS v(id)",
        "CREATE TABLE staging AS SELECT * FROM (VALUES (1)) AS v(id)",
    ] {
        client.simple_query(setup).await.unwrap();
    }

    let error = client
        .simple_query("DELETE FROM target WHERE (id) IN (SELECT DISTINCT id FROM staging)")
        .await
        .expect_err("the delete is refused");
    let db_error = error.as_db_error().expect("server-side error");
    assert!(
        db_error.message().contains("subquery"),
        "the error names the cause, got: {}",
        db_error.message(),
    );

    let rows = client
        .simple_query("SELECT id FROM target ORDER BY id")
        .await
        .unwrap();
    assert_eq!(single_column(&rows), vec!["1", "2", "3"]);
}

/// dbt's relation-dependency walk casts to `regclass`, which DataFusion has
/// no type for; the server answers it with the right shape and no rows.
#[tokio::test]
async fn the_relation_dependency_walk_answers_empty() {
    let (client, _) = connect(Arc::new(MockModel::default())).await;

    let rows = client
        .simple_query(
            "select distinct dependent_namespace.nspname as dependent_schema
             from pg_class as dependent_class
             join pg_depend as d on d.classid = 'pg_rewrite'::regclass",
        )
        .await
        .unwrap();
    assert!(single_column(&rows).is_empty());
}
