//! End-to-end tests for batch jobs: `SUBMIT` returns immediately, the work
//! runs detached from the connection, and status and results come back as
//! ordinary SQL. Same real-TCP + tokio-postgres shape as `server.rs`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use semcast::SemcastContextBuilder;
use semcast::model::{MockModel, ModelProvider};
use semcast::server::{JobRegistry, QueryEngine, jobs, serve};
use tokio_postgres::{NoTls, SimpleQueryMessage};

const MATCHING: &str = "we agreed to ship offline sync in Q3";
const OTHER: &str = "nothing notable happened";

struct Server {
    client: tokio_postgres::Client,
    jobs: Arc<JobRegistry>,
}

/// Serve a fresh jobs-enabled context on an ephemeral port.
async fn connect(model: Arc<dyn ModelProvider>) -> Server {
    connect_with_slots(model, 4).await
}

/// [`connect`] with the concurrency ceiling as a parameter, for tests about
/// what happens to a job that is still waiting for a slot.
async fn connect_with_slots(model: Arc<dyn ModelProvider>, max_concurrent: usize) -> Server {
    let index_root = tempfile::tempdir().unwrap();
    let jobs_root = tempfile::tempdir().unwrap();
    let ctx = Arc::new(
        SemcastContextBuilder::new(model)
            .with_index_root(index_root.path())
            .with_information_schema(true)
            .build(),
    );
    let registry = Arc::new(JobRegistry::new(jobs_root.path(), max_concurrent).unwrap());
    jobs::register(&ctx, &registry).unwrap();
    // Keep Lance datasets and job artifacts alive for the whole test.
    std::mem::forget(index_root);
    std::mem::forget(jobs_root);

    let engine = Arc::new(QueryEngine::new(ctx).with_jobs(Arc::clone(&registry)));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, engine));

    let (client, connection) = tokio_postgres::connect(
        &format!(
            "host=127.0.0.1 port={} user=test dbname=semcast",
            addr.port()
        ),
        NoTls,
    )
    .await
    .unwrap();
    tokio::spawn(connection);

    Server {
        client,
        jobs: registry,
    }
}

impl Server {
    async fn query(&self, sql: &str) -> Vec<SimpleQueryMessage> {
        self.client.simple_query(sql).await.unwrap()
    }

    /// Create the two-row `meetings` table the tests filter over.
    async fn seed(&self) {
        self.query(&format!(
            "CREATE TABLE meetings AS
             SELECT * FROM (VALUES (1, '{MATCHING}'), (2, '{OTHER}')) AS t(meeting_id, transcript)",
        ))
        .await;
    }

    async fn submit(&self, sql: &str) -> String {
        let messages = self.query(&format!("SUBMIT {sql}")).await;
        column(&messages, 0)
            .into_iter()
            .next()
            .expect("SUBMIT answers with a job id")
    }

    /// Poll until the job leaves `queued`. Submission only registers the job;
    /// the task picks up a concurrency slot a moment later.
    async fn wait_until_started(&self, id: &str) -> String {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let status = self.status(id).await;
            if status != "queued" {
                return status;
            }
            assert!(Instant::now() < deadline, "job {id} never started");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Poll until the job reaches a terminal status, or fail after `timeout`.
    async fn wait_for(&self, id: &str, timeout: Duration) -> String {
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.status(id).await;
            if !matches!(status.as_str(), "queued" | "running") {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "job {id} still {status} after {timeout:?}",
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn status(&self, id: &str) -> String {
        let messages = self
            .query(&format!(
                "SELECT status FROM semcast_jobs WHERE job_id = '{id}'"
            ))
            .await;
        column(&messages, 0)
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("job {id} is in semcast_jobs"))
    }
}

/// The server-side message of a failed statement. `Error::to_string` only
/// says "db error", which makes assertions useless.
async fn error_message(server: &Server, sql: &str) -> String {
    let error = server.client.simple_query(sql).await.unwrap_err();
    error
        .as_db_error()
        .unwrap_or_else(|| panic!("`{sql}` failed server-side, got: {error}"))
        .message()
        .to_owned()
}

/// Values of column `index` across every row in the response.
fn column(messages: &[SimpleQueryMessage], index: usize) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(row) => Some(row.get(index).unwrap_or("").to_owned()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn submitted_query_runs_detached_and_its_rows_come_back() {
    let server = connect(Arc::new(MockModel::answering_yes_to(["offline sync"]))).await;
    server.seed().await;

    let id = server
        .submit("SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'")
        .await;
    assert!(id.starts_with("job_"), "got a job id: {id}");
    assert_eq!(
        server.wait_for(&id, Duration::from_secs(10)).await,
        "succeeded"
    );

    let rows = column(
        &server
            .query(&format!("SELECT meeting_id FROM job_result('{id}')"))
            .await,
        0,
    );
    assert_eq!(
        rows,
        vec!["1"],
        "the same row the synchronous query returns"
    );

    // The funnel counts the run spent land as columns, not just as text.
    let messages = server
        .query(&format!(
            "SELECT rows, model_calls FROM semcast_jobs WHERE job_id = '{id}'"
        ))
        .await;
    assert_eq!(column(&messages, 0), vec!["1"]);
    assert_eq!(
        column(&messages, 1),
        vec!["2"],
        "one call per candidate row"
    );
}

#[tokio::test]
async fn the_connection_is_free_while_the_job_runs() {
    // Slow enough that the job is observably mid-flight, fast enough that the
    // test doesn't drag.
    let model =
        MockModel::answering_yes_to(["offline sync"]).with_latency(Duration::from_millis(400));
    let server = connect(Arc::new(model)).await;
    server.seed().await;

    let id = server
        .submit("SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'")
        .await;

    // This is the whole point of the feature: the submitting connection is
    // usable immediately, while the job is still running.
    assert_eq!(server.wait_until_started(&id).await, "running");
    let unrelated = column(&server.query("SELECT 41 + 1").await, 0);
    assert_eq!(unrelated, vec!["42"]);

    assert_eq!(
        server.wait_for(&id, Duration::from_secs(10)).await,
        "succeeded"
    );

    // Progress was recorded while it ran, and mirrored to progress.log.
    let record = server.jobs.get(&id).unwrap();
    let progress = record.progress.expect("a funnel line was captured");
    assert!(progress.contains("model calls"), "got: {progress}");
    let log = std::fs::read_to_string(server.jobs.dir(&id).join("progress.log")).unwrap();
    assert!(
        log.contains("funnel"),
        "progress.log holds the lines: {log}"
    );
}

#[tokio::test]
async fn a_failing_statement_lands_as_a_failed_job() {
    let server = connect(Arc::new(MockModel::default())).await;

    let id = server.submit("SELECT * FROM no_such_table").await;
    assert_eq!(
        server.wait_for(&id, Duration::from_secs(10)).await,
        "failed"
    );

    let messages = server
        .query(&format!(
            "SELECT error, result_path FROM semcast_jobs WHERE job_id = '{id}'"
        ))
        .await;
    let error = column(&messages, 0).into_iter().next().unwrap();
    assert!(
        error.contains("no_such_table"),
        "the error is readable: {error}"
    );
    assert_eq!(
        column(&messages, 1).into_iter().next().unwrap(),
        "",
        "a failed job exposes no result",
    );

    let err = error_message(&server, &format!("SELECT * FROM job_result('{id}')")).await;
    assert!(err.contains("produced no result"), "got: {err}");
}

#[tokio::test]
async fn ddl_jobs_report_a_command_tag_and_no_result() {
    let server = connect(Arc::new(MockModel::answering_yes_to(["offline sync"]))).await;
    server.seed().await;

    let id = server
        .submit("CREATE SEMANTIC INDEX ON meetings(transcript)")
        .await;
    assert_eq!(
        server.wait_for(&id, Duration::from_secs(30)).await,
        "succeeded"
    );

    let messages = server
        .query(&format!(
            "SELECT command_tag, rows, result_path FROM semcast_jobs WHERE job_id = '{id}'"
        ))
        .await;
    assert_eq!(
        column(&messages, 0).into_iter().next().unwrap(),
        "CREATE SEMANTIC INDEX",
    );
    assert_eq!(column(&messages, 1).into_iter().next().unwrap(), "");
    assert_eq!(column(&messages, 2).into_iter().next().unwrap(), "");

    // And the index it built is usable by a later query.
    let rows = column(
        &server
            .query("SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'")
            .await,
        0,
    );
    assert_eq!(rows, vec!["1"]);
}

#[tokio::test]
async fn cancel_job_stops_a_running_job() {
    let model = MockModel::answering_yes_to(["offline sync"]).with_latency(Duration::from_secs(30));
    let server = connect(Arc::new(model)).await;
    server.seed().await;

    let id = server
        .submit("SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'")
        .await;
    assert_eq!(server.wait_until_started(&id).await, "running");

    server.query(&format!("CANCEL JOB '{id}'")).await;
    assert_eq!(server.status(&id).await, "cancelled");

    // Cancelling twice is an error, not a silent no-op.
    let err = error_message(&server, &format!("CANCEL JOB '{id}'")).await;
    assert!(err.contains("already finished"), "got: {err}");
}

#[tokio::test]
async fn cancelling_a_queued_job_keeps_it_from_ever_running() {
    // One slot, so the second job is still waiting when it is cancelled.
    let model = Arc::new(
        MockModel::answering_yes_to(["offline sync"]).with_latency(Duration::from_millis(600)),
    );
    let server = connect_with_slots(Arc::clone(&model) as Arc<dyn ModelProvider>, 1).await;
    server.seed().await;

    let query = "SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'";
    let running = server.submit(query).await;
    assert_eq!(server.wait_until_started(&running).await, "running");
    let queued = server.submit(query).await;
    assert_eq!(server.status(&queued).await, "queued", "the slot is taken");

    server.query(&format!("CANCEL JOB '{queued}'")).await;
    assert_eq!(server.status(&queued).await, "cancelled");

    // The slot frees here. The cancelled job must not pick it up: it was
    // cancelled precisely so it would never spend a model call.
    assert_eq!(
        server.wait_for(&running, Duration::from_secs(10)).await,
        "succeeded"
    );
    let spent = model.completion_calls();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        model.completion_calls(),
        spent,
        "the cancelled job took the free slot and spent model calls anyway",
    );

    let messages = server
        .query(&format!(
            "SELECT status, started_at FROM semcast_jobs WHERE job_id = '{queued}'"
        ))
        .await;
    assert_eq!(column(&messages, 0), vec!["cancelled"]);
    assert_eq!(
        column(&messages, 1),
        vec![""],
        "it never started, so it has no start time",
    );
}

#[tokio::test]
async fn jobs_are_queryable_as_an_ordinary_table() {
    let server = connect(Arc::new(MockModel::default())).await;

    let ok = server.submit("SELECT 1").await;
    let bad = server.submit("SELECT * FROM nope").await;
    server.wait_for(&ok, Duration::from_secs(10)).await;
    server.wait_for(&bad, Duration::from_secs(10)).await;

    let failed = column(
        &server
            .query("SELECT job_id FROM semcast_jobs WHERE status = 'failed'")
            .await,
        0,
    );
    assert_eq!(failed, vec![bad], "the table filters like any other");

    let ordered = column(
        &server
            .query("SELECT job_id FROM semcast_jobs ORDER BY submitted_at DESC, job_id DESC")
            .await,
        0,
    );
    assert_eq!(ordered.len(), 2, "and sorts like any other");
}

#[tokio::test]
async fn malformed_job_statements_explain_themselves() {
    let server = connect(Arc::new(MockModel::default())).await;

    let err = error_message(&server, "CANCEL JOB 'a' extra").await;
    assert!(err.contains("CANCEL JOB '<job_id>'"), "got: {err}");

    let err = error_message(&server, "CANCEL JOB 'job_does_not_exist'").await;
    assert!(err.contains("no such job"), "got: {err}");

    // A bare SUBMIT is not a job statement, so it falls through to the engine
    // and fails there as ordinary invalid SQL.
    assert!(server.client.simple_query("SUBMIT").await.is_err());
}
