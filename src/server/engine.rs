//! Statement execution decoupled from the wire protocol: takes SQL text,
//! sends the results to a [`ResultTarget`], streams progress over a channel.
//! The extended-protocol handler can reuse this unchanged later.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;
use datafusion::parquet::arrow::AsyncArrowWriter;
use datafusion::physical_plan::execute_stream;
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::Result;

use super::jobs::JobRegistry;
use super::progress::{self, ProgressEvent};

/// How often live funnel counters are checked for a progress NOTICE.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(500);

pub struct QueryEngine {
    ctx: Arc<SessionContext>,
    jobs: Option<Arc<JobRegistry>>,
}

/// Where a statement's rows should go.
pub enum ResultTarget {
    /// Buffer in memory — the interactive path, which has to hand the rows
    /// straight back to the connection.
    Memory,
    /// Stream to a Parquet file. Batch jobs use this so a result set never
    /// has to fit in RAM and outlives the connection that asked for it.
    Parquet(PathBuf),
}

pub enum StatementOutcome {
    Rows {
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    },
    Command {
        tag: String,
    },
    Written {
        path: PathBuf,
        rows: u64,
    },
}

impl QueryEngine {
    pub fn new(ctx: Arc<SessionContext>) -> Self {
        Self { ctx, jobs: None }
    }

    /// Enable `SUBMIT` / `CANCEL JOB` against `jobs`.
    pub fn with_jobs(mut self, jobs: Arc<JobRegistry>) -> Self {
        self.jobs = Some(jobs);
        self
    }

    pub fn jobs(&self) -> Option<&Arc<JobRegistry>> {
        self.jobs.as_ref()
    }

    /// Execute one already-split statement, buffering its rows. Progress
    /// events land on `events` while the query runs; send failures are
    /// ignored so a disinterested receiver never blocks execution.
    pub async fn execute_statement(
        &self,
        sql: &str,
        events: mpsc::Sender<ProgressEvent>,
    ) -> Result<StatementOutcome> {
        self.execute_into(sql, events, ResultTarget::Memory).await
    }

    /// [`Self::execute_statement`] with the destination as a parameter.
    pub async fn execute_into(
        &self,
        sql: &str,
        events: mpsc::Sender<ProgressEvent>,
        target: ResultTarget,
    ) -> Result<StatementOutcome> {
        // DDL and CTAS execute eagerly in here; their DataFrame is empty.
        let df = crate::sql(&self.ctx, sql).await?;
        let plan = df.create_physical_plan().await?;

        for line in progress::funnel_summary(&plan) {
            let _ = events.send(ProgressEvent::Plan(line)).await;
        }
        let semantic = progress::snapshot(&plan).is_semantic();

        // A statement with no result shape is a command; it still has to be
        // driven to completion, but it must not produce a result file.
        let schema = plan.schema();
        let mut sink = if schema.fields().is_empty() {
            Sink::Discard
        } else {
            Sink::new(&schema, target).await?
        };

        let mut stream = execute_stream(Arc::clone(&plan), self.ctx.task_ctx())?;
        let mut tick = tokio::time::interval(PROGRESS_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        tick.tick().await; // the first tick is immediate — skip it
        let mut last = progress::FunnelCounts::default();
        loop {
            tokio::select! {
                batch = stream.next() => match batch {
                    Some(batch) => sink.push(batch?).await?,
                    None => break,
                },
                _ = tick.tick(), if semantic => {
                    if let Some(event) = progress::snapshot_if_changed(&plan, &mut last) {
                        let _ = events.send(event).await;
                    }
                }
            }
        }
        if let Some(event) = progress::final_totals(&plan) {
            let _ = events.send(event).await;
        }

        sink.finish(schema, sql).await
    }
}

/// The open destination for one statement's rows.
enum Sink {
    /// No result shape — drive the stream, keep nothing.
    Discard,
    Memory(Vec<RecordBatch>),
    Parquet {
        writer: Box<AsyncArrowWriter<tokio::fs::File>>,
        path: PathBuf,
        rows: u64,
    },
}

impl Sink {
    async fn new(schema: &SchemaRef, target: ResultTarget) -> Result<Self> {
        match target {
            ResultTarget::Memory => Ok(Sink::Memory(Vec::new())),
            ResultTarget::Parquet(path) => {
                let file = tokio::fs::File::create(&path).await.map_err(io_error)?;
                let writer = AsyncArrowWriter::try_new(file, Arc::clone(schema), None)
                    .map_err(parquet_error)?;
                Ok(Sink::Parquet {
                    writer: Box::new(writer),
                    path,
                    rows: 0,
                })
            }
        }
    }

    async fn push(&mut self, batch: RecordBatch) -> Result<()> {
        match self {
            Sink::Discard => {}
            Sink::Memory(batches) => batches.push(batch),
            Sink::Parquet { writer, rows, .. } => {
                *rows += batch.num_rows() as u64;
                writer.write(&batch).await.map_err(parquet_error)?;
            }
        }
        Ok(())
    }

    async fn finish(self, schema: SchemaRef, sql: &str) -> Result<StatementOutcome> {
        match self {
            Sink::Discard => Ok(StatementOutcome::Command {
                tag: command_tag(sql),
            }),
            Sink::Memory(batches) => Ok(StatementOutcome::Rows { schema, batches }),
            Sink::Parquet {
                writer, path, rows, ..
            } => {
                // Closing writes the footer — without it the file is not a
                // readable Parquet file at all.
                writer.close().await.map_err(parquet_error)?;
                Ok(StatementOutcome::Written { path, rows })
            }
        }
    }
}

fn io_error(error: std::io::Error) -> crate::SemcastError {
    DataFusionError::External(Box::new(error)).into()
}

fn parquet_error(error: datafusion::parquet::errors::ParquetError) -> crate::SemcastError {
    DataFusionError::External(Box::new(error)).into()
}

/// Command tag for statements without a result shape, from the leading
/// keywords: `CREATE TABLE`, `CREATE SEMANTIC INDEX`, `DROP TABLE`, ...
fn command_tag(sql: &str) -> String {
    let mut words = sql.split_whitespace().map(|word| word.to_ascii_uppercase());
    match (words.next().as_deref(), words.next().as_deref()) {
        (Some(first @ ("CREATE" | "DROP" | "ALTER")), Some("SEMANTIC")) => match words.next() {
            Some(third) => format!("{first} SEMANTIC {third}"),
            None => format!("{first} SEMANTIC"),
        },
        // psql knows `CREATE TABLE`; `CREATE EXTERNAL` would read as noise.
        (Some("CREATE"), Some("EXTERNAL")) => "CREATE TABLE".to_owned(),
        (Some(first @ ("CREATE" | "DROP" | "ALTER")), Some(second)) => format!("{first} {second}"),
        (Some(first), _) => first.to_owned(),
        (None, _) => "OK".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MockModel;

    const MATCHING: &str = "we agreed to ship offline sync in Q3";
    const OTHER: &str = "nothing notable happened";

    async fn engine() -> QueryEngine {
        let ctx = crate::semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));
        ctx.sql(&format!(
            "CREATE TABLE meetings AS
             SELECT * FROM (VALUES (1, '{MATCHING}'), (2, '{OTHER}')) AS t(meeting_id, transcript)",
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
        QueryEngine::new(Arc::new(ctx))
    }

    #[tokio::test]
    async fn means_query_returns_rows_and_reports_model_calls() {
        let engine = engine().await;
        let (tx, mut rx) = mpsc::channel(16);
        let outcome = engine
            .execute_statement(
                "SELECT meeting_id FROM meetings WHERE transcript MEANS 'offline sync'",
                tx,
            )
            .await
            .unwrap();

        let StatementOutcome::Rows { batches, .. } = outcome else {
            panic!("MEANS query yields rows");
        };
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event.line());
        }
        assert!(
            events.iter().any(|l| l.starts_with("funnel: VerifyExec")),
            "plan summary announced, got: {events:?}",
        );
        assert!(
            events
                .iter()
                .any(|l| l.starts_with("funnel done") && l.contains("2 model calls")),
            "final totals report the verify calls, got: {events:?}",
        );
    }

    #[tokio::test]
    async fn ddl_yields_a_command_tag_and_no_funnel_noise() {
        let engine = engine().await;
        let (tx, mut rx) = mpsc::channel(16);
        let outcome = engine
            .execute_statement("CREATE SEMANTIC INDEX ON meetings(transcript)", tx)
            .await
            .unwrap();

        let StatementOutcome::Command { tag } = outcome else {
            panic!("DDL yields a command tag");
        };
        assert_eq!(tag, "CREATE SEMANTIC INDEX");
        assert!(rx.try_recv().is_err(), "no progress events for DDL");
    }

    #[test]
    fn create_external_table_tags_as_create_table() {
        let tag = command_tag("CREATE EXTERNAL TABLE t STORED AS CSV LOCATION 'x.csv'");
        assert_eq!(tag, "CREATE TABLE");
    }

    #[tokio::test]
    async fn plain_sql_stays_silent() {
        let engine = engine().await;
        let (tx, mut rx) = mpsc::channel(16);
        let outcome = engine
            .execute_statement("SELECT meeting_id FROM meetings ORDER BY meeting_id", tx)
            .await
            .unwrap();

        let StatementOutcome::Rows { batches, .. } = outcome else {
            panic!("SELECT yields rows");
        };
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
        assert!(rx.try_recv().is_err(), "no funnel events for plain SQL");
    }
}
