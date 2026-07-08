//! Statement execution decoupled from the wire protocol: takes SQL text,
//! returns buffered results, streams progress lines over a channel. The
//! extended-protocol handler can reuse this unchanged later.

use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::execute_stream;
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::Result;

use super::progress;

/// How often live funnel counters are checked for a progress NOTICE.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(500);

pub struct QueryEngine {
    ctx: Arc<SessionContext>,
}

pub enum StatementOutcome {
    Rows {
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    },
    Command {
        tag: String,
    },
}

impl QueryEngine {
    pub fn new(ctx: Arc<SessionContext>) -> Self {
        Self { ctx }
    }

    /// Execute one already-split statement. Progress lines land on
    /// `events` while the query runs; send failures are ignored so a
    /// disinterested receiver never blocks execution.
    pub async fn execute_statement(
        &self,
        sql: &str,
        events: mpsc::Sender<String>,
    ) -> Result<StatementOutcome> {
        // DDL and CTAS execute eagerly in here; their DataFrame is empty.
        let df = crate::sql(&self.ctx, sql).await?;
        let plan = df.create_physical_plan().await?;

        for line in progress::funnel_summary(&plan) {
            let _ = events.send(line).await;
        }
        let semantic = progress::snapshot(&plan).is_semantic();

        let mut stream = execute_stream(Arc::clone(&plan), self.ctx.task_ctx())?;
        let mut tick = tokio::time::interval(PROGRESS_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        tick.tick().await; // the first tick is immediate — skip it
        let mut last = progress::FunnelCounts::default();
        let mut batches = Vec::new();
        loop {
            tokio::select! {
                batch = stream.next() => match batch {
                    Some(batch) => batches.push(batch?),
                    None => break,
                },
                _ = tick.tick(), if semantic => {
                    if let Some(line) = progress::snapshot_if_changed(&plan, &mut last) {
                        let _ = events.send(line).await;
                    }
                }
            }
        }
        if let Some(line) = progress::final_totals(&plan) {
            let _ = events.send(line).await;
        }

        let schema = plan.schema();
        if schema.fields().is_empty() {
            Ok(StatementOutcome::Command {
                tag: command_tag(sql),
            })
        } else {
            Ok(StatementOutcome::Rows { schema, batches })
        }
    }
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
        while let Ok(line) = rx.try_recv() {
            events.push(line);
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
