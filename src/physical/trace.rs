//! Stage tracing: wrap an operator's output stream so it logs `begin` on the
//! first poll and `end` (with row count and elapsed time) when the stream is
//! exhausted. DataFusion streams are lazy — `execute()` only sets a stream up —
//! so polling is where the real work happens, and where these boundaries belong.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};
use futures::Stream;

/// Wrap `stream` so the stage's begin/end are logged at info on the
/// `semcast::stage` target. `stage` is the operator name (e.g. `"VerifyExec"`).
pub fn trace_stage(
    stage: &'static str,
    partition: usize,
    stream: SendableRecordBatchStream,
) -> SendableRecordBatchStream {
    let schema = stream.schema();
    Box::pin(TracedStage {
        stage,
        partition,
        schema,
        inner: stream,
        started: None,
        rows: 0,
    })
}

struct TracedStage {
    stage: &'static str,
    partition: usize,
    schema: SchemaRef,
    inner: SendableRecordBatchStream,
    /// Set on the first poll; doubles as the "already announced begin" flag.
    started: Option<Instant>,
    rows: usize,
}

impl Stream for TracedStage {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.started.is_none() {
            this.started = Some(Instant::now());
            tracing::info!(
                target: "semcast::stage",
                stage = this.stage,
                partition = this.partition,
                "begin"
            );
        }
        let poll = this.inner.as_mut().poll_next(cx);
        match &poll {
            Poll::Ready(Some(Ok(batch))) => this.rows += batch.num_rows(),
            Poll::Ready(None) => {
                let elapsed_ms = this.started.map_or(0, |t| t.elapsed().as_millis());
                tracing::info!(
                    target: "semcast::stage",
                    stage = this.stage,
                    partition = this.partition,
                    rows = this.rows,
                    elapsed_ms,
                    "end"
                );
            }
            _ => {}
        }
        poll
    }
}

impl RecordBatchStream for TracedStage {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
