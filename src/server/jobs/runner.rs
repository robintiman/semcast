//! Detached execution of a submitted statement.
//!
//! The task owns the job's lifecycle: it waits for a concurrency slot, drives
//! [`QueryEngine::execute_into`] with a Parquet target, mirrors progress into
//! the record, and writes a terminal status. Cancellation aborts the task
//! outright, so [`JobRegistry::cancel`] — not this module — writes the
//! `cancelled` status before it pulls the trigger. An abort cannot land on a
//! job that has not been polled yet, so the task also refuses to start a job
//! that is already terminal.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::Result;
use crate::server::engine::{QueryEngine, ResultTarget, StatementOutcome};
use crate::server::progress::ProgressEvent;

use super::{JobRecord, JobRegistry, JobStatus, PROGRESS_FILE, RESULT_FILE, now_ms};

/// Depth of the progress channel; matched to the interactive path.
const PROGRESS_BUFFER: usize = 16;

impl QueryEngine {
    /// Register `sql` as a job and start it. Returns as soon as the record
    /// exists — none of the work is awaited.
    pub fn submit(self: &Arc<Self>, sql: &str) -> Result<String> {
        let jobs = self.jobs().ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(
                "batch jobs are not enabled on this server".to_owned(),
            )
        })?;
        let id = jobs
            .create(sql)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let engine = Arc::clone(self);
        let registry = Arc::clone(jobs);
        let sql = sql.to_owned();
        let spawned = id.clone();
        // The spawn happens under the registry lock so the abort handle is in
        // place before the task can start or a cancel can look for it.
        jobs.spawn_attached(&id, || {
            tokio::spawn(run(engine, registry, spawned, sql)).abort_handle()
        });
        Ok(id)
    }
}

async fn run(engine: Arc<QueryEngine>, jobs: Arc<JobRegistry>, id: String, sql: String) {
    // Queued until a slot frees: running jobs share one model backend, so the
    // ceiling is deliberate rather than "however many were submitted".
    let _permit = match jobs.permits().acquire().await {
        Ok(permit) => permit,
        Err(_) => return, // registry shutting down
    };
    // A cancel that landed while this job queued has already written its
    // terminal status; running anyway would spend the model calls the cancel
    // was meant to stop.
    if !jobs.start(&id) {
        return;
    }

    let dir = jobs.dir(&id);
    let (events, progress) = mpsc::channel::<ProgressEvent>(PROGRESS_BUFFER);
    let drain = tokio::spawn(drain_progress(
        Arc::clone(&jobs),
        id.clone(),
        dir.join(PROGRESS_FILE),
        progress,
    ));

    let outcome = engine
        .execute_into(&sql, events, ResultTarget::Parquet(dir.join(RESULT_FILE)))
        .await;
    // The sender is dropped by now, so this closes rather than hangs.
    let _ = drain.await;

    match outcome {
        Ok(StatementOutcome::Written { path, rows }) => finish(&jobs, &id, |record| {
            record.status = JobStatus::Succeeded;
            record.rows = Some(rows as i64);
            record.result_path = Some(path.to_string_lossy().into_owned());
        }),
        Ok(StatementOutcome::Command { tag }) => finish(&jobs, &id, |record| {
            record.status = JobStatus::Succeeded;
            record.command_tag = Some(tag);
        }),
        // Jobs always target Parquet; a buffered result would mean the sink
        // selection above changed without this arm following.
        Ok(StatementOutcome::Rows { .. }) => finish(&jobs, &id, |record| {
            record.status = JobStatus::Failed;
            record.error = Some("internal: job produced buffered rows".to_owned());
        }),
        Err(error) => {
            let message = error.to_string();
            tracing::warn!("semcast: job {id} failed: {message}");
            finish(&jobs, &id, |record| {
                record.status = JobStatus::Failed;
                record.error = Some(message);
                // A failed statement leaves an unclosed Parquet file.
                record.result_path = None;
            });
        }
    }
}

/// Write the job's terminal status — unless something already did.
///
/// `cancel` marks the record before it aborts the task, but the abort can land
/// after the work finished and before this update runs. Cancellation is the
/// user's explicit decision, so it wins the race.
fn finish(jobs: &JobRegistry, id: &str, mutate: impl FnOnce(&mut JobRecord)) {
    jobs.update(id, |record| {
        if record.status.is_terminal() {
            return;
        }
        record.finished_at_ms = Some(now_ms());
        mutate(record);
    });
}

/// Mirror progress events into the record and append them to `progress.log`.
/// Events arrive at most every 500ms (the engine's tick), so this is a couple
/// of small writes per second for a running job.
async fn drain_progress(
    jobs: Arc<JobRegistry>,
    id: String,
    log_path: std::path::PathBuf,
    mut progress: mpsc::Receiver<ProgressEvent>,
) {
    use tokio::io::AsyncWriteExt;

    let mut log = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .await
        .ok();

    while let Some(event) = progress.recv().await {
        let line = event.line();
        if let Some(log) = log.as_mut() {
            let _ = log.write_all(format!("{line}\n").as_bytes()).await;
        }
        jobs.update(&id, |record| {
            record.progress = Some(line.clone());
            match &event {
                ProgressEvent::Funnel(counts) | ProgressEvent::Done(counts) => {
                    record.funnel = counts.clone();
                }
                ProgressEvent::Plan(_) => {}
            }
        });
    }
    if let Some(mut log) = log {
        let _ = log.flush().await;
    }
}
