//! Batch jobs: a statement submitted with `SUBMIT` runs detached from the
//! connection that asked for it, and its state is polled back with ordinary
//! SQL against `semcast_jobs` and `job_result('<id>')`.
//!
//! Semantic queries run for minutes to hours — one model call per surviving
//! row, or a whole-corpus embed for `CREATE SEMANTIC INDEX`. Holding a TCP
//! connection open for that long means a dropped link throws away every model
//! call the run has made so far.
//!
//! This module is state only: it never holds a `SessionContext`. That is what
//! keeps the dependencies acyclic — the context needs the registry (to expose
//! `semcast_jobs`), and the runner needs the context, so the runner lives on
//! [`crate::server::QueryEngine`] instead.

pub mod parse;
pub mod runner;
pub mod table;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::AbortHandle;

use super::progress::FunnelCounts;

pub use table::register;

/// Per-job state file, rewritten on every transition.
const RECORD_FILE: &str = "job.json";
/// Rendered funnel lines, appended as they arrive.
pub const PROGRESS_FILE: &str = "progress.log";
/// Where a job's rows land.
pub const RESULT_FILE: &str = "result.parquet";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Registered, waiting for a concurrency slot.
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    /// The server restarted while this job was in flight.
    Interrupted,
}

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Succeeded => "succeeded",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
            JobStatus::Interrupted => "interrupted",
        }
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self, JobStatus::Queued | JobStatus::Running)
    }
}

/// Everything `semcast_jobs` exposes for one job, and everything `job.json`
/// persists. Timestamps are unix millis — there is no date library in the
/// tree, and the table renders them as Arrow timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub sql: String,
    pub status: JobStatus,
    pub submitted_at_ms: i64,
    pub started_at_ms: Option<i64>,
    pub finished_at_ms: Option<i64>,
    pub rows: Option<i64>,
    /// Set instead of `rows` for statements with no result shape.
    pub command_tag: Option<String>,
    pub error: Option<String>,
    /// The most recent rendered funnel line.
    pub progress: Option<String>,
    #[serde(default)]
    pub funnel: FunnelCounts,
    pub result_path: Option<String>,
}

impl JobRecord {
    fn new(id: String, sql: String, submitted_at_ms: i64) -> Self {
        Self {
            id,
            sql,
            status: JobStatus::Queued,
            submitted_at_ms,
            started_at_ms: None,
            finished_at_ms: None,
            rows: None,
            command_tag: None,
            error: None,
            progress: None,
            funnel: FunnelCounts::default(),
            result_path: None,
        }
    }
}

struct JobEntry {
    record: JobRecord,
    /// Present only while the job's task is alive.
    abort: Option<AbortHandle>,
}

pub struct JobRegistry {
    root: PathBuf,
    permits: Semaphore,
    jobs: Mutex<HashMap<String, JobEntry>>,
    counter: AtomicU64,
}

/// Catalog entries must be `Debug`; dumping every record would be noise.
impl std::fmt::Debug for JobRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobRegistry")
            .field("root", &self.root)
            .field("jobs", &self.jobs.lock().map(|j| j.len()).unwrap_or(0))
            .finish_non_exhaustive()
    }
}

impl JobRegistry {
    pub fn new(root: impl Into<PathBuf>, max_concurrent: usize) -> std::io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            permits: Semaphore::new(max_concurrent.max(1)),
            jobs: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(0),
        })
    }

    pub(crate) fn permits(&self) -> &Semaphore {
        &self.permits
    }

    /// Load jobs left on disk by an earlier process. Anything still queued or
    /// running was killed by the restart: mark it `interrupted` rather than
    /// re-running it. A job can be a CTAS or `CREATE SEMANTIC INDEX` that
    /// already partly applied, so resuming is not idempotent — and silently
    /// starting hours of work at boot is the operator's call to make.
    pub fn recover(&self) -> std::io::Result<usize> {
        let mut recovered = 0;
        let mut interrupted = 0;
        for entry in std::fs::read_dir(&self.root)? {
            let dir = entry?.path();
            if !dir.is_dir() {
                continue;
            }
            let path = dir.join(RECORD_FILE);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let mut record: JobRecord = match serde_json::from_str(&text) {
                Ok(record) => record,
                Err(error) => {
                    tracing::warn!("semcast: ignoring unreadable {}: {error}", path.display());
                    continue;
                }
            };
            if !record.status.is_terminal() {
                record.status = JobStatus::Interrupted;
                record.finished_at_ms = Some(now_ms());
                record.error = Some("the server restarted while this job was running".to_owned());
                // A job killed mid-write left an unclosed Parquet file.
                record.result_path = None;
                let _ = write_record(&dir, &record);
                interrupted += 1;
            }
            recovered += 1;
            self.jobs.lock().unwrap().insert(
                record.id.clone(),
                JobEntry {
                    record,
                    abort: None,
                },
            );
        }
        if interrupted > 0 {
            tracing::warn!("semcast: {interrupted} job(s) interrupted by a restart");
        }
        Ok(recovered)
    }

    pub fn dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    /// Register a job and create its directory. The caller starts the work
    /// through [`Self::spawn_attached`].
    pub fn create(&self, sql: &str) -> std::io::Result<String> {
        let submitted_at_ms = now_ms();
        let seq = self.counter.fetch_add(1, Ordering::Relaxed);
        let id = format!("job_{submitted_at_ms}_{seq:04}");
        let record = JobRecord::new(id.clone(), sql.to_owned(), submitted_at_ms);
        let dir = self.dir(&id);
        std::fs::create_dir_all(&dir)?;
        write_record(&dir, &record)?;
        self.jobs.lock().unwrap().insert(
            id.clone(),
            JobEntry {
                record,
                abort: None,
            },
        );
        Ok(id)
    }

    /// Start the job's task and store its [`AbortHandle`], both under the
    /// registry lock.
    ///
    /// The lock is what makes cancellation reliable: the task's first act is
    /// [`Self::start`], which needs the same lock, so it cannot begin work —
    /// and a concurrent [`Self::cancel`] cannot observe a handle-less record —
    /// until the handle is stored. Spawning outside the lock leaves a window
    /// where a cancel reports success and the job runs on, spending the model
    /// calls the cancel was meant to stop.
    pub fn spawn_attached(&self, id: &str, spawn: impl FnOnce() -> AbortHandle) {
        let mut jobs = self.jobs.lock().unwrap();
        let abort = spawn();
        if let Some(entry) = jobs.get_mut(id) {
            entry.abort = Some(abort);
        }
    }

    /// Move a job to `running`, or refuse if it is already terminal — it was
    /// cancelled while it waited for a slot. Returns whether to do the work.
    pub fn start(&self, id: &str) -> bool {
        let mut jobs = self.jobs.lock().unwrap();
        let Some(entry) = jobs.get_mut(id) else {
            return false;
        };
        if entry.record.status.is_terminal() {
            return false;
        }
        entry.record.status = JobStatus::Running;
        entry.record.started_at_ms = Some(now_ms());
        persist(&self.root.join(id), &entry.record);
        true
    }

    /// Mutate a record and flush it to disk.
    ///
    /// The flush happens under the same lock as the mutation. Cloning the
    /// record and writing it afterwards would let two updaters for one job —
    /// a [`Self::cancel`] racing the progress drain — interleave inside the
    /// write, or land out of order and persist a status that has already been
    /// superseded.
    pub fn update(&self, id: &str, mutate: impl FnOnce(&mut JobRecord)) {
        let mut jobs = self.jobs.lock().unwrap();
        let Some(entry) = jobs.get_mut(id) else {
            return;
        };
        mutate(&mut entry.record);
        persist(&self.root.join(id), &entry.record);
    }

    pub fn get(&self, id: &str) -> Option<JobRecord> {
        self.jobs
            .lock()
            .unwrap()
            .get(id)
            .map(|entry| entry.record.clone())
    }

    /// Every record, newest first.
    pub fn snapshot(&self) -> Vec<JobRecord> {
        let mut records: Vec<JobRecord> = self
            .jobs
            .lock()
            .unwrap()
            .values()
            .map(|entry| entry.record.clone())
            .collect();
        records.sort_unstable_by(|a, b| {
            b.submitted_at_ms
                .cmp(&a.submitted_at_ms)
                .then_with(|| b.id.cmp(&a.id))
        });
        records
    }

    /// Abort a queued or running job. The record is marked `cancelled` and
    /// flushed *before* the abort, because the aborted task never gets to run
    /// its own terminal update.
    pub fn cancel(&self, id: &str) -> Result<(), String> {
        let abort = {
            let mut jobs = self.jobs.lock().unwrap();
            let entry = jobs
                .get_mut(id)
                .ok_or_else(|| format!("no such job: {id}"))?;
            if entry.record.status.is_terminal() {
                return Err(format!(
                    "job {id} already finished ({})",
                    entry.record.status.as_str()
                ));
            }
            entry.record.status = JobStatus::Cancelled;
            entry.record.finished_at_ms = Some(now_ms());
            // Whatever Parquet was written is missing its footer.
            entry.record.result_path = None;
            persist(&self.root.join(id), &entry.record);
            entry.abort.take()
        };
        if let Some(abort) = abort {
            abort.abort();
        }
        Ok(())
    }
}

/// Flush a record whose lock the caller holds. Failures are logged, not
/// propagated — losing a status file must not kill a running job.
fn persist(dir: &Path, record: &JobRecord) {
    if let Err(error) = write_record(dir, record) {
        tracing::warn!("semcast: could not persist job {}: {error}", record.id);
    }
}

fn write_record(dir: &Path, record: &JobRecord) -> std::io::Result<()> {
    // Write-then-rename: a crash mid-write must not leave a half-parsed
    // job.json that recovery then skips. The temp name carries the pid so
    // that two processes sharing a jobs dir cannot scribble over each
    // other's half-written file.
    let tmp = dir.join(format!("job.json.{}.tmp", std::process::id()));
    let json = serde_json::to_vec_pretty(record)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(tmp, dir.join(RECORD_FILE))
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_writes_a_queued_record() {
        let dir = tempfile::tempdir().unwrap();
        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        let id = registry.create("SELECT 1").unwrap();

        let record = registry.get(&id).unwrap();
        assert_eq!(record.status, JobStatus::Queued);
        assert_eq!(record.sql, "SELECT 1");

        let on_disk: JobRecord = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(&id).join(RECORD_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(on_disk.id, id);
        assert_eq!(on_disk.status, JobStatus::Queued);
    }

    #[test]
    fn recover_marks_in_flight_jobs_interrupted() {
        let dir = tempfile::tempdir().unwrap();
        let job_dir = dir.path().join("job_1_0000");
        std::fs::create_dir_all(&job_dir).unwrap();
        std::fs::write(
            job_dir.join(RECORD_FILE),
            r#"{
                "id": "job_1_0000",
                "sql": "SELECT 1",
                "status": "running",
                "submitted_at_ms": 1,
                "started_at_ms": 2,
                "finished_at_ms": null,
                "rows": null,
                "command_tag": null,
                "error": null,
                "progress": "verify: 3 model calls, 0 cache hits, 0 dropped",
                "result_path": "/tmp/x.parquet"
            }"#,
        )
        .unwrap();

        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        assert_eq!(registry.recover().unwrap(), 1);

        let record = registry.get("job_1_0000").unwrap();
        assert_eq!(record.status, JobStatus::Interrupted);
        assert!(record.error.unwrap().contains("restarted"));
        // Partial progress survives; the half-written result does not.
        assert!(record.progress.unwrap().contains("3 model calls"));
        assert_eq!(record.result_path, None);
    }

    #[test]
    fn recover_leaves_finished_jobs_alone() {
        let dir = tempfile::tempdir().unwrap();
        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        let id = registry.create("SELECT 1").unwrap();
        registry.update(&id, |r| {
            r.status = JobStatus::Succeeded;
            r.rows = Some(7);
            r.result_path = Some("/tmp/result.parquet".to_owned());
        });

        let reopened = JobRegistry::new(dir.path(), 2).unwrap();
        reopened.recover().unwrap();
        let record = reopened.get(&id).unwrap();
        assert_eq!(record.status, JobStatus::Succeeded);
        assert_eq!(record.rows, Some(7));
        assert_eq!(record.result_path.as_deref(), Some("/tmp/result.parquet"));
    }

    #[test]
    fn cancel_rejects_unknown_and_finished_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        assert!(registry.cancel("nope").unwrap_err().contains("no such job"));

        let id = registry.create("SELECT 1").unwrap();
        registry.update(&id, |r| r.status = JobStatus::Succeeded);
        assert!(
            registry
                .cancel(&id)
                .unwrap_err()
                .contains("already finished"),
            "a finished job cannot be cancelled",
        );
    }

    #[test]
    fn a_cancelled_job_never_starts() {
        let dir = tempfile::tempdir().unwrap();
        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        let id = registry.create("SELECT 1").unwrap();

        // Cancelled before its task reached a slot: nothing must run, or the
        // job spends the model calls the cancel was meant to stop.
        registry.cancel(&id).unwrap();
        assert!(!registry.start(&id), "a cancelled job refuses to start");
        assert_eq!(registry.get(&id).unwrap().status, JobStatus::Cancelled);

        assert!(!registry.start("nope"), "so does a job that does not exist");
    }

    #[test]
    fn start_moves_a_queued_job_to_running() {
        let dir = tempfile::tempdir().unwrap();
        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        let id = registry.create("SELECT 1").unwrap();

        assert!(registry.start(&id));
        let record = registry.get(&id).unwrap();
        assert_eq!(record.status, JobStatus::Running);
        assert!(record.started_at_ms.is_some());
    }

    #[test]
    fn concurrent_updates_persist_a_readable_record() {
        // The progress drain and a `cancel` update one job at the same
        // instant. Cloning the record and writing it after the lock is
        // released loses this race two ways: the two writes interleave into a
        // half-written job.json, or the drain's older clone lands last and
        // persists `running` over `cancelled`. Both are reproducible within a
        // few dozen attempts, so the margin here is deliberate.
        let dir = tempfile::tempdir().unwrap();
        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        let sql = "SELECT 'x'".repeat(2_000);

        for attempt in 0..300 {
            let id = registry.create(&sql).unwrap();
            registry.start(&id);
            let gate = std::sync::Barrier::new(2);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    gate.wait();
                    registry.update(&id, |r| r.progress = Some("last tick".to_owned()));
                });
                scope.spawn(|| {
                    gate.wait();
                    registry.cancel(&id).unwrap();
                });
            });

            let text = std::fs::read_to_string(dir.path().join(&id).join(RECORD_FILE)).unwrap();
            let on_disk: JobRecord = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("attempt {attempt}: job.json parses: {e}"));
            assert_eq!(
                on_disk.status,
                JobStatus::Cancelled,
                "attempt {attempt}: a stale write won",
            );
        }
    }

    #[test]
    fn snapshot_is_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        let registry = JobRegistry::new(dir.path(), 2).unwrap();
        let first = registry.create("SELECT 1").unwrap();
        let second = registry.create("SELECT 2").unwrap();

        let ids: Vec<String> = registry.snapshot().into_iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![second, first]);
    }
}
