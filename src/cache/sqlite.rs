//! A [`SemanticCache`] that survives the process: one SQLite file.
//!
//! The in-memory cache makes a repeated query free within one run; this one
//! makes it free across restarts, which is what lets the cache compound and
//! act as a checkpoint for resumed jobs. Keys are [`stable_digest`]s of the
//! full [`CacheKey`], so an entry written by one build is found by the next.
//!
//! Operators call the cache from async tasks, row by row, so writes never
//! touch SQLite on the caller's thread: a `put` lands in an in-memory pending
//! map, visible to `get` at once, and a dedicated writer thread commits the
//! backlog in batched transactions. Reads go to SQLite directly — a point
//! lookup, and in WAL mode a reader never waits for a writer. Dropping the
//! cache flushes the backlog; a crash loses at most the uncommitted tail,
//! which costs re-asking the model, never a wrong answer.
//!
//! Rows fail, queries don't: a read or write the database refuses is logged
//! and treated as a miss, never as a query error. The cost of a broken store
//! is model calls, not answers.

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rusqlite::{Connection, OptionalExtension, params};

use super::{CacheKey, CachedValue, SemanticCache, stable_digest};
use crate::{Result, SemcastError};

/// Bumped when the table layout changes; a file from another layout is
/// refused rather than misread.
const SCHEMA_VERSION: i64 = 1;

/// Most rows the writer commits in one transaction.
const MAX_BATCH: usize = 1024;

/// How long a connection waits on another's lock before giving up.
const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

type RowKey = [u8; 16];

/// One verdict on its way to disk.
#[derive(Debug)]
struct Row {
    key: RowKey,
    type_version: String,
    field: String,
    model_id: String,
    prompt_version: String,
    is_error: bool,
    value: String,
}

#[derive(Debug)]
pub struct SqliteCache {
    reader: Mutex<Connection>,
    /// Verdicts handed to the writer and not yet committed.
    pending: Arc<Mutex<HashMap<RowKey, CachedValue>>>,
    /// `None` only while dropping, so the writer sees its channel close.
    rows: Option<Sender<Row>>,
    writer: Option<JoinHandle<()>>,
}

impl SqliteCache {
    /// Open (or create) the store at `path`. Safe to call from several
    /// processes or contexts on the same file at once.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut writer = Connection::open(path).map_err(store_err)?;
        writer.busy_timeout(BUSY_TIMEOUT).map_err(store_err)?;
        // WAL lets readers proceed while a write commits; NORMAL sync is
        // durable across process crashes, which is the failure that matters
        // for a cache (a lost tail costs re-asking the model, nothing more).
        // Switching modes needs the file to itself and can't run in a
        // transaction, and SQLite reports a contended switch as busy at once
        // rather than through the busy handler, so retry it.
        retry_while_busy(|| writer.pragma_update(None, "journal_mode", "WAL"))?;
        writer
            .pragma_update(None, "synchronous", "NORMAL")
            .map_err(store_err)?;
        // Schema setup takes the write lock up front (IMMEDIATE): two openers
        // that both read first and then upgrade would deadlock, which SQLite
        // also reports as busy at once. Claim the version if the file is new,
        // then read whichever version won.
        let version: i64 = {
            let setup = writer
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .map_err(store_err)?;
            setup
                .execute_batch(
                    "CREATE TABLE IF NOT EXISTS meta (
                         key   TEXT PRIMARY KEY,
                         value INTEGER NOT NULL
                     );
                     CREATE TABLE IF NOT EXISTS verdicts (
                         key            BLOB PRIMARY KEY,
                         type_version   TEXT NOT NULL,
                         field          TEXT NOT NULL,
                         model_id       TEXT NOT NULL,
                         prompt_version TEXT NOT NULL,
                         is_error       INTEGER NOT NULL,
                         value          TEXT NOT NULL,
                         created_at     INTEGER NOT NULL DEFAULT (unixepoch())
                     ) WITHOUT ROWID;",
                )
                .map_err(store_err)?;
            setup
                .execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?1)",
                    params![SCHEMA_VERSION],
                )
                .map_err(store_err)?;
            let version = setup
                .query_row(
                    "SELECT value FROM meta WHERE key = 'schema_version'",
                    [],
                    |row| row.get(0),
                )
                .map_err(store_err)?;
            setup.commit().map_err(store_err)?;
            version
        };
        if version != SCHEMA_VERSION {
            return Err(SemcastError::Store(format!(
                "store schema version {version} is not the supported {SCHEMA_VERSION}"
            )));
        }

        let reader = Connection::open(path).map_err(store_err)?;
        reader.busy_timeout(BUSY_TIMEOUT).map_err(store_err)?;
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let (rows, received) = mpsc::channel();
        let writer = {
            let pending = Arc::clone(&pending);
            std::thread::Builder::new()
                .name("semcast-store-writer".to_owned())
                .spawn(move || write_loop(writer, received, pending))
                .map_err(|error| SemcastError::Store(error.to_string()))?
        };
        Ok(Self {
            reader: Mutex::new(reader),
            pending,
            rows: Some(rows),
            writer: Some(writer),
        })
    }

    /// Entries committed to disk, for inspection and tests.
    pub fn len(&self) -> Result<usize> {
        let reader = self.reader.lock().expect("store lock poisoned");
        let count: i64 = reader
            .query_row("SELECT count(*) FROM verdicts", [], |row| row.get(0))
            .map_err(store_err)?;
        Ok(count as usize)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

impl Drop for SqliteCache {
    /// Close the channel and wait for the writer to commit the backlog.
    fn drop(&mut self) {
        self.rows.take();
        if let Some(writer) = self.writer.take()
            && writer.join().is_err()
        {
            tracing::warn!(target: "semcast::store", "store writer panicked; its backlog is lost");
        }
    }
}

/// Commit rows as they arrive, as many per transaction as are waiting (up to
/// [`MAX_BATCH`]), until the cache is dropped.
fn write_loop(
    mut connection: Connection,
    rows: Receiver<Row>,
    pending: Arc<Mutex<HashMap<RowKey, CachedValue>>>,
) {
    while let Ok(first) = rows.recv() {
        let mut batch = vec![first];
        while batch.len() < MAX_BATCH {
            match rows.try_recv() {
                Ok(row) => batch.push(row),
                Err(_) => break,
            }
        }
        if let Err(error) = write_batch(&mut connection, &batch) {
            tracing::warn!(
                target: "semcast::store",
                %error,
                rows = batch.len(),
                "store write failed; these verdicts stay uncached",
            );
        }
        // Committed (or given up on): reads go to disk from here.
        let mut pending = pending.lock().expect("store lock poisoned");
        for row in &batch {
            pending.remove(&row.key);
        }
    }
}

fn write_batch(connection: &mut Connection, batch: &[Row]) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    {
        // First evaluation wins, as in memory: a re-run never overwrites the
        // verdict an earlier query already returned.
        let mut insert = transaction.prepare_cached(
            "INSERT OR IGNORE INTO verdicts
                 (key, type_version, field, model_id, prompt_version, is_error, value)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for row in batch {
            insert.execute(params![
                row.key,
                row.type_version,
                row.field,
                row.model_id,
                row.prompt_version,
                row.is_error,
                row.value,
            ])?;
        }
    }
    transaction.commit()
}

/// The row key: a digest of every field of the [`CacheKey`], so two keys
/// differing anywhere never share a row.
fn row_key(key: &CacheKey) -> RowKey {
    stable_digest(&[
        key.type_version.as_bytes(),
        key.field.as_bytes(),
        &key.input_digest.to_le_bytes(),
        key.model_id.0.as_bytes(),
        key.prompt_version.as_bytes(),
    ])
    .to_le_bytes()
}

impl SemanticCache for SqliteCache {
    fn get(&self, key: &CacheKey) -> Option<CachedValue> {
        let row_key = row_key(key);
        if let Some(value) = self
            .pending
            .lock()
            .expect("store lock poisoned")
            .get(&row_key)
        {
            return Some(value.clone());
        }
        let reader = self.reader.lock().expect("store lock poisoned");
        let row = reader
            .query_row(
                "SELECT is_error, value FROM verdicts WHERE key = ?1",
                params![row_key],
                |row| Ok((row.get::<_, bool>(0)?, row.get::<_, String>(1)?)),
            )
            .optional();
        match row {
            Ok(Some((false, value))) => Some(CachedValue::Value(value)),
            Ok(Some((true, error))) => Some(CachedValue::Error(error)),
            Ok(None) => None,
            Err(error) => {
                tracing::warn!(target: "semcast::store", %error, "store read failed; treating as a miss");
                None
            }
        }
    }

    fn put(&self, key: CacheKey, value: CachedValue) {
        let row_key = row_key(&key);
        {
            let mut pending = self.pending.lock().expect("store lock poisoned");
            if pending.contains_key(&row_key) {
                return; // first evaluation wins
            }
            pending.insert(row_key, value.clone());
        }
        let (is_error, value) = match value {
            CachedValue::Value(value) => (false, value),
            CachedValue::Error(error) => (true, error),
        };
        let row = Row {
            key: row_key,
            type_version: key.type_version,
            field: key.field,
            model_id: key.model_id.0,
            prompt_version: key.prompt_version,
            is_error,
            value,
        };
        if let Some(rows) = &self.rows
            && rows.send(row).is_err()
        {
            tracing::warn!(target: "semcast::store", "store writer is gone; the verdict stays uncached");
        }
    }
}

/// Run `attempt` until it stops reporting a busy database, for up to
/// [`BUSY_TIMEOUT`].
fn retry_while_busy(mut attempt: impl FnMut() -> rusqlite::Result<()>) -> Result<()> {
    let deadline = std::time::Instant::now() + BUSY_TIMEOUT;
    loop {
        match attempt() {
            Err(rusqlite::Error::SqliteFailure(error, _))
                if error.code == rusqlite::ErrorCode::DatabaseBusy
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            result => return result.map_err(store_err),
        }
    }
}

fn store_err(error: rusqlite::Error) -> SemcastError {
    SemcastError::Store(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;

    fn key(input: &str) -> CacheKey {
        CacheKey {
            type_version: "the customer is angry".to_owned(),
            field: "means".to_owned(),
            input_digest: stable_digest(&[input.as_bytes()]),
            model_id: ModelId("mock".to_owned()),
            prompt_version: "means-v2".to_owned(),
        }
    }

    #[test]
    fn verdicts_survive_reopening() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        {
            let cache = SqliteCache::open(&path).unwrap();
            cache.put(key("a"), CachedValue::Value("yes".to_owned()));
            cache.put(key("b"), CachedValue::Error("refused".to_owned()));
        }
        let cache = SqliteCache::open(&path).unwrap();
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("yes".to_owned()))
        );
        assert_eq!(
            cache.get(&key("b")),
            Some(CachedValue::Error("refused".to_owned()))
        );
        assert_eq!(cache.get(&key("c")), None);
        assert_eq!(cache.len().unwrap(), 2);
    }

    #[test]
    fn first_evaluation_wins() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SqliteCache::open(dir.path().join("store.db")).unwrap();
        cache.put(key("a"), CachedValue::Value("yes".to_owned()));
        cache.put(key("a"), CachedValue::Value("no".to_owned()));
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("yes".to_owned()))
        );
    }

    #[test]
    fn every_key_field_separates_rows() {
        let base = key("a");
        let variants = [
            CacheKey {
                type_version: "other".to_owned(),
                ..base.clone()
            },
            CacheKey {
                field: "other".to_owned(),
                ..base.clone()
            },
            CacheKey {
                input_digest: base.input_digest + 1,
                ..base.clone()
            },
            CacheKey {
                model_id: ModelId("other".to_owned()),
                ..base.clone()
            },
            CacheKey {
                prompt_version: "other".to_owned(),
                ..base.clone()
            },
        ];
        for variant in variants {
            assert_ne!(row_key(&base), row_key(&variant), "{variant:?}");
        }
    }

    #[test]
    fn a_put_is_visible_before_it_is_committed() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SqliteCache::open(dir.path().join("store.db")).unwrap();
        cache.put(key("a"), CachedValue::Value("yes".to_owned()));
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("yes".to_owned()))
        );
    }

    #[test]
    fn many_puts_are_all_committed_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        {
            let cache = SqliteCache::open(&path).unwrap();
            for i in 0..3000 {
                cache.put(key(&i.to_string()), CachedValue::Value("no".to_owned()));
            }
        }
        assert_eq!(SqliteCache::open(&path).unwrap().len().unwrap(), 3000);
    }

    /// Two servers (or two contexts) starting against a fresh file at once
    /// must both come up: the schema row is claimed, not blindly inserted.
    #[test]
    fn concurrent_openers_of_a_fresh_store_all_succeed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        let openers: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || SqliteCache::open(&path).map(drop))
            })
            .collect();
        for opener in openers {
            opener.join().unwrap().unwrap();
        }
    }

    #[test]
    fn a_store_from_another_schema_version_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        drop(SqliteCache::open(&path).unwrap());
        Connection::open(&path)
            .unwrap()
            .execute(
                "UPDATE meta SET value = 99 WHERE key = 'schema_version'",
                [],
            )
            .unwrap();
        let error = SqliteCache::open(&path).unwrap_err();
        assert!(error.to_string().contains("schema version 99"), "{error}");
    }
}
