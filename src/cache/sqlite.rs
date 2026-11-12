//! A [`SemanticCache`] that survives the process: one SQLite file.
//!
//! The in-memory cache makes a repeated query free within one run; this one
//! makes it free across restarts, which is what lets the cache compound and
//! act as a checkpoint for resumed jobs. Keys are [`stable_digest`]s of the
//! full [`CacheKey`], so an entry written by one build is found by the next.
//!
//! Operators call the cache from async tasks, row by row, so writes never
//! touch SQLite on the caller's thread: a `put` lands in an in-memory map of
//! every verdict this instance has stored, visible to `get` at once, and a
//! dedicated writer thread persists the backlog in batched transactions. The
//! map is the instance's memory of what it has returned, so first evaluation
//! wins regardless of what the disk does: a write that fails leaves the
//! verdict served from memory, as the in-memory cache would. Reads of
//! verdicts from earlier runs go to SQLite — a point lookup, and in WAL mode
//! a reader never waits for a writer. Dropping the cache flushes the backlog;
//! a crash loses at most the uncommitted tail, which costs re-asking the
//! model after a restart, never a wrong answer.
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

use super::{AskStats, CacheKey, CachedValue, SemanticCache, stable_digest};
use crate::{Result, SemcastError};

/// Bumped when the table layout changes; a file from another layout is
/// refused rather than misread.
const SCHEMA_VERSION: i64 = 1;

/// Most rows the writer commits in one transaction.
const MAX_BATCH: usize = 1024;

/// How long a connection waits on another's lock before giving up.
const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

type RowKey = [u8; 16];

/// What the writer thread is asked to do.
enum Message {
    /// Rows to commit together: never split across transactions, so a group
    /// (an extraction unit's fields) is on disk whole or not at all.
    Rows(Vec<Row>),
    /// Reply once everything sent before this message is committed.
    Flush(Sender<()>),
    /// Add to a question's statistics.
    Stats(u128, AskStats),
}

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
    /// Every verdict this instance has stored, committed or not. Never
    /// shrinks, like the in-memory cache: it is what makes first evaluation
    /// win within the instance whatever happens on disk.
    stored: Mutex<HashMap<RowKey, CachedValue>>,
    /// `None` only while dropping, so the writer sees its channel close.
    messages: Option<Sender<Message>>,
    writer: Option<JoinHandle<()>>,
    /// Statistics recorded but not yet committed, added to what disk holds
    /// when read. The writer subtracts them once committed, so a read racing
    /// that moment may count a batch twice: statistics order filters, never
    /// decide rows, so that costs at most a call.
    pending_stats: Arc<Mutex<HashMap<u128, AskStats>>>,
}

impl SqliteCache {
    /// Open (or create) the store at `path`. Several processes or contexts
    /// may open the same file at once, and all of them read and write it.
    ///
    /// "First evaluation wins" holds within one `SqliteCache`. Two instances
    /// sharing a file that ask the model about the same input at the same
    /// moment can each return their own answer; the file keeps whichever
    /// commits first, and the other instance logs the disagreement. Making
    /// them agree would mean claiming each key on disk before asking the
    /// model — a blocking write on every miss.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let writer = open_store(path)?;

        let reader = Connection::open(path).map_err(store_err)?;
        reader.busy_timeout(BUSY_TIMEOUT).map_err(store_err)?;
        let (messages, received) = mpsc::channel();
        let pending_stats: Arc<Mutex<HashMap<u128, AskStats>>> = Arc::default();
        let pending = Arc::clone(&pending_stats);
        let writer = std::thread::Builder::new()
            .name("semcast-store-writer".to_owned())
            .spawn(move || write_loop(writer, received, pending))
            .map_err(|error| SemcastError::Store(error.to_string()))?;
        Ok(Self {
            reader: Mutex::new(reader),
            stored: Mutex::new(HashMap::new()),
            messages: Some(messages),
            writer: Some(writer),
            pending_stats,
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

    /// Block until every verdict stored so far is committed to disk. For a
    /// clean shutdown when the cache itself can't be dropped (a server whose
    /// connections still hold the session). Blocking: call it off the async
    /// threads.
    pub fn flush(&self) {
        let (done, wait) = mpsc::channel();
        if let Some(messages) = &self.messages
            && messages.send(Message::Flush(done)).is_ok()
        {
            // An error means the writer is gone; nothing more will commit.
            let _ = wait.recv();
        }
    }

    /// Store several verdicts that belong together, committed in one
    /// transaction. First evaluation wins per key, as for `put`.
    fn store(&self, entries: Vec<(CacheKey, CachedValue)>) {
        let mut rows = Vec::with_capacity(entries.len());
        {
            // A key this instance has stored keeps its verdict. No disk read
            // here — a put follows a get that missed, and the map, not the
            // disk, is what this instance has returned.
            let mut stored = self.stored.lock().expect("store lock poisoned");
            for (key, value) in entries {
                let row_key = row_key(&key);
                if stored.contains_key(&row_key) {
                    continue;
                }
                stored.insert(row_key, value.clone());
                let (is_error, value) = match value {
                    CachedValue::Value(value) => (false, value),
                    CachedValue::Error(error) => (true, error),
                };
                rows.push(Row {
                    key: row_key,
                    type_version: key.type_version,
                    field: key.field,
                    model_id: key.model_id.0,
                    prompt_version: key.prompt_version,
                    is_error,
                    value,
                });
            }
        }
        if rows.is_empty() {
            return;
        }
        if let Some(messages) = &self.messages
            && messages.send(Message::Rows(rows)).is_err()
        {
            tracing::warn!(target: "semcast::store", "store writer is gone; verdicts stay uncached");
        }
    }
}

impl Drop for SqliteCache {
    /// Close the channel and wait for the writer to commit the backlog.
    fn drop(&mut self) {
        self.messages.take();
        if let Some(writer) = self.writer.take()
            && writer.join().is_err()
        {
            tracing::warn!(target: "semcast::store", "store writer panicked; its backlog is lost");
        }
    }
}

/// Commit groups of rows as they arrive, as many per transaction as are
/// waiting (up to about [`MAX_BATCH`] rows; a group is never split), and
/// answer flush requests once what came before them is committed. Runs until
/// the cache is dropped.
fn write_loop(
    mut connection: Connection,
    messages: Receiver<Message>,
    pending_stats: Arc<Mutex<HashMap<u128, AskStats>>>,
) {
    while let Ok(first) = messages.recv() {
        let mut batch: Vec<Row> = Vec::new();
        let mut stats: Vec<(u128, AskStats)> = Vec::new();
        let mut flushes: Vec<Sender<()>> = Vec::new();
        let mut next = Some(first);
        while let Some(message) = next.take() {
            match message {
                Message::Rows(rows) => batch.extend(rows),
                Message::Stats(question, delta) => stats.push((question, delta)),
                Message::Flush(done) => {
                    // Commit what precedes the flush before answering it.
                    flushes.push(done);
                    break;
                }
            }
            if batch.len() >= MAX_BATCH {
                break;
            }
            next = messages.try_recv().ok();
        }
        if !batch.is_empty() {
            commit(&mut connection, &batch);
        }
        if !stats.is_empty() {
            if let Err(error) = write_stats(&mut connection, &stats) {
                tracing::warn!(target: "semcast::store", %error, "store could not save statistics");
            }
            // Committed or lost, they no longer count as pending.
            let mut pending = pending_stats.lock().expect("stats lock poisoned");
            for (question, delta) in &stats {
                if let Some(entry) = pending.get_mut(question) {
                    entry.asked = entry.asked.saturating_sub(delta.asked);
                    entry.matched = entry.matched.saturating_sub(delta.matched);
                    entry.input_chars = entry.input_chars.saturating_sub(delta.input_chars);
                }
            }
        }
        for done in flushes {
            let _ = done.send(());
        }
    }
}

fn commit(connection: &mut Connection, batch: &[Row]) {
    match write_batch(connection, batch) {
        Ok(conflicts) => {
            // Two runs of one model can differ in a probability's last
            // digits while agreeing on the verdict: not worth a warning.
            for row in conflicts
                .into_iter()
                .filter(|row| !row.field.ends_with(".yes_probability"))
            {
                tracing::warn!(
                    target: "semcast::store",
                    condition = %row.type_version,
                    field = %row.field,
                    model = %row.model_id,
                    "another process stored a different verdict for this input first; \
                     this process keeps returning its own until it restarts, then \
                     the stored one",
                );
            }
        }
        Err(error) => {
            tracing::warn!(
                target: "semcast::store",
                %error,
                rows = batch.len(),
                "store write failed; these verdicts are kept in memory for this \
                 process only",
            );
        }
    }
}

fn write_stats(connection: &mut Connection, stats: &[(u128, AskStats)]) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    {
        let mut upsert = transaction.prepare_cached(
            "INSERT INTO ask_stats (question, asked, matched, input_chars)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT (question) DO UPDATE SET
                 asked = asked + excluded.asked,
                 matched = matched + excluded.matched,
                 input_chars = input_chars + excluded.input_chars",
        )?;
        for (question, delta) in stats {
            upsert.execute(params![
                question.to_le_bytes().to_vec(),
                delta.asked as i64,
                delta.matched as i64,
                delta.input_chars as i64,
            ])?;
        }
    }
    transaction.commit()
}

/// Commit `batch`, returning the rows whose key already held a different
/// verdict — written by another instance sharing the file, since this one
/// puts each key at most once.
fn write_batch<'a>(
    connection: &mut Connection,
    batch: &'a [Row],
) -> rusqlite::Result<Vec<&'a Row>> {
    let mut conflicts = Vec::new();
    let transaction = connection.transaction()?;
    {
        // First evaluation wins, as in memory: a re-run never overwrites the
        // verdict an earlier query already returned.
        let mut insert = transaction.prepare_cached(
            "INSERT OR IGNORE INTO verdicts
                 (key, type_version, field, model_id, prompt_version, is_error, value)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        let mut stored =
            transaction.prepare_cached("SELECT is_error, value FROM verdicts WHERE key = ?1")?;
        for row in batch {
            let inserted = insert.execute(params![
                row.key,
                row.type_version,
                row.field,
                row.model_id,
                row.prompt_version,
                row.is_error,
                row.value,
            ])?;
            if inserted == 0 {
                let (is_error, value): (bool, String) =
                    stored.query_row(params![row.key], |r| Ok((r.get(0)?, r.get(1)?)))?;
                if is_error != row.is_error || value != row.value {
                    conflicts.push(row);
                }
            }
        }
    }
    transaction.commit()?;
    Ok(conflicts)
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

impl SqliteCache {
    /// The verdict committed to disk under `row_key`, if any.
    fn committed(&self, row_key: &RowKey) -> Option<CachedValue> {
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
}

/// Most keys one batched read binds, well under SQLite's variable limit.
const MAX_READ_BATCH: usize = 512;

impl SqliteCache {
    /// The verdicts committed to disk under `row_keys`, read in batches of
    /// [`MAX_READ_BATCH`]. A failed read is logged and leaves its keys
    /// missing: a miss costs a model call, never an answer.
    fn committed_many(&self, row_keys: &[RowKey]) -> HashMap<RowKey, CachedValue> {
        let mut found = HashMap::new();
        let reader = self.reader.lock().expect("store lock poisoned");
        for chunk in row_keys.chunks(MAX_READ_BATCH) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let sql =
                format!("SELECT key, is_error, value FROM verdicts WHERE key IN ({placeholders})");
            let read = reader.prepare_cached(&sql).and_then(|mut statement| {
                let rows =
                    statement.query_map(rusqlite::params_from_iter(chunk.iter()), |row| {
                        Ok((
                            row.get::<_, Vec<u8>>(0)?,
                            row.get::<_, bool>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            });
            match read {
                Ok(rows) => {
                    for (key, is_error, value) in rows {
                        let Ok(key) = RowKey::try_from(key.as_slice()) else {
                            continue;
                        };
                        let value = if is_error {
                            CachedValue::Error(value)
                        } else {
                            CachedValue::Value(value)
                        };
                        found.insert(key, value);
                    }
                }
                Err(error) => {
                    tracing::warn!(target: "semcast::store", %error, "store read failed; treating as a miss");
                }
            }
        }
        found
    }
}

impl SemanticCache for SqliteCache {
    fn get_many(&self, keys: &[CacheKey]) -> Vec<Option<CachedValue>> {
        let row_keys: Vec<RowKey> = keys.iter().map(row_key).collect();
        // What this instance already holds answers first, as in `get`.
        let mut answers: Vec<Option<CachedValue>> = {
            let stored = self.stored.lock().expect("store lock poisoned");
            row_keys
                .iter()
                .map(|key| stored.get(key).cloned())
                .collect()
        };
        let missing: Vec<RowKey> = row_keys
            .iter()
            .zip(&answers)
            .filter(|(_, answer)| answer.is_none())
            .map(|(key, _)| *key)
            .collect();
        if missing.is_empty() {
            return answers;
        }
        // Not remembered in memory, unlike `get`: a lookup reads every row of
        // a table, and pinning each answer would grow with the table rather
        // than with what this instance has asked. Puts follow misses, so a
        // key read here is put again only by a query that missed it before
        // this read, a race `get` closes and this does not; the store keeps
        // its first value on disk regardless.
        let mut committed = self.committed_many(&missing);
        for (key, answer) in row_keys.iter().zip(answers.iter_mut()) {
            if answer.is_none() {
                *answer = committed.remove(key);
            }
        }
        answers
    }

    fn get(&self, key: &CacheKey) -> Option<CachedValue> {
        let row_key = row_key(key);
        if let Some(value) = self
            .stored
            .lock()
            .expect("store lock poisoned")
            .get(&row_key)
        {
            return Some(value.clone());
        }
        let committed = self.committed(&row_key)?;
        // Remember what was returned, so a put racing in later for the same
        // key backs off and this instance never answers two ways. If a put
        // landed between the two lookups, the map's verdict is the one this
        // instance has already returned, so that is the one to keep.
        Some(
            self.stored
                .lock()
                .expect("store lock poisoned")
                .entry(row_key)
                .or_insert(committed)
                .clone(),
        )
    }

    fn put(&self, key: CacheKey, value: CachedValue) {
        self.store(vec![(key, value)]);
    }

    fn put_many(&self, entries: Vec<(CacheKey, CachedValue)>) {
        self.store(entries);
    }

    fn record_stats(&self, question: u128, delta: AskStats) {
        self.pending_stats
            .lock()
            .expect("stats lock poisoned")
            .entry(question)
            .or_default()
            .add(delta);
        if let Some(messages) = &self.messages {
            let _ = messages.send(Message::Stats(question, delta));
        }
    }

    fn stats(&self, question: u128) -> Option<AskStats> {
        let stored: Option<AskStats> = {
            let reader = self.reader.lock().expect("store lock poisoned");
            reader
                .query_row(
                    "SELECT asked, matched, input_chars FROM ask_stats WHERE question = ?1",
                    params![question.to_le_bytes().to_vec()],
                    |row| {
                        Ok(AskStats {
                            asked: row.get::<_, i64>(0)? as u64,
                            matched: row.get::<_, i64>(1)? as u64,
                            input_chars: row.get::<_, i64>(2)? as u64,
                        })
                    },
                )
                .ok()
        };
        let pending = self
            .pending_stats
            .lock()
            .expect("stats lock poisoned")
            .get(&question)
            .copied();
        match (stored, pending) {
            (None, None) => None,
            (stored, pending) => {
                let mut total = stored.unwrap_or_default();
                total.add(pending.unwrap_or_default());
                Some(total)
            }
        }
    }
}

/// Open a connection to the store file at `path`, creating its tables on
/// first use and refusing a file from another schema version. Both the
/// verdict cache and the catalog open the file through here, so either may
/// be the one that creates it.
pub(crate) fn open_store(path: &Path) -> Result<Connection> {
    let mut connection = Connection::open(path).map_err(store_err)?;
    connection.busy_timeout(BUSY_TIMEOUT).map_err(store_err)?;
    // WAL lets readers proceed while a write commits; NORMAL sync is
    // durable across process crashes, which is the failure that matters
    // for a cache (a lost tail costs re-asking the model, nothing more).
    // Switching modes needs the file to itself and can't run in a
    // transaction, and SQLite reports a contended switch as busy at once
    // rather than through the busy handler, so retry it.
    retry_while_busy(|| connection.pragma_update(None, "journal_mode", "WAL"))?;
    connection
        .pragma_update(None, "synchronous", "NORMAL")
        .map_err(store_err)?;
    // Schema setup takes the write lock up front (IMMEDIATE): two openers
    // that both read first and then upgrade would deadlock, which SQLite
    // also reports as busy at once. Claim the version if the file is new,
    // then read whichever version won.
    let version: i64 = {
        let setup = connection
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
                 ) WITHOUT ROWID;
                 CREATE TABLE IF NOT EXISTS catalog (
                     seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                     kind       TEXT NOT NULL,
                     name       TEXT NOT NULL,
                     definition TEXT NOT NULL,
                     UNIQUE (kind, name)
                 );
                 -- EMBED answers, content-addressed like verdicts: the key is
                 -- a digest of (embed model, chunk text).
                 CREATE TABLE IF NOT EXISTS embeddings (
                     key         BLOB PRIMARY KEY,
                     embed_model TEXT NOT NULL,
                     vector      BLOB NOT NULL
                 ) WITHOUT ROWID;
                 -- Which chunks each semantic index holds, and the embedding
                 -- each one reads. Added in place: an older build ignores it.
                 -- What the model answered per question, for ordering
                 -- filters. Added in place: an older build ignores it.
                 CREATE TABLE IF NOT EXISTS ask_stats (
                     question    BLOB PRIMARY KEY,
                     asked       INTEGER NOT NULL,
                     matched     INTEGER NOT NULL,
                     input_chars INTEGER NOT NULL
                 ) WITHOUT ROWID;
                 CREATE TABLE IF NOT EXISTS index_chunks (
                     index_name  TEXT NOT NULL,
                     doc_hash    INTEGER NOT NULL,
                     chunk_index INTEGER NOT NULL,
                     text        TEXT NOT NULL,
                     embedding   BLOB NOT NULL,
                     dim         INTEGER NOT NULL,
                     PRIMARY KEY (index_name, doc_hash, chunk_index)
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
    Ok(connection)
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

pub(crate) fn store_err(error: rusqlite::Error) -> SemcastError {
    SemcastError::Store(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;

    /// The store's row key must never change: rows written by one build are
    /// read by the next. Pinned on the key SemIR builds for a `MEANS` answer.
    #[test]
    fn row_key_matches_golden_value() {
        // Through `means_cache_key`, the builder every MEANS call site uses.
        let key = crate::physical::verify::means_cache_key(
            "the customer is angry",
            "I am furious",
            &ModelId("ollama/gemma4:e4b".to_owned()),
            "means-v2",
        );
        assert_eq!(u128::from_le_bytes(row_key(&key)), GOLDEN_ROW_KEY);
    }

    const GOLDEN_ROW_KEY: u128 = 137170530867284571163207860107501048400;

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

    /// After the writer commits a verdict, a later put for the same key must
    /// still lose to it.
    #[test]
    fn first_evaluation_wins_after_it_is_committed() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SqliteCache::open(dir.path().join("store.db")).unwrap();
        cache.put(key("a"), CachedValue::Value("yes".to_owned()));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while cache.len().unwrap() == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "writer never committed"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        cache.put(key("a"), CachedValue::Value("no".to_owned()));
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("yes".to_owned()))
        );
    }

    /// Two instances on one file: the first commit wins on disk, and the
    /// instance that lost reports the disagreement rather than hiding it.
    #[test]
    fn a_conflicting_commit_from_another_instance_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        let mut connection = Connection::open(&path).unwrap();
        drop(SqliteCache::open(&path).unwrap());
        let row = |value: &str| Row {
            key: row_key(&key("a")),
            type_version: "the customer is angry".to_owned(),
            field: "means".to_owned(),
            model_id: "mock".to_owned(),
            prompt_version: "means-v2".to_owned(),
            is_error: false,
            value: value.to_owned(),
        };
        let first = [row("yes")];
        assert!(write_batch(&mut connection, &first).unwrap().is_empty());
        let same = [row("yes")];
        assert!(
            write_batch(&mut connection, &same).unwrap().is_empty(),
            "agreeing is fine"
        );
        let other = [row("no")];
        assert_eq!(write_batch(&mut connection, &other).unwrap().len(), 1);
        let cache = SqliteCache::open(&path).unwrap();
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("yes".to_owned()))
        );
    }

    /// A verdict read from disk is what this instance returned: a put for the
    /// same key afterwards, from a query that missed earlier, must lose to it.
    #[test]
    fn a_verdict_read_from_disk_wins_over_a_later_put() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        {
            let other = SqliteCache::open(&path).unwrap();
            other.put(key("a"), CachedValue::Value("no".to_owned()));
        }
        let cache = SqliteCache::open(&path).unwrap();
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("no".to_owned()))
        );
        cache.put(key("a"), CachedValue::Value("yes".to_owned()));
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("no".to_owned()))
        );
    }

    /// A group is committed whole: after a flush, every member is on disk.
    #[test]
    fn put_many_commits_a_group_together_and_flush_waits_for_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        let cache = SqliteCache::open(&path).unwrap();
        cache.put_many(
            (0..10)
                .map(|i| (key(&i.to_string()), CachedValue::Value("yes".to_owned())))
                .collect(),
        );
        cache.flush();
        assert_eq!(cache.len().unwrap(), 10);
        // A second instance sees all of it without the first being dropped.
        let other = SqliteCache::open(&path).unwrap();
        assert_eq!(other.len().unwrap(), 10);
    }

    /// A write the database refuses must not make this instance forget a
    /// verdict it already returned.
    #[test]
    fn a_failed_write_keeps_the_verdict_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        let cache = SqliteCache::open(&path).unwrap();
        Connection::open(&path)
            .unwrap()
            .execute("DROP TABLE verdicts", [])
            .unwrap();
        cache.put(key("a"), CachedValue::Value("yes".to_owned()));
        // Give the writer time to try, and fail, to commit it.
        std::thread::sleep(std::time::Duration::from_millis(50));
        cache.put(key("a"), CachedValue::Value("no".to_owned()));
        assert_eq!(
            cache.get(&key("a")),
            Some(CachedValue::Value("yes".to_owned()))
        );
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
