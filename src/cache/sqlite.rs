//! A [`SemanticCache`] that survives the process: one SQLite file.
//!
//! The in-memory cache makes a repeated query free within one run; this one
//! makes it free across restarts, which is what lets the cache compound and
//! act as a checkpoint for resumed jobs. Keys are [`stable_digest`]s of the
//! full [`CacheKey`], so an entry written by one build is found by the next.
//!
//! Rows fail, queries don't: a read or write the database refuses is logged
//! and treated as a miss, never as a query error. The cost of a broken store
//! is model calls, not answers.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, params};

use super::{CacheKey, CachedValue, SemanticCache, stable_digest};
use crate::{Result, SemcastError};

/// Bumped when the table layout changes; a file from another layout is
/// refused rather than misread.
const SCHEMA_VERSION: i64 = 1;

#[derive(Debug)]
pub struct SqliteCache {
    connection: Mutex<Connection>,
}

impl SqliteCache {
    /// Open (or create) the store at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = Connection::open(path).map_err(store_err)?;
        // WAL lets readers proceed while a write commits; NORMAL sync is
        // durable across process crashes, which is the failure that matters
        // for a cache (a lost tail costs re-asking the model, nothing more).
        connection
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 CREATE TABLE IF NOT EXISTS meta (
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
        let version: Option<i64> = connection
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(store_err)?;
        match version {
            None => {
                connection
                    .execute(
                        "INSERT INTO meta (key, value) VALUES ('schema_version', ?1)",
                        params![SCHEMA_VERSION],
                    )
                    .map_err(store_err)?;
            }
            Some(SCHEMA_VERSION) => {}
            Some(other) => {
                return Err(SemcastError::Store(format!(
                    "store schema version {other} is not the supported {SCHEMA_VERSION}"
                )));
            }
        }
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    /// Entries stored, for inspection and tests.
    pub fn len(&self) -> Result<usize> {
        let connection = self.connection.lock().expect("store lock poisoned");
        let count: i64 = connection
            .query_row("SELECT count(*) FROM verdicts", [], |row| row.get(0))
            .map_err(store_err)?;
        Ok(count as usize)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// The row key: a digest of every field of the [`CacheKey`], so two keys
/// differing anywhere never share a row.
fn row_key(key: &CacheKey) -> [u8; 16] {
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
        let connection = self.connection.lock().expect("store lock poisoned");
        let row = connection
            .query_row(
                "SELECT is_error, value FROM verdicts WHERE key = ?1",
                params![row_key(key)],
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
        let (is_error, value) = match value {
            CachedValue::Value(value) => (false, value),
            CachedValue::Error(error) => (true, error),
        };
        let connection = self.connection.lock().expect("store lock poisoned");
        // First evaluation wins, as in memory: a re-run never overwrites the
        // verdict an earlier query already returned.
        let written = connection.execute(
            "INSERT OR IGNORE INTO verdicts
                 (key, type_version, field, model_id, prompt_version, is_error, value)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                row_key(&key),
                key.type_version,
                key.field,
                key.model_id.0,
                key.prompt_version,
                is_error,
                value,
            ],
        );
        if let Err(error) = written {
            tracing::warn!(target: "semcast::store", %error, "store write failed; the verdict stays uncached");
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
