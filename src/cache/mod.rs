//! Field-level cache with provenance keys (roadmap step 4).
//!
//! Pay once per `(type version, field, input value, model, prompt version)`,
//! shared across every query that ever asks again. First evaluation wins, so
//! re-running a query is deterministic even though the model isn't — and the
//! cache doubles as a checkpoint for resumed jobs.

pub mod sqlite;

use std::collections::HashMap;
use std::sync::Mutex;

use crate::model::ModelId;

/// A stable 128-bit digest of `parts`: BLAKE3 over each part's length (u64,
/// little-endian) followed by its bytes, truncated to the first 16 bytes.
///
/// Cache entries are meant to outlive the process, so — like
/// [`crate::index::doc_hash`] — this must never change; golden tests pin it.
/// `DefaultHasher` is only stable within one binary and must not key
/// anything durable. Length-prefixing keeps `["ab", "c"]` and `["a", "bc"]`
/// apart.
pub fn stable_digest(parts: &[&[u8]]) -> u128 {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut first = [0_u8; 16];
    first.copy_from_slice(&digest.as_bytes()[..16]);
    u128::from_le_bytes(first)
}

/// Full provenance of one model verdict. Editing one field's doc line bumps
/// that field's `type_version` and invalidates exactly its entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Semantic type name + version, or the `MEANS` condition for predicates.
    pub type_version: String,
    pub field: String,
    /// [`stable_digest`] of the input value sent (a transcript, a chunk set).
    pub input_digest: u128,
    pub model_id: ModelId,
    /// Version of the prompt-synthesis scheme itself.
    pub prompt_version: String,
}

/// A cached verdict: the decoded value, or the error the row failed with
/// ("rows fail, queries don't").
#[derive(Debug, Clone, PartialEq)]
pub enum CachedValue {
    Value(String),
    Error(String),
}

pub trait SemanticCache: std::fmt::Debug + Send + Sync {
    fn get(&self, key: &CacheKey) -> Option<CachedValue>;
    fn put(&self, key: CacheKey, value: CachedValue);

    /// Store verdicts that only make sense together (an extraction unit's
    /// fields). A durable cache commits them atomically, so a crash never
    /// leaves part of a group behind to be mixed with a later run's answers.
    fn put_many(&self, entries: Vec<(CacheKey, CachedValue)>) {
        for (key, value) in entries {
            self.put(key, value);
        }
    }
}

/// Process-local cache — enough for tests and single-session use. For one
/// that survives restarts, see [`sqlite::SqliteCache`].
#[derive(Debug, Default)]
pub struct InMemoryCache {
    entries: Mutex<HashMap<CacheKey, CachedValue>>,
}

impl SemanticCache for InMemoryCache {
    fn get(&self, key: &CacheKey) -> Option<CachedValue> {
        self.entries.lock().unwrap().get(key).cloned()
    }

    fn put(&self, key: CacheKey, value: CachedValue) {
        self.entries.lock().unwrap().insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_digest_matches_golden_values() {
        // Golden values lock the cache keyspace: a failure here means every
        // persisted verdict would silently stop matching its request.
        assert_eq!(stable_digest(&[]), GOLDEN_EMPTY);
        assert_eq!(stable_digest(&[b"hello world"]), GOLDEN_HELLO);
        assert_eq!(stable_digest(&[b"ab", b"c"]), GOLDEN_AB_C);
    }

    #[test]
    fn stable_digest_separates_part_boundaries() {
        assert_ne!(stable_digest(&[b"ab", b"c"]), stable_digest(&[b"a", b"bc"]));
        assert_ne!(stable_digest(&[b""]), stable_digest(&[]));
    }

    // BLAKE3("") truncated: the published BLAKE3 test vector af1349b9…
    const GOLDEN_EMPTY: u128 = 98081761841810822994326224503485043631;
    const GOLDEN_HELLO: u128 = 279069530831315160929398200261679935565;
    const GOLDEN_AB_C: u128 = 167606391424009241341458254025850656416;
}
