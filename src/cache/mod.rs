//! Field-level cache with provenance keys (roadmap step 4).
//!
//! Pay once per `(type version, field, input value, model, prompt version)`,
//! shared across every query that ever asks again. First evaluation wins, so
//! re-running a query is deterministic even though the model isn't — and the
//! cache doubles as a checkpoint for resumed jobs.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::model::ModelId;

/// Full provenance of one model verdict. Editing one field's doc line bumps
/// that field's `type_version` and invalidates exactly its entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Semantic type name + version, or the `MEANS` condition for predicates.
    pub type_version: String,
    pub field: String,
    /// Hash of the input value (a transcript, a chunk set).
    pub input_hash: u64,
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
}

/// Process-local cache — enough for tests and single-session use. A
/// persistent (disk-backed) implementation is what makes the cache compound
/// across sessions and act as a checkpoint.
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
