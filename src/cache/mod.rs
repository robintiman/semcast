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

/// What the model has answered for one question about one text column (see
/// `physical::verify::stats_key`): how often it was asked, how often it said
/// yes, and how much text it read. The optimizer orders filters by it; it
/// never decides an answer, so a count that is slightly off costs calls,
/// not rows.
///
/// Recorded only where the rows asked are an unbiased view of the condition:
/// a verify nothing pre-filters, calibration's full-text labels (a uniform
/// sample), and an unguarded label. Not by a verify behind an index or
/// scorer (mostly matches), a guarded label, a scorer, or `CONFIDENCE()`.
/// A filter stacked behind another still sees only that one's survivors, so
/// its rate is conditional on it. An indexed filter records only through a
/// `WITH RECALL` calibration.
///
/// The text column is named as the plan prints it: `tickets.body` once a
/// filter is pushed down to its table, but an alias (`u.bio`) or expression
/// where it stays above one, so two sources under one alias can share
/// history, and one column can have two. Nothing is keyed by data version,
/// so a table recreated under its old name keeps its history. Each costs
/// calls, never rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AskStats {
    pub asked: u64,
    pub matched: u64,
    /// Characters of input sent, summed over `asked`.
    pub input_chars: u64,
}

impl AskStats {
    /// The share of asked inputs the model said yes to.
    pub fn selectivity(&self) -> Option<f64> {
        (self.asked > 0).then(|| self.matched as f64 / self.asked as f64)
    }

    /// Average characters of input per question asked.
    pub fn chars_per_ask(&self) -> Option<f64> {
        (self.asked > 0).then(|| self.input_chars as f64 / self.asked as f64)
    }

    pub(crate) fn add(&mut self, other: AskStats) {
        self.asked += other.asked;
        self.matched += other.matched;
        self.input_chars += other.input_chars;
    }
}

/// How much of its input a filter keeps at each recall level, from its last
/// calibration: `(recall, share of rows kept)`, recall ascending. The recall
/// budget splits a statement's allowed loss by these (see
/// `optimizer::budget`); like [`AskStats`] they price a plan and never decide
/// a row.
pub type PruneCurve = Vec<(f64, f64)>;

pub trait SemanticCache: std::fmt::Debug + Send + Sync {
    fn get(&self, key: &CacheKey) -> Option<CachedValue>;
    fn put(&self, key: CacheKey, value: CachedValue);

    /// Look up several keys at once, answers in the order asked. A durable
    /// cache reads them in one round trip; the default asks one by one.
    fn get_many(&self, keys: &[CacheKey]) -> Vec<Option<CachedValue>> {
        keys.iter().map(|key| self.get(key)).collect()
    }

    /// Store verdicts that only make sense together (an extraction unit's
    /// fields). A durable cache commits them atomically, so a crash never
    /// leaves part of a group behind to be mixed with a later run's answers.
    fn put_many(&self, entries: Vec<(CacheKey, CachedValue)>) {
        for (key, value) in entries {
            self.put(key, value);
        }
    }

    /// Add fresh model answers to a question's statistics. Only answers the
    /// model gave, not cache hits, so a repeat query adds nothing.
    fn record_stats(&self, _question: u128, _delta: AskStats) {}

    /// A question's statistics so far, if any were recorded.
    fn stats(&self, _question: u128) -> Option<AskStats> {
        None
    }

    /// Replace a filter's pruning curve with its latest calibration's.
    fn record_curve(&self, _key: u128, _curve: PruneCurve) {}

    /// A filter's latest pruning curve, if one was recorded.
    fn curve(&self, _key: u128) -> Option<PruneCurve> {
        None
    }
}

/// Process-local cache — enough for tests and single-session use. For one
/// that survives restarts, see [`sqlite::SqliteCache`].
#[derive(Debug, Default)]
pub struct InMemoryCache {
    entries: Mutex<HashMap<CacheKey, CachedValue>>,
    stats: Mutex<HashMap<u128, AskStats>>,
    curves: Mutex<HashMap<u128, PruneCurve>>,
}

impl SemanticCache for InMemoryCache {
    fn get(&self, key: &CacheKey) -> Option<CachedValue> {
        self.entries.lock().unwrap().get(key).cloned()
    }

    fn put(&self, key: CacheKey, value: CachedValue) {
        self.entries.lock().unwrap().insert(key, value);
    }

    fn record_stats(&self, question: u128, delta: AskStats) {
        self.stats
            .lock()
            .unwrap()
            .entry(question)
            .or_default()
            .add(delta);
    }

    fn stats(&self, question: u128) -> Option<AskStats> {
        self.stats.lock().unwrap().get(&question).copied()
    }

    fn record_curve(&self, key: u128, curve: PruneCurve) {
        self.curves.lock().unwrap().insert(key, curve);
    }

    fn curve(&self, key: u128) -> Option<PruneCurve> {
        self.curves.lock().unwrap().get(&key).cloned()
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
