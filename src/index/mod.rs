//! The semantic index (`CREATE SEMANTIC INDEX`) — roadmap step 2.
//!
//! Chunk each document (~512-token slices), embed every chunk, keep it fresh
//! as rows arrive — exactly as incremental as any other index. It pulls
//! double duty: the same chunk vectors that pre-filter candidates also tell
//! the verify step *which* chunks to show the model.
//!
//! Backing store: [Lance] (Arrow-native). Documents are identified by
//! [`doc_hash`] of their text value, not row position — robust to joins,
//! projections, and reordering, and it lets the pre-filter pass unindexed
//! rows through to full-text verification instead of silently dropping them.
//!
//! [Lance]: https://lancedb.github.io/lance/

pub mod chunk;
pub mod lance;
pub mod registry;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, StringArray};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::DataType;
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;

use crate::model::{ModelId, ModelProvider};
use crate::{Result, SemcastError};
use chunk::ChunkConfig;
use registry::SemcastRuntime;

use self::lance::LanceIndex;

/// Stable identity of a document: FNV-1a over the text value. Index entries
/// outlive the process, so this must never change — `DefaultHasher` (which
/// keys the in-memory verdict cache) is only stable within one binary and
/// must not leak into the index.
pub fn doc_hash(text: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A hit from an index search: which document, which chunk, how close —
/// and the chunk text itself, doubling as the verify stage's evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkHit {
    pub doc_hash: u64,
    /// Which chunk of that document matched.
    pub chunk_index: usize,
    /// Cosine similarity to the query; thresholds are best-effort until
    /// `WITH RECALL` calibration (roadmap step 3).
    pub score: f32,
    pub text: String,
}

/// Knobs for the index pre-filter stage. Uncalibrated defaults; `WITH
/// RECALL` (roadmap step 3) will derive `score_floor` from a sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    /// Chunk hits fetched from the vector scan before the floor is applied.
    pub fetch_k: usize,
    /// Minimum cosine similarity for a chunk to count as a hit.
    pub score_floor: f32,
    /// Chunks per document handed to the verify stage as evidence.
    pub chunks_per_doc: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            fetch_k: 256,
            score_floor: 0.35,
            chunks_per_doc: 3,
        }
    }
}

#[async_trait]
pub trait SemanticIndex: std::fmt::Debug + Send + Sync {
    /// Provenance: which model produced the stored vectors. Searching with
    /// a different embedder is an error, never a silent mismatch.
    fn embed_model_id(&self) -> ModelId;

    /// The search knobs this index was created with.
    fn search_params(&self) -> SearchParams;

    /// Chunk, embed, and store documents. The caller pre-dedupes against
    /// [`SemanticIndex::indexed_doc_hashes`]; returns how many documents
    /// were actually indexed (whitespace-only texts chunk to nothing).
    async fn add_documents(&self, texts: Vec<String>) -> Result<usize>;

    /// Every document the index knows. The pre-filter uses this to tell
    /// "indexed but below threshold" (prune) from "never indexed" (pass
    /// through to verify).
    async fn indexed_doc_hashes(&self) -> Result<HashSet<u64>>;

    /// Chunk hits scoring at least `params.score_floor` against the embedded
    /// query, best first, at most `params.fetch_k`.
    async fn search(&self, query: &str, params: &SearchParams) -> Result<Vec<ChunkHit>>;
}

/// Options for [`create_semantic_index`]. The defaults index into a
/// temp-dir Lance dataset using the session's model as embedder.
#[derive(Debug, Clone, Default)]
pub struct IndexOptions {
    /// Embedding provider; defaults to the session model. Bring an Ollama
    /// provider here when the session model can't embed (Anthropic).
    pub embedder: Option<Arc<dyn ModelProvider>>,
    /// Where the Lance dataset lives; defaults under the runtime's index root.
    pub path: Option<PathBuf>,
    pub chunk: ChunkConfig,
    pub search: SearchParams,
}

/// Chunk, embed, and index `table.column`, and register the index with the
/// session so `MEANS` predicates over that column plan an index pre-filter
/// stage. Re-creating an existing index overwrites it.
pub async fn create_semantic_index(
    ctx: &SessionContext,
    table: &str,
    column: &str,
    options: IndexOptions,
) -> Result<Arc<dyn SemanticIndex>> {
    let runtime = runtime(ctx)?;
    let embedder = options
        .embedder
        .unwrap_or_else(|| Arc::clone(&runtime.model));
    let texts = read_column_texts(ctx, table, column).await?;
    let path = options
        .path
        .unwrap_or_else(|| runtime.index_root().join(format!("{table}.{column}.lance")));
    let uri = path
        .to_str()
        .ok_or_else(|| SemcastError::Index("index path is not valid UTF-8".to_owned()))?;
    let index = LanceIndex::create(uri, embedder, options.chunk, options.search).await?;
    index.add_documents(texts).await?;
    let index: Arc<dyn SemanticIndex> = Arc::new(index);
    runtime.register_index(table, column, Arc::clone(&index));
    Ok(index)
}

/// Incremental maintenance: index documents in `table.column` that the
/// registered index has not seen. Returns how many were added.
pub async fn refresh_semantic_index(
    ctx: &SessionContext,
    table: &str,
    column: &str,
) -> Result<usize> {
    let runtime = runtime(ctx)?;
    let index = runtime
        .index_for(table, column)
        .ok_or_else(|| SemcastError::Index(format!("no semantic index on {table}({column})")))?;
    let indexed = index.indexed_doc_hashes().await?;
    let new_texts: Vec<String> = read_column_texts(ctx, table, column)
        .await?
        .into_iter()
        .filter(|text| !indexed.contains(&doc_hash(text)))
        .collect();
    index.add_documents(new_texts).await
}

fn runtime(ctx: &SessionContext) -> Result<Arc<SemcastRuntime>> {
    ctx.state()
        .config()
        .get_extension::<SemcastRuntime>()
        .ok_or_else(|| {
            SemcastError::Index(
                "not a semcast context — build it with semcast_context()".to_owned(),
            )
        })
}

/// All non-null texts of `table.column`, deduped by [`doc_hash`].
async fn read_column_texts(ctx: &SessionContext, table: &str, column: &str) -> Result<Vec<String>> {
    let batches = ctx
        .table(table)
        .await?
        .select_columns(&[column])?
        .collect()
        .await?;
    let mut seen = HashSet::new();
    let mut texts = Vec::new();
    for batch in &batches {
        let strings =
            compute::cast(batch.column(0), &DataType::Utf8).map_err(DataFusionError::from)?;
        let strings = strings
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast to Utf8 yields StringArray");
        for i in 0..strings.len() {
            if strings.is_null(i) {
                continue;
            }
            let text = strings.value(i);
            if seen.insert(doc_hash(text)) {
                texts.push(text.to_owned());
            }
        }
    }
    Ok(texts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_hash_matches_fnv1a_golden_values() {
        // Golden values lock the on-disk keyspace; a failure here means
        // existing indexes would silently stop matching their documents.
        assert_eq!(doc_hash(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(doc_hash("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(doc_hash("hello world"), 0x779a_65e7_023c_d2e7);
    }

    #[test]
    fn doc_hash_distinguishes_texts() {
        assert_ne!(doc_hash("meeting one"), doc_hash("meeting two"));
    }
}
