//! The semantic index (`CREATE SEMANTIC INDEX`) — roadmap step 2.
//!
//! Chunk each document (~512-token slices), embed every chunk, keep it fresh
//! as rows arrive — exactly as incremental as any other index. It pulls
//! double duty: the same chunk vectors that pre-filter candidates also tell
//! the verify step *which* chunks to show the model.
//!
//! Backing store: [`stored`] — chunk vectors are `EMBED` answers kept in the
//! store file (or memory), content-addressed by embedder and chunk text, and
//! searched exactly. Documents are identified by [`doc_hash`] of their text
//! value, not row position — robust to joins, projections, and reordering,
//! and it lets the pre-filter pass unindexed rows through to full-text
//! verification instead of silently dropping them.

pub mod chunk;
pub mod dedupe;
pub mod kmeans;
pub mod registry;
pub mod stored;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, StringArray};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::DataType;
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;

use crate::model::{Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};
use chunk::ChunkConfig;
use registry::SemcastRuntime;

use self::stored::StoredIndex;

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

/// What a filter needs from one consistent read of the index.
#[derive(Debug, Clone, Default)]
pub struct FilterScan {
    /// Every chunk at or above the floor, best first.
    pub hits: Vec<ChunkHit>,
    /// Every document the index held when `hits` were scored.
    pub indexed: HashSet<u64>,
}

/// Knobs for the index pre-filter stage. Uncalibrated defaults; `WITH
/// RECALL` (roadmap step 3) will derive `score_floor` from a sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    /// Chunk hits a top-k [`SemanticIndex::search`] fetches before the floor
    /// is applied. The `MEANS` pre-filter ignores it: it scores every chunk.
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
    /// query, best first, at most `params.fetch_k`. Top-k retrieval: right
    /// for ranking, where the cut is the point, wrong for filtering.
    async fn search(&self, query: &str, params: &SearchParams) -> Result<Vec<ChunkHit>>;

    /// Every chunk scoring at least `score_floor` against the embedded query,
    /// best first, with no cap — and the documents indexed, read from the
    /// same snapshot. A filter must judge each indexed document on its own
    /// score: a top-k cut would prune matches for rank alone, and a
    /// membership set read from a later snapshot would prune documents the
    /// scan never saw.
    async fn filter_scan(&self, query: &str, score_floor: f32) -> Result<FilterScan>;

    /// One unit vector per indexed document — the mean of its chunk vectors,
    /// renormalized.
    ///
    /// Search answers "which documents are near *this* query"; clustering
    /// asks which documents are near *each other*, which needs the vectors
    /// themselves. They are already paid for, so reading them back beats
    /// re-embedding a corpus.
    async fn doc_vectors(&self) -> Result<HashMap<u64, Embedding>>;
}

/// Options for [`create_semantic_index`]. The defaults use the session's
/// embedder and keep the index wherever the session keeps indexes (the
/// store file, or memory).
#[derive(Debug, Clone, Default)]
pub struct IndexOptions {
    /// Embedding provider; defaults to the runtime's embedder — the session
    /// model, unless the context builder set one (e.g. Voyage, or Ollama
    /// when the session model can't embed).
    pub embedder: Option<Arc<dyn ModelProvider>>,
    pub chunk: ChunkConfig,
    pub search: SearchParams,
}

/// Chunk, embed, and index `table.column`, and register the index with the
/// session so `MEANS` predicates over that column plan an index pre-filter
/// stage. Re-creating an existing index starts it over; vectors already
/// stored for the same chunks are read back rather than embedded again.
pub async fn create_semantic_index(
    ctx: &SessionContext,
    table: &str,
    column: &str,
    options: IndexOptions,
) -> Result<Arc<dyn SemanticIndex>> {
    let runtime = runtime(ctx)?;
    let embedder = options
        .embedder
        .unwrap_or_else(|| Arc::clone(runtime.embedder()));
    let texts = read_column_texts(ctx, table, column).await?;
    let stored = StoredIndex::create(
        crate::catalog::index_name(table, column),
        Arc::clone(runtime.chunk_store()),
        Arc::clone(&embedder),
        options.chunk.clone(),
        options.search,
    )
    .await?;
    // Embedded first, swapped in at once: an index being re-created keeps
    // serving its old chunks until the new ones are ready.
    stored.rebuild(texts).await?;
    let dim = stored.dim();
    let index: Arc<dyn SemanticIndex> = Arc::new(stored);
    runtime.register_index(table, column, Arc::clone(&index));
    if let Some(catalog) = runtime.catalog() {
        let _guard = catalog.lock().await;
        // An index over an in-memory table would come back without its
        // table, and attach to whatever takes the name next.
        let state = ctx.state();
        let defaults = &state.config().options().catalog;
        let resolved = datafusion::common::TableReference::from(table)
            .resolve(&defaults.default_catalog, &defaults.default_schema)
            .to_string();
        if !catalog.has_relation(&resolved).await? {
            tracing::info!(
                "semantic index on {table}({column}) is kept for this session only: \
                 {table} is not a saved table"
            );
            return Ok(index);
        }
        let saved = SavedIndex {
            table: table.to_owned(),
            column: column.to_owned(),
            embed_model: Some(embedder.id().0),
            dim: Some(dim),
            chunk_max_tokens: Some(options.chunk.max_tokens),
            chunk_overlap_tokens: Some(options.chunk.overlap_tokens),
            path: None,
            fetch_k: options.search.fetch_k,
            score_floor: options.search.score_floor,
            chunks_per_doc: options.search.chunks_per_doc,
        };
        let entry = crate::catalog::Entry {
            kind: crate::catalog::EntryKind::SemanticIndex,
            name: crate::catalog::index_name(table, column),
            definition: serde_json::to_string(&saved).expect("a saved index serializes"),
        };
        let failure = format!(
            "the semantic index on {table}({column}) is built for this session but will be \
             gone after a restart"
        );
        crate::catalog::save(catalog, vec![crate::catalog::Change::Put(entry)], &failure).await?;
    }
    Ok(index)
}

/// What the durable catalog keeps of a semantic index: enough to reopen its
/// stored chunks without embedding anything. An entry written by a build
/// that kept indexes in Lance has a `path` and none of the chunking fields;
/// it is rebuilt on restore.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SavedIndex {
    table: String,
    column: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embed_model: Option<String>,
    /// The dimension of the index's vectors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dim: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chunk_max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chunk_overlap_tokens: Option<usize>,
    /// The Lance dataset an older build kept the index in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<std::path::PathBuf>,
    fetch_k: usize,
    score_floor: f32,
    chunks_per_doc: usize,
}

/// Reopen a semantic index from its saved catalog `definition` and register
/// it with the session — no documents are read and nothing is embedded.
/// Fails when the index was built with another embed model than the
/// session's. Documents added to the table since stay unindexed until a
/// refresh, and pass through to full verification meanwhile.
///
/// An index saved by a build that kept it in Lance is rebuilt from its
/// column: its vectors were never in the store, so this embeds once.
pub async fn reattach_semantic_index(
    ctx: &SessionContext,
    definition: &str,
) -> Result<Arc<dyn SemanticIndex>> {
    let runtime = runtime(ctx)?;
    let saved: SavedIndex = serde_json::from_str(definition)
        .map_err(|error| SemcastError::Index(format!("unreadable saved index: {error}")))?;
    let search = SearchParams {
        fetch_k: saved.fetch_k,
        score_floor: saved.score_floor,
        chunks_per_doc: saved.chunks_per_doc,
    };
    let embedder = Arc::clone(runtime.embedder());
    let (Some(embed_model), Some(max_tokens), Some(overlap_tokens)) = (
        saved.embed_model.as_ref(),
        saved.chunk_max_tokens,
        saved.chunk_overlap_tokens,
    ) else {
        tracing::warn!(
            table = %saved.table,
            column = %saved.column,
            "semantic index saved by an older build (Lance); rebuilding it from its column"
        );
        return create_semantic_index(
            ctx,
            &saved.table,
            &saved.column,
            IndexOptions {
                embedder: None,
                chunk: ChunkConfig::default(),
                search,
            },
        )
        .await;
    };
    if &embedder.id().0 != embed_model {
        return Err(SemcastError::Index(format!(
            "the semantic index on {}({}) was built with embedder {embed_model}, but the \
             session embeds with {}; recreate it with CREATE SEMANTIC INDEX",
            saved.table,
            saved.column,
            embedder.id().0
        )));
    }
    let name = crate::catalog::index_name(&saved.table, &saved.column);
    let chunk = ChunkConfig {
        max_tokens,
        overlap_tokens,
    };
    let store = Arc::clone(runtime.chunk_store());
    let stored = match saved.dim {
        Some(dim) => StoredIndex::open(name, store, embedder, chunk, search, dim),
        // Saved before the dimension was recorded: learn it with one probe.
        None => StoredIndex::create(name, store, embedder, chunk, search).await?,
    };
    let index: Arc<dyn SemanticIndex> = Arc::new(stored);
    if index.indexed_doc_hashes().await?.is_empty() {
        tracing::warn!(
            table = %saved.table,
            column = %saved.column,
            "restored a semantic index with no documents in the store (was the store \
             file replaced?); rows pass through to full verification until \
             CREATE SEMANTIC INDEX rebuilds it"
        );
    }
    runtime.register_index(&saved.table, &saved.column, Arc::clone(&index));
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

pub(crate) fn runtime(ctx: &SessionContext) -> Result<Arc<SemcastRuntime>> {
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
