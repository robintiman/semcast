//! The stored semantic index — `EMBED` answers kept in the store, replacing
//! Lance (redesign phase 3).
//!
//! An index is two things: which chunks of which documents it holds, and the
//! vector of each chunk. Vectors are `EMBED` answers, content-addressed like
//! verdicts by a digest of (embed model, dimension, chunk text), so a chunk
//! embedded once — by this index, an earlier build of it, or another index
//! over the same text — is never paid for again. The dimension is part of the
//! key because a model id doesn't pin it (a re-pulled tag, a fallback chain):
//! a vector of another size is a different answer, not a stale copy.
//!
//! Search is exact: every chunk is scored against the query, streamed from
//! the store, so memory stays bounded by the hits rather than the corpus. A
//! document with no vector of the index's dimension is not counted as
//! indexed, so it passes through to verify rather than being pruned on a
//! score it never had.
//!
//! Without a store file the same layout lives in memory for the process.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, params};

use crate::cache::sqlite::{open_store, store_err};
use crate::cache::stable_digest;
use crate::model::{Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

use super::chunk::{ChunkConfig, chunk_text};
use super::{ChunkHit, FilterScan, SearchParams, SemanticIndex, doc_hash};

/// Embedded once when an index is built, to learn its vector dimension
/// before any chunk is looked up.
const DIM_PROBE: &str = "semcast embedding dimension probe";

/// Most rows one write transaction carries, so a large index never holds
/// the store's write lock long enough to time out the verdict cache or the
/// catalog writing beside it.
const WRITE_BATCH: usize = 2_000;

/// Most keys one batched vector read binds.
const MAX_READ_BATCH: usize = 512;

/// The store key of one chunk's `EMBED` answer.
fn embedding_key(model: &ModelId, dim: usize, text: &str) -> [u8; 16] {
    stable_digest(&[
        model.0.as_bytes(),
        &(dim as u64).to_le_bytes(),
        text.as_bytes(),
    ])
    .to_le_bytes()
}

/// One chunk an index holds.
#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub doc_hash: u64,
    pub chunk_index: u32,
    pub text: String,
    pub embedding_key: [u8; 16],
    /// The size of the chunk's vector, kept beside it so counting an
    /// index's documents of one size needs no vector read.
    pub dim: usize,
}

/// Called once per scanned chunk that has a vector: its document, its index
/// within the document, its text and its vector.
pub type ChunkVisitor<'a> = dyn FnMut(u64, u32, &str, &[f32]) + 'a;

/// Where indexes keep their chunks and vectors: the store file, or memory.
pub trait ChunkStore: std::fmt::Debug + Send + Sync {
    /// The stored vectors for `keys`, in order; `None` where none is stored.
    fn vectors(&self, keys: &[[u8; 16]]) -> Result<Vec<Option<Embedding>>>;
    /// Store vectors; a key already stored keeps its first vector.
    fn put_vectors(&self, model: &ModelId, entries: &[([u8; 16], Embedding)]) -> Result<()>;
    /// Forget everything `index` holds. Stored vectors stay: they are
    /// answers, and other indexes may read them.
    fn reset(&self, index: &str) -> Result<()>;
    /// Add `chunks` to `index`.
    fn add_chunks(&self, index: &str, chunks: &[StoredChunk]) -> Result<()>;
    /// Make `chunks` all that `index` holds, at once: a reader sees either
    /// the old chunks or the new ones, never an empty index in between.
    fn replace(&self, index: &str, chunks: &[StoredChunk]) -> Result<()>;
    /// Every document `index` holds a chunk of, and each chunk that has a
    /// vector, read from one snapshot.
    fn scan(&self, index: &str, visit: &mut ChunkVisitor<'_>) -> Result<HashSet<u64>>;
    /// Every document `index` holds a vector of `dim` values for.
    fn doc_hashes(&self, index: &str, dim: usize) -> Result<HashSet<u64>>;
    /// Forget `index` and any half-built copy of it a crashed build left.
    fn drop_index(&self, index: &str) -> Result<()> {
        self.reset(index)
    }
}

/// A scratch name no other build of the same index uses, so two concurrent
/// builds never mix their chunks or swap in each other's half.
fn scratch_name(index: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static BUILDS: AtomicU64 = AtomicU64::new(0);
    format!(
        "{index}\u{0}building\u{0}{}\u{0}{}",
        std::process::id(),
        BUILDS.fetch_add(1, Ordering::Relaxed)
    )
}

/// Chunks and vectors for the process only — the default without a store.
#[derive(Debug, Default)]
pub struct MemoryChunkStore {
    vectors: Mutex<HashMap<[u8; 16], Arc<Embedding>>>,
    /// Shared, so a scan takes a handle rather than copying the corpus.
    chunks: Mutex<HashMap<String, Arc<Vec<StoredChunk>>>>,
}

impl ChunkStore for MemoryChunkStore {
    fn vectors(&self, keys: &[[u8; 16]]) -> Result<Vec<Option<Embedding>>> {
        let vectors = self.vectors.lock().expect("chunk store poisoned");
        Ok(keys
            .iter()
            .map(|key| vectors.get(key).map(|vector| vector.as_ref().clone()))
            .collect())
    }

    fn put_vectors(&self, _model: &ModelId, entries: &[([u8; 16], Embedding)]) -> Result<()> {
        let mut vectors = self.vectors.lock().expect("chunk store poisoned");
        for (key, vector) in entries {
            vectors
                .entry(*key)
                .or_insert_with(|| Arc::new(vector.clone()));
        }
        Ok(())
    }

    fn reset(&self, index: &str) -> Result<()> {
        self.chunks
            .lock()
            .expect("chunk store poisoned")
            .remove(index);
        Ok(())
    }

    fn add_chunks(&self, index: &str, chunks: &[StoredChunk]) -> Result<()> {
        let mut all = self.chunks.lock().expect("chunk store poisoned");
        let entry = all.entry(index.to_owned()).or_default();
        let mut merged: Vec<StoredChunk> = entry.as_ref().clone();
        let mut position: HashMap<(u64, u32), usize> = merged
            .iter()
            .enumerate()
            .map(|(i, chunk)| ((chunk.doc_hash, chunk.chunk_index), i))
            .collect();
        for chunk in chunks {
            match position.get(&(chunk.doc_hash, chunk.chunk_index)) {
                Some(&i) => merged[i] = chunk.clone(),
                None => {
                    position.insert((chunk.doc_hash, chunk.chunk_index), merged.len());
                    merged.push(chunk.clone());
                }
            }
        }
        *entry = Arc::new(merged);
        Ok(())
    }

    fn replace(&self, index: &str, chunks: &[StoredChunk]) -> Result<()> {
        self.chunks
            .lock()
            .expect("chunk store poisoned")
            .insert(index.to_owned(), Arc::new(chunks.to_vec()));
        Ok(())
    }

    fn scan(&self, index: &str, visit: &mut ChunkVisitor<'_>) -> Result<HashSet<u64>> {
        let chunks = self
            .chunks
            .lock()
            .expect("chunk store poisoned")
            .get(index)
            .cloned()
            .unwrap_or_default();
        let mut docs = HashSet::new();
        for chunk in chunks.iter() {
            docs.insert(chunk.doc_hash);
            let vector = self
                .vectors
                .lock()
                .expect("chunk store poisoned")
                .get(&chunk.embedding_key)
                .cloned();
            if let Some(vector) = vector {
                visit(chunk.doc_hash, chunk.chunk_index, &chunk.text, &vector);
            }
        }
        Ok(docs)
    }

    fn doc_hashes(&self, index: &str, dim: usize) -> Result<HashSet<u64>> {
        let chunks = self
            .chunks
            .lock()
            .expect("chunk store poisoned")
            .get(index)
            .cloned()
            .unwrap_or_default();
        let vectors = self.vectors.lock().expect("chunk store poisoned");
        Ok(chunks
            .iter()
            .filter(|chunk| chunk.dim == dim && vectors.contains_key(&chunk.embedding_key))
            .map(|chunk| chunk.doc_hash)
            .collect())
    }
}

/// Chunks and vectors in the store file, next to the verdicts and catalog.
///
/// Two connections: scans read on their own, so a long scan never holds up
/// a write (WAL lets the two proceed together), and writes go in batches of
/// [`WRITE_BATCH`] so they never hold the file's write lock for long.
#[derive(Debug)]
pub struct SqliteChunkStore {
    writer: Mutex<Connection>,
    reader: Mutex<Connection>,
}

impl SqliteChunkStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            writer: Mutex::new(open_store(path.as_ref())?),
            reader: Mutex::new(open_store(path.as_ref())?),
        })
    }

    /// Insert `chunks` into `index`, a batch per transaction.
    fn insert_chunks(&self, index: &str, chunks: &[StoredChunk]) -> Result<()> {
        let mut connection = self.writer.lock().expect("chunk store poisoned");
        for batch in chunks.chunks(WRITE_BATCH) {
            let transaction = connection.transaction().map_err(store_err)?;
            {
                let mut statement = transaction
                    .prepare_cached(
                        "INSERT OR REPLACE INTO index_chunks \
                         (index_name, doc_hash, chunk_index, text, embedding, dim) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    )
                    .map_err(store_err)?;
                for chunk in batch {
                    statement
                        .execute(params![
                            index,
                            chunk.doc_hash as i64,
                            chunk.chunk_index,
                            chunk.text,
                            chunk.embedding_key,
                            chunk.dim as i64
                        ])
                        .map_err(store_err)?;
                }
            }
            transaction.commit().map_err(store_err)?;
        }
        Ok(())
    }
}

/// Vectors are stored as little-endian f32s.
fn encode(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode(bytes: &[u8]) -> Embedding {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

impl ChunkStore for SqliteChunkStore {
    fn vectors(&self, keys: &[[u8; 16]]) -> Result<Vec<Option<Embedding>>> {
        let connection = self.reader.lock().expect("chunk store poisoned");
        let mut found: HashMap<[u8; 16], Embedding> = HashMap::new();
        for chunk in keys.chunks(MAX_READ_BATCH) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let mut statement = connection
                .prepare_cached(&format!(
                    "SELECT key, vector FROM embeddings WHERE key IN ({placeholders})"
                ))
                .map_err(store_err)?;
            let rows = statement
                .query_map(rusqlite::params_from_iter(chunk.iter()), |row| {
                    Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .map_err(store_err)?;
            for row in rows {
                let (key, vector) = row.map_err(store_err)?;
                if let Ok(key) = <[u8; 16]>::try_from(key.as_slice()) {
                    found.insert(key, decode(&vector));
                }
            }
        }
        Ok(keys.iter().map(|key| found.remove(key)).collect())
    }

    fn put_vectors(&self, model: &ModelId, entries: &[([u8; 16], Embedding)]) -> Result<()> {
        let mut connection = self.writer.lock().expect("chunk store poisoned");
        for batch in entries.chunks(WRITE_BATCH) {
            let transaction = connection.transaction().map_err(store_err)?;
            {
                let mut statement = transaction
                    .prepare_cached(
                        "INSERT OR IGNORE INTO embeddings (key, embed_model, vector) \
                         VALUES (?1, ?2, ?3)",
                    )
                    .map_err(store_err)?;
                for (key, vector) in batch {
                    statement
                        .execute(params![key, model.0, encode(vector)])
                        .map_err(store_err)?;
                }
            }
            transaction.commit().map_err(store_err)?;
        }
        Ok(())
    }

    fn reset(&self, index: &str) -> Result<()> {
        self.writer
            .lock()
            .expect("chunk store poisoned")
            .execute(
                "DELETE FROM index_chunks WHERE index_name = ?1",
                params![index],
            )
            .map(|_| ())
            .map_err(store_err)
    }

    fn add_chunks(&self, index: &str, chunks: &[StoredChunk]) -> Result<()> {
        self.insert_chunks(index, chunks)
    }

    /// The index and every scratch copy of it: names from `index` NUL up to
    /// (not including) `index` followed by byte 1.
    fn drop_index(&self, index: &str) -> Result<()> {
        self.writer
            .lock()
            .expect("chunk store poisoned")
            .execute(
                "DELETE FROM index_chunks WHERE index_name = ?1 \
                 OR (index_name >= ?2 AND index_name < ?3)",
                params![index, format!("{index}\u{0}"), format!("{index}\u{1}")],
            )
            .map(|_| ())
            .map_err(store_err)
    }

    /// Built under a scratch name in batches, then swapped in with one short
    /// transaction: the index serves its old chunks until the swap.
    fn replace(&self, index: &str, chunks: &[StoredChunk]) -> Result<()> {
        let scratch = scratch_name(index);
        if let Err(error) = self.insert_chunks(&scratch, chunks) {
            // Best effort: a half-written scratch only costs space.
            let _ = self.reset(&scratch);
            return Err(error);
        }
        let mut connection = self.writer.lock().expect("chunk store poisoned");
        let transaction = connection.transaction().map_err(store_err)?;
        transaction
            .execute(
                "DELETE FROM index_chunks WHERE index_name = ?1",
                params![index],
            )
            .map_err(store_err)?;
        transaction
            .execute(
                "UPDATE index_chunks SET index_name = ?1 WHERE index_name = ?2",
                params![index, scratch],
            )
            .map_err(store_err)?;
        transaction.commit().map_err(store_err)
    }

    /// One read transaction on the reader connection, so the documents
    /// listed and the chunks scored are the same snapshot even while a
    /// refresh writes.
    fn scan(&self, index: &str, visit: &mut ChunkVisitor<'_>) -> Result<HashSet<u64>> {
        let mut connection = self.reader.lock().expect("chunk store poisoned");
        let transaction = connection.transaction().map_err(store_err)?;
        let mut docs = HashSet::new();
        {
            let mut statement = transaction
                .prepare_cached(
                    "SELECT c.doc_hash, c.chunk_index, c.text, e.vector \
                     FROM index_chunks c LEFT JOIN embeddings e ON e.key = c.embedding \
                     WHERE c.index_name = ?1",
                )
                .map_err(store_err)?;
            let mut rows = statement.query(params![index]).map_err(store_err)?;
            while let Some(row) = rows.next().map_err(store_err)? {
                let hash = row.get::<_, i64>(0).map_err(store_err)? as u64;
                docs.insert(hash);
                let vector = row.get_ref(3).map_err(store_err)?;
                let Ok(vector) = vector.as_blob() else {
                    continue;
                };
                let chunk_index = row.get::<_, u32>(1).map_err(store_err)?;
                // Borrowed, not copied: most chunks score below the floor.
                let text = row
                    .get_ref(2)
                    .map_err(store_err)?
                    .as_str()
                    .map_err(|error| SemcastError::Store(error.to_string()))?;
                visit(hash, chunk_index, text, &decode(vector));
            }
        }
        transaction.commit().map_err(store_err)?;
        Ok(docs)
    }

    fn doc_hashes(&self, index: &str, dim: usize) -> Result<HashSet<u64>> {
        let connection = self.reader.lock().expect("chunk store poisoned");
        let mut statement = connection
            .prepare_cached(
                "SELECT DISTINCT doc_hash FROM index_chunks \
                 WHERE index_name = ?1 AND dim = ?2",
            )
            .map_err(store_err)?;
        let hashes = statement
            .query_map(params![index, dim as i64], |row| row.get::<_, i64>(0))
            .map_err(store_err)?
            .map(|hash| hash.map(|hash| hash as u64).map_err(store_err))
            .collect::<Result<HashSet<u64>>>()?;
        Ok(hashes)
    }
}

/// A semantic index whose chunks and vectors live in a [`ChunkStore`].
pub struct StoredIndex {
    name: String,
    store: Arc<dyn ChunkStore>,
    embedder: Arc<dyn ModelProvider>,
    chunk_config: ChunkConfig,
    params: SearchParams,
    /// The dimension every vector of this index has.
    dim: usize,
}

impl std::fmt::Debug for StoredIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredIndex")
            .field("name", &self.name)
            .field("embed_model", &self.embedder.id().0)
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl StoredIndex {
    /// A new index named `name` (one per table and column): one probe embed
    /// call learns the embedder's dimension. It holds nothing until
    /// [`Self::rebuild`] or [`SemanticIndex::add_documents`].
    pub async fn create(
        name: impl Into<String>,
        store: Arc<dyn ChunkStore>,
        embedder: Arc<dyn ModelProvider>,
        chunk_config: ChunkConfig,
        params: SearchParams,
    ) -> Result<Self> {
        let probe = embedder.embed(vec![DIM_PROBE.to_owned()]).await?;
        let dim = probe
            .first()
            .map(Vec::len)
            .filter(|dim| *dim > 0)
            .ok_or_else(|| {
                SemcastError::Index(format!(
                    "embedder {} returned an empty embedding",
                    embedder.id().0
                ))
            })?;
        Ok(Self::open(name, store, embedder, chunk_config, params, dim))
    }

    /// An index whose dimension is already known — one being reattached.
    /// Nothing is embedded.
    pub fn open(
        name: impl Into<String>,
        store: Arc<dyn ChunkStore>,
        embedder: Arc<dyn ModelProvider>,
        chunk_config: ChunkConfig,
        params: SearchParams,
        dim: usize,
    ) -> Self {
        Self {
            name: name.into(),
            store,
            embedder,
            chunk_config,
            params,
            dim,
        }
    }

    /// The dimension of this index's vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Make `texts` all this index holds. Everything is embedded first and
    /// swapped in at once, so the index keeps serving what it held until
    /// the new chunks are ready — and keeps them if embedding fails.
    pub async fn rebuild(&self, texts: Vec<String>) -> Result<usize> {
        let (docs, chunks) = self.chunk(texts).await?;
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        blocking(move || store.replace(&name, &chunks)).await?;
        Ok(docs)
    }

    /// Chunk `texts` and make sure every chunk has a stored vector.
    async fn chunk(&self, texts: Vec<String>) -> Result<(usize, Vec<StoredChunk>)> {
        let mut chunk_texts = Vec::new();
        let mut placements = Vec::new();
        let mut docs = 0;
        for text in &texts {
            let doc_chunks = chunk_text(text, &self.chunk_config);
            if doc_chunks.is_empty() {
                continue;
            }
            docs += 1;
            let hash = doc_hash(text);
            for (i, chunk) in doc_chunks.into_iter().enumerate() {
                placements.push((hash, i as u32));
                chunk_texts.push(chunk);
            }
        }
        if chunk_texts.is_empty() {
            return Ok((0, Vec::new()));
        }
        let model = self.embedder.id();
        let keys: Vec<[u8; 16]> = chunk_texts
            .iter()
            .map(|text| embedding_key(&model, self.dim, text))
            .collect();
        self.ensure_vectors(&chunk_texts, &keys).await?;
        let chunks = placements
            .into_iter()
            .zip(chunk_texts)
            .zip(keys)
            .map(
                |(((doc_hash, chunk_index), text), embedding_key)| StoredChunk {
                    doc_hash,
                    chunk_index,
                    text,
                    embedding_key,
                    dim: self.dim,
                },
            )
            .collect();
        Ok((docs, chunks))
    }

    /// Embed (and store) every chunk the store has no vector for.
    async fn ensure_vectors(&self, texts: &[String], keys: &[[u8; 16]]) -> Result<()> {
        let model = self.embedder.id();
        let store = Arc::clone(&self.store);
        let lookup = keys.to_vec();
        let stored = blocking(move || store.vectors(&lookup)).await?;
        let mut seen = HashSet::new();
        let missing: Vec<usize> = (0..texts.len())
            .filter(|&i| stored[i].is_none() && seen.insert(keys[i]))
            .collect();
        for batch in missing.chunks(self.embedder.embed_batch_size().max(1)) {
            let embedded = self
                .embedder
                .embed(batch.iter().map(|&i| texts[i].clone()).collect())
                .await?;
            if embedded.len() != batch.len() {
                return Err(SemcastError::Index(format!(
                    "embedder {} returned {} vectors for {} chunks",
                    model.0,
                    embedded.len(),
                    batch.len()
                )));
            }
            if let Some(vector) = embedded.iter().find(|vector| vector.len() != self.dim) {
                return Err(SemcastError::Index(format!(
                    "embedder {} returned dimension {}, the index stores {}; recreate the \
                     index with CREATE SEMANTIC INDEX",
                    model.0,
                    vector.len(),
                    self.dim
                )));
            }
            let fresh: Vec<([u8; 16], Embedding)> =
                batch.iter().map(|&i| keys[i]).zip(embedded).collect();
            let store = Arc::clone(&self.store);
            let model = model.clone();
            blocking(move || store.put_vectors(&model, &fresh)).await?;
        }
        Ok(())
    }

    async fn embed_query(&self, query: &str) -> Result<Embedding> {
        let mut embedded = self.embedder.embed(vec![query.to_owned()]).await?;
        embedded
            .pop()
            .filter(|vector| vector.len() == self.dim)
            .ok_or_else(|| {
                SemcastError::Index(format!(
                    "embedder {} did not return a {}-dimensional query vector; recreate \
                     the index with CREATE SEMANTIC INDEX if the model changed",
                    self.embedder.id().0,
                    self.dim
                ))
            })
    }

    /// Score every chunk against `query`, keeping those at or above `floor`,
    /// best first; and the documents indexed, from the same snapshot. Only a
    /// document with a vector of the index's dimension counts as indexed.
    async fn scored(&self, query: &str, floor: f32) -> Result<FilterScan> {
        let mut query = self.embed_query(query).await?;
        normalize(&mut query);
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        let dim = self.dim;
        let (mut hits, indexed) = blocking(move || {
            let mut hits = Vec::new();
            let mut scorable = HashSet::new();
            store.scan(&name, &mut |doc_hash, chunk_index, text, vector| {
                if vector.len() != dim {
                    return;
                }
                scorable.insert(doc_hash);
                let score = cosine(&query, vector);
                if score >= floor {
                    hits.push(ChunkHit {
                        doc_hash,
                        chunk_index: chunk_index as usize,
                        score,
                        text: text.to_owned(),
                    });
                }
            })?;
            Ok((hits, scorable))
        })
        .await?;
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(FilterScan { hits, indexed })
    }
}

#[async_trait]
impl SemanticIndex for StoredIndex {
    fn embed_model_id(&self) -> ModelId {
        self.embedder.id()
    }

    fn search_params(&self) -> SearchParams {
        self.params
    }

    async fn add_documents(&self, texts: Vec<String>) -> Result<usize> {
        let (docs, chunks) = self.chunk(texts).await?;
        if chunks.is_empty() {
            return Ok(docs);
        }
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        blocking(move || store.add_chunks(&name, &chunks)).await?;
        Ok(docs)
    }

    /// Only documents with a vector of this index's size: one refreshed
    /// through an index of another size has no score here, and must pass
    /// through rather than read as indexed-and-pruned.
    async fn indexed_doc_hashes(&self) -> Result<HashSet<u64>> {
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        let dim = self.dim;
        blocking(move || store.doc_hashes(&name, dim)).await
    }

    async fn filter_scan(&self, query: &str, score_floor: f32) -> Result<FilterScan> {
        self.scored(query, score_floor).await
    }

    async fn search(&self, query: &str, params: &SearchParams) -> Result<Vec<ChunkHit>> {
        let mut hits = self.scored(query, params.score_floor).await?.hits;
        hits.truncate(params.fetch_k);
        Ok(hits)
    }

    /// The mean of each document's chunk vectors, renormalized: a document
    /// is its chunks, and cosine distance downstream assumes unit vectors.
    async fn doc_vectors(&self) -> Result<HashMap<u64, Embedding>> {
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        let dim = self.dim;
        let sums = blocking(move || {
            let mut sums: HashMap<u64, (Embedding, usize)> = HashMap::new();
            store.scan(&name, &mut |doc_hash, _, _, vector| {
                if vector.len() != dim {
                    return;
                }
                let (sum, count) = sums.entry(doc_hash).or_insert_with(|| (vec![0.0; dim], 0));
                for (slot, value) in sum.iter_mut().zip(vector) {
                    *slot += value;
                }
                *count += 1;
            })?;
            Ok(sums)
        })
        .await?;
        Ok(sums
            .into_iter()
            .map(|(hash, (mut sum, count))| {
                for value in &mut sum {
                    *value /= count as f32;
                }
                normalize(&mut sum);
                (hash, sum)
            })
            .collect())
    }
}

/// Run blocking store work off the async workers.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| SemcastError::Index(format!("index task failed: {error}")))?
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

/// Cosine similarity of a unit `query` and a `vector` of the same length;
/// 0 for a zero vector.
fn cosine(query: &[f32], vector: &[f32]) -> f32 {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    query.iter().zip(vector).map(|(a, b)| a * b).sum::<f32>() / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_round_trip_through_their_encoding() {
        let vector = vec![0.5, -1.25, 3.0, f32::MIN_POSITIVE];
        assert_eq!(decode(&encode(&vector)), vector);
    }

    #[test]
    fn cosine_of_unit_query() {
        let mut query = vec![3.0, 4.0];
        normalize(&mut query);
        assert!((cosine(&query, &[6.0, 8.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&query, &[4.0, -3.0]).abs() < 1e-6);
        assert_eq!(cosine(&query, &[0.0, 0.0]), 0.0);
    }

    /// The `EMBED` answer key must never change: vectors are stored under it.
    #[test]
    fn embedding_key_matches_golden_value() {
        let model = ModelId("ollama/nomic-embed-text".to_owned());
        let key = embedding_key(&model, 768, "hello");
        assert_eq!(u128::from_le_bytes(key), GOLDEN_EMBEDDING_KEY);
        assert_ne!(
            key,
            embedding_key(&model, 1024, "hello"),
            "another dimension is another answer"
        );
    }

    const GOLDEN_EMBEDDING_KEY: u128 = 38664561312866180288865513835267829184;

    fn chunk(key: [u8; 16], text: &str, doc_hash: u64) -> StoredChunk {
        StoredChunk {
            doc_hash,
            chunk_index: 0,
            text: text.to_owned(),
            embedding_key: key,
            dim: 2,
        }
    }

    /// Two builds of one index swapping in at once: the index ends up holding
    /// one complete build, never a mix and never nothing.
    #[test]
    fn concurrent_replaces_leave_one_whole_build() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteChunkStore::open(dir.path().join("s.sqlite")).unwrap());
        let model = ModelId("m".to_owned());
        let key = embedding_key(&model, 2, "chunk");
        store.put_vectors(&model, &[(key, vec![1.0, 0.0])]).unwrap();
        for _ in 0..20 {
            let builds: Vec<Vec<StoredChunk>> = [(0_u64, 50_u64), (1000, 30)]
                .into_iter()
                .map(|(start, n)| (start..start + n).map(|h| chunk(key, "chunk", h)).collect())
                .collect();
            let threads: Vec<_> = builds
                .clone()
                .into_iter()
                .map(|build| {
                    let store = Arc::clone(&store);
                    std::thread::spawn(move || store.replace("t.c", &build).unwrap())
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            let held = store.doc_hashes("t.c", 2).unwrap();
            let whole: Vec<HashSet<u64>> = builds
                .iter()
                .map(|build| build.iter().map(|c| c.doc_hash).collect())
                .collect();
            assert!(whole.contains(&held), "held {} docs", held.len());
        }
        store.drop_index("t.c").unwrap();
        let leftovers: i64 = store
            .writer
            .lock()
            .unwrap()
            .query_row("SELECT count(*) FROM index_chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(leftovers, 0, "no scratch rows left behind");
    }

    #[test]
    fn both_stores_agree() {
        let dir = tempfile::tempdir().unwrap();
        let stores: Vec<Arc<dyn ChunkStore>> = vec![
            Arc::new(MemoryChunkStore::default()),
            Arc::new(SqliteChunkStore::open(dir.path().join("s.sqlite")).unwrap()),
        ];
        let model = ModelId("m".to_owned());
        for store in stores {
            let key = embedding_key(&model, 2, "chunk");
            store.put_vectors(&model, &[(key, vec![1.0, 0.0])]).unwrap();
            store.put_vectors(&model, &[(key, vec![0.0, 1.0])]).unwrap();
            assert_eq!(
                store.vectors(&[key]).unwrap(),
                [Some(vec![1.0, 0.0])],
                "first wins"
            );
            store
                .add_chunks("t.c", &[chunk(key, "chunk", u64::MAX)])
                .unwrap();
            let mut seen = Vec::new();
            let docs = store
                .scan("t.c", &mut |hash, i, text, vector| {
                    seen.push((hash, i, text.to_owned(), vector.to_vec()))
                })
                .unwrap();
            assert_eq!(docs, HashSet::from([u64::MAX]), "u64 survives SQLite's i64");
            assert_eq!(seen, [(u64::MAX, 0, "chunk".to_owned(), vec![1.0, 0.0])]);
            assert_eq!(
                store.doc_hashes("t.c", 2).unwrap(),
                HashSet::from([u64::MAX])
            );
            assert!(store.doc_hashes("other", 2).unwrap().is_empty());

            assert!(
                store.doc_hashes("t.c", 3).unwrap().is_empty(),
                "a vector of another size doesn't count as indexed"
            );

            // Replace swaps the whole index at once.
            store.replace("t.c", &[chunk(key, "chunk", 7)]).unwrap();
            assert_eq!(store.doc_hashes("t.c", 2).unwrap(), HashSet::from([7]));

            store.drop_index("t.c").unwrap();
            assert!(store.doc_hashes("t.c", 2).unwrap().is_empty());
            assert_eq!(
                store.vectors(&[key]).unwrap(),
                [Some(vec![1.0, 0.0])],
                "vectors stay"
            );
        }
    }
}
