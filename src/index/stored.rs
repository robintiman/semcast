//! The stored semantic index — `EMBED` answers kept in the store, replacing
//! Lance (redesign phase 3).
//!
//! An index is two things: which chunks of which documents it holds, and the
//! vector of each chunk. Vectors are `EMBED` answers, content-addressed like
//! verdicts by a digest of (embed model, chunk text), so a chunk embedded
//! once — by this index, an earlier build of it, or another index over the
//! same text — is never paid for again.
//!
//! Search is exact: every chunk is scored against the query, streamed from
//! the store, so memory stays bounded by the hits rather than the corpus.
//! Without a store file the same layout lives in memory for the process.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};

use crate::cache::sqlite::{open_store, store_err};
use crate::cache::stable_digest;
use crate::model::{Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

use super::chunk::{ChunkConfig, chunk_text};
use super::{ChunkHit, FilterScan, SearchParams, SemanticIndex, doc_hash};

/// The store key of one chunk's `EMBED` answer.
fn embedding_key(model: &ModelId, text: &str) -> [u8; 16] {
    stable_digest(&[model.0.as_bytes(), text.as_bytes()]).to_le_bytes()
}

/// One chunk an index holds.
#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub doc_hash: u64,
    pub chunk_index: u32,
    pub text: String,
    pub embedding_key: [u8; 16],
}

/// Called once per scanned chunk: its document, its index within the
/// document, its text and its vector.
pub type ChunkVisitor<'a> = dyn FnMut(u64, u32, &str, &[f32]) + 'a;

/// Where indexes keep their chunks and vectors: the store file, or memory.
pub trait ChunkStore: std::fmt::Debug + Send + Sync {
    /// The stored vectors for `keys`, in order; `None` where none is stored.
    fn vectors(&self, keys: &[[u8; 16]]) -> Result<Vec<Option<Embedding>>>;
    /// Store vectors; a key already stored keeps its first vector.
    fn put_vectors(&self, model: &ModelId, entries: &[([u8; 16], Embedding)]) -> Result<()>;
    /// Replace everything `index` holds with `chunks`.
    fn reset(&self, index: &str) -> Result<()>;
    /// Add `chunks` to `index`.
    fn add_chunks(&self, index: &str, chunks: &[StoredChunk]) -> Result<()>;
    /// Every document `index` holds, and each of its chunks with its vector,
    /// read from one snapshot.
    fn scan(&self, index: &str, visit: &mut ChunkVisitor<'_>) -> Result<HashSet<u64>>;
    /// Every document `index` holds.
    fn doc_hashes(&self, index: &str) -> Result<HashSet<u64>>;
}

/// Chunks and vectors for the process only — the default without a store.
#[derive(Debug, Default)]
pub struct MemoryChunkStore {
    vectors: Mutex<HashMap<[u8; 16], Embedding>>,
    chunks: Mutex<HashMap<String, Vec<StoredChunk>>>,
}

impl ChunkStore for MemoryChunkStore {
    fn vectors(&self, keys: &[[u8; 16]]) -> Result<Vec<Option<Embedding>>> {
        let vectors = self.vectors.lock().expect("chunk store poisoned");
        Ok(keys.iter().map(|key| vectors.get(key).cloned()).collect())
    }

    fn put_vectors(&self, _model: &ModelId, entries: &[([u8; 16], Embedding)]) -> Result<()> {
        let mut vectors = self.vectors.lock().expect("chunk store poisoned");
        for (key, vector) in entries {
            vectors.entry(*key).or_insert_with(|| vector.clone());
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
        self.chunks
            .lock()
            .expect("chunk store poisoned")
            .entry(index.to_owned())
            .or_default()
            .extend(chunks.iter().cloned());
        Ok(())
    }

    fn scan(&self, index: &str, visit: &mut ChunkVisitor<'_>) -> Result<HashSet<u64>> {
        // One snapshot: copy the chunk list under the lock, then score.
        let chunks = self
            .chunks
            .lock()
            .expect("chunk store poisoned")
            .get(index)
            .cloned()
            .unwrap_or_default();
        let vectors = self.vectors.lock().expect("chunk store poisoned");
        let mut docs = HashSet::new();
        for chunk in &chunks {
            docs.insert(chunk.doc_hash);
            if let Some(vector) = vectors.get(&chunk.embedding_key) {
                visit(chunk.doc_hash, chunk.chunk_index, &chunk.text, vector);
            }
        }
        Ok(docs)
    }

    fn doc_hashes(&self, index: &str) -> Result<HashSet<u64>> {
        Ok(self
            .chunks
            .lock()
            .expect("chunk store poisoned")
            .get(index)
            .map(|chunks| chunks.iter().map(|chunk| chunk.doc_hash).collect())
            .unwrap_or_default())
    }
}

/// Chunks and vectors in the store file, next to the verdicts and catalog.
#[derive(Debug)]
pub struct SqliteChunkStore {
    connection: Mutex<Connection>,
}

impl SqliteChunkStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            connection: Mutex::new(open_store(path.as_ref())?),
        })
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
        let connection = self.connection.lock().expect("chunk store poisoned");
        let mut statement = connection
            .prepare_cached("SELECT vector FROM embeddings WHERE key = ?1")
            .map_err(store_err)?;
        keys.iter()
            .map(|key| {
                statement
                    .query_row(params![key], |row| row.get::<_, Vec<u8>>(0))
                    .optional()
                    .map(|bytes| bytes.map(|bytes| decode(&bytes)))
                    .map_err(store_err)
            })
            .collect()
    }

    fn put_vectors(&self, model: &ModelId, entries: &[([u8; 16], Embedding)]) -> Result<()> {
        let mut connection = self.connection.lock().expect("chunk store poisoned");
        let transaction = connection.transaction().map_err(store_err)?;
        {
            let mut statement = transaction
                .prepare_cached(
                    "INSERT OR IGNORE INTO embeddings (key, embed_model, vector) VALUES (?1, ?2, ?3)",
                )
                .map_err(store_err)?;
            for (key, vector) in entries {
                statement
                    .execute(params![key, model.0, encode(vector)])
                    .map_err(store_err)?;
            }
        }
        transaction.commit().map_err(store_err)
    }

    fn reset(&self, index: &str) -> Result<()> {
        self.connection
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
        let mut connection = self.connection.lock().expect("chunk store poisoned");
        let transaction = connection.transaction().map_err(store_err)?;
        {
            let mut statement = transaction
                .prepare_cached(
                    "INSERT OR REPLACE INTO index_chunks \
                     (index_name, doc_hash, chunk_index, text, embedding) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(store_err)?;
            for chunk in chunks {
                statement
                    .execute(params![
                        index,
                        chunk.doc_hash as i64,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.embedding_key
                    ])
                    .map_err(store_err)?;
            }
        }
        transaction.commit().map_err(store_err)
    }

    /// One read transaction, so the documents listed and the chunks scored
    /// are the same snapshot even while a refresh writes.
    fn scan(&self, index: &str, visit: &mut ChunkVisitor<'_>) -> Result<HashSet<u64>> {
        let mut connection = self.connection.lock().expect("chunk store poisoned");
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
                let Some(vector) = row.get::<_, Option<Vec<u8>>>(3).map_err(store_err)? else {
                    continue;
                };
                let chunk_index = row.get::<_, u32>(1).map_err(store_err)?;
                let text = row.get::<_, String>(2).map_err(store_err)?;
                visit(hash, chunk_index, &text, &decode(&vector));
            }
        }
        transaction.commit().map_err(store_err)?;
        Ok(docs)
    }

    fn doc_hashes(&self, index: &str) -> Result<HashSet<u64>> {
        let connection = self.connection.lock().expect("chunk store poisoned");
        let mut statement = connection
            .prepare_cached("SELECT DISTINCT doc_hash FROM index_chunks WHERE index_name = ?1")
            .map_err(store_err)?;
        let hashes = statement
            .query_map(params![index], |row| row.get::<_, i64>(0))
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
}

impl std::fmt::Debug for StoredIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredIndex")
            .field("name", &self.name)
            .field("embed_model", &self.embedder.id().0)
            .finish_non_exhaustive()
    }
}

impl StoredIndex {
    /// An index named `name` (one per table and column) over `store`. Opening
    /// one that already holds chunks reads them as they are; [`Self::reset`]
    /// starts it over.
    pub fn new(
        name: impl Into<String>,
        store: Arc<dyn ChunkStore>,
        embedder: Arc<dyn ModelProvider>,
        chunk_config: ChunkConfig,
        params: SearchParams,
    ) -> Self {
        Self {
            name: name.into(),
            store,
            embedder,
            chunk_config,
            params,
        }
    }

    /// Forget every chunk this index holds. Stored vectors stay: they are
    /// answers, and re-indexing the same text reads them back for free.
    pub async fn reset(&self) -> Result<()> {
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        blocking(move || store.reset(&name)).await
    }

    /// The vectors for `texts`, from the store where they are, embedded (and
    /// stored) where they aren't.
    async fn vectors_for(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let model = self.embedder.id();
        let keys: Vec<[u8; 16]> = texts
            .iter()
            .map(|text| embedding_key(&model, text))
            .collect();
        let store = Arc::clone(&self.store);
        let lookup = keys.clone();
        let mut vectors = blocking(move || store.vectors(&lookup)).await?;
        let missing: Vec<usize> = (0..texts.len()).filter(|&i| vectors[i].is_none()).collect();
        let mut fresh = Vec::with_capacity(missing.len());
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
            for (&i, vector) in batch.iter().zip(embedded) {
                fresh.push((keys[i], vector.clone()));
                vectors[i] = Some(vector);
            }
        }
        if !fresh.is_empty() {
            let store = Arc::clone(&self.store);
            blocking(move || store.put_vectors(&model, &fresh)).await?;
        }
        Ok(vectors
            .into_iter()
            .map(|v| v.expect("filled above"))
            .collect())
    }

    async fn embed_query(&self, query: &str) -> Result<Embedding> {
        let mut embedded = self.embedder.embed(vec![query.to_owned()]).await?;
        embedded.pop().filter(|v| !v.is_empty()).ok_or_else(|| {
            SemcastError::Index(format!(
                "embedder {} returned no query vector",
                self.embedder.id().0
            ))
        })
    }

    /// Score every chunk against `query`, keeping those at or above `floor`,
    /// best first; and the documents indexed, from the same snapshot.
    async fn scored(&self, query: &str, floor: f32) -> Result<FilterScan> {
        let mut query = self.embed_query(query).await?;
        normalize(&mut query);
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        let (mut hits, indexed) = blocking(move || {
            let mut hits = Vec::new();
            let indexed = store.scan(&name, &mut |doc_hash, chunk_index, text, vector| {
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
            Ok((hits, indexed))
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
            return Ok(0);
        }
        // Vectors first: a chunk listed in the index always has one.
        self.vectors_for(&chunk_texts).await?;
        let model = self.embedder.id();
        let chunks: Vec<StoredChunk> = placements
            .into_iter()
            .zip(chunk_texts)
            .map(|((doc_hash, chunk_index), text)| StoredChunk {
                doc_hash,
                chunk_index,
                embedding_key: embedding_key(&model, &text),
                text,
            })
            .collect();
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        blocking(move || store.add_chunks(&name, &chunks)).await?;
        Ok(docs)
    }

    async fn indexed_doc_hashes(&self) -> Result<HashSet<u64>> {
        let store = Arc::clone(&self.store);
        let name = self.name.clone();
        blocking(move || store.doc_hashes(&name)).await
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
        let sums = blocking(move || {
            let mut sums: HashMap<u64, (Embedding, usize)> = HashMap::new();
            store.scan(&name, &mut |doc_hash, _, _, vector| {
                let (sum, count) = sums
                    .entry(doc_hash)
                    .or_insert_with(|| (vec![0.0; vector.len()], 0));
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

/// Cosine similarity of a unit `query` and any `vector`; 0 for a zero
/// vector or a dimension mismatch.
fn cosine(query: &[f32], vector: &[f32]) -> f32 {
    if query.len() != vector.len() {
        return 0.0;
    }
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
        assert_eq!(cosine(&query, &[1.0]), 0.0, "a dimension mismatch scores 0");
    }

    /// The `EMBED` answer key must never change: vectors are stored under it.
    #[test]
    fn embedding_key_matches_golden_value() {
        let key = embedding_key(&ModelId("ollama/nomic-embed-text".to_owned()), "hello");
        assert_eq!(u128::from_le_bytes(key), GOLDEN_EMBEDDING_KEY);
    }

    const GOLDEN_EMBEDDING_KEY: u128 = 289648415725612861662131911017099935454;

    #[test]
    fn both_stores_agree() {
        let dir = tempfile::tempdir().unwrap();
        let stores: Vec<Arc<dyn ChunkStore>> = vec![
            Arc::new(MemoryChunkStore::default()),
            Arc::new(SqliteChunkStore::open(dir.path().join("s.sqlite")).unwrap()),
        ];
        let model = ModelId("m".to_owned());
        for store in stores {
            let key = embedding_key(&model, "chunk");
            store.put_vectors(&model, &[(key, vec![1.0, 0.0])]).unwrap();
            store.put_vectors(&model, &[(key, vec![0.0, 1.0])]).unwrap();
            assert_eq!(
                store.vectors(&[key]).unwrap(),
                [Some(vec![1.0, 0.0])],
                "first wins"
            );
            let chunk = StoredChunk {
                doc_hash: u64::MAX,
                chunk_index: 0,
                text: "chunk".to_owned(),
                embedding_key: key,
            };
            store.add_chunks("t.c", &[chunk]).unwrap();
            let mut seen = Vec::new();
            let docs = store
                .scan("t.c", &mut |hash, i, text, vector| {
                    seen.push((hash, i, text.to_owned(), vector.to_vec()))
                })
                .unwrap();
            assert_eq!(docs, HashSet::from([u64::MAX]), "u64 survives SQLite's i64");
            assert_eq!(seen, [(u64::MAX, 0, "chunk".to_owned(), vec![1.0, 0.0])]);
            assert_eq!(store.doc_hashes("t.c").unwrap(), HashSet::from([u64::MAX]));
            assert!(store.doc_hashes("other").unwrap().is_empty());
            store.reset("t.c").unwrap();
            assert!(store.doc_hashes("t.c").unwrap().is_empty());
            assert_eq!(
                store.vectors(&[key]).unwrap(),
                [Some(vec![1.0, 0.0])],
                "vectors stay"
            );
        }
    }
}
