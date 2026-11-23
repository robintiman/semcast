//! Document vectors for the set-level operators (`MEANING OF`, `SEMANTIC
//! DISTINCT ON`), with or without a semantic index.
//!
//! A document's vector is the normalized mean of its chunks' vectors, as a
//! semantic index builds it. With an index over the column, the vectors it
//! already paid for are read back, and documents it doesn't hold are built
//! the same way (with the index's embedder and chunking). Without one, the
//! column is vectored with the session's embedder and default chunking
//! through a scratch [`StoredIndex`], which adds nothing to the store's
//! indexes; chunk vectors are stored, so the next query reads them back.
//! An index changes the cost, and gives the same vectors as vectoring the
//! column when it was built with those defaults.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::error::Result;
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};

use crate::index::chunk::ChunkConfig;
use crate::index::stored::{ChunkStore, StoredIndex, VECTORS_OF_BATCH};
use crate::index::{SearchParams, SemanticIndex};
use crate::model::{Embedding, ModelId, ModelProvider};

/// Where document vectors come from.
#[derive(Debug, Clone)]
pub enum DocVectors {
    /// A semantic index over the column: its vectors, plus vectors built the
    /// same way for documents it doesn't hold.
    Index(Arc<dyn SemanticIndex>),
    /// No index: vector the column as an index build would (the session's
    /// embedder, default chunking), storing chunk vectors in `store`.
    Embed {
        embedder: Arc<dyn ModelProvider>,
        store: Arc<dyn ChunkStore>,
        /// The vector size per embedder, shared with the session.
        dims: Arc<Mutex<HashMap<String, usize>>>,
    },
}

/// What vectoring cost, as operator metrics the progress NOTICE reads.
pub struct VectorCounts {
    /// Rows an index over the column doesn't hold, vectored for the query.
    pub unindexed_rows: Count,
    /// Documents vectored with no index: chunk vectors already stored are
    /// read back, the rest embedded.
    pub docs_vectored: Count,
    /// Documents an index lacked whose embedding failed: no vector.
    pub docs_unembedded: Count,
}

impl VectorCounts {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            unindexed_rows: MetricBuilder::new(metrics).counter("unindexed_rows", partition),
            docs_vectored: MetricBuilder::new(metrics).counter("docs_vectored", partition),
            docs_unembedded: MetricBuilder::new(metrics).counter("docs_unembedded", partition),
        }
    }
}

impl DocVectors {
    pub fn embed_model_id(&self) -> ModelId {
        match self {
            DocVectors::Index(index) => index.embed_model_id(),
            DocVectors::Embed { embedder, .. } => embedder.embed_id(),
        }
    }

    /// Every distinct document's vector (`hashes`, their `texts`, and how
    /// many input rows each stands for): an index's own, and the rest built
    /// the same way. With an index, the rows it lacks are counted as
    /// unindexed once they have a vector; if embedding them fails, they go
    /// without one rather than failing the query (an index gap degrades).
    /// Without an index, an embedder failure fails the query.
    pub async fn for_documents(
        &self,
        hashes: &[u64],
        texts: &HashMap<u64, String>,
        rows: &HashMap<u64, usize>,
        counts: &VectorCounts,
    ) -> Result<HashMap<u64, Embedding>> {
        match self {
            DocVectors::Index(index) => {
                let mut vectors = index.doc_vectors().await?;
                let missing: Vec<u64> = hashes
                    .iter()
                    .copied()
                    .filter(|hash| !vectors.contains_key(hash))
                    .collect();
                // In batches, so one failure leaves only its own batch
                // ungrouped; what the rest stored is read back next time.
                for batch in missing.chunks(VECTORS_OF_BATCH) {
                    let wanted = batch.iter().map(|hash| texts[hash].clone()).collect();
                    match index.vectors_of(wanted).await {
                        Ok(built) => {
                            for hash in batch {
                                if built.contains_key(hash) {
                                    counts.unindexed_rows.add(rows[hash]);
                                }
                            }
                            vectors.extend(built);
                        }
                        Err(error) => {
                            // An empty text never had a vector to lose.
                            let lost = batch
                                .iter()
                                .filter(|hash| !texts[*hash].trim().is_empty())
                                .count();
                            tracing::warn!(
                                "embedding {lost} documents the index lacks failed; they go \
                                 without a vector: {error}"
                            );
                            counts.docs_unembedded.add(lost);
                        }
                    }
                }
                Ok(vectors)
            }
            DocVectors::Embed {
                embedder,
                store,
                dims,
            } => {
                let wanted: Vec<String> = hashes.iter().map(|hash| texts[hash].clone()).collect();
                let id = embedder.embed_id().0;
                let known = dims.lock().expect("dims lock poisoned").get(&id).copied();
                if let Some(dim) = known {
                    // A scratch index adds nothing to the store's indexes:
                    // it only chunks and embeds as an index build would.
                    let scratch = StoredIndex::open(
                        "",
                        Arc::clone(store),
                        Arc::clone(embedder),
                        ChunkConfig::default(),
                        SearchParams::default(),
                        dim,
                    );
                    match scratch.vectors_of(wanted.clone()).await {
                        Ok(built) => {
                            counts.docs_vectored.add(built.len());
                            return Ok(built);
                        }
                        // The size may have changed under the same id (a
                        // re-pulled model): forget it and probe again.
                        Err(error) => {
                            tracing::warn!(
                                "embedding outside an index failed; probing the embedder again: {error}"
                            );
                            dims.lock().expect("dims lock poisoned").remove(&id);
                        }
                    }
                }
                let scratch = StoredIndex::create(
                    "",
                    Arc::clone(store),
                    Arc::clone(embedder),
                    ChunkConfig::default(),
                    SearchParams::default(),
                )
                .await?;
                let built = scratch.vectors_of(wanted).await?;
                dims.lock()
                    .expect("dims lock poisoned")
                    .insert(id, scratch.dim());
                counts.docs_vectored.add(built.len());
                Ok(built)
            }
        }
    }
}
