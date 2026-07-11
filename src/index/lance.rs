//! Lance-backed semantic index — the Arrow-native store for chunk vectors.
//!
//! Lance pins its own DataFusion internally, so nothing in this module may
//! use DataFusion types: only Arrow (shared at one version across both
//! trees) crosses the boundary.

use std::collections::HashSet;
use std::sync::Arc;

use ::lance::dataset::{Dataset, WriteMode, WriteParams};
use async_trait::async_trait;
use datafusion::arrow::array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt32Array, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::error::ArrowError;
use futures::TryStreamExt;
use lance_linalg::distance::MetricType;
use tokio::sync::Mutex;

use crate::model::{Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

use super::chunk::{ChunkConfig, chunk_text};
use super::{ChunkHit, SearchParams, SemanticIndex, doc_hash};

const META_EMBED_MODEL: &str = "semcast.embed_model";
const META_EMBED_DIM: &str = "semcast.embed_dim";
const META_CHUNK_MAX_TOKENS: &str = "semcast.chunk_max_tokens";
const META_CHUNK_OVERLAP_TOKENS: &str = "semcast.chunk_overlap_tokens";

/// Chunk texts per `ModelProvider::embed` call, so huge corpora don't build
/// one giant request body.
const EMBED_BATCH: usize = 64;

/// Embedded once at creation to learn the vector dimension before any
/// document arrives.
const DIM_PROBE: &str = "semcast embedding dimension probe";

pub struct LanceIndex {
    uri: String,
    embedder: Arc<dyn ModelProvider>,
    chunk_config: ChunkConfig,
    params: SearchParams,
    dim: usize,
    schema: SchemaRef,
    dataset: Mutex<Dataset>,
}

impl std::fmt::Debug for LanceIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceIndex")
            .field("uri", &self.uri)
            .field("embed_model", &self.embedder.embed_model_id().0)
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl LanceIndex {
    /// Create (or overwrite) an empty index at `uri`. One probe embed call
    /// learns the vector dimension; documents arrive via
    /// [`SemanticIndex::add_documents`].
    pub async fn create(
        uri: &str,
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
                    embedder.embed_model_id().0
                ))
            })?;
        let schema = chunk_schema(dim, &embedder.embed_model_id(), &chunk_config);
        let empty = RecordBatchIterator::new(
            std::iter::empty::<std::result::Result<RecordBatch, ArrowError>>(),
            Arc::clone(&schema),
        );
        let overwrite = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(empty, uri, Some(overwrite))
            .await
            .map_err(lance_err)?;
        Ok(Self {
            uri: uri.to_owned(),
            embedder,
            chunk_config,
            params,
            dim,
            schema,
            dataset: Mutex::new(dataset),
        })
    }

    /// Open an existing index, validating that `embedder` matches the model
    /// the stored vectors came from.
    pub async fn open(
        uri: &str,
        embedder: Arc<dyn ModelProvider>,
        params: SearchParams,
    ) -> Result<Self> {
        let dataset = Dataset::open(uri).await.map_err(lance_err)?;
        let metadata = &dataset.schema().metadata;
        let recorded = metadata.get(META_EMBED_MODEL).ok_or_else(|| {
            SemcastError::Index(format!(
                "{uri} is not a semcast index (no embed-model metadata)"
            ))
        })?;
        if *recorded != embedder.embed_model_id().0 {
            return Err(SemcastError::Index(format!(
                "index at {uri} was built with embed model {recorded}, \
                 but the session embeds with {}",
                embedder.embed_model_id().0
            )));
        }
        let dim = parse_meta(metadata.get(META_EMBED_DIM), uri, META_EMBED_DIM)?;
        let chunk_config = ChunkConfig {
            max_tokens: parse_meta(
                metadata.get(META_CHUNK_MAX_TOKENS),
                uri,
                META_CHUNK_MAX_TOKENS,
            )?,
            overlap_tokens: parse_meta(
                metadata.get(META_CHUNK_OVERLAP_TOKENS),
                uri,
                META_CHUNK_OVERLAP_TOKENS,
            )?,
        };
        let schema = chunk_schema(dim, &embedder.embed_model_id(), &chunk_config);
        Ok(Self {
            uri: uri.to_owned(),
            embedder,
            chunk_config,
            params,
            dim,
            schema,
            dataset: Mutex::new(dataset),
        })
    }

    async fn embed_chunks(&self, chunks: &[(u64, u32, String)]) -> Result<Vec<Embedding>> {
        let mut vectors = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(EMBED_BATCH) {
            let embedded = self
                .embedder
                .embed(batch.iter().map(|(_, _, text)| text.clone()).collect())
                .await?;
            for embedding in &embedded {
                if embedding.len() != self.dim {
                    return Err(SemcastError::Index(format!(
                        "embedder {} returned dimension {}, index stores {}",
                        self.embedder.embed_model_id().0,
                        embedding.len(),
                        self.dim
                    )));
                }
            }
            vectors.extend(embedded);
        }
        Ok(vectors)
    }
}

#[async_trait]
impl SemanticIndex for LanceIndex {
    fn embed_model_id(&self) -> ModelId {
        self.embedder.embed_model_id()
    }

    fn search_params(&self) -> SearchParams {
        self.params
    }

    async fn add_documents(&self, texts: Vec<String>) -> Result<usize> {
        let mut chunks: Vec<(u64, u32, String)> = Vec::new();
        let mut docs = 0;
        for text in &texts {
            let hash = doc_hash(text);
            let doc_chunks = chunk_text(text, &self.chunk_config);
            if doc_chunks.is_empty() {
                continue;
            }
            docs += 1;
            for (i, chunk) in doc_chunks.into_iter().enumerate() {
                chunks.push((hash, i as u32, chunk));
            }
        }
        if chunks.is_empty() {
            return Ok(0);
        }
        let vectors = self.embed_chunks(&chunks).await?;
        let batch = chunk_batch(Arc::clone(&self.schema), self.dim, &chunks, &vectors)?;
        let reader = RecordBatchIterator::new([Ok(batch)], Arc::clone(&self.schema));
        let mut dataset = self.dataset.lock().await;
        dataset.append(reader, None).await.map_err(lance_err)?;
        Ok(docs)
    }

    async fn indexed_doc_hashes(&self) -> Result<HashSet<u64>> {
        let dataset = self.dataset.lock().await.clone();
        let mut scan = dataset.scan();
        scan.project(&["doc_hash"]).map_err(lance_err)?;
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .map_err(lance_err)?
            .try_collect()
            .await
            .map_err(lance_err)?;
        let mut hashes = HashSet::new();
        for batch in &batches {
            let column = column::<UInt64Array>(batch, "doc_hash")?;
            hashes.extend(column.iter().flatten());
        }
        Ok(hashes)
    }

    async fn search(&self, query: &str, params: &SearchParams) -> Result<Vec<ChunkHit>> {
        let mut embedded = self.embedder.embed(vec![query.to_owned()]).await?;
        let query_vector = embedded
            .pop()
            .filter(|v| v.len() == self.dim)
            .ok_or_else(|| {
                SemcastError::Index(format!(
                    "embedder {} did not return a {}-dimensional query vector",
                    self.embedder.embed_model_id().0,
                    self.dim
                ))
            })?;
        let dataset = self.dataset.lock().await.clone();
        let mut scan = dataset.scan();
        scan.project(&["doc_hash", "chunk_index", "text"])
            .map_err(lance_err)?;
        scan.nearest("vector", &Float32Array::from(query_vector), params.fetch_k)
            .map_err(lance_err)?;
        scan.distance_metric(MetricType::Cosine);
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .map_err(lance_err)?
            .try_collect()
            .await
            .map_err(lance_err)?;
        let mut hits = Vec::new();
        for batch in &batches {
            let hashes = column::<UInt64Array>(batch, "doc_hash")?;
            let chunk_indexes = column::<UInt32Array>(batch, "chunk_index")?;
            let texts = column::<StringArray>(batch, "text")?;
            let distances = column::<Float32Array>(batch, "_distance")?;
            for i in 0..batch.num_rows() {
                let score = 1.0 - distances.value(i);
                if score >= params.score_floor {
                    hits.push(ChunkHit {
                        doc_hash: hashes.value(i),
                        chunk_index: chunk_indexes.value(i) as usize,
                        score,
                        text: texts.value(i).to_owned(),
                    });
                }
            }
        }
        Ok(hits)
    }
}

fn chunk_schema(dim: usize, embed_model: &ModelId, chunk_config: &ChunkConfig) -> SchemaRef {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(
        Schema::new(vec![
            Field::new("doc_hash", DataType::UInt64, false),
            Field::new("chunk_index", DataType::UInt32, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(item, dim as i32), false),
        ])
        .with_metadata(
            [
                (META_EMBED_MODEL.to_owned(), embed_model.0.clone()),
                (META_EMBED_DIM.to_owned(), dim.to_string()),
                (
                    META_CHUNK_MAX_TOKENS.to_owned(),
                    chunk_config.max_tokens.to_string(),
                ),
                (
                    META_CHUNK_OVERLAP_TOKENS.to_owned(),
                    chunk_config.overlap_tokens.to_string(),
                ),
            ]
            .into(),
        ),
    )
}

fn chunk_batch(
    schema: SchemaRef,
    dim: usize,
    chunks: &[(u64, u32, String)],
    vectors: &[Embedding],
) -> Result<RecordBatch> {
    let values = Float32Array::from_iter_values(vectors.iter().flatten().copied());
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let vectors = FixedSizeListArray::try_new(item, dim as i32, Arc::new(values), None)
        .map_err(|e| SemcastError::Index(e.to_string()))?;
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from_iter_values(
                chunks.iter().map(|(hash, _, _)| *hash),
            )),
            Arc::new(UInt32Array::from_iter_values(
                chunks.iter().map(|(_, index, _)| *index),
            )),
            Arc::new(StringArray::from_iter_values(
                chunks.iter().map(|(_, _, text)| text.as_str()),
            )),
            Arc::new(vectors),
        ],
    )
    .map_err(|e| SemcastError::Index(e.to_string()))
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<T>())
        .ok_or_else(|| SemcastError::Index(format!("index scan returned no {name} column")))
}

fn lance_err(err: ::lance::Error) -> SemcastError {
    SemcastError::Index(err.to_string())
}

fn parse_meta(value: Option<&String>, uri: &str, key: &str) -> Result<usize> {
    value
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| SemcastError::Index(format!("{uri} has missing or invalid {key} metadata")))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion::arrow::datatypes::Float32Type;
    use lance::dataset::{Dataset, WriteMode, WriteParams};

    use super::*;

    const DIM: i32 = 4;

    fn boundary_schema() -> Arc<Schema> {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        Arc::new(
            Schema::new(vec![
                Field::new("doc_hash", DataType::UInt64, false),
                Field::new("chunk_index", DataType::UInt32, false),
                Field::new("text", DataType::Utf8, false),
                Field::new("vector", DataType::FixedSizeList(item, DIM), false),
            ])
            .with_metadata(HashMap::from([(
                "semcast.embed_model".to_string(),
                "mock/test".to_string(),
            )])),
        )
    }

    fn boundary_batch(rows: &[(u64, u32, &str, [f32; 4])]) -> RecordBatch {
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            rows.iter()
                .map(|(_, _, _, v)| Some(v.iter().copied().map(Some).collect::<Vec<_>>())),
            DIM,
        );
        RecordBatch::try_new(
            boundary_schema(),
            vec![
                Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.0))),
                Arc::new(UInt32Array::from_iter_values(rows.iter().map(|r| r.1))),
                Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.2))),
                Arc::new(vectors),
            ],
        )
        .expect("valid chunk batch")
    }

    /// Pins every Lance API the index needs — write, open, schema-metadata
    /// round-trip, flat nearest-vector scan with cosine distance, append —
    /// so API drift surfaces here, not in the index implementation.
    #[tokio::test]
    async fn lance_boundary_roundtrip_nearest_and_append() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("chunks.lance");
        let uri = path.to_str().expect("utf8 path");

        let batch = boundary_batch(&[
            (1, 0, "offline sync launch", [1.0, 0.0, 0.0, 0.0]),
            (2, 0, "quarterly budget", [0.0, 1.0, 0.0, 0.0]),
            (3, 0, "sync rollout planning", [0.7, 0.7, 0.0, 0.0]),
        ]);
        let reader = RecordBatchIterator::new([Ok(batch)], boundary_schema());
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .expect("write dataset");

        let dataset = Dataset::open(uri).await.expect("open dataset");
        assert_eq!(dataset.count_rows(None).await.expect("count"), 3);
        assert_eq!(
            dataset
                .schema()
                .metadata
                .get("semcast.embed_model")
                .map(String::as_str),
            Some("mock/test"),
        );

        let query = Float32Array::from(vec![1.0, 0.0, 0.0, 0.0]);
        let mut scan = dataset.scan();
        scan.nearest("vector", &query, 3).expect("nearest");
        scan.distance_metric(MetricType::Cosine);
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .expect("scan stream")
            .try_collect()
            .await
            .expect("collect hits");

        let mut hits: Vec<(u64, f32)> = Vec::new();
        for batch in &batches {
            let hashes = column::<UInt64Array>(batch, "doc_hash").expect("doc_hash");
            let distances = column::<Float32Array>(batch, "_distance").expect("_distance");
            for i in 0..batch.num_rows() {
                hits.push((hashes.value(i), distances.value(i)));
            }
        }
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, 1, "identical vector is the closest hit");
        assert!(hits[0].1.abs() < 1e-6, "cosine distance to itself ≈ 0");
        assert!(
            hits.windows(2).all(|w| w[0].1 <= w[1].1),
            "hits sorted by ascending distance: {hits:?}",
        );

        let more = boundary_batch(&[(4, 0, "unrelated doc", [0.0, 0.0, 1.0, 0.0])]);
        let reader = RecordBatchIterator::new([Ok(more)], boundary_schema());
        let append = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        Dataset::write(reader, uri, Some(append))
            .await
            .expect("append to dataset");
        let dataset = Dataset::open(uri).await.expect("reopen dataset");
        assert_eq!(dataset.count_rows(None).await.expect("count"), 4);
    }
}
