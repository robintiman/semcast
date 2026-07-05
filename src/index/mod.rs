//! The semantic index (`CREATE SEMANTIC INDEX`) — roadmap step 2.
//!
//! Chunk each document (~512-token slices), embed every chunk, keep it fresh
//! as rows arrive — exactly as incremental as any other index. It pulls
//! double duty: the same chunk vectors that pre-filter candidates also tell
//! the verify step *which* chunks to show the model.
//!
//! Planned backing store: [Lance] (Arrow-native). The dependency is deferred
//! until this trait gets its first real implementation.
//!
//! [Lance]: https://lancedb.github.io/lance/

pub mod chunk;

use async_trait::async_trait;

use crate::Result;

/// A hit from an index search: which row, which chunk, how close.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkHit {
    /// Row id in the indexed table.
    pub row_id: u64,
    /// Which chunk of that row's text matched.
    pub chunk_index: usize,
    /// Similarity score; thresholds over it come from calibration.
    pub score: f32,
}

#[async_trait]
pub trait SemanticIndex: std::fmt::Debug + Send + Sync {
    /// Chunk, embed, and index every existing row of `table.column`.
    async fn build(&self, table: &str, column: &str) -> Result<()>;

    /// Rows scoring above `threshold` against the embedded query, with their
    /// top chunks — the funnel's cheap stage and the verify stage's evidence.
    async fn search(&self, query: &str, threshold: f32) -> Result<Vec<ChunkHit>>;

    /// Incremental maintenance: index rows added since the last call.
    async fn refresh(&self) -> Result<()>;
}
