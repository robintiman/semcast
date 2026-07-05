//! Document chunking for the semantic index.

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkConfig {
    /// Approximate tokens per slice.
    pub max_tokens: usize,
    /// Tokens shared between adjacent slices, so a match straddling a
    /// boundary is not lost.
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 64,
        }
    }
}

/// Split `text` into embedding-sized slices.
pub fn chunk_text(_text: &str, _config: &ChunkConfig) -> Vec<String> {
    todo!("token-aware chunking (roadmap step 2)")
}
