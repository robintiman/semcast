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

impl ChunkConfig {
    /// Window size in words, from the ~0.75 words-per-token heuristic.
    fn words_per_chunk(&self) -> usize {
        (self.max_tokens * 3 / 4).max(1)
    }

    /// Overlap in words, clamped so every window advances.
    fn overlap_words(&self) -> usize {
        (self.overlap_tokens * 3 / 4).min(self.words_per_chunk() - 1)
    }
}

/// Split `text` into embedding-sized slices: sliding word windows with
/// overlap, no tokenizer. Words are rejoined with single spaces — chunks
/// feed embeddings and verify prompts, where original whitespace carries
/// no meaning. Whitespace-only text yields no chunks.
pub fn chunk_text(text: &str, config: &ChunkConfig) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let window = config.words_per_chunk();
    let stride = window - config.overlap_words();
    let mut chunks = Vec::with_capacity(words.len().div_ceil(stride));
    let mut start = 0;
    loop {
        let end = (start + window).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            return chunks;
        }
        start += stride;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn empty_and_whitespace_yield_no_chunks() {
        let config = ChunkConfig::default();
        assert!(chunk_text("", &config).is_empty());
        assert!(chunk_text("  \n\t ", &config).is_empty());
    }

    #[test]
    fn short_text_is_one_normalized_chunk() {
        let chunks = chunk_text("  hello\n  world ", &ChunkConfig::default());
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn default_config_windows_and_overlap() {
        // Defaults: 512 tokens → 384-word windows, 64 tokens → 48-word overlap,
        // so the stride is 336 words.
        let config = ChunkConfig::default();
        let chunks = chunk_text(&words(800), &config);
        assert_eq!(chunks.len(), 3); // starts at 0, 336, 672
        assert_eq!(chunks[0].split_whitespace().count(), 384);
        assert_eq!(chunks[1].split_whitespace().count(), 384);
        assert_eq!(chunks[2].split_whitespace().count(), 128);
        // A word near the first boundary appears in both adjacent chunks.
        assert!(chunks[0].contains("w350"));
        assert!(chunks[1].contains("w350"));
    }

    #[test]
    fn exact_window_length_is_a_single_chunk() {
        let config = ChunkConfig::default();
        let chunks = chunk_text(&words(384), &config);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn degenerate_config_still_advances() {
        // overlap >= window must not loop forever.
        let config = ChunkConfig {
            max_tokens: 4,
            overlap_tokens: 100,
        };
        let chunks = chunk_text(&words(10), &config);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }
}
