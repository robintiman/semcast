//! Model providers — async, batched LLM and embedding calls (`tokio`).
//!
//! Everything above this module talks to models through [`ModelProvider`];
//! swapping a small verify model for a strong extraction model is a matter of
//! handing a different provider to the planner. Real HTTP-backed providers are
//! deliberately absent from the skeleton — [`MockModel`] is deterministic and
//! free, which is what the eval harness (roadmap step 5) needs as a baseline.

mod mock;

pub use mock::MockModel;

use async_trait::async_trait;

use crate::Result;

/// Identifies a model for cache provenance and cost estimation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One completion request. Prompts are synthesized by semcast (from a
/// semantic type or a `MEANS` condition), never written by users.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Synthesized instruction (the condition to check, the fields to extract).
    pub system: String,
    /// The text under scrutiny — a transcript, or its top-scoring chunks.
    pub input: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    /// Token counts feed `EXPLAIN` cost estimates and the `BUDGET` cap.
    pub input_tokens: usize,
    pub output_tokens: usize,
}

pub type Embedding = Vec<f32>;

/// A model endpoint. Batched by design: physical operators hand over whole
/// record batches so providers can pipeline requests.
///
/// `complete` returns one result per request — a single bad row must not fail
/// the batch ("rows fail, queries don't").
#[async_trait]
pub trait ModelProvider: std::fmt::Debug + Send + Sync {
    fn id(&self) -> ModelId;

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>>;

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>>;
}
