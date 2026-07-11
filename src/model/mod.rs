//! Model providers — async, batched LLM and embedding calls (`tokio`).
//!
//! Everything above this module talks to models through [`ModelProvider`];
//! swapping a small verify model for a strong extraction model is a matter of
//! handing a different provider to the planner. [`MockModel`] is deterministic
//! and free — the eval harness (roadmap step 5) baseline; [`OllamaProvider`]
//! runs local models (and embeddings, for the index); [`AnthropicProvider`]
//! is the hosted option for verify-quality answers.

mod anthropic;
mod mock;
mod ollama;
mod voyage;

pub use anthropic::AnthropicProvider;
pub use mock::MockModel;
pub use ollama::{DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL, OllamaProvider};
pub use voyage::{DEFAULT_VOYAGE_MODEL, DEFAULT_VOYAGE_URL, VoyageProvider};

use async_trait::async_trait;
use futures::StreamExt;

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
    /// JSON Schema for a single JSON object the model must return — the
    /// constrained-decoding contract for typed extraction. `None` is today's
    /// free-form behavior (a `MEANS` yes/no verdict).
    pub schema: Option<serde_json::Value>,
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

    /// Provenance key for the semantic index: which model produced the stored
    /// vectors. Distinct from [`id`](ModelProvider::id), which names the
    /// *completion* model — a provider that embeds with a different model than
    /// it completes with (e.g. [`OllamaProvider`], whose `id` is the chat model
    /// but whose `embed` uses a separate embedding model) must override this so
    /// the index's corruption guard keys on the embedding model, not the chat
    /// one. Defaults to `id()`, which is correct for embed-only providers.
    ///
    /// [`OllamaProvider`]: crate::model::OllamaProvider
    fn embed_model_id(&self) -> ModelId {
        self.id()
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>>;

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>>;
}

/// How many requests an HTTP provider keeps in flight at once.
const MAX_IN_FLIGHT: usize = 8;

/// Run `send` over every request with bounded concurrency, preserving order —
/// one result per request, so a single bad row can't fail the batch.
async fn complete_concurrently<F, Fut>(
    requests: Vec<CompletionRequest>,
    send: F,
) -> Vec<Result<Completion>>
where
    F: Fn(CompletionRequest) -> Fut,
    Fut: Future<Output = Result<Completion>>,
{
    futures::stream::iter(requests.into_iter().map(send))
        .buffered(MAX_IN_FLIGHT)
        .collect()
        .await
}
