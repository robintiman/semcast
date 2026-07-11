//! Voyage AI provider — the embeddings API over raw HTTP (there is no official
//! Rust SDK). Voyage is Anthropic's recommended embedding vendor and the hosted
//! counterpart to [`OllamaProvider`]'s local embeddings: it fills the half of
//! [`ModelProvider`] that Anthropic can't (Anthropic ships no embedding models).
//!
//! Embeddings only — Voyage has no chat/completion models, so `complete` errors
//! and verify calls must go to [`AnthropicProvider`] or [`OllamaProvider`].
//! Because this provider *only* embeds, its [`id`](ModelProvider::id) already
//! names the embedding model, so the default
//! [`embed_model_id`](ModelProvider::embed_model_id) (which returns `id()`) is
//! the correct index-provenance key — no override needed.
//!
//! [`OllamaProvider`]: super::OllamaProvider
//! [`AnthropicProvider`]: super::AnthropicProvider

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

pub const DEFAULT_VOYAGE_URL: &str = "https://api.voyageai.com";
pub const DEFAULT_VOYAGE_MODEL: &str = "voyage-3";

pub struct VoyageProvider {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl VoyageProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_VOYAGE_URL.to_owned(),
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Reads `VOYAGE_API_KEY` and uses the default (`voyage-3`) embedding model.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("VOYAGE_API_KEY").map_err(|_| {
            SemcastError::Model(
                "VOYAGE_API_KEY is not set; export it or use OllamaProvider to embed".to_owned(),
            )
        })?;
        Ok(Self::new(api_key, DEFAULT_VOYAGE_MODEL))
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self.base_url
            .truncate(self.base_url.trim_end_matches('/').len());
        self
    }
}

// Manual impl: never print the API key.
impl std::fmt::Debug for VoyageProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoyageProvider")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"[redacted]")
            .finish()
    }
}

#[async_trait]
impl ModelProvider for VoyageProvider {
    fn id(&self) -> ModelId {
        ModelId(format!("voyage/{}", self.model))
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        // Voyage is embeddings-only. One error per request keeps the batch
        // contract ("rows fail, queries don't") intact.
        requests
            .into_iter()
            .map(|_| {
                Err(SemcastError::Model(
                    "Voyage offers no completion models; use AnthropicProvider or \
                     OllamaProvider for verify calls"
                        .to_owned(),
                ))
            })
            .collect()
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        let response = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&EmbedRequest {
                model: &self.model,
                input: &texts,
            })
            .send()
            .await
            .map_err(|e| SemcastError::Model(format!("voyage embed request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(SemcastError::Model(format!(
                "voyage returned {status}: {detail}"
            )));
        }

        let mut body: EmbedResponse = response
            .json()
            .await
            .map_err(|e| SemcastError::Model(format!("invalid voyage embed response: {e}")))?;
        if body.data.len() != texts.len() {
            return Err(SemcastError::Model(format!(
                "voyage returned {} embeddings for {} inputs",
                body.data.len(),
                texts.len()
            )));
        }
        // The API returns results in input order and tags each with its index;
        // sort by it so a reordered response can never misalign vectors.
        body.data.sort_by_key(|d| d.index);
        Ok(body.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Embedding,
    index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_embed_response_in_index_order() {
        // Deliberately out of order to prove the sort realigns vectors.
        let json = r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                {"object": "embedding", "embedding": [0.1, -0.2], "index": 0}
            ],
            "model": "voyage-3",
            "usage": {"total_tokens": 8}
        }"#;
        let mut body: EmbedResponse = serde_json::from_str(json).unwrap();
        body.data.sort_by_key(|d| d.index);
        let embeddings: Vec<Embedding> = body.data.into_iter().map(|d| d.embedding).collect();
        assert_eq!(embeddings, vec![vec![0.1, -0.2], vec![0.3, 0.4]]);
    }

    #[tokio::test]
    async fn complete_reports_no_completion_support() {
        let provider = VoyageProvider::new("voyage-secret", DEFAULT_VOYAGE_MODEL);
        let results = provider
            .complete(vec![CompletionRequest {
                system: "sys".to_owned(),
                input: "in".to_owned(),
                max_tokens: 8,
                schema: None,
            }])
            .await;
        assert_eq!(results.len(), 1);
        let err = results.into_iter().next().unwrap().unwrap_err();
        assert!(
            err.to_string().contains("no completion models"),
            "got: {err}"
        );
    }

    #[test]
    fn id_and_embed_model_id_name_the_embedding_model() {
        let provider = VoyageProvider::new("voyage-secret", "voyage-3");
        assert_eq!(provider.id().0, "voyage/voyage-3");
        // Embed-only: the default embed_model_id (== id) is the right
        // provenance key, no override needed.
        assert_eq!(provider.embed_model_id().0, "voyage/voyage-3");
    }

    #[test]
    fn debug_redacts_api_key() {
        let provider = VoyageProvider::new("voyage-secret", DEFAULT_VOYAGE_MODEL);
        let debug = format!("{provider:?}");
        assert!(!debug.contains("secret"), "{debug}");
    }
}
