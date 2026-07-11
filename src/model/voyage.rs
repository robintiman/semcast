//! [Voyage AI] provider — hosted embeddings over raw HTTP.
//!
//! The mirror image of [`AnthropicProvider`]: embeddings only, no chat models.
//! Pair it with an Ollama or Anthropic session model — the semantic index
//! embeds through this provider while `MEANS` verify calls go elsewhere.
//!
//! Voyage distinguishes `input_type: query|document` for retrieval-tuned
//! embeddings; [`ModelProvider::embed`] carries no such intent, so requests
//! omit it for now — a possible follow-up when the trait grows one.
//!
//! [Voyage AI]: https://docs.voyageai.com
//! [`AnthropicProvider`]: super::AnthropicProvider

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

pub const DEFAULT_VOYAGE_URL: &str = "https://api.voyageai.com";
pub const DEFAULT_VOYAGE_MODEL: &str = "voyage-4-large";
/// Voyage caps `input` at 1000 texts per request.
const MAX_BATCH: usize = 1000;

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

    /// Reads `VOYAGE_API_KEY` and uses the default general-purpose model.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("VOYAGE_API_KEY").map_err(|_| {
            SemcastError::Model(
                "VOYAGE_API_KEY is not set; export it or use OllamaProvider for embeddings"
                    .to_owned(),
            )
        })?;
        Ok(Self::new(api_key, DEFAULT_VOYAGE_MODEL))
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self.base_url
            .truncate(self.base_url.trim_end_matches('/').len());
        self
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let response = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&EmbedRequest {
                input: texts,
                model: &self.model,
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

        let embed: EmbedResponse = response
            .json()
            .await
            .map_err(|e| SemcastError::Model(format!("invalid voyage embed response: {e}")))?;
        if embed.data.len() != texts.len() {
            return Err(SemcastError::Model(format!(
                "voyage returned {} embeddings for {} inputs",
                embed.data.len(),
                texts.len()
            )));
        }
        tracing::debug!(
            target: "semcast::model",
            model = %self.model,
            texts = texts.len(),
            total_tokens = embed.usage.total_tokens,
            "embed response"
        );
        let mut data = embed.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
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
        requests
            .iter()
            .map(|_| {
                Err(SemcastError::Model(
                    "voyage has no chat models; use OllamaProvider or AnthropicProvider \
                     for completions"
                        .to_owned(),
                ))
            })
            .collect()
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH) {
            embeddings.extend(self.embed_batch(batch).await?);
        }
        Ok(embeddings)
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
    usage: Usage,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Embedding,
    index: usize,
}

#[derive(Deserialize)]
struct Usage {
    total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_embeddings_response_ordered_by_index() {
        // `data` deliberately out of index order: outputs must follow inputs.
        let json = r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                {"object": "embedding", "embedding": [0.1, -0.2], "index": 0}
            ],
            "model": "voyage-4-large",
            "usage": {"total_tokens": 10}
        }"#;
        let parsed: EmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.usage.total_tokens, 10);
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);
        let embeddings: Vec<Embedding> = data.into_iter().map(|d| d.embedding).collect();
        assert_eq!(embeddings, vec![vec![0.1, -0.2], vec![0.3, 0.4]]);
    }

    #[test]
    fn embed_request_serializes_input_and_model_only() {
        let texts = vec!["a".to_owned(), "b".to_owned()];
        let body = serde_json::to_value(EmbedRequest {
            input: &texts,
            model: DEFAULT_VOYAGE_MODEL,
        })
        .unwrap();
        assert_eq!(body["input"], serde_json::json!(["a", "b"]));
        assert_eq!(body["model"], DEFAULT_VOYAGE_MODEL);
        assert!(body.get("input_type").is_none(), "no retrieval intent yet");
    }

    #[test]
    fn debug_redacts_api_key() {
        let provider = VoyageProvider::new("pa-secret", DEFAULT_VOYAGE_MODEL);
        let debug = format!("{provider:?}");
        assert!(!debug.contains("secret"), "{debug}");
    }
}
