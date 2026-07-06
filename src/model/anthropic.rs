//! Anthropic provider — the Messages API over raw HTTP (there is no official
//! Rust SDK). Default model is Claude Haiku: the cheapest, fastest tier, which
//! is exactly right for one-word yes/no verify calls.
//!
//! No embeddings — Anthropic doesn't offer embedding models, so the semantic
//! index needs a different provider (e.g. [`OllamaProvider`]) for that half.
//!
//! [`OllamaProvider`]: super::OllamaProvider

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

pub const DEFAULT_ANTHROPIC_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-4-5";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_ANTHROPIC_URL.to_owned(),
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Reads `ANTHROPIC_API_KEY` and uses the default (Haiku) verify model.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            SemcastError::Model(
                "ANTHROPIC_API_KEY is not set; export it or use OllamaProvider".to_owned(),
            )
        })?;
        Ok(Self::new(api_key, DEFAULT_ANTHROPIC_MODEL))
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self.base_url.truncate(self.base_url.trim_end_matches('/').len());
        self
    }

    async fn complete_one(&self, request: CompletionRequest) -> Result<Completion> {
        let body = json!({
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.input}],
        });
        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| SemcastError::Model(format!("anthropic request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(SemcastError::Model(format!(
                "anthropic returned {status}: {detail}"
            )));
        }

        let message: MessagesResponse = response
            .json()
            .await
            .map_err(|e| SemcastError::Model(format!("invalid anthropic response: {e}")))?;

        // A refusal is a row-level failure: the row gets dropped and counted,
        // never treated as a "no".
        if message.stop_reason.as_deref() == Some("refusal") {
            return Err(SemcastError::Model("anthropic refused the request".to_owned()));
        }
        let text = message
            .content
            .iter()
            .find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                ContentBlock::Other => None,
            })
            .ok_or_else(|| {
                SemcastError::Model("anthropic response contained no text block".to_owned())
            })?;
        Ok(Completion {
            text,
            input_tokens: message.usage.input_tokens,
            output_tokens: message.usage.output_tokens,
        })
    }
}

// Manual impl: never print the API key.
impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"[redacted]")
            .finish()
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn id(&self) -> ModelId {
        ModelId(format!("anthropic/{}", self.model))
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        super::complete_concurrently(requests, |req| self.complete_one(req)).await
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Embedding>> {
        Err(SemcastError::Model(
            "the Anthropic API has no embedding models; use OllamaProvider for the \
             semantic index"
                .to_owned(),
        ))
    }
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: Usage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Serialize)]
struct Usage {
    input_tokens: usize,
    output_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_messages_response() {
        let json = r#"{
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "yes"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 212, "output_tokens": 1}
        }"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(&parsed.content[0], ContentBlock::Text { text } if text == "yes"));
        assert_eq!(parsed.usage.input_tokens, 212);
        assert_eq!(parsed.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn skips_non_text_blocks() {
        let json = r#"{
            "content": [
                {"type": "thinking", "thinking": "", "signature": "abc"},
                {"type": "text", "text": "no"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 1}
        }"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        let text = parsed.content.iter().find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Other => None,
        });
        assert_eq!(text, Some("no"));
    }

    #[test]
    fn debug_redacts_api_key() {
        let provider = AnthropicProvider::new("sk-ant-secret", DEFAULT_ANTHROPIC_MODEL);
        let debug = format!("{provider:?}");
        assert!(!debug.contains("secret"), "{debug}");
    }
}
