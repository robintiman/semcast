//! [Ollama] provider — local models via the native HTTP API.
//!
//! The free-local-dev path: completions through `/api/chat`, embeddings
//! through `/api/embed` (which the semantic index needs at roadmap step 2).
//!
//! [Ollama]: https://ollama.com

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
pub const DEFAULT_CHAT_MODEL: &str = "gemma4:e4b";
pub const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    base_url: String,
    model: String,
    embed_model: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    /// A provider talking to Ollama at `http://localhost:11434`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_URL.to_owned(),
            model: model.into(),
            embed_model: DEFAULT_EMBED_MODEL.to_owned(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self.base_url
            .truncate(self.base_url.trim_end_matches('/').len());
        self
    }

    pub fn with_embed_model(mut self, embed_model: impl Into<String>) -> Self {
        self.embed_model = embed_model.into();
        self
    }

    async fn complete_one(&self, request: CompletionRequest) -> Result<Completion> {
        let body = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: &request.system,
                },
                ChatMessage {
                    role: "user",
                    content: &request.input,
                },
            ],
            stream: false,
            // Verify wants a one-word verdict, not a reasoning trace. On a
            // reasoning model the thinking phase would burn the whole
            // `num_predict` budget and leave `content` empty; disabling it
            // yields the bare "yes"/"no" and is ignored by plain models.
            think: false,
            // Ollama's native structured-output field: a JSON Schema the model
            // is constrained to. Absent for `MEANS` verdicts (free-form).
            format: request.schema.as_ref(),
            options: ChatOptions {
                num_predict: request.max_tokens,
            },
        };
        tracing::debug!(
            target: "semcast::model",
            model = %self.model,
            system = %request.system,
            input = %request.input,
            "llm request"
        );
        let response = super::send_with_retry("ollama", || {
            self.client
                .post(format!("{}/api/chat", self.base_url))
                .json(&body)
        })
        .await?;
        let chat: ChatResponse = response
            .json()
            .await
            .map_err(|e| SemcastError::Model(format!("invalid ollama response: {e}")))?;
        tracing::debug!(
            target: "semcast::model",
            model = %self.model,
            input_tokens = chat.prompt_eval_count,
            output_tokens = chat.eval_count,
            response = %chat.message.content,
            "llm response"
        );
        Ok(Completion {
            text: chat.message.content,
            input_tokens: chat.prompt_eval_count,
            output_tokens: chat.eval_count,
        })
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn id(&self) -> ModelId {
        ModelId(format!("ollama/{}", self.model))
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        super::complete_concurrently(requests, |req| self.complete_one(req)).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        let response = super::send_with_retry("ollama", || {
            self.client
                .post(format!("{}/api/embed", self.base_url))
                .json(&EmbedRequest {
                    model: &self.embed_model,
                    input: &texts,
                })
        })
        .await?;
        let embed: EmbedResponse = response
            .json()
            .await
            .map_err(|e| SemcastError::Model(format!("invalid ollama embed response: {e}")))?;
        if embed.embeddings.len() != texts.len() {
            return Err(SemcastError::Model(format!(
                "ollama returned {} embeddings for {} inputs",
                embed.embeddings.len(),
                texts.len()
            )));
        }
        Ok(embed.embeddings)
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    think: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'a serde_json::Value>,
    options: ChatOptions,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatOptions {
    num_predict: usize,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
    #[serde(default)]
    prompt_eval_count: usize,
    #[serde(default)]
    eval_count: usize,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Embedding>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chat_response() {
        let json = r#"{
            "model": "gemma4:e4b",
            "created_at": "2026-07-06T10:00:00Z",
            "message": {"role": "assistant", "content": "yes"},
            "done": true,
            "prompt_eval_count": 214,
            "eval_count": 1
        }"#;
        let parsed: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.message.content, "yes");
        assert_eq!(parsed.prompt_eval_count, 214);
        assert_eq!(parsed.eval_count, 1);
    }

    #[test]
    fn parses_embed_response() {
        let json = r#"{"model": "nomic-embed-text", "embeddings": [[0.1, -0.2], [0.3, 0.4]]}"#;
        let parsed: EmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.embeddings.len(), 2);
        assert_eq!(parsed.embeddings[0], vec![0.1, -0.2]);
    }

    fn chat_request<'a>(format: Option<&'a serde_json::Value>) -> ChatRequest<'a> {
        ChatRequest {
            model: "gemma4:e4b",
            messages: vec![],
            stream: false,
            think: false,
            format,
            options: ChatOptions { num_predict: 8 },
        }
    }

    #[test]
    fn chat_request_omits_format_when_absent() {
        let body = serde_json::to_value(chat_request(None)).unwrap();
        assert!(body.get("format").is_none(), "no format key when free-form");
    }

    #[test]
    fn chat_request_serializes_schema_as_format() {
        let schema = serde_json::json!({"type": "object"});
        let body = serde_json::to_value(chat_request(Some(&schema))).unwrap();
        assert_eq!(body["format"], schema);
    }
}
