//! [Ollama] provider — local models via the native HTTP API.
//!
//! The free-local-dev path: completions through `/api/chat`, embeddings
//! through `/api/embed` (which the semantic index needs at roadmap step 2).
//!
//! Every request sets the context length (`num_ctx`). Left to Ollama, it
//! falls back to a default (16k tokens in 0.40) and keeps only the end of a
//! longer input, so a long document's verdict would be about its last pages
//! with nothing said. An input that can't fit is an error instead, before
//! the call when it is clearly too long and after it when the model's own
//! token count shows the prompt was cut: rows fail, they are never answered
//! from part of the text.
//!
//! [Ollama]: https://ollama.com

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
pub const DEFAULT_CHAT_MODEL: &str = "gemma4:e4b";
pub const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
/// Context length requested by default, in tokens: room for a document of
/// about 120k characters (nine CUAD contracts in ten). Raise it with
/// `--ollama-num-ctx` for longer ones, up to the model's own limit; memory
/// grows with it.
pub const DEFAULT_NUM_CTX: usize = 32_768;

/// Characters per token assumed when checking an input fits before the
/// call: generous, so only an input clearly too long is refused unasked.
const CHARS_PER_TOKEN: usize = 4;

/// Refuse an input that clearly can't fit `num_ctx` with room to answer.
fn check_fits(request: &CompletionRequest, num_ctx: usize) -> Result<()> {
    let estimated = (request.system.len() + request.input.len()) / CHARS_PER_TOKEN;
    if estimated + request.max_tokens > num_ctx {
        return Err(SemcastError::Model(format!(
            "input of about {estimated} tokens doesn't fit the model's context \
             of {num_ctx} (--ollama-num-ctx); refused rather than truncated"
        )));
    }
    Ok(())
}

/// Whether Ollama cut the prompt: it evaluated as many tokens as the context
/// leaves room for, which is what truncation looks like from outside.
fn truncated(prompt_eval_count: usize, max_tokens: usize, num_ctx: usize) -> bool {
    prompt_eval_count + max_tokens >= num_ctx
}

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    base_url: String,
    model: String,
    embed_model: String,
    /// Context length requested on every call, in tokens.
    num_ctx: usize,
    client: reqwest::Client,
}

impl OllamaProvider {
    /// A provider talking to Ollama at `http://localhost:11434`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_URL.to_owned(),
            model: model.into(),
            embed_model: DEFAULT_EMBED_MODEL.to_owned(),
            num_ctx: DEFAULT_NUM_CTX,
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

    /// Request this context length, in tokens (see the module docs).
    pub fn with_num_ctx(mut self, num_ctx: usize) -> Self {
        self.num_ctx = num_ctx;
        self
    }

    async fn complete_one(&self, request: CompletionRequest) -> Result<Completion> {
        check_fits(&request, self.num_ctx)?;
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
                temperature: TEMPERATURE,
                num_ctx: self.num_ctx,
            },
            logprobs: request.logprobs.then_some(true),
            // Enough alternatives to see both answers in their usual
            // spellings ("yes", "Yes", " yes", "no", ...).
            top_logprobs: request.logprobs.then_some(TOP_LOGPROBS),
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
        if truncated(chat.prompt_eval_count, request.max_tokens, self.num_ctx) {
            return Err(SemcastError::Model(format!(
                "the model read {} tokens, all its context of {} \
                 (--ollama-num-ctx) allows: the input was cut, so the answer \
                 is not used",
                chat.prompt_eval_count, self.num_ctx
            )));
        }
        let yes_probability = chat
            .logprobs
            .as_deref()
            .and_then(first_answer_token)
            .and_then(|token| {
                let alternatives: Vec<(&str, f32)> = token
                    .top_logprobs
                    .iter()
                    .map(|alt| (alt.token.as_str(), alt.logprob))
                    .collect();
                super::yes_probability(&alternatives)
            });
        Ok(Completion {
            text: chat.message.content,
            input_tokens: chat.prompt_eval_count,
            output_tokens: chat.eval_count,
            yes_probability,
            merged: false,
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

/// Alternatives asked for at each token when `logprobs` is set.
const TOP_LOGPROBS: usize = 10;

/// Greedy decoding: the same text gets the same verdict on every call. A
/// verdict is cached and reused, so it must not be one sample among several.
const TEMPERATURE: f32 = 0.0;

/// The first token that carries text: some models open with a control
/// token, which says nothing about the answer.
fn first_answer_token(tokens: &[TokenLogprobs]) -> Option<&TokenLogprobs> {
    tokens.iter().find(|token| !token.token.trim().is_empty())
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
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<usize>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatOptions {
    num_predict: usize,
    temperature: f32,
    num_ctx: usize,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
    #[serde(default)]
    prompt_eval_count: usize,
    #[serde(default)]
    eval_count: usize,
    /// One entry per generated token, when the request asked for them. A
    /// shape we don't know (a look-alike server's) reads as none: a missing
    /// probability must not cost the answer.
    #[serde(default, deserialize_with = "lenient_logprobs")]
    logprobs: Option<Vec<TokenLogprobs>>,
}

fn lenient_logprobs<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Vec<TokenLogprobs>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).ok())
}

#[derive(Deserialize)]
struct TokenLogprobs {
    token: String,
    #[serde(default)]
    top_logprobs: Vec<Alternative>,
}

#[derive(Deserialize)]
struct Alternative {
    token: String,
    logprob: f32,
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

    fn request(chars: usize) -> CompletionRequest {
        CompletionRequest {
            system: String::new(),
            input: "x".repeat(chars),
            max_tokens: 8,
            schema: None,
            logprobs: false,
        }
    }

    #[test]
    fn an_input_too_long_for_the_context_is_refused_not_truncated() {
        assert!(check_fits(&request(4 * 1000), 2048).is_ok());
        let error = check_fits(&request(4 * 3000), 2048).unwrap_err();
        assert!(
            error.to_string().contains("refused rather than truncated"),
            "{error}"
        );
        // What Ollama reports when it cut the prompt to fit.
        assert!(truncated(16_380, 8, 16_384));
        assert!(!truncated(12_039, 8, 16_384));
    }

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
            options: ChatOptions {
                num_predict: 8,
                temperature: TEMPERATURE,
                num_ctx: DEFAULT_NUM_CTX,
            },
            logprobs: None,
            top_logprobs: None,
        }
    }

    #[test]
    fn chat_request_pins_temperature_to_zero() {
        let body = serde_json::to_value(chat_request(None)).unwrap();
        assert_eq!(body["options"]["temperature"], 0.0);
    }

    #[test]
    fn chat_request_omits_logprobs_unless_asked() {
        let body = serde_json::to_value(chat_request(None)).unwrap();
        assert!(body.get("logprobs").is_none());
        assert!(body.get("top_logprobs").is_none());
    }

    #[test]
    fn reads_yes_probability_from_the_first_answer_token() {
        // Shape of a live /api/chat reply with logprobs (gemma4, think off);
        // the empty first token stands in for a model's control token.
        let json = r#"{
            "message": {"role": "assistant", "content": "yes"},
            "logprobs": [
                {"token": "", "logprob": -0.01, "top_logprobs": [{"token": "", "logprob": -0.01}]},
                {"token": "yes", "logprob": -0.0048, "top_logprobs": [
                    {"token": "yes", "logprob": -0.0048},
                    {"token": "no", "logprob": -7.26}
                ]}
            ]
        }"#;
        let chat: ChatResponse = serde_json::from_str(json).unwrap();
        let token = first_answer_token(chat.logprobs.as_deref().unwrap()).unwrap();
        assert_eq!(token.token, "yes");
        let alternatives: Vec<(&str, f32)> = token
            .top_logprobs
            .iter()
            .map(|alt| (alt.token.as_str(), alt.logprob))
            .collect();
        let p = crate::model::yes_probability(&alternatives).unwrap();
        assert!(p > 0.99, "{p}");
    }

    #[test]
    fn logprobs_in_an_unknown_shape_cost_the_probability_not_the_answer() {
        let json = r#"{"message": {"role": "assistant", "content": "yes"},
                       "logprobs": {"content": [{"token": "yes", "logprob": -0.1}]}}"#;
        let chat: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(chat.message.content, "yes");
        assert!(chat.logprobs.is_none());
    }

    #[test]
    fn a_reply_without_logprobs_still_parses() {
        let json = r#"{"message": {"role": "assistant", "content": "no"}}"#;
        let chat: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(chat.logprobs.is_none());
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
