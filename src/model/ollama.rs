//! [Ollama] provider — local models via the native HTTP API.
//!
//! The free-local-dev path: completions through `/api/chat`, embeddings
//! through `/api/embed` (which the semantic index needs at roadmap step 2).
//!
//! Every chat request sets the context length (`num_ctx`) and turns
//! truncation off. Left to Ollama, the context falls back to a default that
//! depends on the version and machine (16k tokens on 0.40, for example) and a longer
//! prompt is cut to fit, keeping its end, so a long document's verdict would
//! be about its last pages with nothing said. With `truncate: false` Ollama
//! refuses such a prompt instead (an HTTP 400 that counts with its own
//! tokenizer, before any generation), and the row fails: rows fail, they are
//! never answered from part of the text. With `shift: false` an answer that
//! runs out of context stops instead of sliding the prompt out, and an answer
//! that stopped there, short of its own token cap, fails too.
//!
//! Embeddings turn truncation off too. A chunk is sized by a words-per-token
//! guess, and a run like `______` or `......` tokenizes far denser than
//! words do (22 of CUAD's 12,105 chunks pass nomic-embed-text's 2048
//! tokens), so left to Ollama its vector would describe only its start. A
//! batch Ollama refuses for length is split until the input it refused is
//! alone; that input is embedded in pieces, halving until each fits, and its
//! vector is the pieces' mean weighted by their tokens. For a mean-pooling
//! embedder (nomic is one) that approximates the whole: it loses attention
//! across the cuts, and each piece is normalized before the mean. Every word
//! of every chunk counts toward its vector. A store that embedded chunks
//! before this keeps their cut vectors (they are keyed by text, which did
//! not change); dropping the store's `embeddings` rows re-embeds them.
//!
//! Needs an Ollama that knows `truncate` and `shift` (0.40 does); an older
//! one ignores them and cuts silently.
//!
//! [Ollama]: https://ollama.com

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
pub const DEFAULT_CHAT_MODEL: &str = "qwen3.8:27b";
pub const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";

/// Context length requested by default, in tokens: room for a document of
/// about 120k characters (nine CUAD contracts in ten). Raise it with
/// `--ollama-num-ctx` for longer ones, up to the model's own limit; memory
/// grows with it.
pub const DEFAULT_NUM_CTX: usize = 32_768;

/// Whether generation stopped because the context filled, not because the
/// answer reached its own cap: a `"length"` stop short of `max_tokens`.
/// A server that reports no count is given the benefit of the doubt.
fn ran_out_of_context(
    done_reason: Option<&str>,
    eval_count: Option<usize>,
    max_tokens: usize,
) -> bool {
    done_reason == Some("length") && eval_count.is_some_and(|count| count < max_tokens)
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

    /// One `/api/embed` call.
    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        match self.embed_request(&texts).await {
            Ok((embeddings, _)) => Ok(embeddings),
            Err(error) if is_too_long(&error) && texts.len() > 1 => {
                // Split the batch until the input Ollama refused is alone.
                let right = texts[texts.len() / 2..].to_vec();
                let mut left =
                    Box::pin(self.embed_texts(texts[..texts.len() / 2].to_vec())).await?;
                left.extend(Box::pin(self.embed_texts(right)).await?);
                Ok(left)
            }
            Err(error) if is_too_long(&error) => {
                let (mut vector, _) = self.embed_in_halves(&texts[0]).await?;
                unit(&mut vector);
                Ok(vec![vector])
            }
            Err(error) => Err(error),
        }
    }

    /// One text too long for the embedder: its pieces' vectors, averaged by
    /// their tokens, and the weight of the whole (its tokens, less each
    /// piece's two special ones).
    async fn embed_in_halves(&self, text: &str) -> Result<(Embedding, f32)> {
        let Some((first, second)) = halves(text) else {
            return Err(SemcastError::Model(format!(
                "ollama cannot embed {:?} in its context",
                text.chars().take(40).collect::<String>()
            )));
        };
        let mut parts = Vec::with_capacity(2);
        for half in [first, second] {
            parts.push(match self.embed_request(&[half.to_owned()]).await {
                Ok((mut vectors, tokens)) => (
                    vectors.pop().unwrap_or_default(),
                    piece_weight(half, tokens),
                ),
                Err(error) if is_too_long(&error) => Box::pin(self.embed_in_halves(half)).await?,
                Err(error) => return Err(error),
            });
        }
        let weight: f32 = parts.iter().map(|(_, weight)| weight).sum();
        let dim = parts[0].0.len();
        if parts[1].0.len() != dim {
            return Err(SemcastError::Model(
                "ollama embedded two halves of one text at different dimensions".to_owned(),
            ));
        }
        let mut mean = vec![0.0_f32; dim];
        for (vector, part) in &parts {
            let share = part / weight;
            for (sum, value) in mean.iter_mut().zip(vector) {
                *sum += share * value;
            }
        }
        Ok((mean, weight))
    }

    /// One `/api/embed` call with truncation off: the vectors, and the
    /// tokens Ollama counted for the whole input (0 if it didn't say).
    async fn embed_request(&self, texts: &[String]) -> Result<(Vec<Embedding>, usize)> {
        let response = super::send_with_retry("ollama", || {
            self.client
                .post(format!("{}/api/embed", self.base_url))
                .json(&EmbedRequest {
                    model: &self.embed_model,
                    input: texts,
                    truncate: false,
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
        Ok((embed.embeddings, embed.prompt_eval_count.unwrap_or(0)))
    }

    /// Request this context length, in tokens (see the module docs).
    pub fn with_num_ctx(mut self, num_ctx: usize) -> Self {
        self.num_ctx = num_ctx;
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
                temperature: TEMPERATURE,
                num_ctx: self.num_ctx,
            },
            logprobs: request.logprobs.then_some(true),
            // Enough alternatives to see both answers in their usual
            // spellings ("yes", "Yes", " yes", "no", ...).
            top_logprobs: request.logprobs.then_some(TOP_LOGPROBS),
            truncate: false,
            shift: false,
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
            output_tokens = chat.eval_count.unwrap_or_default(),
            response = %chat.message.content,
            "llm response"
        );
        if ran_out_of_context(
            chat.done_reason.as_deref(),
            chat.eval_count,
            request.max_tokens,
        ) {
            return Err(SemcastError::Model(format!(
                "the answer ran out of context after {} tokens (--ollama-num-ctx {}): \
                 a cut-off answer is not used",
                chat.eval_count.unwrap_or_default(),
                self.num_ctx
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
            output_tokens: chat.eval_count.unwrap_or_default(),
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
        self.embed_texts(texts).await
    }

    /// The embed model, not the chat model: it made the vectors.
    ///
    /// Queries and documents are embedded alike. nomic-embed-text's model
    /// card asks for `search_query: ` / `search_document: ` prefixes, but on
    /// CUAD (509 contracts, 5 clauses) they ranked worse on every clause
    /// (AUC 0.64 to 0.84 against 0.67 to 0.85 without). Adding prefixes
    /// later changes every vector, so it must change this id.
    fn embed_id(&self) -> ModelId {
        ModelId(format!("ollama/{}", self.embed_model))
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
    /// Always false: refuse a prompt longer than the context rather than cut
    /// it (see the module docs).
    truncate: bool,
    /// Always false: never slide the prompt out to make room for the answer.
    shift: bool,
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
    eval_count: Option<usize>,
    /// Why generation stopped: `"length"` when it hit a limit.
    #[serde(default)]
    done_reason: Option<String>,
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
    /// Refuse an input past the context (HTTP 400) instead of cutting it.
    truncate: bool,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Embedding>,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
}

/// Ollama's refusal of an input past the embedder's context, under
/// `truncate: false`.
/// Worded "the input length exceeds the context length" on 0.40, "input length
/// exceeds maximum context length" on older releases.
fn is_too_long(error: &SemcastError) -> bool {
    matches!(error, SemcastError::Model(message)
        if message.contains("exceeds") && message.contains("context length"))
}

/// A piece's weight in its text's mean: its tokens less the two special
/// ones Ollama counts with every input, or its characters when Ollama
/// reported no count (so a side cut into more pieces doesn't count more).
fn piece_weight(piece: &str, tokens: usize) -> f32 {
    match tokens {
        0 => piece.chars().count().max(1) as f32,
        tokens => tokens.saturating_sub(2).max(1) as f32,
    }
}

/// `text`, trimmed, cut in two at its middle whitespace, or at the middle
/// character of a single word; both halves trimmed and never empty. `None`
/// for one character, which has no halves.
fn halves(text: &str) -> Option<(&str, &str)> {
    let text = text.trim();
    let gaps: Vec<usize> = text
        .char_indices()
        .filter(|(_, c)| c.is_whitespace())
        .map(|(at, _)| at)
        .collect();
    if let Some(&at) = gaps.get(gaps.len() / 2) {
        return Some((text[..at].trim_end(), text[at..].trim_start()));
    }
    let chars: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
    let &at = chars.get(chars.len() / 2).filter(|&&at| at > 0)?;
    Some((&text[..at], &text[at..]))
}

/// Scale to unit length, as Ollama's own vectors are.
fn unit(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake `/api/embed`: refuses, as Ollama does under `truncate: false`,
    /// any request with an input over `max_chars` (and fails one with an
    /// input saying "boom" for another reason). It embeds the rest as
    /// `[chars, 1]`, counting a token per character plus Ollama's two
    /// special ones unless `count` is false. Hands back every request body.
    async fn embed_server(
        max_chars: usize,
        count: bool,
    ) -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = std::sync::Arc::clone(&bodies);
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                let body = loop {
                    let n = socket.read(&mut buf).await.unwrap();
                    if n == 0 {
                        break None;
                    }
                    request.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&request).to_string();
                    if let Some(end) = text.find("\r\n\r\n") {
                        let length: usize = text[..end]
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse().unwrap())
                            })
                            .unwrap_or(0);
                        if request.len() >= end + 4 + length {
                            break Some(
                                serde_json::from_slice::<serde_json::Value>(
                                    &request[end + 4..end + 4 + length],
                                )
                                .unwrap(),
                            );
                        }
                    }
                };
                let Some(body) = body else { continue };
                let inputs: Vec<String> = body["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|input| input.as_str().unwrap().to_owned())
                    .collect();
                seen.lock().unwrap().push(body);
                let (status, answer) = if inputs.iter().any(|input| input.contains("boom")) {
                    (
                        "400 Bad Request",
                        r#"{"error":"something else"}"#.to_owned(),
                    )
                } else if inputs.iter().any(|input| input.len() > max_chars) {
                    (
                        "400 Bad Request",
                        r#"{"error":"the input length exceeds the context length"}"#.to_owned(),
                    )
                } else {
                    let mut answer = serde_json::json!({
                        "embeddings": inputs.iter().map(|input| vec![input.len() as f32, 1.0]).collect::<Vec<_>>(),
                    });
                    if count {
                        answer["prompt_eval_count"] = inputs
                            .iter()
                            .map(|input| input.len() + 2)
                            .sum::<usize>()
                            .into();
                    }
                    ("200 OK", answer.to_string())
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                    answer.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        (url, bodies)
    }

    fn inputs(bodies: &std::sync::Mutex<Vec<serde_json::Value>>) -> Vec<serde_json::Value> {
        bodies
            .lock()
            .unwrap()
            .iter()
            .map(|body| body["input"].clone())
            .collect()
    }

    /// The unit mean of `[len, 1]` vectors weighted by `len`.
    fn expected(lens: &[f32]) -> Vec<f32> {
        let total: f32 = lens.iter().sum();
        let mut mean = vec![lens.iter().map(|len| len * len).sum::<f32>() / total, 1.0];
        let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
        mean.iter_mut().for_each(|x| *x /= norm);
        mean
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[tokio::test]
    async fn queries_and_documents_are_embedded_alike() {
        for queries in [false, true] {
            let (url, bodies) = embed_server(usize::MAX, true).await;
            let ollama = OllamaProvider::new("qwen3.8:27b").with_base_url(&url);
            let texts = vec!["non-compete".to_owned()];
            if queries {
                ollama.embed_queries(texts).await.unwrap();
            } else {
                ollama.embed(texts).await.unwrap();
            }
            let body = bodies.lock().unwrap()[0].clone();
            assert_eq!(body["model"], "nomic-embed-text");
            assert_eq!(body["input"], serde_json::json!(["non-compete"]));
            assert_eq!(
                body["truncate"], false,
                "an over-long input is refused, not cut"
            );
        }
    }

    /// An input past the context is never cut: the batch is split until it
    /// is alone, and its vector is its pieces' token-weighted mean.
    #[tokio::test]
    async fn an_over_long_input_is_embedded_in_pieces_not_cut() {
        let (url, bodies) = embed_server(40, true).await;
        let ollama = OllamaProvider::new("qwen3.8:27b").with_base_url(&url);
        // Two words of unequal length, 47 chars: the halves fit, and the
        // longer one weighs more.
        let long = format!("{} {}", "a".repeat(10), "b".repeat(36));
        let vectors = ollama
            .embed(vec!["short".to_owned(), long, "also short".to_owned()])
            .await
            .unwrap();
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0], vec![5.0, 1.0]);
        assert_eq!(vectors[2], vec![10.0, 1.0]);
        assert!(
            close(&vectors[1], &expected(&[10.0, 36.0])),
            "{:?}",
            vectors[1]
        );
        let sent = inputs(&bodies);
        assert!(sent.contains(&serde_json::json!(["a".repeat(10)])));
        assert!(sent.contains(&serde_json::json!(["b".repeat(36)])));
        // The batch, its halves (["short"] fits, the other two don't), the
        // pair split again, then the long input's two halves.
        assert_eq!(sent.len(), 7, "{sent:?}");
    }

    /// A half that is still too long is halved again; a single word is cut
    /// by characters. Weights follow characters when Ollama reports no
    /// token count, so a side cut into more pieces doesn't count more.
    #[tokio::test]
    async fn pieces_are_halved_until_they_fit() {
        let (url, _) = embed_server(10, false).await;
        let ollama = OllamaProvider::new("qwen3.8:27b").with_base_url(&url);
        let rule = "_".repeat(36);
        let vectors = ollama.embed(vec![rule]).await.unwrap();
        // 36 → 18 + 18 → four pieces of 9, all equal.
        assert!(close(&vectors[0], &expected(&[9.0; 4])), "{:?}", vectors[0]);
    }

    /// Only a length refusal is split; any other failure fails the batch.
    #[tokio::test]
    async fn another_failure_still_fails_the_batch() {
        let (url, bodies) = embed_server(40, true).await;
        let ollama = OllamaProvider::new("qwen3.8:27b").with_base_url(&url);
        let error = ollama
            .embed(vec!["boom".to_owned(), "fine".to_owned()])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("something else"), "{error}");
        assert_eq!(bodies.lock().unwrap().len(), 1, "no split, no retry");
    }

    #[test]
    fn length_refusals_of_old_and_new_ollama_are_recognized() {
        for message in [
            "ollama returned 400 Bad Request: {\"error\":\"the input length exceeds the context length\"}",
            "ollama returned 400 Bad Request: {\"error\":\"input length exceeds maximum context length\"}",
        ] {
            assert!(is_too_long(&SemcastError::Model(message.to_owned())));
        }
        assert!(!is_too_long(&SemcastError::Model(
            "ollama returned 400: boom".into()
        )));
    }

    #[test]
    fn halves_split_at_the_middle_word_or_character() {
        assert_eq!(halves("a b c d"), Some(("a b", "c d")));
        assert_eq!(
            halves(" a\n\nb "),
            Some(("a", "b")),
            "trimmed, any whitespace"
        );
        assert_eq!(halves("  "), None, "nothing to embed in either half");
        assert_eq!(halves("ab"), Some(("a", "b")));
        assert_eq!(halves("é"), None);
        assert_eq!(halves("ééé"), Some(("é", "éé")));
    }

    /// Vectors are keyed by the embedder, not the chat model: changing
    /// `--model` keeps them, changing `--embed-model` does not.
    #[test]
    fn the_embedder_has_its_own_id() {
        let ollama = OllamaProvider::new("qwen3.8:27b");
        assert_eq!(ollama.id(), ModelId("ollama/qwen3.8:27b".into()));
        assert_eq!(ollama.embed_id(), ModelId("ollama/nomic-embed-text".into()));
        assert_eq!(
            ollama.with_embed_model("mxbai-embed-large").embed_id(),
            ModelId("ollama/mxbai-embed-large".into())
        );
    }

    #[test]
    fn an_answer_cut_by_the_context_is_not_used() {
        assert!(ran_out_of_context(Some("length"), Some(3), 128));
        // Reaching its own cap is an answer, not a cut.
        assert!(!ran_out_of_context(Some("length"), Some(8), 8));
        assert!(!ran_out_of_context(Some("stop"), Some(3), 128));
        assert!(!ran_out_of_context(None, Some(3), 128));
        // No count: no evidence of a cut.
        assert!(!ran_out_of_context(Some("length"), None, 8));
        let chat: ChatResponse = serde_json::from_str(
            r#"{"message":{"content":"ye"},"done_reason":"length","eval_count":1}"#,
        )
        .unwrap();
        assert_eq!(chat.done_reason.as_deref(), Some("length"));
    }

    #[test]
    fn chat_request_refuses_truncation() {
        let body = serde_json::to_value(chat_request(None)).unwrap();
        assert_eq!(body["truncate"], false);
        assert_eq!(body["shift"], false);
        assert_eq!(body["options"]["num_ctx"], DEFAULT_NUM_CTX);
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
        assert_eq!(parsed.eval_count, Some(1));
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
            truncate: false,
            shift: false,
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
