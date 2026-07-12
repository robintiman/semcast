//! Model providers — async, batched LLM and embedding calls (`tokio`).
//!
//! Everything above this module talks to models through [`ModelProvider`];
//! swapping a small verify model for a strong extraction model is a matter of
//! handing a different provider to the planner. [`MockModel`] is deterministic
//! and free — the eval harness (roadmap step 5) baseline; [`OllamaProvider`]
//! runs local models (and embeddings, for the index); [`AnthropicProvider`]
//! is the hosted option for verify-quality answers; [`VoyageProvider`] is the
//! hosted option for embeddings (its inverse: no completions).

mod anthropic;
mod mock;
mod ollama;
mod voyage;

pub use anthropic::{AnthropicProvider, DEFAULT_ANTHROPIC_MODEL, DEFAULT_ANTHROPIC_URL};
pub use mock::MockModel;
pub use ollama::{DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL, OllamaProvider};
pub use voyage::{DEFAULT_VOYAGE_MODEL, DEFAULT_VOYAGE_URL, VoyageProvider};

use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::{RequestBuilder, Response, StatusCode};

use crate::{Result, SemcastError};

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

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>>;

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>>;

    /// Preferred number of texts per embed request. The index groups chunks
    /// into windows of this size before calling [`embed`](Self::embed), so a
    /// hosted provider can amortize its per-request overhead into fewer, larger
    /// calls. The default keeps the historical call-site batch of 64.
    fn embed_batch_size(&self) -> usize {
        64
    }
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

/// Retries a throttled or transiently-failed request before giving up.
const MAX_RETRIES: usize = 4;
/// First backoff step; doubles each attempt, capped at [`MAX_BACKOFF`].
const BASE_BACKOFF: Duration = Duration::from_millis(250);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Send an HTTP request with bounded retries on rate-limit (429), transient
/// server errors, and transport failures. `build` is a factory because
/// [`RequestBuilder::send`] consumes the builder, so each attempt needs a fresh
/// one. On success the 2xx [`Response`] is returned for the caller to parse;
/// fatal statuses (4xx other than 429/408/425) and exhausted retries surface as
/// `SemcastError::Model`. `label` names the provider in log lines and errors.
async fn send_with_retry(label: &str, build: impl Fn() -> RequestBuilder) -> Result<Response> {
    let mut attempt = 0;
    loop {
        match build().send().await {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return Ok(response);
                }
                if is_retryable(status) && attempt < MAX_RETRIES {
                    // Honor a server-provided Retry-After (Anthropic sends one;
                    // Voyage may) over our own guess.
                    let delay = retry_after(&response).unwrap_or_else(|| backoff(attempt));
                    tracing::warn!(
                        target: "semcast::model",
                        provider = label,
                        %status,
                        attempt = attempt + 1,
                        delay_ms = delay.as_millis(),
                        "retrying throttled/failed request",
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                let detail = response.text().await.unwrap_or_default();
                return Err(SemcastError::Model(format!(
                    "{label} returned {status}: {detail}"
                )));
            }
            Err(e) if attempt < MAX_RETRIES => {
                let delay = backoff(attempt);
                tracing::warn!(
                    target: "semcast::model",
                    provider = label,
                    attempt = attempt + 1,
                    delay_ms = delay.as_millis(),
                    error = %e,
                    "retrying after transport error",
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(e) => {
                return Err(SemcastError::Model(format!("{label} request failed: {e}")));
            }
        }
    }
}

/// Retryable = rate limits, request-timeout/too-early, and 5xx. Everything
/// else (400/401/403/404/422 …) is a client error that retrying won't fix.
fn is_retryable(status: StatusCode) -> bool {
    matches!(
        status.as_u16(),
        408 | 425 | 429 | 500 | 502 | 503 | 504 | 529
    )
}

/// The `Retry-After` delay a response asks for, if any.
fn retry_after(response: &Response) -> Option<Duration> {
    parse_retry_after(
        response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
    )
}

/// Parse a `Retry-After` header value expressed in integer seconds. The
/// HTTP-date form is treated as absent, falling back to exponential backoff.
fn parse_retry_after(value: Option<&str>) -> Option<Duration> {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Exponential backoff with equal jitter: at least half of `BASE * 2^attempt`
/// (capped at `MAX_BACKOFF`), plus a random slice of the rest — so concurrent
/// retries neither thunder together nor fire near-instantly.
fn backoff(attempt: usize) -> Duration {
    let exp = BASE_BACKOFF.saturating_mul(1u32 << attempt.min(16));
    let capped = exp.min(MAX_BACKOFF);
    let half = capped / 2;
    half + jitter(half)
}

/// A pseudo-random `Duration` in `[0, span]`, seeded from the wall clock to
/// avoid pulling in a full RNG crate. Good enough to decorrelate retries.
fn jitter(span: Duration) -> Duration {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let frac = f64::from(nanos) / 1_000_000_000.0;
    span.mul_f64(frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_covers_rate_limits_and_server_errors() {
        for code in [408, 425, 429, 500, 502, 503, 504, 529] {
            assert!(
                is_retryable(StatusCode::from_u16(code).unwrap()),
                "{code} should be retryable",
            );
        }
        // Client errors retrying won't fix.
        for code in [400, 401, 403, 404, 422] {
            assert!(
                !is_retryable(StatusCode::from_u16(code).unwrap()),
                "{code} should be fatal",
            );
        }
    }

    #[test]
    fn parse_retry_after_reads_integer_seconds_only() {
        assert_eq!(parse_retry_after(Some("2")), Some(Duration::from_secs(2)));
        assert_eq!(
            parse_retry_after(Some("  5 ")),
            Some(Duration::from_secs(5))
        );
        assert_eq!(parse_retry_after(Some("0")), Some(Duration::from_secs(0)));
        // HTTP-date form is not integer seconds → fall back to backoff.
        assert_eq!(
            parse_retry_after(Some("Wed, 21 Oct 2026 07:28:00 GMT")),
            None
        );
        assert_eq!(parse_retry_after(Some("")), None);
        assert_eq!(parse_retry_after(None), None);
    }

    #[test]
    fn backoff_grows_but_stays_capped() {
        // Every step is bounded by MAX_BACKOFF and at least half the (capped)
        // exponential floor, so it never fires near-instantly at high attempts.
        for attempt in 0..8 {
            let delay = backoff(attempt);
            assert!(
                delay <= MAX_BACKOFF,
                "attempt {attempt}: {delay:?} over cap"
            );
        }
        // The lower bound (half the capped exponential) climbs with attempts
        // until it saturates at MAX_BACKOFF/2.
        assert!(backoff(0) >= BASE_BACKOFF / 2);
        // By attempt 10 the exponential has saturated at the cap.
        assert!(backoff(10) >= MAX_BACKOFF / 2);
    }
}
