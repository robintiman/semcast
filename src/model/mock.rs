//! Deterministic in-process model for tests and the eval baseline.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde_json::Value;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::Result;

type JsonResponder = Box<dyn Fn(&CompletionRequest) -> Value + Send + Sync>;

/// Answers "yes" to a completion iff the input contains one of the configured
/// substrings; embeds by byte histogram. Deterministic and free. For typed
/// extraction (`request.schema.is_some()`), a configured JSON responder
/// produces the object instead.
#[derive(Default)]
pub struct MockModel {
    truthy: Vec<String>,
    /// Serves typed-extraction requests: input+schema → the JSON object the
    /// "model" returns. `None` falls back to the yes/no path.
    json: Option<JsonResponder>,
    calls: AtomicUsize,
    inputs: Mutex<Vec<String>>,
    schemas: Mutex<Vec<Option<Value>>>,
    embed_calls: AtomicUsize,
}

impl std::fmt::Debug for MockModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockModel")
            .field("truthy", &self.truthy)
            .field("json", &self.json.as_ref().map(|_| "<responder>"))
            .field("calls", &self.calls)
            .finish_non_exhaustive()
    }
}

impl MockModel {
    /// A mock that answers "yes" whenever the input text contains any of
    /// `needles`, and "no" otherwise. `MockModel::default()` always says "no".
    pub fn answering_yes_to<I, S>(needles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            truthy: needles.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }

    /// A mock that answers typed-extraction requests (`schema.is_some()`) by
    /// calling `f` with the request; the returned JSON object is serialized as
    /// the completion text. Requests without a schema still use the yes/no
    /// path (so a `MEANS` + `CAST` query exercises both).
    pub fn answering_json_with<F>(f: F) -> Self
    where
        F: Fn(&CompletionRequest) -> Value + Send + Sync + 'static,
    {
        Self {
            json: Some(Box::new(f)),
            ..Default::default()
        }
    }

    /// Also answer schemaless (`MEANS`) requests "yes" for these needles — so
    /// one mock can gate a filter *and* serve typed extraction on the
    /// survivors.
    pub fn also_answering_yes_to<I, S>(mut self, needles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.truthy = needles.into_iter().map(Into::into).collect();
        self
    }

    /// Completion requests served so far — lets tests assert what the cache
    /// saved, and the eval harness report calls against the baseline.
    pub fn completion_calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    /// The input of every completion request served, in order — lets tests
    /// assert what the model actually read (full text vs. chunks).
    pub fn completion_inputs(&self) -> Vec<String> {
        self.inputs.lock().expect("inputs poisoned").clone()
    }

    /// The schema of every completion request served, in order — lets tests
    /// assert field pushdown by inspecting which fields the request carried.
    pub fn completion_schemas(&self) -> Vec<Option<Value>> {
        self.schemas.lock().expect("schemas poisoned").clone()
    }

    /// `embed` requests served so far (one per call, however many texts).
    pub fn embed_calls(&self) -> usize {
        self.embed_calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ModelProvider for MockModel {
    fn id(&self) -> ModelId {
        ModelId("mock".to_owned())
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        self.calls.fetch_add(requests.len(), Ordering::Relaxed);
        {
            let mut inputs = self.inputs.lock().expect("inputs poisoned");
            let mut schemas = self.schemas.lock().expect("schemas poisoned");
            for req in &requests {
                inputs.push(req.input.clone());
                schemas.push(req.schema.clone());
            }
        }
        requests
            .into_iter()
            .map(|req| {
                let text = match (&self.json, &req.schema) {
                    (Some(responder), Some(_)) => responder(&req).to_string(),
                    _ => {
                        let matched = self.truthy.iter().any(|n| req.input.contains(n.as_str()));
                        if matched { "yes" } else { "no" }.to_owned()
                    }
                };
                Ok(Completion {
                    output_tokens: text.len() / 4 + 1,
                    input_tokens: req.input.len() / 4,
                    text,
                })
            })
            .collect()
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        self.embed_calls.fetch_add(1, Ordering::Relaxed);
        Ok(texts.iter().map(|t| byte_histogram(t)).collect())
    }
}

const MOCK_EMBEDDING_DIM: usize = 16;

fn byte_histogram(text: &str) -> Embedding {
    let mut v = vec![0.0_f32; MOCK_EMBEDDING_DIM];
    for (i, b) in text.bytes().enumerate() {
        v[i % MOCK_EMBEDDING_DIM] += f32::from(b) / 255.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(input: &str, schema: Option<Value>) -> CompletionRequest {
        CompletionRequest {
            system: "sys".to_owned(),
            input: input.to_owned(),
            max_tokens: 64,
            schema,
        }
    }

    #[tokio::test]
    async fn json_responder_serves_schema_requests_and_records_schemas() {
        let mock = MockModel::answering_json_with(|req| serde_json::json!({"echo": req.input}));
        let schema = serde_json::json!({"type": "object"});
        let out = mock
            .complete(vec![
                request("has schema", Some(schema.clone())),
                request("no schema", None),
            ])
            .await;
        // Schema request → JSON object; schemaless request → yes/no path.
        assert_eq!(out[0].as_ref().unwrap().text, r#"{"echo":"has schema"}"#);
        assert_eq!(out[1].as_ref().unwrap().text, "no");
        assert_eq!(
            mock.completion_schemas(),
            vec![Some(schema), None],
            "schemas recorded in order for pushdown assertions",
        );
    }
}
