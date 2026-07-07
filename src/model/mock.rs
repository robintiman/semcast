//! Deterministic in-process model for tests and the eval baseline.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::Result;

/// Answers "yes" to a completion iff the input contains one of the configured
/// substrings; embeds by byte histogram. Deterministic and free.
#[derive(Debug, Default)]
pub struct MockModel {
    truthy: Vec<String>,
    calls: AtomicUsize,
    inputs: Mutex<Vec<String>>,
    embed_calls: AtomicUsize,
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
        self.inputs
            .lock()
            .expect("inputs poisoned")
            .extend(requests.iter().map(|req| req.input.clone()));
        requests
            .into_iter()
            .map(|req| {
                let answer = if self.truthy.iter().any(|n| req.input.contains(n.as_str())) {
                    "yes"
                } else {
                    "no"
                };
                Ok(Completion {
                    text: answer.to_owned(),
                    input_tokens: req.input.len() / 4,
                    output_tokens: 1,
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
