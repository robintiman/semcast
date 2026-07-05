//! Deterministic in-process model for tests and the eval baseline.

use async_trait::async_trait;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::Result;

/// Answers "yes" to a completion iff the input contains one of the configured
/// substrings; embeds by byte histogram. Deterministic and free.
#[derive(Debug, Default)]
pub struct MockModel {
    truthy: Vec<String>,
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
        }
    }
}

#[async_trait]
impl ModelProvider for MockModel {
    fn id(&self) -> ModelId {
        ModelId("mock".to_owned())
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
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
