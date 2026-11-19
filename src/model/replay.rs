//! A model that answers `MEANS` from a saved store instead of asking: the
//! replay harness behind `semcast serve --provider replay`.
//!
//! An eval run leaves every verdict it paid for in its `--store` file, keyed
//! by condition, model, prompt version and input text. Replaying a query
//! against that store with a fresh session store reruns the funnel exactly as
//! it would run cold — sampling, calibration, pruning, verify — while each
//! request is answered from the recording rather than the model. A funnel
//! change is then measured in seconds, at no model cost, against the same
//! answers.
//!
//! A request the recording doesn't hold (a chunk set the original run never
//! asked, a condition it never ran) fails its row, as a model error would,
//! and is counted: a replay with misses is not exact. Only yes/no `MEANS`
//! prompts are answered; the provider doesn't embed, so a session pairs it
//! with a live embedder.
//!
//! Each request is appended to an optional log, one JSON object per line
//! (`{"chunked":false,"chars":51234,"answered":true}`), which is how the
//! harness prices a replay: by input characters, the measure that tokens
//! track.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::cache::{CachedValue, SemanticCache};
use crate::physical::verify::{
    MEANS_CHUNKED_PROMPT_VERSION, MEANS_PROMPT_VERSION, cached_score, means_cache_key,
    parse_means_prompt,
};
use crate::semir::normalize_instruction;
use crate::{Result, SemcastError};

/// Characters per token for the estimate a replayed answer reports, which
/// only feeds `EXPLAIN` and budgets; the harness prices by characters.
const CHARS_PER_TOKEN: usize = 4;

/// Answers `MEANS` requests from a recorded store; see the module docs.
#[derive(Debug)]
pub struct ReplayProvider {
    answers: Arc<dyn SemanticCache>,
    model: ModelId,
    log: Option<Mutex<File>>,
    misses: AtomicUsize,
}

impl ReplayProvider {
    /// Replay the answers `model` (its id as stored, e.g.
    /// `ollama/gemma4:e4b`) gave in `answers`.
    pub fn new(answers: Arc<dyn SemanticCache>, model: impl Into<String>) -> Self {
        Self {
            answers,
            model: ModelId(model.into()),
            log: None,
            misses: AtomicUsize::new(0),
        }
    }

    /// Append one line per request to `path`, created if missing.
    pub fn with_log(mut self, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::options().create(true).append(true).open(path)?;
        self.log = Some(Mutex::new(file));
        Ok(self)
    }

    /// Requests the recording held no answer for.
    pub fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    fn answer(&self, request: &CompletionRequest) -> Result<Completion> {
        let Some((condition, chunked)) = parse_means_prompt(&request.system) else {
            return Err(SemcastError::Model(
                "replay answers only MEANS yes/no prompts".to_owned(),
            ));
        };
        let version = if chunked {
            MEANS_CHUNKED_PROMPT_VERSION
        } else {
            MEANS_PROMPT_VERSION
        };
        let key = means_cache_key(
            &normalize_instruction(condition),
            &request.input,
            &self.model,
            version,
        );
        let verdict = match self.answers.get(&key) {
            Some(CachedValue::Value(verdict)) if verdict == "yes" || verdict == "no" => {
                Some(verdict)
            }
            _ => None,
        };
        let chars = request.system.len() + request.input.len();
        self.record(chunked, chars, verdict.is_some());
        let Some(verdict) = verdict else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return Err(SemcastError::Model(format!(
                "replay: no recorded answer for this {} request",
                if chunked { "chunked" } else { "full-text" }
            )));
        };
        let yes_probability = if request.logprobs {
            cached_score(self.answers.as_ref(), &key).flatten()
        } else {
            None
        };
        Ok(Completion {
            text: verdict,
            input_tokens: chars.div_ceil(CHARS_PER_TOKEN),
            output_tokens: 1,
            yes_probability,
            merged: false,
        })
    }

    fn record(&self, chunked: bool, chars: usize, answered: bool) {
        let Some(log) = &self.log else { return };
        let line = format!("{{\"chunked\":{chunked},\"chars\":{chars},\"answered\":{answered}}}\n");
        let mut file = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Err(error) = file.write_all(line.as_bytes()) {
            tracing::warn!("replay log: {error}");
        }
    }
}

#[async_trait]
impl ModelProvider for ReplayProvider {
    fn id(&self) -> ModelId {
        self.model.clone()
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        requests
            .iter()
            .map(|request| self.answer(request))
            .collect()
    }

    fn embeds(&self) -> bool {
        false
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Embedding>> {
        Err(SemcastError::Model(
            "the replay provider doesn't embed; pair it with an embedder".to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use crate::physical::verify::{
        cache_verdict, synthesize_means_prompt, synthesize_means_prompt_chunked,
    };

    const MODEL: &str = "ollama/recorded";

    fn request(system: String, input: &str) -> CompletionRequest {
        CompletionRequest {
            system,
            input: input.to_owned(),
            max_tokens: 8,
            schema: None,
            logprobs: true,
        }
    }

    fn recording() -> InMemoryCache {
        let cache = InMemoryCache::default();
        let model = ModelId(MODEL.to_owned());
        let full = means_cache_key("is angry", "I am furious", &model, MEANS_PROMPT_VERSION);
        cache_verdict(&cache, full, true, Some(0.9));
        let chunked = means_cache_key("is angry", "furious", &model, MEANS_CHUNKED_PROMPT_VERSION);
        cache_verdict(&cache, chunked, false, None);
        cache
    }

    #[tokio::test]
    async fn recorded_answers_are_replayed_and_the_rest_counted_as_misses() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("replay.jsonl");
        let replay = ReplayProvider::new(Arc::new(recording()), MODEL)
            .with_log(&log)
            .unwrap();
        let answers = replay
            .complete(vec![
                // A condition with extra whitespace keys as the normalized one.
                request(synthesize_means_prompt(" is  angry"), "I am furious"),
                request(synthesize_means_prompt_chunked("is angry"), "furious"),
                request(synthesize_means_prompt("is angry"), "I am calm"),
                request("Extract the fields.".to_owned(), "I am furious"),
            ])
            .await;
        let full = answers[0].as_ref().unwrap();
        assert_eq!(
            (full.text.as_str(), full.yes_probability),
            ("yes", Some(0.9))
        );
        let chunked = answers[1].as_ref().unwrap();
        assert_eq!(
            (chunked.text.as_str(), chunked.yes_probability),
            ("no", None)
        );
        assert!(answers[2].is_err(), "never recorded");
        assert!(answers[3].is_err(), "not a MEANS prompt");
        assert_eq!(replay.misses(), 1);

        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 3, "one line per MEANS request");
        assert_eq!(lines[1]["chunked"], true);
        assert_eq!(lines[2]["answered"], false);
        let sent = synthesize_means_prompt(" is  angry").len() + "I am furious".len();
        assert_eq!(lines[0]["chars"], sent);
    }
}
