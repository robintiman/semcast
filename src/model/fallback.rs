//! An ordered chain of providers: the next one answers what the previous
//! could not.
//!
//! "Rows fail, queries don't" is the floor, not the ceiling. A row whose model
//! call fails is excluded from the result and counted — honest, but a row lost
//! to one provider's outage is a row a second provider would have answered.
//! [`FallbackProvider`] retries exactly the failed requests against the next
//! model in the chain, so an outage costs latency instead of rows.
//!
//! Scope: the chain reacts to `Err`, which the HTTP providers only return
//! after their own retry/backoff is exhausted — a genuine failure, not a
//! blip. It cannot react to a *parseable-but-wrong* answer, because deciding
//! that is the caller's job (`parse_verdict` lives in the verify stage) and
//! this layer only sees `Result<Completion>`.

use std::sync::Arc;

use async_trait::async_trait;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::{Result, SemcastError};

/// A primary model plus the models that cover for it, tried in order.
#[derive(Debug)]
pub struct FallbackProvider {
    /// Non-empty by construction: `models[0]` is the primary.
    models: Vec<Arc<dyn ModelProvider>>,
    id: ModelId,
}

impl FallbackProvider {
    /// `primary` answers first; each of `fallbacks` in turn retries whatever
    /// the previous model failed on.
    ///
    /// With no fallbacks the composite id degenerates to the primary's own id,
    /// so wrapping a lone model changes no cache key.
    pub fn new(primary: Arc<dyn ModelProvider>, fallbacks: Vec<Arc<dyn ModelProvider>>) -> Self {
        let mut models = Vec::with_capacity(fallbacks.len() + 1);
        models.push(primary);
        models.extend(fallbacks);
        let id = ModelId(
            models
                .iter()
                .map(|m| m.id().0)
                .collect::<Vec<_>>()
                .join("+"),
        );
        Self { models, id }
    }
}

#[async_trait]
impl ModelProvider for FallbackProvider {
    /// Names the whole chain, because the chain is what produced the answer.
    ///
    /// [`CacheKey`] carries a model id as provenance, so attributing a
    /// fallback's verdict to the primary would make the cache lie about who
    /// said it. Naming every link means adding a fallback invalidates cached
    /// verdicts — the same honesty the prompt-version constants buy.
    ///
    /// [`CacheKey`]: crate::cache::CacheKey
    fn id(&self) -> ModelId {
        self.id.clone()
    }

    /// One result per request, in request order — a failed row moves down the
    /// chain, a successful one is never asked twice.
    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        let mut slots: Vec<Option<Result<Completion>>> =
            (0..requests.len()).map(|_| None).collect();
        // Indices into `requests` still looking for an answer.
        let mut pending: Vec<usize> = (0..requests.len()).collect();

        for (rank, model) in self.models.iter().enumerate() {
            if pending.is_empty() {
                break;
            }
            let last = rank + 1 == self.models.len();
            let batch: Vec<CompletionRequest> =
                pending.iter().map(|&i| requests[i].clone()).collect();
            let sent = batch.len();
            let mut outcomes = model.complete(batch).await;
            // The trait promises one result per request; a provider that
            // breaks that promise must not silently drop rows.
            if outcomes.len() != sent {
                outcomes.resize_with(sent, || {
                    Err(SemcastError::Model(format!(
                        "{} returned the wrong number of results",
                        model.id()
                    )))
                });
            }

            let mut still_pending = Vec::new();
            for (&slot, outcome) in pending.iter().zip(outcomes) {
                match outcome {
                    Ok(completion) => slots[slot] = Some(Ok(completion)),
                    // The last link has nobody to defer to: its error is the
                    // row's error, and the verify stage drops the row.
                    Err(error) if last => slots[slot] = Some(Err(error)),
                    Err(error) => {
                        tracing::warn!(
                            target: "semcast::model",
                            failed = %model.id(),
                            next = %self.models[rank + 1].id(),
                            error = %error,
                            "falling back to the next model",
                        );
                        still_pending.push(slot);
                    }
                }
            }
            pending = still_pending;
        }

        slots
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    Err(SemcastError::Model("no model answered the request".into()))
                })
            })
            .collect()
    }

    /// All-or-nothing, unlike `complete`: [`ModelProvider::embed`] returns one
    /// `Result` for the whole batch, so there are no individual failures to
    /// route and the next model re-embeds everything.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        let mut last_error = None;
        for (rank, model) in self.models.iter().enumerate() {
            match model.embed(texts.clone()).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(error) => {
                    if rank + 1 < self.models.len() {
                        tracing::warn!(
                            target: "semcast::model",
                            failed = %model.id(),
                            next = %self.models[rank + 1].id(),
                            error = %error,
                            "falling back to the next embedder",
                        );
                    }
                    last_error = Some(error);
                }
            }
        }
        Err(last_error
            .unwrap_or_else(|| SemcastError::Model("no embedder answered the batch".into())))
    }

    /// The primary's preference. A fallback batches to whatever the model it
    /// covers for asked for, which keeps one embed call one embed call.
    fn embed_batch_size(&self) -> usize {
        self.models[0].embed_batch_size()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Fails every request whose input contains `poison`, answers the rest
    /// with its own name so tests can tell who spoke.
    #[derive(Debug)]
    struct Flaky {
        name: &'static str,
        poison: &'static str,
        seen: Mutex<Vec<String>>,
    }

    impl Flaky {
        fn new(name: &'static str, poison: &'static str) -> Arc<Self> {
            Arc::new(Self {
                name,
                poison,
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl ModelProvider for Flaky {
        fn id(&self) -> ModelId {
            ModelId(self.name.to_owned())
        }

        async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
            let mut seen = self.seen.lock().expect("seen poisoned");
            requests
                .into_iter()
                .map(|req| {
                    seen.push(req.input.clone());
                    if req.input.contains(self.poison) {
                        Err(SemcastError::Model(format!("{} refused", self.name)))
                    } else {
                        Ok(Completion {
                            text: self.name.to_owned(),
                            input_tokens: 0,
                            output_tokens: 0,
                        })
                    }
                })
                .collect()
        }

        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
            if texts.iter().any(|t| t.contains(self.poison)) {
                return Err(SemcastError::Model(format!("{} refused", self.name)));
            }
            Ok(texts.iter().map(|_| vec![1.0; 4]).collect())
        }
    }

    fn ask(inputs: &[&str]) -> Vec<CompletionRequest> {
        inputs
            .iter()
            .map(|input| CompletionRequest {
                system: "s".into(),
                input: (*input).to_owned(),
                max_tokens: 8,
                schema: None,
            })
            .collect()
    }

    fn texts(completions: &[Result<Completion>]) -> Vec<String> {
        completions
            .iter()
            .map(|c| match c {
                Ok(completion) => completion.text.clone(),
                Err(error) => format!("ERR({error})"),
            })
            .collect()
    }

    #[tokio::test]
    async fn the_secondary_answers_only_what_the_primary_failed() {
        let primary = Flaky::new("primary", "bad");
        let secondary = Flaky::new("secondary", "never");
        let chain = FallbackProvider::new(primary.clone(), vec![secondary.clone()]);

        let answers = chain.complete(ask(&["fine", "bad", "also fine"])).await;

        // Order preserved, and only the failed row moved down the chain.
        assert_eq!(texts(&answers), ["primary", "secondary", "primary"]);
        assert_eq!(
            *secondary.seen.lock().unwrap(),
            ["bad"],
            "the secondary must not re-answer rows the primary handled"
        );
    }

    #[tokio::test]
    async fn a_row_every_model_fails_still_returns_one_error() {
        let chain = FallbackProvider::new(
            Flaky::new("primary", "bad"),
            vec![Flaky::new("secondary", "bad")],
        );

        let answers = chain.complete(ask(&["fine", "bad"])).await;

        assert_eq!(answers.len(), 2, "one result per request, always");
        assert!(answers[0].is_ok());
        // The last link's error surfaces, so the verify stage drops that row
        // rather than failing the query.
        assert!(matches!(answers[1], Err(SemcastError::Model(_))));
    }

    #[tokio::test]
    async fn an_empty_batch_asks_nobody() {
        let primary = Flaky::new("primary", "bad");
        let chain = FallbackProvider::new(primary.clone(), vec![Flaky::new("secondary", "x")]);

        assert!(chain.complete(Vec::new()).await.is_empty());
        assert!(primary.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn embedding_falls_through_as_a_whole_batch() {
        let chain = FallbackProvider::new(
            Flaky::new("primary", "bad"),
            vec![Flaky::new("secondary", "never")],
        );

        let embeddings = chain
            .embed(vec!["fine".into(), "bad".into()])
            .await
            .expect("the secondary embeds the batch");
        assert_eq!(embeddings.len(), 2);
    }

    #[test]
    fn a_lone_model_keeps_its_own_id() {
        // The cache keys on this id, so wrapping without fallbacks must not
        // invalidate a single cached verdict.
        let primary = Flaky::new("primary", "x");
        let chain = FallbackProvider::new(primary.clone(), Vec::new());
        assert_eq!(chain.id(), primary.id());
    }

    #[test]
    fn a_chain_names_every_link() {
        let chain = FallbackProvider::new(
            Flaky::new("primary", "x"),
            vec![Flaky::new("secondary", "x"), Flaky::new("tertiary", "x")],
        );
        assert_eq!(chain.id(), ModelId("primary+secondary+tertiary".into()));
    }
}
