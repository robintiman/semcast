//! One queue in front of a model: a shared cap on requests in flight, and
//! identical requests merged into one call.
//!
//! Each operator call used to pace itself (a fixed number in flight per
//! `complete` batch). Partitions, operators and concurrent queries each have
//! their own batches, so the load a provider saw grew with all three — the
//! surest way to hit a rate limit, or to queue requests inside Ollama where
//! nothing can see them. [`Scheduler`] holds one semaphore per provider, so
//! the cap holds across everything the session runs.
//!
//! It also merges duplicates. The verdict cache is filled when an answer
//! arrives, so two partitions asking about the same text at the same moment
//! both miss it and both pay. A request identical to one already in flight
//! waits for that one's answer instead. Same request, same answer — which is
//! what "first evaluation wins" asks for anyway.
//!
//! The scheduler changes when requests are sent, never what is answered, so
//! it reports its inner model's id and leaves cache keys alone.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use tokio::sync::Semaphore;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::cache::stable_digest;
use crate::{Result, SemcastError};

/// The default cap on requests in flight per provider.
pub const DEFAULT_MAX_IN_FLIGHT: usize = 8;

/// An answer shared by every caller that asked the same thing. The error is
/// a string because it is cloned to each of them.
type SharedAnswer = Shared<BoxFuture<'static, std::result::Result<Completion, String>>>;

/// Counters for logs and tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerStats {
    /// Requests sent to the model.
    pub sent: usize,
    /// Requests answered by a call already in flight.
    pub merged: usize,
}

/// A [`ModelProvider`] that paces and merges requests to `inner`.
pub struct Scheduler {
    inner: Arc<dyn ModelProvider>,
    permits: Arc<Semaphore>,
    max_in_flight: usize,
    in_flight: Arc<Mutex<HashMap<u128, SharedAnswer>>>,
    sent: Arc<AtomicUsize>,
    merged: AtomicUsize,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("inner", &self.inner)
            .field("max_in_flight", &self.max_in_flight)
            .finish_non_exhaustive()
    }
}

impl Scheduler {
    /// At most `max_in_flight` requests to `inner` at once (at least one).
    pub fn new(inner: Arc<dyn ModelProvider>, max_in_flight: usize) -> Self {
        let max_in_flight = max_in_flight.max(1);
        Self {
            inner,
            permits: Arc::new(Semaphore::new(max_in_flight)),
            max_in_flight,
            in_flight: Arc::new(Mutex::new(HashMap::new())),
            sent: Arc::new(AtomicUsize::new(0)),
            merged: AtomicUsize::new(0),
        }
    }

    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            sent: self.sent.load(Ordering::Relaxed),
            merged: self.merged.load(Ordering::Relaxed),
        }
    }

    async fn complete_one(&self, request: CompletionRequest) -> Result<Completion> {
        let key = request_key(&request);
        let answer = {
            let mut in_flight = self.in_flight.lock().expect("scheduler lock poisoned");
            match in_flight.get(&key) {
                Some(answer) => {
                    self.merged.fetch_add(1, Ordering::Relaxed);
                    answer.clone()
                }
                None => {
                    let answer = self.send(key, request);
                    in_flight.insert(key, answer.clone());
                    answer
                }
            }
        };
        answer.await.map_err(SemcastError::Model)
    }

    /// The call itself, as a future any number of callers can await. It
    /// removes its own entry when done, so a later identical request — one
    /// the cache will answer, or a deliberate retry — is sent afresh.
    fn send(&self, key: u128, request: CompletionRequest) -> SharedAnswer {
        let inner = Arc::clone(&self.inner);
        let permits = Arc::clone(&self.permits);
        let in_flight = Arc::clone(&self.in_flight);
        let sent = Arc::clone(&self.sent);
        async move {
            let answer = {
                let _permit = permits
                    .acquire_owned()
                    .await
                    .expect("the scheduler never closes its semaphore");
                sent.fetch_add(1, Ordering::Relaxed);
                inner.complete(vec![request]).await.into_iter().next()
            };
            in_flight
                .lock()
                .expect("scheduler lock poisoned")
                .remove(&key);
            match answer {
                Some(answer) => answer.map_err(|error| error.to_string()),
                None => Err(format!("{} returned no result", inner.id())),
            }
        }
        .boxed()
        .shared()
    }
}

/// Every field that can change the answer.
fn request_key(request: &CompletionRequest) -> u128 {
    let schema = request
        .schema
        .as_ref()
        .map(|schema| schema.to_string())
        .unwrap_or_default();
    stable_digest(&[
        request.system.as_bytes(),
        request.input.as_bytes(),
        &request.max_tokens.to_le_bytes(),
        schema.as_bytes(),
    ])
}

#[async_trait]
impl ModelProvider for Scheduler {
    fn id(&self) -> ModelId {
        self.inner.id()
    }

    /// One result per request, in order. Requests are sent one per call so
    /// each takes its own place in the queue.
    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        futures::future::join_all(
            requests
                .into_iter()
                .map(|request| self.complete_one(request)),
        )
        .await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        let _permit = self
            .permits
            .acquire()
            .await
            .expect("the scheduler never closes its semaphore");
        self.inner.embed(texts).await
    }

    fn embed_batch_size(&self) -> usize {
        self.inner.embed_batch_size()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::model::MockModel;

    fn request(input: &str) -> CompletionRequest {
        CompletionRequest {
            system: "Answer yes or no.".to_owned(),
            input: input.to_owned(),
            max_tokens: 4,
            schema: None,
        }
    }

    /// Counts how many calls are inside it at once.
    #[derive(Debug, Default)]
    struct Gauge {
        now: AtomicUsize,
        peak: AtomicUsize,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ModelProvider for Gauge {
        fn id(&self) -> ModelId {
            ModelId("gauge".to_owned())
        }

        async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
            let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            self.calls.fetch_add(requests.len(), Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            self.now.fetch_sub(1, Ordering::SeqCst);
            requests
                .into_iter()
                .map(|request| {
                    Ok(Completion {
                        text: request.input,
                        input_tokens: 0,
                        output_tokens: 0,
                    })
                })
                .collect()
        }

        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
            Ok(texts.iter().map(|_| vec![1.0]).collect())
        }
    }

    #[tokio::test]
    async fn the_cap_holds_across_concurrent_batches() {
        let gauge = Arc::new(Gauge::default());
        let scheduler = Arc::new(Scheduler::new(Arc::clone(&gauge) as _, 3));
        let batches = (0..4).map(|batch| {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move {
                let requests = (0..5).map(|i| request(&format!("{batch}-{i}"))).collect();
                scheduler.complete(requests).await
            })
        });
        for batch in futures::future::join_all(batches).await {
            assert!(batch.unwrap().iter().all(Result::is_ok));
        }
        assert_eq!(gauge.calls.load(Ordering::SeqCst), 20);
        assert_eq!(
            gauge.peak.load(Ordering::SeqCst),
            3,
            "never more than the cap"
        );
    }

    #[tokio::test]
    async fn identical_requests_in_flight_share_one_call() {
        let model = Arc::new(
            MockModel::answering_yes_to(["angry"]).with_latency(Duration::from_millis(50)),
        );
        let scheduler = Arc::new(Scheduler::new(Arc::clone(&model) as _, 8));
        let callers = (0..4).map(|_| {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move { scheduler.complete(vec![request("angry customer")]).await })
        });
        for answers in futures::future::join_all(callers).await {
            assert_eq!(answers.unwrap()[0].as_ref().unwrap().text, "yes");
        }
        assert_eq!(model.completion_calls(), 1);
        assert_eq!(scheduler.stats(), SchedulerStats { sent: 1, merged: 3 });
    }

    #[tokio::test]
    async fn a_finished_request_is_sent_again_when_asked_again() {
        let model = Arc::new(MockModel::default());
        let scheduler = Scheduler::new(Arc::clone(&model) as _, 8);
        scheduler.complete(vec![request("x")]).await;
        scheduler.complete(vec![request("x")]).await;
        assert_eq!(
            model.completion_calls(),
            2,
            "merging is for calls in flight only"
        );
    }

    #[tokio::test]
    async fn results_keep_request_order_and_errors_stay_per_row() {
        let model = Arc::new(MockModel::answering_yes_to(["a"]).failing_on(["bad"]));
        let scheduler = Scheduler::new(Arc::clone(&model) as _, 2);
        let answers = scheduler
            .complete(vec![request("a"), request("bad"), request("b")])
            .await;
        assert_eq!(answers[0].as_ref().unwrap().text, "yes");
        assert!(answers[1].is_err());
        assert_eq!(answers[2].as_ref().unwrap().text, "no");
    }

    #[tokio::test]
    async fn the_id_is_the_inner_models() {
        let scheduler = Scheduler::new(Arc::new(MockModel::default()), 1);
        assert_eq!(scheduler.id(), ModelId("mock".to_owned()));
    }
}
