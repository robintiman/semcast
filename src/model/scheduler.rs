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
//! A call lives only as long as someone waits for it: the map holds it
//! weakly, so when every caller is gone (a cancelled query, an aborted job)
//! the call is dropped, its permit comes back, and its entry is cleared.
//!
//! Each `complete` batch keeps at most the cap in the queue at a time, so a
//! large batch takes turns with other queries instead of queueing all its
//! rows ahead of them.
//!
//! The scheduler changes when requests are sent, not what is answered, so
//! it reports its inner model's id and leaves cache keys alone. One
//! exception: a provider that panics fails that row (logged as an error)
//! instead of the query — rows fail, queries don't. And the inner provider
//! is handed one request per call, so a provider that packs a whole batch
//! into one request gets no batches behind the scheduler. A fallback
//! chain sits inside it and shares the one cap.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::{FutureExt, StreamExt};
use tokio::sync::Semaphore;

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider};
use crate::cache::stable_digest;
use crate::{Result, SemcastError};

/// The default cap on requests in flight per provider.
pub const DEFAULT_MAX_IN_FLIGHT: usize = 8;

/// An answer shared by every caller that asked the same thing. The error is
/// a string because it is cloned to each of them.
type SharedAnswer = Shared<BoxFuture<'static, std::result::Result<Completion, String>>>;

/// A call in flight, held weakly, with an id that tells it from a later
/// call for the same key.
type InFlight = HashMap<
    u128,
    (
        u64,
        WeakShared<BoxFuture<'static, std::result::Result<Completion, String>>>,
    ),
>;

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
    in_flight: Arc<Mutex<InFlight>>,
    next_call: AtomicU64,
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
            next_call: AtomicU64::new(0),
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
        let (answer, merged) = {
            let mut in_flight = self.in_flight.lock().expect("scheduler lock poisoned");
            match in_flight.get(&key).and_then(|(_, call)| call.upgrade()) {
                Some(answer) => {
                    self.merged.fetch_add(1, Ordering::Relaxed);
                    (answer, true)
                }
                None => {
                    let call = self.next_call.fetch_add(1, Ordering::Relaxed);
                    let answer = self.send(key, call, request);
                    let weak = answer.downgrade().expect("a fresh call has not finished");
                    in_flight.insert(key, (call, weak));
                    (answer, false)
                }
            }
        };
        answer
            .await
            .map(|completion| Completion {
                merged,
                ..completion
            })
            .map_err(|error| {
                if merged {
                    SemcastError::MergedModel(error)
                } else {
                    SemcastError::Model(error)
                }
            })
    }

    /// The call itself, as a future any number of callers can await. It
    /// clears its own entry however it ends — answered, panicked, or dropped
    /// with its last caller — so a later identical request is sent afresh.
    fn send(&self, key: u128, call: u64, request: CompletionRequest) -> SharedAnswer {
        let inner = Arc::clone(&self.inner);
        let permits = Arc::clone(&self.permits);
        let clear = ClearOnDrop {
            in_flight: Arc::clone(&self.in_flight),
            key,
            call,
        };
        let sent = Arc::clone(&self.sent);
        async move {
            let _clear = clear;
            let answer = {
                let _permit = permits
                    .acquire_owned()
                    .await
                    .expect("the scheduler never closes its semaphore");
                sent.fetch_add(1, Ordering::Relaxed);
                // A panic must end the call like any failure: `Shared` does
                // not wake its other waiters when the future it drives
                // panics, so a merged caller would hang forever.
                // Inside the guarded future, so a provider that panics before
                // returning its future is caught too.
                std::panic::AssertUnwindSafe(async { inner.complete(vec![request]).await })
                    .catch_unwind()
                    .await
            };
            match answer {
                Ok(answers) => match answers.into_iter().next() {
                    Some(answer) => answer.map_err(error_text),
                    None => Err(format!("{} returned no result", inner.id())),
                },
                Err(panic) => {
                    let message = format!("{} panicked: {}", inner.id(), panic_text(&*panic));
                    // A row error is easy to miss in a result set; a provider
                    // bug should not be.
                    tracing::error!(target: "semcast::model", "{message}");
                    Err(message)
                }
            }
        }
        .boxed()
        .shared()
    }
}

/// Removes a call's entry when the call ends, unless a later call for the
/// same key has taken its place.
struct ClearOnDrop {
    in_flight: Arc<Mutex<InFlight>>,
    key: u128,
    call: u64,
}

impl Drop for ClearOnDrop {
    fn drop(&mut self) {
        // Unwinding from a panic must not panic again on a poisoned lock.
        let mut in_flight = match self.in_flight.lock() {
            Ok(in_flight) => in_flight,
            Err(poisoned) => poisoned.into_inner(),
        };
        if in_flight
            .get(&self.key)
            .is_some_and(|(call, _)| *call == self.call)
        {
            in_flight.remove(&self.key);
        }
    }
}

/// What a panic said, when it said it with a string.
fn panic_text(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
}

/// An error's message, without the `model error:` prefix the caller will
/// add back when it wraps it.
fn error_text(error: SemcastError) -> String {
    match error {
        SemcastError::Model(message) | SemcastError::MergedModel(message) => message,
        other => other.to_string(),
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
        &[u8::from(request.logprobs)],
    ])
}

#[async_trait]
impl ModelProvider for Scheduler {
    fn id(&self) -> ModelId {
        self.inner.id()
    }

    /// One result per request, in order. Requests are sent one per call so
    /// each takes its own place in the queue, and at most the cap of them
    /// wait at once, so other batches get their turn.
    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        futures::stream::iter(
            requests
                .into_iter()
                .map(|request| self.complete_one(request)),
        )
        .buffered(self.max_in_flight)
        .collect()
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
            logprobs: false,
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
                        yes_probability: None,
                        merged: false,
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

    /// A different request must still get through after a cancelled one,
    /// whether the cancelled call was running or only queued.
    #[tokio::test]
    async fn a_cancelled_caller_gives_its_place_back() {
        let model = Arc::new(MockModel::default().with_latency(Duration::from_millis(200)));
        let scheduler = Arc::new(Scheduler::new(Arc::clone(&model) as _, 1));
        let spawn = |input: &'static str| {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move { scheduler.complete(vec![request(input)]).await })
        };
        let running = spawn("running");
        tokio::time::sleep(Duration::from_millis(20)).await;
        let queued = spawn("queued");
        tokio::time::sleep(Duration::from_millis(20)).await;
        running.abort();
        queued.abort();

        let other = tokio::time::timeout(
            Duration::from_secs(2),
            scheduler.complete(vec![request("other")]),
        )
        .await
        .expect("the permits came back");
        assert!(other[0].is_ok());
        assert!(
            scheduler.in_flight.lock().unwrap().is_empty(),
            "entries cleared"
        );
    }

    /// Panics on its first call for an input, then answers.
    #[derive(Debug, Default)]
    struct PanicsOnce {
        panicked: std::sync::atomic::AtomicBool,
    }

    #[async_trait]
    impl ModelProvider for PanicsOnce {
        fn id(&self) -> ModelId {
            ModelId("panics-once".to_owned())
        }

        async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
            if !self.panicked.swap(true, Ordering::SeqCst) {
                panic!("provider bug");
            }
            requests
                .into_iter()
                .map(|_| {
                    Ok(Completion {
                        text: "yes".to_owned(),
                        input_tokens: 0,
                        output_tokens: 0,
                        yes_probability: None,
                        merged: false,
                    })
                })
                .collect()
        }

        async fn embed(&self, _: Vec<String>) -> Result<Vec<Embedding>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn a_panicking_call_fails_its_row_and_does_not_poison_its_request() {
        let scheduler = Scheduler::new(Arc::new(PanicsOnce::default()), 1);
        let first = scheduler.complete(vec![request("x")]).await;
        let error = first[0].as_ref().unwrap_err().to_string();
        assert!(error.contains("panicked: provider bug"), "{error}");
        let again = scheduler.complete(vec![request("x")]).await;
        assert_eq!(again[0].as_ref().unwrap().text, "yes", "asked afresh");
    }

    /// Sleeps, then panics.
    #[derive(Debug)]
    struct SlowPanic;

    #[async_trait]
    impl ModelProvider for SlowPanic {
        fn id(&self) -> ModelId {
            ModelId("slow-panic".to_owned())
        }

        async fn complete(&self, _: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            panic!("provider bug");
        }

        async fn embed(&self, _: Vec<String>) -> Result<Vec<Embedding>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn every_caller_merged_onto_a_panicking_call_gets_an_error() {
        let scheduler = Arc::new(Scheduler::new(Arc::new(SlowPanic), 4));
        let callers = (0..4).map(|_| {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move { scheduler.complete(vec![request("x")]).await })
        });
        let answers =
            tokio::time::timeout(Duration::from_secs(5), futures::future::join_all(callers))
                .await
                .expect("no merged caller hangs");
        for answer in answers {
            assert!(answer.expect("no caller panics")[0].is_err());
        }
        assert_eq!(
            scheduler.stats(),
            SchedulerStats { sent: 1, merged: 3 },
            "the callers shared one call"
        );
    }

    #[tokio::test]
    async fn a_model_error_is_not_prefixed_twice() {
        let scheduler = Scheduler::new(Arc::new(MockModel::default().failing_on(["bad"])), 1);
        let error = scheduler
            .complete(vec![request("bad")])
            .await
            .remove(0)
            .unwrap_err();
        let text = error.to_string();
        assert_eq!(text.matches("model error").count(), 1, "{text}");
    }

    #[tokio::test]
    async fn a_large_batch_takes_turns_with_other_queries() {
        let model = Arc::new(MockModel::default().with_latency(Duration::from_millis(20)));
        let scheduler = Arc::new(Scheduler::new(Arc::clone(&model) as _, 2));
        let big = {
            let scheduler = Arc::clone(&scheduler);
            tokio::spawn(async move {
                let requests = (0..100).map(|i| request(&format!("row {i}"))).collect();
                scheduler.complete(requests).await
            })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        let started = std::time::Instant::now();
        scheduler.complete(vec![request("interactive")]).await;
        // Queued behind the whole batch it would wait ~50 × 20ms = 1s.
        assert!(
            started.elapsed() < Duration::from_millis(300),
            "waited {:?}",
            started.elapsed()
        );
        big.await.unwrap();
    }

    #[tokio::test]
    async fn the_id_is_the_inner_models() {
        let scheduler = Scheduler::new(Arc::new(MockModel::default()), 1);
        assert_eq!(scheduler.id(), ModelId("mock".to_owned()));
    }
}
