//! Logging setup. One idempotent initializer that every entry point (the
//! server binary, the examples, tests) can call. Output is controlled by
//! `RUST_LOG`; when it is unset the default filter shows all three semcast
//! logging categories — the optimised plan and stage boundaries at info, and
//! full model prompts/responses at debug:
//!
//! - `semcast::plan`  — the optimised physical plan, before execution (info)
//! - `semcast::stage` — begin/end of each execution stage (info)
//! - `semcast::model` — the prompt sent to, and response from, the LLM (debug)
//!
//! `RUST_LOG=semcast=info` hides the verbose prompt/response bodies;
//! `RUST_LOG=semcast=debug` (the default) shows everything. The
//! `tracing-log` bridge also surfaces DataFusion/Lance internal `log` records.

/// Install the global tracing subscriber. Safe to call more than once — the
/// first call wins and later calls are ignored, so examples and tests can call
/// it freely without panicking.
pub fn init() {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,semcast=debug"));
    let _ = fmt().with_env_filter(filter).with_target(true).try_init();
}
