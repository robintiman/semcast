//! Planner-integrated semantic operators for [Apache DataFusion].
//!
//! semcast puts LLM calls inside the query planner as first-class operators —
//! `text MEANS 'condition'` becomes a very expensive predicate that DataFusion
//! can prune, reorder, and cache like any other. See `README.md` for the full
//! design; the roadmap ("Status" section) drives what is implemented here.
//!
//! Module map (mirrors the README architecture table):
//!
//! | Module      | DataFusion hook                                        |
//! |-------------|--------------------------------------------------------|
//! | [`sql`]     | `means()` UDF entry point, `CREATE SEMANTIC ...` DDL   |
//! | [`logical`] | `UserDefinedLogicalNodeCore` extension nodes           |
//! | [`optimizer`] | `OptimizerRule`s: rewrite, funnel, calibration       |
//! | [`physical`] | `QueryPlanner` / `ExtensionPlanner` / `ExecutionPlan` |
//! | [`index`]   | semantic index (chunk, embed, incremental maintenance) |
//! | [`model`]   | async batched model providers                          |
//! | [`cache`]   | field-level cache with provenance keys                 |
//! | [`types`]   | semantic type registry (`CREATE SEMANTIC TYPE`)        |
//!
//! [Apache DataFusion]: https://datafusion.apache.org/

pub mod cache;
pub mod error;
pub mod index;
pub mod logical;
pub mod model;
pub mod optimizer;
pub mod physical;
pub mod sql;
pub mod types;

pub use error::{Result, SemcastError};

use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use datafusion::execution::session_state::SessionStateBuilder;

use crate::cache::{InMemoryCache, SemanticCache};
use crate::model::ModelProvider;
use crate::optimizer::rewrite::MeansRewriteRule;
use crate::physical::planner::SemcastQueryPlanner;

/// Build a [`SessionContext`] with all semcast machinery registered:
/// the `means()` UDF, the `MEANS`-rewrite optimizer rule, and a query
/// planner that knows how to execute semcast's logical extension nodes.
///
/// Verdicts are cached in memory for the lifetime of the context, so
/// re-running a query — or asking a new question that shares a `means()`
/// predicate — costs zero new model calls for the rows already seen. Bring
/// a persistent [`SemanticCache`] with [`semcast_context_with_cache`] to
/// compound across sessions.
///
/// ```
/// use std::sync::Arc;
/// use semcast::{model::MockModel, semcast_context};
///
/// let ctx = semcast_context(Arc::new(MockModel::default()));
/// ```
pub fn semcast_context(model: Arc<dyn ModelProvider>) -> SessionContext {
    semcast_context_with_cache(model, Arc::new(InMemoryCache::default()))
}

/// [`semcast_context`] with a caller-provided verdict cache.
pub fn semcast_context_with_cache(
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_optimizer_rule(Arc::new(MeansRewriteRule))
        .with_query_planner(Arc::new(SemcastQueryPlanner::new(model, cache)))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(sql::means_udf::means_udf());
    ctx
}
