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
#[cfg(feature = "server")]
pub mod server;
pub mod sql;
pub mod types;

pub use error::{Result, SemcastError};

use std::sync::Arc;

use datafusion::dataframe::DataFrame;
use datafusion::error::DataFusionError;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::sql::parser::Statement as DFStatement;

use crate::cache::{InMemoryCache, SemanticCache};
use crate::index::registry::SemcastRuntime;
use crate::model::ModelProvider;
use crate::optimizer::rewrite::MeansRewriteRule;
use crate::physical::planner::SemcastQueryPlanner;

pub use crate::index::{IndexOptions, create_semantic_index, refresh_semantic_index};

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

/// Run a query written in semcast SQL — standard SQL plus the infix `MEANS`
/// operator and a trailing `WITH RECALL <target>` — against a context from
/// [`semcast_context`].
///
/// This is `ctx.sql()` with [`sql::SemcastDialect`] in front. `ctx.sql()`
/// only accepts sqlparser's built-in dialects, so the custom syntax needs its
/// own entry point; queries that call `means()` directly work through either.
pub async fn sql(ctx: &SessionContext, query: &str) -> Result<DataFrame> {
    // `CREATE SEMANTIC ...` is semcast syntax, not SQL — intercept it before
    // the parser. DDL yields an empty DataFrame, like DataFusion's own DDL.
    if let Some(ddl) = sql::ddl::parse_semantic_ddl(query)? {
        return match ddl {
            sql::ddl::SemanticDdl::CreateIndex { table, column } => {
                create_semantic_index(ctx, &table, &column, IndexOptions::default()).await?;
                let no_rows = LogicalPlanBuilder::empty(false).build()?;
                Ok(DataFrame::new(ctx.state(), no_rows))
            }
            other => Err(DataFusionError::Plan(format!(
                "semantic DDL not implemented yet: {other:?}"
            ))
            .into()),
        };
    }
    let (statement, recall) = sql::recall::parse_statement_with_recall(query)?;
    let statement = DFStatement::Statement(Box::new(statement));
    let mut plan = ctx.state().statement_to_plan(statement).await?;
    if let Some(recall) = recall {
        plan = optimizer::rewrite::apply_recall(plan, recall)?;
    }
    Ok(ctx.execute_logical_plan(plan).await?)
}

/// [`semcast_context`] with a caller-provided verdict cache.
pub fn semcast_context_with_cache(
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
) -> SessionContext {
    SemcastContextBuilder::new(model).with_cache(cache).build()
}

/// [`semcast_context`] with every knob exposed — the server binary needs an
/// index root outside the temp dir and `information_schema` for `SHOW`.
pub struct SemcastContextBuilder {
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    index_root: Option<std::path::PathBuf>,
    information_schema: bool,
}

impl SemcastContextBuilder {
    pub fn new(model: Arc<dyn ModelProvider>) -> Self {
        Self {
            model,
            cache: Arc::new(InMemoryCache::default()),
            index_root: None,
            information_schema: false,
        }
    }

    pub fn with_cache(mut self, cache: Arc<dyn SemanticCache>) -> Self {
        self.cache = cache;
        self
    }

    /// Where `CREATE SEMANTIC INDEX` puts Lance datasets; defaults to the temp dir.
    pub fn with_index_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.index_root = Some(root.into());
        self
    }

    pub fn with_information_schema(mut self, on: bool) -> Self {
        self.information_schema = on;
        self
    }

    pub fn build(self) -> SessionContext {
        let mut runtime = SemcastRuntime::new(Arc::clone(&self.model));
        if let Some(root) = self.index_root {
            runtime = runtime.with_index_root(root);
        }
        let config = SessionConfig::new()
            .with_extension(Arc::new(runtime))
            .with_information_schema(self.information_schema);
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config)
            .with_optimizer_rule(Arc::new(MeansRewriteRule))
            .with_query_planner(Arc::new(SemcastQueryPlanner::new(self.model, self.cache)))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_udf(sql::means_udf::means_udf());
        ctx
    }
}
