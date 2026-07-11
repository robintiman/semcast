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
pub mod server;
pub mod sql;
pub mod telemetry;
pub mod types;

pub use error::{Result, SemcastError};

use std::sync::Arc;

use datafusion::dataframe::DataFrame;
use datafusion::error::DataFusionError;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlanBuilder;

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
            sql::ddl::SemanticDdl::CreateType(ty) => {
                let runtime = ctx
                    .state()
                    .config()
                    .get_extension::<SemcastRuntime>()
                    .ok_or_else(|| {
                        DataFusionError::Plan("semcast runtime is not registered".to_owned())
                    })?;
                runtime.type_registry().register(ty)?;
                let no_rows = LogicalPlanBuilder::empty(false).build()?;
                Ok(DataFrame::new(ctx.state(), no_rows))
            }
            other => Err(DataFusionError::Plan(format!(
                "semantic DDL not implemented yet: {other:?}"
            ))
            .into()),
        };
    }
    let (mut statement, recall) = sql::recall::parse_statement_with_recall(query)?;
    // Desugar `CAST(col AS SemanticType)[.field]` into the marker UDFs before
    // planning — DataFusion can't plan the field access itself.
    if let Some(runtime) = ctx.state().config().get_extension::<SemcastRuntime>() {
        sql::typed::rewrite_semantic_casts(&mut statement, runtime.type_registry())?;
    }
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
///
/// Contexts can query files by path (`SELECT * FROM 'data.parquet'`), so any
/// SQL client can read any local file the process can.
pub struct SemcastContextBuilder {
    model: Arc<dyn ModelProvider>,
    embedder: Option<Arc<dyn ModelProvider>>,
    cache: Arc<dyn SemanticCache>,
    index_root: Option<std::path::PathBuf>,
    information_schema: bool,
}

impl SemcastContextBuilder {
    pub fn new(model: Arc<dyn ModelProvider>) -> Self {
        Self {
            model,
            embedder: None,
            cache: Arc::new(InMemoryCache::default()),
            index_root: None,
            information_schema: false,
        }
    }

    /// Embed semantic indexes (and their queries) through `embedder` instead
    /// of the session model — for a dedicated embedding provider (Voyage), or
    /// when the session model can't embed (Anthropic).
    pub fn with_embedder(mut self, embedder: Arc<dyn ModelProvider>) -> Self {
        self.embedder = Some(embedder);
        self
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
        // One type registry, shared between the runtime (DDL dispatch) and the
        // marker UDFs (which resolve type fields at plan time).
        let types = Arc::new(crate::types::registry::TypeRegistry::default());
        let mut runtime =
            SemcastRuntime::new(Arc::clone(&self.model)).with_type_registry(Arc::clone(&types));
        if let Some(embedder) = self.embedder {
            runtime = runtime.with_embedder(embedder);
        }
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
            .with_optimizer_rule(Arc::new(
                crate::optimizer::extract::ExtractRewriteRule::new(Arc::clone(&types)),
            ))
            .with_query_planner(Arc::new(SemcastQueryPlanner::new(self.model, self.cache)))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_udf(sql::means_udf::means_udf());
        // Marker UDFs for typed extraction, sharing the runtime's registry so
        // they resolve type fields at plan time.
        for udf in sql::extract_udf::extract_udfs(types) {
            ctx.register_udf(udf);
        }
        // Must stay last: it consumes and rebuilds the context (everything
        // registered above carries over via `new_from_existing`).
        ctx.enable_url_table()
    }
}
