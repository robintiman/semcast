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

use datafusion::dataframe::DataFrame;
use datafusion::error::DataFusionError;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlanBuilder;
use datafusion::sql::parser::Statement as DFStatement;
use datafusion::sql::sqlparser::parser::Parser;

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
/// operator — against a context from [`semcast_context`].
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
    let mut statements =
        Parser::parse_sql(&sql::SemcastDialect::default(), query).map_err(DataFusionError::from)?;
    if statements.len() != 1 {
        return Err(DataFusionError::Plan(format!(
            "expected exactly one statement, got {}",
            statements.len()
        ))
        .into());
    }
    let statement = DFStatement::Statement(Box::new(statements.pop().expect("checked len")));
    let plan = ctx.state().statement_to_plan(statement).await?;
    Ok(ctx.execute_logical_plan(plan).await?)
}

/// [`semcast_context`] with a caller-provided verdict cache.
pub fn semcast_context_with_cache(
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
) -> SessionContext {
    let runtime = Arc::new(SemcastRuntime::new(Arc::clone(&model)));
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_extension(runtime))
        .with_optimizer_rule(Arc::new(MeansRewriteRule))
        .with_query_planner(Arc::new(SemcastQueryPlanner::new(model, cache)))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(sql::means_udf::means_udf());
    ctx
}
