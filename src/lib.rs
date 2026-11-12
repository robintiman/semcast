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
pub mod catalog;
pub mod error;
pub mod index;
pub mod logical;
pub mod model;
pub mod optimizer;
pub mod physical;
pub mod semantic_column;
pub mod semir;
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
use crate::optimizer::cluster::ClusterRewriteRule;
use crate::optimizer::distinct::DistinctRewriteRule;
use crate::optimizer::rank::RelevanceRewriteRule;
use crate::optimizer::rewrite::MeansRewriteRule;
use crate::physical::planner::SemcastQueryPlanner;

pub use crate::catalog::restore as restore_catalog;
pub use crate::index::{
    IndexOptions, create_semantic_index, reattach_semantic_index, refresh_semantic_index,
};

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
    run_sql(ctx, query, true).await
}

/// [`sql`], saving what a DDL statement defines to the session's durable
/// catalog when `record` is set. Restoring the catalog replays with it
/// unset: the definitions are already saved.
pub(crate) async fn run_sql(ctx: &SessionContext, text: &str, record: bool) -> Result<DataFrame> {
    // `SEMANTIC` qualifies a select modifier, where sqlparser has no dialect
    // hook — remove the word, then mark what it qualified in the AST. Safe to
    // run ahead of the DDL branch: it only fires on `SEMANTIC DISTINCT`, and
    // `CREATE SEMANTIC INDEX`/`TYPE` never is.
    let (query, semantic_distinct) = sql::distinct::strip_semantic_distinct(text);
    // `CREATE SEMANTIC ...` is semcast syntax, not SQL. The statement parser
    // dispatches it on its own token stream; DDL is executed here rather than
    // planned, and yields an empty DataFrame like DataFusion's own DDL.
    let (mut statement, clauses) = match sql::parse_semcast_statement(&query)? {
        sql::SemcastStatement::Ddl(ddl) => {
            return match ddl {
                sql::ddl::SemanticDdl::CreateIndex { table, column } => {
                    create_semantic_index(ctx, &table, &column, IndexOptions::default()).await?;
                    let no_rows = LogicalPlanBuilder::empty(false).build()?;
                    Ok(DataFrame::new(ctx.state(), no_rows))
                }
                sql::ddl::SemanticDdl::CreateType(ty) => {
                    let runtime = index::runtime(ctx)?;
                    let catalog = runtime.catalog().filter(|_| record);
                    let _guard = match catalog {
                        Some(catalog) => Some(catalog.lock().await),
                        None => None,
                    };
                    let name = ty.name.to_lowercase();
                    runtime.type_registry().register(ty)?;
                    if let Some(catalog) = catalog {
                        let entry = catalog::Entry {
                            kind: catalog::EntryKind::SemanticType,
                            name: name.clone(),
                            definition: text.to_owned(),
                        };
                        let failure = format!(
                            "semantic type {name} is defined for this session but will be \
                             gone after a restart"
                        );
                        catalog::save(catalog, vec![catalog::Change::Put(entry)], &failure).await?;
                    }
                    let no_rows = LogicalPlanBuilder::empty(false).build()?;
                    Ok(DataFrame::new(ctx.state(), no_rows))
                }
                sql::ddl::SemanticDdl::DropType { name, if_exists } => {
                    let runtime = index::runtime(ctx)?;
                    let catalog = runtime.catalog().filter(|_| record);
                    let _guard = match catalog {
                        Some(catalog) => Some(catalog.lock().await),
                        None => None,
                    };
                    let name = name.to_lowercase();
                    if !runtime.type_registry().unregister(&name) && !if_exists {
                        return Err(DataFusionError::Plan(format!(
                            "semantic type {name} does not exist"
                        ))
                        .into());
                    }
                    if let Some(catalog) = catalog {
                        let forget = catalog::Change::Forget {
                            kind: catalog::EntryKind::SemanticType,
                            name: name.clone(),
                        };
                        let failure = format!(
                            "semantic type {name} is dropped for this session but will come \
                             back after a restart"
                        );
                        catalog::save(catalog, vec![forget], &failure).await?;
                    }
                    let no_rows = LogicalPlanBuilder::empty(false).build()?;
                    Ok(DataFrame::new(ctx.state(), no_rows))
                }
                sql::ddl::SemanticDdl::AddColumn {
                    table,
                    table_if_exists,
                    name,
                    text: column,
                    condition,
                } => {
                    if table_if_exists && !ctx.table_exist(table.as_str())? {
                        let no_rows = LogicalPlanBuilder::empty(false).build()?;
                        return Ok(DataFrame::new(ctx.state(), no_rows));
                    }
                    let column = semantic_column::SemanticColumn {
                        name,
                        text: column,
                        condition: semir::normalize_instruction(&condition),
                    };
                    semantic_column::add_column(ctx, &table, column, text, record).await?;
                    let no_rows = LogicalPlanBuilder::empty(false).build()?;
                    Ok(DataFrame::new(ctx.state(), no_rows))
                }
                sql::ddl::SemanticDdl::DropColumn {
                    table,
                    table_if_exists,
                    name,
                    if_exists,
                } => {
                    if table_if_exists && !ctx.table_exist(table.as_str())? {
                        let no_rows = LogicalPlanBuilder::empty(false).build()?;
                        return Ok(DataFrame::new(ctx.state(), no_rows));
                    }
                    semantic_column::drop_column(ctx, &table, &name, if_exists, record).await?;
                    let no_rows = LogicalPlanBuilder::empty(false).build()?;
                    Ok(DataFrame::new(ctx.state(), no_rows))
                }
                sql::ddl::SemanticDdl::RefreshColumn {
                    table,
                    name,
                    explain: true,
                } => semantic_column::explain_refresh(ctx, &table, &name).await,
                sql::ddl::SemanticDdl::RefreshColumn { table, name, .. } => {
                    semantic_column::refresh_column(ctx, &table, &name).await?;
                    let no_rows = LogicalPlanBuilder::empty(false).build()?;
                    Ok(DataFrame::new(ctx.state(), no_rows))
                }
                other => Err(DataFusionError::Plan(format!(
                    "semantic DDL not implemented yet: {other:?}"
                ))
                .into()),
            };
        }
        sql::SemcastStatement::Sql(statement, clauses) => (statement, clauses),
    };
    if semantic_distinct && !sql::distinct::mark_semantic_distinct(&mut statement) {
        return Err(DataFusionError::Plan(
            "SEMANTIC only qualifies DISTINCT ON (<column>); a plain SEMANTIC \
             DISTINCT has no column to compare"
                .to_owned(),
        )
        .into());
    }
    // Desugar `CAST(col AS SemanticType)[.field]` into the marker UDFs before
    // planning — DataFusion can't plan the field access itself.
    if let Some(runtime) = ctx.state().config().get_extension::<SemcastRuntime>() {
        sql::typed::rewrite_semantic_casts(&mut statement, runtime.type_registry())?;
    }
    // A relevance score is best-first; SQL's default is ascending. Fix the
    // direction while it is still `None` — the logical plan has no such thing.
    sql::rank::default_relevance_desc(&mut statement);
    // `GROUP BY MEANING OF t AS topic` names a column the optimizer has not
    // created yet; bind it before the planner tries to resolve it.
    sql::cluster::bind_meaning_labels(&mut statement);
    let mut plan = ctx.state().statement_to_plan(statement).await?;
    reject_handwritten_recall(&plan)?;
    match (clauses.recall, clauses.confidence) {
        // A view's plan runs inside whatever statement reads it, and the
        // recall promise is that statement's to make (optimizer::budget): a
        // target stored in the view would leak into statements that never
        // asked for one.
        (Some(_), _)
            if matches!(
                plan,
                datafusion::logical_expr::LogicalPlan::Ddl(
                    datafusion::logical_expr::DdlStatement::CreateView(_)
                )
            ) =>
        {
            return Err(datafusion::error::DataFusionError::Plan(
                "WITH RECALL is a promise about one statement's results, so a view \
                 cannot carry it: put WITH RECALL on the query that reads the view"
                    .to_owned(),
            )
            .into());
        }
        (Some(recall), confidence) => {
            plan = optimizer::rewrite::apply_recall(plan, recall, confidence)?;
        }
        // A confidence qualifies a recall target; alone it qualifies nothing.
        (None, Some(_)) => {
            return Err(datafusion::error::DataFusionError::Plan(
                "WITH CONFIDENCE certifies a WITH RECALL target; give a recall \
                 target for it to certify"
                    .to_owned(),
            )
            .into());
        }
        (None, None) => {}
    }
    if let Some(similarity) = clauses.similarity {
        plan = optimizer::distinct::apply_similarity(plan, similarity)?;
    }
    if matches!(plan, datafusion::logical_expr::LogicalPlan::Ddl(_)) {
        return catalog::execute_ddl(ctx, plan, text, record).await;
    }
    reject_semantic_dml(&plan)?;
    Ok(ctx.execute_logical_plan(plan).await?)
}

/// `UPDATE`/`DELETE` hand the table only the predicates DataFusion can push
/// down. A semantic one becomes an extension node it can't push, and a
/// subquery predicate is dropped outright: either way the statement would
/// touch every row. Refuse both, wherever the statement sits in the plan
/// (`EXPLAIN ANALYZE` runs it too).
fn reject_semantic_dml(plan: &datafusion::logical_expr::LogicalPlan) -> Result<()> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::logical_expr::{Expr, LogicalPlan, WriteOp};

    let mut refusal: Option<String> = None;
    plan.apply_with_subqueries(|node| {
        let LogicalPlan::Dml(dml) = node else {
            return Ok(TreeNodeRecursion::Continue);
        };
        if !matches!(dml.op, WriteOp::Delete | WriteOp::Update) {
            return Ok(TreeNodeRecursion::Continue);
        }
        let op = dml.op.to_string();
        // A filter on a semantic column is a MEANS too, under another name;
        // the planner hasn't expanded it yet, so look for its name.
        let semantic: Vec<String> = datafusion::datasource::source_as_provider(&dml.target)
            .ok()
            .and_then(|provider| {
                provider
                    .downcast_ref::<semantic_column::SemanticColumnsTable>()
                    .map(|table| table.columns().iter().map(|c| c.name.clone()).collect())
            })
            .unwrap_or_default();
        // An UPDATE assigns every column; a semantic one may only keep its
        // own value, and nothing may be computed from one — before planning,
        // that would be a MEANS hidden in an assignment.
        if dml.op == WriteOp::Update
            && !semantic.is_empty()
            && let LogicalPlan::Projection(assignments) = dml.input.as_ref()
        {
            for (field, value) in assignments.schema.fields().iter().zip(&assignments.expr) {
                let value = match value {
                    Expr::Alias(alias) => alias.expr.as_ref(),
                    other => other,
                };
                if semantic.contains(field.name()) {
                    let unchanged =
                        matches!(value, Expr::Column(column) if &column.name == field.name());
                    if !unchanged {
                        refusal = Some(format!(
                            "{} is a semantic column: it is computed, not stored, so \
                             UPDATE cannot set it",
                            field.name()
                        ));
                        return Ok(TreeNodeRecursion::Stop);
                    }
                    continue;
                }
                if value.exists(|e| {
                    Ok(matches!(e, Expr::Column(column) if semantic.contains(&column.name)))
                })? {
                    refusal = Some(format!(
                        "UPDATE cannot compute {} from a semantic column: it would ask \
                         the model for every row. Read the column into a table first",
                        field.name()
                    ));
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
        }
        dml.input.apply_with_subqueries(|node| {
            if let LogicalPlan::Filter(filter) = node
                && filter.predicate.exists(|e| {
                    Ok(matches!(e, Expr::Column(column) if semantic.contains(&column.name)))
                })?
            {
                refusal = Some(format!(
                    "{op} cannot filter on a semantic column: it would affect every row \
                     without asking the model. Keep the other rows instead: CREATE TABLE \
                     <new> AS SELECT * FROM <table> WHERE NOT <column>"
                ));
                return Ok(TreeNodeRecursion::Stop);
            }
            for expr in node.expressions() {
                if optimizer::rewrite::contains_means(&expr)? {
                    refusal = Some(format!(
                        "{op} cannot filter on MEANS or CONFIDENCE(): it would \
                         affect every row without asking the model. Keep the other \
                         rows instead: CREATE TABLE <new> AS SELECT * FROM <table> \
                         WHERE <key> NOT IN (SELECT <key> FROM <table> WHERE ... MEANS ...)"
                    ));
                    return Ok(TreeNodeRecursion::Stop);
                }
                let subquery = expr.exists(|e| {
                    Ok(matches!(
                        e,
                        Expr::InSubquery(_) | Expr::Exists(_) | Expr::ScalarSubquery(_)
                    ))
                })?;
                if subquery {
                    refusal = Some(format!(
                        "{op} cannot use a subquery: DataFusion drops subquery \
                         predicates from {op}, so it would affect every row. Keep the \
                         other rows instead: CREATE TABLE <new> AS SELECT * FROM \
                         <table> WHERE <key> NOT IN (SELECT ...)"
                    ));
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(if refusal.is_some() {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    match refusal {
        Some(message) => Err(datafusion::error::DataFusionError::Plan(message).into()),
        None => Ok(()),
    }
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
    fallbacks: Vec<Arc<dyn ModelProvider>>,
    embedder: Option<Arc<dyn ModelProvider>>,
    scorer: Option<Arc<dyn ModelProvider>>,
    cache: Arc<dyn SemanticCache>,
    chunk_store: Option<Arc<dyn crate::index::stored::ChunkStore>>,
    information_schema: bool,
    memory_limit: Option<usize>,
    catalog: Option<catalog::Catalog>,
    max_in_flight: usize,
}

impl SemcastContextBuilder {
    pub fn new(model: Arc<dyn ModelProvider>) -> Self {
        Self {
            model,
            fallbacks: Vec::new(),
            embedder: None,
            scorer: None,
            cache: Arc::new(InMemoryCache::default()),
            chunk_store: None,
            information_schema: false,
            memory_limit: None,
            catalog: None,
            max_in_flight: crate::model::DEFAULT_MAX_IN_FLIGHT,
        }
    }

    /// Cover for the session model: a request it fails on is retried against
    /// each of `fallbacks` in turn, so a provider outage costs latency instead
    /// of rows.
    ///
    /// The chain's id names every model in it, so adding a fallback
    /// invalidates cached verdicts rather than attributing one model's answer
    /// to another. An embedder left unset inherits the chain; one set
    /// explicitly through [`with_embedder`] does not.
    ///
    /// [`with_embedder`]: Self::with_embedder
    pub fn with_fallback_models(mut self, fallbacks: Vec<Arc<dyn ModelProvider>>) -> Self {
        self.fallbacks = fallbacks;
        self
    }

    /// Embed semantic indexes (and their queries) through `embedder` instead
    /// of the session model — for a dedicated embedding provider (Voyage), or
    /// when the session model can't embed (Anthropic).
    pub fn with_embedder(mut self, embedder: Arc<dyn ModelProvider>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// A cheap chat model that pre-filters a `MEANS` no index covers, when
    /// the statement asks `WITH RECALL`: it scores every row by its P("yes")
    /// (it must report logprobs, as Ollama does), and the session model
    /// verifies only rows above a floor calibrated to the target on a
    /// labeled sample. Without `WITH RECALL` it is never asked. It gets its
    /// own cap on requests in flight.
    pub fn with_scorer(mut self, scorer: Arc<dyn ModelProvider>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    pub fn with_cache(mut self, cache: Arc<dyn SemanticCache>) -> Self {
        self.cache = cache;
        self
    }

    /// Keep semantic-index chunks and vectors in `store` (the store file,
    /// via [`index::stored::SqliteChunkStore`]) so indexes survive a
    /// restart without re-embedding. In memory by default.
    pub fn with_chunk_store(mut self, store: Arc<dyn crate::index::stored::ChunkStore>) -> Self {
        self.chunk_store = Some(store);
        self
    }

    pub fn with_information_schema(mut self, on: bool) -> Self {
        self.information_schema = on;
        self
    }

    /// Cap the memory operators may reserve, in bytes. Operators that buffer
    /// (a sort, a `WITH RECALL` scan sampling its whole input) fail a query
    /// that would exceed it with a resources-exhausted error, rather than
    /// exhausting the process. Unbounded by default.
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = Some(bytes);
        self
    }

    /// Save schemas, tables, views, semantic types and semantic indexes to
    /// `catalog` as they are defined; [`restore_catalog`] brings them back
    /// in the next process.
    pub fn with_catalog(mut self, catalog: catalog::Catalog) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// Cap the model requests in flight at once, across every query the
    /// session runs (default [`DEFAULT_MAX_IN_FLIGHT`]). The session model
    /// and its fallbacks share the cap; a separate embedder gets its own.
    ///
    /// [`DEFAULT_MAX_IN_FLIGHT`]: crate::model::DEFAULT_MAX_IN_FLIGHT
    pub fn with_max_in_flight(mut self, requests: usize) -> Self {
        self.max_in_flight = requests;
        self
    }

    pub fn build(self) -> SessionContext {
        // One type registry, shared between the runtime (DDL dispatch) and the
        // marker UDFs (which resolve type fields at plan time).
        let types = Arc::new(crate::types::registry::TypeRegistry::default());
        // Wrap once, here, so the runtime and the physical planner hold the
        // same chain — they take separate handles to the session model.
        let model: Arc<dyn ModelProvider> = if self.fallbacks.is_empty() {
            self.model
        } else {
            Arc::new(crate::model::FallbackProvider::new(
                self.model,
                self.fallbacks,
            ))
        };
        // One queue per provider, in front of everything that calls it.
        let model: Arc<dyn ModelProvider> =
            Arc::new(crate::model::Scheduler::new(model, self.max_in_flight));
        let mut runtime = SemcastRuntime::new(Arc::clone(&model))
            .with_type_registry(Arc::clone(&types))
            .with_cache(Arc::clone(&self.cache));
        if let Some(embedder) = self.embedder {
            runtime = runtime.with_embedder(Arc::new(crate::model::Scheduler::new(
                embedder,
                self.max_in_flight,
            )));
        }
        let scorer: Option<Arc<dyn ModelProvider>> = self
            .scorer
            .map(|scorer| Arc::new(crate::model::Scheduler::new(scorer, self.max_in_flight)) as _);
        // The physical planner uses a scorer only when it is not the session
        // model, which would save nothing; the budget rule must agree.
        runtime = runtime.with_scorer(
            scorer
                .as_ref()
                .is_some_and(|scorer| scorer.id() != model.id()),
        );
        if let Some(store) = self.chunk_store {
            runtime = runtime.with_chunk_store(store);
        }
        if let Some(catalog) = self.catalog {
            runtime = runtime.with_catalog(catalog);
        }
        // One runtime, two holders: the session config (where the physical
        // planner and the public API find it) and the means rewrite rule,
        // which needs the index map at plan time and has no route to a
        // `SessionConfig` extension of its own.
        let runtime = Arc::new(runtime);
        let config = SessionConfig::new()
            .with_extension(Arc::clone(&runtime))
            .with_information_schema(self.information_schema);
        let mut state = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config);
        if let Some(bytes) = self.memory_limit {
            let runtime_env = datafusion::execution::runtime_env::RuntimeEnvBuilder::new()
                .with_memory_limit(bytes, 1.0)
                .build_arc()
                .expect("a memory-limited runtime env always builds");
            state = state.with_runtime_env(runtime_env);
        }
        let state = state
            .with_analyzer_rule(Arc::new(semantic_column::SemanticColumnsRule))
            .with_optimizer_rule(Arc::new(MeansRewriteRule::new(Arc::clone(&runtime))))
            .with_optimizer_rule(Arc::new(
                crate::optimizer::extract::ExtractRewriteRule::new(Arc::clone(&types)),
            ))
            .with_optimizer_rule(Arc::new(RelevanceRewriteRule))
            .with_optimizer_rule(Arc::new(ClusterRewriteRule))
            .with_optimizer_rule(Arc::new(DistinctRewriteRule))
            // Last: it reads where every MEANS filter ended up.
            .with_optimizer_rule(Arc::new(crate::optimizer::budget::RecallBudgetRule::new(
                Arc::clone(&runtime),
            )))
            .with_query_planner(Arc::new(
                SemcastQueryPlanner::new(model, self.cache).with_scorer(scorer),
            ))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_udf(sql::means_udf::means_udf());
        ctx.register_udf(sql::confidence_udf::confidence_udf());
        ctx.register_udf(sql::rank_udf::relevance_udf());
        ctx.register_udf(sql::cluster_udf::meaning_of_udf());
        ctx.register_udf(sql::distinct_udf::semantic_key_udf());
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

/// A recall target is the statement's (`WITH RECALL`, split across its lossy
/// filters by `optimizer::budget`), so a `means()` written with a third or
/// fourth argument is refused here: in a view it would leak into statements
/// that read it, and beside other filters it would claim to be the
/// statement's target. `apply_recall` adds those arguments after this check.
fn reject_handwritten_recall(plan: &datafusion::logical_expr::LogicalPlan) -> Result<()> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    let mut found = false;
    plan.apply_with_subqueries(|node| {
        node.apply_expressions(|expr| {
            expr.apply(|e| {
                if let datafusion::logical_expr::Expr::ScalarFunction(call) = e
                    && call.func.name() == sql::means_udf::MEANS_UDF_NAME
                    && call.args.len() > 2
                {
                    found = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })
        })
    })?;
    if found {
        return Err(datafusion::error::DataFusionError::Plan(
            "a means() with a recall argument (written by hand, or stored in a \
             view this statement reads) is refused: a recall target belongs to \
             the whole statement, so end the statement WITH RECALL r instead"
                .to_owned(),
        )
        .into());
    }
    Ok(())
}
