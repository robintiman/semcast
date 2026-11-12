//! Semantic columns — a named `MEANS` question attached to a table
//! (redesign phase 3).
//!
//! ```sql
//! ALTER TABLE tickets ADD SEMANTIC COLUMN angry AS (body MEANS 'the customer is angry');
//! SELECT id FROM tickets WHERE angry;
//! ```
//!
//! The table is re-registered as a [`SemanticColumnsTable`], which adds one
//! nullable boolean per semantic column to its schema, so ordinary SQL binds
//! `angry` like any column. Nothing is materialized: before the optimizer
//! runs, [`SemanticColumnsRule`] replaces the scan with a projection that
//! computes each semantic column as the `MEANS` it stands for. From there the
//! existing machinery takes over, and its answers are the stored answers of
//! that question: a column read twice, or a `MEANS` asked by hand over the
//! same text, is paid for once. Reading a column computes what isn't stored
//! yet; `REFRESH SEMANTIC COLUMN` pays ahead of time.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchema, Result, plan_err};
use datafusion::datasource::{TableProvider, TableType, provider_as_source, source_as_provider};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{Expr, LogicalPlan, Projection, TableScan, lit};
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_plan::ExecutionPlan;

/// One semantic column: `name AS (text MEANS 'condition')`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SemanticColumn {
    pub name: String,
    /// The table column the question reads.
    pub text: String,
    pub condition: String,
}

/// A table plus its semantic columns. Its schema is the table's, then one
/// nullable boolean per semantic column; scanning it directly is refused,
/// since only [`SemanticColumnsRule`] knows how to compute the columns.
///
/// The columns change in place, so a table keeps one registration: adding or
/// dropping a column never leaves a moment with no table under the name.
#[derive(Debug)]
pub struct SemanticColumnsTable {
    inner: Arc<dyn TableProvider>,
    state: std::sync::RwLock<Columns>,
}

#[derive(Debug, Clone)]
struct Columns {
    columns: Vec<SemanticColumn>,
    schema: SchemaRef,
}

impl SemanticColumnsTable {
    /// `inner` with `columns` added. A name already in the table, or a text
    /// column that isn't one, is a plan error.
    pub fn new(inner: Arc<dyn TableProvider>, columns: Vec<SemanticColumn>) -> Result<Self> {
        let state = Columns::new(&inner, columns)?;
        Ok(Self {
            inner,
            state: std::sync::RwLock::new(state),
        })
    }

    /// The table without its semantic columns.
    pub fn inner(&self) -> &Arc<dyn TableProvider> {
        &self.inner
    }

    pub fn columns(&self) -> Vec<SemanticColumn> {
        self.snapshot().columns
    }

    fn snapshot(&self) -> Columns {
        self.state
            .read()
            .expect("semantic columns lock poisoned")
            .clone()
    }

    /// Make `columns` the table's semantic columns, validated as in
    /// [`Self::new`].
    fn set_columns(&self, columns: Vec<SemanticColumn>) -> Result<()> {
        let state = Columns::new(&self.inner, columns)?;
        *self.state.write().expect("semantic columns lock poisoned") = state;
        Ok(())
    }
}

impl Columns {
    fn new(inner: &Arc<dyn TableProvider>, columns: Vec<SemanticColumn>) -> Result<Self> {
        let base = inner.schema();
        let mut fields: Vec<Arc<Field>> = base.fields().iter().cloned().collect();
        for column in &columns {
            if fields.iter().any(|field| field.name() == &column.name) {
                return plan_err!("the table already has a column named {}", column.name);
            }
            match base.field_with_name(&column.text) {
                Ok(field)
                    if matches!(
                        field.data_type(),
                        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                    ) => {}
                Ok(field) => {
                    return plan_err!(
                        "a semantic column reads text, but {} is {}",
                        column.text,
                        field.data_type()
                    );
                }
                Err(_) => return plan_err!("the table has no column named {}", column.text),
            }
            fields.push(Arc::new(Field::new(&column.name, DataType::Boolean, true)));
        }
        let schema = Arc::new(Schema::new_with_metadata(fields, base.metadata().clone()));
        Ok(Self { columns, schema })
    }
}

#[async_trait]
impl TableProvider for SemanticColumnsTable {
    fn schema(&self) -> SchemaRef {
        self.snapshot().schema
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan_err!(
            "a table with semantic columns is read through semcast::sql, which computes \
             them; this query bypassed it"
        )
    }
    fn get_table_definition(&self) -> Option<&str> {
        self.inner.get_table_definition()
    }

    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        self.inner.get_column_default(column)
    }

    /// Rows go to the table underneath; a semantic column isn't stored, so a
    /// value given for one is dropped (it is computed when read).
    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: datafusion::logical_expr::dml::InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_expr::expressions::col;
        use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
        let base = self.inner.schema();
        let input_schema = input.schema();
        let exprs = base
            .fields()
            .iter()
            .map(|field| {
                Ok(ProjectionExpr::new(
                    col(field.name(), &input_schema)?,
                    field.name().clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let stored = Arc::new(ProjectionExec::try_new(exprs, input)?);
        self.inner.insert_into(state, stored, insert_op).await
    }

    /// A filter on a semantic column never gets here: `semcast::sql`
    /// refuses it (see `lib.rs::reject_semantic_dml`), as it refuses any
    /// `MEANS` in a `DELETE`.
    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.delete_from(state, filters).await
    }

    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // DataFusion assigns every column, an unchanged one its current value.
        // Those are dropped for a semantic column; any other value is refused.
        let semantic: Vec<String> = self.columns().into_iter().map(|c| c.name).collect();
        let mut stored = Vec::with_capacity(assignments.len());
        for (name, value) in assignments {
            if !semantic.contains(&name) {
                stored.push((name, value));
                continue;
            }
            // Unchanged reads as the column itself, or as what the planner
            // made of it: the `MEANS` the column stands for, or the column a
            // classify node computes it into.
            let value = match value {
                Expr::Alias(alias) => *alias.expr,
                other => other,
            };
            let unchanged = match &value {
                Expr::Column(column) => {
                    column.name == name
                        || column.name.starts_with("__sem_class_")
                        || column.name.starts_with("__sem_conf_")
                }
                other => crate::optimizer::rewrite::is_means_call(other),
            };
            if !unchanged {
                return plan_err!("{name} is a semantic column: it is computed, not stored");
            }
        }
        self.inner.update(state, stored, filters).await
    }

    async fn truncate(&self, state: &dyn Session) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.truncate(state).await
    }
}

/// Rewrites a scan of a [`SemanticColumnsTable`] into a projection over its
/// underlying table that computes each semantic column as its `MEANS`.
/// Runs as an analyzer rule, before any optimizer rule (the `MEANS` rewrite
/// included) sees the plan.
#[derive(Debug, Default)]
pub struct SemanticColumnsRule;

impl AnalyzerRule for SemanticColumnsRule {
    fn name(&self) -> &str {
        "semcast_semantic_columns"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up_with_subqueries(|plan| {
            if let LogicalPlan::Filter(filter) = &plan
                && expansion_under(&filter.input).is_some()
                && !has_subquery(&filter.predicate)?
            {
                let LogicalPlan::Filter(filter) = plan else {
                    unreachable!("matched above")
                };
                return push_below_expansion(filter);
            }
            let LogicalPlan::TableScan(scan) = &plan else {
                return Ok(Transformed::no(plan));
            };
            let Ok(provider) = source_as_provider(&scan.source) else {
                return Ok(Transformed::no(plan));
            };
            let Some(table) = provider.downcast_ref::<SemanticColumnsTable>() else {
                return Ok(Transformed::no(plan));
            };
            let LogicalPlan::TableScan(scan) = plan else {
                unreachable!("matched above")
            };
            Ok(Transformed::yes(expand(scan, table)?))
        })
        .map(|transformed| transformed.data)
    }
}

/// The expansion a filter's input is, directly or under a table alias
/// (`FROM tickets t`).
fn expansion_under(plan: &LogicalPlan) -> Option<&Projection> {
    let plan = match plan {
        LogicalPlan::SubqueryAlias(alias) => alias.input.as_ref(),
        other => other,
    };
    match plan {
        LogicalPlan::Projection(projection) if is_expansion(plan) => Some(projection),
        _ => None,
    }
}

/// A predicate with a subquery is left where it is: its correlated columns
/// live inside the subquery's plan, which the substitution doesn't reach.
fn has_subquery(predicate: &Expr) -> Result<bool> {
    predicate.exists(|expr| {
        Ok(matches!(
            expr,
            Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
        ))
    })
}

/// Whether `plan` is a projection [`expand`] built: every expression a
/// column or a `MEANS` under the semantic column's name, over a scan.
fn is_expansion(plan: &LogicalPlan) -> bool {
    let LogicalPlan::Projection(projection) = plan else {
        return false;
    };
    matches!(projection.input.as_ref(), LogicalPlan::TableScan(_))
        && projection.expr.iter().any(is_means_alias)
        && projection
            .expr
            .iter()
            .all(|expr| matches!(expr, Expr::Column(_)) || is_means_alias(expr))
}

fn is_means_alias(expr: &Expr) -> bool {
    matches!(expr, Expr::Alias(alias) if crate::optimizer::rewrite::is_means_call(&alias.expr))
}

/// `WHERE angry` over an expanded scan: filter the scan by the `MEANS` the
/// column stands for instead, below the projection. The filter is then a
/// `MEANS` filter like any other — store lookup, index, scorer — rather than
/// a label computed for every row and filtered afterwards. DataFusion won't
/// push a filter through the projection itself: `means()` is volatile, so
/// it can't tell the expression is the same question either way.
fn push_below_expansion(
    filter: datafusion::logical_expr::Filter,
) -> Result<Transformed<LogicalPlan>> {
    let alias = match filter.input.as_ref() {
        LogicalPlan::SubqueryAlias(alias) => Some(alias.alias.clone()),
        _ => None,
    };
    let projection = expansion_under(&filter.input).expect("checked by the caller");
    // Each output column, by name, and what it computes over the scan. Under
    // an alias the predicate names columns by the alias, so match by name:
    // a single table's columns are unique by name.
    let definitions: Vec<(String, Expr)> = projection
        .schema
        .fields()
        .iter()
        .zip(projection.expr.iter())
        .map(|(field, expr)| {
            let definition = match expr {
                Expr::Alias(alias) => alias.expr.as_ref().clone(),
                other => other.clone(),
            };
            (field.name().clone(), definition)
        })
        .collect();
    let predicate = filter.predicate.clone().transform_up(|expr| match &expr {
        Expr::Column(column) => Ok(definitions
            .iter()
            .find(|(name, _)| name == &column.name)
            .map(|(_, definition)| Transformed::yes(definition.clone()))
            .unwrap_or(Transformed::no(expr))),
        _ => Ok(Transformed::no(expr)),
    })?;
    let below = LogicalPlan::Filter(datafusion::logical_expr::Filter::try_new(
        predicate.data,
        Arc::clone(&projection.input),
    )?);
    let projected = LogicalPlan::Projection(Projection::try_new(
        projection.expr.clone(),
        Arc::new(below),
    )?);
    Ok(Transformed::yes(match alias {
        Some(alias) => LogicalPlan::SubqueryAlias(
            datafusion::logical_expr::SubqueryAlias::try_new(Arc::new(projected), alias)?,
        ),
        None => projected,
    }))
}

/// The scan of `table`'s underlying table, then a projection giving back
/// exactly the columns the original scan produced, under the same qualifier.
fn expand(scan: TableScan, table: &SemanticColumnsTable) -> Result<LogicalPlan> {
    if !scan.filters.is_empty() || scan.fetch.is_some() {
        return plan_err!("a pushed-down filter or limit on a table with semantic columns");
    }
    let qualifier = scan.table_name.clone();
    let inner = LogicalPlan::TableScan(TableScan::try_new(
        qualifier.clone(),
        provider_as_source(Arc::clone(&table.inner)),
        None,
        vec![],
        None,
    )?);
    let base = table.inner.schema();
    let state = table.snapshot();
    let means = crate::sql::means_udf::means_udf();
    let wanted: Vec<usize> = match &scan.projection {
        Some(projection) => projection.clone(),
        None => (0..state.schema.fields().len()).collect(),
    };
    // The scan was planned against the columns as they were then; an ALTER
    // since may have changed them under the same index.
    let planned = scan.projected_schema.fields();
    let unchanged = planned.len() == wanted.len()
        && wanted.iter().zip(planned.iter()).all(|(&i, field)| {
            i < state.schema.fields().len() && state.schema.field(i).name() == field.name()
        });
    if !unchanged {
        return plan_err!(
            "the semantic columns of {} changed while this query was planned; run it again",
            scan.table_name
        );
    }
    let exprs = wanted
        .into_iter()
        .map(|i| {
            let field = state.schema.field(i);
            if i < base.fields().len() {
                return Expr::Column(Column::new(Some(qualifier.clone()), field.name()));
            }
            let column = &state.columns[i - base.fields().len()];
            Expr::ScalarFunction(ScalarFunction::new_udf(
                Arc::new(means.clone()),
                vec![
                    Expr::Column(Column::new(Some(qualifier.clone()), &column.text)),
                    lit(column.condition.clone()),
                ],
            ))
            .alias_qualified(Some(qualifier.clone()), &column.name)
        })
        .collect::<Vec<_>>();
    let projection = Projection::try_new(exprs, Arc::new(inner))?;
    // The projection must stand exactly where the scan stood.
    debug_assert_eq!(
        projection.schema.fields().len(),
        scan.projected_schema.fields().len()
    );
    let _: &DFSchema = &projection.schema;
    Ok(LogicalPlan::Projection(projection))
}

impl fmt::Display for SemanticColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} AS ({} MEANS '{}')",
            self.name,
            self.text,
            self.condition.replace('\'', "''")
        )
    }
}

/// `ALTER TABLE table ADD SEMANTIC COLUMN`: re-register `table` with the
/// column added, and save the statement `text` to the catalog when the
/// table is a saved one and `record` is set.
pub async fn add_column(
    ctx: &datafusion::execution::context::SessionContext,
    table: &str,
    column: SemanticColumn,
    text: &str,
    record: bool,
) -> crate::Result<()> {
    let runtime = crate::index::runtime(ctx)?;
    let catalog = runtime.catalog().filter(|_| record);
    let _guard = match catalog {
        Some(catalog) => Some(catalog.lock().await),
        None => None,
    };
    let _serial = ALTERS.lock().await;
    let provider = ctx.table_provider(table).await?;
    let name = column.name.clone();
    match provider.downcast_ref::<SemanticColumnsTable>() {
        // Changed in place: the name never goes without a table.
        Some(wrapped) => {
            let mut columns = wrapped.columns();
            columns.push(column);
            wrapped.set_columns(columns)?;
        }
        // A table's first semantic column wraps it, once. DataFusion offers
        // no swap, so for that moment the name is free.
        None => {
            let wrapped = SemanticColumnsTable::new(provider, vec![column])?;
            ctx.deregister_table(table)?;
            ctx.register_table(table, Arc::new(wrapped))?;
        }
    }
    if let Some(catalog) = catalog {
        let resolved = resolve(ctx, table);
        if catalog.has_relation(&resolved).await? {
            let entry = crate::catalog::Entry {
                kind: crate::catalog::EntryKind::SemanticColumn,
                name: catalog_key(table, &name),
                definition: text.to_owned(),
            };
            let failure = format!(
                "the semantic column {table}.{name} is defined for this session but will be \
                 gone after a restart"
            );
            crate::catalog::save(catalog, vec![crate::catalog::Change::Put(entry)], &failure)
                .await?;
        } else {
            tracing::info!(
                "semantic column {table}.{name} is kept for this session only: {table} is \
                 not a saved table"
            );
        }
    }
    Ok(())
}

/// `ALTER TABLE table DROP SEMANTIC COLUMN`. Its stored answers stay: they
/// are answers to the question, which a `MEANS` or a re-added column reads.
pub async fn drop_column(
    ctx: &datafusion::execution::context::SessionContext,
    table: &str,
    name: &str,
    if_exists: bool,
    record: bool,
) -> crate::Result<()> {
    let runtime = crate::index::runtime(ctx)?;
    let catalog = runtime.catalog().filter(|_| record);
    let _guard = match catalog {
        Some(catalog) => Some(catalog.lock().await),
        None => None,
    };
    let _serial = ALTERS.lock().await;
    let provider = ctx.table_provider(table).await?;
    let Some(wrapped) = provider.downcast_ref::<SemanticColumnsTable>() else {
        if if_exists {
            return Ok(());
        }
        return Err(datafusion::error::DataFusionError::Plan(format!(
            "{table} has no semantic column named {name}"
        ))
        .into());
    };
    let mut columns = wrapped.columns();
    let before = columns.len();
    columns.retain(|column| column.name != name);
    if columns.len() == before {
        if if_exists {
            return Ok(());
        }
        return Err(datafusion::error::DataFusionError::Plan(format!(
            "{table} has no semantic column named {name}"
        ))
        .into());
    }
    // In place, even to none: the wrapper stays, and with no columns reads
    // exactly like the table it wraps.
    wrapped.set_columns(columns)?;
    if let Some(catalog) = catalog {
        let forget = crate::catalog::Change::Forget {
            kind: crate::catalog::EntryKind::SemanticColumn,
            name: catalog_key(table, name),
        };
        let failure = format!(
            "the semantic column {table}.{name} is dropped for this session but will come \
             back after a restart"
        );
        crate::catalog::save(catalog, vec![forget], &failure).await?;
    }
    Ok(())
}

/// `REFRESH SEMANTIC COLUMN table(name)`: read the column over every row, so
/// every answer it lacks is asked now and stored. Returns the rows read.
pub async fn refresh_column(
    ctx: &datafusion::execution::context::SessionContext,
    table: &str,
    name: &str,
) -> crate::Result<usize> {
    let provider = ctx.table_provider(table).await?;
    let known = provider
        .downcast_ref::<SemanticColumnsTable>()
        .is_some_and(|wrapped| wrapped.columns().iter().any(|column| column.name == name));
    if !known {
        return Err(datafusion::error::DataFusionError::Plan(format!(
            "{table} has no semantic column named {name}"
        ))
        .into());
    }
    // The analyzer and optimizer rules apply to DataFrame plans too, so this
    // reads the column exactly as a query would.
    let batches = ctx
        .table(table)
        .await?
        .select_columns(&[name])?
        .collect()
        .await?;
    Ok(batches.iter().map(|batch| batch.num_rows()).sum())
}

/// One `ALTER` at a time: each reads the table's columns, changes them, and
/// writes them back (or wraps the table), and two interleaved would lose one's
/// change.
static ALTERS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The catalog key of a semantic column: `table(name)` by the table's bare
/// name, as indexes are keyed, so `public.tickets` and `tickets` name one
/// entry, and dropping the table forgets it.
fn catalog_key(table: &str, name: &str) -> String {
    let bare = datafusion::common::TableReference::from(table)
        .table()
        .to_owned();
    crate::catalog::index_name(&bare, name)
}

/// `table` as the catalog names relations: fully qualified.
fn resolve(ctx: &datafusion::execution::context::SessionContext, table: &str) -> String {
    let state = ctx.state();
    let defaults = &state.config().options().catalog;
    datafusion::common::TableReference::from(table)
        .resolve(&defaults.default_catalog, &defaults.default_schema)
        .to_string()
}
