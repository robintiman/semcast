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
#[derive(Debug)]
pub struct SemanticColumnsTable {
    inner: Arc<dyn TableProvider>,
    columns: Vec<SemanticColumn>,
    schema: SchemaRef,
}

impl SemanticColumnsTable {
    /// `inner` with `columns` added. A name already in the table, or a text
    /// column that isn't one, is a plan error.
    pub fn new(inner: Arc<dyn TableProvider>, columns: Vec<SemanticColumn>) -> Result<Self> {
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
        Ok(Self {
            inner,
            columns,
            schema,
        })
    }

    /// The table without its semantic columns.
    pub fn inner(&self) -> &Arc<dyn TableProvider> {
        &self.inner
    }

    pub fn columns(&self) -> &[SemanticColumn] {
        &self.columns
    }
}

#[async_trait]
impl TableProvider for SemanticColumnsTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
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
                && is_expansion(&filter.input)
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
    let LogicalPlan::Projection(projection) = filter.input.as_ref() else {
        unreachable!("checked by is_expansion");
    };
    // Each output column of the projection, by name, and what it computes.
    let definitions: Vec<(Column, Expr)> = projection
        .schema
        .columns()
        .into_iter()
        .zip(projection.expr.iter())
        .map(|(column, expr)| {
            let definition = match expr {
                Expr::Alias(alias) => alias.expr.as_ref().clone(),
                other => other.clone(),
            };
            (column, definition)
        })
        .collect();
    let predicate = filter.predicate.clone().transform_up(|expr| match &expr {
        Expr::Column(column) => Ok(definitions
            .iter()
            .find(|(name, _)| name == column)
            .map(|(_, definition)| Transformed::yes(definition.clone()))
            .unwrap_or(Transformed::no(expr))),
        _ => Ok(Transformed::no(expr)),
    })?;
    let below = LogicalPlan::Filter(datafusion::logical_expr::Filter::try_new(
        predicate.data,
        Arc::clone(&projection.input),
    )?);
    Ok(Transformed::yes(LogicalPlan::Projection(
        Projection::try_new(projection.expr.clone(), Arc::new(below))?,
    )))
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
    let means = crate::sql::means_udf::means_udf();
    let wanted: Vec<usize> = match &scan.projection {
        Some(projection) => projection.clone(),
        None => (0..table.schema.fields().len()).collect(),
    };
    let exprs = wanted
        .into_iter()
        .map(|i| {
            let field = table.schema.field(i);
            if i < base.fields().len() {
                return Expr::Column(Column::new(Some(qualifier.clone()), field.name()));
            }
            let column = &table.columns[i - base.fields().len()];
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
    let provider = ctx.table_provider(table).await?;
    let (inner, mut columns) = unwrap(provider);
    let name = column.name.clone();
    columns.push(column);
    let wrapped = SemanticColumnsTable::new(inner, columns)?;
    ctx.deregister_table(table)?;
    ctx.register_table(table, Arc::new(wrapped))?;
    if let Some(catalog) = catalog {
        let resolved = resolve(ctx, table);
        if catalog.has_relation(&resolved).await? {
            let entry = crate::catalog::Entry {
                kind: crate::catalog::EntryKind::SemanticColumn,
                name: crate::catalog::index_name(table, &name),
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
    let provider = ctx.table_provider(table).await?;
    let (inner, mut columns) = unwrap(provider);
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
    let provider: Arc<dyn TableProvider> = if columns.is_empty() {
        inner
    } else {
        Arc::new(SemanticColumnsTable::new(inner, columns)?)
    };
    ctx.deregister_table(table)?;
    ctx.register_table(table, provider)?;
    if let Some(catalog) = catalog {
        let forget = crate::catalog::Change::Forget {
            kind: crate::catalog::EntryKind::SemanticColumn,
            name: crate::catalog::index_name(table, name),
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
        .is_some_and(|wrapped| wrapped.columns.iter().any(|column| column.name == name));
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

/// The table under any semantic columns, and those columns.
fn unwrap(provider: Arc<dyn TableProvider>) -> (Arc<dyn TableProvider>, Vec<SemanticColumn>) {
    match provider.downcast_ref::<SemanticColumnsTable>() {
        Some(wrapped) => (Arc::clone(&wrapped.inner), wrapped.columns.clone()),
        None => (provider, Vec::new()),
    }
}

/// `table` as the catalog names relations: fully qualified.
fn resolve(ctx: &datafusion::execution::context::SessionContext, table: &str) -> String {
    let state = ctx.state();
    let defaults = &state.config().options().catalog;
    datafusion::common::TableReference::from(table)
        .resolve(&defaults.default_catalog, &defaults.default_schema)
        .to_string()
}
