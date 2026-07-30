//! The queryable half of batch jobs: `semcast_jobs` and `job_result('<id>')`.
//!
//! Both are ordinary DataFusion catalog entries rather than intercepted
//! statements, so `WHERE status = 'failed'`, `ORDER BY submitted_at DESC`,
//! and joins against the result all work without any extra syntax.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray, TimestampMillisecondArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::catalog::memory::MemTable;
use datafusion::catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;

use super::{JobRecord, JobRegistry};

/// Register `semcast_jobs` and `job_result()` on `ctx`.
///
/// Must run *after* `SemcastContextBuilder::build()`: `enable_url_table()`
/// rebuilds the context, and registrations made before it would be dropped.
pub fn register(ctx: &SessionContext, jobs: &Arc<JobRegistry>) -> DfResult<()> {
    ctx.register_table("semcast_jobs", Arc::new(JobsTable::new(Arc::clone(jobs))))?;
    ctx.register_udtf("job_result", Arc::new(JobResultFunc::new(Arc::clone(jobs))));
    Ok(())
}

/// The registry as a table. Each scan takes a fresh snapshot, so polling in a
/// loop sees live status without any cache-invalidation machinery.
#[derive(Debug)]
pub struct JobsTable {
    jobs: Arc<JobRegistry>,
    schema: SchemaRef,
}

impl JobsTable {
    pub fn new(jobs: Arc<JobRegistry>) -> Self {
        Self {
            jobs,
            schema: jobs_schema(),
        }
    }

    fn batch(&self) -> DfResult<RecordBatch> {
        let records = self.jobs.snapshot();
        RecordBatch::try_new(Arc::clone(&self.schema), columns(&records))
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

#[async_trait]
impl TableProvider for JobsTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let table = MemTable::try_new(Arc::clone(&self.schema), vec![vec![self.batch()?]])?;
        table.scan(state, projection, filters, limit).await
    }
}

fn jobs_schema() -> SchemaRef {
    let ts = || DataType::Timestamp(TimeUnit::Millisecond, None);
    Arc::new(Schema::new(vec![
        Field::new("job_id", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("sql", DataType::Utf8, false),
        Field::new("submitted_at", ts(), false),
        Field::new("started_at", ts(), true),
        Field::new("finished_at", ts(), true),
        Field::new("rows", DataType::Int64, true),
        Field::new("command_tag", DataType::Utf8, true),
        Field::new("error", DataType::Utf8, true),
        Field::new("progress", DataType::Utf8, true),
        Field::new("index_hits", DataType::Int64, false),
        Field::new("rows_pruned", DataType::Int64, false),
        Field::new("model_calls", DataType::Int64, false),
        Field::new("cache_hits", DataType::Int64, false),
        Field::new("extract_model_calls", DataType::Int64, false),
        Field::new("result_path", DataType::Utf8, true),
    ]))
}

fn columns(records: &[JobRecord]) -> Vec<datafusion::arrow::array::ArrayRef> {
    let text = |f: fn(&JobRecord) -> Option<String>| -> datafusion::arrow::array::ArrayRef {
        Arc::new(records.iter().map(f).collect::<StringArray>())
    };
    let count = |f: fn(&JobRecord) -> usize| -> datafusion::arrow::array::ArrayRef {
        Arc::new(records.iter().map(|r| f(r) as i64).collect::<Int64Array>())
    };
    vec![
        text(|r| Some(r.id.clone())),
        text(|r| Some(r.status.as_str().to_owned())),
        text(|r| Some(r.sql.clone())),
        Arc::new(
            records
                .iter()
                .map(|r| Some(r.submitted_at_ms))
                .collect::<TimestampMillisecondArray>(),
        ),
        Arc::new(
            records
                .iter()
                .map(|r| r.started_at_ms)
                .collect::<TimestampMillisecondArray>(),
        ),
        Arc::new(
            records
                .iter()
                .map(|r| r.finished_at_ms)
                .collect::<TimestampMillisecondArray>(),
        ),
        Arc::new(records.iter().map(|r| r.rows).collect::<Int64Array>()),
        text(|r| r.command_tag.clone()),
        text(|r| r.error.clone()),
        text(|r| r.progress.clone()),
        count(|r| r.funnel.index_hits),
        count(|r| r.funnel.rows_pruned),
        count(|r| r.funnel.model_calls),
        count(|r| r.funnel.cache_hits),
        count(|r| r.funnel.extract_model_calls),
        text(|r| r.result_path.clone()),
    ]
}

/// `job_result('<id>')` — the rows a finished job wrote.
#[derive(Debug)]
pub struct JobResultFunc {
    jobs: Arc<JobRegistry>,
}

impl JobResultFunc {
    pub fn new(jobs: Arc<JobRegistry>) -> Self {
        Self { jobs }
    }
}

impl TableFunctionImpl for JobResultFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let id = match args.exprs() {
            [Expr::Literal(ScalarValue::Utf8(Some(id)), _)] => id.clone(),
            _ => {
                return Err(DataFusionError::Plan(
                    "job_result takes one argument: job_result('<job_id>')".to_owned(),
                ));
            }
        };

        let record = self
            .jobs
            .get(&id)
            .ok_or_else(|| DataFusionError::Plan(format!("no such job: {id}")))?;
        let path = match (&record.result_path, record.command_tag.as_deref()) {
            (Some(path), _) => path.clone(),
            // A command statement succeeded but has no result shape — say so
            // rather than reporting a missing file.
            (None, Some(tag)) => {
                return Err(DataFusionError::Plan(format!(
                    "job {id} ran `{tag}`, which produces no rows"
                )));
            }
            (None, None) if !record.status.is_terminal() => {
                return Err(DataFusionError::Plan(format!(
                    "job {id} is {} — no result yet",
                    record.status.as_str()
                )));
            }
            (None, None) => {
                return Err(DataFusionError::Plan(format!(
                    "job {id} is {} and produced no result",
                    record.status.as_str()
                )));
            }
        };

        // TableProvider::schema is synchronous, so read the Arrow schema
        // straight out of the Parquet footer instead of inferring it async.
        let file = std::fs::File::open(&path)
            .map_err(|e| DataFusionError::Plan(format!("job {id}: cannot read {path}: {e}")))?;
        let reader =
            datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                file,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let schema = Arc::clone(reader.schema());

        let url = ListingTableUrl::parse(&path)?;
        let options =
            ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
        let config = ListingTableConfig::new(url)
            .with_listing_options(options)
            .with_schema(schema);
        Ok(Arc::new(ListingTable::try_new(config)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::jobs::JobStatus;

    async fn registry_with_jobs() -> (tempfile::TempDir, Arc<JobRegistry>) {
        let dir = tempfile::tempdir().unwrap();
        let jobs = Arc::new(JobRegistry::new(dir.path(), 2).unwrap());
        let done = jobs.create("SELECT 1").unwrap();
        jobs.update(&done, |r| {
            r.status = JobStatus::Succeeded;
            r.rows = Some(3);
            r.funnel.model_calls = 12;
        });
        jobs.create("SELECT 2").unwrap();
        (dir, jobs)
    }

    #[tokio::test]
    async fn jobs_table_is_queryable_sql() {
        let (_dir, jobs) = registry_with_jobs().await;
        let ctx = SessionContext::new();
        register(&ctx, &jobs).unwrap();

        let batches = ctx
            .sql("SELECT job_id, status, rows, model_calls FROM semcast_jobs WHERE status = 'succeeded'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "the filter reaches the snapshot");

        let counts = batches[0]
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(counts.value(0), 12, "funnel counts land as columns");
    }

    #[tokio::test]
    async fn jobs_table_reflects_later_updates() {
        let (_dir, jobs) = registry_with_jobs().await;
        let ctx = SessionContext::new();
        register(&ctx, &jobs).unwrap();

        let before = ctx
            .sql("SELECT * FROM semcast_jobs")
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        jobs.create("SELECT 3").unwrap();
        let after = ctx
            .sql("SELECT * FROM semcast_jobs")
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        assert_eq!(after, before + 1, "each scan re-snapshots the registry");
    }

    #[tokio::test]
    async fn job_result_explains_why_there_are_no_rows() {
        let (_dir, jobs) = registry_with_jobs().await;
        let ctx = SessionContext::new();
        register(&ctx, &jobs).unwrap();

        let err = ctx
            .sql("SELECT * FROM job_result('nope')")
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no such job"), "got: {err}");

        let queued = jobs
            .snapshot()
            .into_iter()
            .find(|r| r.sql == "SELECT 2")
            .unwrap();
        let err = ctx
            .sql(&format!("SELECT * FROM job_result('{}')", queued.id))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no result yet"), "got: {err}");

        let ddl = jobs.create("CREATE TABLE t AS SELECT 1").unwrap();
        jobs.update(&ddl, |r| {
            r.status = JobStatus::Succeeded;
            r.command_tag = Some("CREATE TABLE".to_owned());
        });
        let err = ctx
            .sql(&format!("SELECT * FROM job_result('{ddl}')"))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("produces no rows"), "got: {err}");
    }
}
