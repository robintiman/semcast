//! Tests for roadmap step 8: ingestion — DuckDB-like local file access.
//! Path-literal SELECT (with globs), `CREATE EXTERNAL TABLE`, CTAS
//! materialization, and `COPY TO` export, all through `semcast::sql`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::execution::context::SessionContext;
use semcast::model::MockModel;
use semcast::semcast_context;

const MATCHING: &str = "we agreed to ship offline sync in Q3";
const OTHER: &str = "nothing notable happened";

fn test_context() -> SessionContext {
    semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])))
}

async fn run(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
}

/// Values of the first column, which every query here shapes as Int64.
fn first_column(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("first column is Int64")
                .values()
                .to_vec()
        })
        .collect()
}

/// meetings.csv with a header row: meeting_id, transcript.
fn write_meetings_csv(dir: &Path) -> PathBuf {
    let path = dir.join("meetings.csv");
    std::fs::write(
        &path,
        format!("meeting_id,transcript\n1,{MATCHING}\n2,{OTHER}\n"),
    )
    .unwrap();
    path
}

#[tokio::test]
async fn selects_from_a_csv_path_literal() {
    let ctx = test_context();
    let dir = tempfile::tempdir().unwrap();
    let path = write_meetings_csv(dir.path());

    let batches = run(
        &ctx,
        &format!(
            "SELECT meeting_id FROM '{}' ORDER BY meeting_id",
            path.display()
        ),
    )
    .await;
    assert_eq!(first_column(&batches), vec![1, 2]);
}

#[tokio::test]
async fn copies_to_parquet_and_selects_back() {
    let ctx = test_context();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meetings.parquet");

    run(
        &ctx,
        &format!(
            "COPY (SELECT * FROM (VALUES (1, '{MATCHING}'), (2, '{OTHER}'))
                    AS t(meeting_id, transcript))
             TO '{}' STORED AS PARQUET",
            path.display(),
        ),
    )
    .await;

    let batches = run(
        &ctx,
        &format!(
            "SELECT meeting_id FROM '{}' ORDER BY meeting_id",
            path.display()
        ),
    )
    .await;
    assert_eq!(first_column(&batches), vec![1, 2]);
}

#[tokio::test]
async fn globs_across_parquet_parts() {
    let ctx = test_context();
    let dir = tempfile::tempdir().unwrap();
    for (part, id) in [("part-0", 1), ("part-1", 2)] {
        run(
            &ctx,
            &format!(
                "COPY (SELECT {id} AS meeting_id) TO '{}' STORED AS PARQUET",
                dir.path().join(format!("{part}.parquet")).display(),
            ),
        )
        .await;
    }

    let batches = run(
        &ctx,
        &format!(
            "SELECT count(*) FROM '{}'",
            dir.path().join("part-*.parquet").display(),
        ),
    )
    .await;
    assert_eq!(first_column(&batches), vec![2]);
}

#[tokio::test]
async fn create_external_table_reads_csv_with_options() {
    let ctx = test_context();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meetings.csv");
    std::fs::write(&path, format!("1;{MATCHING}\n2;{OTHER}\n")).unwrap();

    run(
        &ctx,
        &format!(
            "CREATE EXTERNAL TABLE meetings (meeting_id BIGINT, transcript VARCHAR)
             STORED AS CSV LOCATION '{}'
             OPTIONS ('format.has_header' 'false', 'format.delimiter' ';')",
            path.display(),
        ),
    )
    .await;

    let batches = run(&ctx, "SELECT meeting_id FROM meetings ORDER BY meeting_id").await;
    assert_eq!(first_column(&batches), vec![1, 2]);
}

#[tokio::test]
async fn ctas_materializes_a_file_into_the_semantic_path() {
    let ctx = test_context();
    let dir = tempfile::tempdir().unwrap();
    let path = write_meetings_csv(dir.path());

    run(
        &ctx,
        &format!(
            "CREATE TABLE meetings AS SELECT * FROM '{}'",
            path.display()
        ),
    )
    .await;

    let batches = run(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'discussed offline sync'",
    )
    .await;
    assert_eq!(first_column(&batches), vec![1]);
}

#[tokio::test]
async fn means_with_recall_works_directly_over_a_file() {
    let ctx = test_context();
    let dir = tempfile::tempdir().unwrap();
    let path = write_meetings_csv(dir.path());

    let batches = run(
        &ctx,
        &format!(
            "SELECT meeting_id FROM '{}'
             WHERE transcript MEANS 'discussed offline sync'
             WITH RECALL 0.9",
            path.display(),
        ),
    )
    .await;
    assert_eq!(first_column(&batches), vec![1]);
}

#[tokio::test]
async fn semantic_index_builds_on_an_external_table() {
    let ctx = test_context();
    let dir = tempfile::tempdir().unwrap();
    let path = write_meetings_csv(dir.path());

    run(
        &ctx,
        &format!(
            "CREATE EXTERNAL TABLE meetings STORED AS CSV LOCATION '{}'",
            path.display(),
        ),
    )
    .await;
    run(&ctx, "CREATE SEMANTIC INDEX ON meetings(transcript)").await;

    let batches = run(
        &ctx,
        "SELECT meeting_id FROM meetings WHERE transcript MEANS 'discussed offline sync'",
    )
    .await;
    assert_eq!(first_column(&batches), vec![1]);
}
