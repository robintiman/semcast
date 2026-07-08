//! Arrow results → Postgres wire format. Text encoding only — the simple
//! query protocol never requests binary. The declared type OID is display
//! metadata for clients; values travel as Postgres-shaped text.

use std::sync::Arc;

use datafusion::arrow::array::{Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use futures::stream;
use pgwire::api::Type;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo, QueryResponse, Response};
use pgwire::error::{PgWireError, PgWireResult};

/// Postgres shapes: space-separated timestamps (arrow's default `T`
/// separator confuses strict clients).
const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";
const TIMESTAMP_TZ_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f%:z";

pub fn pg_type(datatype: &DataType) -> Type {
    match datatype {
        DataType::Boolean => Type::BOOL,
        DataType::Int8 | DataType::Int16 => Type::INT2,
        DataType::Int32 | DataType::UInt8 | DataType::UInt16 => Type::INT4,
        DataType::Int64 | DataType::UInt32 => Type::INT8,
        DataType::UInt64 | DataType::Decimal128(..) | DataType::Decimal256(..) => Type::NUMERIC,
        DataType::Float16 | DataType::Float32 => Type::FLOAT4,
        DataType::Float64 => Type::FLOAT8,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Type::TEXT,
        DataType::Timestamp(_, None) => Type::TIMESTAMP,
        DataType::Timestamp(_, Some(_)) => Type::TIMESTAMPTZ,
        DataType::Date32 | DataType::Date64 => Type::DATE,
        DataType::Time32(_) | DataType::Time64(_) => Type::TIME,
        // Anything exotic degrades to its arrow text rendering.
        _ => Type::TEXT,
    }
}

pub fn field_infos(schema: &Schema) -> Arc<Vec<FieldInfo>> {
    Arc::new(
        schema
            .fields()
            .iter()
            .map(|f| {
                FieldInfo::new(
                    f.name().clone(),
                    None,
                    None,
                    pg_type(f.data_type()),
                    FieldFormat::Text,
                )
            })
            .collect(),
    )
}

/// Buffered batches → a complete simple-protocol query response.
pub fn rows_response(schema: &Schema, batches: &[RecordBatch]) -> PgWireResult<Response> {
    let fields = field_infos(schema);
    let mut rows = Vec::new();
    for batch in batches {
        encode_batch(batch, &fields, &mut rows)?;
    }
    Ok(Response::Query(QueryResponse::new(
        fields,
        stream::iter(rows.into_iter().map(Ok)),
    )))
}

/// One-row, one-text-column response for canned `SHOW` answers.
pub fn canned_response(column: &str, value: &str) -> PgWireResult<Response> {
    let fields = Arc::new(vec![FieldInfo::new(
        column.to_owned(),
        None,
        None,
        Type::TEXT,
        FieldFormat::Text,
    )]);
    let mut encoder = DataRowEncoder::new(Arc::clone(&fields));
    encoder.encode_field(&value)?;
    let row = encoder.take_row();
    Ok(Response::Query(QueryResponse::new(
        fields,
        stream::iter([Ok(row)]),
    )))
}

fn encode_batch(
    batch: &RecordBatch,
    fields: &Arc<Vec<FieldInfo>>,
    rows: &mut Vec<pgwire::messages::data::DataRow>,
) -> PgWireResult<()> {
    let options = FormatOptions::new()
        .with_timestamp_format(Some(TIMESTAMP_FORMAT))
        .with_timestamp_tz_format(Some(TIMESTAMP_TZ_FORMAT))
        .with_date_format(Some("%Y-%m-%d"));
    let formatters = batch
        .columns()
        .iter()
        .map(|col| ArrayFormatter::try_new(col.as_ref(), &options))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PgWireError::ApiError(Box::new(e)))?;

    for row in 0..batch.num_rows() {
        let mut encoder = DataRowEncoder::new(Arc::clone(fields));
        for (col, formatter) in batch.columns().iter().zip(&formatters) {
            if col.is_null(row) {
                encoder.encode_field(&None::<&str>)?;
            } else {
                let text = match col.data_type() {
                    // Postgres booleans are `t`/`f`, not arrow's true/false.
                    DataType::Boolean => {
                        if formatter.value(row).to_string() == "true" {
                            "t".to_owned()
                        } else {
                            "f".to_owned()
                        }
                    }
                    _ => formatter.value(row).to_string(),
                };
                encoder.encode_field(&text.as_str())?;
            }
        }
        rows.push(encoder.take_row());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{
        BooleanArray, Date32Array, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
    };
    use datafusion::arrow::datatypes::{Field, TimeUnit};
    use futures::StreamExt;

    fn text_cells(response: Response) -> Vec<Vec<Option<String>>> {
        let Response::Query(mut query) = response else {
            panic!("expected a query response");
        };
        let mut rows = Vec::new();
        while let Some(row) = futures::executor::block_on(query.data_rows.next()) {
            let row = row.unwrap();
            let mut cells = Vec::new();
            // DataRow buffer: per field, big-endian i32 length (-1 = NULL)
            // then that many value bytes.
            let data: &[u8] = row.data.as_ref();
            let mut off = 0;
            for _ in 0..row.field_count {
                let len = i32::from_be_bytes(data[off..off + 4].try_into().unwrap());
                off += 4;
                if len < 0 {
                    cells.push(None);
                } else {
                    let end = off + len as usize;
                    cells.push(Some(String::from_utf8(data[off..end].to_vec()).unwrap()));
                    off = end;
                }
            }
            rows.push(cells);
        }
        rows
    }

    #[test]
    fn oid_map_covers_the_common_types() {
        assert_eq!(pg_type(&DataType::Utf8), Type::TEXT);
        assert_eq!(pg_type(&DataType::Boolean), Type::BOOL);
        assert_eq!(pg_type(&DataType::Int32), Type::INT4);
        assert_eq!(pg_type(&DataType::Int64), Type::INT8);
        assert_eq!(pg_type(&DataType::Float64), Type::FLOAT8);
        assert_eq!(
            pg_type(&DataType::Timestamp(TimeUnit::Microsecond, None)),
            Type::TIMESTAMP,
        );
        assert_eq!(
            pg_type(&DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into())
            )),
            Type::TIMESTAMPTZ,
        );
        assert_eq!(pg_type(&DataType::Date32), Type::DATE);
        assert_eq!(
            pg_type(&DataType::List(Arc::new(Field::new(
                "x",
                DataType::Int64,
                true
            )))),
            Type::TEXT
        );
    }

    #[test]
    fn rows_encode_as_postgres_text() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
            Field::new("b", DataType::Boolean, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![Some("hello"), None])),
                Arc::new(Int64Array::from(vec![Some(42), None])),
                Arc::new(Float64Array::from(vec![Some(1.5), None])),
                Arc::new(BooleanArray::from(vec![Some(true), Some(false)])),
            ],
        )
        .unwrap();

        let rows = text_cells(rows_response(&schema, &[batch]).unwrap());
        assert_eq!(
            rows,
            vec![
                vec![
                    Some("hello".into()),
                    Some("42".into()),
                    Some("1.5".into()),
                    Some("t".into()),
                ],
                vec![None, None, None, Some("f".into())],
            ],
        );
    }

    #[test]
    fn timestamps_and_dates_use_postgres_shapes() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("d", DataType::Date32, false),
        ]));
        // 2021-01-02 03:04:05.000006 UTC; 2021-01-02 in days.
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![1609556645000006])),
                Arc::new(Date32Array::from(vec![18629])),
            ],
        )
        .unwrap();

        let rows = text_cells(rows_response(&schema, &[batch]).unwrap());
        assert_eq!(
            rows,
            vec![vec![
                Some("2021-01-02 03:04:05.000006".into()),
                Some("2021-01-02".into()),
            ]],
        );
    }

    #[test]
    fn canned_show_answer_is_one_text_row() {
        let rows = text_cells(canned_response("server_version", "15.0").unwrap());
        assert_eq!(rows, vec![vec![Some("15.0".into())]]);
    }
}
