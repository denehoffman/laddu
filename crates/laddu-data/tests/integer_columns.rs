//! Public exact column access, transport, and projection contracts.

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Int8Array, UInt64Array},
    datatypes::{DataType, Field, Schema as ArrowSchema},
    record_batch::RecordBatch,
};
use laddu_data::{
    BatchLayout,
    columns::{Column, ColumnDType, ColumnValue},
    data::{Dataset, EventBatch},
    io::parquet::ParquetSource,
    schema::{Precision, Schema},
};
use parquet::arrow::ArrowWriter;

#[test]
fn typed_batch_row_access_selection_concat_and_memory_use_exact_widths() {
    let schema = Arc::new(
        Schema::new(Vec::<String>::new(), ["x"], false)
            .unwrap()
            .with_columns([("id", ColumnDType::U64), ("run", ColumnDType::I8)])
            .unwrap(),
    );
    let batch = EventBatch::new_with_columns(
        schema.clone(),
        vec![],
        vec![Arc::from([1., 2., 3.])],
        vec![
            Column::U64(Arc::from([u64::MAX - 1, u64::MAX, u64::MAX])),
            Column::I8(Arc::from([-2, 7, 7])),
        ],
        None,
    )
    .unwrap();
    assert_eq!(
        batch.event(1).column_named("id"),
        Some(ColumnValue::U64(u64::MAX))
    );
    assert_eq!(batch.bytes_per_event(), 17);
    assert_eq!(
        BatchLayout::from_schema(&schema)
            .bytes_per_event(Precision::F32)
            .unwrap(),
        13
    );
    let selected = batch.select(&[2, 0]);
    let joined = EventBatch::concat(&[selected.slice(0, 1), selected.slice(1, 2)]).unwrap();
    assert_eq!(
        joined.column_named("run"),
        Some(&Column::I8(Arc::from([7, -2])))
    );
    assert_eq!(
        Dataset::from_batch(joined).column("id").unwrap(),
        Column::U64(Arc::from([u64::MAX, u64::MAX - 1]))
    );
}

#[test]
fn parquet_projection_ignores_unselected_nulls_but_selected_integer_nulls_fail() {
    let path =
        std::env::temp_dir().join(format!("laddu-integer-null-{}.parquet", std::process::id()));
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("unrelated", DataType::Int8, true),
    ]));
    let rb = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![u64::MAX - 1, u64::MAX])) as ArrayRef,
            Arc::new(Int8Array::from(vec![Some(2), None])) as ArrayRef,
        ],
    )
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&rb).unwrap();
    writer.close().unwrap();
    let selected = Arc::new(
        Schema::new(Vec::<String>::new(), Vec::<String>::new(), false)
            .unwrap()
            .with_columns([("id", ColumnDType::U64)])
            .unwrap(),
    );
    let projected = Dataset::new(
        ParquetSource::builder(path.to_str().unwrap())
            .schema(selected)
            .build()
            .unwrap(),
    );
    assert_eq!(
        projected.column("id").unwrap(),
        Column::U64(Arc::from([u64::MAX - 1, u64::MAX]))
    );
    let inferred = Dataset::new(ParquetSource::open(path.to_str().unwrap()).unwrap());
    assert!(
        inferred
            .column("unrelated")
            .unwrap_err()
            .to_string()
            .contains("null")
    );
    std::fs::remove_file(path).unwrap();
}
