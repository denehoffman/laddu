use std::{fs::File, path::Path, sync::Arc};

use arrow::{
    array::{Array, Float32Array, Float64Array},
    datatypes::{DataType, Schema as ArrowSchema, SchemaRef},
    record_batch::RecordBatch,
};
use laddu_physics::vectors::RealVec4;
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    schema::types::SchemaDescriptor,
};

use crate::{
    LadduDataError, LadduDataResult, Name,
    data::{BatchAssembler, EventBatch},
    io::SliceBatchIter,
    schema::{ColumnInfo, ColumnType, Schema, SchemaColumnNames},
};

use super::{NullHandling, ParquetFragmentKey, ParquetReadOptions};

pub(super) fn open_parquet_fragment_reader(
    schema: Arc<Schema>,
    options: ParquetReadOptions,
    key: ParquetFragmentKey,
    local_start: usize,
    local_len: usize,
    chunk_size: Option<usize>,
) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
    let file = File::open(key.file.as_ref()).map_err(|e| LadduDataError::Source(e.to_string()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LadduDataError::Source(e.to_string()))?
        .with_row_groups(vec![key.row_group]);
    let projection = parquet_projection_for_schema(
        builder.parquet_schema(),
        builder.schema().as_ref(),
        &schema,
        &options.schema_inference.column_names,
    )?;
    builder = builder.with_projection(projection);
    if matches!(chunk_size, Some(0)) {
        return Err(LadduDataError::InvalidArgument(
            "chunk_size must be nonzero",
        ));
    }
    let end = local_start
        .checked_add(local_len)
        .ok_or(LadduDataError::InvalidArgument(
            "slice range overflows usize",
        ))?;
    let batch_size = chunk_size.unwrap_or(end.max(1));
    builder = builder.with_batch_size(batch_size);
    let reader = builder
        .build()
        .map_err(|e| LadduDataError::Source(e.to_string()))?;
    let batches = reader.map(move |rb| {
        let rb = rb.map_err(|e| LadduDataError::Source(e.to_string()))?;
        record_batch_to_event_batch(rb, Arc::clone(&schema), &options)
    });
    Ok(Box::new(SliceBatchIter::new(
        batches,
        local_start,
        local_len,
    )?))
}

pub(super) fn parquet_projection_for_schema(
    parquet_schema: &SchemaDescriptor,
    arrow_schema: &ArrowSchema,
    schema: &Schema,
    column_names: &SchemaColumnNames,
) -> LadduDataResult<ProjectionMask> {
    let mut indices = Vec::new();
    for name in schema.physical_columns(column_names) {
        let index = arrow_schema
            .index_of(name.as_ref())
            .map_err(|_| LadduDataError::MissingColumn(name))?;
        indices.push(index);
    }
    indices.sort_unstable();
    indices.dedup();
    Ok(ProjectionMask::roots(parquet_schema, indices))
}

pub(super) fn parquet_arrow_schema(path: &Path) -> LadduDataResult<SchemaRef> {
    let file = File::open(path).map_err(|e| LadduDataError::Source(e.to_string()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LadduDataError::Source(e.to_string()))?;
    Ok(builder.schema().clone())
}

pub(super) fn arrow_columns(schema: &ArrowSchema) -> impl Iterator<Item = ColumnInfo<'_>> {
    schema.fields().iter().map(|field| ColumnInfo {
        name: field.name().as_str(),
        dtype: arrow_column_type(field.data_type()),
    })
}

fn arrow_column_type(data_type: &DataType) -> ColumnType {
    match data_type {
        DataType::Float64 => ColumnType::F64,
        DataType::Float32 => ColumnType::F32,
        _ => ColumnType::Other,
    }
}

pub(super) fn record_batch_to_event_batch(
    rb: RecordBatch,
    schema: Arc<Schema>,
    options: &ParquetReadOptions,
) -> LadduDataResult<EventBatch> {
    let arrow_schema = rb.schema();
    let mut p4s: Vec<Arc<[RealVec4]>> = Vec::with_capacity(schema.n_p4s());
    let mut scalars: Vec<Arc<[f64]>> = Vec::with_capacity(schema.n_scalars());
    for name in schema.p4s() {
        let [e_name, px_name, py_name, pz_name] = options
            .schema_inference
            .column_names
            .p4_suffixes
            .physical_p4_names(name);
        let e = read_f64_column(&rb, &arrow_schema, &e_name, options)?;
        let px = read_f64_column(&rb, &arrow_schema, &px_name, options)?;
        let py = read_f64_column(&rb, &arrow_schema, &py_name, options)?;
        let pz = read_f64_column(&rb, &arrow_schema, &pz_name, options)?;
        let col: Arc<[RealVec4]> = (0..rb.num_rows())
            .map(|i| RealVec4 {
                e: e[i],
                px: px[i],
                py: py[i],
                pz: pz[i],
            })
            .collect();
        p4s.push(col);
    }
    for name in schema.scalars() {
        scalars.push(read_f64_column(&rb, &arrow_schema, name, options)?.into());
    }
    let weights: Option<Arc<[f64]>> = if schema.has_weight() {
        Some(
            read_f64_column(
                &rb,
                &arrow_schema,
                &options.schema_inference.column_names.weight_column,
                options,
            )?
            .into(),
        )
    } else {
        None
    };
    BatchAssembler::from_columns(schema, p4s, scalars, weights)
}

fn read_f64_column(
    rb: &RecordBatch,
    arrow_schema: &SchemaRef,
    name: &str,
    options: &ParquetReadOptions,
) -> LadduDataResult<Vec<f64>> {
    let index = arrow_schema
        .index_of(name)
        .map_err(|_| LadduDataError::MissingColumn(Name::from(name)))?;
    let array = rb.column(index);
    match array.data_type() {
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| LadduDataError::Source(format!("failed to read {name} as f64")))?;
            collect_f64(array, name, options.null_handling)
        }
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| LadduDataError::Source(format!("failed to read {name} as f32")))?;
            collect_f32(array, name, options.null_handling)
        }
        other => Err(LadduDataError::Source(format!(
            "column {name} has unsupported type {other:?}"
        ))),
    }
}

fn collect_f64(array: &Float64Array, name: &str, nulls: NullHandling) -> LadduDataResult<Vec<f64>> {
    let mut out = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            match nulls {
                NullHandling::Error => {
                    return Err(LadduDataError::Source(format!("null in column {name}")));
                }
                NullHandling::NaN => out.push(f64::NAN),
            }
        } else {
            out.push(array.value(i));
        }
    }
    Ok(out)
}

fn collect_f32(array: &Float32Array, name: &str, nulls: NullHandling) -> LadduDataResult<Vec<f64>> {
    let mut out = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            match nulls {
                NullHandling::Error => {
                    return Err(LadduDataError::Source(format!("null in column {name}")));
                }
                NullHandling::NaN => out.push(f64::NAN),
            }
        } else {
            out.push(array.value(i) as f64);
        }
    }
    Ok(out)
}
