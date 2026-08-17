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
    io::{SliceBatchIter, source_error},
    schema::{
        ColumnInfo, ColumnType, PhysicalColumnRole, PhysicalSchemaPlan, Schema, SchemaColumnNames,
    },
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
    let resource = key.file.as_ref().display().to_string();
    let file = File::open(key.file.as_ref())
        .map_err(|e| source_error("open Parquet file", &resource, e))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| source_error("read Parquet metadata", &resource, e))?
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
        .map_err(|e| source_error("build Parquet reader", &resource, e))?;
    let batches = reader.map(move |rb| {
        let rb = rb.map_err(|e| source_error("read Parquet batch", &resource, e))?;
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
    for column in PhysicalSchemaPlan::for_read(schema, column_names).columns() {
        let index = arrow_schema
            .index_of(column.name().as_ref())
            .map_err(|_| LadduDataError::MissingColumn(column.name().clone()))?;
        indices.push(index);
    }
    indices.sort_unstable();
    indices.dedup();
    Ok(ProjectionMask::roots(parquet_schema, indices))
}

pub(super) fn parquet_arrow_schema(path: &Path) -> LadduDataResult<SchemaRef> {
    let resource = path.display().to_string();
    let file = File::open(path).map_err(|e| source_error("open Parquet file", &resource, e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| source_error("read Parquet metadata", &resource, e))?;
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
    let plan = PhysicalSchemaPlan::for_read(&schema, &options.schema_inference.column_names);
    let mut p4s: Vec<[Option<Vec<f64>>; 4]> = (0..schema.n_p4s())
        .map(|_| std::array::from_fn(|_| None))
        .collect();
    let mut scalars: Vec<Option<Vec<f64>>> = (0..schema.n_scalars()).map(|_| None).collect();
    let mut weights = None;

    for column in plan.columns() {
        let values = read_f64_column(&rb, &arrow_schema, column.name().as_ref(), options)?;
        match column.role() {
            PhysicalColumnRole::P4 { index, component } => {
                p4s[index][component] = Some(values);
            }
            PhysicalColumnRole::Scalar { index } => {
                scalars[index] = Some(values);
            }
            PhysicalColumnRole::Weight => {
                weights = Some(values);
            }
        }
    }

    let mut p4_columns = Vec::with_capacity(p4s.len());
    for parts in p4s {
        let [e, px, py, pz] = parts;
        let (Some(e), Some(px), Some(py), Some(pz)) = (e, px, py, pz) else {
            return Err(LadduDataError::Source(
                "physical schema plan did not bind all four-momentum components".into(),
            ));
        };
        let col: Arc<[RealVec4]> = (0..rb.num_rows())
            .map(|i| RealVec4 {
                e: e[i],
                px: px[i],
                py: py[i],
                pz: pz[i],
            })
            .collect();
        p4_columns.push(col);
    }

    let scalar_columns = scalars
        .into_iter()
        .map(|values| {
            values.ok_or_else(|| {
                LadduDataError::Source("physical schema plan did not bind a scalar column".into())
            })
        })
        .collect::<LadduDataResult<Vec<_>>>()?;

    BatchAssembler::from_columns(
        schema,
        p4_columns,
        scalar_columns.into_iter().map(Into::into).collect(),
        weights.map(Into::into),
    )
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
                .ok_or_else(|| {
                    source_error(
                        "decode Parquet column",
                        name,
                        format!("failed to read {name} as f64"),
                    )
                })?;
            collect_numeric(array, name, options.null_handling)
        }
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    source_error(
                        "decode Parquet column",
                        name,
                        format!("failed to read {name} as f32"),
                    )
                })?;
            collect_numeric(array, name, options.null_handling)
        }
        other => Err(source_error(
            "decode Parquet column",
            name,
            format!("column {name} has unsupported type {other:?}"),
        )),
    }
}

/// Borrowed numeric view shared by Arrow's supported floating-point arrays.
///
/// Parquet keeps the concrete Arrow array so null bitmaps and values remain
/// zero-copy until this boundary. The adapter only centralizes the conversion
/// to the logical `f64` representation used by [`EventBatch`].
trait NumericArray {
    fn len(&self) -> usize;

    fn is_null(&self, index: usize) -> bool;

    fn value_as_f64(&self, index: usize) -> f64;
}

impl NumericArray for Float64Array {
    fn len(&self) -> usize {
        Array::len(self)
    }

    fn is_null(&self, index: usize) -> bool {
        Array::is_null(self, index)
    }

    fn value_as_f64(&self, index: usize) -> f64 {
        self.value(index)
    }
}

impl NumericArray for Float32Array {
    fn len(&self) -> usize {
        Array::len(self)
    }

    fn is_null(&self, index: usize) -> bool {
        Array::is_null(self, index)
    }

    fn value_as_f64(&self, index: usize) -> f64 {
        self.value(index) as f64
    }
}

fn collect_numeric(
    array: &dyn NumericArray,
    name: &str,
    nulls: NullHandling,
) -> LadduDataResult<Vec<f64>> {
    let mut out = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            match nulls {
                NullHandling::Error => {
                    return Err(source_error(
                        "decode Parquet column",
                        name,
                        format!("null in column {name}"),
                    ));
                }
                NullHandling::NaN => out.push(f64::NAN),
            }
        } else {
            out.push(array.value_as_f64(i));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, Float32Array, Float64Array, Int32Array},
        datatypes::{DataType, Field, Schema as ArrowSchema},
        record_batch::RecordBatch,
    };

    use super::*;

    #[test]
    fn numeric_collector_handles_precision_nulls_and_non_finite_values() {
        let f64_values =
            Float64Array::from(vec![Some(1.25), None, Some(f64::NAN), Some(f64::INFINITY)]);

        let error = collect_numeric(&f64_values, "value", NullHandling::Error).unwrap_err();
        assert!(matches!(
            error,
            LadduDataError::Source(message)
                if message == "decode Parquet column `value`: null in column value"
        ));

        let values = collect_numeric(&f64_values, "value", NullHandling::NaN).unwrap();
        assert_eq!(values[0], 1.25);
        assert!(values[1].is_nan());
        assert!(values[2].is_nan());
        assert!(values[3].is_infinite());

        let f32_values = Float32Array::from(vec![
            Some(f32::MAX),
            Some(f32::NAN),
            Some(f32::NEG_INFINITY),
            None,
        ]);
        let values = collect_numeric(&f32_values, "value", NullHandling::NaN).unwrap();
        assert_eq!(values[0], f32::MAX as f64);
        assert!(values[1].is_nan());
        assert!(values[2].is_infinite() && values[2].is_sign_negative());
        assert!(values[3].is_nan());
    }

    #[test]
    fn unsupported_arrow_type_preserves_cause_and_context() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef],
        )
        .unwrap();

        let error =
            read_f64_column(&batch, &schema, "value", &ParquetReadOptions::default()).unwrap_err();
        assert!(matches!(
            error,
            LadduDataError::Source(message)
                if message.contains("decode Parquet column `value`")
                    && message.contains("unsupported type Int32")
        ));
    }
}
