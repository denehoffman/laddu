use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Float32Array, Float64Array},
    datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef},
    record_batch::RecordBatch,
};

use crate::{
    LadduDataResult,
    columns::{Column, ColumnDType},
    data::EventBatch,
    io::sink_error,
    schema::{
        PhysicalColumnRole, PhysicalSchemaPlan, Precision, Schema, SchemaWriteOptions,
        WriteWeightColumn,
    },
};

pub(super) fn arrow_schema_from_event_schema(
    schema: &Schema,
    write_weight: WriteWeightColumn,
    options: &SchemaWriteOptions,
) -> ArrowSchema {
    let plan = PhysicalSchemaPlan::for_write(schema, options, write_weight);
    let mut fields = Vec::with_capacity(plan.columns().len());

    let data_type = match options.precision {
        Precision::F64 => DataType::Float64,
        Precision::F32 => DataType::Float32,
    };

    for column in plan.columns() {
        let dtype = match column.role() {
            PhysicalColumnRole::Column { dtype, .. } => integer_arrow_type(dtype),
            _ => data_type.clone(),
        };
        fields.push(Field::new(column.name().as_ref(), dtype, false));
    }

    ArrowSchema::new(fields)
}

fn array_from_iter<I: IntoIterator<Item = f64>>(iter: I, precision: Precision) -> ArrayRef {
    match precision {
        Precision::F64 => Arc::new(Float64Array::from_iter_values(iter)),
        Precision::F32 => Arc::new(Float32Array::from_iter_values(
            iter.into_iter().map(|f| f as f32),
        )),
    }
}

pub(super) fn event_batch_to_record_batch(
    batch: &EventBatch,
    arrow_schema: SchemaRef,
    write_weight: WriteWeightColumn,
    options: &SchemaWriteOptions,
    precision: Precision,
) -> LadduDataResult<RecordBatch> {
    let plan = PhysicalSchemaPlan::for_write(batch.schema(), options, write_weight);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());

    // The Arrow schema passed by the sink is created with the same plan.  The
    // role match below keeps the physical order and the logical-to-physical
    // binding in one place for both schema creation and encoding.
    for column in plan.columns() {
        match column.role() {
            PhysicalColumnRole::Column { index, .. } => {
                columns.push(integer_arrow_array(batch.column(index)))
            }
            PhysicalColumnRole::P4 { index, component } => {
                columns.push(array_from_iter(
                    batch
                        .vec4_column(index)
                        .iter()
                        .map(|x| x.components()[component]),
                    precision,
                ));
            }
            PhysicalColumnRole::Scalar { index } => {
                columns.push(array_from_iter(
                    batch.scalar_column(index).iter().copied(),
                    precision,
                ));
            }
            PhysicalColumnRole::Weight => {
                columns.push(array_from_iter(
                    (0..batch.len()).map(|i| batch.weights_at(i)),
                    precision,
                ));
            }
        }
    }

    RecordBatch::try_new(arrow_schema, columns)
        .map_err(|error| sink_error("assemble Parquet batch", "record batch", error))
}

fn integer_arrow_type(dtype: ColumnDType) -> DataType {
    match dtype {
        ColumnDType::I8 => DataType::Int8,
        ColumnDType::U8 => DataType::UInt8,
        ColumnDType::I16 => DataType::Int16,
        ColumnDType::U16 => DataType::UInt16,
        ColumnDType::I32 => DataType::Int32,
        ColumnDType::U32 => DataType::UInt32,
        ColumnDType::I64 => DataType::Int64,
        ColumnDType::U64 => DataType::UInt64,
    }
}
fn integer_arrow_array(column: &Column) -> ArrayRef {
    match column {
        Column::I8(values) => Arc::new(arrow::array::Int8Array::from_iter_values(
            values.iter().copied(),
        )),
        Column::U8(values) => Arc::new(arrow::array::UInt8Array::from_iter_values(
            values.iter().copied(),
        )),
        Column::I16(values) => Arc::new(arrow::array::Int16Array::from_iter_values(
            values.iter().copied(),
        )),
        Column::U16(values) => Arc::new(arrow::array::UInt16Array::from_iter_values(
            values.iter().copied(),
        )),
        Column::I32(values) => Arc::new(arrow::array::Int32Array::from_iter_values(
            values.iter().copied(),
        )),
        Column::U32(values) => Arc::new(arrow::array::UInt32Array::from_iter_values(
            values.iter().copied(),
        )),
        Column::I64(values) => Arc::new(arrow::array::Int64Array::from_iter_values(
            values.iter().copied(),
        )),
        Column::U64(values) => Arc::new(arrow::array::UInt64Array::from_iter_values(
            values.iter().copied(),
        )),
    }
}
