use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Float32Array, Float64Array},
    datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef},
    record_batch::RecordBatch,
};

use crate::{
    LadduDataError, LadduDataResult,
    data::EventBatch,
    schema::{Precision, Schema, SchemaWriteOptions, WriteWeightColumn},
};

pub(super) fn arrow_schema_from_event_schema(
    schema: &Schema,
    write_weight: WriteWeightColumn,
    options: &SchemaWriteOptions,
) -> ArrowSchema {
    let should_write_weight =
        matches!(write_weight, WriteWeightColumn::Always) || schema.has_weight();

    let mut fields = Vec::with_capacity(
        4 * schema.n_p4s() + schema.n_scalars() + usize::from(should_write_weight),
    );

    let data_type = match options.precision {
        Precision::F64 => DataType::Float64,
        Precision::F32 => DataType::Float32,
    };

    for name in schema.p4s() {
        let [e, px, py, pz] = options.column_names.p4_suffixes.physical_p4_names(name);
        fields.push(Field::new(e, data_type.clone(), false));
        fields.push(Field::new(px, data_type.clone(), false));
        fields.push(Field::new(py, data_type.clone(), false));
        fields.push(Field::new(pz, data_type.clone(), false));
    }

    for name in schema.scalars() {
        fields.push(Field::new(name.as_ref(), data_type.clone(), false));
    }

    if should_write_weight {
        fields.push(Field::new(
            options.column_names.weight_column.as_ref(),
            data_type,
            false,
        ));
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
    precision: Precision,
) -> LadduDataResult<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());

    for col in 0..batch.schema().n_p4s() {
        let p = batch.vec4_column(col);

        for component in 0..4 {
            columns.push(array_from_iter(
                p.iter().map(|x| x.components()[component]),
                precision,
            ));
        }
    }

    for col in 0..batch.schema().n_scalars() {
        columns.push(array_from_iter(
            batch.scalar_column(col).iter().copied(),
            precision,
        ));
    }

    let should_write_weight =
        matches!(write_weight, WriteWeightColumn::Always) || batch.schema().has_weight();

    if should_write_weight {
        columns.push(array_from_iter(
            (0..batch.len()).map(|i| batch.weights_at(i)),
            precision,
        ));
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| LadduDataError::Sink(e.to_string()))
}
