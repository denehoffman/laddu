use crate::{
    Name,
    schema::{PhysicalSchemaPlan, Schema, SchemaWriteOptions, WriteWeightColumn},
};

pub(super) fn root_output_columns(
    schema: &Schema,
    write_weight: WriteWeightColumn,
    options: &SchemaWriteOptions,
) -> Vec<Name> {
    PhysicalSchemaPlan::for_write(schema, options, write_weight)
        .columns()
        .iter()
        .map(|column| column.name().clone())
        .collect()
}
