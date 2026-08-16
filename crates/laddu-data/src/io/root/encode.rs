use crate::{
    Name,
    schema::{Schema, SchemaWriteOptions, WriteWeightColumn},
};

pub(super) fn root_output_columns(
    schema: &Schema,
    write_weight: WriteWeightColumn,
    options: &SchemaWriteOptions,
) -> Vec<Name> {
    let should_write_weight =
        matches!(write_weight, WriteWeightColumn::Always) || schema.has_weight();

    let mut columns = Vec::with_capacity(
        4 * schema.n_p4s() + schema.n_scalars() + usize::from(should_write_weight),
    );

    for name in schema.p4s() {
        let [e, px, py, pz] = options.column_names.p4_suffixes.physical_p4_names(name);
        columns.push(e.into());
        columns.push(px.into());
        columns.push(py.into());
        columns.push(pz.into());
    }

    for name in schema.scalars() {
        columns.push(name.clone());
    }

    if should_write_weight {
        columns.push(options.column_names.weight_column.clone());
    }

    columns
}
