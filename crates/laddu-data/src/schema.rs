use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{LadduDataError, LadduDataResult, Name};

/// Logical names and lookup tables for four-momentum, scalar, and weight columns.
#[derive(Clone, Debug)]
pub struct Schema {
    p4s: Vec<Name>,
    scalars: Vec<Name>,
    has_weight: bool,

    p4_index: Arc<HashMap<Name, usize>>,
    scalar_index: Arc<HashMap<Name, usize>>,
}

impl PartialEq for Schema {
    fn eq(&self, other: &Self) -> bool {
        self.p4s == other.p4s
            && self.scalars == other.scalars
            && self.has_weight == other.has_weight
    }
}

impl Schema {
    /// Validates and constructs a logical schema.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when four-momentum or scalar column names are
    /// duplicated.
    pub fn new(
        p4s: impl IntoIterator<Item = impl Into<Name>>,
        scalars: impl IntoIterator<Item = impl Into<Name>>,
        has_weight: bool,
    ) -> LadduDataResult<Self> {
        let p4s: Vec<Name> = p4s.into_iter().map(Into::into).collect();
        let scalars: Vec<Name> = scalars.into_iter().map(Into::into).collect();
        let p4_index = Arc::new(make_index(&p4s, "p4")?);
        let scalar_index = Arc::new(make_index(&scalars, "scalar")?);
        Ok(Self {
            p4s,
            scalars,
            has_weight,
            p4_index,
            scalar_index,
        })
    }

    /// Returns the four-momentum column index for `name`.
    pub fn p4_index(&self, name: &str) -> Option<usize> {
        self.p4_index.get(name).copied()
    }

    /// Returns the scalar column index for `name`.
    pub fn scalar_index(&self, name: &str) -> Option<usize> {
        self.scalar_index.get(name).copied()
    }

    /// Returns four-momentum names in column order.
    pub fn p4s(&self) -> &[Name] {
        &self.p4s
    }

    /// Returns scalar names in column order.
    pub fn scalars(&self) -> &[Name] {
        &self.scalars
    }

    /// Returns whether events carry explicit weights.
    pub fn has_weight(&self) -> bool {
        self.has_weight
    }

    /// Returns the number of four-momentum columns.
    pub fn n_p4s(&self) -> usize {
        self.p4s.len()
    }

    /// Returns the number of scalar columns.
    pub fn n_scalars(&self) -> usize {
        self.scalars.len()
    }

    /// Requires and returns a four-momentum column index.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError::MissingColumn`] when `name` is not a
    /// four-momentum column.
    pub fn require_p4(&self, name: &str) -> LadduDataResult<usize> {
        self.p4_index(name)
            .ok_or_else(|| LadduDataError::MissingColumn(Name::from(name)))
    }

    /// Requires and returns a scalar column index.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError::MissingColumn`] when `name` is not a scalar
    /// column.
    pub fn require_scalar(&self, name: &str) -> LadduDataResult<usize> {
        self.scalar_index(name)
            .ok_or_else(|| LadduDataError::MissingColumn(Name::from(name)))
    }
}

fn make_index(names: &[Name], kind: &'static str) -> LadduDataResult<HashMap<Name, usize>> {
    let mut out = HashMap::with_capacity(names.len());
    for (i, name) in names.iter().cloned().enumerate() {
        if out.insert(name.clone(), i).is_some() {
            return Err(LadduDataError::Schema(format!(
                "duplicate {kind} column: {name}"
            )));
        }
    }
    Ok(out)
}

/// Physical naming conventions used to map a logical schema to storage columns.
#[derive(Clone, Debug)]
pub struct SchemaColumnNames {
    /// Physical weight-column name.
    pub weight_column: Name,
    /// Suffixes for four-momentum components.
    pub p4_suffixes: P4Suffixes,
}

impl Default for SchemaColumnNames {
    fn default() -> Self {
        Self {
            weight_column: Name::from("weight"),
            p4_suffixes: P4Suffixes::default(),
        }
    }
}

/// Options controlling logical schema inference from physical columns.
#[derive(Clone, Debug)]
pub struct SchemaInferenceOptions {
    /// Physical naming conventions.
    pub column_names: SchemaColumnNames,
    /// Whether inference fails if the weight column is absent.
    pub require_weight: bool,
    /// Whether incomplete four-momenta become independent scalar columns.
    pub incomplete_p4_components_are_scalars: bool,
}

impl Default for SchemaInferenceOptions {
    fn default() -> Self {
        Self {
            column_names: SchemaColumnNames::default(),
            require_weight: false,
            incomplete_p4_components_are_scalars: true,
        }
    }
}

/// Physical suffixes for `(E, px, py, pz)` columns.
#[derive(Clone, Debug)]
pub struct P4Suffixes {
    /// Energy suffix.
    pub e: &'static str,
    /// X-momentum suffix.
    pub px: &'static str,
    /// Y-momentum suffix.
    pub py: &'static str,
    /// Z-momentum suffix.
    pub pz: &'static str,
}

impl Default for P4Suffixes {
    fn default() -> Self {
        Self {
            e: "_e",
            px: "_px",
            py: "_py",
            pz: "_pz",
        }
    }
}

impl P4Suffixes {
    /// Splits a matching physical name into logical prefix and component index.
    pub fn component<'a>(&'a self, name: &'a str) -> Option<(&'a str, usize)> {
        if let Some(prefix) = name.strip_suffix(self.e) {
            Some((prefix, 0))
        } else if let Some(prefix) = name.strip_suffix(self.px) {
            Some((prefix, 1))
        } else if let Some(prefix) = name.strip_suffix(self.py) {
            Some((prefix, 2))
        } else if let Some(prefix) = name.strip_suffix(self.pz) {
            Some((prefix, 3))
        } else {
            None
        }
    }

    /// Produces the four physical names for a logical four-momentum prefix.
    pub fn physical_p4_names(&self, prefix: &str) -> [String; 4] {
        [
            format!("{prefix}{}", self.e),
            format!("{prefix}{}", self.px),
            format!("{prefix}{}", self.py),
            format!("{prefix}{}", self.pz),
        ]
    }
}

/// Physical storage type relevant to schema inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    /// 64-bit floating point.
    F64,
    /// 32-bit floating point.
    F32,
    /// Any unsupported type.
    Other,
}

impl ColumnType {
    /// Returns whether this type can populate an event-data column.
    pub fn is_supported_float(self) -> bool {
        matches!(self, Self::F64 | Self::F32)
    }
}

/// Name and physical type of one available storage column.
#[derive(Clone, Copy, Debug)]
pub struct ColumnInfo<'a> {
    /// Physical column name.
    pub name: &'a str,
    /// Physical column type.
    pub dtype: ColumnType,
}

impl Schema {
    /// Infers a logical schema from available physical columns.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when required momentum components or weights
    /// are missing, names are ambiguous, or inferred logical names conflict.
    pub fn infer_from_columns<'a>(
        columns: impl IntoIterator<Item = ColumnInfo<'a>>,
        options: &SchemaInferenceOptions,
    ) -> LadduDataResult<Self> {
        let mut p4_candidates = BTreeMap::<String, [bool; 4]>::new();
        let mut scalar_names = Vec::<Name>::new();
        let mut has_weight = false;

        for col in columns {
            if !col.dtype.is_supported_float() {
                continue;
            }

            if col.name == options.column_names.weight_column.as_ref() {
                has_weight = true;
                continue;
            }

            if let Some((prefix, component)) = options.column_names.p4_suffixes.component(col.name)
            {
                p4_candidates.entry(prefix.to_owned()).or_default()[component] = true;
            } else {
                scalar_names.push(Name::from(col.name));
            }
        }

        let mut p4s = Vec::<Name>::new();

        for (prefix, seen) in p4_candidates {
            if seen == [true, true, true, true] {
                p4s.push(Name::from(prefix));
            } else if options.incomplete_p4_components_are_scalars {
                let names = options.column_names.p4_suffixes.physical_p4_names(&prefix);

                for (i, name) in names.into_iter().enumerate() {
                    if seen[i] {
                        scalar_names.push(Name::from(name));
                    }
                }
            }
        }

        if options.require_weight && !has_weight {
            return Err(LadduDataError::MissingColumn(Arc::clone(
                &options.column_names.weight_column,
            )));
        }

        Schema::new(p4s, scalar_names, has_weight)
    }

    /// Returns all physical columns required to store this schema.
    pub fn physical_columns(&self, column_names: &SchemaColumnNames) -> Vec<Name> {
        PhysicalSchemaPlan::for_read(self, column_names)
            .columns()
            .iter()
            .map(|column| Arc::clone(column.name()))
            .collect()
    }

    /// Validates that all physical columns required by this schema are available.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError::MissingColumn`] when a required physical
    /// column is absent or has an unsupported type.
    pub fn validate_required_columns<'a>(
        &self,
        available: impl IntoIterator<Item = ColumnInfo<'a>>,
        options: &SchemaInferenceOptions,
    ) -> LadduDataResult<()> {
        let available: HashSet<&str> = available
            .into_iter()
            .filter(|c| c.dtype.is_supported_float())
            .map(|c| c.name)
            .collect();

        for required in PhysicalSchemaPlan::for_read(self, &options.column_names).columns() {
            if !available.contains(required.name().as_ref()) {
                return Err(LadduDataError::MissingColumn(Arc::clone(required.name())));
            }
        }

        Ok(())
    }
}

/// Floating-point precision used when writing physical columns.
#[derive(Copy, Clone, Debug, Default)]
pub enum Precision {
    /// Write 64-bit floating-point values.
    #[default]
    F64,
    /// Write 32-bit floating-point values.
    F32,
}

/// Policy controlling whether sinks emit a weight column.
#[derive(Clone, Copy, Debug, Default)]
pub enum WriteWeightColumn {
    /// Always write weights, using unit weights when the schema has none.
    #[default]
    Always,
    /// Write weights only when the logical schema contains them.
    OnlyIfPresent,
}

/// Physical naming, precision, and weight policy for event sinks.
#[derive(Clone, Debug, Default)]
pub struct SchemaWriteOptions {
    /// Physical column naming conventions.
    pub column_names: SchemaColumnNames,
    /// Floating-point output precision.
    pub precision: Precision,
    /// Weight-column emission policy.
    pub write_weight_column: WriteWeightColumn,
}

/// The semantic role of one physical storage column.
///
/// This is deliberately crate-private: callers work with logical [`Schema`]
/// values while the file-format adapters use this plan to keep physical
/// column ordering and binding identical across backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PhysicalColumnRole {
    /// One component of a logical four-momentum column.
    P4 {
        /// Logical four-momentum column index.
        index: usize,
        /// Component index in `(E, px, py, pz)` order.
        component: usize,
    },
    /// One logical scalar column.
    Scalar {
        /// Logical scalar column index.
        index: usize,
    },
    /// The physical event-weight column.
    Weight,
}

/// Ordered physical representation of a logical event schema.
///
/// A plan is built once at each storage boundary and then consumed by schema
/// creation, projection, encoding, and decoding.  Keeping the role alongside
/// the name avoids each backend reimplementing the canonical ordering.
#[derive(Clone, Debug)]
pub(crate) struct PhysicalSchemaPlan {
    columns: Vec<PhysicalColumn>,
}

#[derive(Clone, Debug)]
pub(crate) struct PhysicalColumn {
    name: Name,
    role: PhysicalColumnRole,
}

impl PhysicalSchemaPlan {
    /// Builds the physical columns used when reading or validating a schema.
    pub(crate) fn for_read(schema: &Schema, column_names: &SchemaColumnNames) -> Self {
        Self::build(schema, column_names, schema.has_weight())
    }

    /// Builds the physical columns emitted by a sink.
    pub(crate) fn for_write(
        schema: &Schema,
        options: &SchemaWriteOptions,
        write_weight: WriteWeightColumn,
    ) -> Self {
        let should_write_weight =
            matches!(write_weight, WriteWeightColumn::Always) || schema.has_weight();
        Self::build(schema, &options.column_names, should_write_weight)
    }

    fn build(schema: &Schema, column_names: &SchemaColumnNames, include_weight: bool) -> Self {
        let mut columns = Vec::with_capacity(
            4 * schema.n_p4s() + schema.n_scalars() + usize::from(include_weight),
        );

        for (index, p4) in schema.p4s().iter().enumerate() {
            for (component, name) in column_names
                .p4_suffixes
                .physical_p4_names(p4)
                .into_iter()
                .enumerate()
            {
                columns.push(PhysicalColumn {
                    name: Name::from(name),
                    role: PhysicalColumnRole::P4 { index, component },
                });
            }
        }

        for (index, name) in schema.scalars().iter().cloned().enumerate() {
            columns.push(PhysicalColumn {
                name,
                role: PhysicalColumnRole::Scalar { index },
            });
        }

        if include_weight {
            columns.push(PhysicalColumn {
                name: Arc::clone(&column_names.weight_column),
                role: PhysicalColumnRole::Weight,
            });
        }

        Self { columns }
    }

    /// Returns physical columns in canonical storage order.
    pub(crate) fn columns(&self) -> &[PhysicalColumn] {
        &self.columns
    }
}

impl PhysicalColumn {
    /// Returns the physical storage name.
    pub(crate) fn name(&self) -> &Name {
        &self.name
    }

    /// Returns the logical role of this physical column.
    pub(crate) fn role(&self) -> PhysicalColumnRole {
        self.role
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(name: &'static str, dtype: ColumnType) -> ColumnInfo<'static> {
        ColumnInfo { name, dtype }
    }

    #[test]
    fn schema_new_rejects_duplicates_and_required_lookup_reports_missing_column() {
        let duplicate_p4 = Schema::new(["p", "p"], ["mass"], false);
        assert!(matches!(duplicate_p4, Err(LadduDataError::Schema(_))));

        let duplicate_scalar = Schema::new(["p"], ["mass", "mass"], false);
        assert!(matches!(duplicate_scalar, Err(LadduDataError::Schema(_))));

        let schema = Schema::new(["beam", "recoil"], ["mass", "costheta"], true).unwrap();

        assert_eq!(schema.require_p4("recoil").unwrap(), 1);
        assert_eq!(schema.require_scalar("costheta").unwrap(), 1);

        let err = schema.require_scalar("missing").unwrap_err();
        assert!(matches!(err, LadduDataError::MissingColumn(name) if name.as_ref() == "missing"));
    }

    #[test]
    fn infer_from_columns_groups_complete_p4s_keeps_incomplete_components_as_scalars_and_ignores_nonfloats()
     {
        let options = SchemaInferenceOptions::default();

        let schema = Schema::infer_from_columns(
            [
                col("gamma_px", ColumnType::F64),
                col("gamma_py", ColumnType::F64),
                col("gamma_pz", ColumnType::F32),
                col("gamma_e", ColumnType::F64),
                col("partial_px", ColumnType::F64),
                col("partial_e", ColumnType::F64),
                col("mass", ColumnType::F32),
                col("ignored", ColumnType::Other),
                col("weight", ColumnType::F64),
            ],
            &options,
        )
        .unwrap();

        assert_eq!(
            schema
                .p4s()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>(),
            vec!["gamma"]
        );

        assert_eq!(
            schema
                .scalars()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>(),
            vec!["mass", "partial_e", "partial_px"]
        );

        assert!(schema.has_weight());
    }

    #[test]
    fn infer_from_columns_can_discard_incomplete_p4_components_and_require_weight() {
        let options = SchemaInferenceOptions {
            incomplete_p4_components_are_scalars: false,
            ..Default::default()
        };

        let schema = Schema::infer_from_columns(
            [
                col("partial_px", ColumnType::F64),
                col("partial_e", ColumnType::F64),
                col("mass", ColumnType::F64),
                col("weight", ColumnType::F64),
            ],
            &options,
        )
        .unwrap();

        assert!(schema.p4s().is_empty());
        assert_eq!(
            schema
                .scalars()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>(),
            vec!["mass"]
        );

        let require_weight = SchemaInferenceOptions {
            require_weight: true,
            ..Default::default()
        };

        let err = Schema::infer_from_columns([col("mass", ColumnType::F64)], &require_weight)
            .unwrap_err();

        assert!(matches!(err, LadduDataError::MissingColumn(name) if name.as_ref() == "weight"));
    }

    #[test]
    fn physical_columns_and_validation_respect_custom_names_and_float_types_only() {
        let schema = Schema::new(["p"], ["mass"], true).unwrap();

        let names = SchemaColumnNames {
            weight_column: Name::from("event_weight"),
            ..Default::default()
        };

        let physical = schema
            .physical_columns(&names)
            .into_iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            physical,
            vec!["p_e", "p_px", "p_py", "p_pz", "mass", "event_weight"]
        );

        let options = SchemaInferenceOptions {
            column_names: names,
            ..Default::default()
        };

        let ok = schema.validate_required_columns(
            [
                col("p_e", ColumnType::F32),
                col("p_px", ColumnType::F64),
                col("p_py", ColumnType::F64),
                col("p_pz", ColumnType::F32),
                col("mass", ColumnType::F64),
                col("event_weight", ColumnType::F64),
            ],
            &options,
        );

        assert!(ok.is_ok());

        let missing_because_not_float = schema
            .validate_required_columns(
                [
                    col("p_e", ColumnType::F32),
                    col("p_px", ColumnType::F64),
                    col("p_py", ColumnType::Other),
                    col("p_pz", ColumnType::F32),
                    col("mass", ColumnType::F64),
                    col("event_weight", ColumnType::F64),
                ],
                &options,
            )
            .unwrap_err();

        assert!(
            matches!(missing_because_not_float, LadduDataError::MissingColumn(name) if name.as_ref() == "p_py")
        );
    }

    #[test]
    fn physical_schema_plan_preserves_order_roles_and_weight_policy() {
        let schema = Schema::new(["p"], ["mass"], false).unwrap();
        let options = SchemaWriteOptions {
            column_names: SchemaColumnNames {
                weight_column: Name::from("event_weight"),
                ..Default::default()
            },
            ..Default::default()
        };

        let only_if_present =
            PhysicalSchemaPlan::for_write(&schema, &options, WriteWeightColumn::OnlyIfPresent);
        assert_eq!(
            only_if_present
                .columns()
                .iter()
                .map(|column| column.name().to_string())
                .collect::<Vec<_>>(),
            ["p_e", "p_px", "p_py", "p_pz", "mass"]
        );
        assert_eq!(
            only_if_present
                .columns()
                .iter()
                .map(PhysicalColumn::role)
                .collect::<Vec<_>>(),
            [
                PhysicalColumnRole::P4 {
                    index: 0,
                    component: 0,
                },
                PhysicalColumnRole::P4 {
                    index: 0,
                    component: 1,
                },
                PhysicalColumnRole::P4 {
                    index: 0,
                    component: 2,
                },
                PhysicalColumnRole::P4 {
                    index: 0,
                    component: 3,
                },
                PhysicalColumnRole::Scalar { index: 0 },
            ]
        );

        let always = PhysicalSchemaPlan::for_write(&schema, &options, WriteWeightColumn::Always);
        assert_eq!(
            always.columns().last().map(|column| column.name().as_ref()),
            Some("event_weight")
        );
        assert_eq!(
            always.columns().last().map(PhysicalColumn::role),
            Some(PhysicalColumnRole::Weight)
        );
    }
}
