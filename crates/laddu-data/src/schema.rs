use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{LadduDataError, LadduDataResult, Name};

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

    pub fn p4_index(&self, name: &str) -> Option<usize> {
        self.p4_index.get(name).copied()
    }

    pub fn scalar_index(&self, name: &str) -> Option<usize> {
        self.scalar_index.get(name).copied()
    }

    pub fn p4s(&self) -> &[Name] {
        &self.p4s
    }

    pub fn scalars(&self) -> &[Name] {
        &self.scalars
    }

    pub fn has_weight(&self) -> bool {
        self.has_weight
    }

    pub fn n_p4s(&self) -> usize {
        self.p4s.len()
    }

    pub fn n_scalars(&self) -> usize {
        self.scalars.len()
    }

    pub fn require_p4(&self, name: &str) -> LadduDataResult<usize> {
        self.p4_index(name)
            .ok_or_else(|| LadduDataError::MissingColumn(Name::from(name)))
    }

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

#[derive(Clone, Debug)]
pub struct SchemaColumnNames {
    pub weight_column: Name,
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

#[derive(Clone, Debug)]
pub struct SchemaInferenceOptions {
    pub column_names: SchemaColumnNames,
    pub require_weight: bool,
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

#[derive(Clone, Debug)]
pub struct P4Suffixes {
    pub e: &'static str,
    pub px: &'static str,
    pub py: &'static str,
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

    pub fn physical_p4_names(&self, prefix: &str) -> [String; 4] {
        [
            format!("{prefix}{}", self.e),
            format!("{prefix}{}", self.px),
            format!("{prefix}{}", self.py),
            format!("{prefix}{}", self.pz),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    F64,
    F32,
    Other,
}

impl ColumnType {
    pub fn is_supported_float(self) -> bool {
        matches!(self, Self::F64 | Self::F32)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ColumnInfo<'a> {
    pub name: &'a str,
    pub dtype: ColumnType,
}

impl Schema {
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

    pub fn physical_columns(&self, column_names: &SchemaColumnNames) -> Vec<Name> {
        let mut names = Vec::with_capacity(4 * self.n_p4s() + self.n_scalars() + 1);

        for p4 in self.p4s() {
            for name in column_names.p4_suffixes.physical_p4_names(p4) {
                names.push(Name::from(name));
            }
        }

        names.extend(self.scalars().iter().cloned());

        if self.has_weight() {
            names.push(Arc::clone(&column_names.weight_column));
        }

        names
    }

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

        for required in self.physical_columns(&options.column_names) {
            if !available.contains(required.as_ref()) {
                return Err(LadduDataError::MissingColumn(required));
            }
        }

        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub enum Precision {
    #[default]
    F64,
    F32,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum WriteWeightColumn {
    #[default]
    Always,
    OnlyIfPresent,
}

#[derive(Clone, Debug, Default)]
pub struct SchemaWriteOptions {
    pub column_names: SchemaColumnNames,
    pub precision: Precision,
    pub write_weight_column: WriteWeightColumn,
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
}
