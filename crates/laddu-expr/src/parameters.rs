use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type ParamResult<T> = Result<T, ParamError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ParamError {
    #[error("parameter name cannot be empty")]
    EmptyName,
    #[error("duplicate parameter name: {0}")]
    DuplicateName(String),
    #[error("parameter conflict for {name}: {reason}")]
    ParameterConflict { name: String, reason: String },
    #[error("invalid parameter id #{id} for layout of size {len}")]
    InvalidParamId { id: usize, len: usize },
    #[error("invalid free parameter id #{id} for layout with {len} free parameters")]
    InvalidFreeParamId { id: usize, len: usize },
    #[error("expected {expected} free parameters, got {actual}")]
    FreeLengthMismatch { expected: usize, actual: usize },
    #[error("expected {expected} total parameters, got {actual}")]
    FullLengthMismatch { expected: usize, actual: usize },
    #[error("invalid bounds for {name}: min {min} is greater than max {max}")]
    InvalidBounds { name: String, min: f64, max: f64 },
    #[error("invalid uniform initial range for {name}: min {min} is greater than max {max}")]
    InvalidInitialRange { name: String, min: f64, max: f64 },
    #[error("initial value {value} for {name} is outside bounds")]
    InitialOutOfBounds { name: String, value: f64 },
    #[error("fixed value {value} for {name} is outside bounds")]
    FixedValueOutOfBounds { name: String, value: f64 },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParamId(u32);

impl ParamId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FreeParamId(u32);

impl FreeParamId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InitialSpec {
    Default,
    Value(f64),
    Uniform { min: f64, max: f64 },
}

impl Default for InitialSpec {
    fn default() -> Self {
        Self::Default
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamState {
    Free,
    Fixed(f64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl Bounds {
    pub fn new(min: impl Into<Option<f64>>, max: impl Into<Option<f64>>) -> Self {
        Self {
            min: min.into(),
            max: max.into(),
        }
    }

    fn validate(&self, name: &str) -> ParamResult<()> {
        if let (Some(min), Some(max)) = (self.min, self.max) {
            if min > max {
                return Err(ParamError::InvalidBounds {
                    name: name.to_owned(),
                    min,
                    max,
                });
            }
        }
        Ok(())
    }

    pub fn contains(&self, value: f64) -> bool {
        self.min.is_none_or(|min| value >= min) && self.max.is_none_or(|max| value <= max)
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            min: None,
            max: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    name: Arc<str>,
    state: ParamState,
    initial: InitialSpec,
    bounds: Bounds,
    unit: Option<Arc<str>>,
    latex: Option<Arc<str>>,
    description: Option<Arc<str>>,
}

impl Parameter {
    pub fn free(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            state: ParamState::Free,
            initial: InitialSpec::Default,
            bounds: Bounds::default(),
            unit: None,
            latex: None,
            description: None,
        }
    }

    pub fn fixed(name: impl Into<Arc<str>>, value: f64) -> Self {
        Self {
            name: name.into(),
            state: ParamState::Fixed(value),
            initial: InitialSpec::Value(value),
            bounds: Bounds::default(),
            unit: None,
            latex: None,
            description: None,
        }
    }

    pub fn set_fixed_value(&mut self, value: f64) {
        self.state = ParamState::Fixed(value);
        self.initial = InitialSpec::Value(value);
    }

    pub fn with_fixed_value(mut self, value: f64) -> Self {
        self.set_fixed_value(value);
        self
    }

    pub fn set_initial(&mut self, initial: impl Into<InitialSpec>) {
        self.initial = initial.into();
    }

    pub fn with_initial(mut self, initial: impl Into<InitialSpec>) -> Self {
        self.set_initial(initial);
        self
    }

    pub fn set_bounds(&mut self, min: impl Into<Option<f64>>, max: impl Into<Option<f64>>) {
        self.bounds = Bounds::new(min, max);
    }

    pub fn with_bounds(mut self, min: impl Into<Option<f64>>, max: impl Into<Option<f64>>) -> Self {
        self.set_bounds(min, max);
        self
    }

    pub fn set_unit(&mut self, unit: impl Into<Arc<str>>) {
        self.unit = Some(unit.into());
    }

    pub fn with_unit(mut self, unit: impl Into<Arc<str>>) -> Self {
        self.set_unit(unit);
        self
    }

    pub fn set_latex(&mut self, latex: impl Into<Arc<str>>) {
        self.latex = Some(latex.into());
    }

    pub fn with_latex(mut self, latex: impl Into<Arc<str>>) -> Self {
        self.set_latex(latex);
        self
    }

    pub fn set_description(&mut self, description: impl Into<Arc<str>>) {
        self.description = Some(description.into());
    }

    pub fn with_description(mut self, description: impl Into<Arc<str>>) -> Self {
        self.set_description(description);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn state(&self) -> &ParamState {
        &self.state
    }

    pub fn is_free(&self) -> bool {
        matches!(self.state, ParamState::Free)
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self.state, ParamState::Fixed(_))
    }

    pub fn initial_spec(&self) -> &InitialSpec {
        &self.initial
    }

    pub fn bounds_spec(&self) -> &Bounds {
        &self.bounds
    }

    pub fn unit_label(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    pub fn latex_label(&self) -> Option<&str> {
        self.latex.as_deref()
    }

    pub fn description_text(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl From<f64> for InitialSpec {
    fn from(value: f64) -> Self {
        Self::Value(value)
    }
}

impl From<(f64, f64)> for InitialSpec {
    fn from((min, max): (f64, f64)) -> Self {
        Self::Uniform { min, max }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamLayout {
    specs: Arc<[Parameter]>,
    names: Arc<HashMap<Arc<str>, ParamId>>,
    free_params: Arc<[ParamId]>,
    full_to_free: Arc<[Option<FreeParamId>]>,
    defaults: Arc<[f64]>,
}

impl ParamLayout {
    pub fn new<S>(specs: impl IntoIterator<Item = S>) -> ParamResult<Self>
    where
        S: Into<Parameter>,
    {
        let specs: Vec<_> = specs.into_iter().map(Into::into).collect();
        let mut names = HashMap::with_capacity(specs.len());
        let mut free_params = Vec::new();
        let mut full_to_free = Vec::with_capacity(specs.len());
        let mut defaults = Vec::with_capacity(specs.len());

        for (index, spec) in specs.iter().enumerate() {
            if spec.name().is_empty() {
                return Err(ParamError::EmptyName);
            }
            spec.bounds.validate(spec.name())?;
            validate_initial(spec)?;
            let id = ParamId(index as u32);
            if names.insert(Arc::clone(&spec.name), id).is_some() {
                return Err(ParamError::DuplicateName(spec.name().to_owned()));
            }
            defaults.push(default_value(spec));
            match spec.state {
                ParamState::Free => {
                    let free_id = FreeParamId(free_params.len() as u32);
                    free_params.push(id);
                    full_to_free.push(Some(free_id));
                }
                ParamState::Fixed(_) => full_to_free.push(None),
            }
        }

        Ok(Self {
            specs: specs.into(),
            names: Arc::new(names),
            free_params: free_params.into(),
            full_to_free: full_to_free.into(),
            defaults: defaults.into(),
        })
    }

    pub fn specs(&self) -> &[Parameter] {
        &self.specs
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn n_free(&self) -> usize {
        self.free_params.len()
    }

    pub fn id(&self, name: &str) -> Option<ParamId> {
        self.names.get(name).copied()
    }

    pub fn name(&self, id: ParamId) -> ParamResult<&str> {
        self.check_id(id)?;
        Ok(self.specs[id.index()].name())
    }

    pub fn spec(&self, id: ParamId) -> ParamResult<&Parameter> {
        self.check_id(id)?;
        Ok(&self.specs[id.index()])
    }

    pub fn free_id(&self, id: ParamId) -> ParamResult<Option<FreeParamId>> {
        self.check_id(id)?;
        Ok(self.full_to_free[id.index()])
    }

    pub fn free_param(&self, id: FreeParamId) -> ParamResult<ParamId> {
        self.check_free_id(id)?;
        Ok(self.free_params[id.index()])
    }

    pub fn free_params(&self) -> &[ParamId] {
        &self.free_params
    }

    pub fn defaults(&self) -> &[f64] {
        &self.defaults
    }

    pub fn default_values(&self) -> ParamValues {
        ParamValues {
            layout: Arc::new(self.clone()),
            values: self.defaults.to_vec(),
        }
    }

    pub fn default_free_values(&self) -> Vec<f64> {
        self.free_params
            .iter()
            .map(|id| self.defaults[id.index()])
            .collect()
    }

    pub fn expand_free_values(&self, free: &[f64]) -> ParamResult<ParamValues> {
        let mut values = self.defaults.to_vec();
        self.fill_full_from_free(free, &mut values)?;
        Ok(ParamValues {
            layout: Arc::new(self.clone()),
            values,
        })
    }

    pub fn extract_free_values(&self, full: &[f64]) -> ParamResult<Vec<f64>> {
        let mut free = vec![0.0; self.n_free()];
        self.fill_free_from_full(full, &mut free)?;
        Ok(free)
    }

    pub fn fill_full_from_free(&self, free: &[f64], full: &mut [f64]) -> ParamResult<()> {
        if free.len() != self.n_free() {
            return Err(ParamError::FreeLengthMismatch {
                expected: self.n_free(),
                actual: free.len(),
            });
        }
        if full.len() != self.len() {
            return Err(ParamError::FullLengthMismatch {
                expected: self.len(),
                actual: full.len(),
            });
        }
        full.copy_from_slice(&self.defaults);
        for (free_index, id) in self.free_params.iter().enumerate() {
            full[id.index()] = free[free_index];
        }
        Ok(())
    }

    pub fn fill_free_from_full(&self, full: &[f64], free: &mut [f64]) -> ParamResult<()> {
        if full.len() != self.len() {
            return Err(ParamError::FullLengthMismatch {
                expected: self.len(),
                actual: full.len(),
            });
        }
        if free.len() != self.n_free() {
            return Err(ParamError::FreeLengthMismatch {
                expected: self.n_free(),
                actual: free.len(),
            });
        }
        for (free_index, id) in self.free_params.iter().enumerate() {
            free[free_index] = full[id.index()];
        }
        Ok(())
    }

    fn check_id(&self, id: ParamId) -> ParamResult<()> {
        if id.index() >= self.len() {
            Err(ParamError::InvalidParamId {
                id: id.index(),
                len: self.len(),
            })
        } else {
            Ok(())
        }
    }

    fn check_free_id(&self, id: FreeParamId) -> ParamResult<()> {
        if id.index() >= self.n_free() {
            Err(ParamError::InvalidFreeParamId {
                id: id.index(),
                len: self.n_free(),
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ParamRegistry {
    specs: Vec<Parameter>,
    names: HashMap<Arc<str>, ParamId>,
}

impl ParamRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<S>(&mut self, spec: S) -> ParamResult<ParamId>
    where
        S: Into<Parameter>,
    {
        let spec = spec.into();
        if spec.name().is_empty() {
            return Err(ParamError::EmptyName);
        }
        if let Some(id) = self.names.get(spec.name()).copied() {
            let existing = &self.specs[id.index()];
            if existing != &spec {
                return Err(ParamError::ParameterConflict {
                    name: spec.name().to_owned(),
                    reason: "duplicate parameter name has incompatible metadata".into(),
                });
            }
            return Ok(id);
        }

        let id = ParamId(self.specs.len() as u32);
        self.names.insert(Arc::clone(&spec.name), id);
        self.specs.push(spec);
        Ok(id)
    }

    pub fn specs(&self) -> &[Parameter] {
        &self.specs
    }

    pub fn layout(&self) -> ParamResult<ParamLayout> {
        ParamLayout::new(self.specs.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ParamValues {
    layout: Arc<ParamLayout>,
    values: Vec<f64>,
}

impl ParamValues {
    pub fn from_full(layout: Arc<ParamLayout>, values: Vec<f64>) -> ParamResult<Self> {
        if values.len() != layout.len() {
            return Err(ParamError::FullLengthMismatch {
                expected: layout.len(),
                actual: values.len(),
            });
        }
        Ok(Self { layout, values })
    }

    pub fn layout(&self) -> &Arc<ParamLayout> {
        &self.layout
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    pub fn get(&self, id: ParamId) -> ParamResult<f64> {
        self.layout.check_id(id)?;
        Ok(self.values[id.index()])
    }

    pub fn free_values(&self) -> Vec<f64> {
        self.layout
            .free_params()
            .iter()
            .map(|id| self.values[id.index()])
            .collect()
    }

    pub fn set_full(&mut self, id: ParamId, value: f64) -> ParamResult<()> {
        self.layout.check_id(id)?;
        self.values[id.index()] = value;
        Ok(())
    }

    pub fn set_free(&mut self, id: FreeParamId, value: f64) -> ParamResult<()> {
        let full_id = self.layout.free_param(id)?;
        self.values[full_id.index()] = value;
        Ok(())
    }
}

fn default_value(spec: &Parameter) -> f64 {
    match spec.state {
        ParamState::Fixed(value) => value,
        ParamState::Free => match spec.initial {
            InitialSpec::Default => 0.0,
            InitialSpec::Value(value) => value,
            InitialSpec::Uniform { min, max } => 0.5 * (min + max),
        },
    }
}

fn validate_initial(spec: &Parameter) -> ParamResult<()> {
    match spec.state {
        ParamState::Fixed(value) => {
            if !spec.bounds.contains(value) {
                return Err(ParamError::FixedValueOutOfBounds {
                    name: spec.name().to_owned(),
                    value,
                });
            }
        }
        ParamState::Free => match spec.initial {
            InitialSpec::Default => {}
            InitialSpec::Value(value) => {
                if !spec.bounds.contains(value) {
                    return Err(ParamError::InitialOutOfBounds {
                        name: spec.name().to_owned(),
                        value,
                    });
                }
            }
            InitialSpec::Uniform { min, max } => {
                if min > max {
                    return Err(ParamError::InvalidInitialRange {
                        name: spec.name().to_owned(),
                        min,
                        max,
                    });
                }
                let value = 0.5 * (min + max);
                if !spec.bounds.contains(value) {
                    return Err(ParamError::InitialOutOfBounds {
                        name: spec.name().to_owned(),
                        value,
                    });
                }
            }
        },
    }
    Ok(())
}

/// Convenience macro for creating parameters. Usage:
/// `parameter!("name")` for a free parameter, or `parameter!("name", 1.0)` for a fixed one.
#[macro_export]
macro_rules! parameter {
    ($name:expr) => {{
        $crate::parameters::Parameter::free($name)
    }};

    ($name:expr, $value:expr) => {{
        let mut p = $crate::parameters::Parameter::free($name);
        p.set_fixed_value(Some($value));
        p
    }};

    ($name:expr, $($rest:tt)+) => {{
        let mut p = $crate::parameters::Parameter::free($name);
        $crate::parameter!(@parse p, [fixed = false, initial = false]; $($rest)+);
        p
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; ) => {};

    (@parse $p:ident, [fixed = false, initial = false]; fixed : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_fixed_value(Some($value));
        $crate::parameter!(@parse $p, [fixed = true, initial = false]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = false, initial = false]; initial : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_initial($value);
        $crate::parameter!(@parse $p, [fixed = false, initial = true]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = true, initial = false]; initial : $value:expr $(, $($rest:tt)*)?) => {
        compile_error!("parameter!: cannot specify both `fixed` and `initial`");
    };

    (@parse $p:ident, [fixed = false, initial = true]; fixed : $value:expr $(, $($rest:tt)*)?) => {
        compile_error!("parameter!: cannot specify both `fixed` and `initial`");
    };

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; bounds : ($min:expr, $max:expr) $(, $($rest:tt)*)?) => {{
        $p.set_bounds($min, $max);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; unit : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_unit($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; latex : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_latex($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; description : $value:expr $(, $($rest:tt)*)?) => {{
        $p.set_description($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_tracks_free_and_fixed_values() {
        let layout = ParamLayout::new([
            Parameter::free("mass")
                .with_initial(1.2)
                .with_bounds(Some(0.0), Some(2.0)),
            Parameter::fixed("pi", std::f64::consts::PI),
            Parameter::free("width").with_initial((0.0, 1.0)),
        ])
        .unwrap();

        assert_eq!(layout.len(), 3);
        assert_eq!(layout.n_free(), 2);
        assert_eq!(layout.default_free_values(), vec![1.2, 0.5]);
        assert_eq!(layout.id("mass").map(ParamId::index), Some(0));
        assert_eq!(layout.id("pi").map(ParamId::index), Some(1));
        assert_eq!(layout.id("width").map(ParamId::index), Some(2));
        assert_eq!(
            layout
                .free_params()
                .iter()
                .map(|id| layout.name(*id).unwrap())
                .collect::<Vec<_>>(),
            vec!["mass", "width"]
        );

        let values = layout.expand_free_values(&[1.4, 0.2]).unwrap();
        assert_eq!(values.as_slice(), &[1.4, std::f64::consts::PI, 0.2]);
        assert_eq!(values.free_values(), vec![1.4, 0.2]);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let err = ParamLayout::new([Parameter::free("x"), Parameter::fixed("x", 1.0)]).unwrap_err();
        assert_eq!(err, ParamError::DuplicateName("x".into()));
    }

    #[test]
    fn free_length_is_checked() {
        let layout = ParamLayout::new([Parameter::free("x"), Parameter::free("y")]).unwrap();
        let err = layout.expand_free_values(&[1.0]).unwrap_err();
        assert_eq!(
            err,
            ParamError::FreeLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn full_and_free_vectors_round_trip_in_stable_order() {
        let layout = ParamLayout::new([
            Parameter::fixed("offset", -1.0),
            Parameter::free("mass").with_initial(1.2),
            Parameter::fixed("scale", 2.0),
            Parameter::free("width").with_initial(0.1),
        ])
        .unwrap();

        let full = layout.expand_free_values(&[1.4, 0.2]).unwrap();
        assert_eq!(full.as_slice(), &[-1.0, 1.4, 2.0, 0.2]);
        assert_eq!(
            layout.extract_free_values(full.as_slice()).unwrap(),
            vec![1.4, 0.2]
        );

        let mut rewritten = vec![0.0; layout.len()];
        layout
            .fill_full_from_free(&[1.5, 0.3], &mut rewritten)
            .unwrap();
        assert_eq!(rewritten, vec![-1.0, 1.5, 2.0, 0.3]);

        let mut free = vec![0.0; layout.n_free()];
        layout.fill_free_from_full(&rewritten, &mut free).unwrap();
        assert_eq!(free, vec![1.5, 0.3]);
    }

    #[test]
    fn values_can_be_mutated_by_full_or_free_id() {
        let layout = ParamLayout::new([
            Parameter::fixed("fixed", 1.0),
            Parameter::free("x"),
            Parameter::free("y"),
        ])
        .unwrap();
        let fixed_id = layout.id("fixed").unwrap();
        let x_id = layout.id("x").unwrap();
        let y_id = layout.id("y").unwrap();
        let x_free = layout.free_id(x_id).unwrap().unwrap();
        let y_free = layout.free_id(y_id).unwrap().unwrap();

        let mut values = layout.default_values();
        values.set_full(fixed_id, 2.0).unwrap();
        values.set_free(x_free, 3.0).unwrap();
        values.set_free(y_free, 4.0).unwrap();

        assert_eq!(values.as_slice(), &[2.0, 3.0, 4.0]);
        assert_eq!(values.free_values(), vec![3.0, 4.0]);
    }

    #[test]
    fn invalid_specs_are_rejected() {
        assert_eq!(
            ParamLayout::new([Parameter::free("")]).unwrap_err(),
            ParamError::EmptyName
        );

        assert_eq!(
            ParamLayout::new([Parameter::free("x").with_bounds(Some(2.0), Some(1.0))]).unwrap_err(),
            ParamError::InvalidBounds {
                name: "x".into(),
                min: 2.0,
                max: 1.0
            }
        );

        assert_eq!(
            ParamLayout::new([Parameter::free("x").with_initial((2.0, 1.0))]).unwrap_err(),
            ParamError::InvalidInitialRange {
                name: "x".into(),
                min: 2.0,
                max: 1.0
            }
        );

        assert_eq!(
            ParamLayout::new([Parameter::free("x")
                .with_initial(3.0)
                .with_bounds(Some(0.0), Some(2.0))])
            .unwrap_err(),
            ParamError::InitialOutOfBounds {
                name: "x".into(),
                value: 3.0
            }
        );

        assert_eq!(
            ParamLayout::new([Parameter::fixed("x", 3.0).with_bounds(Some(0.0), Some(2.0))])
                .unwrap_err(),
            ParamError::FixedValueOutOfBounds {
                name: "x".into(),
                value: 3.0
            }
        );
    }

    #[test]
    fn vector_lengths_are_checked_for_both_directions() {
        let layout = ParamLayout::new([
            Parameter::fixed("a", 0.0),
            Parameter::free("x"),
            Parameter::free("y"),
        ])
        .unwrap();

        assert_eq!(
            layout
                .fill_full_from_free(&[1.0], &mut [0.0, 0.0, 0.0])
                .unwrap_err(),
            ParamError::FreeLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(
            layout
                .fill_full_from_free(&[1.0, 2.0], &mut [0.0, 0.0])
                .unwrap_err(),
            ParamError::FullLengthMismatch {
                expected: 3,
                actual: 2
            }
        );
        assert_eq!(
            layout
                .fill_free_from_full(&[0.0, 1.0], &mut [0.0, 0.0])
                .unwrap_err(),
            ParamError::FullLengthMismatch {
                expected: 3,
                actual: 2
            }
        );
        assert_eq!(
            layout
                .fill_free_from_full(&[0.0, 1.0, 2.0], &mut [0.0])
                .unwrap_err(),
            ParamError::FreeLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn registry_merges_identical_parameters_in_first_seen_order() {
        let mut registry = ParamRegistry::new();
        let y = registry
            .register(Parameter::free("y").with_initial(1.0).with_bounds(0.0, 2.0))
            .unwrap();
        let x = registry.register(Parameter::free("x")).unwrap();
        let y_again = registry
            .register(Parameter::free("y").with_initial(1.0).with_bounds(0.0, 2.0))
            .unwrap();

        assert_eq!(y.index(), 0);
        assert_eq!(x.index(), 1);
        assert_eq!(y_again, y);

        let layout = registry.layout().unwrap();
        assert_eq!(
            layout
                .specs()
                .iter()
                .map(Parameter::name)
                .collect::<Vec<_>>(),
            vec!["y", "x"]
        );
    }

    #[test]
    fn registry_rejects_incompatible_parameter_reuse() {
        let mut registry = ParamRegistry::new();
        registry
            .register(Parameter::free("x").with_initial(1.0))
            .unwrap();

        assert!(matches!(
            registry.register(Parameter::free("x").with_initial(2.0)),
            Err(ParamError::ParameterConflict { name, .. }) if name == "x"
        ));
    }
}
