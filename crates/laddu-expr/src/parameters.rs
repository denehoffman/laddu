use std::{collections::HashMap, fmt, sync::Arc};

pub use crate::{ParamError, ParamResult};
use fastrand::Rng;
use fastrand_contrib::RngExt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Stable identifier for a parameter in a [`ParamLayout`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParamId(u32);

impl ParamId {
    /// Returns the zero-based position in the full parameter layout.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Stable identifier for a free parameter in free-parameter order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FreeParamId(u32);

impl FreeParamId {
    /// Returns the zero-based position among free parameters.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Rule used to choose a free parameter's initial value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum InitialSpec {
    /// Use the default value zero.
    #[default]
    Default,
    /// Use a specific initial value.
    Value(f64),
    /// Sample uniformly from an inclusive range.
    Uniform {
        /// Range minimum.
        min: f64,
        /// Range maximum.
        max: f64,
    },
}

/// Whether a parameter is varied or held at a fixed value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamState {
    /// The parameter is supplied by the optimizer or caller.
    Free,
    /// The parameter is fixed at the contained value.
    Fixed(f64),
}

/// Classified failure from validating a free-parameter value vector.
///
/// This separates structural input errors from invalid numeric input and
/// values that are finite but outside the parameter support.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FreeValueValidationError {
    /// Parameter-layout validation failed before individual values were classified.
    #[error(transparent)]
    Parameter(#[from] ParamError),
    /// A free parameter was assigned a non-finite value.
    #[error("non-finite value {value} for free parameter {name} ({id:?})")]
    NonFiniteValue {
        /// Identifier in free-parameter order.
        id: FreeParamId,
        /// Parameter name.
        name: String,
        /// Invalid value.
        value: f64,
    },
    /// A finite free-parameter value was outside its declared support.
    #[error("value {value} for free parameter {name} ({id:?}) is outside its support")]
    OutsideSupport {
        /// Identifier in free-parameter order.
        id: FreeParamId,
        /// Parameter name.
        name: String,
        /// Unsupported value.
        value: f64,
    },
}

/// Optional inclusive lower and upper bounds for a parameter.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    /// Inclusive lower bound, or no lower bound.
    pub min: Option<f64>,
    /// Inclusive upper bound, or no upper bound.
    pub max: Option<f64>,
}

impl Bounds {
    /// Creates bounds from optional lower and upper endpoints.
    pub fn new(min: impl Into<Option<f64>>, max: impl Into<Option<f64>>) -> Self {
        Self {
            min: min.into(),
            max: max.into(),
        }
    }

    fn validate(&self, name: &str) -> ParamResult<()> {
        if let (Some(min), Some(max)) = (self.min, self.max)
            && min > max
        {
            return Err(ParamError::InvalidBounds {
                name: name.to_owned(),
                min,
                max,
            });
        }
        Ok(())
    }

    /// Returns whether `value` lies within both configured bounds.
    pub fn contains(&self, value: f64) -> bool {
        self.min.is_none_or(|min| value >= min) && self.max.is_none_or(|max| value <= max)
    }
}

/// Complete definition and user-facing metadata for one scalar parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    name: Arc<str>,
    state: ParamState,
    initial: InitialSpec,
    bounds: Bounds,
    #[serde(default)]
    periodic: bool,
    #[serde(default)]
    scale: Option<f64>,
    unit: Option<Arc<str>>,
    latex: Option<Arc<str>>,
    description: Option<Arc<str>>,
}

impl Parameter {
    /// Creates an unbounded free parameter with a default initial value.
    pub fn free(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            state: ParamState::Free,
            initial: InitialSpec::Default,
            bounds: Bounds::default(),
            periodic: false,
            scale: None,
            unit: None,
            latex: None,
            description: None,
        }
    }

    /// Creates an unbounded parameter fixed at `value`.
    pub fn fixed(name: impl Into<Arc<str>>, value: f64) -> Self {
        Self {
            name: name.into(),
            state: ParamState::Fixed(value),
            initial: InitialSpec::Value(value),
            bounds: Bounds::default(),
            periodic: false,
            scale: None,
            unit: None,
            latex: None,
            description: None,
        }
    }

    fn set_fixed_value(&mut self, value: f64) {
        self.state = ParamState::Fixed(value);
        self.initial = InitialSpec::Value(value);
    }

    /// Returns this parameter fixed at `value`.
    pub fn with_fixed_value(mut self, value: f64) -> Self {
        self.set_fixed_value(value);
        self
    }

    fn set_free(&mut self) {
        self.state = ParamState::Free;
    }

    /// Returns this parameter marked as free.
    pub fn with_free(mut self) -> Self {
        self.set_free();
        self
    }

    fn set_initial(&mut self, initial: impl Into<InitialSpec>) {
        self.initial = initial.into();
    }

    /// Returns this parameter with the specified initialization rule.
    pub fn with_initial(mut self, initial: impl Into<InitialSpec>) -> Self {
        self.set_initial(initial);
        self
    }

    fn set_bounds(&mut self, min: impl Into<Option<f64>>, max: impl Into<Option<f64>>) {
        self.bounds = Bounds::new(min, max);
    }

    /// Returns this parameter with inclusive optional bounds.
    pub fn with_bounds(mut self, min: impl Into<Option<f64>>, max: impl Into<Option<f64>>) -> Self {
        self.set_bounds(min, max);
        self
    }

    /// Mark this parameter as periodic over its finite two-sided bounds.
    pub fn with_periodic(mut self) -> Self {
        self.periodic = true;
        self
    }

    /// Sets whether the parameter is periodic.
    pub fn with_periodicity(mut self, periodic: bool) -> Self {
        self.periodic = periodic;
        self
    }

    /// Set the characteristic optimizer scale for this parameter.
    ///
    /// The scale is metadata: fit integrations may use it to condition the
    /// optimizer coordinate system, while direct model evaluation is unchanged.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    fn set_unit(&mut self, unit: impl Into<Arc<str>>) {
        self.unit = Some(unit.into());
    }

    /// Attaches a human-readable unit label.
    pub fn with_unit(mut self, unit: impl Into<Arc<str>>) -> Self {
        self.set_unit(unit);
        self
    }

    fn set_latex(&mut self, latex: impl Into<Arc<str>>) {
        self.latex = Some(latex.into());
    }

    /// Attaches a LaTeX-formatted label.
    pub fn with_latex(mut self, latex: impl Into<Arc<str>>) -> Self {
        self.set_latex(latex);
        self
    }

    fn set_description(&mut self, description: impl Into<Arc<str>>) {
        self.description = Some(description.into());
    }

    /// Attaches a longer human-readable description.
    pub fn with_description(mut self, description: impl Into<Arc<str>>) -> Self {
        self.set_description(description);
        self
    }

    /// Returns the unique parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns whether the parameter is free or fixed.
    pub fn state(&self) -> &ParamState {
        &self.state
    }

    /// Returns whether the parameter is free.
    pub fn is_free(&self) -> bool {
        matches!(self.state, ParamState::Free)
    }

    /// Returns whether the parameter is fixed.
    pub fn is_fixed(&self) -> bool {
        matches!(self.state, ParamState::Fixed(_))
    }

    /// Returns the initialization rule.
    pub fn initial_spec(&self) -> &InitialSpec {
        &self.initial
    }

    /// Returns the optional parameter bounds.
    pub fn bounds_spec(&self) -> &Bounds {
        &self.bounds
    }

    /// Returns whether the parameter is periodic over its bounds.
    pub fn is_periodic(&self) -> bool {
        self.periodic
    }

    /// Return the validated canonical half-open periodic interval.
    pub fn periodic_bounds(&self) -> Option<(f64, f64)> {
        match (self.periodic, self.bounds.min, self.bounds.max) {
            (true, Some(min), Some(max)) if min.is_finite() && max.is_finite() && min < max => {
                Some((min, max))
            }
            _ => None,
        }
    }

    /// Returns the optional characteristic optimizer scale.
    pub fn scale(&self) -> Option<f64> {
        self.scale
    }

    /// Returns the optional human-readable unit label.
    pub fn unit_label(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// Returns the optional LaTeX-formatted label.
    pub fn latex_label(&self) -> Option<&str> {
        self.latex.as_deref()
    }

    /// Returns the optional longer description.
    pub fn description_text(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn validate(&self) -> ParamResult<()> {
        if self.name().is_empty() {
            return Err(ParamError::EmptyName);
        }
        self.bounds.validate(self.name())?;
        if self.periodic && self.periodic_bounds().is_none() {
            return Err(ParamError::PeriodicRequiresFiniteBounds {
                name: self.name().to_owned(),
            });
        }
        if let Some(scale) = self.scale
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(ParamError::InvalidScale {
                name: self.name().to_owned(),
                scale,
            });
        }
        self.validate_initial()
    }

    fn validate_initial(&self) -> ParamResult<()> {
        match self.state {
            ParamState::Fixed(value) => {
                if !self.bounds.contains(value) {
                    return Err(ParamError::FixedValueOutOfBounds {
                        name: self.name().to_owned(),
                        value,
                    });
                }
                self.validate_periodic_value(value)
            }
            ParamState::Free => self.validate_free_initial(),
        }
    }

    fn validate_free_initial(&self) -> ParamResult<()> {
        match self.initial {
            InitialSpec::Default | InitialSpec::Value(_) => {
                let value = self.initial.representative_value();
                if !self.bounds.contains(value) {
                    return Err(ParamError::InitialOutOfBounds {
                        name: self.name().to_owned(),
                        value,
                    });
                }
                self.validate_periodic_value(value)
            }
            InitialSpec::Uniform { min, max } => {
                if min > max {
                    return Err(ParamError::InvalidInitialRange {
                        name: self.name().to_owned(),
                        min,
                        max,
                    });
                }
                if !self.bounds.contains(min) || !self.bounds.contains(max) {
                    return Err(ParamError::InitialRangeOutOfBounds {
                        name: self.name().to_owned(),
                        min,
                        max,
                    });
                }
                if let Some((domain_min, domain_max)) = self.periodic_bounds()
                    && (min < domain_min || max > domain_max)
                {
                    let value = if min < domain_min { min } else { max };
                    return Err(ParamError::ValueOutsidePeriodicDomain {
                        name: self.name().to_owned(),
                        value,
                        min: domain_min,
                        max: domain_max,
                    });
                }
                Ok(())
            }
        }
    }

    fn default_value(&self) -> f64 {
        match self.state {
            ParamState::Fixed(value) => value,
            ParamState::Free => self.initial.representative_value(),
        }
    }

    fn validate_periodic_value(&self, value: f64) -> ParamResult<()> {
        if let Some((min, max)) = self.periodic_bounds()
            && !(value.is_finite() && value >= min && value < max)
        {
            return Err(ParamError::ValueOutsidePeriodicDomain {
                name: self.name().to_owned(),
                value,
                min,
                max,
            });
        }
        Ok(())
    }

    fn validate_value(&self, value: f64) -> ParamResult<()> {
        if !self.bounds.contains(value) {
            return Err(ParamError::ValueOutOfBounds {
                name: self.name().to_owned(),
                value,
            });
        }
        self.validate_periodic_value(value)
    }

    fn contains_in_support(&self, value: f64) -> bool {
        self.bounds.contains(value)
            && self
                .periodic_bounds()
                .is_none_or(|(min, max)| value >= min && value < max)
    }
}

impl InitialSpec {
    fn representative_value(&self) -> f64 {
        match *self {
            Self::Default => 0.0,
            Self::Value(value) => value,
            Self::Uniform { min, max } => 0.5 * (min + max),
        }
    }

    fn sample_with(&self, rng: &mut Rng) -> f64 {
        match *self {
            Self::Default => 0.0,
            Self::Value(value) => value,
            Self::Uniform { min, max } => rng.f64_range(min..max),
        }
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

/// Validated parameter ordering and mapping between full and free values.
#[derive(Clone)]
pub struct ParamLayout {
    specs: Arc<[Parameter]>,
    names: Arc<HashMap<Arc<str>, ParamId>>,
    projection: ParamLayoutProjection,
}

impl fmt::Debug for ParamLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParamLayout")
            .field("specs", &self.specs)
            .field("names", &self.names)
            .field("free_params", &self.projection.free_params)
            .field("full_to_free", &self.projection.full_to_free)
            .field("defaults", &self.projection.defaults)
            .finish()
    }
}

#[derive(Clone, Debug)]
// Owns the stable mappings and defaults that define full/free projection.
struct ParamLayoutProjection {
    free_params: Arc<[ParamId]>,
    full_to_free: Arc<[Option<FreeParamId>]>,
    defaults: Arc<[f64]>,
}

impl ParamLayoutProjection {
    fn n_free(&self) -> usize {
        self.free_params.len()
    }

    fn free_params(&self) -> &[ParamId] {
        &self.free_params
    }

    fn free_id(&self, id: ParamId) -> Option<FreeParamId> {
        self.full_to_free[id.index()]
    }

    fn full_id(&self, id: FreeParamId) -> ParamId {
        self.free_params[id.index()]
    }

    fn validate_free_dimension<T>(&self, values: &[T]) -> ParamResult<()> {
        if values.len() == self.n_free() {
            Ok(())
        } else {
            Err(ParamError::FreeLengthMismatch {
                expected: self.n_free(),
                actual: values.len(),
            })
        }
    }

    fn initial_free_values(&self) -> Vec<f64> {
        self.free_params
            .iter()
            .map(|id| self.defaults[id.index()])
            .collect()
    }

    fn fill_full_from_free(&self, free: &[f64], full: &mut [f64]) -> ParamResult<()> {
        self.validate_free_dimension(free)?;
        debug_assert_eq!(full.len(), self.defaults.len());
        full.copy_from_slice(&self.defaults);
        for (value, id) in free.iter().zip(self.free_params.iter()) {
            full[id.index()] = *value;
        }
        Ok(())
    }

    fn free_values_from_full(&self, full: &[f64]) -> Vec<f64> {
        debug_assert_eq!(full.len(), self.defaults.len());
        self.free_params.iter().map(|id| full[id.index()]).collect()
    }
}

/// Validated projection from one parameter layout into another.
///
/// The target layout owns the values produced by [`Self::project`], while the
/// source layout owns the values accepted by it.  Only free parameters are
/// projected: fixed target parameters retain their configured values, and a
/// target free parameter must also be free in the source layout.
#[derive(Clone, Debug)]
pub struct ParamProjection {
    source: Arc<ParamLayout>,
    target: Arc<ParamLayout>,
    source_free_ids: Arc<[FreeParamId]>,
}

#[derive(Serialize, Deserialize)]
// Keep ParamLayout's established flat serialized representation while its
// projection fields are grouped behind one internal artifact.
struct ParamLayoutSerde {
    specs: Arc<[Parameter]>,
    names: Arc<HashMap<Arc<str>, ParamId>>,
    free_params: Arc<[ParamId]>,
    full_to_free: Arc<[Option<FreeParamId>]>,
    defaults: Arc<[f64]>,
}

impl Serialize for ParamLayout {
    fn serialize<__S>(&self, serializer: __S) -> Result<__S::Ok, __S::Error>
    where
        __S: Serializer,
    {
        ParamLayoutSerde {
            specs: Arc::clone(&self.specs),
            names: Arc::clone(&self.names),
            free_params: Arc::clone(&self.projection.free_params),
            full_to_free: Arc::clone(&self.projection.full_to_free),
            defaults: Arc::clone(&self.projection.defaults),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ParamLayout {
    fn deserialize<__D>(deserializer: __D) -> Result<Self, __D::Error>
    where
        __D: Deserializer<'de>,
    {
        let serialized = ParamLayoutSerde::deserialize(deserializer)?;
        Ok(Self {
            specs: serialized.specs,
            names: serialized.names,
            projection: ParamLayoutProjection {
                free_params: serialized.free_params,
                full_to_free: serialized.full_to_free,
                defaults: serialized.defaults,
            },
        })
    }
}

impl ParamProjection {
    /// Projects values from the source layout into the target layout.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::UnknownName`] when the supplied values do not
    /// contain a required target parameter.
    pub fn project(&self, source: &ParamValues) -> ParamResult<ParamValues> {
        let free = if source.layout().specs() == self.source.specs() {
            self.source_free_ids
                .iter()
                .map(|id| source.get(self.source.projection.full_id(*id)))
                .collect::<ParamResult<Vec<_>>>()?
        } else {
            self.target
                .free_params()
                .iter()
                .map(|target_id| {
                    let name = self.target.name(*target_id)?;
                    let source_id = source
                        .layout()
                        .id(name)
                        .ok_or_else(|| ParamError::UnknownName(name.to_owned()))?;
                    source.get(source_id)
                })
                .collect::<ParamResult<Vec<_>>>()?
        };
        self.target.values(&free)
    }

    /// Scatters a target-layout free gradient into a source-layout gradient.
    ///
    /// Values are added to `source`, allowing several projections to
    /// contribute to one gradient.  Both dimensions are validated against the
    /// layouts captured by this artifact.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::FreeLengthMismatch`] when either slice has an
    /// incompatible free-parameter dimension.
    pub fn scatter_add(&self, target: &[f64], source: &mut [f64]) -> ParamResult<()> {
        self.target.projection.validate_free_dimension(target)?;
        self.source.projection.validate_free_dimension(source)?;
        for (value, id) in target.iter().zip(self.source_free_ids.iter()) {
            source[id.index()] += value;
        }
        Ok(())
    }
}

struct LayoutBuilder {
    specs: Vec<Parameter>,
    names: HashMap<Arc<str>, ParamId>,
    free_params: Vec<ParamId>,
    full_to_free: Vec<Option<FreeParamId>>,
    defaults: Vec<f64>,
}

impl LayoutBuilder {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            specs: Vec::with_capacity(capacity),
            names: HashMap::with_capacity(capacity),
            free_params: Vec::new(),
            full_to_free: Vec::with_capacity(capacity),
            defaults: Vec::with_capacity(capacity),
        }
    }

    fn push_validated(&mut self, spec: Parameter) -> ParamResult<()> {
        spec.validate()?;
        let id = ParamId(self.specs.len() as u32);
        if self.names.insert(Arc::clone(&spec.name), id).is_some() {
            return Err(ParamError::DuplicateName(spec.name().to_owned()));
        }
        self.defaults.push(spec.default_value());
        match spec.state {
            ParamState::Free => {
                let free_id = FreeParamId(self.free_params.len() as u32);
                self.free_params.push(id);
                self.full_to_free.push(Some(free_id));
            }
            ParamState::Fixed(_) => self.full_to_free.push(None),
        }
        self.specs.push(spec);
        Ok(())
    }

    fn finish(self) -> ParamLayout {
        ParamLayout {
            specs: self.specs.into(),
            names: Arc::new(self.names),
            projection: ParamLayoutProjection {
                free_params: self.free_params.into(),
                full_to_free: self.full_to_free.into(),
                defaults: self.defaults.into(),
            },
        }
    }
}

impl ParamLayout {
    /// Validates parameter definitions and constructs a layout.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when a definition has an empty or duplicate
    /// name, invalid bounds, invalid periodic metadata, an invalid scale, or
    /// an initial or fixed value outside its permitted domain.
    pub fn new<S>(specs: impl IntoIterator<Item = S>) -> ParamResult<Self>
    where
        S: Into<Parameter>,
    {
        let specs: Vec<_> = specs.into_iter().map(Into::into).collect();
        let mut builder = LayoutBuilder::with_capacity(specs.len());
        for spec in specs {
            builder.push_validated(spec)?;
        }
        Ok(builder.finish())
    }

    /// Returns all parameter definitions in full-layout order.
    pub fn specs(&self) -> &[Parameter] {
        &self.specs
    }

    /// Returns the total number of free and fixed parameters.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Returns whether the layout contains no parameters.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Returns the number of free parameters.
    pub fn n_free(&self) -> usize {
        self.projection.n_free()
    }

    /// Looks up a full-layout identifier by parameter name.
    pub fn id(&self, name: &str) -> Option<ParamId> {
        self.names.get(name).copied()
    }

    /// Returns the name associated with a full-layout identifier.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::InvalidParamId`] when `id` is outside this
    /// layout.
    pub fn name(&self, id: ParamId) -> ParamResult<&str> {
        self.check_id(id)?;
        Ok(self.specs[id.index()].name())
    }

    /// Returns the definition associated with a full-layout identifier.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::InvalidParamId`] when `id` is outside this
    /// layout.
    pub fn spec(&self, id: ParamId) -> ParamResult<&Parameter> {
        self.check_id(id)?;
        Ok(&self.specs[id.index()])
    }

    /// Maps a full-layout identifier to free-parameter order.
    ///
    /// Fixed parameters return `None`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::InvalidParamId`] when `id` is outside this
    /// layout.
    pub fn free_id(&self, id: ParamId) -> ParamResult<Option<FreeParamId>> {
        self.check_id(id)?;
        Ok(self.projection.free_id(id))
    }

    fn free_param(&self, id: FreeParamId) -> ParamResult<ParamId> {
        self.check_free_id(id)?;
        Ok(self.projection.full_id(id))
    }

    /// Returns full-layout identifiers in free-parameter order.
    pub fn free_params(&self) -> &[ParamId] {
        self.projection.free_params()
    }

    /// Builds a validated projection from `source` into this layout.
    ///
    /// Free parameters are matched by name and must be free in both layouts.
    /// Parameters present only in the source are permitted; fixed parameters
    /// in this layout use their own configured values.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::UnknownName`] when a free target parameter is
    /// absent from `source`, or [`ParamError::ParameterConflict`] when it is
    /// fixed there.
    pub fn projection_from(&self, source: &ParamLayout) -> ParamResult<ParamProjection> {
        let source_free_ids = self
            .free_params()
            .iter()
            .map(|target_id| {
                let name = self.name(*target_id)?;
                let source_id = source
                    .id(name)
                    .ok_or_else(|| ParamError::UnknownName(name.to_owned()))?;
                source
                    .free_id(source_id)?
                    .ok_or_else(|| ParamError::ParameterConflict {
                        name: name.to_owned(),
                        reason: "target free parameter is fixed in the source layout".to_owned(),
                    })
            })
            .collect::<ParamResult<Arc<[_]>>>()?;
        Ok(ParamProjection {
            source: Arc::new(source.clone()),
            target: Arc::new(self.clone()),
            source_free_ids,
        })
    }

    /// Iterates parameter definitions in stable free-parameter order.
    pub fn free_parameters(
        &self,
    ) -> impl ExactSizeIterator<Item = &Parameter> + DoubleEndedIterator {
        self.free_params().iter().map(|id| &self.specs[id.index()])
    }

    /// Creates a full value set using fixed and deterministic initial values.
    pub fn default_values(&self) -> ParamValues {
        ParamValues {
            layout: Arc::new(self.clone()),
            values: self.projection.defaults.to_vec(),
        }
    }

    /// Return deterministic initial values in free-parameter order.
    ///
    /// Uniform initial ranges use their midpoint.
    pub fn initial_free_values(&self) -> Vec<f64> {
        self.projection.initial_free_values()
    }

    /// Expand a free-parameter slice while restoring fixed values from the layout.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::FreeLengthMismatch`] when `free` does not contain
    /// exactly one value per free parameter.
    pub fn values(&self, free: &[f64]) -> ParamResult<ParamValues> {
        let mut values = self.projection.defaults.to_vec();
        self.fill_full_from_free(free, &mut values)?;
        Ok(ParamValues {
            layout: Arc::new(self.clone()),
            values,
        })
    }

    /// Generate one value per free parameter in layout order.
    pub fn free_values_with(&self, mut value: impl FnMut(&Parameter) -> f64) -> Vec<f64> {
        self.free_parameters().map(&mut value).collect()
    }

    /// Generate initial free values, invoking `uniform` only for uniform initial ranges.
    pub fn sample_initial(&self, seed: u64) -> Vec<f64> {
        let mut rng = Rng::with_seed(seed);
        self.free_values_with(|parameter| parameter.initial.sample_with(&mut rng))
    }

    /// Validate free values against ordinary bounds and canonical periodic domains.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::FreeLengthMismatch`] when `free` has the wrong
    /// length, or a value-related [`ParamError`] when a value lies outside its
    /// parameter's bounds or canonical periodic domain.
    pub fn validate_free_values(&self, free: &[f64]) -> ParamResult<()> {
        self.projection.validate_free_dimension(free)?;
        for (value, parameter) in free.iter().zip(self.free_parameters()) {
            parameter.validate_value(*value)?;
        }
        Ok(())
    }

    /// Validate free values while classifying structural, numeric, and support failures.
    ///
    /// Unlike [`Self::validate_free_values`], this method treats every non-finite
    /// value as invalid input, including for an otherwise unbounded parameter.
    ///
    /// # Errors
    ///
    /// Returns [`FreeValueValidationError::Parameter`] for layout-level
    /// validation failures such as the wrong number of values,
    /// [`FreeValueValidationError::NonFiniteValue`] for NaN or infinity, and
    /// [`FreeValueValidationError::OutsideSupport`] for a finite value outside
    /// ordinary bounds or a canonical periodic domain.
    pub fn validate_free_values_classified(
        &self,
        free: &[f64],
    ) -> Result<(), FreeValueValidationError> {
        self.projection.validate_free_dimension(free)?;
        for (index, (value, parameter)) in free.iter().zip(self.free_parameters()).enumerate() {
            let id = FreeParamId(index as u32);
            if !value.is_finite() {
                return Err(FreeValueValidationError::NonFiniteValue {
                    id,
                    name: parameter.name().to_owned(),
                    value: *value,
                });
            }
            if !parameter.contains_in_support(*value) {
                return Err(FreeValueValidationError::OutsideSupport {
                    id,
                    name: parameter.name().to_owned(),
                    value: *value,
                });
            }
        }
        Ok(())
    }

    /// Return free values with periodic parameters mapped into their canonical domains.
    /// Non-periodic values are unchanged; ordinary bounds are not clamped.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::FreeLengthMismatch`] when `free` does not contain
    /// exactly one value per free parameter.
    pub fn wrap_periodic_free_values(&self, free: &[f64]) -> ParamResult<Vec<f64>> {
        self.projection.validate_free_dimension(free)?;
        Ok(free
            .iter()
            .zip(self.free_params().iter())
            .map(|(value, id)| {
                let parameter = &self.specs[id.index()];
                parameter.periodic_bounds().map_or(*value, |(min, max)| {
                    min + (*value - min).rem_euclid(max - min)
                })
            })
            .collect())
    }

    fn fill_full_from_free(&self, free: &[f64], full: &mut [f64]) -> ParamResult<()> {
        self.projection.fill_full_from_free(free, full)
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

/// Incremental collection of uniquely named parameter definitions.
#[derive(Clone, Debug, Default)]
pub struct ParamRegistry {
    specs: Vec<Parameter>,
    names: HashMap<Arc<str>, ParamId>,
}

impl ParamRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a parameter and returns its stable identifier.
    ///
    /// Re-registering an identical definition returns its existing identifier;
    /// incompatible definitions with the same name return an error.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::EmptyName`] when the parameter name is empty, or
    /// [`ParamError::ParameterConflict`] when the name is already associated
    /// with a different definition.
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

    /// Validates the registered parameters and builds their layout.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when any registered definition has invalid
    /// bounds, periodic metadata, scale, or initial or fixed values.
    pub fn layout(&self) -> ParamResult<ParamLayout> {
        ParamLayout::new(self.specs.clone())
    }
}

/// Concrete full-layout parameter values paired with their defining layout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamValues {
    layout: Arc<ParamLayout>,
    values: Vec<f64>,
}

impl ParamValues {
    /// Returns the shared layout that interprets these values.
    pub fn layout(&self) -> &Arc<ParamLayout> {
        &self.layout
    }

    /// Returns all values in full-layout order.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Returns the value associated with a full-layout identifier.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::InvalidParamId`] when `id` is outside the shared
    /// layout.
    pub fn get(&self, id: ParamId) -> ParamResult<f64> {
        self.layout.check_id(id)?;
        Ok(self.values[id.index()])
    }

    /// Copies the values of free parameters in free-parameter order.
    pub fn free_values(&self) -> Vec<f64> {
        self.layout.projection.free_values_from_full(&self.values)
    }

    /// Assigns one value by its free-parameter identifier.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::InvalidFreeParamId`] when `id` is outside the
    /// shared layout's free-parameter ordering.
    pub fn set_free(&mut self, id: FreeParamId, value: f64) -> ParamResult<()> {
        let full_id = self.layout.free_param(id)?;
        self.values[full_id.index()] = value;
        Ok(())
    }

    /// Replaces all free values and restores fixed values from the layout.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::FreeLengthMismatch`] when `values` does not
    /// contain exactly one value per free parameter.
    pub fn set_free_values(&mut self, values: &[f64]) -> ParamResult<()> {
        let layout = Arc::clone(&self.layout);
        layout.fill_full_from_free(values, &mut self.values)
    }
}

/// Convenience macro for creating parameters. Usage:
/// `parameter!("name")` for a free parameter, or `parameter!("name", 1.0)` for a fixed one.
#[macro_export]
macro_rules! parameter {
    ($name:expr) => {{
        $crate::parameters::Parameter::free($name)
    }};

    ($name:expr, $value:expr) => {{
        $crate::parameters::Parameter::fixed($name, $value)
    }};

    ($name:expr, $($rest:tt)+) => {{
        let mut p = $crate::parameters::Parameter::free($name);
        $crate::parameter!(@parse p, [fixed = false, initial = false]; $($rest)+);
        p
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; ) => {};

    (@parse $p:ident, [fixed = false, initial = false]; fixed : $value:expr $(, $($rest:tt)*)?) => {{
        $p = $p.with_fixed_value($value);
        $crate::parameter!(@parse $p, [fixed = true, initial = false]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = false, initial = false]; initial : $value:expr $(, $($rest:tt)*)?) => {{
        $p = $p.with_initial($value);
        $crate::parameter!(@parse $p, [fixed = false, initial = true]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = true, initial = false]; initial : $value:expr $(, $($rest:tt)*)?) => {
        compile_error!("parameter!: cannot specify both `fixed` and `initial`");
    };

    (@parse $p:ident, [fixed = false, initial = true]; fixed : $value:expr $(, $($rest:tt)*)?) => {
        compile_error!("parameter!: cannot specify both `fixed` and `initial`");
    };

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; bounds : ($min:expr, $max:expr) $(, $($rest:tt)*)?) => {{
        $p = $p.with_bounds($min, $max);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; periodic : $value:expr $(, $($rest:tt)*)?) => {{
        $p = $p.with_periodicity($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; periodic $(, $($rest:tt)*)?) => {{
        $p = $p.with_periodic();
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; scale : $value:expr $(, $($rest:tt)*)?) => {{
        $p = $p.with_scale($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; unit : $value:expr $(, $($rest:tt)*)?) => {{
        $p = $p.with_unit($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; latex : $value:expr $(, $($rest:tt)*)?) => {{
        $p = $p.with_latex($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};

    (@parse $p:ident, [fixed = $f:tt, initial = $i:tt]; description : $value:expr $(, $($rest:tt)*)?) => {{
        $p = $p.with_description($value);
        $crate::parameter!(@parse $p, [fixed = $f, initial = $i]; $($($rest)*)?);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_macro_constructs_fixed_parameters() {
        let positional = crate::parameter!("positional", 1.25);
        let named = crate::parameter!("named", fixed: -0.5);

        assert_eq!(positional.state(), &ParamState::Fixed(1.25));
        assert_eq!(named.state(), &ParamState::Fixed(-0.5));
    }

    #[test]
    fn parameter_scale_is_validated_and_supported_by_the_macro() {
        let scaled = crate::parameter!("scaled", initial: 2.0, scale: 0.25);
        let layout = ParamLayout::new([scaled]).unwrap();
        assert_eq!(layout.specs()[0].scale(), Some(0.25));

        let error = ParamLayout::new([Parameter::free("bad").with_scale(0.0)]).unwrap_err();
        assert!(matches!(error, ParamError::InvalidScale { .. }));
    }

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
        assert_eq!(layout.initial_free_values(), vec![1.2, 0.5]);
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

        let values = layout.values(&[1.4, 0.2]).unwrap();
        assert_eq!(values.as_slice(), &[1.4, std::f64::consts::PI, 0.2]);
        assert_eq!(values.free_values(), vec![1.4, 0.2]);
    }

    #[test]
    fn free_values_can_be_generated_or_sampled_in_layout_order() {
        let layout = ParamLayout::new([
            Parameter::fixed("fixed", 8.0),
            Parameter::free("uniform").with_initial((-2.0, 4.0)),
            Parameter::free("value").with_initial(3.0),
            Parameter::free("default"),
        ])
        .unwrap();

        assert_eq!(layout.initial_free_values(), vec![1.0, 3.0, 0.0]);
        assert_eq!(layout.sample_initial(0), vec![1.6157656431461036, 3.0, 0.0]);
        assert_eq!(
            layout.free_values_with(|parameter| parameter.name().len() as f64),
            vec![7.0, 5.0, 7.0]
        );
        assert_eq!(
            layout
                .free_parameters()
                .map(Parameter::name)
                .collect::<Vec<_>>(),
            vec!["uniform", "value", "default"]
        );
    }

    #[test]
    fn deterministic_and_sampled_initial_values_share_initial_spec_semantics() {
        let layout = ParamLayout::new([
            Parameter::free("default"),
            Parameter::free("value").with_initial(2.5),
            Parameter::free("uniform").with_initial((-4.0, 6.0)),
        ])
        .unwrap();

        assert_eq!(layout.initial_free_values(), vec![0.0, 2.5, 1.0]);
        for seed in 0..32 {
            let sampled = layout.sample_initial(seed);
            assert_eq!(sampled[0], 0.0);
            assert_eq!(sampled[1], 2.5);
            assert!((-4.0..6.0).contains(&sampled[2]));
        }
    }

    #[test]
    fn classified_free_value_validation_separates_failure_kinds() {
        let layout = ParamLayout::new([
            Parameter::free("bounded").with_bounds(-1.0, 1.0),
            Parameter::free("phase")
                .with_bounds(0.0, std::f64::consts::TAU)
                .with_periodic(),
        ])
        .unwrap();

        assert_eq!(
            layout.validate_free_values_classified(&[0.0]),
            Err(FreeValueValidationError::Parameter(
                ParamError::FreeLengthMismatch {
                    expected: 2,
                    actual: 1,
                }
            ))
        );
        assert!(matches!(
            layout.validate_free_values_classified(&[f64::NAN, 0.0]),
            Err(FreeValueValidationError::NonFiniteValue { id, name, value })
                if id.index() == 0 && name == "bounded" && value.is_nan()
        ));
        assert_eq!(
            layout.validate_free_values_classified(&[2.0, 0.0]),
            Err(FreeValueValidationError::OutsideSupport {
                id: FreeParamId(0),
                name: "bounded".into(),
                value: 2.0,
            })
        );
        assert_eq!(
            layout.validate_free_values_classified(&[0.0, std::f64::consts::TAU]),
            Err(FreeValueValidationError::OutsideSupport {
                id: FreeParamId(1),
                name: "phase".into(),
                value: std::f64::consts::TAU,
            })
        );
        assert!(layout.validate_free_values_classified(&[1.0, 0.0]).is_ok());
    }

    #[test]
    fn periodic_domains_wrap_and_validate_without_changing_bounds() {
        let tau = std::f64::consts::TAU;
        let phase = Parameter::free("phase")
            .with_initial(0.25)
            .with_bounds(0.0, tau)
            .with_periodic();
        assert_eq!(phase.periodic_bounds(), Some((0.0, tau)));

        let layout = ParamLayout::new([phase]).unwrap();
        assert_eq!(
            layout.wrap_periodic_free_values(&[-0.25]).unwrap(),
            vec![tau - 0.25]
        );
        assert!(layout.validate_free_values(&[tau - 0.25]).is_ok());
        assert!(matches!(
            layout.validate_free_values(&[tau]),
            Err(ParamError::ValueOutsidePeriodicDomain { .. })
        ));
    }

    #[test]
    fn invalid_periodic_metadata_and_initial_values_are_rejected() {
        assert!(matches!(
            ParamLayout::new([Parameter::free("phase").with_periodic()]),
            Err(ParamError::PeriodicRequiresFiniteBounds { .. })
        ));
        assert!(matches!(
            ParamLayout::new([Parameter::free("phase")
                .with_initial(std::f64::consts::TAU)
                .with_bounds(0.0, std::f64::consts::TAU)
                .with_periodic(),]),
            Err(ParamError::ValueOutsidePeriodicDomain { .. })
        ));
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let err = ParamLayout::new([Parameter::free("x"), Parameter::fixed("x", 1.0)]).unwrap_err();
        assert_eq!(err, ParamError::DuplicateName("x".into()));
    }

    #[test]
    fn free_length_is_checked() {
        let layout = ParamLayout::new([Parameter::free("x"), Parameter::free("y")]).unwrap();
        let err = layout.values(&[1.0]).unwrap_err();
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

        let full = layout.values(&[1.4, 0.2]).unwrap();
        assert_eq!(full.as_slice(), &[-1.0, 1.4, 2.0, 0.2]);
        let mut rewritten = vec![0.0; layout.len()];
        layout
            .fill_full_from_free(&[1.5, 0.3], &mut rewritten)
            .unwrap();
        assert_eq!(rewritten, vec![-1.0, 1.5, 2.0, 0.3]);
    }

    #[test]
    fn values_only_mutate_free_parameters() {
        let layout = ParamLayout::new([
            Parameter::fixed("fixed", 1.0),
            Parameter::free("x"),
            Parameter::free("y"),
        ])
        .unwrap();
        let x_id = layout.id("x").unwrap();
        let y_id = layout.id("y").unwrap();
        let x_free = layout.free_id(x_id).unwrap().unwrap();
        let y_free = layout.free_id(y_id).unwrap().unwrap();

        let mut values = layout.default_values();
        values.set_free(x_free, 3.0).unwrap();
        values.set_free(y_free, 4.0).unwrap();

        assert_eq!(values.as_slice(), &[1.0, 3.0, 4.0]);
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
            ParamLayout::new([Parameter::free("x")
                .with_initial((-1.0, 1.0))
                .with_bounds(Some(0.0), Some(2.0))])
            .unwrap_err(),
            ParamError::InitialRangeOutOfBounds {
                name: "x".into(),
                min: -1.0,
                max: 1.0
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
    fn direct_and_registry_layouts_report_the_same_invalid_spec_errors() {
        let invalid = [
            Parameter::fixed("fixed", 2.0).with_bounds(0.0, 1.0),
            Parameter::free("default").with_bounds(1.0, 2.0),
            Parameter::free("value")
                .with_initial(2.0)
                .with_bounds(0.0, 1.0),
            Parameter::free("range")
                .with_initial((-1.0, 0.5))
                .with_bounds(0.0, 1.0),
            Parameter::free("periodic-value")
                .with_initial(std::f64::consts::TAU)
                .with_bounds(0.0, std::f64::consts::TAU)
                .with_periodic(),
        ];

        for parameter in invalid {
            let direct = ParamLayout::new([parameter.clone()]).unwrap_err();
            let mut registry = ParamRegistry::new();
            registry.register(parameter).unwrap();
            assert_eq!(registry.layout().unwrap_err(), direct);
        }
    }

    #[test]
    fn free_vector_lengths_are_checked() {
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
    }

    #[test]
    fn free_dimension_contract_is_shared_by_projection_operations() {
        let layout = ParamLayout::new([
            Parameter::fixed("fixed", 4.0),
            Parameter::free("x"),
            Parameter::free("y"),
        ])
        .unwrap();
        let expected = ParamError::FreeLengthMismatch {
            expected: 2,
            actual: 1,
        };

        assert_eq!(layout.values(&[1.0]).unwrap_err(), expected);
        assert_eq!(layout.validate_free_values(&[1.0]).unwrap_err(), expected);
        assert_eq!(
            layout.wrap_periodic_free_values(&[1.0]).unwrap_err(),
            expected
        );

        let mut values = layout.default_values();
        assert_eq!(values.set_free_values(&[1.0]).unwrap_err(), expected);
        assert_eq!(values.as_slice(), &[4.0, 0.0, 0.0]);
        assert_eq!(
            layout.validate_free_values_classified(&[1.0]),
            Err(FreeValueValidationError::Parameter(
                ParamError::FreeLengthMismatch {
                    expected: 2,
                    actual: 1,
                }
            ))
        );
    }

    #[test]
    fn cross_layout_projection_reorders_values_and_scatters_gradients() {
        let source = ParamLayout::new([
            Parameter::fixed("offset", -1.0),
            Parameter::free("x"),
            Parameter::free("y"),
        ])
        .unwrap();
        let target = ParamLayout::new([
            Parameter::free("y"),
            Parameter::fixed("scale", 2.0),
            Parameter::free("x"),
        ])
        .unwrap();
        let projection = target.projection_from(&source).unwrap();
        let source_values = source.values(&[1.0, 2.0]).unwrap();

        assert_eq!(
            projection.project(&source_values).unwrap().as_slice(),
            &[2.0, 2.0, 1.0]
        );

        let mut source_gradient = vec![10.0, 20.0];
        projection
            .scatter_add(&[0.5, 1.5], &mut source_gradient)
            .unwrap();
        assert_eq!(source_gradient, vec![11.5, 20.5]);
    }

    #[test]
    fn cross_layout_projection_rejects_missing_and_fixed_source_parameters() {
        let target = ParamLayout::new([Parameter::free("x")]).unwrap();
        let missing = ParamLayout::new([Parameter::free("y")]).unwrap();
        assert_eq!(
            target.projection_from(&missing).unwrap_err(),
            ParamError::UnknownName("x".into())
        );

        let fixed = ParamLayout::new([Parameter::fixed("x", 1.0)]).unwrap();
        assert!(matches!(
            target.projection_from(&fixed),
            Err(ParamError::ParameterConflict { name, .. }) if name == "x"
        ));
    }

    #[test]
    fn cross_layout_projection_supports_zero_free_layouts() {
        let source = ParamLayout::new([Parameter::fixed("source", 3.0)]).unwrap();
        let target = ParamLayout::new([Parameter::fixed("target", 4.0)]).unwrap();
        let projection = target.projection_from(&source).unwrap();
        let values = source.default_values();

        assert_eq!(projection.project(&values).unwrap().as_slice(), &[4.0]);
        let mut gradient = Vec::new();
        projection.scatter_add(&[], &mut gradient).unwrap();
    }

    #[test]
    fn cross_layout_projection_accepts_reordered_compatible_source_layouts() {
        let source = ParamLayout::new([Parameter::free("x")]).unwrap();
        let other =
            ParamLayout::new([Parameter::fixed("extra", 9.0), Parameter::fixed("x", 3.0)]).unwrap();
        let target = ParamLayout::new([Parameter::free("x")]).unwrap();
        let projection = target.projection_from(&source).unwrap();
        let values = other.default_values();

        assert_eq!(projection.project(&values).unwrap().as_slice(), &[3.0]);
        assert_eq!(values.as_slice(), &[9.0, 3.0]);
    }

    #[test]
    fn cross_layout_projection_rejects_missing_alternate_source_name() {
        let source = ParamLayout::new([Parameter::free("x")]).unwrap();
        let other = ParamLayout::new([Parameter::free("y")]).unwrap();
        let target = ParamLayout::new([Parameter::free("x")]).unwrap();
        let projection = target.projection_from(&source).unwrap();
        let values = other.values(&[7.0]).unwrap();

        assert_eq!(
            projection.project(&values).unwrap_err(),
            ParamError::UnknownName("x".into())
        );
        assert_eq!(values.as_slice(), &[7.0]);
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
