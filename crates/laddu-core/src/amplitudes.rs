use std::{
    fmt::{Debug, Display},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use auto_ops::*;
use dyn_clone::DynClone;
use nalgebra::{ComplexField, DVector};
use num::complex::Complex64;

use parking_lot::RwLock;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

static AMPLITUDE_INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_amplitude_id() -> u64 {
    AMPLITUDE_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

use crate::{
    data::{Dataset, DatasetMetadata, EventData},
    resources::{Cache, ParameterTransform, Parameters, Resources},
    LadduError, LadduResult, ParameterID, ReadWrite,
};

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

/// An enum containing either a named free parameter or a constant value.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Parameter {
    /// The name of the parameter.
    pub name: String,
    /// If `Some`, this parameter is fixed to the given value. If `None`, it is free.
    pub fixed: Option<f64>,
}

impl Parameter {
    /// Create a free (floating) parameter with the given name.
    pub fn free(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fixed: None,
        }
    }

    /// Create a fixed parameter with the given name and value.
    pub fn fixed(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            fixed: Some(value),
        }
    }

    /// An uninitialized parameter placeholder.
    pub fn uninit() -> Self {
        Self {
            name: String::new(),
            fixed: None,
        }
    }

    /// Is this parameter free?
    pub fn is_free(&self) -> bool {
        self.fixed.is_none()
    }

    /// Is this parameter fixed?
    pub fn is_fixed(&self) -> bool {
        self.fixed.is_some()
    }

    /// Get the parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Maintains naming used across the crate.
pub type ParameterLike = Parameter;

/// Shorthand for generating a named free parameter.
pub fn parameter(name: &str) -> Parameter {
    Parameter::free(name)
}

/// Shorthand for generating a fixed parameter with the given name and value.
pub fn constant(name: &str, value: f64) -> Parameter {
    Parameter::fixed(name, value)
}

/// Convenience macro for creating parameters. Usage:
/// `parameter!(\"name\")` for a free parameter, or `parameter!(\"name\", 1.0)` for a fixed one.
#[macro_export]
macro_rules! parameter {
    ($name:expr) => {
        $crate::amplitudes::Parameter::free($name)
    };
    ($name:expr, $value:expr) => {
        $crate::amplitudes::Parameter::fixed($name, $value)
    };
}

/// This is the only required trait for writing new amplitude-like structures for this
/// crate. Users need only implement the [`register`](Amplitude::register)
/// method to register parameters, cached values, and the amplitude itself with an input
/// [`Resources`] struct and the [`compute`](Amplitude::compute) method to actually carry
/// out the calculation. [`Amplitude`]-implementors are required to implement [`Clone`] and can
/// optionally implement a [`precompute`](Amplitude::precompute) method to calculate and
/// cache values which do not depend on free parameters.
#[typetag::serde(tag = "type")]
pub trait Amplitude: DynClone + Send + Sync {
    /// This method should be used to tell the [`Resources`] manager about all of
    /// the free parameters and cached values used by this [`Amplitude`]. It should end by
    /// returning an [`AmplitudeID`], which can be obtained from the
    /// [`Resources::register_amplitude`] method.
    ///
    /// [`register`](Amplitude::register) is invoked once when an amplitude is first added to a
    /// [`Manager`]. Use it to allocate parameter/cache state within [`Resources`] without assuming
    /// any dataset context.
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID>;
    /// Bind this [`Amplitude`] to a concrete [`Dataset`] by using the provided metadata to wire up
    /// [`Variable`](crate::utils::variables::Variable)s or other dataset-specific state. This will
    /// be invoked when a [`Model`] is loaded with data, after [`register`](Amplitude::register)
    /// has already succeeded. The default implementation is a no-op for amplitudes that do not
    /// depend on metadata.
    fn bind(&mut self, _metadata: &DatasetMetadata) -> LadduResult<()> {
        Ok(())
    }
    /// This method can be used to do some critical calculations ahead of time and
    /// store them in a [`Cache`]. These values can only depend on the data in an [`EventData`],
    /// not on any free parameters in the fit. This method is opt-in since it is not required
    /// to make a functioning [`Amplitude`].
    #[allow(unused_variables)]
    fn precompute(&self, event: &EventData, cache: &mut Cache) {}
    /// Evaluates [`Amplitude::precompute`] over ever [`EventData`] in a [`Dataset`].
    #[cfg(feature = "rayon")]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        dataset
            .events
            .par_iter()
            .zip(resources.caches.par_iter_mut())
            .for_each(|(event, cache)| {
                self.precompute(event, cache);
            })
    }
    /// Evaluates [`Amplitude::precompute`] over ever [`EventData`] in a [`Dataset`].
    #[cfg(not(feature = "rayon"))]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        dataset
            .events
            .iter()
            .zip(resources.caches.iter_mut())
            .for_each(|(event, cache)| self.precompute(event, cache))
    }
    /// This method constitutes the main machinery of an [`Amplitude`], returning the actual
    /// calculated value for a particular [`EventData`] and set of [`Parameters`]. See those
    /// structs, as well as [`Cache`], for documentation on their available methods. For the
    /// most part, [`EventData`]s can be interacted with via
    /// [`Variable`](crate::utils::variables::Variable)s, while [`Parameters`] and the
    /// [`Cache`] are more like key-value storage accessed by
    /// [`ParameterID`]s and several different types of cache
    /// IDs.
    fn compute(&self, parameters: &Parameters, event: &EventData, cache: &Cache) -> Complex64;

    /// This method yields the gradient of a particular [`Amplitude`] at a point specified
    /// by a particular [`EventData`] and set of [`Parameters`]. See those structs, as well as
    /// [`Cache`], for documentation on their available methods. For the most part,
    /// [`EventData`]s can be interacted with via [`Variable`](crate::utils::variables::Variable)s,
    /// while [`Parameters`] and the [`Cache`] are more like key-value storage accessed by
    /// [`ParameterID`]s and several different types of cache
    /// IDs. If the analytic version of the gradient is known, this method can be overwritten to
    /// improve performance for some derivative-using methods of minimization. The default
    /// implementation calculates a central finite difference across all parameters, regardless of
    /// whether or not they are used in the [`Amplitude`].
    ///
    /// In the future, it may be possible to automatically implement this with the indices of
    /// registered free parameters, but until then, the [`Amplitude::central_difference_with_indices`]
    /// method can be used to conveniently only calculate central differences for the parameters
    /// which are used by the [`Amplitude`].
    fn compute_gradient(
        &self,
        parameters: &Parameters,
        event: &EventData,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        self.central_difference_with_indices(
            &Vec::from_iter(0..parameters.len()),
            parameters,
            event,
            cache,
            gradient,
        )
    }

    /// A helper function to implement a central difference only on indices which correspond to
    /// free parameters in the [`Amplitude`]. For example, if an [`Amplitude`] contains free
    /// parameters registered to indices 1, 3, and 5 of the its internal parameters array, then
    /// running this with those indices will compute a central finite difference derivative for
    /// those coordinates only, since the rest can be safely assumed to be zero.
    fn central_difference_with_indices(
        &self,
        indices: &[usize],
        parameters: &Parameters,
        event: &EventData,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let x = parameters.parameters.to_owned();
        let constants = parameters.constants.to_owned();
        let h: DVector<f64> = x
            .iter()
            .map(|&xi| f64::cbrt(f64::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        for i in indices {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[*i] += h[*i];
            x_minus[*i] -= h[*i];
            let f_plus = self.compute(&Parameters::new(&x_plus, &constants), event, cache);
            let f_minus = self.compute(&Parameters::new(&x_minus, &constants), event, cache);
            gradient[*i] = (f_plus - f_minus) / (2.0 * h[*i]);
        }
    }

    /// Convenience helper to wrap an amplitude into an [`Expression`].
    ///
    /// This allows amplitude constructors to return `LadduResult<Expression>` without duplicating
    /// boxing/registration boilerplate.
    fn into_expression(self) -> LadduResult<Expression>
    where
        Self: Sized + 'static,
    {
        Expression::from_amplitude(Box::new(self))
    }
}

/// Utility function to calculate a central finite difference gradient.
pub fn central_difference<F: Fn(&[f64]) -> f64>(parameters: &[f64], func: F) -> DVector<f64> {
    let mut gradient = DVector::zeros(parameters.len());
    let x = parameters.to_owned();
    let h: DVector<f64> = x
        .iter()
        .map(|&xi| f64::cbrt(f64::EPSILON) * (xi.abs() + 1.0))
        .collect::<Vec<_>>()
        .into();
    for i in 0..parameters.len() {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += h[i];
        x_minus[i] -= h[i];
        let f_plus = func(&x_plus);
        let f_minus = func(&x_minus);
        gradient[i] = (f_plus - f_minus) / (2.0 * h[i]);
    }
    gradient
}

dyn_clone::clone_trait_object!(Amplitude);

/// A helper struct that contains the value of each amplitude for a particular event
#[derive(Debug)]
pub struct AmplitudeValues(pub Vec<Complex64>);

/// A helper struct that contains the gradient of each amplitude for a particular event
#[derive(Debug)]
pub struct GradientValues(pub usize, pub Vec<DVector<Complex64>>);

/// A tag which refers to a registered [`Amplitude`]. This is the base object which can be used to
/// build [`Expression`]s and should be obtained from the [`Resources::register`] method.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct AmplitudeID(pub(crate) String, pub(crate) usize);

impl Display for AmplitudeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(id={})", self.0, self.1)
    }
}

/// A holder struct that owns both an expression tree and the registered amplitudes.
#[allow(missing_docs)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Expression {
    registry: ExpressionRegistry,
    tree: ExpressionNode,
}

impl ReadWrite for Expression {
    fn create_null() -> Self {
        Self {
            registry: ExpressionRegistry::default(),
            tree: ExpressionNode::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
#[derive(Default)]
pub struct ExpressionRegistry {
    amplitudes: Vec<Box<dyn Amplitude>>,
    amplitude_names: Vec<String>,
    amplitude_ids: Vec<u64>,
    resources: Resources,
}

impl ExpressionRegistry {
    fn singleton(mut amplitude: Box<dyn Amplitude>) -> LadduResult<Self> {
        let mut resources = Resources::default();
        let aid = amplitude.register(&mut resources)?;
        let amp_id = next_amplitude_id();
        Ok(Self {
            amplitudes: vec![amplitude],
            amplitude_names: vec![aid.0],
            amplitude_ids: vec![amp_id],
            resources,
        })
    }

    fn merge(&self, other: &Self) -> LadduResult<(Self, Vec<usize>, Vec<usize>)> {
        let mut resources = Resources::default();
        let mut amplitudes = Vec::new();
        let mut amplitude_names = Vec::new();
        let mut amplitude_ids = Vec::new();
        let mut name_to_index = std::collections::HashMap::new();

        let mut left_map = Vec::with_capacity(self.amplitudes.len());
        for ((amp, name), amp_id) in self
            .amplitudes
            .iter()
            .zip(&self.amplitude_names)
            .zip(&self.amplitude_ids)
        {
            let mut cloned_amp = dyn_clone::clone_box(&**amp);
            let aid = cloned_amp.register(&mut resources)?;
            amplitudes.push(cloned_amp);
            amplitude_names.push(name.clone());
            amplitude_ids.push(*amp_id);
            name_to_index.insert(name.clone(), aid.1);
            left_map.push(aid.1);
        }

        let mut right_map = Vec::with_capacity(other.amplitudes.len());
        for ((amp, name), amp_id) in other
            .amplitudes
            .iter()
            .zip(&other.amplitude_names)
            .zip(&other.amplitude_ids)
        {
            if let Some(existing) = name_to_index.get(name) {
                let existing_amp_id = amplitude_ids[*existing];
                if existing_amp_id != *amp_id {
                    return Err(LadduError::Custom(format!(
                        "Amplitude name \"{name}\" refers to different underlying amplitudes; rename to avoid conflicts"
                    )));
                }
                right_map.push(*existing);
                continue;
            }
            let mut cloned_amp = dyn_clone::clone_box(&**amp);
            let aid = cloned_amp.register(&mut resources)?;
            amplitudes.push(cloned_amp);
            amplitude_names.push(name.clone());
            amplitude_ids.push(*amp_id);
            name_to_index.insert(name.clone(), aid.1);
            right_map.push(aid.1);
        }

        Ok((
            Self {
                amplitudes,
                amplitude_names,
                amplitude_ids,
                resources,
            },
            left_map,
            right_map,
        ))
    }

    fn rebuild_with_transform(&self, transform: ParameterTransform) -> LadduResult<Self> {
        let mut resources = Resources::with_transform(transform);
        let mut amplitudes = Vec::new();
        let mut amplitude_names = Vec::new();
        let mut amplitude_ids = Vec::new();
        for ((amp, name), amp_id) in self
            .amplitudes
            .iter()
            .zip(&self.amplitude_names)
            .zip(&self.amplitude_ids)
        {
            let mut cloned_amp = dyn_clone::clone_box(&**amp);
            let aid = cloned_amp.register(&mut resources)?;
            if aid.0 != *name {
                return Err(LadduError::ParameterConflict {
                    name: aid.0,
                    reason: "amplitude renamed during rebuild".to_string(),
                });
            }
            amplitudes.push(cloned_amp);
            amplitude_names.push(name.clone());
            amplitude_ids.push(*amp_id);
        }
        Ok(Self {
            amplitudes,
            amplitude_names,
            amplitude_ids,
            resources,
        })
    }
}

/// Expression tree used by [`Expression`].
#[allow(missing_docs)]
#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub enum ExpressionNode {
    #[default]
    /// A expression equal to zero.
    Zero,
    /// A expression equal to one.
    One,
    /// A registered [`Amplitude`] referenced by index.
    Amp(usize),
    /// The sum of two [`ExpressionNode`]s.
    Add(Box<ExpressionNode>, Box<ExpressionNode>),
    /// The difference of two [`ExpressionNode`]s.
    Sub(Box<ExpressionNode>, Box<ExpressionNode>),
    /// The product of two [`ExpressionNode`]s.
    Mul(Box<ExpressionNode>, Box<ExpressionNode>),
    /// The division of two [`ExpressionNode`]s.
    Div(Box<ExpressionNode>, Box<ExpressionNode>),
    /// The additive inverse of an [`ExpressionNode`].
    Neg(Box<ExpressionNode>),
    /// The real part of an [`ExpressionNode`].
    Real(Box<ExpressionNode>),
    /// The imaginary part of an [`ExpressionNode`].
    Imag(Box<ExpressionNode>),
    /// The complex conjugate of an [`ExpressionNode`].
    Conj(Box<ExpressionNode>),
    /// The absolute square of an [`ExpressionNode`].
    NormSqr(Box<ExpressionNode>),
}

impl ExpressionNode {
    fn remap(&self, mapping: &[usize]) -> Self {
        match self {
            Self::Amp(idx) => Self::Amp(mapping[*idx]),
            Self::Add(a, b) => Self::Add(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Sub(a, b) => Self::Sub(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Mul(a, b) => Self::Mul(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Div(a, b) => Self::Div(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Neg(a) => Self::Neg(Box::new(a.remap(mapping))),
            Self::Real(a) => Self::Real(Box::new(a.remap(mapping))),
            Self::Imag(a) => Self::Imag(Box::new(a.remap(mapping))),
            Self::Conj(a) => Self::Conj(Box::new(a.remap(mapping))),
            Self::NormSqr(a) => Self::NormSqr(Box::new(a.remap(mapping))),
            Self::Zero => Self::Zero,
            Self::One => Self::One,
        }
    }

    /// Evaluate an [`ExpressionNode`] over a single event using calculated [`AmplitudeValues`]
    ///
    /// This method parses the underlying [`ExpressionNode`] but doesn't actually calculate the values
    /// from the [`Amplitude`]s themselves.
    pub fn evaluate(&self, amplitude_values: &[Complex64]) -> Complex64 {
        match self {
            ExpressionNode::Amp(idx) => amplitude_values[*idx],
            ExpressionNode::Add(a, b) => {
                a.evaluate(amplitude_values) + b.evaluate(amplitude_values)
            }
            ExpressionNode::Sub(a, b) => {
                a.evaluate(amplitude_values) - b.evaluate(amplitude_values)
            }
            ExpressionNode::Mul(a, b) => {
                a.evaluate(amplitude_values) * b.evaluate(amplitude_values)
            }
            ExpressionNode::Div(a, b) => {
                a.evaluate(amplitude_values) / b.evaluate(amplitude_values)
            }
            ExpressionNode::Neg(a) => -a.evaluate(amplitude_values),
            ExpressionNode::Real(a) => Complex64::new(a.evaluate(amplitude_values).re, 0.0),
            ExpressionNode::Imag(a) => Complex64::new(a.evaluate(amplitude_values).im, 0.0),
            ExpressionNode::Conj(a) => a.evaluate(amplitude_values).conj(),
            ExpressionNode::NormSqr(a) => {
                let value = a.evaluate(amplitude_values);
                Complex64::new(value.norm_sqr(), 0.0)
            }
            ExpressionNode::Zero => Complex64::ZERO,
            ExpressionNode::One => Complex64::ONE,
        }
    }

    /// Evaluate the gradient of an [`ExpressionNode`] over a single event using calculated [`AmplitudeValues`]
    ///
    /// This method parses the underlying [`ExpressionNode`] but doesn't actually calculate the
    /// gradient from the [`Amplitude`]s themselves.
    pub fn evaluate_gradient(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        match self {
            ExpressionNode::Amp(idx) => gradient_values[*idx].clone(),
            ExpressionNode::Add(a, b) => {
                a.evaluate_gradient(amplitude_values, gradient_values)
                    + b.evaluate_gradient(amplitude_values, gradient_values)
            }
            ExpressionNode::Sub(a, b) => {
                a.evaluate_gradient(amplitude_values, gradient_values)
                    - b.evaluate_gradient(amplitude_values, gradient_values)
            }
            ExpressionNode::Mul(a, b) => {
                let f_a = a.evaluate(amplitude_values);
                let f_b = b.evaluate(amplitude_values);
                b.evaluate_gradient(amplitude_values, gradient_values)
                    .map(|g| g * f_a)
                    + a.evaluate_gradient(amplitude_values, gradient_values)
                        .map(|g| g * f_b)
            }
            ExpressionNode::Div(a, b) => {
                let f_a = a.evaluate(amplitude_values);
                let f_b = b.evaluate(amplitude_values);
                (a.evaluate_gradient(amplitude_values, gradient_values)
                    .map(|g| g * f_b)
                    - b.evaluate_gradient(amplitude_values, gradient_values)
                        .map(|g| g * f_a))
                    / (f_b * f_b)
            }
            ExpressionNode::Neg(a) => -a.evaluate_gradient(amplitude_values, gradient_values),
            ExpressionNode::Real(a) => a
                .evaluate_gradient(amplitude_values, gradient_values)
                .map(|g| Complex64::new(g.re, 0.0)),
            ExpressionNode::Imag(a) => a
                .evaluate_gradient(amplitude_values, gradient_values)
                .map(|g| Complex64::new(g.im, 0.0)),
            ExpressionNode::Conj(a) => a
                .evaluate_gradient(amplitude_values, gradient_values)
                .map(|g| g.conj()),
            ExpressionNode::NormSqr(a) => {
                let conj_f_a = a.evaluate(amplitude_values).conjugate();
                a.evaluate_gradient(amplitude_values, gradient_values)
                    .map(|g| Complex64::new(2.0 * (g * conj_f_a).re, 0.0))
            }
            ExpressionNode::Zero | ExpressionNode::One => {
                let max_dim = gradient_values.first().map(|g| g.len()).unwrap_or(0);
                DVector::zeros(max_dim)
            }
        }
    }
}

impl Expression {
    /// Build an [`Expression`] from a single [`Amplitude`].
    pub fn from_amplitude(amplitude: Box<dyn Amplitude>) -> LadduResult<Self> {
        let registry = ExpressionRegistry::singleton(amplitude)?;
        Ok(Self {
            tree: ExpressionNode::Amp(0),
            registry,
        })
    }

    /// Create an expression representing zero, the additive identity.
    pub fn zero() -> Self {
        Self {
            registry: ExpressionRegistry::default(),
            tree: ExpressionNode::Zero,
        }
    }

    /// Create an expression representing one, the multiplicative identity.
    pub fn one() -> Self {
        Self {
            registry: ExpressionRegistry::default(),
            tree: ExpressionNode::One,
        }
    }

    fn binary_op(
        a: &Expression,
        b: &Expression,
        build: impl Fn(Box<ExpressionNode>, Box<ExpressionNode>) -> ExpressionNode,
    ) -> Expression {
        let (registry, left_map, right_map) = a
            .registry
            .merge(&b.registry)
            .expect("merging expression registries should not fail");
        let left_tree = a.tree.remap(&left_map);
        let right_tree = b.tree.remap(&right_map);
        Expression {
            registry,
            tree: build(Box::new(left_tree), Box::new(right_tree)),
        }
    }

    fn unary_op(a: &Expression, build: impl Fn(Box<ExpressionNode>) -> ExpressionNode) -> Self {
        Expression {
            registry: a.registry.clone(),
            tree: build(Box::new(a.tree.clone())),
        }
    }

    /// Get the list of parameter names in the order they appear in the underlying resources.
    pub fn parameters(&self) -> Vec<String> {
        self.registry.resources.parameter_names()
    }

    /// Get the list of free parameter names.
    pub fn free_parameters(&self) -> Vec<String> {
        self.registry.resources.free_parameter_names()
    }

    /// Get the list of fixed parameter names.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.registry.resources.fixed_parameter_names()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.registry.resources.n_free_parameters()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.registry.resources.n_fixed_parameters()
    }

    /// Total number of parameters.
    pub fn n_params(&self) -> usize {
        self.registry.resources.n_parameters()
    }

    fn with_transform(&self, transform: ParameterTransform) -> LadduResult<Self> {
        let merged = self
            .registry
            .resources
            .parameter_overrides
            .merged(&transform);
        let registry = self.registry.rebuild_with_transform(merged)?;
        Ok(Self {
            registry,
            tree: self.tree.clone(),
        })
    }

    fn assert_parameter_exists(&self, name: &str) -> LadduResult<()> {
        if self.parameters().iter().any(|p| p == name) {
            Ok(())
        } else {
            Err(LadduError::UnregisteredParameter {
                name: name.to_string(),
                reason: "parameter not found".to_string(),
            })
        }
    }

    /// Return a new [`Expression`] with the given parameter fixed to a value.
    pub fn fix(&self, name: &str, value: f64) -> LadduResult<Self> {
        self.assert_parameter_exists(name)?;
        let mut transform = ParameterTransform::default();
        transform.fixed.insert(name.to_string(), value);
        self.with_transform(transform)
    }

    /// Return a new [`Expression`] with the given parameter freed.
    pub fn free(&self, name: &str) -> LadduResult<Self> {
        self.assert_parameter_exists(name)?;
        let mut transform = ParameterTransform::default();
        transform.freed.insert(name.to_string());
        self.with_transform(transform)
    }

    /// Return a new [`Expression`] with a single parameter renamed.
    pub fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<Self> {
        self.assert_parameter_exists(old)?;
        if old == new {
            return Ok(self.clone());
        }
        if self.parameters().iter().any(|p| p == new) {
            return Err(LadduError::ParameterConflict {
                name: new.to_string(),
                reason: "rename target already exists".to_string(),
            });
        }
        let mut transform = ParameterTransform::default();
        transform.renames.insert(old.to_string(), new.to_string());
        self.with_transform(transform)
    }

    /// Return a new [`Expression`] with several parameters renamed.
    pub fn rename_parameters(
        &self,
        mapping: &std::collections::HashMap<String, String>,
    ) -> LadduResult<Self> {
        for old in mapping.keys() {
            self.assert_parameter_exists(old)?;
        }
        let mut final_names: std::collections::HashSet<String> =
            self.parameters().into_iter().collect();
        for (old, new) in mapping {
            if old == new {
                continue;
            }
            final_names.remove(old);
            if final_names.contains(new) {
                return Err(LadduError::ParameterConflict {
                    name: new.clone(),
                    reason: "rename target already exists".to_string(),
                });
            }
            final_names.insert(new.clone());
        }
        let mut transform = ParameterTransform::default();
        for (old, new) in mapping {
            transform.renames.insert(old.clone(), new.clone());
        }
        self.with_transform(transform)
    }

    /// Load an [`Expression`] against a dataset, binding amplitudes and reserving caches.
    pub fn load(&self, dataset: &Arc<Dataset>) -> LadduResult<Evaluator> {
        let mut resources = self.registry.resources.clone();
        let metadata = dataset.metadata();
        resources.reserve_cache(dataset.n_events());
        let mut amplitudes: Vec<Box<dyn Amplitude>> = self
            .registry
            .amplitudes
            .iter()
            .map(|amp| dyn_clone::clone_box(&**amp))
            .collect();
        {
            for amplitude in amplitudes.iter_mut() {
                amplitude.bind(metadata)?;
                amplitude.precompute_all(dataset, &mut resources);
            }
        }
        Ok(Evaluator {
            amplitudes,
            resources: Arc::new(RwLock::new(resources)),
            dataset: dataset.clone(),
            expression: self.tree.clone(),
            registry: self.registry.clone(),
        })
    }

    /// Takes the real part of the given [`Expression`].
    pub fn real(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Real)
    }
    /// Takes the imaginary part of the given [`Expression`].
    pub fn imag(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Imag)
    }
    /// Takes the complex conjugate of the given [`Expression`].
    pub fn conj(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Conj)
    }
    /// Takes the absolute square of the given [`Expression`].
    pub fn norm_sqr(&self) -> Self {
        Self::unary_op(self, ExpressionNode::NormSqr)
    }

    fn write_tree(
        &self,
        t: &ExpressionNode,
        f: &mut std::fmt::Formatter<'_>,
        parent_prefix: &str,
        immediate_prefix: &str,
        parent_suffix: &str,
    ) -> std::fmt::Result {
        let display_string = match t {
            ExpressionNode::Amp(idx) => {
                let name = self
                    .registry
                    .amplitude_names
                    .get(*idx)
                    .cloned()
                    .unwrap_or_else(|| "<unregistered>".to_string());
                format!("{name}(id={idx})")
            }
            ExpressionNode::Add(_, _) => "+".to_string(),
            ExpressionNode::Sub(_, _) => "-".to_string(),
            ExpressionNode::Mul(_, _) => "×".to_string(),
            ExpressionNode::Div(_, _) => "÷".to_string(),
            ExpressionNode::Neg(_) => "-".to_string(),
            ExpressionNode::Real(_) => "Re".to_string(),
            ExpressionNode::Imag(_) => "Im".to_string(),
            ExpressionNode::Conj(_) => "*".to_string(),
            ExpressionNode::NormSqr(_) => "NormSqr".to_string(),
            ExpressionNode::Zero => "0".to_string(),
            ExpressionNode::One => "1".to_string(),
        };
        writeln!(f, "{}{}{}", parent_prefix, immediate_prefix, display_string)?;
        match t {
            ExpressionNode::Amp(_) | ExpressionNode::Zero | ExpressionNode::One => {}
            ExpressionNode::Add(a, b)
            | ExpressionNode::Sub(a, b)
            | ExpressionNode::Mul(a, b)
            | ExpressionNode::Div(a, b) => {
                let terms = [a, b];
                let mut it = terms.iter().peekable();
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                while let Some(child) = it.next() {
                    match it.peek() {
                        Some(_) => self.write_tree(child, f, &child_prefix, "├─ ", "│  "),
                        None => self.write_tree(child, f, &child_prefix, "└─ ", "   "),
                    }?;
                }
            }
            ExpressionNode::Neg(a)
            | ExpressionNode::Real(a)
            | ExpressionNode::Imag(a)
            | ExpressionNode::Conj(a)
            | ExpressionNode::NormSqr(a) => {
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                self.write_tree(a, f, &child_prefix, "└─ ", "   ")?;
            }
        }
        Ok(())
    }
}

impl Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(&self.tree, f, "", "", "")
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(&self.tree, f, "", "", "")
    }
}

#[rustfmt::skip]
impl_op_ex!(+ |a: &Expression, b: &Expression| -> Expression {
    Expression::binary_op(a, b, ExpressionNode::Add)
});
#[rustfmt::skip]
impl_op_ex!(- |a: &Expression, b: &Expression| -> Expression {
    Expression::binary_op(a, b, ExpressionNode::Sub)
});
#[rustfmt::skip]
impl_op_ex!(* |a: &Expression, b: &Expression| -> Expression {
    Expression::binary_op(a, b, ExpressionNode::Mul)
});
#[rustfmt::skip]
impl_op_ex!(/ |a: &Expression, b: &Expression| -> Expression {
    Expression::binary_op(a, b, ExpressionNode::Div)
});
#[rustfmt::skip]
impl_op_ex!(- |a: &Expression| -> Expression {
    Expression::unary_op(a, ExpressionNode::Neg)
});

/// Evaluator for [`Expression`] that mirrors the existing evaluator behavior.
#[allow(missing_docs)]
#[derive(Clone)]
pub struct Evaluator {
    pub amplitudes: Vec<Box<dyn Amplitude>>,
    pub resources: Arc<RwLock<Resources>>,
    pub dataset: Arc<Dataset>,
    pub expression: ExpressionNode,
    registry: ExpressionRegistry,
}

#[allow(missing_docs)]
impl Evaluator {
    /// Get the list of parameter names in the order they appear in the [`Evaluator::evaluate`]
    /// method.
    pub fn parameters(&self) -> Vec<String> {
        self.resources.read().parameter_names()
    }

    /// Get the list of free parameter names.
    pub fn free_parameters(&self) -> Vec<String> {
        self.resources.read().free_parameter_names()
    }

    /// Get the list of fixed parameter names.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.resources.read().fixed_parameter_names()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.resources.read().n_free_parameters()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.resources.read().n_fixed_parameters()
    }

    /// Total number of parameters.
    pub fn n_params(&self) -> usize {
        self.resources.read().n_parameters()
    }

    fn as_expression(&self) -> Expression {
        Expression {
            registry: self.registry.clone(),
            tree: self.expression.clone(),
        }
    }

    /// Return a new [`Evaluator`] with the given parameter fixed to a value.
    pub fn fix(&self, name: &str, value: f64) -> LadduResult<Self> {
        self.as_expression().fix(name, value)?.load(&self.dataset)
    }

    /// Return a new [`Evaluator`] with the given parameter freed.
    pub fn free(&self, name: &str) -> LadduResult<Self> {
        self.as_expression().free(name)?.load(&self.dataset)
    }

    /// Return a new [`Evaluator`] with a single parameter renamed.
    pub fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<Self> {
        self.as_expression()
            .rename_parameter(old, new)?
            .load(&self.dataset)
    }

    /// Return a new [`Evaluator`] with several parameters renamed.
    pub fn rename_parameters(
        &self,
        mapping: &std::collections::HashMap<String, String>,
    ) -> LadduResult<Self> {
        self.as_expression()
            .rename_parameters(mapping)?
            .load(&self.dataset)
    }

    /// Activate an [`Amplitude`] by name, skipping missing entries.
    pub fn activate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().activate(name);
    }
    /// Activate an [`Amplitude`] by name and return an error if it is missing.
    pub fn activate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().activate_strict(name)
    }

    /// Activate several [`Amplitude`]s by name, skipping missing entries.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().activate_many(names);
    }
    /// Activate several [`Amplitude`]s by name and return an error if any are missing.
    pub fn activate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().activate_many_strict(names)
    }

    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&self) {
        self.resources.write().activate_all();
    }

    /// Dectivate an [`Amplitude`] by name, skipping missing entries.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().deactivate(name);
    }

    /// Dectivate an [`Amplitude`] by name and return an error if it is missing.
    pub fn deactivate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().deactivate_strict(name)
    }

    /// Deactivate several [`Amplitude`]s by name, skipping missing entries.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().deactivate_many(names);
    }
    /// Dectivate several [`Amplitude`]s by name and return an error if any are missing.
    pub fn deactivate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().deactivate_many_strict(names)
    }

    /// Deactivate all registered [`Amplitude`]s.
    pub fn deactivate_all(&self) {
        self.resources.write().deactivate_all();
    }

    /// Isolate an [`Amplitude`] by name (deactivate the rest), skipping missing entries.
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().isolate(name);
    }

    /// Isolate an [`Amplitude`] by name (deactivate the rest) and return an error if it is missing.
    pub fn isolate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().isolate_strict(name)
    }

    /// Isolate several [`Amplitude`]s by name (deactivate the rest), skipping missing entries.
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().isolate_many(names);
    }

    /// Isolate several [`Amplitude`]s by name (deactivate the rest) and return an error if any are missing.
    pub fn isolate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().isolate_many_strict(names)
    }

    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`Evaluator::evaluate`] instead.
    pub fn evaluate_local(&self, parameters: &[f64]) -> Vec<Complex64> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        #[cfg(feature = "rayon")]
        {
            self.dataset
                .events
                .par_iter()
                .zip(resources.caches.par_iter())
                .map(|(event, cache)| {
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();
                    self.expression.evaluate(&amplitude_values)
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.dataset
                .events
                .iter()
                .zip(resources.caches.iter())
                .map(|(event, cache)| {
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();
                    self.expression.evaluate(&amplitude_values)
                })
                .collect()
        }
    }

    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`Evaluator::evaluate`] instead.
    #[cfg(feature = "mpi")]
    fn evaluate_mpi(&self, parameters: &[f64], world: &SimpleCommunicator) -> Vec<Complex64> {
        let local_evaluation = self.evaluate_local(parameters);
        let n_events = self.dataset.n_events();
        let mut buffer: Vec<Complex64> = vec![Complex64::ZERO; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_evaluation, &mut partitioned_buffer);
        }
        buffer
    }

    /// Evaluate the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    pub fn evaluate(&self, parameters: &[f64]) -> Vec<Complex64> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.evaluate_mpi(parameters, &world);
            }
        }
        self.evaluate_local(parameters)
    }

    /// See [`Evaluator::evaluate_local`]. This method evaluates over a subset of events rather
    /// than all events in the total dataset.
    pub fn evaluate_batch_local(&self, parameters: &[f64], indices: &[usize]) -> Vec<Complex64> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        #[cfg(feature = "rayon")]
        {
            indices
                .par_iter()
                .map(|&idx| {
                    let event = &self.dataset.events[idx];
                    let cache = &resources.caches[idx];
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();
                    self.expression.evaluate(&amplitude_values)
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            indices
                .iter()
                .map(|&idx| {
                    let event = &self.dataset.events[idx];
                    let cache = &resources.caches[idx];
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();
                    self.expression.evaluate(&amplitude_values)
                })
                .collect()
        }
    }

    /// See [`Evaluator::evaluate_mpi`]. This method evaluates over a subset of events rather
    /// than all events in the total dataset.
    #[cfg(feature = "mpi")]
    fn evaluate_batch_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> Vec<Complex64> {
        let total = self.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let local_evaluation = self.evaluate_batch_local(parameters, &locals);
        world.all_gather_batched_partitioned(&local_evaluation, indices, total, None)
    }

    /// Evaluate the stored [`Expression`] over a subset of events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters. See also [`Expression::evaluate`].
    pub fn evaluate_batch(&self, parameters: &[f64], indices: &[usize]) -> Vec<Complex64> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.evaluate_batch_mpi(parameters, indices, &world);
            }
        }
        self.evaluate_batch_local(parameters, indices)
    }

    /// Evaluate the gradient of the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`Evaluator::evaluate_gradient`] instead.
    pub fn evaluate_gradient_local(&self, parameters: &[f64]) -> Vec<DVector<Complex64>> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        #[cfg(feature = "rayon")]
        {
            self.dataset
                .events
                .par_iter()
                .zip(resources.caches.par_iter())
                .map(|(event, cache)| {
                    let mut gradient_values =
                        vec![DVector::zeros(parameters.len()); self.amplitudes.len()];
                    self.amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .zip(gradient_values.iter_mut())
                        .for_each(|((amp, active), grad)| {
                            if *active {
                                amp.compute_gradient(&parameters, event, cache, grad)
                            }
                        });
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();
                    self.expression
                        .evaluate_gradient(&amplitude_values, &gradient_values)
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.dataset
                .events
                .iter()
                .zip(resources.caches.iter())
                .map(|(event, cache)| {
                    let mut gradient_values =
                        vec![DVector::zeros(parameters.len()); self.amplitudes.len()];
                    self.amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .zip(gradient_values.iter_mut())
                        .for_each(|((amp, active), grad)| {
                            if *active {
                                amp.compute_gradient(&parameters, event, cache, grad)
                            }
                        });
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();

                    self.expression
                        .evaluate_gradient(&amplitude_values, &gradient_values)
                })
                .collect()
        }
    }

    /// Evaluate the gradient of the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`Evaluator::evaluate_gradient`] instead.
    #[cfg(feature = "mpi")]
    fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> Vec<DVector<Complex64>> {
        let local_evaluation = self.evaluate_gradient_local(parameters);
        let n_events = self.dataset.n_events();
        let mut buffer: Vec<Complex64> = vec![Complex64::ZERO; n_events * parameters.len()];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(
                &local_evaluation
                    .iter()
                    .flat_map(|v| v.data.as_vec())
                    .copied()
                    .collect::<Vec<_>>(),
                &mut partitioned_buffer,
            );
        }
        buffer
            .chunks(parameters.len())
            .map(|chunk| DVector::from_row_slice(chunk))
            .collect()
    }

    /// Evaluate the gradient of the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters.
    pub fn evaluate_gradient(&self, parameters: &[f64]) -> Vec<DVector<Complex64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.evaluate_gradient_mpi(parameters, &world);
            }
        }
        self.evaluate_gradient_local(parameters)
    }

    /// See [`Evaluator::evaluate_gradient_local`]. This method evaluates over a subset
    /// of events rather than all events in the total dataset.
    pub fn evaluate_gradient_batch_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> Vec<DVector<Complex64>> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        #[cfg(feature = "rayon")]
        {
            indices
                .par_iter()
                .map(|&idx| {
                    let event = &self.dataset.events[idx];
                    let cache = &resources.caches[idx];
                    let mut gradient_values =
                        vec![DVector::zeros(parameters.len()); self.amplitudes.len()];
                    self.amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .zip(gradient_values.iter_mut())
                        .for_each(|((amp, active), grad)| {
                            if *active {
                                amp.compute_gradient(&parameters, event, cache, grad)
                            }
                        });
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();
                    self.expression
                        .evaluate_gradient(&amplitude_values, &gradient_values)
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            indices
                .iter()
                .map(|&idx| {
                    let event = &self.dataset.events[idx];
                    let cache = &resources.caches[idx];
                    let mut gradient_values =
                        vec![DVector::zeros(parameters.len()); self.amplitudes.len()];
                    self.amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .zip(gradient_values.iter_mut())
                        .for_each(|((amp, active), grad)| {
                            if *active {
                                amp.compute_gradient(&parameters, event, cache, grad)
                            }
                        });
                    let amplitude_values: Vec<Complex64> = self
                        .amplitudes
                        .iter()
                        .zip(resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&parameters, event, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect();

                    self.expression
                        .evaluate_gradient(&amplitude_values, &gradient_values)
                })
                .collect()
        }
    }

    /// See [`Evaluator::evaluate_gradient_mpi`]. This method evaluates over a subset
    /// of events rather than all events in the total dataset.
    #[cfg(feature = "mpi")]
    fn evaluate_gradient_batch_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> Vec<DVector<Complex64>> {
        let total = self.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let flattened_local_evaluation = self
            .evaluate_gradient_batch_local(parameters, &locals)
            .iter()
            .flat_map(|g| g.data.as_vec().to_vec())
            .collect::<Vec<Complex64>>();
        world
            .all_gather_batched_partitioned(
                &flattened_local_evaluation,
                indices,
                total,
                Some(parameters.len()),
            )
            .chunks(parameters.len())
            .map(DVector::from_row_slice)
            .collect()
    }

    /// Evaluate the gradient of the stored [`Expression`] over a subset of the
    /// events in the [`Dataset`] stored by the [`Evaluator`] with the given values
    /// for free parameters. See also [`Expression::evaluate_gradient`].
    pub fn evaluate_gradient_batch(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> Vec<DVector<Complex64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.evaluate_gradient_batch_mpi(parameters, indices, &world);
            }
        }
        self.evaluate_gradient_batch_local(parameters, indices)
    }
}

/// A testing [`Amplitude`].
#[derive(Clone, Serialize, Deserialize)]
pub struct TestAmplitude {
    name: String,
    re: ParameterLike,
    pid_re: ParameterID,
    im: ParameterLike,
    pid_im: ParameterID,
}

impl TestAmplitude {
    /// Create a new testing [`Amplitude`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(name: &str, re: ParameterLike, im: ParameterLike) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            re,
            pid_re: Default::default(),
            im,
            pid_im: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for TestAmplitude {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_re = resources.register_parameter(&self.re)?;
        self.pid_im = resources.register_parameter(&self.im)?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, event: &EventData, _cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid_re), parameters.get(self.pid_im)) * event.p4s[0].e()
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        event: &EventData,
        _cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let ParameterID::Parameter(ind) = self.pid_re {
            gradient[ind] = Complex64::ONE * event.p4s[0].e();
        }
        if let ParameterID::Parameter(ind) = self.pid_im {
            gradient[ind] = Complex64::I * event.p4s[0].e();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::data::{test_dataset, test_event, DatasetMetadata};

    use super::*;
    use crate::{
        data::EventData,
        resources::{Cache, ParameterID, Parameters, Resources},
    };
    use approx::assert_relative_eq;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Serialize, Deserialize)]
    pub struct ComplexScalar {
        name: String,
        re: ParameterLike,
        pid_re: ParameterID,
        im: ParameterLike,
        pid_im: ParameterID,
    }

    impl ComplexScalar {
        #[allow(clippy::new_ret_no_self)]
        pub fn new(name: &str, re: ParameterLike, im: ParameterLike) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                re,
                pid_re: Default::default(),
                im,
                pid_im: Default::default(),
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for ComplexScalar {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.pid_re = resources.register_parameter(&self.re)?;
            self.pid_im = resources.register_parameter(&self.im)?;
            resources.register_amplitude(&self.name)
        }

        fn compute(
            &self,
            parameters: &Parameters,
            _event: &EventData,
            _cache: &Cache,
        ) -> Complex64 {
            Complex64::new(parameters.get(self.pid_re), parameters.get(self.pid_im))
        }

        fn compute_gradient(
            &self,
            _parameters: &Parameters,
            _event: &EventData,
            _cache: &Cache,
            gradient: &mut DVector<Complex64>,
        ) {
            if let ParameterID::Parameter(ind) = self.pid_re {
                gradient[ind] = Complex64::ONE;
            }
            if let ParameterID::Parameter(ind) = self.pid_im {
                gradient[ind] = Complex64::I;
            }
        }
    }

    #[test]
    fn test_batch_evaluation() {
        let expr = TestAmplitude::new("test", parameter("real"), parameter("imag")).unwrap();
        let mut event1 = test_event();
        event1.p4s[0].t = 10.0;
        let mut event2 = test_event();
        event2.p4s[0].t = 11.0;
        let mut event3 = test_event();
        event3.p4s[0].t = 12.0;
        let dataset = Arc::new(Dataset::new_with_metadata(
            vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
            Arc::new(DatasetMetadata::default()),
        ));
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate_batch(&[1.1, 2.2], &[0, 2]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Complex64::new(1.1, 2.2) * 10.0);
        assert_eq!(result[1], Complex64::new(1.1, 2.2) * 12.0);
        let result_grad = evaluator.evaluate_gradient_batch(&[1.1, 2.2], &[0, 2]);
        assert_eq!(result_grad.len(), 2);
        assert_eq!(result_grad[0][0], Complex64::new(10.0, 0.0));
        assert_eq!(result_grad[0][1], Complex64::new(0.0, 10.0));
        assert_eq!(result_grad[1][0], Complex64::new(12.0, 0.0));
        assert_eq!(result_grad[1][1], Complex64::new(0.0, 12.0));
    }

    #[test]
    fn test_constant_amplitude() {
        let expr = ComplexScalar::new(
            "constant",
            constant("const_re", 2.0),
            constant("const_im", 3.0),
        )
        .unwrap();
        let dataset = Arc::new(Dataset::new_with_metadata(
            vec![Arc::new(test_event())],
            Arc::new(DatasetMetadata::default()),
        ));
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex64::new(2.0, 3.0));
    }

    #[test]
    fn test_parametric_amplitude() {
        let expr = ComplexScalar::new(
            "parametric",
            parameter("test_param_re"),
            parameter("test_param_im"),
        )
        .unwrap();
        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[2.0, 3.0]);
        assert_eq!(result[0], Complex64::new(2.0, 3.0));
    }

    #[test]
    fn test_expression_operations() {
        let expr1 = ComplexScalar::new(
            "const1",
            constant("const1_re", 2.0),
            constant("const1_im", 0.0),
        )
        .unwrap();
        let expr2 = ComplexScalar::new(
            "const2",
            constant("const2_re", 0.0),
            constant("const2_im", 1.0),
        )
        .unwrap();
        let expr3 = ComplexScalar::new(
            "const3",
            constant("const3_re", 3.0),
            constant("const3_im", 4.0),
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());

        // Test (amp) addition
        let expr_add = &expr1 + &expr2;
        let result_add = expr_add.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_add[0], Complex64::new(2.0, 1.0));

        // Test (amp) subtraction
        let expr_sub = &expr1 - &expr2;
        let result_sub = expr_sub.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_sub[0], Complex64::new(2.0, -1.0));

        // Test (amp) multiplication
        let expr_mul = &expr1 * &expr2;
        let result_mul = expr_mul.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_mul[0], Complex64::new(0.0, 2.0));

        // Test (amp) division
        let expr_div = &expr1 / &expr3;
        let result_div = expr_div.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_div[0], Complex64::new(6.0 / 25.0, -8.0 / 25.0));

        // Test (amp) neg
        let expr_neg = -&expr3;
        let result_neg = expr_neg.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_neg[0], Complex64::new(-3.0, -4.0));

        // Test (expr) addition
        let expr_add2 = &expr_add + &expr_mul;
        let result_add2 = expr_add2.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_add2[0], Complex64::new(2.0, 3.0));

        // Test (expr) subtraction
        let expr_sub2 = &expr_add - &expr_mul;
        let result_sub2 = expr_sub2.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_sub2[0], Complex64::new(2.0, -1.0));

        // Test (expr) multiplication
        let expr_mul2 = &expr_add * &expr_mul;
        let result_mul2 = expr_mul2.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_mul2[0], Complex64::new(-2.0, 4.0));

        // Test (expr) division
        let expr_div2 = &expr_add / &expr_add2;
        let result_div2 = expr_div2.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_div2[0], Complex64::new(7.0 / 13.0, -4.0 / 13.0));

        // Test (expr) neg
        let expr_neg2 = -&expr_mul2;
        let result_neg2 = expr_neg2.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_neg2[0], Complex64::new(2.0, -4.0));

        // Test (amp) real
        let expr_real = expr3.real();
        let result_real = expr_real.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_real[0], Complex64::new(3.0, 0.0));

        // Test (expr) real
        let expr_mul2_real = expr_mul2.real();
        let result_mul2_real = expr_mul2_real.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_mul2_real[0], Complex64::new(-2.0, 0.0));

        // Test (amp) imag
        let expr_imag = expr3.imag();
        let result_imag = expr_imag.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_imag[0], Complex64::new(4.0, 0.0));

        // Test (expr) imag
        let expr_mul2_imag = expr_mul2.imag();
        let result_mul2_imag = expr_mul2_imag.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_mul2_imag[0], Complex64::new(4.0, 0.0));

        // Test (amp) conj
        let expr_conj = expr3.conj();
        let result_conj = expr_conj.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_conj[0], Complex64::new(3.0, -4.0));

        // Test (expr) conj
        let expr_mul2_conj = expr_mul2.conj();
        let result_mul2_conj = expr_mul2_conj.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_mul2_conj[0], Complex64::new(-2.0, -4.0));

        // Test (amp) norm_sqr
        let expr_norm = expr1.norm_sqr();
        let result_norm = expr_norm.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_norm[0], Complex64::new(4.0, 0.0));

        // Test (expr) norm_sqr
        let expr_mul2_norm = expr_mul2.norm_sqr();
        let result_mul2_norm = expr_mul2_norm.load(&dataset).unwrap().evaluate(&[]);
        assert_eq!(result_mul2_norm[0], Complex64::new(20.0, 0.0));
    }

    #[test]
    fn test_amplitude_activation() {
        let expr1 = ComplexScalar::new(
            "const1",
            constant("const1_re_act", 1.0),
            constant("const1_im_act", 0.0),
        )
        .unwrap();
        let expr2 = ComplexScalar::new(
            "const2",
            constant("const2_re_act", 2.0),
            constant("const2_im_act", 0.0),
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = &expr1 + &expr2;
        let evaluator = expr.load(&dataset).unwrap();

        // Test initial state (all active)
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex64::new(3.0, 0.0));

        // Test deactivation
        evaluator.deactivate_strict("const1").unwrap();
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex64::new(2.0, 0.0));

        // Test isolation
        evaluator.isolate_strict("const1").unwrap();
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex64::new(1.0, 0.0));

        // Test reactivation
        evaluator.activate_all();
        let result = evaluator.evaluate(&[]);
        assert_eq!(result[0], Complex64::new(3.0, 0.0));
    }

    #[test]
    fn test_gradient() {
        let expr1 = ComplexScalar::new(
            "parametric_1",
            parameter("test_param_re_1"),
            parameter("test_param_im_1"),
        )
        .unwrap();
        let expr2 = ComplexScalar::new(
            "parametric_2",
            parameter("test_param_re_2"),
            parameter("test_param_im_2"),
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let params = vec![2.0, 3.0, 4.0, 5.0];

        let expr = &expr1 + &expr2;
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 1.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 1.0);
        assert_relative_eq!(gradient[0][2].re, 1.0);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 0.0);
        assert_relative_eq!(gradient[0][3].im, 1.0);

        let expr = &expr1 - &expr2;
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 1.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 1.0);
        assert_relative_eq!(gradient[0][2].re, -1.0);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 0.0);
        assert_relative_eq!(gradient[0][3].im, -1.0);

        let expr = &expr1 * &expr2;
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 4.0);
        assert_relative_eq!(gradient[0][0].im, 5.0);
        assert_relative_eq!(gradient[0][1].re, -5.0);
        assert_relative_eq!(gradient[0][1].im, 4.0);
        assert_relative_eq!(gradient[0][2].re, 2.0);
        assert_relative_eq!(gradient[0][2].im, 3.0);
        assert_relative_eq!(gradient[0][3].re, -3.0);
        assert_relative_eq!(gradient[0][3].im, 2.0);

        let expr = &expr1 / &expr2;
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 4.0 / 41.0);
        assert_relative_eq!(gradient[0][0].im, -5.0 / 41.0);
        assert_relative_eq!(gradient[0][1].re, 5.0 / 41.0);
        assert_relative_eq!(gradient[0][1].im, 4.0 / 41.0);
        assert_relative_eq!(gradient[0][2].re, -102.0 / 1681.0);
        assert_relative_eq!(gradient[0][2].im, 107.0 / 1681.0);
        assert_relative_eq!(gradient[0][3].re, -107.0 / 1681.0);
        assert_relative_eq!(gradient[0][3].im, -102.0 / 1681.0);

        let expr = -(&expr1 * &expr2);
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, -4.0);
        assert_relative_eq!(gradient[0][0].im, -5.0);
        assert_relative_eq!(gradient[0][1].re, 5.0);
        assert_relative_eq!(gradient[0][1].im, -4.0);
        assert_relative_eq!(gradient[0][2].re, -2.0);
        assert_relative_eq!(gradient[0][2].im, -3.0);
        assert_relative_eq!(gradient[0][3].re, 3.0);
        assert_relative_eq!(gradient[0][3].im, -2.0);

        let expr = (&expr1 * &expr2).real();
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 4.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, -5.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 2.0);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, -3.0);
        assert_relative_eq!(gradient[0][3].im, 0.0);

        let expr = (&expr1 * &expr2).imag();
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 5.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 4.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 3.0);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 2.0);
        assert_relative_eq!(gradient[0][3].im, 0.0);

        let expr = (&expr1 * &expr2).conj();
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 4.0);
        assert_relative_eq!(gradient[0][0].im, -5.0);
        assert_relative_eq!(gradient[0][1].re, -5.0);
        assert_relative_eq!(gradient[0][1].im, -4.0);
        assert_relative_eq!(gradient[0][2].re, 2.0);
        assert_relative_eq!(gradient[0][2].im, -3.0);
        assert_relative_eq!(gradient[0][3].re, -3.0);
        assert_relative_eq!(gradient[0][3].im, -2.0);

        let expr = (&expr1 * &expr2).norm_sqr();
        let evaluator = expr.load(&dataset).unwrap();

        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 164.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 246.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 104.0);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 130.0);
        assert_relative_eq!(gradient[0][3].im, 0.0);
    }

    #[test]
    fn test_zeros_and_ones() {
        let amp = ComplexScalar::new(
            "parametric",
            parameter("test_param_re"),
            constant("fixed_two", 2.0),
        )
        .unwrap();
        let dataset = Arc::new(test_dataset());
        let expr = (amp * Expression::one() + Expression::zero()).norm_sqr();
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![2.0];
        let value = evaluator.evaluate(&params);
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x) * 1 + 0|^2 where f(x) = x+2i, the value should be x^2 + 4
        assert_relative_eq!(value[0].re, 8.0);
        assert_relative_eq!(value[0].im, 0.0);

        // For |f(x) * 1 + 0|^2 where f(x) = x+2i, the derivative should be 2x
        assert_relative_eq!(gradient[0][0].re, 4.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
    }

    #[test]
    fn test_parameter_registration() {
        let expr = ComplexScalar::new(
            "parametric",
            parameter("test_param_re"),
            constant("fixed_two", 2.0),
        )
        .unwrap();
        let parameters = expr.free_parameters();
        assert_eq!(parameters.len(), 1);
        assert_eq!(parameters[0], "test_param_re");
    }

    #[test]
    #[should_panic(expected = "refers to different underlying amplitudes")]
    fn test_duplicate_amplitude_registration() {
        let amp1 = ComplexScalar::new(
            "same_name",
            constant("dup_re1", 1.0),
            constant("dup_im1", 0.0),
        )
        .unwrap();
        let amp2 = ComplexScalar::new(
            "same_name",
            constant("dup_re2", 2.0),
            constant("dup_im2", 0.0),
        )
        .unwrap();
        let _expr = amp1 + amp2;
    }

    #[test]
    fn test_tree_printing() {
        let amp1 = ComplexScalar::new(
            "parametric_1",
            parameter("test_param_re_1"),
            parameter("test_param_im_1"),
        )
        .unwrap();
        let amp2 = ComplexScalar::new(
            "parametric_2",
            parameter("test_param_re_2"),
            parameter("test_param_im_2"),
        )
        .unwrap();
        let expr = &amp1.real() + &amp2.conj().imag() + Expression::one() * -Expression::zero()
            - Expression::zero() / Expression::one()
            + (&amp1 * &amp2).norm_sqr();
        assert_eq!(
            expr.to_string(),
            "+
├─ -
│  ├─ +
│  │  ├─ +
│  │  │  ├─ Re
│  │  │  │  └─ parametric_1(id=0)
│  │  │  └─ Im
│  │  │     └─ *
│  │  │        └─ parametric_2(id=1)
│  │  └─ ×
│  │     ├─ 1
│  │     └─ -
│  │        └─ 0
│  └─ ÷
│     ├─ 0
│     └─ 1
└─ NormSqr
   └─ ×
      ├─ parametric_1(id=0)
      └─ parametric_2(id=1)
"
        );
    }
}
