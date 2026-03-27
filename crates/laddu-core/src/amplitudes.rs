use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use auto_ops::*;
use dyn_clone::DynClone;
use nalgebra::DVector;
use num::complex::Complex64;

use parking_lot::RwLock;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

static AMPLITUDE_INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_amplitude_id() -> u64 {
    AMPLITUDE_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "expression-ir")]
#[allow(dead_code)]
mod ir;
#[cfg(feature = "expression-ir")]
#[allow(dead_code)]
mod lowered;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Dependence classification used by expression-IR diagnostics.
pub enum ExpressionDependence {
    /// Depends only on fixed/free parameter values.
    ParameterOnly,
    /// Depends only on event-local cached values.
    CacheOnly,
    /// Depends on both parameter values and cached event values.
    Mixed,
}

#[cfg(feature = "expression-ir")]
impl From<ir::DependenceClass> for ExpressionDependence {
    fn from(value: ir::DependenceClass) -> Self {
        match value {
            ir::DependenceClass::ParameterOnly => Self::ParameterOnly,
            ir::DependenceClass::CacheOnly => Self::CacheOnly,
            ir::DependenceClass::Mixed => Self::Mixed,
        }
    }
}

#[cfg(feature = "expression-ir")]
#[derive(Clone, Debug, PartialEq, Eq)]
/// Explain/debug view of the IR normalization planning decomposition.
pub struct NormalizationPlanExplain {
    /// Dependence classification at the expression root.
    pub root_dependence: ExpressionDependence,
    /// Warning-level diagnostics collected during planning.
    pub warnings: Vec<String>,
    /// Candidate multiply node indices identified as separable.
    pub separable_mul_candidate_nodes: Vec<usize>,
    /// Candidate separable node indices selected for caching.
    pub cached_separable_nodes: Vec<usize>,
    /// Node indices planned for residual per-event evaluation.
    pub residual_terms: Vec<usize>,
}

#[cfg(feature = "expression-ir")]
#[derive(Clone, Debug, PartialEq, Eq)]
/// Explain/debug view of amplitude execution sets used by normalization evaluation.
pub struct NormalizationExecutionSetsExplain {
    /// Amplitudes required to evaluate parameter factors for cached separable terms.
    pub cached_parameter_amplitudes: Vec<usize>,
    /// Amplitudes required to evaluate cache factors for cached separable terms.
    pub cached_cache_amplitudes: Vec<usize>,
    /// Amplitudes required for residual (non-cached) normalization evaluation.
    pub residual_amplitudes: Vec<usize>,
}

#[cfg(feature = "expression-ir")]
#[derive(Clone, Debug, PartialEq)]
/// Load-time precomputed integral metadata for a separable cached term.
pub struct PrecomputedCachedIntegral {
    /// Node index of the separable multiplication term.
    pub mul_node_index: usize,
    /// Node index of the parameter-dependent factor.
    pub parameter_node_index: usize,
    /// Node index of the cache-dependent factor.
    pub cache_node_index: usize,
    /// Signed extraction coefficient induced by Add/Sub/Neg ancestry to the root.
    pub coefficient: i32,
    /// Weighted sum over local events of the cache-dependent factor.
    pub weighted_cache_sum: Complex64,
}

#[cfg(feature = "expression-ir")]
#[derive(Clone, Debug, PartialEq)]
/// Parameter-gradient contribution for a load-time precomputed cached integral term.
pub struct PrecomputedCachedIntegralGradientTerm {
    /// Node index of the separable multiplication term.
    pub mul_node_index: usize,
    /// Node index of the parameter-dependent factor.
    pub parameter_node_index: usize,
    /// Node index of the cache-dependent factor.
    pub cache_node_index: usize,
    /// Signed extraction coefficient induced by Add/Sub/Neg ancestry to the root.
    pub coefficient: i32,
    /// Gradient contribution `(d/dp parameter_factor) * weighted_cache_sum`.
    pub weighted_gradient: DVector<Complex64>,
}

#[cfg(feature = "expression-ir")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CachedIntegralCacheKey {
    active_mask: Vec<bool>,
    n_events_local: usize,
    events_local_len: usize,
    weighted_sum_bits: u64,
    events_ptr: usize,
}

#[cfg(feature = "expression-ir")]
#[derive(Clone, Debug)]
struct CachedIntegralCacheState {
    key: CachedIntegralCacheKey,
    expression_ir: ir::ExpressionIR,
    values: Vec<PrecomputedCachedIntegral>,
    execution_sets: ir::NormalizationExecutionSets,
}

#[cfg(feature = "expression-ir")]
#[derive(Clone)]
struct ExpressionSpecializationState {
    cached_integrals: CachedIntegralCacheState,
    lowered_runtime: Option<lowered::LoweredExpressionRuntime>,
}

#[cfg(feature = "expression-ir")]
impl From<ir::NormalizationPlanExplain> for NormalizationPlanExplain {
    fn from(value: ir::NormalizationPlanExplain) -> Self {
        Self {
            root_dependence: value.root_dependence.into(),
            warnings: value.warnings,
            separable_mul_candidate_nodes: value
                .separable_mul_candidates
                .into_iter()
                .map(|candidate| candidate.node_index)
                .collect(),
            cached_separable_nodes: value.cached_separable_nodes,
            residual_terms: value.residual_terms,
        }
    }
}

#[cfg(feature = "expression-ir")]
impl From<ir::NormalizationExecutionSets> for NormalizationExecutionSetsExplain {
    fn from(value: ir::NormalizationExecutionSets) -> Self {
        Self {
            cached_parameter_amplitudes: value.cached_parameter_amplitudes,
            cached_cache_amplitudes: value.cached_cache_amplitudes,
            residual_amplitudes: value.residual_amplitudes,
        }
    }
}

#[cfg(feature = "expression-ir")]
impl From<ExpressionDependence> for ir::DependenceClass {
    fn from(value: ExpressionDependence) -> Self {
        match value {
            ExpressionDependence::ParameterOnly => Self::ParameterOnly,
            ExpressionDependence::CacheOnly => Self::CacheOnly,
            ExpressionDependence::Mixed => Self::Mixed,
        }
    }
}

use crate::{
    data::{Dataset, DatasetMetadata, NamedEventView},
    parameter_manager::{ParameterManager, ParameterTransform},
    resources::{Cache, Parameters, Resources},
    LadduError, LadduResult, ParameterID, ReadWrite,
};
#[cfg(feature = "execution-context-prototype")]
use crate::{ExecutionContext, ThreadPolicy};

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
    /// Optional dependence hint used by expression-IR diagnostics/planning.
    ///
    /// The default returns [`ExpressionDependence::Mixed`] for backward compatibility.
    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::Mixed
    }
    /// This method can be used to do some critical calculations ahead of time and
    /// store them in a [`Cache`]. These values can only depend on event data,
    /// not on any free parameters in the fit. This method is opt-in since it is
    /// not required to make a functioning [`Amplitude`].
    #[allow(unused_variables)]
    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {}

    /// Evaluate [`Amplitude::precompute`] over columnar event views in a [`Dataset`].
    #[cfg(feature = "rayon")]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        resources
            .caches
            .par_iter_mut()
            .enumerate()
            .for_each(|(event_index, cache)| {
                let event = dataset.event_view(event_index);
                self.precompute(&event, cache);
            });
    }

    /// Evaluate [`Amplitude::precompute`] over columnar event views in a [`Dataset`].
    #[cfg(not(feature = "rayon"))]
    fn precompute_all(&self, dataset: &Dataset, resources: &mut Resources) {
        dataset.for_each_named_event_local(|event_index, event| {
            let cache = &mut resources.caches[event_index];
            self.precompute(&event, cache);
        });
    }
    /// This method constitutes the main machinery of an [`Amplitude`], returning the actual
    /// calculated value for a particular set of [`Parameters`] and event [`Cache`].
    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64;

    /// This method yields the gradient of a particular [`Amplitude`] at a point specified
    /// by a set of [`Parameters`]. See those structs, as well as
    /// [`Cache`], for documentation on their available methods. For the most part,
    /// [`Parameters`] and the [`Cache`] are key-value storage accessed by [`ParameterID`]s and
    /// several different types of cache
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
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        self.central_difference_with_indices(
            &Vec::from_iter(0..parameters.len()),
            parameters,
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
            let f_plus = self.compute(&Parameters::new(&x_plus, &constants), cache);
            let f_minus = self.compute(&Parameters::new(&x_minus, &constants), cache);
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

#[derive(Clone, Debug)]
struct ExpressionProgram {
    ops: Vec<ExpressionOp>,
    slot_count: usize,
    root_slot: usize,
}

#[derive(Clone, Debug)]
enum ExpressionOp {
    LoadZero {
        dst: usize,
    },
    LoadOne {
        dst: usize,
    },
    LoadAmp {
        dst: usize,
        amp_idx: usize,
    },
    Add {
        dst: usize,
        left: usize,
        right: usize,
    },
    Sub {
        dst: usize,
        left: usize,
        right: usize,
    },
    Mul {
        dst: usize,
        left: usize,
        right: usize,
    },
    Div {
        dst: usize,
        left: usize,
        right: usize,
    },
    Neg {
        dst: usize,
        input: usize,
    },
    Real {
        dst: usize,
        input: usize,
    },
    Imag {
        dst: usize,
        input: usize,
    },
    Conj {
        dst: usize,
        input: usize,
    },
    NormSqr {
        dst: usize,
        input: usize,
    },
}

#[derive(Default)]
struct ExpressionProgramBuilder {
    ops: Vec<ExpressionOp>,
    next_slot: usize,
}

impl ExpressionProgramBuilder {
    fn alloc_slot(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        slot
    }

    fn build(self, root: usize) -> ExpressionProgram {
        ExpressionProgram {
            ops: self.ops,
            slot_count: self.next_slot,
            root_slot: root,
        }
    }

    fn emit(&mut self, op: ExpressionOp) {
        self.ops.push(op);
    }

    fn compile(&mut self, node: &ExpressionNode) -> usize {
        match node {
            ExpressionNode::Zero => {
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::LoadZero { dst });
                dst
            }
            ExpressionNode::One => {
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::LoadOne { dst });
                dst
            }
            ExpressionNode::Amp(idx) => {
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::LoadAmp { dst, amp_idx: *idx });
                dst
            }
            ExpressionNode::Add(a, b) => {
                let left = self.compile(a);
                let right = self.compile(b);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Add { dst, left, right });
                dst
            }
            ExpressionNode::Sub(a, b) => {
                let left = self.compile(a);
                let right = self.compile(b);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Sub { dst, left, right });
                dst
            }
            ExpressionNode::Mul(a, b) => {
                let left = self.compile(a);
                let right = self.compile(b);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Mul { dst, left, right });
                dst
            }
            ExpressionNode::Div(a, b) => {
                let left = self.compile(a);
                let right = self.compile(b);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Div { dst, left, right });
                dst
            }
            ExpressionNode::Neg(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Neg { dst, input });
                dst
            }
            ExpressionNode::Real(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Real { dst, input });
                dst
            }
            ExpressionNode::Imag(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Imag { dst, input });
                dst
            }
            ExpressionNode::Conj(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Conj { dst, input });
                dst
            }
            ExpressionNode::NormSqr(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::NormSqr { dst, input });
                dst
            }
        }
    }
}

impl ExpressionProgram {
    fn from_node(node: &ExpressionNode) -> Self {
        let mut builder = ExpressionProgramBuilder::default();
        let root = builder.compile(node);
        builder.build(root)
    }

    #[allow(dead_code)]
    fn slot_count(&self) -> usize {
        self.slot_count
    }

    fn fill_values(&self, amplitude_values: &[Complex64], slots: &mut [Complex64]) {
        debug_assert!(slots.len() >= self.slot_count);
        for op in &self.ops {
            match *op {
                ExpressionOp::LoadZero { dst } => slots[dst] = Complex64::ZERO,
                ExpressionOp::LoadOne { dst } => slots[dst] = Complex64::ONE,
                ExpressionOp::LoadAmp { dst, amp_idx } => {
                    slots[dst] = amplitude_values.get(amp_idx).copied().unwrap_or_default();
                }
                ExpressionOp::Add { dst, left, right } => {
                    slots[dst] = slots[left] + slots[right];
                }
                ExpressionOp::Sub { dst, left, right } => {
                    slots[dst] = slots[left] - slots[right];
                }
                ExpressionOp::Mul { dst, left, right } => {
                    slots[dst] = slots[left] * slots[right];
                }
                ExpressionOp::Div { dst, left, right } => {
                    slots[dst] = slots[left] / slots[right];
                }
                ExpressionOp::Neg { dst, input } => {
                    slots[dst] = -slots[input];
                }
                ExpressionOp::Real { dst, input } => {
                    slots[dst] = Complex64::new(slots[input].re, 0.0);
                }
                ExpressionOp::Imag { dst, input } => {
                    slots[dst] = Complex64::new(slots[input].im, 0.0);
                }
                ExpressionOp::Conj { dst, input } => {
                    slots[dst] = slots[input].conj();
                }
                ExpressionOp::NormSqr { dst, input } => {
                    slots[dst] = Complex64::new(slots[input].norm_sqr(), 0.0);
                }
            }
        }
    }

    fn evaluate_into(&self, amplitude_values: &[Complex64], slots: &mut [Complex64]) -> Complex64 {
        if self.slot_count == 0 {
            return Complex64::ZERO;
        }
        self.fill_values(amplitude_values, slots);
        slots[self.root_slot]
    }

    pub fn evaluate(&self, amplitude_values: &[Complex64]) -> Complex64 {
        if self.slot_count == 0 {
            return Complex64::ZERO;
        }
        let mut slots = vec![Complex64::ZERO; self.slot_count];
        self.evaluate_into(amplitude_values, &mut slots)
    }

    pub fn evaluate_gradient_into(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_slots: &mut [Complex64],
        gradient_slots: &mut [DVector<Complex64>],
    ) -> DVector<Complex64> {
        if self.slot_count == 0 {
            let dim = gradient_values.first().map(|g| g.len()).unwrap_or(0);
            return DVector::zeros(dim);
        }
        self.fill_values(amplitude_values, value_slots);
        self.fill_gradients(gradient_values, value_slots, gradient_slots);
        gradient_slots[self.root_slot].clone()
    }

    pub fn evaluate_value_gradient_into(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_slots: &mut [Complex64],
        gradient_slots: &mut [DVector<Complex64>],
    ) -> (Complex64, DVector<Complex64>) {
        if self.slot_count == 0 {
            let dim = gradient_values.first().map(|g| g.len()).unwrap_or(0);
            return (Complex64::ZERO, DVector::zeros(dim));
        }
        self.fill_values(amplitude_values, value_slots);
        self.fill_gradients(gradient_values, value_slots, gradient_slots);
        (
            value_slots[self.root_slot],
            gradient_slots[self.root_slot].clone(),
        )
    }

    pub fn evaluate_gradient(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        let grad_dim = gradient_values.first().map(|g| g.len()).unwrap_or(0);
        let mut value_slots = vec![Complex64::ZERO; self.slot_count];
        let mut gradient_slots: Vec<DVector<Complex64>> = (0..self.slot_count)
            .map(|_| DVector::zeros(grad_dim))
            .collect();
        self.evaluate_gradient_into(
            amplitude_values,
            gradient_values,
            &mut value_slots,
            &mut gradient_slots,
        )
    }

    fn fill_gradients(
        &self,
        amplitude_gradients: &[DVector<Complex64>],
        values: &[Complex64],
        gradients: &mut [DVector<Complex64>],
    ) {
        debug_assert!(gradients.len() >= self.slot_count);
        debug_assert!(values.len() >= self.slot_count);
        fn borrow_dst(
            gradients: &mut [DVector<Complex64>],
            dst: usize,
        ) -> (&[DVector<Complex64>], &mut DVector<Complex64>) {
            let (before, tail) = gradients.split_at_mut(dst);
            let (dst_ref, _) = tail.split_first_mut().expect("dst slot should exist");
            (before, dst_ref)
        }
        for op in &self.ops {
            match *op {
                ExpressionOp::LoadZero { dst } | ExpressionOp::LoadOne { dst } => {
                    let (_, dst_grad) = borrow_dst(gradients, dst);
                    for item in dst_grad.iter_mut() {
                        *item = Complex64::ZERO;
                    }
                }
                ExpressionOp::LoadAmp { dst, amp_idx } => {
                    let (_, dst_grad) = borrow_dst(gradients, dst);
                    if let Some(source) = amplitude_gradients.get(amp_idx) {
                        dst_grad.clone_from(source);
                    } else {
                        for item in dst_grad.iter_mut() {
                            *item = Complex64::ZERO;
                        }
                    }
                }
                ExpressionOp::Add { dst, left, right } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    dst_grad.clone_from(&before_dst[left]);
                    for (dst_item, right_item) in dst_grad.iter_mut().zip(before_dst[right].iter())
                    {
                        *dst_item += *right_item;
                    }
                }
                ExpressionOp::Sub { dst, left, right } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    dst_grad.clone_from(&before_dst[left]);
                    for (dst_item, right_item) in dst_grad.iter_mut().zip(before_dst[right].iter())
                    {
                        *dst_item -= *right_item;
                    }
                }
                ExpressionOp::Mul { dst, left, right } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let f_left = values[left];
                    let f_right = values[right];
                    dst_grad.clone_from(&before_dst[right]);
                    for item in dst_grad.iter_mut() {
                        *item *= f_left;
                    }
                    for (dst_item, left_item) in dst_grad.iter_mut().zip(before_dst[left].iter()) {
                        *dst_item += *left_item * f_right;
                    }
                }
                ExpressionOp::Div { dst, left, right } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let f_left = values[left];
                    let f_right = values[right];
                    let denom = f_right * f_right;
                    dst_grad.clone_from(&before_dst[left]);
                    for item in dst_grad.iter_mut() {
                        *item *= f_right;
                    }
                    for (dst_item, right_item) in dst_grad.iter_mut().zip(before_dst[right].iter())
                    {
                        *dst_item -= *right_item * f_left;
                    }
                    for item in dst_grad.iter_mut() {
                        *item /= denom;
                    }
                }
                ExpressionOp::Neg { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    dst_grad.clone_from(&before_dst[input]);
                    for item in dst_grad.iter_mut() {
                        *item = -*item;
                    }
                }
                ExpressionOp::Real { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = Complex64::new(input_item.re, 0.0);
                    }
                }
                ExpressionOp::Imag { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = Complex64::new(input_item.im, 0.0);
                    }
                }
                ExpressionOp::Conj { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = input_item.conj();
                    }
                }
                ExpressionOp::NormSqr { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let conj_value = values[input].conj();
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = Complex64::new(2.0 * (*input_item * conj_value).re, 0.0);
                    }
                }
            }
        }
    }
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

    fn program(&self) -> ExpressionProgram {
        ExpressionProgram::from_node(self)
    }

    /// Evaluate an [`ExpressionNode`] by compiling it to bytecode on the fly.
    ///
    /// For repeated evaluations prefer [`ExpressionProgram`] to avoid recompilation.
    pub fn evaluate(&self, amplitude_values: &[Complex64]) -> Complex64 {
        self.program().evaluate(amplitude_values)
    }

    /// Evaluate the gradient of an [`ExpressionNode`].
    ///
    /// For repeated evaluations prefer [`ExpressionProgram`] to avoid recompilation.
    pub fn evaluate_gradient(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        self.program()
            .evaluate_gradient(amplitude_values, gradient_values)
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
    pub fn n_parameters(&self) -> usize {
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
        resources.reserve_cache(dataset.n_events_local());
        resources.refresh_active_indices();
        let parameter_manager = ParameterManager::with_fixed_values(
            &resources.parameter_names(),
            &resources.fixed_parameter_values(),
        );
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
        #[cfg(feature = "expression-ir")]
        let expression_ir = {
            let mut active_amplitudes = vec![false; amplitudes.len()];
            for &index in resources.active_indices() {
                active_amplitudes[index] = true;
            }
            let amplitude_dependencies = amplitudes
                .iter()
                .map(|amp| ir::DependenceClass::from(amp.dependence_hint()))
                .collect::<Vec<_>>();
            ir::compile_expression_ir(&self.tree, &active_amplitudes, &amplitude_dependencies)
        };
        #[cfg(feature = "expression-ir")]
        let cached_integrals = Evaluator::precompute_cached_integrals_at_load(
            &expression_ir,
            &amplitudes,
            &resources,
            dataset,
            parameter_manager.n_free_parameters(),
        );
        #[cfg(feature = "expression-ir")]
        let lowered_runtime =
            lowered::LoweredExpressionRuntime::from_ir_value_gradient(&expression_ir).ok();
        #[cfg(feature = "expression-ir")]
        let execution_sets = expression_ir.normalization_execution_sets().clone();
        #[cfg(feature = "expression-ir")]
        let cached_integral_key =
            Evaluator::cached_integral_cache_key(resources.active.clone(), dataset);
        #[cfg(feature = "expression-ir")]
        let cached_integral_state = CachedIntegralCacheState {
            key: cached_integral_key.clone(),
            expression_ir,
            values: cached_integrals,
            execution_sets,
        };
        #[cfg(feature = "expression-ir")]
        let specialization_state = ExpressionSpecializationState {
            cached_integrals: cached_integral_state.clone(),
            lowered_runtime: lowered_runtime.clone(),
        };
        #[cfg(feature = "expression-ir")]
        let specialization_cache = HashMap::from([(cached_integral_key, specialization_state)]);
        Ok(Evaluator {
            amplitudes,
            resources: Arc::new(RwLock::new(resources)),
            dataset: dataset.clone(),
            expression: self.tree.clone(),
            runtime_backend: {
                #[cfg(feature = "expression-ir")]
                {
                    ExpressionRuntimeBackend::Lowered
                }
                #[cfg(not(feature = "expression-ir"))]
                {
                    ExpressionRuntimeBackend::LegacyProgram
                }
            },
            expression_program: ExpressionProgram::from_node(&self.tree),
            #[cfg(feature = "expression-ir")]
            ir_planning: ExpressionIrPlanningState {
                expression_ir: cached_integral_state.expression_ir.clone(),
                cached_integrals: Arc::new(RwLock::new(Some(cached_integral_state))),
                specialization_cache: Arc::new(RwLock::new(specialization_cache)),
            },
            #[cfg(feature = "expression-ir")]
            runtime_state: ExpressionRuntimeState {
                lowered_runtime: Arc::new(RwLock::new(lowered_runtime)),
            },
            registry: self.registry.clone(),
            parameter_manager,
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

    /// Credit to Daniel Janus: <https://blog.danieljanus.pl/2023/07/20/iterating-trees/>
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpressionRuntimeBackend {
    LegacyProgram,
    #[cfg(feature = "expression-ir")]
    IrInterpreter,
    #[cfg(feature = "expression-ir")]
    Lowered,
}

#[cfg(feature = "expression-ir")]
#[derive(Clone)]
/// IR-planning state derived from the semantic expression tree plus the current active mask.
///
/// Invariants:
/// - `expression_ir` is never a source of truth; it is always derived from `Evaluator::expression`.
/// - `cached_integrals` are specialization-dependent and must be treated as invalid once the
///   active mask or dataset identity changes.
struct ExpressionIrPlanningState {
    expression_ir: ir::ExpressionIR,
    cached_integrals: Arc<RwLock<Option<CachedIntegralCacheState>>>,
    specialization_cache:
        Arc<RwLock<HashMap<CachedIntegralCacheKey, ExpressionSpecializationState>>>,
}

#[cfg(feature = "expression-ir")]
#[allow(dead_code)]
#[derive(Clone)]
/// Runtime-only lowered execution artifacts derived from IR planning state.
///
/// Invariants:
/// - Lowered runtimes must never outlive the specialization assumptions used to build them.
/// - Activation-mask changes invalidate any stored lowered runtime until it is rebuilt.
/// - Lowered runtime is an execution cache, not a semantic source of truth.
struct ExpressionRuntimeState {
    lowered_runtime: Arc<RwLock<Option<lowered::LoweredExpressionRuntime>>>,
}

/// Evaluator for [`Expression`] that mirrors the existing evaluator behavior.
#[allow(missing_docs)]
#[derive(Clone)]
pub struct Evaluator {
    pub amplitudes: Vec<Box<dyn Amplitude>>,
    pub resources: Arc<RwLock<Resources>>,
    pub dataset: Arc<Dataset>,
    pub expression: ExpressionNode,
    runtime_backend: ExpressionRuntimeBackend,
    expression_program: ExpressionProgram,
    #[cfg(feature = "expression-ir")]
    ir_planning: ExpressionIrPlanningState,
    #[cfg(feature = "expression-ir")]
    runtime_state: ExpressionRuntimeState,
    registry: ExpressionRegistry,
    parameter_manager: ParameterManager,
}

#[allow(missing_docs)]
impl Evaluator {
    #[cfg(feature = "expression-ir")]
    /// Internal benchmarking/debug hook for forcing a specific expression execution backend.
    pub fn set_expression_runtime_backend(&mut self, backend: ExpressionRuntimeBackend) {
        self.runtime_backend = backend;
    }

    #[cfg(feature = "expression-ir")]
    fn expression_ir(&self) -> ir::ExpressionIR {
        self.ir_planning
            .cached_integrals
            .read()
            .as_ref()
            .map(|state| state.expression_ir.clone())
            .unwrap_or_else(|| self.ir_planning.expression_ir.clone())
    }

    #[cfg(feature = "expression-ir")]
    #[allow(dead_code)]
    fn lowered_runtime(&self) -> Option<lowered::LoweredExpressionRuntime> {
        self.runtime_state.lowered_runtime.read().clone()
    }

    #[cfg(feature = "expression-ir")]
    fn lowered_runtime_slot_count(&self) -> usize {
        self.lowered_runtime()
            .map(|runtime| {
                [
                    runtime
                        .value_program()
                        .map(|program| program.scratch_slots())
                        .unwrap_or(0),
                    runtime
                        .gradient_program()
                        .map(|program| program.scratch_slots())
                        .unwrap_or(0),
                    runtime
                        .value_gradient_program()
                        .map(|program| program.scratch_slots())
                        .unwrap_or(0),
                ]
                .into_iter()
                .max()
                .unwrap_or(0)
            })
            .unwrap_or_else(|| self.expression_ir().node_count())
    }

    #[cfg(feature = "expression-ir")]
    #[cfg(test)]
    fn specialization_cache_len(&self) -> usize {
        self.ir_planning.specialization_cache.read().len()
    }

    #[cfg(feature = "expression-ir")]
    fn install_expression_specialization(&self, specialization: &ExpressionSpecializationState) {
        *self.ir_planning.cached_integrals.write() = Some(specialization.cached_integrals.clone());
        *self.runtime_state.lowered_runtime.write() = specialization.lowered_runtime.clone();
    }

    #[cfg(feature = "expression-ir")]
    fn build_expression_specialization(
        &self,
        resources: &Resources,
        key: CachedIntegralCacheKey,
    ) -> ExpressionSpecializationState {
        let expression_ir = self.compile_expression_ir_for_active_mask(&resources.active);
        let values = Self::precompute_cached_integrals_at_load(
            &expression_ir,
            &self.amplitudes,
            resources,
            &self.dataset,
            self.parameter_manager.n_free_parameters(),
        );
        let execution_sets = expression_ir.normalization_execution_sets().clone();
        let lowered_runtime =
            lowered::LoweredExpressionRuntime::from_ir_value_gradient(&expression_ir).ok();
        ExpressionSpecializationState {
            cached_integrals: CachedIntegralCacheState {
                key,
                expression_ir,
                values,
                execution_sets,
            },
            lowered_runtime,
        }
    }

    #[cfg(feature = "expression-ir")]
    fn ensure_expression_specialization(
        &self,
        resources: &Resources,
    ) -> ExpressionSpecializationState {
        let key = Self::cached_integral_cache_key(resources.active.clone(), &self.dataset);
        if let Some(state) = self.ir_planning.cached_integrals.read().as_ref() {
            if state.key == key {
                return ExpressionSpecializationState {
                    cached_integrals: state.clone(),
                    lowered_runtime: self.runtime_state.lowered_runtime.read().clone(),
                };
            }
        }
        if let Some(specialization) = self
            .ir_planning
            .specialization_cache
            .read()
            .get(&key)
            .cloned()
        {
            self.install_expression_specialization(&specialization);
            return specialization;
        }
        let specialization = self.build_expression_specialization(resources, key.clone());
        self.ir_planning
            .specialization_cache
            .write()
            .insert(key, specialization.clone());
        self.install_expression_specialization(&specialization);
        specialization
    }

    #[cfg(feature = "expression-ir")]
    fn rebuild_runtime_specializations(&self, resources: &Resources) {
        let _ = self.ensure_expression_specialization(resources);
    }

    #[cfg(feature = "expression-ir")]
    fn refresh_runtime_specializations(&self) {
        let resources = self.resources.read();
        self.rebuild_runtime_specializations(&resources);
    }

    #[cfg(feature = "expression-ir")]
    fn cached_integral_cache_key(
        active_mask: Vec<bool>,
        dataset: &Dataset,
    ) -> CachedIntegralCacheKey {
        CachedIntegralCacheKey {
            active_mask,
            n_events_local: dataset.n_events_local(),
            events_local_len: dataset.events_local().len(),
            weighted_sum_bits: dataset.n_events_weighted_local().to_bits(),
            events_ptr: dataset.events_local().as_ptr() as usize,
        }
    }

    #[cfg(feature = "expression-ir")]
    fn precompute_cached_integrals_at_load(
        expression_ir: &ir::ExpressionIR,
        amplitudes: &[Box<dyn Amplitude>],
        resources: &Resources,
        dataset: &Dataset,
        n_free_parameters: usize,
    ) -> Vec<PrecomputedCachedIntegral> {
        let descriptors = expression_ir.cached_integral_descriptors();
        if descriptors.is_empty() {
            return Vec::new();
        }
        let execution_sets = expression_ir.normalization_execution_sets();
        let seed_parameters = vec![0.0; n_free_parameters];
        let parameters = Parameters::new(&seed_parameters, &resources.constants);
        let mut amplitude_values = vec![Complex64::ZERO; amplitudes.len()];
        let mut value_slots = vec![Complex64::ZERO; expression_ir.node_count()];
        let active_set = resources.active_indices();
        let cache_active_indices = execution_sets
            .cached_cache_amplitudes
            .iter()
            .copied()
            .filter(|index| active_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        let mut weighted_cache_sums = vec![Complex64::ZERO; descriptors.len()];
        for (cache, event) in resources.caches.iter().zip(dataset.events_local().iter()) {
            amplitude_values.fill(Complex64::ZERO);
            for &amp_idx in &cache_active_indices {
                amplitude_values[amp_idx] = amplitudes[amp_idx].compute(&parameters, cache);
            }
            expression_ir.evaluate_into(&amplitude_values, &mut value_slots);
            let weight = event.weight();
            for (descriptor_index, descriptor) in descriptors.iter().enumerate() {
                weighted_cache_sums[descriptor_index] +=
                    value_slots[descriptor.cache_node_index] * weight;
            }
        }
        descriptors
            .iter()
            .zip(weighted_cache_sums)
            .map(
                |(descriptor, weighted_cache_sum)| PrecomputedCachedIntegral {
                    mul_node_index: descriptor.mul_node_index,
                    parameter_node_index: descriptor.parameter_node_index,
                    cache_node_index: descriptor.cache_node_index,
                    coefficient: descriptor.coefficient,
                    weighted_cache_sum,
                },
            )
            .collect()
    }

    #[inline]
    fn fill_amplitude_values(
        &self,
        amplitude_values: &mut [Complex64],
        active_indices: &[usize],
        parameters: &Parameters,
        cache: &Cache,
    ) {
        amplitude_values.fill(Complex64::ZERO);
        for &amp_idx in active_indices {
            amplitude_values[amp_idx] = self.amplitudes[amp_idx].compute(parameters, cache);
        }
    }

    #[inline]
    fn fill_amplitude_gradients(
        &self,
        gradient_values: &mut [DVector<Complex64>],
        active_mask: &[bool],
        parameters: &Parameters,
        cache: &Cache,
    ) {
        for ((amp, active), grad) in self
            .amplitudes
            .iter()
            .zip(active_mask.iter())
            .zip(gradient_values.iter_mut())
        {
            grad.fill(Complex64::ZERO);
            if *active {
                amp.compute_gradient(parameters, cache, grad);
            }
        }
    }

    #[inline]
    fn fill_amplitude_values_and_gradients(
        &self,
        amplitude_values: &mut [Complex64],
        gradient_values: &mut [DVector<Complex64>],
        active_indices: &[usize],
        active_mask: &[bool],
        parameters: &Parameters,
        cache: &Cache,
    ) {
        self.fill_amplitude_values(amplitude_values, active_indices, parameters, cache);
        self.fill_amplitude_gradients(gradient_values, active_mask, parameters, cache);
    }

    #[inline]
    fn evaluate_cache_gradient_with_scratch(
        &self,
        amplitude_values: &mut [Complex64],
        gradient_values: &mut [DVector<Complex64>],
        value_slots: &mut [Complex64],
        gradient_slots: &mut [DVector<Complex64>],
        active_indices: &[usize],
        active_mask: &[bool],
        parameters: &Parameters,
        cache: &Cache,
    ) -> DVector<Complex64> {
        self.fill_amplitude_values_and_gradients(
            amplitude_values,
            gradient_values,
            active_indices,
            active_mask,
            parameters,
            cache,
        );
        self.evaluate_expression_gradient_with_scratch(
            amplitude_values,
            gradient_values,
            value_slots,
            gradient_slots,
        )
    }

    #[inline]
    fn evaluate_cache_value_gradient_with_scratch(
        &self,
        amplitude_values: &mut [Complex64],
        gradient_values: &mut [DVector<Complex64>],
        value_slots: &mut [Complex64],
        gradient_slots: &mut [DVector<Complex64>],
        active_indices: &[usize],
        active_mask: &[bool],
        parameters: &Parameters,
        cache: &Cache,
    ) -> (Complex64, DVector<Complex64>) {
        self.fill_amplitude_values_and_gradients(
            amplitude_values,
            gradient_values,
            active_indices,
            active_mask,
            parameters,
            cache,
        );
        self.evaluate_expression_value_gradient_with_scratch(
            amplitude_values,
            gradient_values,
            value_slots,
            gradient_slots,
        )
    }

    pub fn expression_slot_count(&self) -> usize {
        match self.runtime_backend {
            ExpressionRuntimeBackend::LegacyProgram => self.expression_program.slot_count(),
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::IrInterpreter => self.expression_ir().node_count(),
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::Lowered => self.lowered_runtime_slot_count(),
        }
    }

    #[cfg(feature = "expression-ir")]
    fn compile_expression_ir_for_active_mask(&self, active_mask: &[bool]) -> ir::ExpressionIR {
        let amplitude_dependencies = self
            .amplitudes
            .iter()
            .map(|amp| ir::DependenceClass::from(amp.dependence_hint()))
            .collect::<Vec<_>>();
        ir::compile_expression_ir(&self.expression, active_mask, &amplitude_dependencies)
    }

    #[cfg(feature = "expression-ir")]
    fn ensure_cached_integral_cache_state(
        &self,
        resources: &Resources,
    ) -> CachedIntegralCacheState {
        self.ensure_expression_specialization(resources)
            .cached_integrals
    }

    fn evaluate_expression_runtime_value_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        scratch: &mut [Complex64],
    ) -> Complex64 {
        match self.runtime_backend {
            ExpressionRuntimeBackend::LegacyProgram => self
                .expression_program
                .evaluate_into(amplitude_values, scratch),
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::IrInterpreter => self
                .expression_ir()
                .evaluate_into(amplitude_values, scratch),
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::Lowered => {
                let lowered_runtime = self.runtime_state.lowered_runtime.read();
                if let Some(program) = lowered_runtime
                    .as_ref()
                    .and_then(|runtime| runtime.value_program())
                {
                    program.evaluate_into(amplitude_values, scratch)
                } else {
                    self.expression_ir()
                        .evaluate_into(amplitude_values, scratch)
                }
            }
        }
    }

    fn evaluate_expression_runtime_gradient_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> DVector<Complex64> {
        match self.runtime_backend {
            ExpressionRuntimeBackend::LegacyProgram => {
                self.expression_program.evaluate_gradient_into(
                    amplitude_values,
                    gradient_values,
                    value_scratch,
                    gradient_scratch,
                )
            }
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::IrInterpreter => self.expression_ir().evaluate_gradient_into(
                amplitude_values,
                gradient_values,
                value_scratch,
                gradient_scratch,
            ),
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::Lowered => {
                let lowered_runtime = self.runtime_state.lowered_runtime.read();
                if let Some(program) = lowered_runtime
                    .as_ref()
                    .and_then(|runtime| runtime.gradient_program())
                {
                    program.evaluate_gradient_into(
                        amplitude_values,
                        gradient_values,
                        value_scratch,
                        gradient_scratch,
                    )
                } else {
                    self.expression_ir().evaluate_gradient_into(
                        amplitude_values,
                        gradient_values,
                        value_scratch,
                        gradient_scratch,
                    )
                }
            }
        }
    }

    fn evaluate_expression_runtime_value_gradient_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> (Complex64, DVector<Complex64>) {
        match self.runtime_backend {
            ExpressionRuntimeBackend::LegacyProgram => {
                self.expression_program.evaluate_value_gradient_into(
                    amplitude_values,
                    gradient_values,
                    value_scratch,
                    gradient_scratch,
                )
            }
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::IrInterpreter => {
                self.expression_ir().evaluate_value_gradient_into(
                    amplitude_values,
                    gradient_values,
                    value_scratch,
                    gradient_scratch,
                )
            }
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::Lowered => {
                let lowered_runtime = self.runtime_state.lowered_runtime.read();
                if let Some(program) = lowered_runtime
                    .as_ref()
                    .and_then(|runtime| runtime.value_gradient_program())
                {
                    program.evaluate_value_gradient_into(
                        amplitude_values,
                        gradient_values,
                        value_scratch,
                        gradient_scratch,
                    )
                } else {
                    self.expression_ir().evaluate_value_gradient_into(
                        amplitude_values,
                        gradient_values,
                        value_scratch,
                        gradient_scratch,
                    )
                }
            }
        }
    }

    fn evaluate_expression_runtime_value(&self, amplitude_values: &[Complex64]) -> Complex64 {
        match self.runtime_backend {
            ExpressionRuntimeBackend::LegacyProgram => {
                self.expression_program.evaluate(amplitude_values)
            }
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::IrInterpreter => {
                self.expression_ir().evaluate(amplitude_values)
            }
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::Lowered => {
                let lowered_runtime = self.runtime_state.lowered_runtime.read();
                if let Some(program) = lowered_runtime
                    .as_ref()
                    .and_then(|runtime| runtime.value_program())
                {
                    let mut scratch = vec![Complex64::ZERO; program.scratch_slots()];
                    program.evaluate_into(amplitude_values, &mut scratch)
                } else {
                    self.expression_ir().evaluate(amplitude_values)
                }
            }
        }
    }

    fn evaluate_expression_runtime_gradient(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        match self.runtime_backend {
            ExpressionRuntimeBackend::LegacyProgram => self
                .expression_program
                .evaluate_gradient(amplitude_values, gradient_values),
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::IrInterpreter => self
                .expression_ir()
                .evaluate_gradient(amplitude_values, gradient_values),
            #[cfg(feature = "expression-ir")]
            ExpressionRuntimeBackend::Lowered => {
                let lowered_runtime = self.runtime_state.lowered_runtime.read();
                if let Some(program) = lowered_runtime
                    .as_ref()
                    .and_then(|runtime| runtime.gradient_program())
                {
                    let mut value_scratch = vec![Complex64::ZERO; program.scratch_slots()];
                    let grad_dim = gradient_values.first().map(|g| g.len()).unwrap_or(0);
                    let mut gradient_scratch = (0..program.scratch_slots())
                        .map(|_| DVector::zeros(grad_dim))
                        .collect::<Vec<_>>();
                    program.evaluate_gradient_into(
                        amplitude_values,
                        gradient_values,
                        &mut value_scratch,
                        &mut gradient_scratch,
                    )
                } else {
                    self.expression_ir()
                        .evaluate_gradient(amplitude_values, gradient_values)
                }
            }
        }
    }

    #[cfg(feature = "expression-ir")]
    /// Dependence classification for the compiled expression root.
    pub fn expression_root_dependence(&self) -> ExpressionDependence {
        let resources = self.resources.read();
        self.ensure_cached_integral_cache_state(&resources)
            .expression_ir
            .root_dependence()
            .into()
    }

    #[cfg(feature = "expression-ir")]
    /// Dependence classification for each compiled expression node.
    pub fn expression_node_dependence_annotations(&self) -> Vec<ExpressionDependence> {
        let resources = self.resources.read();
        self.ensure_cached_integral_cache_state(&resources)
            .expression_ir
            .node_dependence_annotations()
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    #[cfg(feature = "expression-ir")]
    /// Warning-level diagnostics for potentially inconsistent dependence hints.
    pub fn expression_dependence_warnings(&self) -> Vec<String> {
        let resources = self.resources.read();
        self.ensure_cached_integral_cache_state(&resources)
            .expression_ir
            .dependence_warnings()
            .to_vec()
    }

    #[cfg(feature = "expression-ir")]
    /// Explain/debug view of IR normalization planning decomposition.
    pub fn expression_normalization_plan_explain(&self) -> NormalizationPlanExplain {
        let resources = self.resources.read();
        self.ensure_cached_integral_cache_state(&resources)
            .expression_ir
            .normalization_plan_explain()
            .into()
    }

    #[cfg(feature = "expression-ir")]
    /// Explain/debug view of amplitude execution sets used by normalization evaluation.
    pub fn expression_normalization_execution_sets(&self) -> NormalizationExecutionSetsExplain {
        let resources = self.resources.read();
        self.ensure_cached_integral_cache_state(&resources)
            .execution_sets
            .clone()
            .into()
    }

    #[cfg(feature = "expression-ir")]
    /// Cached integral terms precomputed at evaluator load.
    pub fn expression_precomputed_cached_integrals(&self) -> Vec<PrecomputedCachedIntegral> {
        let resources = self.resources.read();
        self.ensure_cached_integral_cache_state(&resources).values
    }

    #[cfg(feature = "expression-ir")]
    /// Derivative rules for cached separable terms evaluated at the given parameter point.
    ///
    /// Each returned term corresponds to a cached separable descriptor and contributes
    /// `weighted_gradient` to `d(normalization)/dp` prior to residual-term combination.
    pub fn expression_precomputed_cached_integral_gradient_terms(
        &self,
        parameters: &[f64],
    ) -> Vec<PrecomputedCachedIntegralGradientTerm> {
        let resources = self.resources.read();
        let state = self.ensure_cached_integral_cache_state(&resources);
        if state.values.is_empty() {
            return Vec::new();
        }

        let Some(cache) = resources.caches.first() else {
            return state
                .values
                .into_iter()
                .map(|descriptor| PrecomputedCachedIntegralGradientTerm {
                    mul_node_index: descriptor.mul_node_index,
                    parameter_node_index: descriptor.parameter_node_index,
                    cache_node_index: descriptor.cache_node_index,
                    coefficient: descriptor.coefficient,
                    weighted_gradient: DVector::zeros(parameters.len()),
                })
                .collect();
        };

        let parameter_values = Parameters::new(parameters, &resources.constants);
        let mut amplitude_values = vec![Complex64::ZERO; self.amplitudes.len()];
        self.fill_amplitude_values(
            &mut amplitude_values,
            resources.active_indices(),
            &parameter_values,
            cache,
        );
        let mut amplitude_gradients = (0..self.amplitudes.len())
            .map(|_| DVector::zeros(parameters.len()))
            .collect::<Vec<_>>();
        self.fill_amplitude_gradients(
            &mut amplitude_gradients,
            &resources.active,
            &parameter_values,
            cache,
        );
        let mut value_slots = vec![Complex64::ZERO; state.expression_ir.node_count()];
        let mut gradient_slots = (0..state.expression_ir.node_count())
            .map(|_| DVector::zeros(parameters.len()))
            .collect::<Vec<_>>();
        let _ = state.expression_ir.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut value_slots,
            &mut gradient_slots,
        );

        state
            .values
            .into_iter()
            .map(|descriptor| {
                let parameter_gradient = gradient_slots
                    .get(descriptor.parameter_node_index)
                    .cloned()
                    .unwrap_or_else(|| DVector::zeros(parameters.len()));
                let weighted_gradient = parameter_gradient.map(|value| {
                    value * descriptor.weighted_cache_sum * descriptor.coefficient as f64
                });
                PrecomputedCachedIntegralGradientTerm {
                    mul_node_index: descriptor.mul_node_index,
                    parameter_node_index: descriptor.parameter_node_index,
                    cache_node_index: descriptor.cache_node_index,
                    coefficient: descriptor.coefficient,
                    weighted_gradient,
                }
            })
            .collect()
    }

    fn evaluate_weighted_value_sum_local_components(&self, parameters: &[f64]) -> (f64, f64) {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        #[cfg(not(feature = "expression-ir"))]
        let active_indices = resources.active_indices().to_vec();
        #[cfg(feature = "expression-ir")]
        let state = self.ensure_cached_integral_cache_state(&resources);
        #[cfg(feature = "expression-ir")]
        let zeroed_nodes = {
            let mut nodes = vec![false; state.expression_ir.node_count()];
            for descriptor in &state.values {
                if descriptor.mul_node_index < nodes.len() {
                    nodes[descriptor.mul_node_index] = true;
                }
            }
            nodes
        };
        #[cfg(feature = "expression-ir")]
        let active_index_set = resources.active_indices();
        #[cfg(feature = "expression-ir")]
        let cached_parameter_indices = state
            .execution_sets
            .cached_parameter_amplitudes
            .iter()
            .copied()
            .filter(|index| active_index_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        #[cfg(feature = "expression-ir")]
        let residual_active_indices = state
            .execution_sets
            .residual_amplitudes
            .iter()
            .copied()
            .filter(|index| active_index_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        #[cfg(not(feature = "expression-ir"))]
        let residual_active_indices = active_indices.clone();
        #[cfg(feature = "expression-ir")]
        let slot_count = state.expression_ir.node_count();
        #[cfg(not(feature = "expression-ir"))]
        let slot_count = self.expression_slot_count();

        #[cfg(feature = "expression-ir")]
        let cached_value_sum = {
            if let Some(cache) = resources.caches.first() {
                let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
                self.fill_amplitude_values(
                    &mut amplitude_values,
                    &cached_parameter_indices,
                    &parameters,
                    cache,
                );
                let mut value_slots = vec![Complex64::ZERO; state.expression_ir.node_count()];
                let _ = state
                    .expression_ir
                    .evaluate_into(&amplitude_values, &mut value_slots);
                state
                    .values
                    .iter()
                    .map(|descriptor| {
                        let parameter_factor = value_slots[descriptor.parameter_node_index];
                        (parameter_factor
                            * descriptor.weighted_cache_sum
                            * descriptor.coefficient as f64)
                            .re
                    })
                    .sum::<f64>()
            } else {
                0.0
            }
        };

        #[cfg(feature = "rayon")]
        let residual_sum: f64 = {
            resources
                .caches
                .par_iter()
                .zip(self.dataset.events_local().par_iter())
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                        )
                    },
                    |(amplitude_values, value_slots), (cache, event)| {
                        self.fill_amplitude_values(
                            amplitude_values,
                            &residual_active_indices,
                            &parameters,
                            cache,
                        );
                        #[cfg(feature = "expression-ir")]
                        {
                            let value = state.expression_ir.evaluate_into_with_zeroed_nodes(
                                amplitude_values,
                                value_slots,
                                &zeroed_nodes,
                            );
                            event.weight * value.re
                        }
                        #[cfg(not(feature = "expression-ir"))]
                        {
                            let value = self.evaluate_expression_value_with_scratch(
                                amplitude_values,
                                value_slots,
                            );
                            event.weight * value.re
                        }
                    },
                )
                .sum()
        };

        #[cfg(not(feature = "rayon"))]
        let residual_sum: f64 = {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; slot_count];
            resources
                .caches
                .iter()
                .zip(self.dataset.events_local().iter())
                .map(|(cache, event)| {
                    self.fill_amplitude_values(
                        &mut amplitude_values,
                        &residual_active_indices,
                        &parameters,
                        cache,
                    );
                    #[cfg(feature = "expression-ir")]
                    {
                        let value = state.expression_ir.evaluate_into_with_zeroed_nodes(
                            &amplitude_values,
                            &mut value_slots,
                            &zeroed_nodes,
                        );
                        event.weight * value.re
                    }
                    #[cfg(not(feature = "expression-ir"))]
                    {
                        let value = self.evaluate_expression_value_with_scratch(
                            &amplitude_values,
                            &mut value_slots,
                        );
                        event.weight * value.re
                    }
                })
                .sum()
        };

        #[cfg(feature = "expression-ir")]
        {
            (residual_sum, cached_value_sum)
        }
        #[cfg(not(feature = "expression-ir"))]
        {
            (residual_sum, 0.0)
        }
    }

    /// Weighted sum over local events of the real expression value.
    ///
    /// This returns `sum_e(weight_e * Re(L_e))`.
    pub fn evaluate_weighted_value_sum_local(&self, parameters: &[f64]) -> f64 {
        let (residual_sum, cached_value_sum) =
            self.evaluate_weighted_value_sum_local_components(parameters);
        residual_sum + cached_value_sum
    }

    #[cfg(feature = "mpi")]
    /// Weighted sum over all ranks of the real expression value.
    ///
    /// This returns `sum_{r,e}(weight_{r,e} * Re(L_{r,e}))`.
    pub fn evaluate_weighted_value_sum_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> f64 {
        let (residual_sum_local, cached_value_sum_local) =
            self.evaluate_weighted_value_sum_local_components(parameters);
        let mut residual_sum = 0.0;
        world.all_reduce_into(
            &residual_sum_local,
            &mut residual_sum,
            mpi::collective::SystemOperation::sum(),
        );
        let mut cached_value_sum = 0.0;
        world.all_reduce_into(
            &cached_value_sum_local,
            &mut cached_value_sum,
            mpi::collective::SystemOperation::sum(),
        );
        residual_sum + cached_value_sum
    }

    /// Weighted sum over local events of the real gradient of the expression.
    ///
    /// This returns `sum_e(weight_e * Re(dL_e/dp))` for all free parameters.
    fn evaluate_weighted_gradient_sum_local_components(
        &self,
        parameters: &[f64],
    ) -> (DVector<f64>, DVector<f64>) {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let grad_dim = parameters.len();
        #[cfg(not(feature = "expression-ir"))]
        let active_indices = resources.active_indices().to_vec();
        #[cfg(feature = "expression-ir")]
        let state = self.ensure_cached_integral_cache_state(&resources);
        #[cfg(feature = "expression-ir")]
        let zeroed_nodes = {
            let mut nodes = vec![false; state.expression_ir.node_count()];
            for descriptor in &state.values {
                if descriptor.mul_node_index < nodes.len() {
                    nodes[descriptor.mul_node_index] = true;
                }
            }
            nodes
        };
        #[cfg(feature = "expression-ir")]
        let active_index_set = resources.active_indices();
        #[cfg(feature = "expression-ir")]
        let cached_parameter_indices = state
            .execution_sets
            .cached_parameter_amplitudes
            .iter()
            .copied()
            .filter(|index| active_index_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        #[cfg(feature = "expression-ir")]
        let residual_active_indices = state
            .execution_sets
            .residual_amplitudes
            .iter()
            .copied()
            .filter(|index| active_index_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        #[cfg(not(feature = "expression-ir"))]
        let residual_active_indices = active_indices.clone();
        #[cfg(feature = "expression-ir")]
        let mut cached_parameter_mask = vec![false; amplitude_len];
        #[cfg(feature = "expression-ir")]
        for &index in &cached_parameter_indices {
            cached_parameter_mask[index] = true;
        }
        #[cfg(feature = "expression-ir")]
        let mut residual_active_mask = vec![false; amplitude_len];
        #[cfg(feature = "expression-ir")]
        for &index in &residual_active_indices {
            residual_active_mask[index] = true;
        }
        #[cfg(not(feature = "expression-ir"))]
        let residual_active_mask = resources.active.clone();
        #[cfg(feature = "expression-ir")]
        let slot_count = state.expression_ir.node_count();
        #[cfg(not(feature = "expression-ir"))]
        let slot_count = self.expression_slot_count();

        #[cfg(feature = "expression-ir")]
        let cached_term_sum = {
            if let Some(cache) = resources.caches.first() {
                let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
                self.fill_amplitude_values(
                    &mut amplitude_values,
                    &cached_parameter_indices,
                    &parameters,
                    cache,
                );
                let mut amplitude_gradients = (0..amplitude_len)
                    .map(|_| DVector::zeros(grad_dim))
                    .collect::<Vec<_>>();
                self.fill_amplitude_gradients(
                    &mut amplitude_gradients,
                    &cached_parameter_mask,
                    &parameters,
                    cache,
                );
                let mut value_slots = vec![Complex64::ZERO; slot_count];
                let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
                let _ = state.expression_ir.evaluate_gradient_into(
                    &amplitude_values,
                    &amplitude_gradients,
                    &mut value_slots,
                    &mut gradient_slots,
                );
                state
                    .values
                    .iter()
                    .fold(DVector::zeros(grad_dim), |mut accum, descriptor| {
                        let parameter_gradient = &gradient_slots[descriptor.parameter_node_index];
                        let coefficient = descriptor.coefficient as f64;
                        for (accum_item, gradient_item) in
                            accum.iter_mut().zip(parameter_gradient.iter())
                        {
                            *accum_item +=
                                (*gradient_item * descriptor.weighted_cache_sum * coefficient).re;
                        }
                        accum
                    })
            } else {
                DVector::zeros(grad_dim)
            }
        };

        #[cfg(feature = "rayon")]
        let residual_sum = {
            resources
                .caches
                .par_iter()
                .zip(self.dataset.events_local().par_iter())
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![DVector::zeros(grad_dim); slot_count],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots),
                     (cache, event)| {
                        self.fill_amplitude_values_and_gradients(
                            amplitude_values,
                            gradient_values,
                            &residual_active_indices,
                            &residual_active_mask,
                            &parameters,
                            cache,
                        );
                        #[cfg(feature = "expression-ir")]
                        let gradient = state
                            .expression_ir
                            .evaluate_gradient_into_with_zeroed_nodes(
                                amplitude_values,
                                gradient_values,
                                value_slots,
                                gradient_slots,
                                &zeroed_nodes,
                            );
                        #[cfg(not(feature = "expression-ir"))]
                        let gradient = self.evaluate_expression_gradient_with_scratch(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                        );
                        gradient.map(|value| value.re).scale(event.weight)
                    },
                )
                .reduce(
                    || DVector::zeros(grad_dim),
                    |mut accum, value| {
                        accum += value;
                        accum
                    },
                )
        };

        #[cfg(not(feature = "rayon"))]
        let residual_sum = {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; slot_count];
            let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
            resources
                .caches
                .iter()
                .zip(self.dataset.events_local().iter())
                .map(|(cache, event)| {
                    self.fill_amplitude_values_and_gradients(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &residual_active_indices,
                        &residual_active_mask,
                        &parameters,
                        cache,
                    );
                    #[cfg(feature = "expression-ir")]
                    let gradient = state
                        .expression_ir
                        .evaluate_gradient_into_with_zeroed_nodes(
                            &amplitude_values,
                            &gradient_values,
                            &mut value_slots,
                            &mut gradient_slots,
                            &zeroed_nodes,
                        );
                    #[cfg(not(feature = "expression-ir"))]
                    let gradient = self.evaluate_expression_gradient_with_scratch(
                        &amplitude_values,
                        &gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                    );
                    gradient.map(|value| value.re).scale(event.weight)
                })
                .sum()
        };

        #[cfg(feature = "expression-ir")]
        {
            (residual_sum, cached_term_sum)
        }
        #[cfg(not(feature = "expression-ir"))]
        {
            (residual_sum, DVector::zeros(grad_dim))
        }
    }

    /// Weighted sum over local events of the real gradient of the expression.
    ///
    /// This returns `sum_e(weight_e * Re(dL_e/dp))` for all free parameters.
    pub fn evaluate_weighted_gradient_sum_local(&self, parameters: &[f64]) -> DVector<f64> {
        let (residual_sum, cached_term_sum) =
            self.evaluate_weighted_gradient_sum_local_components(parameters);
        residual_sum + cached_term_sum
    }

    #[cfg(feature = "mpi")]
    /// Weighted sum over all ranks of the real gradient of the expression.
    ///
    /// This returns `sum_{r,e}(weight_{r,e} * Re(dL_{r,e}/dp))`.
    pub fn evaluate_weighted_gradient_sum_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> DVector<f64> {
        let (residual_sum_local, cached_term_sum_local) =
            self.evaluate_weighted_gradient_sum_local_components(parameters);
        let mut residual_sum = vec![0.0; residual_sum_local.len()];
        world.all_reduce_into(
            residual_sum_local.as_slice(),
            &mut residual_sum,
            mpi::collective::SystemOperation::sum(),
        );
        let mut cached_term_sum = vec![0.0; cached_term_sum_local.len()];
        world.all_reduce_into(
            cached_term_sum_local.as_slice(),
            &mut cached_term_sum,
            mpi::collective::SystemOperation::sum(),
        );
        let mut total = DVector::from_vec(residual_sum);
        total += DVector::from_vec(cached_term_sum);
        total
    }

    pub fn evaluate_expression_value_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        scratch: &mut [Complex64],
    ) -> Complex64 {
        self.evaluate_expression_runtime_value_with_scratch(amplitude_values, scratch)
    }

    pub fn evaluate_expression_gradient_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> DVector<Complex64> {
        self.evaluate_expression_runtime_gradient_with_scratch(
            amplitude_values,
            gradient_values,
            value_scratch,
            gradient_scratch,
        )
    }

    pub fn evaluate_expression_value_gradient_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> (Complex64, DVector<Complex64>) {
        self.evaluate_expression_runtime_value_gradient_with_scratch(
            amplitude_values,
            gradient_values,
            value_scratch,
            gradient_scratch,
        )
    }

    pub fn evaluate_expression_value(&self, amplitude_values: &[Complex64]) -> Complex64 {
        self.evaluate_expression_runtime_value(amplitude_values)
    }

    pub fn evaluate_expression_gradient(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        self.evaluate_expression_runtime_gradient(amplitude_values, gradient_values)
    }

    /// Get the list of parameter names in the order they appear in the [`Evaluator::evaluate`]
    /// method.
    pub fn parameters(&self) -> Vec<String> {
        self.parameter_manager.parameters()
    }

    /// Get the list of free parameter names.
    pub fn free_parameters(&self) -> Vec<String> {
        self.parameter_manager.free_parameters()
    }

    /// Get the list of fixed parameter names.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.parameter_manager.fixed_parameters()
    }

    /// Values of parameters fixed to constants.
    pub fn fixed_parameter_values(&self) -> HashMap<String, f64> {
        self.resources.read().fixed_parameter_values()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.parameter_manager.n_free_parameters()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.parameter_manager.n_fixed_parameters()
    }

    /// Total number of parameters.
    pub fn n_parameters(&self) -> usize {
        self.parameter_manager.n_parameters()
    }

    /// Access the parameter manager carried by this evaluator.
    pub fn parameter_manager(&self) -> &ParameterManager {
        &self.parameter_manager
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
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }
    /// Activate an [`Amplitude`] by name and return an error if it is missing.
    pub fn activate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().activate_strict(name)?;
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Activate several [`Amplitude`]s by name, skipping missing entries.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().activate_many(names);
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }
    /// Activate several [`Amplitude`]s by name and return an error if any are missing.
    pub fn activate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().activate_many_strict(names)?;
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&self) {
        self.resources.write().activate_all();
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }

    /// Dectivate an [`Amplitude`] by name, skipping missing entries.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().deactivate(name);
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }

    /// Dectivate an [`Amplitude`] by name and return an error if it is missing.
    pub fn deactivate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().deactivate_strict(name)?;
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Deactivate several [`Amplitude`]s by name, skipping missing entries.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().deactivate_many(names);
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }
    /// Dectivate several [`Amplitude`]s by name and return an error if any are missing.
    pub fn deactivate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().deactivate_many_strict(names)?;
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Deactivate all registered [`Amplitude`]s.
    pub fn deactivate_all(&self) {
        self.resources.write().deactivate_all();
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }

    /// Isolate an [`Amplitude`] by name (deactivate the rest), skipping missing entries.
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().isolate(name);
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }

    /// Isolate an [`Amplitude`] by name (deactivate the rest) and return an error if it is missing.
    pub fn isolate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().isolate_strict(name)?;
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Isolate several [`Amplitude`]s by name (deactivate the rest), skipping missing entries.
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().isolate_many(names);
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
    }

    /// Isolate several [`Amplitude`]s by name (deactivate the rest) and return an error if any are missing.
    pub fn isolate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().isolate_many_strict(names)?;
        #[cfg(feature = "expression-ir")]
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Return a copy of the current active-amplitude mask.
    pub fn active_mask(&self) -> Vec<bool> {
        self.resources.read().active.clone()
    }

    /// Apply a precomputed active-amplitude mask.
    pub fn set_active_mask(&self, mask: &[bool]) -> LadduResult<()> {
        #[cfg(feature = "expression-ir")]
        let resources = {
            let mut resources = self.resources.write();
            if mask.len() != resources.active.len() {
                return Err(LadduError::LengthMismatch {
                    context: "active amplitude mask".to_string(),
                    expected: resources.active.len(),
                    actual: mask.len(),
                });
            }
            resources.active.clone_from_slice(mask);
            resources.refresh_active_indices();
            resources.clone()
        };
        #[cfg(not(feature = "expression-ir"))]
        {
            let mut resources = self.resources.write();
            if mask.len() != resources.active.len() {
                return Err(LadduError::LengthMismatch {
                    context: "active amplitude mask".to_string(),
                    expected: resources.active.len(),
                    actual: mask.len(),
                });
            }
            resources.active.clone_from_slice(mask);
            resources.refresh_active_indices();
        }
        #[cfg(feature = "expression-ir")]
        self.rebuild_runtime_specializations(&resources);
        Ok(())
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
        let amplitude_len = self.amplitudes.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            resources
                .caches
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                        )
                    },
                    |(amplitude_values, expr_slots), cache| {
                        self.fill_amplitude_values(
                            amplitude_values,
                            &active_indices,
                            &parameters,
                            cache,
                        );
                        self.evaluate_expression_value_with_scratch(amplitude_values, expr_slots)
                    },
                )
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut expr_slots = vec![Complex64::ZERO; slot_count];
            resources
                .caches
                .iter()
                .map(|cache| {
                    self.fill_amplitude_values(
                        &mut amplitude_values,
                        &active_indices,
                        &parameters,
                        cache,
                    );
                    self.evaluate_expression_value_with_scratch(&amplitude_values, &mut expr_slots)
                })
                .collect()
        }
    }

    /// Evaluate local events using an explicit active-amplitude mask without mutating evaluator state.
    pub fn evaluate_local_with_active_mask(
        &self,
        parameters: &[f64],
        active_mask: &[bool],
    ) -> LadduResult<Vec<Complex64>> {
        let resources = self.resources.read();
        if active_mask.len() != resources.active.len() {
            return Err(LadduError::LengthMismatch {
                context: "active amplitude mask".to_string(),
                expected: resources.active.len(),
                actual: active_mask.len(),
            });
        }
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let active_indices = active_mask
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| if active { Some(index) } else { None })
            .collect::<Vec<_>>();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            Ok(resources
                .caches
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                        )
                    },
                    |(amplitude_values, expr_slots), cache| {
                        self.fill_amplitude_values(
                            amplitude_values,
                            &active_indices,
                            &parameters,
                            cache,
                        );
                        self.evaluate_expression_value_with_scratch(amplitude_values, expr_slots)
                    },
                )
                .collect())
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut expr_slots = vec![Complex64::ZERO; slot_count];
            Ok(resources
                .caches
                .iter()
                .map(|cache| {
                    self.fill_amplitude_values(
                        &mut amplitude_values,
                        &active_indices,
                        &parameters,
                        cache,
                    );
                    self.evaluate_expression_value_with_scratch(&amplitude_values, &mut expr_slots)
                })
                .collect())
        }
    }

    /// Evaluate the stored expression over local events using a reusable execution context.
    #[cfg(feature = "execution-context-prototype")]
    pub fn evaluate_local_with_ctx(
        &self,
        parameters: &[f64],
        execution_context: &ExecutionContext,
    ) -> Vec<Complex64> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            if !matches!(execution_context.thread_policy(), ThreadPolicy::Single) {
                return execution_context.install(|| {
                    resources
                        .caches
                        .par_iter()
                        .map_init(
                            || {
                                (
                                    vec![Complex64::ZERO; amplitude_len],
                                    vec![Complex64::ZERO; slot_count],
                                )
                            },
                            |(amplitude_values, expr_slots), cache| {
                                self.fill_amplitude_values(
                                    amplitude_values,
                                    &active_indices,
                                    &parameters,
                                    cache,
                                );
                                self.evaluate_expression_value_with_scratch(
                                    amplitude_values,
                                    expr_slots,
                                )
                            },
                        )
                        .collect()
                });
            }
        }
        execution_context.with_scratch(|scratch| {
            let (amplitude_values, expr_slots) =
                scratch.reserve_value_workspaces(amplitude_len, slot_count);
            resources
                .caches
                .iter()
                .map(|cache| {
                    self.fill_amplitude_values(
                        amplitude_values,
                        &active_indices,
                        &parameters,
                        cache,
                    );
                    self.evaluate_expression_value_with_scratch(amplitude_values, expr_slots)
                })
                .collect()
        })
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
            // NOTE: gather is required here because the public MPI API returns full per-event outputs.
            // Do not replace with all-reduce unless semantics change to scalar aggregates only.
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_evaluation, &mut partitioned_buffer);
        }
        buffer
    }

    #[cfg(all(feature = "mpi", feature = "execution-context-prototype"))]
    fn evaluate_mpi_with_ctx(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
        execution_context: &ExecutionContext,
    ) -> Vec<Complex64> {
        let local_evaluation = self.evaluate_local_with_ctx(parameters, execution_context);
        let n_events = self.dataset.n_events();
        let mut buffer: Vec<Complex64> = vec![Complex64::ZERO; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required here because the public MPI API returns full per-event outputs.
            // Do not replace with all-reduce unless semantics change to scalar aggregates only.
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

    /// Evaluate the stored expression with a reusable execution context.
    ///
    /// This is intended for repeated calls with the same context instance.
    /// Thread behavior follows [`ThreadPolicy`](crate::ThreadPolicy) configured on
    /// [`ExecutionContext`](crate::ExecutionContext).
    #[cfg(feature = "execution-context-prototype")]
    pub fn evaluate_with_ctx(
        &self,
        parameters: &[f64],
        execution_context: &ExecutionContext,
    ) -> Vec<Complex64> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.evaluate_mpi_with_ctx(parameters, &world, execution_context);
            }
        }
        self.evaluate_local_with_ctx(parameters, execution_context)
    }

    /// See [`Evaluator::evaluate_local`]. This method evaluates over a subset of events rather
    /// than all events in the total dataset.
    pub fn evaluate_batch_local(&self, parameters: &[f64], indices: &[usize]) -> Vec<Complex64> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            indices
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                        )
                    },
                    |(amplitude_values, expr_slots), &idx| {
                        let cache = &resources.caches[idx];
                        self.fill_amplitude_values(
                            amplitude_values,
                            &active_indices,
                            &parameters,
                            cache,
                        );
                        self.evaluate_expression_value_with_scratch(amplitude_values, expr_slots)
                    },
                )
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut expr_slots = vec![Complex64::ZERO; slot_count];
            indices
                .iter()
                .map(|&idx| {
                    let cache = &resources.caches[idx];
                    self.fill_amplitude_values(
                        &mut amplitude_values,
                        &active_indices,
                        &parameters,
                        cache,
                    );
                    self.evaluate_expression_value_with_scratch(&amplitude_values, &mut expr_slots)
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
        let amplitude_len = self.amplitudes.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            resources
                .caches
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![DVector::zeros(grad_dim); slot_count],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), cache| {
                        self.evaluate_cache_gradient_with_scratch(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        )
                    },
                )
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; slot_count];
            let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
            resources
                .caches
                .iter()
                .map(|cache| {
                    self.evaluate_cache_gradient_with_scratch(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    )
                })
                .collect()
        }
    }

    /// Evaluate the gradient over local events using a reusable execution context.
    #[cfg(feature = "execution-context-prototype")]
    pub fn evaluate_gradient_local_with_ctx(
        &self,
        parameters: &[f64],
        execution_context: &ExecutionContext,
    ) -> Vec<DVector<Complex64>> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            if !matches!(execution_context.thread_policy(), ThreadPolicy::Single) {
                return execution_context.install(|| {
                    resources
                        .caches
                        .par_iter()
                        .map_init(
                            || {
                                (
                                    vec![Complex64::ZERO; amplitude_len],
                                    vec![DVector::zeros(grad_dim); amplitude_len],
                                    vec![Complex64::ZERO; slot_count],
                                    vec![DVector::zeros(grad_dim); slot_count],
                                )
                            },
                            |(amplitude_values, gradient_values, value_slots, gradient_slots),
                             cache| {
                                self.evaluate_cache_gradient_with_scratch(
                                    amplitude_values,
                                    gradient_values,
                                    value_slots,
                                    gradient_slots,
                                    &active_indices,
                                    &resources.active,
                                    &parameters,
                                    cache,
                                )
                            },
                        )
                        .collect()
                });
            }
        }
        execution_context.with_scratch(|scratch| {
            let (amplitude_values, value_slots, gradient_values, gradient_slots) =
                scratch.reserve_gradient_workspaces(amplitude_len, slot_count, grad_dim);
            resources
                .caches
                .iter()
                .map(|cache| {
                    self.evaluate_cache_gradient_with_scratch(
                        amplitude_values,
                        gradient_values,
                        value_slots,
                        gradient_slots,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    )
                })
                .collect()
        })
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
        let (counts, displs) = world.get_flattened_counts_displs(n_events, parameters.len());
        {
            // NOTE: gather is required here because the public MPI API returns full per-event gradients.
            // Do not replace with all-reduce unless semantics change to aggregate-only outputs.
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
            .map(DVector::from_row_slice)
            .collect()
    }

    #[cfg(all(feature = "mpi", feature = "execution-context-prototype"))]
    fn evaluate_gradient_mpi_with_ctx(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
        execution_context: &ExecutionContext,
    ) -> Vec<DVector<Complex64>> {
        let local_evaluation = self.evaluate_gradient_local_with_ctx(parameters, execution_context);
        let n_events = self.dataset.n_events();
        let mut buffer: Vec<Complex64> = vec![Complex64::ZERO; n_events * parameters.len()];
        let (counts, displs) = world.get_flattened_counts_displs(n_events, parameters.len());
        {
            // NOTE: gather is required here because the public MPI API returns full per-event gradients.
            // Do not replace with all-reduce unless semantics change to aggregate-only outputs.
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
            .map(DVector::from_row_slice)
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

    /// Evaluate the gradient with a reusable execution context.
    ///
    /// This is intended for repeated calls with the same context instance.
    /// Thread behavior follows [`ThreadPolicy`](crate::ThreadPolicy) configured on
    /// [`ExecutionContext`](crate::ExecutionContext).
    #[cfg(feature = "execution-context-prototype")]
    pub fn evaluate_gradient_with_ctx(
        &self,
        parameters: &[f64],
        execution_context: &ExecutionContext,
    ) -> Vec<DVector<Complex64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.evaluate_gradient_mpi_with_ctx(parameters, &world, execution_context);
            }
        }
        self.evaluate_gradient_local_with_ctx(parameters, execution_context)
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
        let amplitude_len = self.amplitudes.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            indices
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![DVector::zeros(grad_dim); slot_count],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), &idx| {
                        let cache = &resources.caches[idx];
                        self.evaluate_cache_gradient_with_scratch(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        )
                    },
                )
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; slot_count];
            let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
            resources
                .caches
                .iter()
                .map(|cache| {
                    self.evaluate_cache_gradient_with_scratch(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    )
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

    /// Evaluate the stored expression and its gradient over local events in one fused pass.
    pub fn evaluate_with_gradient_local(
        &self,
        parameters: &[f64],
    ) -> Vec<(Complex64, DVector<Complex64>)> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            resources
                .caches
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![DVector::zeros(grad_dim); slot_count],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), cache| {
                        self.evaluate_cache_value_gradient_with_scratch(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        )
                    },
                )
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; slot_count];
            let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
            resources
                .caches
                .iter()
                .map(|cache| {
                    self.evaluate_cache_value_gradient_with_scratch(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    )
                })
                .collect()
        }
    }

    /// Evaluate local events and gradients with an explicit active-amplitude mask without mutating evaluator state.
    pub fn evaluate_with_gradient_local_with_active_mask(
        &self,
        parameters: &[f64],
        active_mask: &[bool],
    ) -> LadduResult<Vec<(Complex64, DVector<Complex64>)>> {
        let resources = self.resources.read();
        if active_mask.len() != resources.active.len() {
            return Err(LadduError::LengthMismatch {
                context: "active amplitude mask".to_string(),
                expected: resources.active.len(),
                actual: active_mask.len(),
            });
        }
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let grad_dim = parameters.len();
        let active_indices = active_mask
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| if active { Some(index) } else { None })
            .collect::<Vec<_>>();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            Ok(resources
                .caches
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![DVector::zeros(grad_dim); slot_count],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), cache| {
                        self.evaluate_cache_value_gradient_with_scratch(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            &active_indices,
                            active_mask,
                            &parameters,
                            cache,
                        )
                    },
                )
                .collect())
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; slot_count];
            let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
            Ok(resources
                .caches
                .iter()
                .map(|cache| {
                    self.evaluate_cache_value_gradient_with_scratch(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        &active_indices,
                        active_mask,
                        &parameters,
                        cache,
                    )
                })
                .collect())
        }
    }

    /// Evaluate the stored expression and its gradient over a local subset of events in one fused pass.
    pub fn evaluate_with_gradient_batch_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> Vec<(Complex64, DVector<Complex64>)> {
        let resources = self.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = self.amplitudes.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_slot_count();
        #[cfg(feature = "rayon")]
        {
            indices
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![DVector::zeros(grad_dim); slot_count],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), &idx| {
                        let cache = &resources.caches[idx];
                        self.evaluate_cache_value_gradient_with_scratch(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        )
                    },
                )
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; slot_count];
            let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
            indices
                .iter()
                .map(|&idx| {
                    let cache = &resources.caches[idx];
                    self.evaluate_cache_value_gradient_with_scratch(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    )
                })
                .collect()
        }
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
    beam_energy: crate::ScalarID,
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
            beam_energy: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for TestAmplitude {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_re = resources.register_parameter(&self.re)?;
        self.pid_im = resources.register_parameter(&self.im)?;
        self.beam_energy = resources.register_scalar(Some(&format!("{}.beam_energy", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        let beam = event.p4_at(0);
        cache.store_scalar(self.beam_energy, beam.e());
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid_re), parameters.get(self.pid_im))
            * cache.get_scalar(self.beam_energy)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let beam_energy = cache.get_scalar(self.beam_energy);
        if let ParameterID::Parameter(ind) = self.pid_re {
            gradient[ind] = Complex64::ONE * beam_energy;
        }
        if let ParameterID::Parameter(ind) = self.pid_im {
            gradient[ind] = Complex64::I * beam_energy;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::data::{test_dataset, test_event, DatasetMetadata, EventData};

    use super::*;
    use crate::resources::{Cache, ParameterID, Parameters, Resources, ScalarID};
    use crate::utils::vectors::Vec4;
    use approx::assert_relative_eq;
    #[cfg(feature = "mpi")]
    use mpi_test::mpi_test;
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

        fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
            Complex64::new(parameters.get(self.pid_re), parameters.get(self.pid_im))
        }

        fn compute_gradient(
            &self,
            _parameters: &Parameters,
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

    #[derive(Clone, Serialize, Deserialize)]
    pub struct ParameterOnlyScalar {
        name: String,
        value: ParameterLike,
        pid: ParameterID,
    }

    impl ParameterOnlyScalar {
        #[allow(clippy::new_ret_no_self)]
        pub fn new(name: &str, value: ParameterLike) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                value,
                pid: Default::default(),
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for ParameterOnlyScalar {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.pid = resources.register_parameter(&self.value)?;
            resources.register_amplitude(&self.name)
        }

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::ParameterOnly
        }

        fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
            Complex64::new(parameters.get(self.pid), 0.0)
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub struct CacheOnlyScalar {
        name: String,
        beam_energy: ScalarID,
    }

    impl CacheOnlyScalar {
        #[allow(clippy::new_ret_no_self)]
        pub fn new(name: &str) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                beam_energy: Default::default(),
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for CacheOnlyScalar {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.beam_energy =
                resources.register_scalar(Some(&format!("{}.beam_energy", self.name)));
            resources.register_amplitude(&self.name)
        }

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::CacheOnly
        }

        fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
            cache.store_scalar(self.beam_energy, event.p4_at(0).e());
        }

        fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
            Complex64::new(cache.get_scalar(self.beam_energy), 0.0)
        }
    }

    #[derive(Clone, Copy)]
    enum DeterministicFixtureKind {
        Separable,
        Partial,
        NonSeparable,
    }

    struct DeterministicFixture {
        expression: Expression,
        dataset: Arc<Dataset>,
        parameters: Vec<f64>,
    }

    const DETERMINISTIC_STRICT_ABS_TOL: f64 = 1e-12;
    const DETERMINISTIC_STRICT_REL_TOL: f64 = 1e-10;

    fn deterministic_fixture_dataset() -> Arc<Dataset> {
        let metadata = Arc::new(DatasetMetadata::default());
        let events = vec![
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
                aux: vec![],
                weight: 0.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
                aux: vec![],
                weight: -1.25,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
                aux: vec![],
                weight: 2.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
                aux: vec![],
                weight: 0.75,
            }),
        ];
        Arc::new(Dataset::new_with_metadata(events, metadata))
    }

    fn make_deterministic_fixture(kind: DeterministicFixtureKind) -> DeterministicFixture {
        let dataset = deterministic_fixture_dataset();
        match kind {
            DeterministicFixtureKind::Separable => {
                let p1 = ParameterOnlyScalar::new("p1", parameter("p1"))
                    .expect("separable p1 should build");
                let p2 = ParameterOnlyScalar::new("p2", parameter("p2"))
                    .expect("separable p2 should build");
                let c1 = CacheOnlyScalar::new("c1").expect("separable c1 should build");
                let c2 = CacheOnlyScalar::new("c2").expect("separable c2 should build");
                DeterministicFixture {
                    expression: (&p1 * &c1) + &(&p2 * &c2),
                    dataset,
                    parameters: vec![0.4, -0.3],
                }
            }
            DeterministicFixtureKind::Partial => {
                let p =
                    ParameterOnlyScalar::new("p", parameter("p")).expect("partial p should build");
                let c = CacheOnlyScalar::new("c").expect("partial c should build");
                let m = TestAmplitude::new("m", parameter("mr"), parameter("mi"))
                    .expect("partial m should build");
                DeterministicFixture {
                    expression: (&p * &c) + &m,
                    dataset,
                    parameters: vec![0.55, 0.2, -0.15],
                }
            }
            DeterministicFixtureKind::NonSeparable => {
                let m1 = TestAmplitude::new("m1", parameter("m1r"), parameter("m1i"))
                    .expect("non-separable m1 should build");
                let m2 = TestAmplitude::new("m2", parameter("m2r"), parameter("m2i"))
                    .expect("non-separable m2 should build");
                DeterministicFixture {
                    expression: &m1 * &m2,
                    dataset,
                    parameters: vec![0.25, -0.4, 0.6, 0.1],
                }
            }
        }
    }

    fn assert_weighted_sum_matches_eventwise_baseline(fixture: &DeterministicFixture) {
        let evaluator = fixture
            .expression
            .load(&fixture.dataset)
            .expect("fixture evaluator should load");
        let expected_value = evaluator
            .evaluate_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let expected_gradient = evaluator
            .evaluate_gradient_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(
                DVector::zeros(fixture.parameters.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        let actual_value = evaluator.evaluate_weighted_value_sum_local(&fixture.parameters);
        let actual_gradient = evaluator.evaluate_weighted_gradient_sum_local(&fixture.parameters);
        assert_relative_eq!(
            actual_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }
    }

    #[test]
    fn test_deterministic_fixture_weighted_sums_stable_across_activation_mask_toggle() {
        let fixture = make_deterministic_fixture(DeterministicFixtureKind::Partial);
        let evaluator = fixture
            .expression
            .load(&fixture.dataset)
            .expect("fixture evaluator should load");
        let original_mask = evaluator.active_mask();

        let expected_value = evaluator
            .evaluate_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let expected_gradient = evaluator
            .evaluate_gradient_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(
                DVector::zeros(fixture.parameters.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        let actual_value = evaluator.evaluate_weighted_value_sum_local(&fixture.parameters);
        assert_relative_eq!(
            actual_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );
        let actual_gradient = evaluator.evaluate_weighted_gradient_sum_local(&fixture.parameters);
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }

        evaluator.isolate_many(&["p", "c"]);
        let expected_value = evaluator
            .evaluate_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let expected_gradient = evaluator
            .evaluate_gradient_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(
                DVector::zeros(fixture.parameters.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        let actual_value = evaluator.evaluate_weighted_value_sum_local(&fixture.parameters);
        assert_relative_eq!(
            actual_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );
        let actual_gradient = evaluator.evaluate_weighted_gradient_sum_local(&fixture.parameters);
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }

        evaluator
            .set_active_mask(&original_mask)
            .expect("original fixture active mask should restore");
        let expected_value = evaluator
            .evaluate_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let expected_gradient = evaluator
            .evaluate_gradient_local(&fixture.parameters)
            .iter()
            .zip(fixture.dataset.events_local().iter())
            .fold(
                DVector::zeros(fixture.parameters.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        let actual_value = evaluator.evaluate_weighted_value_sum_local(&fixture.parameters);
        assert_relative_eq!(
            actual_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );
        let actual_gradient = evaluator.evaluate_weighted_gradient_sum_local(&fixture.parameters);
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }
    }

    #[test]
    fn test_deterministic_fixtures_match_eventwise_weighted_sums() {
        let separable = make_deterministic_fixture(DeterministicFixtureKind::Separable);
        let partial = make_deterministic_fixture(DeterministicFixtureKind::Partial);
        let non_separable = make_deterministic_fixture(DeterministicFixtureKind::NonSeparable);

        assert_weighted_sum_matches_eventwise_baseline(&separable);
        assert_weighted_sum_matches_eventwise_baseline(&partial);
        assert_weighted_sum_matches_eventwise_baseline(&non_separable);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_deterministic_fixtures_cover_separable_partial_non_separable_models() {
        let separable = make_deterministic_fixture(DeterministicFixtureKind::Separable);
        let partial = make_deterministic_fixture(DeterministicFixtureKind::Partial);
        let non_separable = make_deterministic_fixture(DeterministicFixtureKind::NonSeparable);

        let separable_evaluator = separable
            .expression
            .load(&separable.dataset)
            .expect("separable evaluator should load");
        let partial_evaluator = partial
            .expression
            .load(&partial.dataset)
            .expect("partial evaluator should load");
        let non_separable_evaluator = non_separable
            .expression
            .load(&non_separable.dataset)
            .expect("non-separable evaluator should load");

        assert_eq!(
            separable_evaluator
                .expression_precomputed_cached_integrals()
                .len(),
            2
        );
        assert_eq!(
            partial_evaluator
                .expression_precomputed_cached_integrals()
                .len(),
            1
        );
        assert!(non_separable_evaluator
            .expression_precomputed_cached_integrals()
            .is_empty());
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
    fn test_load_compiles_expression_ir_once() {
        let expr = (TestAmplitude::new("a", parameter("ar"), parameter("ai")).unwrap()
            + TestAmplitude::new("b", parameter("br"), parameter("bi")).unwrap())
        .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert!(evaluator.expression_slot_count() > 0);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_value_matches_program() {
        let expr = ((TestAmplitude::new("a", parameter("ar"), parameter("ai")).unwrap()
            + TestAmplitude::new("b", parameter("br"), parameter("bi")).unwrap())
            * TestAmplitude::new("c", parameter("cr"), parameter("ci")).unwrap())
        .conj()
        .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let resources = evaluator.resources.read();
        let parameters = Parameters::new(&[1.0, 0.25, -0.8, 0.5, 0.2, -1.1], &resources.constants);
        let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
        evaluator.fill_amplitude_values(
            &mut amplitude_values,
            resources.active_indices(),
            &parameters,
            &resources.caches[0],
        );
        let mut ir_slots = vec![Complex64::ZERO; evaluator.expression_ir().node_count()];
        let mut program_slots = vec![Complex64::ZERO; evaluator.expression_program.slot_count()];
        let lowered_runtime = evaluator.lowered_runtime().unwrap();
        let lowered_program = lowered_runtime.value_program().unwrap();
        let mut lowered_slots = vec![Complex64::ZERO; lowered_program.scratch_slots()];
        let lowered_value =
            evaluator.evaluate_expression_value_with_scratch(&amplitude_values, &mut ir_slots);
        let direct_lowered_value =
            lowered_program.evaluate_into(&amplitude_values, &mut lowered_slots);
        let ir_value = evaluator
            .expression_ir()
            .evaluate_into(&amplitude_values, &mut ir_slots);
        let program_value = evaluator
            .expression_program
            .evaluate_into(&amplitude_values, &mut program_slots);
        assert_relative_eq!(lowered_value.re, direct_lowered_value.re);
        assert_relative_eq!(lowered_value.im, direct_lowered_value.im);
        assert_relative_eq!(lowered_value.re, ir_value.re);
        assert_relative_eq!(lowered_value.im, ir_value.im);
        assert_relative_eq!(ir_value.re, program_value.re);
        assert_relative_eq!(ir_value.im, program_value.im);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_load_initializes_with_lowered_value_runtime() {
        let expr = TestAmplitude::new("a", parameter("ar"), parameter("ai"))
            .unwrap()
            .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let lowered_runtime = evaluator.lowered_runtime().unwrap();
        assert!(lowered_runtime.value_program().is_some());
        assert!(lowered_runtime.gradient_program().is_some());
        assert!(lowered_runtime.value_gradient_program().is_some());
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_gradient_matches_program() {
        let expr = (TestAmplitude::new("a", parameter("ar"), parameter("ai")).unwrap()
            * TestAmplitude::new("b", parameter("br"), parameter("bi")).unwrap())
        .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let resources = evaluator.resources.read();
        let parameters = Parameters::new(&[1.0, 0.25, -0.8, 0.5], &resources.constants);
        let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
        evaluator.fill_amplitude_values(
            &mut amplitude_values,
            resources.active_indices(),
            &parameters,
            &resources.caches[0],
        );
        let mut active_mask = vec![false; evaluator.amplitudes.len()];
        for &index in resources.active_indices() {
            active_mask[index] = true;
        }
        let mut amplitude_gradients = (0..evaluator.amplitudes.len())
            .map(|_| DVector::zeros(parameters.len()))
            .collect::<Vec<_>>();
        evaluator.fill_amplitude_gradients(
            &mut amplitude_gradients,
            &active_mask,
            &parameters,
            &resources.caches[0],
        );
        let mut ir_value_slots = vec![Complex64::ZERO; evaluator.expression_ir().node_count()];
        let mut ir_gradient_slots: Vec<DVector<Complex64>> =
            (0..evaluator.expression_ir().node_count())
                .map(|_| DVector::zeros(parameters.len()))
                .collect();
        let mut program_value_slots =
            vec![Complex64::ZERO; evaluator.expression_program.slot_count()];
        let mut program_gradient_slots: Vec<DVector<Complex64>> =
            (0..evaluator.expression_program.slot_count())
                .map(|_| DVector::zeros(parameters.len()))
                .collect();
        let lowered_runtime = evaluator.lowered_runtime().unwrap();
        let lowered_program = lowered_runtime.gradient_program().unwrap();
        let mut lowered_value_slots = vec![Complex64::ZERO; lowered_program.scratch_slots()];
        let mut lowered_gradient_slots: Vec<DVector<Complex64>> = (0..lowered_program
            .scratch_slots())
            .map(|_| DVector::zeros(parameters.len()))
            .collect();
        let active_gradient = evaluator.evaluate_expression_gradient_with_scratch(
            &amplitude_values,
            &amplitude_gradients,
            &mut ir_value_slots,
            &mut ir_gradient_slots,
        );
        let ir_gradient = evaluator.expression_ir().evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut ir_value_slots,
            &mut ir_gradient_slots,
        );
        let program_gradient = evaluator.expression_program.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut program_value_slots,
            &mut program_gradient_slots,
        );
        let lowered_gradient = lowered_program.evaluate_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut lowered_value_slots,
            &mut lowered_gradient_slots,
        );
        for (active, lowered) in active_gradient.iter().zip(lowered_gradient.iter()) {
            assert_relative_eq!(active.re, lowered.re);
            assert_relative_eq!(active.im, lowered.im);
        }
        for (lowered, ir) in lowered_gradient.iter().zip(ir_gradient.iter()) {
            assert_relative_eq!(lowered.re, ir.re);
            assert_relative_eq!(lowered.im, ir.im);
        }
        for (ir, program) in ir_gradient.iter().zip(program_gradient.iter()) {
            assert_relative_eq!(ir.re, program.re);
            assert_relative_eq!(ir.im, program.im);
        }
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_value_gradient_matches_program() {
        let expr = ((TestAmplitude::new("a", parameter("ar"), parameter("ai")).unwrap()
            + TestAmplitude::new("b", parameter("br"), parameter("bi")).unwrap())
            * TestAmplitude::new("c", parameter("cr"), parameter("ci")).unwrap())
        .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let resources = evaluator.resources.read();
        let parameters = Parameters::new(&[1.0, 0.25, -0.8, 0.5, 0.2, -1.1], &resources.constants);
        let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
        evaluator.fill_amplitude_values(
            &mut amplitude_values,
            resources.active_indices(),
            &parameters,
            &resources.caches[0],
        );
        let mut active_mask = vec![false; evaluator.amplitudes.len()];
        for &index in resources.active_indices() {
            active_mask[index] = true;
        }
        let mut amplitude_gradients = (0..evaluator.amplitudes.len())
            .map(|_| DVector::zeros(parameters.len()))
            .collect::<Vec<_>>();
        evaluator.fill_amplitude_gradients(
            &mut amplitude_gradients,
            &active_mask,
            &parameters,
            &resources.caches[0],
        );
        let mut ir_value_slots = vec![Complex64::ZERO; evaluator.expression_ir().node_count()];
        let mut ir_gradient_slots: Vec<DVector<Complex64>> =
            (0..evaluator.expression_ir().node_count())
                .map(|_| DVector::zeros(parameters.len()))
                .collect();
        let mut program_value_slots =
            vec![Complex64::ZERO; evaluator.expression_program.slot_count()];
        let mut program_gradient_slots: Vec<DVector<Complex64>> =
            (0..evaluator.expression_program.slot_count())
                .map(|_| DVector::zeros(parameters.len()))
                .collect();
        let lowered_runtime = evaluator.lowered_runtime().unwrap();
        let lowered_program = lowered_runtime.value_gradient_program().unwrap();
        let mut lowered_value_slots = vec![Complex64::ZERO; lowered_program.scratch_slots()];
        let mut lowered_gradient_slots: Vec<DVector<Complex64>> = (0..lowered_program
            .scratch_slots())
            .map(|_| DVector::zeros(parameters.len()))
            .collect();

        let active_value_gradient = evaluator.evaluate_expression_value_gradient_with_scratch(
            &amplitude_values,
            &amplitude_gradients,
            &mut ir_value_slots,
            &mut ir_gradient_slots,
        );
        let ir_value_gradient = evaluator.expression_ir().evaluate_value_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut ir_value_slots,
            &mut ir_gradient_slots,
        );
        let program_value_gradient = evaluator.expression_program.evaluate_value_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut program_value_slots,
            &mut program_gradient_slots,
        );
        let lowered_value_gradient = lowered_program.evaluate_value_gradient_into(
            &amplitude_values,
            &amplitude_gradients,
            &mut lowered_value_slots,
            &mut lowered_gradient_slots,
        );

        assert_relative_eq!(active_value_gradient.0.re, lowered_value_gradient.0.re);
        assert_relative_eq!(active_value_gradient.0.im, lowered_value_gradient.0.im);
        for (active, lowered) in active_value_gradient
            .1
            .iter()
            .zip(lowered_value_gradient.1.iter())
        {
            assert_relative_eq!(active.re, lowered.re);
            assert_relative_eq!(active.im, lowered.im);
        }
        assert_relative_eq!(lowered_value_gradient.0.re, ir_value_gradient.0.re);
        assert_relative_eq!(lowered_value_gradient.0.im, ir_value_gradient.0.im);
        for (lowered, ir) in lowered_value_gradient
            .1
            .iter()
            .zip(ir_value_gradient.1.iter())
        {
            assert_relative_eq!(lowered.re, ir.re);
            assert_relative_eq!(lowered.im, ir.im);
        }
        assert_relative_eq!(ir_value_gradient.0.re, program_value_gradient.0.re);
        assert_relative_eq!(ir_value_gradient.0.im, program_value_gradient.0.im);
        for (ir, program) in ir_value_gradient
            .1
            .iter()
            .zip(program_value_gradient.1.iter())
        {
            assert_relative_eq!(ir.re, program.re);
            assert_relative_eq!(ir.im, program.im);
        }
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_runtime_backends_match_on_identical_workload() {
        let expr = ((TestAmplitude::new("a", parameter("ar"), parameter("ai")).unwrap()
            + TestAmplitude::new("b", parameter("br"), parameter("bi")).unwrap())
            * TestAmplitude::new("c", parameter("cr"), parameter("ci")).unwrap())
        .conj()
        .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let (amplitude_values, amplitude_gradients, grad_dim) = {
            let resources = evaluator.resources.read();
            let parameters =
                Parameters::new(&[1.0, 0.25, -0.8, 0.5, 0.2, -1.1], &resources.constants);
            let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
            evaluator.fill_amplitude_values(
                &mut amplitude_values,
                resources.active_indices(),
                &parameters,
                &resources.caches[0],
            );
            let mut active_mask = vec![false; evaluator.amplitudes.len()];
            for &index in resources.active_indices() {
                active_mask[index] = true;
            }
            let mut amplitude_gradients = (0..evaluator.amplitudes.len())
                .map(|_| DVector::zeros(parameters.len()))
                .collect::<Vec<_>>();
            evaluator.fill_amplitude_gradients(
                &mut amplitude_gradients,
                &active_mask,
                &parameters,
                &resources.caches[0],
            );
            (amplitude_values, amplitude_gradients, parameters.len())
        };

        let backends = [
            ExpressionRuntimeBackend::LegacyProgram,
            ExpressionRuntimeBackend::IrInterpreter,
            ExpressionRuntimeBackend::Lowered,
        ];

        let mut value_results = Vec::new();
        let mut gradient_results = Vec::new();
        let mut value_gradient_results = Vec::new();
        for backend in backends {
            let mut backend_evaluator = evaluator.clone();
            backend_evaluator.set_expression_runtime_backend(backend);
            let slot_count = backend_evaluator.expression_slot_count();
            let mut value_scratch = vec![Complex64::ZERO; slot_count];
            let mut gradient_value_scratch = vec![Complex64::ZERO; slot_count];
            let mut fused_value_scratch = vec![Complex64::ZERO; slot_count];
            let mut gradient_scratch = (0..slot_count)
                .map(|_| DVector::zeros(grad_dim))
                .collect::<Vec<_>>();
            let mut fused_gradient_scratch = (0..slot_count)
                .map(|_| DVector::zeros(grad_dim))
                .collect::<Vec<_>>();

            value_results.push(
                backend_evaluator
                    .evaluate_expression_value_with_scratch(&amplitude_values, &mut value_scratch),
            );
            gradient_results.push(backend_evaluator.evaluate_expression_gradient_with_scratch(
                &amplitude_values,
                &amplitude_gradients,
                &mut gradient_value_scratch,
                &mut gradient_scratch,
            ));
            value_gradient_results.push(
                backend_evaluator.evaluate_expression_value_gradient_with_scratch(
                    &amplitude_values,
                    &amplitude_gradients,
                    &mut fused_value_scratch,
                    &mut fused_gradient_scratch,
                ),
            );
        }

        for pair in value_results.windows(2) {
            assert_relative_eq!(pair[0].re, pair[1].re);
            assert_relative_eq!(pair[0].im, pair[1].im);
        }
        for pair in gradient_results.windows(2) {
            for (left, right) in pair[0].iter().zip(pair[1].iter()) {
                assert_relative_eq!(left.re, right.re);
                assert_relative_eq!(left.im, right.im);
            }
        }
        for pair in value_gradient_results.windows(2) {
            assert_relative_eq!(pair[0].0.re, pair[1].0.re);
            assert_relative_eq!(pair[0].0.im, pair[1].0.im);
            for (left, right) in pair[0].1.iter().zip(pair[1].1.iter()) {
                assert_relative_eq!(left.re, right.re);
                assert_relative_eq!(left.im, right.im);
            }
        }
    }

    #[cfg(feature = "expression-ir")]
    fn assert_runtime_backends_match_fixture(kind: DeterministicFixtureKind) {
        let fixture = make_deterministic_fixture(kind);
        let evaluator = fixture
            .expression
            .load(&fixture.dataset)
            .expect("fixture evaluator should load");
        let resources = evaluator.resources.read();
        let parameters = Parameters::new(&fixture.parameters, &resources.constants);
        let mut amplitude_values = vec![Complex64::ZERO; evaluator.amplitudes.len()];
        evaluator.fill_amplitude_values(
            &mut amplitude_values,
            resources.active_indices(),
            &parameters,
            &resources.caches[0],
        );
        let mut active_mask = vec![false; evaluator.amplitudes.len()];
        for &index in resources.active_indices() {
            active_mask[index] = true;
        }
        let mut amplitude_gradients = (0..evaluator.amplitudes.len())
            .map(|_| DVector::zeros(parameters.len()))
            .collect::<Vec<_>>();
        evaluator.fill_amplitude_gradients(
            &mut amplitude_gradients,
            &active_mask,
            &parameters,
            &resources.caches[0],
        );
        let grad_dim = parameters.len();
        drop(resources);

        let backends = [
            ExpressionRuntimeBackend::LegacyProgram,
            ExpressionRuntimeBackend::IrInterpreter,
            ExpressionRuntimeBackend::Lowered,
        ];

        let mut local_values = Vec::new();
        let mut local_gradients = Vec::new();
        let mut fused_values = Vec::new();
        let mut fused_gradients = Vec::new();
        for backend in backends {
            let mut backend_evaluator = evaluator.clone();
            backend_evaluator.set_expression_runtime_backend(backend);
            local_values.push(backend_evaluator.evaluate_local(&fixture.parameters));
            local_gradients.push(backend_evaluator.evaluate_gradient_local(&fixture.parameters));

            let slot_count = backend_evaluator.expression_slot_count();
            let mut value_scratch = vec![Complex64::ZERO; slot_count];
            let mut gradient_scratch = (0..slot_count)
                .map(|_| DVector::zeros(grad_dim))
                .collect::<Vec<_>>();
            let (value, gradient) = backend_evaluator
                .evaluate_expression_value_gradient_with_scratch(
                    &amplitude_values,
                    &amplitude_gradients,
                    &mut value_scratch,
                    &mut gradient_scratch,
                );
            fused_values.push(value);
            fused_gradients.push(gradient);
        }

        for pair in local_values.windows(2) {
            for (left, right) in pair[0].iter().zip(pair[1].iter()) {
                assert_relative_eq!(left.re, right.re);
                assert_relative_eq!(left.im, right.im);
            }
        }
        for pair in local_gradients.windows(2) {
            for (left_vec, right_vec) in pair[0].iter().zip(pair[1].iter()) {
                for (left, right) in left_vec.iter().zip(right_vec.iter()) {
                    assert_relative_eq!(left.re, right.re);
                    assert_relative_eq!(left.im, right.im);
                }
            }
        }
        for pair in fused_values.windows(2) {
            assert_relative_eq!(pair[0].re, pair[1].re);
            assert_relative_eq!(pair[0].im, pair[1].im);
        }
        for pair in fused_gradients.windows(2) {
            for (left, right) in pair[0].iter().zip(pair[1].iter()) {
                assert_relative_eq!(left.re, right.re);
                assert_relative_eq!(left.im, right.im);
            }
        }
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_runtime_backends_match_representative_fixture_models() {
        assert_runtime_backends_match_fixture(DeterministicFixtureKind::Separable);
        assert_runtime_backends_match_fixture(DeterministicFixtureKind::Partial);
        assert_runtime_backends_match_fixture(DeterministicFixtureKind::NonSeparable);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_dependence_diagnostics_surface() {
        let expr = (TestAmplitude::new("a", parameter("ar"), parameter("ai")).unwrap()
            + TestAmplitude::new("b", parameter("br"), parameter("bi")).unwrap())
        .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let annotations = evaluator.expression_node_dependence_annotations();
        assert_eq!(annotations.len(), evaluator.expression_ir().node_count());
        assert!(annotations
            .iter()
            .all(|dependence| *dependence == ExpressionDependence::Mixed));
        assert_eq!(
            evaluator.expression_root_dependence(),
            ExpressionDependence::Mixed
        );
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_default_dependence_hint_is_mixed() {
        let expr = ComplexScalar::new("c", parameter("cr"), parameter("ci")).unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert_eq!(
            evaluator.expression_root_dependence(),
            ExpressionDependence::Mixed
        );
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_parameter_only_dependence_hint_propagates() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert_eq!(
            evaluator.expression_root_dependence(),
            ExpressionDependence::ParameterOnly
        );
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_cache_only_dependence_hint_propagates() {
        let expr = CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert_eq!(
            evaluator.expression_root_dependence(),
            ExpressionDependence::CacheOnly
        );
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_dependence_warnings_surface() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            + &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert!(evaluator
            .expression_dependence_warnings()
            .iter()
            .any(|warning| warning.contains("both ParameterOnly and CacheOnly")));
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_normalization_plan_explain_surface() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let explain = evaluator.expression_normalization_plan_explain();
        assert_eq!(explain.root_dependence, ExpressionDependence::Mixed);
        assert_eq!(explain.separable_mul_candidate_nodes.len(), 1);
        assert_eq!(
            explain.cached_separable_nodes,
            explain.separable_mul_candidate_nodes
        );
        assert!(explain.residual_terms.iter().all(|index| {
            !explain
                .separable_mul_candidate_nodes
                .iter()
                .any(|candidate| candidate == index)
        }));
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_normalization_execution_sets_surface() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let sets = evaluator.expression_normalization_execution_sets();
        assert_eq!(sets.cached_parameter_amplitudes, vec![0]);
        assert_eq!(sets.cached_cache_amplitudes, vec![1]);
        assert!(sets.residual_amplitudes.is_empty());
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_normalization_execution_sets_partial_surface() {
        let expr = (ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap())
            + &TestAmplitude::new("m", parameter("mr"), parameter("mi")).unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let sets = evaluator.expression_normalization_execution_sets();
        assert_eq!(sets.cached_parameter_amplitudes, vec![0]);
        assert_eq!(sets.cached_cache_amplitudes, vec![1]);
        assert_eq!(sets.residual_amplitudes, vec![2]);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_precomputed_cached_integrals_at_load() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        let precomputed = evaluator.expression_precomputed_cached_integrals();
        assert_eq!(precomputed.len(), 1);
        let cache_reference = CacheOnlyScalar::new("k_ref")
            .unwrap()
            .load(&dataset)
            .unwrap();
        let cache_values = cache_reference.evaluate_local(&[]);
        let expected_weighted_sum = cache_values
            .iter()
            .zip(dataset.events_local().iter())
            .fold(Complex64::ZERO, |acc, (value, event)| {
                acc + (*value * event.weight())
            });
        assert_relative_eq!(
            precomputed[0].weighted_cache_sum.re,
            expected_weighted_sum.re
        );
        assert_relative_eq!(
            precomputed[0].weighted_cache_sum.im,
            expected_weighted_sum.im
        );
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_precomputed_cached_integrals_empty_when_non_separable() {
        let expr = TestAmplitude::new("m", parameter("mr"), parameter("mi")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert!(evaluator
            .expression_precomputed_cached_integrals()
            .is_empty());
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_precomputed_cached_integrals_recompute_on_activation_change() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert_eq!(evaluator.expression_precomputed_cached_integrals().len(), 1);

        evaluator.isolate_many(&["p"]);
        assert!(evaluator
            .expression_precomputed_cached_integrals()
            .is_empty());

        evaluator.activate_many(&["k"]);
        assert_eq!(evaluator.expression_precomputed_cached_integrals().len(), 1);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_precomputed_cached_integrals_recompute_on_dataset_change() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let mut evaluator = expr.load(&dataset).unwrap();
        drop(dataset);
        let before = evaluator.expression_precomputed_cached_integrals();
        assert_eq!(before.len(), 1);

        Arc::get_mut(&mut evaluator.dataset)
            .expect("evaluator should own dataset Arc in this test")
            .clear_events_local();
        let after = evaluator.expression_precomputed_cached_integrals();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].weighted_cache_sum, Complex64::ZERO);
        assert!(before[0].weighted_cache_sum != after[0].weighted_cache_sum);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_precomputed_cached_integral_gradient_terms_scale_by_cache_integrals() {
        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![
            Arc::new(test_event()),
            Arc::new(test_event()),
        ]));
        let evaluator = expr.load(&dataset).unwrap();
        let cached_integrals = evaluator.expression_precomputed_cached_integrals();
        assert_eq!(cached_integrals.len(), 1);
        let gradient_terms =
            evaluator.expression_precomputed_cached_integral_gradient_terms(&[1.25]);
        assert_eq!(gradient_terms.len(), 1);
        assert_eq!(gradient_terms[0].weighted_gradient.len(), 1);
        assert_relative_eq!(
            gradient_terms[0].weighted_gradient[0].re,
            cached_integrals[0].weighted_cache_sum.re,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            gradient_terms[0].weighted_gradient[0].im,
            cached_integrals[0].weighted_cache_sum.im,
            epsilon = 1e-6
        );
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_precomputed_cached_integral_gradient_terms_empty_when_not_separable() {
        let expr = TestAmplitude::new("m", parameter("mr"), parameter("mi")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();
        assert!(evaluator
            .expression_precomputed_cached_integral_gradient_terms(&[0.1, -0.2])
            .is_empty());
    }

    #[test]
    fn test_evaluate_weighted_gradient_sum_local_matches_eventwise_baseline() {
        let p1 = ParameterOnlyScalar::new("p1", parameter("p1")).unwrap();
        let p2 = ParameterOnlyScalar::new("p2", parameter("p2")).unwrap();
        let c1 = CacheOnlyScalar::new("c1").unwrap();
        let c2 = CacheOnlyScalar::new("c2").unwrap();
        let c3 = CacheOnlyScalar::new("c3").unwrap();
        let m1 = TestAmplitude::new("m1", parameter("m1r"), parameter("m1i")).unwrap();
        let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        #[cfg(feature = "expression-ir")]
        assert_eq!(evaluator.expression_precomputed_cached_integrals().len(), 2);
        let params = vec![0.2, -0.3, 1.1, -0.7];
        let expected = evaluator
            .evaluate_gradient_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(
                DVector::zeros(params.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        let actual = evaluator.evaluate_weighted_gradient_sum_local(&params);
        for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(*actual_item, *expected_item, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_evaluate_weighted_value_sum_local_matches_eventwise_baseline() {
        let p1 = ParameterOnlyScalar::new("p1", parameter("p1")).unwrap();
        let p2 = ParameterOnlyScalar::new("p2", parameter("p2")).unwrap();
        let c1 = CacheOnlyScalar::new("c1").unwrap();
        let c2 = CacheOnlyScalar::new("c2").unwrap();
        let c3 = CacheOnlyScalar::new("c3").unwrap();
        let m1 = TestAmplitude::new("m1", parameter("m1r"), parameter("m1i")).unwrap();
        let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        #[cfg(feature = "expression-ir")]
        assert_eq!(evaluator.expression_precomputed_cached_integrals().len(), 2);
        let params = vec![0.2, -0.3, 1.1, -0.7];
        let expected = evaluator
            .evaluate_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let actual = evaluator.evaluate_weighted_value_sum_local(&params);
        assert_relative_eq!(actual, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_sums_match_hardcoded_reference_values() {
        let p1 = ParameterOnlyScalar::new("p1", parameter("p1")).unwrap();
        let p2 = ParameterOnlyScalar::new("p2", parameter("p2")).unwrap();
        let c1 = CacheOnlyScalar::new("c1").unwrap();
        let c2 = CacheOnlyScalar::new("c2").unwrap();
        let c3 = CacheOnlyScalar::new("c3").unwrap();
        let m1 = TestAmplitude::new("m1", parameter("m1r"), parameter("m1i")).unwrap();
        let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);

        let metadata = Arc::new(DatasetMetadata::default());
        let dataset = Arc::new(Dataset::new_with_metadata(
            vec![
                Arc::new(EventData {
                    p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
                    aux: vec![],
                    weight: 0.5,
                }),
                Arc::new(EventData {
                    p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
                    aux: vec![],
                    weight: -1.25,
                }),
                Arc::new(EventData {
                    p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
                    aux: vec![],
                    weight: 2.0,
                }),
            ],
            metadata,
        ));
        let evaluator = expr.load(&dataset).unwrap();
        let params = vec![0.7, -1.1, 0.9, -0.4];

        let weighted_value_sum = evaluator.evaluate_weighted_value_sum_local(&params);
        assert_relative_eq!(weighted_value_sum, 22.7725, epsilon = 1e-12);

        let weighted_gradient_sum = evaluator.evaluate_weighted_gradient_sum_local(&params);
        let free_parameters = evaluator
            .free_parameters()
            .into_iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(free_parameters, vec!["p1", "p2", "m1r", "m1i"]);
        let expected_gradient = [43.925, 7.25, 28.525, 0.0];
        assert_eq!(weighted_gradient_sum.len(), expected_gradient.len());
        for (actual, expected) in weighted_gradient_sum.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-9);
        }
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_evaluate_weighted_gradient_sum_local_respects_signed_cached_terms() {
        let expr = Expression::one()
            - &(ParameterOnlyScalar::new("p", parameter("p")).unwrap()
                * &CacheOnlyScalar::new("k").unwrap());
        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        assert_eq!(evaluator.expression_precomputed_cached_integrals().len(), 1);
        assert_eq!(
            evaluator.expression_precomputed_cached_integrals()[0].coefficient,
            -1
        );
        let params = vec![0.75];
        let expected = evaluator
            .evaluate_gradient_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(
                DVector::zeros(params.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        let actual = evaluator.evaluate_weighted_gradient_sum_local(&params);
        for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(*actual_item, *expected_item, epsilon = 1e-10);
        }
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_evaluate_weighted_value_sum_local_respects_signed_cached_terms() {
        let expr = Expression::one()
            - &(ParameterOnlyScalar::new("p", parameter("p")).unwrap()
                * &CacheOnlyScalar::new("k").unwrap());
        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        assert_eq!(evaluator.expression_precomputed_cached_integrals().len(), 1);
        assert_eq!(
            evaluator.expression_precomputed_cached_integrals()[0].coefficient,
            -1
        );
        let params = vec![0.75];
        let expected = evaluator
            .evaluate_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let actual = evaluator.evaluate_weighted_value_sum_local(&params);
        assert_relative_eq!(actual, expected, epsilon = 1e-10);
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_diagnostics_follow_activation_changes() {
        let expr = (ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap())
            + &TestAmplitude::new("m", parameter("mr"), parameter("mi")).unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();

        let all_active = evaluator.expression_normalization_plan_explain();
        assert_eq!(all_active.cached_separable_nodes.len(), 1);
        assert_eq!(
            evaluator.expression_root_dependence(),
            ExpressionDependence::Mixed
        );

        evaluator.isolate_many(&["p"]);
        let param_only = evaluator.expression_normalization_plan_explain();
        assert!(param_only.cached_separable_nodes.is_empty());
        assert_eq!(
            evaluator.expression_root_dependence(),
            ExpressionDependence::ParameterOnly
        );

        evaluator.activate_many(&["k", "m"]);
        let restored = evaluator.expression_normalization_plan_explain();
        assert_eq!(restored.cached_separable_nodes.len(), 1);
        assert_eq!(
            evaluator.expression_root_dependence(),
            ExpressionDependence::Mixed
        );
    }

    #[cfg(feature = "expression-ir")]
    #[test]
    fn test_expression_ir_specialization_cache_reuses_prior_mask_specializations() {
        let expr = (ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap())
            + &TestAmplitude::new("m", parameter("mr"), parameter("mi")).unwrap();
        let dataset = Arc::new(Dataset::new(vec![Arc::new(test_event())]));
        let evaluator = expr.load(&dataset).unwrap();

        assert_eq!(evaluator.specialization_cache_len(), 1);
        let all_active_cached_integrals = evaluator.expression_precomputed_cached_integrals();
        let all_active_slot_count = evaluator.lowered_runtime_slot_count();

        evaluator.isolate_many(&["p"]);
        assert_eq!(evaluator.specialization_cache_len(), 2);
        assert!(evaluator
            .expression_precomputed_cached_integrals()
            .is_empty());
        let parameter_only_slot_count = evaluator.lowered_runtime_slot_count();
        assert!(parameter_only_slot_count <= all_active_slot_count);

        evaluator.activate_many(&["k", "m"]);
        assert_eq!(evaluator.specialization_cache_len(), 2);
        assert_eq!(
            evaluator.expression_precomputed_cached_integrals(),
            all_active_cached_integrals
        );
        assert_eq!(
            evaluator.lowered_runtime_slot_count(),
            all_active_slot_count
        );

        evaluator.deactivate_many(&["k"]);
        assert_eq!(evaluator.specialization_cache_len(), 3);
        assert!(evaluator
            .expression_precomputed_cached_integrals()
            .is_empty());

        evaluator.activate_many(&["k"]);
        assert_eq!(evaluator.specialization_cache_len(), 2 + 1);
        assert_eq!(
            evaluator.expression_precomputed_cached_integrals(),
            all_active_cached_integrals
        );
        assert_eq!(
            evaluator.lowered_runtime_slot_count(),
            all_active_slot_count
        );
    }

    #[test]
    fn test_weighted_sums_match_baseline_after_activation_changes() {
        let p1 = ParameterOnlyScalar::new("p1", parameter("p1")).unwrap();
        let p2 = ParameterOnlyScalar::new("p2", parameter("p2")).unwrap();
        let c1 = CacheOnlyScalar::new("c1").unwrap();
        let c2 = CacheOnlyScalar::new("c2").unwrap();
        let c3 = CacheOnlyScalar::new("c3").unwrap();
        let m1 = TestAmplitude::new("m1", parameter("m1r"), parameter("m1i")).unwrap();
        let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let params = vec![0.2, -0.3, 1.1, -0.7];

        let expected_value = evaluator
            .evaluate_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let expected_gradient = evaluator
            .evaluate_gradient_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(
                DVector::zeros(params.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        assert_relative_eq!(
            evaluator.evaluate_weighted_value_sum_local(&params),
            expected_value,
            epsilon = 1e-10
        );
        let actual_gradient = evaluator.evaluate_weighted_gradient_sum_local(&params);
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(*actual_item, *expected_item, epsilon = 1e-10);
        }

        evaluator.isolate_many(&["p1", "c1", "m1", "c3"]);

        let expected_value = evaluator
            .evaluate_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let expected_gradient = evaluator
            .evaluate_gradient_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(
                DVector::zeros(params.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        assert_relative_eq!(
            evaluator.evaluate_weighted_value_sum_local(&params),
            expected_value,
            epsilon = 1e-10
        );
        let actual_gradient = evaluator.evaluate_weighted_gradient_sum_local(&params);
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(*actual_item, *expected_item, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_evaluate_local_does_not_depend_on_dataset_rows() {
        let expr = TestAmplitude::new("test", parameter("real"), parameter("imag"))
            .unwrap()
            .norm_sqr();
        let mut event1 = test_event();
        event1.p4s[0].t = 7.5;
        let mut event2 = test_event();
        event2.p4s[0].t = 8.25;
        let mut event3 = test_event();
        event3.p4s[0].t = 9.0;
        let dataset = Arc::new(Dataset::new_with_metadata(
            vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
            Arc::new(DatasetMetadata::default()),
        ));
        let mut evaluator = expr.load(&dataset).unwrap();
        drop(dataset);
        let expected_len = evaluator.resources.read().caches.len();
        Arc::get_mut(&mut evaluator.dataset)
            .expect("evaluator should own dataset Arc in this test")
            .clear_events_local();
        let cached = evaluator.evaluate_local(&[1.25, -0.75]);
        assert_eq!(cached.len(), expected_len);
    }

    #[test]
    fn test_evaluate_gradient_local_does_not_depend_on_dataset_rows() {
        let expr = TestAmplitude::new("test", parameter("real"), parameter("imag"))
            .unwrap()
            .norm_sqr();
        let mut event1 = test_event();
        event1.p4s[0].t = 7.5;
        let mut event2 = test_event();
        event2.p4s[0].t = 8.25;
        let mut event3 = test_event();
        event3.p4s[0].t = 9.0;
        let dataset = Arc::new(Dataset::new_with_metadata(
            vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
            Arc::new(DatasetMetadata::default()),
        ));
        let mut evaluator = expr.load(&dataset).unwrap();
        drop(dataset);
        let expected_len = evaluator.resources.read().caches.len();
        Arc::get_mut(&mut evaluator.dataset)
            .expect("evaluator should own dataset Arc in this test")
            .clear_events_local();
        let cached = evaluator.evaluate_gradient_local(&[1.25, -0.75]);
        assert_eq!(cached.len(), expected_len);
    }

    #[test]
    fn test_evaluate_with_gradient_local_matches_separate_paths() {
        let expr = TestAmplitude::new("test", parameter("real"), parameter("imag"))
            .unwrap()
            .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![
            Arc::new(test_event()),
            Arc::new(test_event()),
            Arc::new(test_event()),
        ]));
        let evaluator = expr.load(&dataset).unwrap();
        let params = [1.25, -0.75];
        let values = evaluator.evaluate_local(&params);
        let gradients = evaluator.evaluate_gradient_local(&params);
        let fused = evaluator.evaluate_with_gradient_local(&params);
        assert_eq!(fused.len(), values.len());
        assert_eq!(fused.len(), gradients.len());
        for ((value_gradient, value), gradient) in
            fused.iter().zip(values.iter()).zip(gradients.iter())
        {
            let (fused_value, fused_gradient) = value_gradient;
            assert_relative_eq!(fused_value.re, value.re, epsilon = 1e-12);
            assert_relative_eq!(fused_value.im, value.im, epsilon = 1e-12);
            assert_eq!(fused_gradient.len(), gradient.len());
            for (fused_item, item) in fused_gradient.iter().zip(gradient.iter()) {
                assert_relative_eq!(fused_item.re, item.re, epsilon = 1e-12);
                assert_relative_eq!(fused_item.im, item.im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_evaluate_with_gradient_batch_local_matches_separate_paths() {
        let expr = TestAmplitude::new("test", parameter("real"), parameter("imag"))
            .unwrap()
            .norm_sqr();
        let dataset = Arc::new(Dataset::new(vec![
            Arc::new(test_event()),
            Arc::new(test_event()),
            Arc::new(test_event()),
            Arc::new(test_event()),
        ]));
        let evaluator = expr.load(&dataset).unwrap();
        let params = [0.5, -1.25];
        let indices = vec![0, 2, 3];
        let values = evaluator.evaluate_batch_local(&params, &indices);
        let gradients = evaluator.evaluate_gradient_batch_local(&params, &indices);
        let fused = evaluator.evaluate_with_gradient_batch_local(&params, &indices);
        assert_eq!(fused.len(), values.len());
        assert_eq!(fused.len(), gradients.len());
        for ((value_gradient, value), gradient) in
            fused.iter().zip(values.iter()).zip(gradients.iter())
        {
            let (fused_value, fused_gradient) = value_gradient;
            assert_relative_eq!(fused_value.re, value.re, epsilon = 1e-12);
            assert_relative_eq!(fused_value.im, value.im, epsilon = 1e-12);
            assert_eq!(fused_gradient.len(), gradient.len());
            for (fused_item, item) in fused_gradient.iter().zip(gradient.iter()) {
                assert_relative_eq!(fused_item.re, item.re, epsilon = 1e-12);
                assert_relative_eq!(fused_item.im, item.im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_precompute_all_columnar_populates_cache() {
        let mut event1 = test_event();
        event1.p4s[0].t = 7.5;
        let mut event2 = test_event();
        event2.p4s[0].t = 8.25;
        let mut event3 = test_event();
        event3.p4s[0].t = 9.0;
        let dataset = Dataset::new_with_metadata(
            vec![Arc::new(event1), Arc::new(event2), Arc::new(event3)],
            Arc::new(DatasetMetadata::default()),
        );
        let mut amplitude = TestAmplitude {
            name: "test".to_string(),
            re: parameter("real"),
            pid_re: ParameterID::default(),
            im: parameter("imag"),
            pid_im: ParameterID::default(),
            beam_energy: Default::default(),
        };
        let mut resources = Resources::default();
        amplitude
            .register(&mut resources)
            .expect("test amplitude should register");
        resources.reserve_cache(dataset.n_events());
        amplitude.precompute_all(&dataset, &mut resources);
        for cache in &resources.caches {
            assert!(cache.get_scalar(amplitude.beam_energy) > 0.0);
        }
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn test_load_reserves_local_cache_size_in_mpi() {
        use crate::mpi::{finalize_mpi, get_world, use_mpi};

        use_mpi(true);
        assert!(get_world().is_some(), "MPI world should be initialized");

        let expr = ComplexScalar::new(
            "constant",
            constant("const_re", 2.0),
            constant("const_im", 3.0),
        )
        .expect("constant amplitude should construct");
        let events = vec![
            Arc::new(test_event()),
            Arc::new(test_event()),
            Arc::new(test_event()),
            Arc::new(test_event()),
        ];
        let dataset = Arc::new(Dataset::new_with_metadata(
            events,
            Arc::new(DatasetMetadata::default()),
        ));
        let evaluator = expr.load(&dataset).expect("evaluator should load");
        let local_events = dataset.n_events_local();
        let cache_len = evaluator.resources.read().caches.len();

        assert_eq!(
            cache_len, local_events,
            "cache length must match local event count under MPI"
        );
        finalize_mpi();
    }

    #[cfg(all(feature = "mpi", feature = "expression-ir"))]
    #[mpi_test(np = [2])]
    fn test_expression_ir_cached_integrals_are_rank_local_in_mpi() {
        use crate::mpi::{finalize_mpi, get_world, use_mpi};
        use mpi::{collective::SystemOperation, topology::Communicator, traits::*};

        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");

        let expr = ParameterOnlyScalar::new("p", parameter("p")).unwrap()
            * &CacheOnlyScalar::new("k").unwrap();
        let events = vec![
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
                aux: vec![],
                weight: 0.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
                aux: vec![],
                weight: 1.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
                aux: vec![],
                weight: 1.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 4.0)],
                aux: vec![],
                weight: 2.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
                aux: vec![],
                weight: 2.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 6.0)],
                aux: vec![],
                weight: 3.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 7.0)],
                aux: vec![],
                weight: 3.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 8.0)],
                aux: vec![],
                weight: 4.0,
            }),
        ];
        let dataset = Arc::new(Dataset::new_with_metadata(
            events,
            Arc::new(DatasetMetadata::default()),
        ));
        let evaluator = expr.load(&dataset).expect("evaluator should load");
        let cached_integrals = evaluator.expression_precomputed_cached_integrals();
        assert_eq!(cached_integrals.len(), 1);

        let local_expected = dataset.events_local().iter().fold(0.0, |acc, event| {
            acc + event.weight() * event.data().p4s[0].e()
        });
        let cached_local = cached_integrals[0].weighted_cache_sum;
        assert_relative_eq!(cached_local.re, local_expected, epsilon = 1e-12);
        assert_relative_eq!(cached_local.im, 0.0, epsilon = 1e-12);

        let weighted_value_sum = evaluator.evaluate_weighted_value_sum_local(&[2.0]);
        assert_relative_eq!(weighted_value_sum, 2.0 * local_expected, epsilon = 1e-10);

        let mut global_expected = 0.0;
        world.all_reduce_into(
            &local_expected,
            &mut global_expected,
            SystemOperation::sum(),
        );
        if world.size() > 1 {
            assert!(
                (cached_local.re - global_expected).abs() > 1e-12,
                "cached integral should remain rank-local before MPI reduction"
            );
        }
        finalize_mpi();
    }

    #[cfg(all(feature = "mpi", feature = "expression-ir"))]
    #[mpi_test(np = [2])]
    fn test_expression_ir_weighted_sum_mpi_matches_global_eventwise_baseline() {
        use crate::mpi::{finalize_mpi, get_world, use_mpi};
        use mpi::{collective::SystemOperation, traits::*};

        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");

        let p1 = ParameterOnlyScalar::new("p1", parameter("p1")).unwrap();
        let p2 = ParameterOnlyScalar::new("p2", parameter("p2")).unwrap();
        let c1 = CacheOnlyScalar::new("c1").unwrap();
        let c2 = CacheOnlyScalar::new("c2").unwrap();
        let c3 = CacheOnlyScalar::new("c3").unwrap();
        let m1 = TestAmplitude::new("m1", parameter("m1r"), parameter("m1i")).unwrap();
        let expr = (&p1 * &c1) + &(&p2 * &c2) + &(&(&m1 * &p1) * &c3);
        let events = vec![
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
                aux: vec![],
                weight: 0.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 2.0)],
                aux: vec![],
                weight: -1.25,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 3.0)],
                aux: vec![],
                weight: 0.75,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 4.0)],
                aux: vec![],
                weight: 1.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 5.0)],
                aux: vec![],
                weight: 2.25,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 6.0)],
                aux: vec![],
                weight: -0.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 7.0)],
                aux: vec![],
                weight: 3.5,
            }),
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 0.0, 8.0)],
                aux: vec![],
                weight: 1.25,
            }),
        ];
        let dataset = Arc::new(Dataset::new_with_metadata(
            events,
            Arc::new(DatasetMetadata::default()),
        ));
        let evaluator = expr.load(&dataset).expect("evaluator should load");
        let params = vec![0.2, -0.3, 1.1, -0.7];

        let local_expected_value = evaluator
            .evaluate_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(0.0, |accum, (value, event)| {
                accum + event.weight() * value.re
            });
        let mut global_expected_value = 0.0;
        world.all_reduce_into(
            &local_expected_value,
            &mut global_expected_value,
            SystemOperation::sum(),
        );
        let mpi_value = evaluator.evaluate_weighted_value_sum_mpi(&params, &world);
        assert_relative_eq!(mpi_value, global_expected_value, epsilon = 1e-10);

        let local_expected_gradient = evaluator
            .evaluate_gradient_local(&params)
            .iter()
            .zip(dataset.events_local().iter())
            .fold(
                DVector::zeros(params.len()),
                |mut accum, (gradient, event)| {
                    accum += gradient.map(|value| value.re).scale(event.weight());
                    accum
                },
            );
        let mut global_expected_gradient = vec![0.0; local_expected_gradient.len()];
        world.all_reduce_into(
            local_expected_gradient.as_slice(),
            &mut global_expected_gradient,
            SystemOperation::sum(),
        );
        let mpi_gradient = evaluator.evaluate_weighted_gradient_sum_mpi(&params, &world);
        for (actual, expected) in mpi_gradient.iter().zip(global_expected_gradient.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-10);
        }

        finalize_mpi();
    }

    #[test]
    fn test_evaluate_local_succeeds_for_constant_amplitude() {
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
        let values = evaluator.evaluate_local(&[]);
        assert_eq!(values.len(), 1);
        let gradients = evaluator.evaluate_gradient_local(&[]);
        assert_eq!(gradients.len(), 1);
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
