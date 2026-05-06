use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use auto_ops::*;
use dyn_clone::DynClone;
use nalgebra::DVector;
use num::complex::Complex64;
use parking_lot::RwLock;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{ir, lowered};

static AMPLITUDE_INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_amplitude_id() -> u64 {
    AMPLITUDE_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed)
}
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
impl From<ir::DependenceClass> for ExpressionDependence {
    fn from(value: ir::DependenceClass) -> Self {
        match value {
            ir::DependenceClass::ParameterOnly => Self::ParameterOnly,
            ir::DependenceClass::CacheOnly => Self::CacheOnly,
            ir::DependenceClass::Mixed => Self::Mixed,
        }
    }
}
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CachedIntegralCacheKey {
    active_mask: Vec<bool>,
    n_events_local: usize,
    weights_local_len: usize,
    weighted_sum_bits: u64,
    weights_ptr: usize,
}
#[derive(Clone, Debug)]
struct CachedIntegralCacheState {
    key: CachedIntegralCacheKey,
    expression_ir: ir::ExpressionIR,
    values: Vec<PrecomputedCachedIntegral>,
    execution_sets: ir::NormalizationExecutionSets,
}
#[derive(Clone, Debug)]
struct LoweredArtifactCacheState {
    parameter_node_indices: Vec<usize>,
    mul_node_indices: Vec<usize>,
    lowered_parameter_factors: Vec<Option<lowered::LoweredFactorRuntime>>,
    residual_runtime: Option<lowered::LoweredExpressionRuntime>,
    lowered_runtime: lowered::LoweredExpressionRuntime,
}
#[derive(Clone)]
struct ExpressionSpecializationState {
    cached_integrals: Arc<CachedIntegralCacheState>,
    lowered_artifacts: Arc<LoweredArtifactCacheState>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Debug/benchmark counters for active-mask specialization reuse under `expression-ir`.
pub struct ExpressionSpecializationMetrics {
    /// Number of specialization cache hits served without recompilation.
    pub cache_hits: usize,
    /// Number of specialization cache misses that required a fresh compile/lower pass.
    pub cache_misses: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Staged compile/lowering metrics for expression-IR construction and specialization refreshes.
pub struct ExpressionCompileMetrics {
    /// Nanoseconds spent compiling the semantic expression tree into IR during initial load.
    pub initial_ir_compile_nanos: u64,
    /// Nanoseconds spent precomputing cached-integral planning artifacts during initial load.
    pub initial_cached_integrals_nanos: u64,
    /// Nanoseconds spent lowering IR-derived runtimes during initial load.
    pub initial_lowering_nanos: u64,
    /// Number of specialization cache hits restored without recompilation.
    pub specialization_cache_hits: usize,
    /// Number of specialization cache misses that required recompilation.
    pub specialization_cache_misses: usize,
    /// Accumulated nanoseconds spent compiling active-mask-specialized IR after initial load.
    pub specialization_ir_compile_nanos: u64,
    /// Accumulated nanoseconds spent recomputing cached-integral planning artifacts after load.
    pub specialization_cached_integrals_nanos: u64,
    /// Accumulated nanoseconds spent lowering specialized runtimes after load.
    pub specialization_lowering_nanos: u64,
    /// Number of specialization rebuilds that reused cached lowered artifacts.
    pub specialization_lowering_cache_hits: usize,
    /// Number of specialization rebuilds that had to lower fresh artifacts.
    pub specialization_lowering_cache_misses: usize,
    /// Accumulated nanoseconds spent restoring specializations from cache.
    pub specialization_cache_restore_nanos: u64,
}
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
impl From<ir::NormalizationExecutionSets> for NormalizationExecutionSetsExplain {
    fn from(value: ir::NormalizationExecutionSets) -> Self {
        Self {
            cached_parameter_amplitudes: value.cached_parameter_amplitudes,
            cached_cache_amplitudes: value.cached_cache_amplitudes,
            residual_amplitudes: value.residual_amplitudes,
        }
    }
}
impl From<ExpressionDependence> for ir::DependenceClass {
    fn from(value: ExpressionDependence) -> Self {
        match value {
            ExpressionDependence::ParameterOnly => Self::ParameterOnly,
            ExpressionDependence::CacheOnly => Self::CacheOnly,
            ExpressionDependence::Mixed => Self::Mixed,
        }
    }
}

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;
#[cfg(feature = "execution-context-prototype")]
use crate::ExecutionContext;
#[cfg(all(feature = "execution-context-prototype", feature = "rayon"))]
use crate::ThreadPolicy;
use crate::{
    data::{Dataset, DatasetMetadata, Event},
    parameters::ParameterMap,
    resources::{Cache, Parameters, Resources},
    LadduError, LadduResult,
};

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
    /// [`register`](Amplitude::register) is invoked once when an amplitude is first converted into
    /// an [`Expression`]. Use it to allocate parameter/cache state within [`Resources`] without assuming
    /// any dataset context.
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID>;
    /// Optional semantic identity key for deduplicating equivalent amplitude computations.
    ///
    /// Return `Some` only when two independently constructed instances with equal keys are
    /// interchangeable after registration, binding, precomputation, value evaluation, and gradient
    /// evaluation. The key should include the concrete amplitude type and all user-facing
    /// configuration, but must ignore registration-assigned IDs like [`ParameterID`]s and cache IDs.
    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        None
    }
    /// Bind this [`Amplitude`] to a concrete [`Dataset`] by using the provided metadata to wire up
    /// [`Variable`](crate::variables::Variable)s or other dataset-specific state. This will
    /// be invoked when a [`Expression`] is loaded with data, after [`register`](Amplitude::register)
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
    /// Optional hint that this amplitude always evaluates to a purely real complex value.
    ///
    /// This must be conservative. Returning `true` allows `expression-ir` to erase
    /// redundant `imag`, `real`, and `conj` work under the assumption that the
    /// amplitude output always has zero imaginary component.
    fn real_valued_hint(&self) -> bool {
        false
    }
    /// This method can be used to do some critical calculations ahead of time and
    /// store them in a [`Cache`]. These values can only depend on event data,
    /// not on any free parameters in the fit. This method is opt-in since it is
    /// not required to make a functioning [`Amplitude`].
    #[allow(unused_variables)]
    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {}

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
        dataset.for_each_event_local(|event_index, event| {
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
        let x = parameters.values().to_owned();
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
            let f_plus = self.compute(&parameters.with_values(x_plus), cache);
            let f_minus = self.compute(&parameters.with_values(x_minus), cache);
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

/// Normalized activation tags for an amplitude use-site.
///
/// Empty strings are ignored, and duplicate tags are removed while preserving the first
/// occurrence. An empty tag set is a normal untagged amplitude use-site; it participates in
/// expression evaluation but cannot be selected by tag activation APIs.
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tags(Vec<String>);

impl Tags {
    /// Construct normalized tags from string-like inputs.
    pub fn new(tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut normalized = Vec::new();
        for tag in tags {
            let tag = tag.into();
            if tag.is_empty() || normalized.contains(&tag) {
                continue;
            }
            normalized.push(tag);
        }
        Self(normalized)
    }

    /// Construct an empty tag set.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Borrow the normalized tags.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// Return true if this tag set has no selectable tags.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn display_label(&self) -> String {
        if self.is_empty() {
            "<untagged>".to_string()
        } else {
            self.0.join(",")
        }
    }
}

/// Convert user-facing tag inputs into [`Tags`].
pub trait IntoTags {
    /// Convert into normalized tags.
    fn into_tags(self) -> Tags;
}

impl IntoTags for Tags {
    fn into_tags(self) -> Tags {
        self
    }
}

impl IntoTags for () {
    fn into_tags(self) -> Tags {
        Tags::empty()
    }
}

impl IntoTags for &str {
    fn into_tags(self) -> Tags {
        Tags::new([self])
    }
}

impl IntoTags for String {
    fn into_tags(self) -> Tags {
        Tags::new([self])
    }
}

impl IntoTags for &String {
    fn into_tags(self) -> Tags {
        Tags::new([self.clone()])
    }
}

impl<T: Into<String>> IntoTags for Vec<T> {
    fn into_tags(self) -> Tags {
        Tags::new(self)
    }
}

impl<T: Clone + Into<String>> IntoTags for &[T] {
    fn into_tags(self) -> Tags {
        Tags::new(self.iter().cloned())
    }
}

impl<T: Into<String>, const N: usize> IntoTags for [T; N] {
    fn into_tags(self) -> Tags {
        Tags::new(self)
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

/// A tag set which refers to a registered [`Amplitude`]. This is the base object which can be used to
/// build [`Expression`]s and should be obtained from the [`Resources::register_amplitude`] method.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct AmplitudeID(pub(crate) Tags, pub(crate) usize);

impl Display for AmplitudeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(id={})", self.0.display_label(), self.1)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AmplitudeUseSite {
    amplitude_index: usize,
    tags: Tags,
}

/// A single named field in an [`AmplitudeSemanticKey`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmplitudeSemanticField {
    name: String,
    value: String,
}

impl AmplitudeSemanticField {
    /// Construct a semantic key field.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A semantic identity key used to opt into deduplicating equivalent amplitude computations.
///
/// The key must include enough type/configuration information to prove that two independently
/// constructed amplitudes can safely share one registered computation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmplitudeSemanticKey {
    kind: String,
    fields: Vec<AmplitudeSemanticField>,
}

impl AmplitudeSemanticKey {
    /// Construct a semantic key for the given amplitude kind.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            fields: Vec::new(),
        }
    }

    /// Add a named field to this semantic key.
    pub fn with_field(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.push(AmplitudeSemanticField::new(name, value));
        self
    }
}

/// A holder struct that owns both an expression tree and the registered amplitudes.
#[allow(missing_docs)]
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct Expression {
    registry: ExpressionRegistry,
    tree: ExpressionNode,
}

#[derive(Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
#[derive(Default)]
pub struct ExpressionRegistry {
    amplitudes: Vec<Box<dyn Amplitude>>,
    amplitude_names: Vec<String>,
    amplitude_ids: Vec<u64>,
    amplitude_use_sites: Vec<AmplitudeUseSite>,
    amplitude_use_site_ids: Vec<u64>,
    resources: Resources,
}

impl ExpressionRegistry {
    fn singleton(mut amplitude: Box<dyn Amplitude>) -> LadduResult<Self> {
        let mut resources = Resources::default();
        let aid = amplitude.register(&mut resources)?;
        let compute_id = next_amplitude_id();
        let use_site_id = next_amplitude_id();
        resources.configure_amplitude_tags(std::slice::from_ref(&aid.0));
        Ok(Self {
            amplitudes: vec![amplitude],
            amplitude_names: vec![aid.0.display_label()],
            amplitude_ids: vec![compute_id],
            amplitude_use_sites: vec![AmplitudeUseSite {
                amplitude_index: 0,
                tags: aid.0,
            }],
            amplitude_use_site_ids: vec![use_site_id],
            resources,
        })
    }

    fn merge(&self, other: &Self) -> LadduResult<(Self, Vec<usize>, Vec<usize>)> {
        let mut resources = Resources::default();
        let mut amplitudes = Vec::new();
        let mut amplitude_ids = Vec::new();
        let mut compute_id_to_index = HashMap::new();
        let mut semantic_key_to_index = HashMap::new();

        let mut left_compute_map = Vec::with_capacity(self.amplitudes.len());
        for (amp_index, (amp, amp_id)) in
            self.amplitudes.iter().zip(&self.amplitude_ids).enumerate()
        {
            let semantic_key = amp.semantic_key();
            let mut cloned_amp = dyn_clone::clone_box(&**amp);
            let aid = cloned_amp.register(&mut resources)?;
            if let Some(key) = semantic_key.clone() {
                semantic_key_to_index.insert(key, aid.1);
            }
            amplitudes.push(cloned_amp);
            amplitude_ids.push(*amp_id);
            compute_id_to_index.insert(*amp_id, aid.1);
            left_compute_map.push(aid.1);
            debug_assert_eq!(amp_index, aid.1);
        }

        let mut right_compute_map = Vec::with_capacity(other.amplitudes.len());
        for (amp, amp_id) in other.amplitudes.iter().zip(&other.amplitude_ids) {
            if let Some(existing) = compute_id_to_index.get(amp_id) {
                right_compute_map.push(*existing);
                continue;
            }
            let incoming_semantic_key = amp.semantic_key();
            if let Some(existing) = incoming_semantic_key
                .as_ref()
                .and_then(|key| semantic_key_to_index.get(key))
            {
                right_compute_map.push(*existing);
                continue;
            }
            let mut cloned_amp = dyn_clone::clone_box(&**amp);
            let aid = cloned_amp.register(&mut resources)?;
            if let Some(key) = incoming_semantic_key.clone() {
                semantic_key_to_index.insert(key, aid.1);
            }
            amplitudes.push(cloned_amp);
            amplitude_ids.push(*amp_id);
            compute_id_to_index.insert(*amp_id, aid.1);
            right_compute_map.push(aid.1);
        }

        let mut amplitude_use_sites = Vec::new();
        let mut amplitude_use_site_ids = Vec::new();
        let mut amplitude_names = Vec::new();
        let mut use_site_id_to_index = HashMap::new();

        let mut left_map = Vec::with_capacity(self.amplitude_use_sites.len());
        for (use_site, use_site_id) in self
            .amplitude_use_sites
            .iter()
            .zip(&self.amplitude_use_site_ids)
        {
            let mapped_index = left_compute_map[use_site.amplitude_index];
            let new_index = amplitude_use_sites.len();
            left_map.push(new_index);
            use_site_id_to_index.insert(*use_site_id, new_index);
            amplitude_use_site_ids.push(*use_site_id);
            amplitude_names.push(use_site.tags.display_label());
            amplitude_use_sites.push(AmplitudeUseSite {
                amplitude_index: mapped_index,
                tags: use_site.tags.clone(),
            });
        }

        let mut right_map = Vec::with_capacity(other.amplitude_use_sites.len());
        for (use_site, use_site_id) in other
            .amplitude_use_sites
            .iter()
            .zip(&other.amplitude_use_site_ids)
        {
            if let Some(existing) = use_site_id_to_index.get(use_site_id) {
                right_map.push(*existing);
                continue;
            }
            let mapped_index = right_compute_map[use_site.amplitude_index];
            let new_index = amplitude_use_sites.len();
            right_map.push(new_index);
            use_site_id_to_index.insert(*use_site_id, new_index);
            amplitude_use_site_ids.push(*use_site_id);
            amplitude_names.push(use_site.tags.display_label());
            amplitude_use_sites.push(AmplitudeUseSite {
                amplitude_index: mapped_index,
                tags: use_site.tags.clone(),
            });
        }
        let use_site_tags = amplitude_use_sites
            .iter()
            .map(|use_site| use_site.tags.clone())
            .collect::<Vec<_>>();
        resources.configure_amplitude_tags(&use_site_tags);

        Ok((
            Self {
                amplitudes,
                amplitude_names,
                amplitude_ids,
                amplitude_use_sites,
                amplitude_use_site_ids,
                resources,
            },
            left_map,
            right_map,
        ))
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
    /// A real-valued constant.
    Constant(f64),
    /// A complex-valued constant.
    ComplexConstant(Complex64),
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
    Sqrt(Box<ExpressionNode>),
    Pow(Box<ExpressionNode>, Box<ExpressionNode>),
    PowI(Box<ExpressionNode>, i32),
    PowF(Box<ExpressionNode>, f64),
    Exp(Box<ExpressionNode>),
    Sin(Box<ExpressionNode>),
    Cos(Box<ExpressionNode>),
    Log(Box<ExpressionNode>),
    Cis(Box<ExpressionNode>),
}

#[derive(Clone, Debug)]
/// Standalone bytecode executor compiled directly from the semantic expression tree.
///
/// This is retained for direct tree-level helpers on [`ExpressionNode`] and debugging of the
/// unfactored semantic expression shape. It is intentionally distinct from the lowered runtime:
/// current lowering carries slot reuse, peephole rewrites, root-specific lowering, and
/// specialized normalization helpers that would be awkward to force back into this form.
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
    LoadConstant {
        dst: usize,
        value: f64,
    },
    LoadComplexConstant {
        dst: usize,
        value: Complex64,
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
    Sqrt {
        dst: usize,
        input: usize,
    },
    Pow {
        dst: usize,
        value: usize,
        power: usize,
    },
    PowI {
        dst: usize,
        input: usize,
        power: i32,
    },
    PowF {
        dst: usize,
        input: usize,
        power: f64,
    },
    Exp {
        dst: usize,
        input: usize,
    },
    Sin {
        dst: usize,
        input: usize,
    },
    Cos {
        dst: usize,
        input: usize,
    },
    Log {
        dst: usize,
        input: usize,
    },
    Cis {
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
            ExpressionNode::Constant(value) => {
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::LoadConstant { dst, value: *value });
                dst
            }
            ExpressionNode::ComplexConstant(value) => {
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::LoadComplexConstant { dst, value: *value });
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
            ExpressionNode::Sqrt(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Sqrt { dst, input });
                dst
            }
            ExpressionNode::Pow(a, b) => {
                let value = self.compile(a);
                let power = self.compile(b);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Pow { dst, value, power });
                dst
            }
            ExpressionNode::PowI(a, power) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::PowI {
                    dst,
                    input,
                    power: *power,
                });
                dst
            }
            ExpressionNode::PowF(a, power) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::PowF {
                    dst,
                    input,
                    power: *power,
                });
                dst
            }
            ExpressionNode::Exp(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Exp { dst, input });
                dst
            }
            ExpressionNode::Sin(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Sin { dst, input });
                dst
            }
            ExpressionNode::Cos(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Cos { dst, input });
                dst
            }
            ExpressionNode::Log(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Log { dst, input });
                dst
            }
            ExpressionNode::Cis(a) => {
                let input = self.compile(a);
                let dst = self.alloc_slot();
                self.emit(ExpressionOp::Cis { dst, input });
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

    fn fill_values(&self, amplitude_values: &[Complex64], slots: &mut [Complex64]) {
        debug_assert!(slots.len() >= self.slot_count);
        for op in &self.ops {
            match *op {
                ExpressionOp::LoadZero { dst } => slots[dst] = Complex64::ZERO,
                ExpressionOp::LoadOne { dst } => slots[dst] = Complex64::ONE,
                ExpressionOp::LoadConstant { dst, value } => slots[dst] = Complex64::from(value),
                ExpressionOp::LoadComplexConstant { dst, value } => slots[dst] = value,
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
                ExpressionOp::Sqrt { dst, input } => {
                    slots[dst] = slots[input].sqrt();
                }
                ExpressionOp::Pow { dst, value, power } => {
                    slots[dst] = slots[value].powc(slots[power]);
                }
                ExpressionOp::PowI { dst, input, power } => {
                    slots[dst] = slots[input].powi(power);
                }
                ExpressionOp::PowF { dst, input, power } => {
                    slots[dst] = slots[input].powc(Complex64::new(power, 0.0));
                }
                ExpressionOp::Exp { dst, input } => {
                    slots[dst] = slots[input].exp();
                }
                ExpressionOp::Sin { dst, input } => {
                    slots[dst] = slots[input].sin();
                }
                ExpressionOp::Cos { dst, input } => {
                    slots[dst] = slots[input].cos();
                }
                ExpressionOp::Log { dst, input } => {
                    slots[dst] = slots[input].ln();
                }
                ExpressionOp::Cis { dst, input } => {
                    slots[dst] = (Complex64::new(0.0, 1.0) * slots[input]).exp();
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
                ExpressionOp::LoadZero { dst }
                | ExpressionOp::LoadOne { dst }
                | ExpressionOp::LoadConstant { dst, .. }
                | ExpressionOp::LoadComplexConstant { dst, .. } => {
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
                ExpressionOp::Sqrt { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let factor = Complex64::new(0.5, 0.0) / values[input].sqrt();
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * factor;
                    }
                }
                ExpressionOp::Pow { dst, value, power } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let base = values[value];
                    let exponent = values[power];
                    let output = values[dst];
                    for ((dst_item, value_item), power_item) in dst_grad
                        .iter_mut()
                        .zip(before_dst[value].iter())
                        .zip(before_dst[power].iter())
                    {
                        *dst_item =
                            output * (*power_item * base.ln() + exponent * *value_item / base);
                    }
                }
                ExpressionOp::PowI { dst, input, power } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let factor = match power {
                        0 => Complex64::ZERO,
                        1 => Complex64::ONE,
                        _ => {
                            let base = values[input];
                            let multiplier = Complex64::new(power as f64, 0.0);
                            if let Some(derivative_power) = power.checked_sub(1) {
                                multiplier * base.powi(derivative_power)
                            } else {
                                multiplier * base.powi(power) / base
                            }
                        }
                    };
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * factor;
                    }
                }
                ExpressionOp::PowF { dst, input, power } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let factor = if power == 0.0 {
                        Complex64::ZERO
                    } else {
                        Complex64::new(power, 0.0)
                            * values[input].powc(Complex64::new(power - 1.0, 0.0))
                    };
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * factor;
                    }
                }
                ExpressionOp::Exp { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let output = values[dst];
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * output;
                    }
                }
                ExpressionOp::Sin { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let factor = values[input].cos();
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * factor;
                    }
                }
                ExpressionOp::Cos { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let factor = -values[input].sin();
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * factor;
                    }
                }
                ExpressionOp::Log { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let factor = Complex64::ONE / values[input];
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * factor;
                    }
                }
                ExpressionOp::Cis { dst, input } => {
                    let (before_dst, dst_grad) = borrow_dst(gradients, dst);
                    let factor = Complex64::new(0.0, 1.0) * values[dst];
                    for (dst_item, input_item) in dst_grad.iter_mut().zip(before_dst[input].iter())
                    {
                        *dst_item = *input_item * factor;
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
            Self::Constant(v) => Self::Constant(*v),
            Self::ComplexConstant(v) => Self::ComplexConstant(*v),
            Self::Sqrt(a) => Self::Sqrt(Box::new(a.remap(mapping))),
            Self::Pow(a, b) => Self::Pow(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::PowI(a, power) => Self::PowI(Box::new(a.remap(mapping)), *power),
            Self::PowF(a, power) => Self::PowF(Box::new(a.remap(mapping)), *power),
            Self::Exp(a) => Self::Exp(Box::new(a.remap(mapping))),
            Self::Sin(a) => Self::Sin(Box::new(a.remap(mapping))),
            Self::Cos(a) => Self::Cos(Box::new(a.remap(mapping))),
            Self::Log(a) => Self::Log(Box::new(a.remap(mapping))),
            Self::Cis(a) => Self::Cis(Box::new(a.remap(mapping))),
        }
    }

    fn program(&self) -> ExpressionProgram {
        ExpressionProgram::from_node(self)
    }

    /// Evaluate an [`ExpressionNode`] by compiling it to bytecode on the fly.
    pub fn evaluate(&self, amplitude_values: &[Complex64]) -> Complex64 {
        self.program().evaluate(amplitude_values)
    }

    /// Evaluate the gradient of an [`ExpressionNode`].
    pub fn evaluate_gradient(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        self.program()
            .evaluate_gradient(amplitude_values, gradient_values)
    }
}

impl From<f64> for Expression {
    fn from(value: f64) -> Self {
        if value == 0.0 {
            Self {
                registry: ExpressionRegistry::default(),
                tree: ExpressionNode::Zero,
            }
        } else if value == 1.0 {
            Self {
                registry: ExpressionRegistry::default(),
                tree: ExpressionNode::One,
            }
        } else {
            Self {
                registry: ExpressionRegistry::default(),
                tree: ExpressionNode::Constant(value),
            }
        }
    }
}
impl From<&f64> for Expression {
    fn from(value: &f64) -> Self {
        (*value).into()
    }
}
impl From<Complex64> for Expression {
    fn from(value: Complex64) -> Self {
        if value == Complex64::ZERO {
            Self {
                registry: ExpressionRegistry::default(),
                tree: ExpressionNode::Zero,
            }
        } else if value == Complex64::ONE {
            Self {
                registry: ExpressionRegistry::default(),
                tree: ExpressionNode::One,
            }
        } else {
            Self {
                registry: ExpressionRegistry::default(),
                tree: ExpressionNode::ComplexConstant(value),
            }
        }
    }
}
impl From<&Complex64> for Expression {
    fn from(value: &Complex64) -> Self {
        (*value).into()
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

    /// Get the parameters used by this expression.
    pub fn parameters(&self) -> ParameterMap {
        self.registry.resources.parameters()
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

    /// Returns a tree-like diagnostic snapshot of this expression's compiled form.
    ///
    /// This compiles the expression on each call with every registered amplitude active. Use
    /// [`Evaluator::compiled_expression`] when you need the compiled form for a loaded evaluator's
    /// current active-amplitude mask.
    pub fn compiled_expression(&self) -> CompiledExpression {
        let active_amplitudes = vec![true; self.registry.amplitude_use_sites.len()];
        let amplitude_dependencies = self
            .registry
            .amplitude_use_sites
            .iter()
            .map(|use_site| {
                ir::DependenceClass::from(
                    self.registry.amplitudes[use_site.amplitude_index].dependence_hint(),
                )
            })
            .collect::<Vec<_>>();
        let amplitude_realness = self
            .registry
            .amplitude_use_sites
            .iter()
            .map(|use_site| self.registry.amplitudes[use_site.amplitude_index].real_valued_hint())
            .collect::<Vec<_>>();
        let expression_ir = ir::compile_expression_ir_with_real_hints(
            &self.tree,
            &active_amplitudes,
            &amplitude_dependencies,
            &amplitude_realness,
        );
        CompiledExpression::from_ir(&expression_ir, &self.registry.amplitude_names)
    }

    /// Fix a parameter used by this expression's evaluator resources.
    pub fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.registry.resources.fix_parameter(name, value)
    }

    /// Mark a parameter used by this expression's evaluator resources as free.
    pub fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.registry.resources.free_parameter(name)
    }

    /// Return a new [`Expression`] with a single parameter renamed.
    pub fn rename_parameter(&mut self, old: &str, new: &str) -> LadduResult<()> {
        self.registry.resources.rename_parameter(old, new)
    }

    /// Return a new [`Expression`] with several parameters renamed.
    pub fn rename_parameters(&mut self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.registry.resources.rename_parameters(mapping)
    }

    /// Load an [`Expression`] against a dataset, binding amplitudes and reserving caches.
    pub fn load(&self, dataset: &Arc<Dataset>) -> LadduResult<Evaluator> {
        let mut resources = self.registry.resources.clone();
        let metadata = dataset.metadata();
        resources.reserve_cache(dataset.n_events_local());
        resources.refresh_active_indices();
        let parameter_map = resources.parameter_map.clone();
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
        let ir_compile_start = Instant::now();
        let expression_ir = {
            let mut active_amplitudes = vec![false; self.registry.amplitude_use_sites.len()];
            for &index in resources.active_indices() {
                active_amplitudes[index] = true;
            }
            let amplitude_dependencies = self
                .registry
                .amplitude_use_sites
                .iter()
                .map(|use_site| {
                    ir::DependenceClass::from(
                        amplitudes[use_site.amplitude_index].dependence_hint(),
                    )
                })
                .collect::<Vec<_>>();
            let amplitude_realness = self
                .registry
                .amplitude_use_sites
                .iter()
                .map(|use_site| amplitudes[use_site.amplitude_index].real_valued_hint())
                .collect::<Vec<_>>();
            ir::compile_expression_ir_with_real_hints(
                &self.tree,
                &active_amplitudes,
                &amplitude_dependencies,
                &amplitude_realness,
            )
        };
        let initial_ir_compile_nanos = ir_compile_start.elapsed().as_nanos() as u64;
        let cached_integrals_start = Instant::now();
        let cached_integrals = Evaluator::precompute_cached_integrals_at_load(
            &expression_ir,
            &amplitudes,
            &self.registry.amplitude_use_sites,
            &resources,
            dataset,
            parameter_map.free().len(),
        )?;
        let initial_cached_integrals_nanos = cached_integrals_start.elapsed().as_nanos() as u64;
        let lowering_start = Instant::now();
        let lowered_artifacts = Arc::new(Evaluator::lower_expression_runtime_artifacts(
            &expression_ir,
            &cached_integrals,
        )?);
        let initial_lowering_nanos = lowering_start.elapsed().as_nanos() as u64;
        let execution_sets = expression_ir.normalization_execution_sets().clone();
        let cached_integral_key =
            Evaluator::cached_integral_cache_key(resources.active.clone(), dataset);
        let cached_integral_state = Arc::new(CachedIntegralCacheState {
            key: cached_integral_key.clone(),
            expression_ir,
            values: cached_integrals,
            execution_sets,
        });
        let specialization_state = ExpressionSpecializationState {
            cached_integrals: cached_integral_state.clone(),
            lowered_artifacts: lowered_artifacts.clone(),
        };
        let specialization_cache = HashMap::from([(cached_integral_key, specialization_state)]);
        let lowered_artifact_cache =
            HashMap::from([(resources.active.clone(), lowered_artifacts.clone())]);
        Ok(Evaluator {
            amplitudes,
            amplitude_use_sites: self.registry.amplitude_use_sites.clone(),
            resources: Arc::new(RwLock::new(resources)),
            dataset: dataset.clone(),
            expression: self.tree.clone(),
            ir_planning: ExpressionIrPlanningState {
                cached_integrals: Arc::new(RwLock::new(Some(cached_integral_state))),
                specialization_cache: Arc::new(RwLock::new(specialization_cache)),
                specialization_metrics: Arc::new(RwLock::new(ExpressionSpecializationMetrics {
                    cache_hits: 0,
                    cache_misses: 1,
                })),
                lowered_artifact_cache: Arc::new(RwLock::new(lowered_artifact_cache)),
                active_lowered_artifacts: Arc::new(RwLock::new(Some(lowered_artifacts.clone()))),
                specialization_status: Arc::new(RwLock::new(Some(
                    ExpressionSpecializationStatus {
                        origin: ExpressionSpecializationOrigin::InitialLoad,
                    },
                ))),
                compile_metrics: Arc::new(RwLock::new(ExpressionCompileMetrics {
                    initial_ir_compile_nanos,
                    initial_cached_integrals_nanos,
                    initial_lowering_nanos,
                    specialization_lowering_cache_misses: 1,
                    ..Default::default()
                })),
            },
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
    /// Takes the square root of the given [`Expression`].
    pub fn sqrt(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Sqrt)
    }
    /// Raises the given [`Expression`] to an expression-valued power.
    pub fn pow(&self, power: &Expression) -> Self {
        Self::binary_op(self, power, ExpressionNode::Pow)
    }
    /// Raises the given [`Expression`] to an integer power.
    pub fn powi(&self, power: i32) -> Self {
        Self::unary_op(self, |input| ExpressionNode::PowI(input, power))
    }
    /// Raises the given [`Expression`] to a real-valued power.
    pub fn powf(&self, power: f64) -> Self {
        Self::unary_op(self, |input| ExpressionNode::PowF(input, power))
    }
    /// Takes the exponential of the given [`Expression`].
    pub fn exp(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Exp)
    }
    /// Takes the sine of the given [`Expression`].
    pub fn sin(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Sin)
    }
    /// Takes the cosine of the given [`Expression`].
    pub fn cos(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Cos)
    }
    /// Takes the natural logarithm of the given [`Expression`].
    pub fn log(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Log)
    }
    /// Takes the complex phase factor exp(i * expression).
    pub fn cis(&self) -> Self {
        Self::unary_op(self, ExpressionNode::Cis)
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
            ExpressionNode::Zero => "0 (exact)".to_string(),
            ExpressionNode::One => "1 (exact)".to_string(),
            ExpressionNode::Constant(v) => v.to_string(),
            ExpressionNode::ComplexConstant(v) => v.to_string(),
            ExpressionNode::Sqrt(_) => "Sqrt".to_string(),
            ExpressionNode::Pow(_, _) => "Pow".to_string(),
            ExpressionNode::PowI(_, power) => format!("PowI({power})"),
            ExpressionNode::PowF(_, power) => format!("PowF({power})"),
            ExpressionNode::Exp(_) => "Exp".to_string(),
            ExpressionNode::Sin(_) => "Sin".to_string(),
            ExpressionNode::Cos(_) => "Cos".to_string(),
            ExpressionNode::Log(_) => "Log".to_string(),
            ExpressionNode::Cis(_) => "Cis".to_string(),
        };
        writeln!(f, "{}{}{}", parent_prefix, immediate_prefix, display_string)?;
        match t {
            ExpressionNode::Amp(_)
            | ExpressionNode::Zero
            | ExpressionNode::One
            | ExpressionNode::Constant(_)
            | ExpressionNode::ComplexConstant(_) => {}
            ExpressionNode::Add(a, b)
            | ExpressionNode::Sub(a, b)
            | ExpressionNode::Mul(a, b)
            | ExpressionNode::Div(a, b)
            | ExpressionNode::Pow(a, b) => {
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
            | ExpressionNode::NormSqr(a)
            | ExpressionNode::Sqrt(a)
            | ExpressionNode::PowI(a, _)
            | ExpressionNode::PowF(a, _)
            | ExpressionNode::Exp(a)
            | ExpressionNode::Sin(a)
            | ExpressionNode::Cos(a)
            | ExpressionNode::Log(a)
            | ExpressionNode::Cis(a) => {
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
impl_op_ex!(+ |a: &Expression, b: &f64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Add)
});
#[rustfmt::skip]
impl_op_ex!(+ |a: &f64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Add)
});
#[rustfmt::skip]
impl_op_ex!(+ |a: &Expression, b: &Complex64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Add)
});
#[rustfmt::skip]
impl_op_ex!(+ |a: &Complex64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Add)
});

#[rustfmt::skip]
impl_op_ex!(- |a: &Expression, b: &Expression| -> Expression {
    Expression::binary_op(a, b, ExpressionNode::Sub)
});
#[rustfmt::skip]
impl_op_ex!(- |a: &Expression, b: &f64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Sub)
});
#[rustfmt::skip]
impl_op_ex!(- |a: &f64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Sub)
});
#[rustfmt::skip]
impl_op_ex!(- |a: &Expression, b: &Complex64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Sub)
});
#[rustfmt::skip]
impl_op_ex!(- |a: &Complex64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Sub)
});

#[rustfmt::skip]
impl_op_ex!(* |a: &Expression, b: &Expression| -> Expression {
    Expression::binary_op(a, b, ExpressionNode::Mul)
});
#[rustfmt::skip]
impl_op_ex!(* |a: &Expression, b: &f64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Mul)
});
#[rustfmt::skip]
impl_op_ex!(* |a: &f64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Mul)
});
#[rustfmt::skip]
impl_op_ex!(* |a: &Expression, b: &Complex64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Mul)
});
#[rustfmt::skip]
impl_op_ex!(* |a: &Complex64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Mul)
});

#[rustfmt::skip]
impl_op_ex!(/ |a: &Expression, b: &Expression| -> Expression {
    Expression::binary_op(a, b, ExpressionNode::Div)
});
#[rustfmt::skip]
impl_op_ex!(/ |a: &Expression, b: &f64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Div)
});
#[rustfmt::skip]
impl_op_ex!(/ |a: &f64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Div)
});
#[rustfmt::skip]
impl_op_ex!(/ |a: &Expression, b: &Complex64| -> Expression {
    Expression::binary_op(a, &Expression::from(b), ExpressionNode::Div)
});
#[rustfmt::skip]
impl_op_ex!(/ |a: &Complex64, b: &Expression| -> Expression {
    Expression::binary_op(&Expression::from(a), b, ExpressionNode::Div)
});

#[rustfmt::skip]
impl_op_ex!(- |a: &Expression| -> Expression {
    Expression::unary_op(a, ExpressionNode::Neg)
});
// NOTE: no need to add an impl for negating f64 or complex!

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ExpressionValueProgramSnapshot {
    lowered_program: lowered::LoweredProgram,
}

#[derive(Clone, Debug, PartialEq)]
/// A node in a compiled expression diagnostic snapshot.
pub enum CompiledExpressionNode {
    /// A complex constant.
    Constant(Complex64),
    /// A registered amplitude use-site by index and display label.
    Amplitude {
        /// The amplitude index used by the compiled evaluator.
        index: usize,
        /// The registered amplitude display label.
        name: String,
    },
    /// A unary operation and its input node.
    Unary {
        /// The display label for the operation.
        op: String,
        /// The input node index.
        input: usize,
    },
    /// A binary operation and its input nodes.
    Binary {
        /// The display label for the operation.
        op: String,
        /// The left input node index.
        left: usize,
        /// The right input node index.
        right: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
/// Tree-like diagnostic view of the compiled expression DAG.
///
/// The compiled expression is a directed acyclic graph because common subexpressions can be
/// deduplicated during compilation. The display format expands the graph from the root once and
/// marks later visits to the same node with `(ref)`.
pub struct CompiledExpression {
    nodes: Vec<CompiledExpressionNode>,
    root: usize,
}

impl CompiledExpression {
    fn from_ir(ir: &ir::ExpressionIR, amplitude_names: &[String]) -> Self {
        let nodes = ir
            .nodes()
            .iter()
            .map(|node| match node {
                ir::IrNode::Constant(value) => CompiledExpressionNode::Constant(*value),
                ir::IrNode::Amp(index) => CompiledExpressionNode::Amplitude {
                    index: *index,
                    name: amplitude_names
                        .get(*index)
                        .cloned()
                        .unwrap_or_else(|| "<unregistered>".to_string()),
                },
                ir::IrNode::Unary { op, input } => CompiledExpressionNode::Unary {
                    op: compiled_unary_op_label(*op),
                    input: *input,
                },
                ir::IrNode::Binary { op, left, right } => CompiledExpressionNode::Binary {
                    op: compiled_binary_op_label(*op),
                    left: *left,
                    right: *right,
                },
            })
            .collect();
        Self {
            nodes,
            root: ir.root(),
        }
    }

    /// Returns the compiled expression node list in evaluator execution order.
    pub fn nodes(&self) -> &[CompiledExpressionNode] {
        &self.nodes
    }

    /// Returns the root node index.
    pub fn root(&self) -> usize {
        self.root
    }

    fn node_label(&self, index: usize) -> String {
        let Some(node) = self.nodes.get(index) else {
            return format!("#{index} <missing>");
        };
        let label = match node {
            CompiledExpressionNode::Constant(value) => format!("const {value}"),
            CompiledExpressionNode::Amplitude { index, name } => {
                format!("{name}(id={index})")
            }
            CompiledExpressionNode::Unary { op, .. }
            | CompiledExpressionNode::Binary { op, .. } => op.clone(),
        };
        format!("#{index} {label}")
    }

    /// Credit to Daniel Janus: <https://blog.danieljanus.pl/2023/07/20/iterating-trees/>
    fn write_tree(
        &self,
        index: usize,
        f: &mut std::fmt::Formatter<'_>,
        parent_prefix: &str,
        immediate_prefix: &str,
        parent_suffix: &str,
        expanded: &mut [bool],
    ) -> std::fmt::Result {
        let already_expanded = expanded.get(index).copied().unwrap_or(false);
        if let Some(slot) = expanded.get_mut(index) {
            *slot = true;
        }
        let ref_suffix = if already_expanded { " (ref)" } else { "" };
        writeln!(
            f,
            "{}{}{}{}",
            parent_prefix,
            immediate_prefix,
            self.node_label(index),
            ref_suffix
        )?;
        if already_expanded {
            return Ok(());
        }
        let Some(node) = self.nodes.get(index) else {
            return Ok(());
        };
        match node {
            CompiledExpressionNode::Constant(_) | CompiledExpressionNode::Amplitude { .. } => {}
            CompiledExpressionNode::Unary { input, .. } => {
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                self.write_tree(*input, f, &child_prefix, "└─ ", "   ", expanded)?;
            }
            CompiledExpressionNode::Binary { left, right, .. } => {
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                self.write_tree(*left, f, &child_prefix, "├─ ", "│  ", expanded)?;
                self.write_tree(*right, f, &child_prefix, "└─ ", "   ", expanded)?;
            }
        }
        Ok(())
    }
}

impl Display for CompiledExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.nodes.is_empty() {
            return writeln!(f, "<empty>");
        }
        let mut expanded = vec![false; self.nodes.len()];
        self.write_tree(self.root, f, "", "", "", &mut expanded)
    }
}

fn compiled_unary_op_label(op: ir::IrUnaryOp) -> String {
    match op {
        ir::IrUnaryOp::Neg => "-".to_string(),
        ir::IrUnaryOp::Real => "Re".to_string(),
        ir::IrUnaryOp::Imag => "Im".to_string(),
        ir::IrUnaryOp::Conj => "*".to_string(),
        ir::IrUnaryOp::NormSqr => "NormSqr".to_string(),
        ir::IrUnaryOp::Sqrt => "Sqrt".to_string(),
        ir::IrUnaryOp::PowI(power) => format!("PowI({power})"),
        ir::IrUnaryOp::PowF(bits) => format!("PowF({})", f64::from_bits(bits)),
        ir::IrUnaryOp::Exp => "Exp".to_string(),
        ir::IrUnaryOp::Sin => "Sin".to_string(),
        ir::IrUnaryOp::Cos => "Cos".to_string(),
        ir::IrUnaryOp::Log => "Log".to_string(),
        ir::IrUnaryOp::Cis => "Cis".to_string(),
    }
}

fn compiled_binary_op_label(op: ir::IrBinaryOp) -> String {
    match op {
        ir::IrBinaryOp::Add => "+".to_string(),
        ir::IrBinaryOp::Sub => "-".to_string(),
        ir::IrBinaryOp::Mul => "×".to_string(),
        ir::IrBinaryOp::Div => "÷".to_string(),
        ir::IrBinaryOp::Pow => "Pow".to_string(),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Origin of the currently installed expression specialization.
pub enum ExpressionSpecializationOrigin {
    /// The specialization installed during evaluator construction.
    InitialLoad,
    /// The specialization was rebuilt because no cached entry matched the current state.
    CacheMissRebuild,
    /// The specialization was restored from an existing cache entry.
    CacheHitRestore,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Current specialization status for the evaluator runtime.
pub struct ExpressionSpecializationStatus {
    /// How the active specialization was obtained most recently.
    pub origin: ExpressionSpecializationOrigin,
}
#[derive(Clone, Debug, PartialEq, Eq)]
/// Diagnostic snapshot of the active runtime specialization state.
pub struct ExpressionRuntimeDiagnostics {
    /// Whether IR planning state is present for the evaluator.
    pub ir_planning_enabled: bool,
    /// Whether a lowered value-only program is available for the active specialization.
    pub lowered_value_program_present: bool,
    /// Whether a lowered gradient-only program is available for the active specialization.
    pub lowered_gradient_program_present: bool,
    /// Whether a lowered fused value+gradient program is available for the active specialization.
    pub lowered_value_gradient_program_present: bool,
    /// Number of cached parameter-factor descriptors in the active specialization.
    pub cached_parameter_factor_count: usize,
    /// Number of cached parameter factors with lowered runtimes available.
    pub lowered_cached_parameter_factor_count: usize,
    /// Whether a lowered residual normalization runtime is available.
    pub residual_runtime_present: bool,
    /// Number of cached specialization entries currently retained.
    pub specialization_cache_entries: usize,
    /// Number of cached lowered-artifact entries currently retained.
    pub lowered_artifact_cache_entries: usize,
    /// Origin of the currently installed specialization, when available.
    pub specialization_status: Option<ExpressionSpecializationStatus>,
}
#[derive(Clone)]
/// IR-planning state derived from the semantic expression tree plus the current active mask.
///
/// Invariants:
/// - `expression_ir` is never a source of truth; it is always derived from `Evaluator::expression`.
/// - `cached_integrals` are specialization-dependent and must be treated as invalid once the
///   active mask or dataset identity changes.
struct ExpressionIrPlanningState {
    cached_integrals: Arc<RwLock<Option<Arc<CachedIntegralCacheState>>>>,
    specialization_cache:
        Arc<RwLock<HashMap<CachedIntegralCacheKey, ExpressionSpecializationState>>>,
    specialization_metrics: Arc<RwLock<ExpressionSpecializationMetrics>>,
    lowered_artifact_cache: Arc<RwLock<HashMap<Vec<bool>, Arc<LoweredArtifactCacheState>>>>,
    active_lowered_artifacts: Arc<RwLock<Option<Arc<LoweredArtifactCacheState>>>>,
    specialization_status: Arc<RwLock<Option<ExpressionSpecializationStatus>>>,
    compile_metrics: Arc<RwLock<ExpressionCompileMetrics>>,
}
/// Evaluator for [`Expression`] that mirrors the existing evaluator behavior.
#[allow(missing_docs)]
#[derive(Clone)]
pub struct Evaluator {
    pub amplitudes: Vec<Box<dyn Amplitude>>,
    amplitude_use_sites: Vec<AmplitudeUseSite>,
    pub resources: Arc<RwLock<Resources>>,
    pub dataset: Arc<Dataset>,
    pub expression: ExpressionNode,
    ir_planning: ExpressionIrPlanningState,
    registry: ExpressionRegistry,
}

#[allow(missing_docs)]
impl Evaluator {
    /// Internal benchmarking/debug counters for specialization cache reuse.
    pub fn expression_specialization_metrics(&self) -> ExpressionSpecializationMetrics {
        *self.ir_planning.specialization_metrics.read()
    }
    /// Reset specialization cache counters while leaving cached specializations intact.
    pub fn reset_expression_specialization_metrics(&self) {
        *self.ir_planning.specialization_metrics.write() =
            ExpressionSpecializationMetrics::default();
    }
    /// Internal benchmarking/debug metrics for staged IR compile and lowering costs.
    pub fn expression_compile_metrics(&self) -> ExpressionCompileMetrics {
        *self.ir_planning.compile_metrics.read()
    }
    /// Internal diagnostics surface for active runtime specialization state.
    pub fn expression_runtime_diagnostics(&self) -> ExpressionRuntimeDiagnostics {
        let active_artifacts = self.active_lowered_artifacts();
        let cached_parameter_factor_count = self
            .ir_planning
            .cached_integrals
            .read()
            .as_ref()
            .map(|state| state.values.len())
            .unwrap_or(0);
        let lowered_cached_parameter_factor_count = active_artifacts
            .as_ref()
            .map(|artifacts| {
                artifacts
                    .lowered_parameter_factors
                    .iter()
                    .filter(|factor| factor.is_some())
                    .count()
            })
            .unwrap_or(0);
        let residual_runtime_present = active_artifacts
            .as_ref()
            .and_then(|artifacts| artifacts.residual_runtime.as_ref())
            .is_some();
        ExpressionRuntimeDiagnostics {
            ir_planning_enabled: true,
            lowered_value_program_present: true,
            lowered_gradient_program_present: true,
            lowered_value_gradient_program_present: true,
            cached_parameter_factor_count,
            lowered_cached_parameter_factor_count,
            residual_runtime_present,
            specialization_cache_entries: self.ir_planning.specialization_cache.read().len(),
            lowered_artifact_cache_entries: self.ir_planning.lowered_artifact_cache.read().len(),
            specialization_status: *self.ir_planning.specialization_status.read(),
        }
    }
    /// Reset post-load compile/lowering counters while preserving initial-load metrics.
    pub fn reset_expression_compile_metrics(&self) {
        let mut metrics = self.ir_planning.compile_metrics.write();
        metrics.specialization_cache_hits = 0;
        metrics.specialization_cache_misses = 0;
        metrics.specialization_ir_compile_nanos = 0;
        metrics.specialization_cached_integrals_nanos = 0;
        metrics.specialization_lowering_nanos = 0;
        metrics.specialization_lowering_cache_hits = 0;
        metrics.specialization_lowering_cache_misses = 0;
        metrics.specialization_cache_restore_nanos = 0;
    }
    #[cfg(test)]
    fn expression_ir(&self) -> ir::ExpressionIR {
        self.ir_planning
            .cached_integrals
            .read()
            .as_ref()
            .map(|state| state.expression_ir.clone())
            .expect("cached integral state should exist for evaluator IR access")
    }
    fn lowered_runtime(&self) -> lowered::LoweredExpressionRuntime {
        self.active_lowered_artifacts()
            .expect("active lowered artifacts should exist for the current specialization")
            .lowered_runtime
            .clone()
    }
    fn active_lowered_artifacts(&self) -> Option<Arc<LoweredArtifactCacheState>> {
        self.ir_planning.active_lowered_artifacts.read().clone()
    }
    fn lowered_runtime_slot_count(&self) -> usize {
        let runtime = self.lowered_runtime();
        [
            runtime.value_program().scratch_slots(),
            runtime.gradient_program().scratch_slots(),
            runtime.value_gradient_program().scratch_slots(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
    fn lowered_value_runtime_slot_count(&self) -> usize {
        self.lowered_runtime().value_program().scratch_slots()
    }

    #[doc(hidden)]
    pub fn expression_value_program_snapshot(&self) -> ExpressionValueProgramSnapshot {
        ExpressionValueProgramSnapshot {
            lowered_program: self.lowered_runtime().value_program().clone(),
        }
    }

    #[doc(hidden)]
    pub fn expression_value_program_snapshot_for_active_mask(
        &self,
        active_mask: &[bool],
    ) -> LadduResult<ExpressionValueProgramSnapshot> {
        let expression_ir = self.compile_expression_ir_for_active_mask(active_mask);
        let lowered_program =
            lowered::LoweredProgram::from_ir_value_only(&expression_ir).map_err(|error| {
                LadduError::Custom(format!(
                    "Failed to lower value-only active-mask runtime: {error:?}"
                ))
            })?;
        Ok(ExpressionValueProgramSnapshot { lowered_program })
    }

    #[doc(hidden)]
    pub fn expression_value_program_snapshot_slot_count(
        &self,
        snapshot: &ExpressionValueProgramSnapshot,
    ) -> usize {
        let _ = self;
        snapshot.lowered_program.scratch_slots()
    }

    /// Returns a tree-like diagnostic snapshot of the compiled expression for the evaluator's
    /// current active-amplitude mask.
    pub fn compiled_expression(&self) -> CompiledExpression {
        let expression_ir = self.compile_expression_ir_for_active_mask(&self.active_mask());
        CompiledExpression::from_ir(&expression_ir, &self.registry.amplitude_names)
    }

    /// Returns the expression represented by this evaluator.
    pub fn expression(&self) -> Expression {
        Expression {
            tree: self.expression.clone(),
            registry: self.registry.clone(),
        }
    }
    fn lowered_gradient_runtime_slot_count(&self) -> usize {
        self.lowered_runtime().gradient_program().scratch_slots()
    }
    fn lowered_value_gradient_runtime_slot_count(&self) -> usize {
        self.lowered_runtime()
            .value_gradient_program()
            .scratch_slots()
    }

    fn expression_value_slot_count(&self) -> usize {
        self.lowered_value_runtime_slot_count()
    }
    fn expression_gradient_slot_count(&self) -> usize {
        self.lowered_gradient_runtime_slot_count()
    }
    fn expression_value_gradient_slot_count(&self) -> usize {
        self.lowered_value_gradient_runtime_slot_count()
    }

    #[doc(hidden)]
    pub fn expression_value_gradient_slot_count_public(&self) -> usize {
        self.expression_value_gradient_slot_count()
    }
    #[cfg(test)]
    fn specialization_cache_len(&self) -> usize {
        self.ir_planning.specialization_cache.read().len()
    }
    #[cfg(test)]
    fn lowered_artifact_cache_len(&self) -> usize {
        self.ir_planning.lowered_artifact_cache.read().len()
    }
    fn install_expression_specialization(&self, specialization: &ExpressionSpecializationState) {
        debug_assert!(Self::lowered_artifact_signature_matches(
            &specialization.lowered_artifacts,
            &specialization.cached_integrals.values,
        ));
        *self.ir_planning.cached_integrals.write() = Some(specialization.cached_integrals.clone());
        *self.ir_planning.active_lowered_artifacts.write() =
            Some(specialization.lowered_artifacts.clone());
        debug_assert_eq!(
            self.active_lowered_artifacts()
                .as_ref()
                .map(|artifacts| Arc::ptr_eq(artifacts, &specialization.lowered_artifacts)),
            Some(true)
        );
        debug_assert_eq!(
            self.lowered_runtime().value_program().scratch_slots(),
            specialization
                .lowered_artifacts
                .lowered_runtime
                .value_program()
                .scratch_slots()
        );
    }
    fn lower_expression_runtime_artifacts(
        expression_ir: &ir::ExpressionIR,
        values: &[PrecomputedCachedIntegral],
    ) -> LadduResult<LoweredArtifactCacheState> {
        let parameter_node_indices = values
            .iter()
            .map(|value| value.parameter_node_index)
            .collect();
        let mul_node_indices = values.iter().map(|value| value.mul_node_index).collect();
        let lowered_parameter_factors = Self::lower_cached_parameter_factors(expression_ir);
        let residual_runtime = Self::lower_residual_runtime(expression_ir, values);
        let lowered_runtime = lowered::LoweredExpressionRuntime::from_ir_value_gradient(
            expression_ir,
        )
        .map_err(|error| {
            LadduError::Custom(format!(
                "Failed to lower expression runtime for specialized IR: {error:?}"
            ))
        })?;
        Ok(LoweredArtifactCacheState {
            parameter_node_indices,
            mul_node_indices,
            lowered_parameter_factors,
            residual_runtime,
            lowered_runtime,
        })
    }
    fn lowered_artifact_signature_matches(
        artifacts: &LoweredArtifactCacheState,
        values: &[PrecomputedCachedIntegral],
    ) -> bool {
        artifacts.parameter_node_indices.len() == values.len()
            && artifacts.mul_node_indices.len() == values.len()
            && artifacts
                .parameter_node_indices
                .iter()
                .copied()
                .eq(values.iter().map(|value| value.parameter_node_index))
            && artifacts
                .mul_node_indices
                .iter()
                .copied()
                .eq(values.iter().map(|value| value.mul_node_index))
    }
    fn build_expression_specialization(
        &self,
        resources: &Resources,
        key: CachedIntegralCacheKey,
    ) -> LadduResult<ExpressionSpecializationState> {
        let ir_compile_start = Instant::now();
        let expression_ir = self.compile_expression_ir_for_active_mask(&resources.active);
        let ir_compile_nanos = ir_compile_start.elapsed().as_nanos() as u64;
        let cached_integrals_start = Instant::now();
        let values = Self::precompute_cached_integrals_at_load(
            &expression_ir,
            &self.amplitudes,
            &self.amplitude_use_sites,
            resources,
            &self.dataset,
            self.resources.read().n_free_parameters(),
        )?;
        let cached_integrals_nanos = cached_integrals_start.elapsed().as_nanos() as u64;
        let execution_sets = expression_ir.normalization_execution_sets().clone();
        let active_mask_key = resources.active.clone();
        let cached_lowered_artifacts = {
            let lowered_artifact_cache = self.ir_planning.lowered_artifact_cache.read();
            lowered_artifact_cache
                .get(&active_mask_key)
                .cloned()
                .filter(|artifacts| Self::lowered_artifact_signature_matches(artifacts, &values))
        };
        let lowered_artifacts = if let Some(artifacts) = cached_lowered_artifacts {
            self.ir_planning
                .compile_metrics
                .write()
                .specialization_lowering_cache_hits += 1;
            artifacts
        } else {
            let lowering_start = Instant::now();
            let artifacts = Arc::new(
                Self::lower_expression_runtime_artifacts(&expression_ir, &values)
                    .expect("specialized lowered runtime should build"),
            );
            let lowering_nanos = lowering_start.elapsed().as_nanos() as u64;
            self.ir_planning
                .lowered_artifact_cache
                .write()
                .insert(active_mask_key, artifacts.clone());
            let mut compile_metrics = self.ir_planning.compile_metrics.write();
            compile_metrics.specialization_lowering_cache_misses += 1;
            compile_metrics.specialization_lowering_nanos += lowering_nanos;
            artifacts
        };
        let mut compile_metrics = self.ir_planning.compile_metrics.write();
        compile_metrics.specialization_cache_misses += 1;
        compile_metrics.specialization_ir_compile_nanos += ir_compile_nanos;
        compile_metrics.specialization_cached_integrals_nanos += cached_integrals_nanos;
        Ok(ExpressionSpecializationState {
            cached_integrals: Arc::new(CachedIntegralCacheState {
                key,
                expression_ir,
                values,
                execution_sets,
            }),
            lowered_artifacts,
        })
    }
    fn ensure_expression_specialization(
        &self,
        resources: &Resources,
    ) -> LadduResult<ExpressionSpecializationState> {
        let key = Self::cached_integral_cache_key(resources.active.clone(), &self.dataset);
        if let Some(state) = self.ir_planning.cached_integrals.read().as_ref() {
            if state.key == key {
                return Ok(ExpressionSpecializationState {
                    cached_integrals: state.clone(),
                    lowered_artifacts: self
                        .active_lowered_artifacts()
                        .expect("active lowered artifacts should exist for cached specialization"),
                });
            }
        }
        let cached_specialization = {
            let specialization_cache = self.ir_planning.specialization_cache.read();
            specialization_cache.get(&key).cloned()
        };
        if let Some(specialization) = cached_specialization {
            let restore_start = Instant::now();
            self.ir_planning.specialization_metrics.write().cache_hits += 1;
            self.install_expression_specialization(&specialization);
            *self.ir_planning.specialization_status.write() =
                Some(ExpressionSpecializationStatus {
                    origin: ExpressionSpecializationOrigin::CacheHitRestore,
                });
            let restore_nanos = restore_start.elapsed().as_nanos() as u64;
            let mut compile_metrics = self.ir_planning.compile_metrics.write();
            compile_metrics.specialization_cache_hits += 1;
            compile_metrics.specialization_cache_restore_nanos += restore_nanos;
            return Ok(specialization);
        }
        let specialization = self.build_expression_specialization(resources, key.clone())?;
        self.ir_planning.specialization_metrics.write().cache_misses += 1;
        self.ir_planning
            .specialization_cache
            .write()
            .insert(key, specialization.clone());
        self.install_expression_specialization(&specialization);
        let origin = if self.ir_planning.specialization_cache.read().len() == 1 {
            ExpressionSpecializationOrigin::InitialLoad
        } else {
            ExpressionSpecializationOrigin::CacheMissRebuild
        };
        *self.ir_planning.specialization_status.write() =
            Some(ExpressionSpecializationStatus { origin });
        Ok(specialization)
    }
    fn rebuild_runtime_specializations(&self, resources: &Resources) {
        let _ = self.ensure_expression_specialization(resources);
    }
    fn refresh_runtime_specializations(&self) {
        let resources = self.resources.read();
        self.rebuild_runtime_specializations(&resources);
    }
    fn cached_integral_cache_key(
        active_mask: Vec<bool>,
        dataset: &Dataset,
    ) -> CachedIntegralCacheKey {
        CachedIntegralCacheKey {
            active_mask,
            n_events_local: dataset.n_events_local(),
            weights_local_len: dataset.columnar.weights.len(),
            weighted_sum_bits: dataset.n_events_weighted_local().to_bits(),
            weights_ptr: dataset.columnar.weights.as_ptr() as usize,
        }
    }
    fn precompute_cached_integrals_at_load(
        expression_ir: &ir::ExpressionIR,
        amplitudes: &[Box<dyn Amplitude>],
        amplitude_use_sites: &[AmplitudeUseSite],
        resources: &Resources,
        dataset: &Dataset,
        n_free_parameters: usize,
    ) -> LadduResult<Vec<PrecomputedCachedIntegral>> {
        let descriptors = expression_ir.cached_integral_descriptors();
        if descriptors.is_empty() {
            return Ok(Vec::new());
        }
        let execution_sets = expression_ir.normalization_execution_sets();
        let seed_parameters = vec![0.0; n_free_parameters];
        let parameters = resources.parameter_map.assemble(&seed_parameters)?;
        let mut amplitude_values = vec![Complex64::ZERO; amplitude_use_sites.len()];
        let mut compute_values = vec![Complex64::ZERO; amplitudes.len()];
        let mut value_slots = vec![Complex64::ZERO; expression_ir.node_count()];
        let active_set = resources.active_indices();
        let cache_active_indices = execution_sets
            .cached_cache_amplitudes
            .iter()
            .copied()
            .filter(|index| active_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        let mut weighted_cache_sums = vec![Complex64::ZERO; descriptors.len()];
        for (cache, event) in resources.caches.iter().zip(dataset.weights_local().iter()) {
            amplitude_values.fill(Complex64::ZERO);
            compute_values.fill(Complex64::ZERO);
            let mut computed = vec![false; amplitudes.len()];
            for &use_site_idx in &cache_active_indices {
                let amp_idx = amplitude_use_sites[use_site_idx].amplitude_index;
                if !computed[amp_idx] {
                    compute_values[amp_idx] = amplitudes[amp_idx].compute(&parameters, cache);
                    computed[amp_idx] = true;
                }
                amplitude_values[use_site_idx] = compute_values[amp_idx];
            }
            expression_ir.evaluate_into(&amplitude_values, &mut value_slots);
            let weight = *event;
            for (descriptor_index, descriptor) in descriptors.iter().enumerate() {
                weighted_cache_sums[descriptor_index] +=
                    value_slots[descriptor.cache_node_index] * weight;
            }
        }
        Ok(descriptors
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
            .collect())
    }
    fn lower_cached_parameter_factors(
        expression_ir: &ir::ExpressionIR,
    ) -> Vec<Option<lowered::LoweredFactorRuntime>> {
        expression_ir
            .cached_integral_descriptors()
            .iter()
            .map(|descriptor| {
                lowered::LoweredFactorRuntime::from_ir_root_value_gradient(
                    expression_ir,
                    descriptor.parameter_node_index,
                )
                .ok()
            })
            .collect()
    }
    fn lower_residual_runtime(
        expression_ir: &ir::ExpressionIR,
        descriptors: &[PrecomputedCachedIntegral],
    ) -> Option<lowered::LoweredExpressionRuntime> {
        let mut zeroed_nodes = vec![false; expression_ir.node_count()];
        for descriptor in descriptors {
            if descriptor.mul_node_index < zeroed_nodes.len() {
                zeroed_nodes[descriptor.mul_node_index] = true;
            }
        }
        lowered::LoweredExpressionRuntime::from_ir_zeroed_value_gradient(
            expression_ir,
            &zeroed_nodes,
        )
        .ok()
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
        let mut compute_values = vec![Complex64::ZERO; self.amplitudes.len()];
        let mut computed = vec![false; self.amplitudes.len()];
        for &use_site_idx in active_indices {
            let amp_idx = self.amplitude_use_sites[use_site_idx].amplitude_index;
            if !computed[amp_idx] {
                compute_values[amp_idx] = self.amplitudes[amp_idx].compute(parameters, cache);
                computed[amp_idx] = true;
            }
            amplitude_values[use_site_idx] = compute_values[amp_idx];
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
        let mut compute_gradients = vec![DVector::zeros(parameters.len()); self.amplitudes.len()];
        let mut computed = vec![false; self.amplitudes.len()];
        for ((use_site, active), grad) in self
            .amplitude_use_sites
            .iter()
            .zip(active_mask.iter())
            .zip(gradient_values.iter_mut())
        {
            grad.fill(Complex64::ZERO);
            if *active {
                let amp_idx = use_site.amplitude_index;
                if !computed[amp_idx] {
                    self.amplitudes[amp_idx].compute_gradient(
                        parameters,
                        cache,
                        &mut compute_gradients[amp_idx],
                    );
                    computed[amp_idx] = true;
                }
                grad.copy_from(&compute_gradients[amp_idx]);
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

    #[doc(hidden)]
    pub fn fill_amplitude_values_and_gradients_public(
        &self,
        amplitude_values: &mut [Complex64],
        gradient_values: &mut [DVector<Complex64>],
        active_indices: &[usize],
        active_mask: &[bool],
        parameters: &Parameters,
        cache: &Cache,
    ) {
        self.fill_amplitude_values_and_gradients(
            amplitude_values,
            gradient_values,
            active_indices,
            active_mask,
            parameters,
            cache,
        );
    }

    #[cfg(feature = "execution-context-prototype")]
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

    #[cfg(feature = "execution-context-prototype")]
    #[allow(dead_code)]
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
        self.lowered_runtime_slot_count()
    }
    fn compile_expression_ir_for_active_mask(&self, active_mask: &[bool]) -> ir::ExpressionIR {
        let amplitude_dependencies = self
            .amplitude_use_sites
            .iter()
            .map(|use_site| {
                ir::DependenceClass::from(
                    self.amplitudes[use_site.amplitude_index].dependence_hint(),
                )
            })
            .collect::<Vec<_>>();
        let amplitude_realness = self
            .amplitude_use_sites
            .iter()
            .map(|use_site| self.amplitudes[use_site.amplitude_index].real_valued_hint())
            .collect::<Vec<_>>();
        ir::compile_expression_ir_with_real_hints(
            &self.expression,
            active_mask,
            &amplitude_dependencies,
            &amplitude_realness,
        )
    }
    fn lower_expression_runtime_for_active_mask(
        &self,
        active_mask: &[bool],
    ) -> LadduResult<lowered::LoweredExpressionRuntime> {
        let expression_ir = self.compile_expression_ir_for_active_mask(active_mask);
        lowered::LoweredExpressionRuntime::from_ir_value_gradient(&expression_ir).map_err(|error| {
            LadduError::Custom(format!(
                "Failed to lower active-mask runtime specialization: {error:?}"
            ))
        })
    }
    fn ensure_cached_integral_cache_state(
        &self,
        resources: &Resources,
    ) -> LadduResult<Arc<CachedIntegralCacheState>> {
        Ok(self
            .ensure_expression_specialization(resources)?
            .cached_integrals)
    }

    fn evaluate_expression_runtime_value_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        scratch: &mut [Complex64],
    ) -> Complex64 {
        let lowered_runtime = self.lowered_runtime();
        lowered_runtime
            .value_program()
            .evaluate_into(amplitude_values, scratch)
    }

    #[doc(hidden)]
    pub fn evaluate_expression_value_with_program_snapshot(
        &self,
        program_snapshot: &ExpressionValueProgramSnapshot,
        amplitude_values: &[Complex64],
        scratch: &mut [Complex64],
    ) -> Complex64 {
        program_snapshot
            .lowered_program
            .evaluate_into(amplitude_values, scratch)
    }

    fn evaluate_expression_runtime_gradient_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> DVector<Complex64> {
        let lowered_runtime = self.lowered_runtime();
        lowered_runtime.gradient_program().evaluate_gradient_into(
            amplitude_values,
            gradient_values,
            value_scratch,
            gradient_scratch,
        )
    }

    fn evaluate_expression_runtime_value_gradient_with_scratch(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
        value_scratch: &mut [Complex64],
        gradient_scratch: &mut [DVector<Complex64>],
    ) -> (Complex64, DVector<Complex64>) {
        let lowered_runtime = self.lowered_runtime();
        lowered_runtime
            .value_gradient_program()
            .evaluate_value_gradient_into(
                amplitude_values,
                gradient_values,
                value_scratch,
                gradient_scratch,
            )
    }

    fn evaluate_expression_runtime_value(&self, amplitude_values: &[Complex64]) -> Complex64 {
        let lowered_runtime = self.lowered_runtime();
        let program = lowered_runtime.value_program();
        let mut scratch = vec![Complex64::ZERO; program.scratch_slots()];
        program.evaluate_into(amplitude_values, &mut scratch)
    }

    fn evaluate_expression_runtime_gradient(
        &self,
        amplitude_values: &[Complex64],
        gradient_values: &[DVector<Complex64>],
    ) -> DVector<Complex64> {
        let lowered_runtime = self.lowered_runtime();
        let program = lowered_runtime.gradient_program();
        let mut value_scratch = vec![Complex64::ZERO; program.scratch_slots()];
        let grad_dim = gradient_values.first().map(|g| g.len()).unwrap_or(0);
        let mut gradient_scratch = vec![Complex64::ZERO; program.scratch_slots() * grad_dim];
        program.evaluate_gradient_into_flat(
            amplitude_values,
            gradient_values,
            &mut value_scratch,
            &mut gradient_scratch,
            grad_dim,
        )
    }
    /// Dependence classification for the compiled expression root.
    pub fn expression_root_dependence(&self) -> LadduResult<ExpressionDependence> {
        let resources = self.resources.read();
        Ok(self
            .ensure_cached_integral_cache_state(&resources)?
            .expression_ir
            .root_dependence()
            .into())
    }
    /// Dependence classification for each compiled expression node.
    pub fn expression_node_dependence_annotations(&self) -> LadduResult<Vec<ExpressionDependence>> {
        let resources = self.resources.read();
        Ok(self
            .ensure_cached_integral_cache_state(&resources)?
            .expression_ir
            .node_dependence_annotations()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }
    /// Warning-level diagnostics for potentially inconsistent dependence hints.
    pub fn expression_dependence_warnings(&self) -> LadduResult<Vec<String>> {
        let resources = self.resources.read();
        Ok(self
            .ensure_cached_integral_cache_state(&resources)?
            .expression_ir
            .dependence_warnings()
            .to_vec())
    }
    /// Explain/debug view of IR normalization planning decomposition.
    pub fn expression_normalization_plan_explain(&self) -> LadduResult<NormalizationPlanExplain> {
        let resources = self.resources.read();
        Ok(self
            .ensure_cached_integral_cache_state(&resources)?
            .expression_ir
            .normalization_plan_explain()
            .into())
    }
    /// Explain/debug view of amplitude execution sets used by normalization evaluation.
    pub fn expression_normalization_execution_sets(
        &self,
    ) -> LadduResult<NormalizationExecutionSetsExplain> {
        let resources = self.resources.read();
        Ok(self
            .ensure_cached_integral_cache_state(&resources)?
            .execution_sets
            .clone()
            .into())
    }
    /// Cached integral terms precomputed at evaluator load.
    pub fn expression_precomputed_cached_integrals(
        &self,
    ) -> LadduResult<Vec<PrecomputedCachedIntegral>> {
        let resources = self.resources.read();
        Ok(self
            .ensure_cached_integral_cache_state(&resources)?
            .values
            .clone())
    }
    /// Derivative rules for cached separable terms evaluated at the given parameter point.
    ///
    /// Each returned term corresponds to a cached separable descriptor and contributes
    /// `weighted_gradient` to `d(normalization)/dp` prior to residual-term combination.
    pub fn expression_precomputed_cached_integral_gradient_terms(
        &self,
        parameters: &[f64],
    ) -> LadduResult<Vec<PrecomputedCachedIntegralGradientTerm>> {
        let resources = self.resources.read();
        let state = self.ensure_cached_integral_cache_state(&resources)?;
        if state.values.is_empty() {
            return Ok(Vec::new());
        }

        let Some(cache) = resources.caches.first() else {
            return Ok(state
                .values
                .iter()
                .map(|descriptor| PrecomputedCachedIntegralGradientTerm {
                    mul_node_index: descriptor.mul_node_index,
                    parameter_node_index: descriptor.parameter_node_index,
                    cache_node_index: descriptor.cache_node_index,
                    coefficient: descriptor.coefficient,
                    weighted_gradient: DVector::zeros(parameters.len()),
                })
                .collect());
        };

        let parameter_values = resources.parameter_map.assemble(parameters)?;
        let mut amplitude_values = vec![Complex64::ZERO; self.amplitude_use_sites.len()];
        self.fill_amplitude_values(
            &mut amplitude_values,
            resources.active_indices(),
            &parameter_values,
            cache,
        );
        let mut amplitude_gradients = (0..self.amplitude_use_sites.len())
            .map(|_| DVector::zeros(parameters.len()))
            .collect::<Vec<_>>();
        self.fill_amplitude_gradients(
            &mut amplitude_gradients,
            &resources.active,
            &parameter_values,
            cache,
        );
        let lowered_artifacts = self.active_lowered_artifacts();
        let mut value_slots = vec![Complex64::ZERO; state.expression_ir.node_count()];
        let mut gradient_slots = (0..state.expression_ir.node_count())
            .map(|_| DVector::zeros(parameters.len()))
            .collect::<Vec<_>>();
        let max_lowered_slots = lowered_artifacts
            .as_ref()
            .map(|artifacts| {
                artifacts
                    .lowered_parameter_factors
                    .iter()
                    .filter_map(|runtime| {
                        runtime
                            .as_ref()
                            .and_then(|runtime| runtime.gradient_program())
                            .map(|program| program.scratch_slots())
                    })
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        let mut lowered_value_slots = vec![Complex64::ZERO; max_lowered_slots];
        let mut lowered_gradient_slots = vec![DVector::zeros(parameters.len()); max_lowered_slots];
        let use_lowered = lowered_artifacts.as_ref().is_some_and(|artifacts| {
            artifacts.lowered_parameter_factors.len() == state.values.len()
                && artifacts.lowered_parameter_factors.iter().all(|runtime| {
                    runtime
                        .as_ref()
                        .and_then(|runtime| runtime.gradient_program())
                        .is_some()
                })
        });

        if !use_lowered {
            let _ = state.expression_ir.evaluate_gradient_into(
                &amplitude_values,
                &amplitude_gradients,
                &mut value_slots,
                &mut gradient_slots,
            );
        }

        if use_lowered {
            let lowered_artifacts = lowered_artifacts.expect("lowered artifacts should exist");
            Ok(state
                .values
                .iter()
                .cloned()
                .zip(lowered_artifacts.lowered_parameter_factors.iter())
                .map(|(descriptor, runtime)| {
                    let parameter_gradient = runtime
                        .as_ref()
                        .and_then(|runtime| runtime.gradient_program())
                        .map(|program| {
                            program.evaluate_gradient_into(
                                &amplitude_values,
                                &amplitude_gradients,
                                &mut lowered_value_slots[..program.scratch_slots()],
                                &mut lowered_gradient_slots[..program.scratch_slots()],
                            )
                        })
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
                .collect())
        } else {
            Ok(state
                .values
                .iter()
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
                .collect())
        }
    }
    fn evaluate_cached_weighted_value_sum_ir(
        &self,
        state: &CachedIntegralCacheState,
        amplitude_values: &[Complex64],
    ) -> f64 {
        let mut value_slots = vec![Complex64::ZERO; state.expression_ir.node_count()];
        let _ = state
            .expression_ir
            .evaluate_into(amplitude_values, &mut value_slots);
        state
            .values
            .iter()
            .map(|descriptor| {
                let parameter_factor = value_slots[descriptor.parameter_node_index];
                (parameter_factor * descriptor.weighted_cache_sum * descriptor.coefficient as f64)
                    .re
            })
            .sum()
    }
    fn evaluate_cached_weighted_value_sum_lowered(
        &self,
        state: &CachedIntegralCacheState,
        lowered_artifacts: &LoweredArtifactCacheState,
        amplitude_values: &[Complex64],
    ) -> Option<f64> {
        let max_slots = lowered_artifacts
            .lowered_parameter_factors
            .iter()
            .filter_map(|runtime| {
                runtime
                    .as_ref()
                    .and_then(|runtime| runtime.value_program())
                    .map(|program| program.scratch_slots())
            })
            .max()
            .unwrap_or(0);
        let mut value_slots = vec![Complex64::ZERO; max_slots];
        let mut total = 0.0;
        for (descriptor, runtime) in state
            .values
            .iter()
            .zip(lowered_artifacts.lowered_parameter_factors.iter())
        {
            let parameter_factor = runtime
                .as_ref()
                .and_then(|runtime| runtime.value_program())
                .map(|program| {
                    program.evaluate_into(
                        amplitude_values,
                        &mut value_slots[..program.scratch_slots()],
                    )
                })?;
            total +=
                (parameter_factor * descriptor.weighted_cache_sum * descriptor.coefficient as f64)
                    .re;
        }
        Some(total)
    }
    fn evaluate_cached_weighted_gradient_sum_ir(
        &self,
        state: &CachedIntegralCacheState,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        grad_dim: usize,
    ) -> DVector<f64> {
        let mut value_slots = vec![Complex64::ZERO; state.expression_ir.node_count()];
        let mut gradient_slots = vec![DVector::zeros(grad_dim); state.expression_ir.node_count()];
        let _ = state.expression_ir.evaluate_gradient_into(
            amplitude_values,
            amplitude_gradients,
            &mut value_slots,
            &mut gradient_slots,
        );
        state
            .values
            .iter()
            .fold(DVector::zeros(grad_dim), |mut accum, descriptor| {
                let parameter_gradient = &gradient_slots[descriptor.parameter_node_index];
                let coefficient = descriptor.coefficient as f64;
                for (accum_item, gradient_item) in accum.iter_mut().zip(parameter_gradient.iter()) {
                    *accum_item +=
                        (*gradient_item * descriptor.weighted_cache_sum * coefficient).re;
                }
                accum
            })
    }
    fn evaluate_cached_weighted_gradient_sum_lowered(
        &self,
        state: &CachedIntegralCacheState,
        lowered_artifacts: &LoweredArtifactCacheState,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        grad_dim: usize,
    ) -> Option<DVector<f64>> {
        let max_value_slots = lowered_artifacts
            .lowered_parameter_factors
            .iter()
            .filter_map(|runtime| {
                runtime
                    .as_ref()
                    .and_then(|runtime| runtime.gradient_program())
                    .map(|program| program.scratch_slots())
            })
            .max()
            .unwrap_or(0);
        let mut value_slots = vec![Complex64::ZERO; max_value_slots];
        let mut gradient_slots = vec![Complex64::ZERO; max_value_slots * grad_dim];
        let mut total = DVector::zeros(grad_dim);
        for (descriptor, runtime) in state
            .values
            .iter()
            .zip(lowered_artifacts.lowered_parameter_factors.iter())
        {
            let parameter_gradient = runtime
                .as_ref()
                .and_then(|runtime| runtime.gradient_program())
                .map(|program| {
                    program.evaluate_gradient_into_flat(
                        amplitude_values,
                        amplitude_gradients,
                        &mut value_slots[..program.scratch_slots()],
                        &mut gradient_slots[..program.scratch_slots() * grad_dim],
                        grad_dim,
                    )
                })?;
            let coefficient = descriptor.coefficient as f64;
            for (accum_item, gradient_item) in total.iter_mut().zip(parameter_gradient.iter()) {
                *accum_item += (*gradient_item * descriptor.weighted_cache_sum * coefficient).re;
            }
        }
        Some(total)
    }
    fn evaluate_residual_value_ir(
        &self,
        state: &CachedIntegralCacheState,
        amplitude_values: &[Complex64],
    ) -> Complex64 {
        let mut zeroed_nodes = vec![false; state.expression_ir.node_count()];
        for descriptor in &state.values {
            if descriptor.mul_node_index < zeroed_nodes.len() {
                zeroed_nodes[descriptor.mul_node_index] = true;
            }
        }
        let mut value_slots = vec![Complex64::ZERO; state.expression_ir.node_count()];
        state.expression_ir.evaluate_into_with_zeroed_nodes(
            amplitude_values,
            &mut value_slots,
            &zeroed_nodes,
        )
    }
    fn evaluate_residual_gradient_ir(
        &self,
        state: &CachedIntegralCacheState,
        amplitude_values: &[Complex64],
        amplitude_gradients: &[DVector<Complex64>],
        grad_dim: usize,
    ) -> DVector<Complex64> {
        let mut zeroed_nodes = vec![false; state.expression_ir.node_count()];
        for descriptor in &state.values {
            if descriptor.mul_node_index < zeroed_nodes.len() {
                zeroed_nodes[descriptor.mul_node_index] = true;
            }
        }
        let mut value_slots = vec![Complex64::ZERO; state.expression_ir.node_count()];
        let mut gradient_slots = vec![DVector::zeros(grad_dim); state.expression_ir.node_count()];
        state
            .expression_ir
            .evaluate_gradient_into_with_zeroed_nodes(
                amplitude_values,
                amplitude_gradients,
                &mut value_slots,
                &mut gradient_slots,
                &zeroed_nodes,
            )
    }

    fn evaluate_weighted_value_sum_local_components(
        &self,
        parameters: &[f64],
    ) -> LadduResult<(f64, f64)> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let state = self.ensure_cached_integral_cache_state(&resources)?;
        let lowered_artifacts = self.active_lowered_artifacts();
        let residual_value_slot_count = lowered_artifacts
            .as_ref()
            .and_then(|artifacts| {
                artifacts
                    .residual_runtime
                    .as_ref()
                    .map(|runtime| runtime.value_program())
                    .map(|program| program.scratch_slots())
            })
            .unwrap_or_else(|| self.expression_slot_count());
        let residual_value_program = lowered_artifacts
            .as_ref()
            .and_then(|artifacts| artifacts.residual_runtime.as_ref())
            .map(|runtime| runtime.value_program());
        let cached_parameter_indices = &state.execution_sets.cached_parameter_amplitudes;
        let residual_active_indices = &state.execution_sets.residual_amplitudes;
        debug_assert!(cached_parameter_indices.iter().all(|&index| resources
            .active
            .get(index)
            .copied()
            .unwrap_or(false)));
        debug_assert!(residual_active_indices.iter().all(|&index| resources
            .active
            .get(index)
            .copied()
            .unwrap_or(false)));
        let cached_value_sum = {
            if let Some(cache) = resources.caches.first() {
                let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
                self.fill_amplitude_values(
                    &mut amplitude_values,
                    cached_parameter_indices,
                    &parameters,
                    cache,
                );
                lowered_artifacts
                    .as_ref()
                    .and_then(|artifacts| {
                        self.evaluate_cached_weighted_value_sum_lowered(
                            &state,
                            artifacts,
                            &amplitude_values,
                        )
                    })
                    .unwrap_or_else(|| {
                        self.evaluate_cached_weighted_value_sum_ir(&state, &amplitude_values)
                    })
            } else {
                0.0
            }
        };

        #[cfg(feature = "rayon")]
        let residual_sum: f64 = {
            resources
                .caches
                .par_iter()
                .zip(self.dataset.weights_local().par_iter())
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![Complex64::ZERO; residual_value_slot_count],
                        )
                    },
                    |(amplitude_values, value_slots), (cache, event)| {
                        self.fill_amplitude_values(
                            amplitude_values,
                            residual_active_indices,
                            &parameters,
                            cache,
                        );
                        {
                            let value = residual_value_program
                                .as_ref()
                                .map(|program| {
                                    program.evaluate_into(
                                        amplitude_values,
                                        &mut value_slots[..program.scratch_slots()],
                                    )
                                })
                                .unwrap_or_else(|| {
                                    self.evaluate_residual_value_ir(&state, amplitude_values)
                                });
                            *event * value.re
                        }
                    },
                )
                .sum()
        };

        #[cfg(not(feature = "rayon"))]
        let residual_sum: f64 = {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut value_slots = vec![Complex64::ZERO; residual_value_slot_count];
            resources
                .caches
                .iter()
                .zip(self.dataset.weights_local().iter())
                .map(|(cache, event)| {
                    self.fill_amplitude_values(
                        &mut amplitude_values,
                        &residual_active_indices,
                        &parameters,
                        cache,
                    );
                    {
                        let value = residual_value_program
                            .as_ref()
                            .map(|program| {
                                program.evaluate_into(
                                    &amplitude_values,
                                    &mut value_slots[..program.scratch_slots()],
                                )
                            })
                            .unwrap_or_else(|| {
                                self.evaluate_residual_value_ir(&state, &amplitude_values)
                            });
                        *event * value.re
                    }
                })
                .sum()
        };
        Ok((residual_sum, cached_value_sum))
    }

    /// Weighted sum over local events of the real expression value.
    ///
    /// This returns `sum_e(weight_e * Re(L_e))`.
    pub fn evaluate_weighted_value_sum_local(&self, parameters: &[f64]) -> LadduResult<f64> {
        let (residual_sum, cached_value_sum) =
            self.evaluate_weighted_value_sum_local_components(parameters)?;
        Ok(residual_sum + cached_value_sum)
    }

    #[cfg(feature = "mpi")]
    /// Weighted sum over all ranks of the real expression value.
    ///
    /// This returns `sum_{r,e}(weight_{r,e} * Re(L_{r,e}))`.
    pub fn evaluate_weighted_value_sum_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> LadduResult<f64> {
        let (residual_sum_local, cached_value_sum_local) =
            self.evaluate_weighted_value_sum_local_components(parameters)?;
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
        Ok(residual_sum + cached_value_sum)
    }

    /// Weighted sum over local events of the real gradient of the expression.
    ///
    /// This returns `sum_e(weight_e * Re(dL_e/dp))` for all free parameters.
    fn evaluate_weighted_gradient_sum_local_components(
        &self,
        parameters: &[f64],
    ) -> LadduResult<(DVector<f64>, DVector<f64>)> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let grad_dim = parameters.len();
        let state = self.ensure_cached_integral_cache_state(&resources)?;
        let lowered_artifacts = self.active_lowered_artifacts();
        let active_index_set = resources.active_indices();
        let cached_parameter_indices = state
            .execution_sets
            .cached_parameter_amplitudes
            .iter()
            .copied()
            .filter(|index| active_index_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        let residual_active_indices = state
            .execution_sets
            .residual_amplitudes
            .iter()
            .copied()
            .filter(|index| active_index_set.binary_search(index).is_ok())
            .collect::<Vec<_>>();
        let mut cached_parameter_mask = vec![false; amplitude_len];
        for &index in &cached_parameter_indices {
            cached_parameter_mask[index] = true;
        }
        let mut residual_active_mask = vec![false; amplitude_len];
        for &index in &residual_active_indices {
            residual_active_mask[index] = true;
        }
        let residual_gradient_program = lowered_artifacts
            .as_ref()
            .and_then(|artifacts| artifacts.residual_runtime.as_ref())
            .map(|runtime| runtime.gradient_program());
        let residual_gradient_slot_count = residual_gradient_program
            .as_ref()
            .map(|program| program.scratch_slots())
            .unwrap_or_else(|| state.expression_ir.node_count());
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
                lowered_artifacts
                    .as_ref()
                    .and_then(|artifacts| {
                        self.evaluate_cached_weighted_gradient_sum_lowered(
                            &state,
                            artifacts,
                            &amplitude_values,
                            &amplitude_gradients,
                            grad_dim,
                        )
                    })
                    .unwrap_or_else(|| {
                        self.evaluate_cached_weighted_gradient_sum_ir(
                            &state,
                            &amplitude_values,
                            &amplitude_gradients,
                            grad_dim,
                        )
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
                .zip(self.dataset.weights_local().par_iter())
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; residual_gradient_slot_count],
                            vec![Complex64::ZERO; residual_gradient_slot_count * grad_dim],
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
                        let gradient = residual_gradient_program
                            .as_ref()
                            .map(|program| {
                                program.evaluate_gradient_into_flat(
                                    amplitude_values,
                                    gradient_values,
                                    value_slots,
                                    gradient_slots,
                                    grad_dim,
                                )
                            })
                            .unwrap_or_else(|| {
                                self.evaluate_residual_gradient_ir(
                                    &state,
                                    amplitude_values,
                                    gradient_values,
                                    grad_dim,
                                )
                            });
                        gradient.map(|value| value.re).scale(*event)
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
            let mut value_slots = vec![Complex64::ZERO; residual_gradient_slot_count];
            let mut gradient_slots = vec![Complex64::ZERO; residual_gradient_slot_count * grad_dim];
            resources
                .caches
                .iter()
                .zip(self.dataset.weights_local().iter())
                .map(|(cache, event)| {
                    self.fill_amplitude_values_and_gradients(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &residual_active_indices,
                        &residual_active_mask,
                        &parameters,
                        cache,
                    );
                    let gradient = residual_gradient_program
                        .as_ref()
                        .map(|program| {
                            program.evaluate_gradient_into_flat(
                                &amplitude_values,
                                &gradient_values,
                                &mut value_slots,
                                &mut gradient_slots,
                                grad_dim,
                            )
                        })
                        .unwrap_or_else(|| {
                            self.evaluate_residual_gradient_ir(
                                &state,
                                &amplitude_values,
                                &gradient_values,
                                grad_dim,
                            )
                        });
                    gradient.map(|value| value.re).scale(*event)
                })
                .sum()
        };
        Ok((residual_sum, cached_term_sum))
    }

    /// Weighted sum over local events of the real gradient of the expression.
    ///
    /// This returns `sum_e(weight_e * Re(dL_e/dp))` for all free parameters.
    pub fn evaluate_weighted_gradient_sum_local(
        &self,
        parameters: &[f64],
    ) -> LadduResult<DVector<f64>> {
        let (residual_sum, cached_term_sum) =
            self.evaluate_weighted_gradient_sum_local_components(parameters)?;
        Ok(residual_sum + cached_term_sum)
    }

    #[cfg(feature = "mpi")]
    /// Weighted sum over all ranks of the real gradient of the expression.
    ///
    /// This returns `sum_{r,e}(weight_{r,e} * Re(dL_{r,e}/dp))`.
    pub fn evaluate_weighted_gradient_sum_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> LadduResult<DVector<f64>> {
        let (residual_sum_local, cached_term_sum_local) =
            self.evaluate_weighted_gradient_sum_local_components(parameters)?;
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
        Ok(total)
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

    /// Get the parameters used by this evaluator.
    pub fn parameters(&self) -> ParameterMap {
        self.resources.read().parameters()
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
    pub fn n_parameters(&self) -> usize {
        self.resources.read().n_parameters()
    }

    pub fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.resources.read().fix_parameter(name, value)
    }

    pub fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.resources.read().free_parameter(name)
    }

    pub fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<()> {
        self.resources.write().rename_parameter(old, new)
    }

    pub fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.resources.write().rename_parameters(mapping)
    }

    /// Activate [`Amplitude`] use-sites by tag or glob selector, skipping missing selectors.
    pub fn activate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().activate(name);
        self.refresh_runtime_specializations();
    }
    /// Activate [`Amplitude`] use-sites by tag or glob selector and return an error if no use-site matches.
    pub fn activate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().activate_strict(name)?;
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Activate several [`Amplitude`] use-sites by tag or glob selector, skipping missing selectors.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().activate_many(names);
        self.refresh_runtime_specializations();
    }
    /// Activate several [`Amplitude`] use-sites by tag or glob selector and return an error if any selector has no matches.
    pub fn activate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().activate_many_strict(names)?;
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Activate all registered [`Amplitude`]s.
    pub fn activate_all(&self) {
        self.resources.write().activate_all();
        self.refresh_runtime_specializations();
    }

    /// Deactivate [`Amplitude`] use-sites by tag or glob selector, skipping missing selectors.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().deactivate(name);
        self.refresh_runtime_specializations();
    }

    /// Deactivate [`Amplitude`] use-sites by tag or glob selector and return an error if no use-site matches.
    pub fn deactivate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().deactivate_strict(name)?;
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Deactivate several [`Amplitude`] use-sites by tag or glob selector, skipping missing selectors.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().deactivate_many(names);
        self.refresh_runtime_specializations();
    }
    /// Deactivate several [`Amplitude`] use-sites by tag or glob selector and return an error if any selector has no matches.
    pub fn deactivate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().deactivate_many_strict(names)?;
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Deactivate all tagged [`Amplitude`] use-sites.
    pub fn deactivate_all(&self) {
        self.resources.write().deactivate_all();
        self.refresh_runtime_specializations();
    }

    /// Isolate [`Amplitude`] use-sites by tag or glob selector, skipping missing selectors.
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.resources.write().isolate(name);
        self.refresh_runtime_specializations();
    }

    /// Isolate [`Amplitude`] use-sites by tag or glob selector and return an error if no use-site matches.
    pub fn isolate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.resources.write().isolate_strict(name)?;
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Isolate several [`Amplitude`] use-sites by tag or glob selector, skipping missing selectors.
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.resources.write().isolate_many(names);
        self.refresh_runtime_specializations();
    }

    /// Isolate several [`Amplitude`] use-sites by tag or glob selector and return an error if any selector has no matches.
    pub fn isolate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.resources.write().isolate_many_strict(names)?;
        self.refresh_runtime_specializations();
        Ok(())
    }

    /// Return a copy of the current active-amplitude mask.
    pub fn active_mask(&self) -> Vec<bool> {
        self.resources.read().active.clone()
    }

    /// Apply a precomputed active-amplitude mask. Untagged use-sites cannot be deactivated.
    pub fn set_active_mask(&self, mask: &[bool]) -> LadduResult<()> {
        let resources = {
            let mut resources = self.resources.write();
            if mask.len() != resources.active.len() {
                return Err(LadduError::LengthMismatch {
                    context: "active amplitude mask".to_string(),
                    expected: resources.active.len(),
                    actual: mask.len(),
                });
            }
            resources.apply_active_mask(mask)?;
            resources.clone()
        };
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
    pub fn evaluate_local(&self, parameters: &[f64]) -> LadduResult<Vec<Complex64>> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_value_slot_count();
        let program_snapshot = self.expression_value_program_snapshot();
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
                        self.evaluate_expression_value_with_program_snapshot(
                            &program_snapshot,
                            amplitude_values,
                            expr_slots,
                        )
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
                    self.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        &amplitude_values,
                        &mut expr_slots,
                    )
                })
                .collect())
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
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let active_indices = active_mask
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| if active { Some(index) } else { None })
            .collect::<Vec<_>>();
        let program_snapshot =
            self.expression_value_program_snapshot_for_active_mask(active_mask)?;
        let slot_count = self.expression_value_program_snapshot_slot_count(&program_snapshot);
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
                        self.evaluate_expression_value_with_program_snapshot(
                            &program_snapshot,
                            amplitude_values,
                            expr_slots,
                        )
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
                    self.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        &amplitude_values,
                        &mut expr_slots,
                    )
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
        let parameters = resources
            .parameter_map
            .assemble(parameters)
            .expect("parameter slice must match evaluator resources");
        let amplitude_len = self.amplitude_use_sites.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_value_slot_count();
        let program_snapshot = self.expression_value_program_snapshot();
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
                                self.evaluate_expression_value_with_program_snapshot(
                                    &program_snapshot,
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
                    self.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        amplitude_values,
                        expr_slots,
                    )
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
    fn evaluate_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> LadduResult<Vec<Complex64>> {
        let local_evaluation = self.evaluate_local(parameters)?;
        let n_events = self.dataset.n_events();
        let mut buffer: Vec<Complex64> = vec![Complex64::ZERO; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required here because the public MPI API returns full per-event outputs.
            // Do not replace with all-reduce unless semantics change to scalar aggregates only.
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_evaluation, &mut partitioned_buffer);
        }
        Ok(buffer)
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
    pub fn evaluate(&self, parameters: &[f64]) -> LadduResult<Vec<Complex64>> {
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
    pub fn evaluate_batch_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> LadduResult<Vec<Complex64>> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let active_indices = resources.active_indices().to_vec();
        let slot_count = self.expression_value_slot_count();
        let program_snapshot = self.expression_value_program_snapshot();
        #[cfg(feature = "rayon")]
        {
            Ok(indices
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
                        self.evaluate_expression_value_with_program_snapshot(
                            &program_snapshot,
                            amplitude_values,
                            expr_slots,
                        )
                    },
                )
                .collect())
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
            let mut expr_slots = vec![Complex64::ZERO; slot_count];
            Ok(indices
                .iter()
                .map(|&idx| {
                    let cache = &resources.caches[idx];
                    self.fill_amplitude_values(
                        &mut amplitude_values,
                        &active_indices,
                        &parameters,
                        cache,
                    );
                    self.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        &amplitude_values,
                        &mut expr_slots,
                    )
                })
                .collect())
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
    ) -> LadduResult<Vec<Complex64>> {
        let total = self.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let local_evaluation = self.evaluate_batch_local(parameters, &locals)?;
        Ok(world.all_gather_batched_partitioned(&local_evaluation, indices, total, None))
    }

    /// Evaluate the stored [`Expression`] over a subset of events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters. See also [`Evaluator::evaluate`].
    pub fn evaluate_batch(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> LadduResult<Vec<Complex64>> {
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
    pub fn evaluate_gradient_local(
        &self,
        parameters: &[f64],
    ) -> LadduResult<Vec<DVector<Complex64>>> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let lowered_runtime = self.lowered_runtime();
        let gradient_program = lowered_runtime.gradient_program();
        let slot_count = self.expression_gradient_slot_count();
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
                            vec![Complex64::ZERO; slot_count * grad_dim],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), cache| {
                        self.fill_amplitude_values_and_gradients(
                            amplitude_values,
                            gradient_values,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        );
                        gradient_program.evaluate_gradient_into_flat(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            grad_dim,
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
            let mut gradient_slots = vec![Complex64::ZERO; slot_count * grad_dim];
            Ok(resources
                .caches
                .iter()
                .map(|cache| {
                    self.fill_amplitude_values_and_gradients(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    );
                    gradient_program.evaluate_gradient_into_flat(
                        &amplitude_values,
                        &gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        grad_dim,
                    )
                })
                .collect())
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
        let parameters = resources
            .parameter_map
            .assemble(parameters)
            .expect("parameter slice must match evaluator resources");
        let amplitude_len = self.amplitude_use_sites.len();
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
    ) -> LadduResult<Vec<DVector<Complex64>>> {
        let local_evaluation = self.evaluate_gradient_local(parameters)?;
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
        Ok(buffer
            .chunks(parameters.len())
            .map(DVector::from_row_slice)
            .collect())
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
    pub fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<Vec<DVector<Complex64>>> {
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
    ) -> LadduResult<Vec<DVector<Complex64>>> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let lowered_runtime = self.lowered_runtime();
        let gradient_program = lowered_runtime.gradient_program();
        let slot_count = self.expression_gradient_slot_count();
        #[cfg(feature = "rayon")]
        {
            Ok(indices
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![Complex64::ZERO; slot_count * grad_dim],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), &idx| {
                        let cache = &resources.caches[idx];
                        self.fill_amplitude_values_and_gradients(
                            amplitude_values,
                            gradient_values,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        );
                        gradient_program.evaluate_gradient_into_flat(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            grad_dim,
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
            let mut gradient_slots = vec![Complex64::ZERO; slot_count * grad_dim];
            Ok(indices
                .iter()
                .map(|&idx| {
                    let cache = &resources.caches[idx];
                    self.fill_amplitude_values_and_gradients(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    );
                    gradient_program.evaluate_gradient_into_flat(
                        &amplitude_values,
                        &gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        grad_dim,
                    )
                })
                .collect())
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
    ) -> LadduResult<Vec<DVector<Complex64>>> {
        let total = self.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let flattened_local_evaluation = self
            .evaluate_gradient_batch_local(parameters, &locals)?
            .iter()
            .flat_map(|g| g.data.as_vec().to_vec())
            .collect::<Vec<Complex64>>();
        Ok(world
            .all_gather_batched_partitioned(
                &flattened_local_evaluation,
                indices,
                total,
                Some(parameters.len()),
            )
            .chunks(parameters.len())
            .map(DVector::from_row_slice)
            .collect())
    }

    /// Evaluate the gradient of the stored [`Expression`] over a subset of the
    /// events in the [`Dataset`] stored by the [`Evaluator`] with the given values
    /// for free parameters. See also [`Evaluator::evaluate_gradient`].
    pub fn evaluate_gradient_batch(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> LadduResult<Vec<DVector<Complex64>>> {
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
    ) -> LadduResult<Vec<(Complex64, DVector<Complex64>)>> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let lowered_runtime = self.lowered_runtime();
        let value_gradient_program = lowered_runtime.value_gradient_program();
        let slot_count = self.expression_value_gradient_slot_count();
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
                            vec![Complex64::ZERO; slot_count * grad_dim],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), cache| {
                        self.fill_amplitude_values_and_gradients(
                            amplitude_values,
                            gradient_values,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        );
                        value_gradient_program.evaluate_value_gradient_into_flat(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            grad_dim,
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
            let mut gradient_slots = vec![Complex64::ZERO; slot_count * grad_dim];
            Ok(resources
                .caches
                .iter()
                .map(|cache| {
                    self.fill_amplitude_values_and_gradients(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    );
                    value_gradient_program.evaluate_value_gradient_into_flat(
                        &amplitude_values,
                        &gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        grad_dim,
                    )
                })
                .collect())
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
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let grad_dim = parameters.len();
        let active_indices = active_mask
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| if active { Some(index) } else { None })
            .collect::<Vec<_>>();
        let lowered_runtime = self.lower_expression_runtime_for_active_mask(active_mask)?;
        let slot_count = lowered_runtime.value_gradient_program().scratch_slots();
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
                            vec![Complex64::ZERO; slot_count * grad_dim],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), cache| {
                        self.fill_amplitude_values_and_gradients(
                            amplitude_values,
                            gradient_values,
                            &active_indices,
                            active_mask,
                            &parameters,
                            cache,
                        );
                        lowered_runtime
                            .value_gradient_program()
                            .evaluate_value_gradient_into_flat(
                                amplitude_values,
                                gradient_values,
                                value_slots,
                                gradient_slots,
                                grad_dim,
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
            let mut gradient_slots = vec![Complex64::ZERO; slot_count * grad_dim];
            Ok(resources
                .caches
                .iter()
                .map(|cache| {
                    self.fill_amplitude_values_and_gradients(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &active_indices,
                        active_mask,
                        &parameters,
                        cache,
                    );
                    lowered_runtime
                        .value_gradient_program()
                        .evaluate_value_gradient_into_flat(
                            &amplitude_values,
                            &gradient_values,
                            &mut value_slots,
                            &mut gradient_slots,
                            grad_dim,
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
    ) -> LadduResult<Vec<(Complex64, DVector<Complex64>)>> {
        let resources = self.resources.read();
        let parameters = resources.parameter_map.assemble(parameters)?;
        let amplitude_len = self.amplitude_use_sites.len();
        let grad_dim = parameters.len();
        let active_indices = resources.active_indices().to_vec();
        let lowered_runtime = self.lowered_runtime();
        let value_gradient_program = lowered_runtime.value_gradient_program();
        let slot_count = self.expression_value_gradient_slot_count();
        #[cfg(feature = "rayon")]
        {
            Ok(indices
                .par_iter()
                .map_init(
                    || {
                        (
                            vec![Complex64::ZERO; amplitude_len],
                            vec![DVector::zeros(grad_dim); amplitude_len],
                            vec![Complex64::ZERO; slot_count],
                            vec![Complex64::ZERO; slot_count * grad_dim],
                        )
                    },
                    |(amplitude_values, gradient_values, value_slots, gradient_slots), &idx| {
                        let cache = &resources.caches[idx];
                        self.fill_amplitude_values_and_gradients(
                            amplitude_values,
                            gradient_values,
                            &active_indices,
                            &resources.active,
                            &parameters,
                            cache,
                        );
                        value_gradient_program.evaluate_value_gradient_into_flat(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                            grad_dim,
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
            let mut gradient_slots = vec![Complex64::ZERO; slot_count * grad_dim];
            Ok(indices
                .iter()
                .map(|&idx| {
                    let cache = &resources.caches[idx];
                    self.fill_amplitude_values_and_gradients(
                        &mut amplitude_values,
                        &mut gradient_values,
                        &active_indices,
                        &resources.active,
                        &parameters,
                        cache,
                    );
                    value_gradient_program.evaluate_value_gradient_into_flat(
                        &amplitude_values,
                        &gradient_values,
                        &mut value_slots,
                        &mut gradient_slots,
                        grad_dim,
                    )
                })
                .collect())
        }
    }
}

#[cfg(test)]
mod tests;
