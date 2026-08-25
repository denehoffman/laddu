//! High-level cross-section analyses and uncertainty propagation.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use auto_ops::impl_op_ex;
use laddu_data::data::Dataset;
use laddu_expr::{Expr, ExprNodeStructuralKey};
use laddu_runtime::{DatasetExprExt, Execution};

use crate::{CrossSectionIntegrals, Likelihood, LikelihoodError, LikelihoodResult};

static NEXT_SOURCE_ID: AtomicU64 = AtomicU64::new(1);

/// Returns a process-local identifier for an independent uncertainty source.
pub fn next_uncertainty_source_id() -> u64 {
    NEXT_SOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

fn invalid(message: impl Into<String>) -> LikelihoodError {
    LikelihoodError::InvalidCrossSection(message.into())
}

#[cfg(test)]
thread_local! {
    static SELECTION_INTENSITY_EVALUATIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static PREPARED_INTENSITY_EVALUATIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static BIN_ASSIGNMENT_EVALUATIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

fn record_selection_intensity_evaluation() {
    #[cfg(test)]
    SELECTION_INTENSITY_EVALUATIONS.with(|count| count.set(count.get() + 1));
}

fn record_prepared_intensity_evaluation() {
    #[cfg(test)]
    PREPARED_INTENSITY_EVALUATIONS.with(|count| count.set(count.get() + 1));
}

fn record_bin_assignment_evaluation() {
    #[cfg(test)]
    BIN_ASSIGNMENT_EVALUATIONS.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn reset_selection_intensity_evaluation_count() {
    SELECTION_INTENSITY_EVALUATIONS.with(|count| count.set(0));
}

#[cfg(test)]
fn selection_intensity_evaluation_count() -> usize {
    SELECTION_INTENSITY_EVALUATIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn reset_projection_evaluation_counts() {
    PREPARED_INTENSITY_EVALUATIONS.with(|count| count.set(0));
    BIN_ASSIGNMENT_EVALUATIONS.with(|count| count.set(0));
}

#[cfg(test)]
fn projection_evaluation_counts() -> (usize, usize) {
    let intensities = PREPARED_INTENSITY_EVALUATIONS.with(std::cell::Cell::get);
    let assignments = BIN_ASSIGNMENT_EVALUATIONS.with(std::cell::Cell::get);
    (intensities, assignments)
}

/// Named parameter draws with optional paired bootstrap likelihood replicas.
#[derive(Clone)]
pub struct Ensemble {
    parameter_names: Vec<String>,
    draws: Vec<Vec<f64>>,
    source_id: u64,
    replicas: Vec<Arc<Likelihood>>,
    replicas_share_event_rows: bool,
}

/// Failure while constructing a paired bootstrap-fit ensemble.
#[derive(Debug, thiserror::Error)]
pub enum BootstrapFitError<E> {
    /// A likelihood replica or final ensemble could not be prepared.
    #[error(transparent)]
    Likelihood(#[from] LikelihoodError),
    /// The user-supplied fit operation failed for one replica.
    #[error("bootstrap fit {index} failed: {source}")]
    Fit {
        /// Zero-based bootstrap replica index.
        index: usize,
        /// Fit error returned by the supplied operation.
        #[source]
        source: E,
    },
}

impl std::fmt::Debug for Ensemble {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Ensemble")
            .field("parameter_names", &self.parameter_names)
            .field("draws", &self.draws)
            .field("source_id", &self.source_id)
            .field("replicas", &self.replicas.len())
            .field("replicas_share_event_rows", &self.replicas_share_event_rows)
            .finish()
    }
}

impl Ensemble {
    /// Constructs an ensemble from rows of free-parameter values.
    ///
    /// # Errors
    /// Returns an error for empty, non-finite, or incorrectly sized draws.
    pub fn new(parameter_names: Vec<String>, draws: Vec<Vec<f64>>) -> LikelihoodResult<Self> {
        Self::with_source_id(parameter_names, draws, next_uncertainty_source_id())
    }

    /// Constructs an ensemble with an explicit correlation/provenance ID.
    ///
    /// # Errors
    /// Returns an error for empty, non-finite, or incorrectly sized draws.
    pub fn with_source_id(
        parameter_names: Vec<String>,
        draws: Vec<Vec<f64>>,
        source_id: u64,
    ) -> LikelihoodResult<Self> {
        if draws.is_empty() {
            return Err(invalid("an ensemble must contain at least one draw"));
        }
        if draws
            .iter()
            .any(|draw| draw.len() != parameter_names.len() || draw.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "every ensemble draw must be finite and match the parameter-name count",
            ));
        }
        Ok(Self {
            parameter_names,
            draws,
            source_id,
            replicas: Vec::new(),
            replicas_share_event_rows: false,
        })
    }

    /// Constructs a paired bootstrap ensemble.
    ///
    /// # Errors
    /// Returns an error when draws are invalid or replica counts differ.
    pub fn with_replicas(
        parameter_names: Vec<String>,
        draws: Vec<Vec<f64>>,
        replicas: Vec<Arc<Likelihood>>,
    ) -> LikelihoodResult<Self> {
        let mut ensemble = Self::new(parameter_names, draws)?;
        if replicas.len() != ensemble.draws.len() {
            return Err(invalid(
                "bootstrap replica count must match the parameter draw count",
            ));
        }
        ensemble.replicas = replicas;
        Ok(ensemble)
    }

    /// Flattens a `(walkers, steps, parameters)` chain after burn-in and thinning.
    ///
    /// # Errors
    /// Returns an error for invalid thinning, discard, or draw shapes.
    pub fn from_chain(
        parameter_names: Vec<String>,
        chain: &[Vec<Vec<f64>>],
        discard: usize,
        thin: usize,
    ) -> LikelihoodResult<Self> {
        if thin == 0 {
            return Err(invalid("MCMC thinning must be positive"));
        }
        if chain.is_empty()
            || chain
                .iter()
                .any(|walker| discard >= walker.len() || walker.is_empty())
        {
            return Err(invalid(
                "MCMC discard must leave at least one step in every walker",
            ));
        }
        let draws = chain
            .iter()
            .flat_map(|walker| {
                (discard..walker.len())
                    .step_by(thin)
                    .map(|step| walker[step].clone())
            })
            .collect();
        Self::new(parameter_names, draws)
    }

    /// Poisson-bootstraps a likelihood and fits every paired replica.
    ///
    /// The callback receives the prepared replica and its zero-based index.
    /// Its returned free-parameter vector is retained beside that exact
    /// likelihood, ensuring later cross-section evaluations use the matching
    /// resampled dataset.
    ///
    /// # Errors
    /// Returns an error when replica preparation, fitting, or validation fails.
    pub fn bootstrap_fit<E>(
        likelihood: &Arc<Likelihood>,
        samples: usize,
        seed: u64,
        mut fit: impl FnMut(&Arc<Likelihood>, usize) -> Result<Vec<f64>, E>,
    ) -> Result<Self, BootstrapFitError<E>> {
        if samples == 0 {
            return Err(BootstrapFitError::Likelihood(invalid(
                "bootstrap sample count must be positive",
            )));
        }
        let parameter_names = likelihood
            .params()
            .free_params()
            .iter()
            .map(|id| likelihood.params().name(*id).map(str::to_owned))
            .collect::<Result<Vec<_>, _>>()
            .map_err(LikelihoodError::from)?;
        let mut draws = Vec::with_capacity(samples);
        let mut replicas = Vec::with_capacity(samples);
        for index in 0..samples {
            let replica = Arc::new(likelihood.bootstrap(seed.wrapping_add(index as u64))?);
            let draw =
                fit(&replica, index).map_err(|source| BootstrapFitError::Fit { index, source })?;
            draws.push(draw);
            replicas.push(replica);
        }
        let mut ensemble = Self::with_replicas(parameter_names, draws, replicas)?;
        ensemble.replicas_share_event_rows = true;
        Ok(ensemble)
    }

    /// Parameter names in draw-column order.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Parameter draw rows.
    pub fn draws(&self) -> &[Vec<f64>] {
        &self.draws
    }

    /// Correlation/provenance identifier.
    pub fn source_id(&self) -> u64 {
        self.source_id
    }

    /// Paired bootstrap likelihood replicas, if present.
    pub fn replicas(&self) -> &[Arc<Likelihood>] {
        &self.replicas
    }

    fn replica_bin_assignments(
        &self,
        dataset: Option<&Dataset>,
        axes: &[Axis],
        execution: &Execution,
    ) -> LikelihoodResult<Option<BinAssignments>> {
        if self.replicas_share_event_rows {
            Ok(None)
        } else {
            dataset
                .map(|dataset| evaluate_bin_assignments(dataset, axes, execution))
                .transpose()
        }
    }

    /// Number of draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether no draws are present.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }
}

/// A central scalar estimate with optional uncertainty draws.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    central: f64,
    draws: Vec<f64>,
    source_id: Option<u64>,
}

impl Estimate {
    /// Constructs a central-only estimate.
    ///
    /// # Errors
    /// Returns an error when the central value is not finite.
    pub fn central(central: f64) -> LikelihoodResult<Self> {
        Self::with_source_id(central, Vec::new(), None)
    }

    /// Constructs an estimate with an independent uncertainty source.
    ///
    /// # Errors
    /// Returns an error when the central value or a draw is not finite.
    pub fn new(central: f64, draws: Vec<f64>) -> LikelihoodResult<Self> {
        let source_id = (!draws.is_empty()).then(next_uncertainty_source_id);
        Self::with_source_id(central, draws, source_id)
    }

    /// Constructs an estimate with an explicit correlation/provenance ID.
    ///
    /// # Errors
    /// Returns an error when the central value or a draw is not finite.
    pub fn with_source_id(
        central: f64,
        draws: Vec<f64>,
        source_id: Option<u64>,
    ) -> LikelihoodResult<Self> {
        if !central.is_finite() || draws.iter().any(|value| !value.is_finite()) {
            return Err(invalid("estimate central value and draws must be finite"));
        }
        Ok(Self {
            central,
            draws,
            source_id,
        })
    }

    fn from_evaluation(central: f64, draws: Vec<f64>, source_id: Option<u64>) -> Self {
        Self {
            central,
            draws,
            source_id,
        }
    }

    /// Central estimate.
    pub fn value(&self) -> f64 {
        self.central
    }

    /// Uncertainty draws.
    pub fn draws(&self) -> &[f64] {
        &self.draws
    }

    /// Correlation/provenance identifier.
    pub fn source_id(&self) -> Option<u64> {
        self.source_id
    }

    /// Draw mean.
    ///
    /// # Errors
    /// Returns an error when there are no uncertainty draws.
    pub fn mean(&self) -> LikelihoodResult<f64> {
        if self.draws.is_empty() {
            return Err(invalid("estimate has no uncertainty draws"));
        }
        Ok(self.draws.iter().sum::<f64>() / self.draws.len() as f64)
    }

    /// Draw standard deviation using Bessel's correction.
    ///
    /// # Errors
    /// Returns an error when fewer than two draws are available.
    pub fn std(&self) -> LikelihoodResult<f64> {
        if self.draws.len() < 2 {
            return Err(invalid("estimate needs at least two uncertainty draws"));
        }
        let mean = self.mean()?;
        Ok((self
            .draws
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (self.draws.len() - 1) as f64)
            .sqrt())
    }

    /// Linearly interpolated draw quantile.
    ///
    /// # Errors
    /// Returns an error for an invalid probability or absent draws.
    pub fn quantile(&self, probability: f64) -> LikelihoodResult<f64> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(invalid("quantile probability must lie in [0, 1]"));
        }
        if self.draws.is_empty() {
            return Err(invalid("estimate has no uncertainty draws"));
        }
        let mut values = self.draws.clone();
        values.sort_by(f64::total_cmp);
        let position = probability * (values.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let fraction = position - lower as f64;
        Ok(values[lower] * (1.0 - fraction) + values[upper] * fraction)
    }

    /// Median draw.
    ///
    /// # Errors
    /// Returns an error when there are no uncertainty draws.
    pub fn median(&self) -> LikelihoodResult<f64> {
        self.quantile(0.5)
    }

    /// Equal-tailed uncertainty interval.
    ///
    /// # Errors
    /// Returns an error for an invalid level or absent draws.
    pub fn interval(&self, level: f64) -> LikelihoodResult<(f64, f64)> {
        if !(0.0 < level && level < 1.0) {
            return Err(invalid("interval level must lie in (0, 1)"));
        }
        let tail = (1.0 - level) * 0.5;
        Ok((self.quantile(tail)?, self.quantile(1.0 - tail)?))
    }

    fn binary(&self, other: &Self, op: impl Fn(f64, f64) -> f64) -> Self {
        let count = match (self.draws.len(), other.draws.len()) {
            (0, 0) => 0,
            (0, right) => right,
            (left, 0) => left,
            (left, right) => left.min(right),
        };
        let draws = (0..count)
            .map(|index| {
                let left = self.draws.get(index).copied().unwrap_or(self.central);
                let right_index = if self.source_id == other.source_id {
                    index
                } else {
                    (index.wrapping_mul(6364136223846793005usize).wrapping_add(1))
                        % other.draws.len().max(1)
                };
                let right = other
                    .draws
                    .get(right_index)
                    .copied()
                    .unwrap_or(other.central);
                op(left, right)
            })
            .collect();
        let source_id = match (self.draws.is_empty(), other.draws.is_empty()) {
            (false, true) => self.source_id,
            (true, false) => other.source_id,
            (false, false) if self.source_id == other.source_id => self.source_id,
            (false, false) => Some(next_uncertainty_source_id()),
            (true, true) => None,
        };
        Self::from_evaluation(op(self.central, other.central), draws, source_id)
    }
}

impl_op_ex!(+ |left: &Estimate, right: &Estimate| -> Estimate {
    left.binary(right, |a, b| a + b)
});
impl_op_ex!(-|left: &Estimate, right: &Estimate| -> Estimate { left.binary(right, |a, b| a - b) });
impl_op_ex!(*|left: &Estimate, right: &Estimate| -> Estimate { left.binary(right, |a, b| a * b) });
impl_op_ex!(/ |left: &Estimate, right: &Estimate| -> Estimate {
    left.binary(right, |a, b| a / b)
});

impl_op_ex!(+ |left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a + b,
    )
});
impl_op_ex!(-|left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a - b,
    )
});
impl_op_ex!(*|left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a * b,
    )
});
impl_op_ex!(/ |left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a / b,
    )
});

/// An expression and monotonically increasing bin edges.
#[derive(Clone, Debug)]
pub struct Axis {
    expression: Expr,
    edges: Vec<f64>,
}

impl Axis {
    /// Constructs a differential axis.
    ///
    /// # Errors
    /// Returns an error unless edges are finite and strictly increasing.
    pub fn new(expression: Expr, edges: Vec<f64>) -> LikelihoodResult<Self> {
        if edges.len() < 2
            || edges.iter().any(|value| !value.is_finite())
            || edges.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "axis edges must contain at least two finite increasing values",
            ));
        }
        Ok(Self { expression, edges })
    }

    /// Axis expression.
    pub fn expression(&self) -> &Expr {
        &self.expression
    }

    /// Bin edges.
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// Number of bins.
    pub fn bins(&self) -> usize {
        self.edges.len() - 1
    }
}

/// Central bin values with optional uncertainty draws.
#[derive(Clone, Debug, PartialEq)]
pub struct BinnedEstimate {
    central: Vec<f64>,
    draws: Vec<Vec<f64>>,
}

impl BinnedEstimate {
    fn new(central: Vec<f64>, draws: Vec<Vec<f64>>) -> Self {
        Self { central, draws }
    }

    /// Central flattened bin values.
    pub fn values(&self) -> &[f64] {
        &self.central
    }

    /// Flattened bin values for every uncertainty draw.
    pub fn draws(&self) -> &[Vec<f64>] {
        &self.draws
    }

    /// Equal-tailed interval for every bin.
    ///
    /// # Errors
    /// Returns an error for an invalid level or absent draws.
    pub fn interval(&self, level: f64) -> LikelihoodResult<(Vec<f64>, Vec<f64>)> {
        if self.draws.is_empty() {
            return Err(invalid("binned estimate has no uncertainty draws"));
        }
        let mut lower = Vec::with_capacity(self.central.len());
        let mut upper = Vec::with_capacity(self.central.len());
        for bin in 0..self.central.len() {
            let estimate = Estimate::from_evaluation(
                self.central[bin],
                self.draws.iter().map(|draw| draw[bin]).collect(),
                None,
            );
            let interval = estimate.interval(level)?;
            lower.push(interval.0);
            upper.push(interval.1);
        }
        Ok((lower, upper))
    }

    /// Sample covariance matrix between flattened bins.
    ///
    /// # Errors
    /// Returns an error when fewer than two draws are available.
    pub fn covariance(&self) -> LikelihoodResult<Vec<Vec<f64>>> {
        if self.draws.len() < 2 {
            return Err(invalid(
                "binned estimate needs at least two uncertainty draws",
            ));
        }
        let count = self.draws.len() as f64;
        let means: Vec<_> = (0..self.central.len())
            .map(|bin| self.draws.iter().map(|draw| draw[bin]).sum::<f64>() / count)
            .collect();
        Ok((0..self.central.len())
            .map(|left| {
                (0..self.central.len())
                    .map(|right| {
                        self.draws
                            .iter()
                            .map(|draw| (draw[left] - means[left]) * (draw[right] - means[right]))
                            .sum::<f64>()
                            / (count - 1.0)
                    })
                    .collect()
            })
            .collect())
    }
}

/// Data, coherent-model, and tagged-component differential cross sections.
#[derive(Clone, Debug)]
pub struct DifferentialCrossSection {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    data: BinnedEstimate,
    model: BinnedEstimate,
    components: HashMap<String, BinnedEstimate>,
}

type DifferentialValues = (Vec<f64>, Vec<f64>, HashMap<String, Vec<f64>>);

impl DifferentialCrossSection {
    /// Edge arrays for every differential axis.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }

    /// Multidimensional bin shape; values are flattened in row-major order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Acceptance-corrected observed distribution.
    pub fn data(&self) -> &BinnedEstimate {
        &self.data
    }

    /// Coherent fitted-model distribution.
    pub fn model(&self) -> &BinnedEstimate {
        &self.model
    }

    /// Tagged, separately evaluated component distributions.
    pub fn components(&self) -> &HashMap<String, BinnedEstimate> {
        &self.components
    }
}

/// One named differential cross section within a projection-set request.
#[derive(Clone, Debug)]
pub struct Projection {
    name: String,
    axes: Vec<Axis>,
}

impl Projection {
    /// Constructs a named projection specification.
    ///
    /// # Errors
    /// Returns an error when the name or axis group is empty.
    pub fn new(name: impl Into<String>, axes: Vec<Axis>) -> LikelihoodResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(invalid("projection names must not be empty"));
        }
        if axes.is_empty() {
            return Err(invalid("each projection must contain at least one axis"));
        }
        Ok(Self { name, axes })
    }

    /// Public projection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Axes forming this entry's joint differential cross section.
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
}

/// Ordered results from a multi-projection cross-section request.
#[derive(Clone, Debug)]
pub struct ProjectionSet {
    entries: Vec<(String, DifferentialCrossSection)>,
}

impl ProjectionSet {
    /// Number of named projection results.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the result contains no projections.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up a projection result by its public name.
    pub fn get(&self, name: &str) -> Option<&DifferentialCrossSection> {
        self.entries
            .iter()
            .find_map(|(candidate, result)| (candidate == name).then_some(result))
    }

    /// Iterates over projection names and results in request order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &DifferentialCrossSection)> {
        self.entries
            .iter()
            .map(|(name, result)| (name.as_str(), result))
    }
}

/// A prepared total, tagged, differential, and combinable cross-section analysis.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CanonicalTags(Vec<String>);

impl CanonicalTags {
    fn new(tags: &[String]) -> Self {
        let mut canonical = tags.to_vec();
        canonical.sort();
        canonical.dedup();
        Self(canonical)
    }

    fn as_slice(&self) -> &[String] {
        &self.0
    }
}

type IntegralCacheKey = (usize, Option<CanonicalTags>);
type IntegralCache = Arc<Mutex<HashMap<IntegralCacheKey, CrossSectionIntegrals>>>;

/// A prepared total, tagged, differential, and combinable cross-section analysis.
#[derive(Clone)]
pub struct CrossSection {
    likelihood: Arc<Likelihood>,
    term_name: String,
    generated_mc: Dataset,
    full_integrals: CrossSectionIntegrals,
    luminosity: f64,
    parameters: Vec<f64>,
    ensemble: Option<Ensemble>,
    members: Option<Arc<Vec<(CrossSection, Estimate)>>>,
    integral_cache: IntegralCache,
    cache_hits: Arc<AtomicU64>,
    cache_misses: Arc<AtomicU64>,
}

/// Integral-preparation cache statistics for a cross-section analysis.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CrossSectionDiagnostics {
    cache_hits: u64,
    cache_misses: u64,
    cached_integrals: usize,
    prepared_bytes: usize,
}

impl CrossSectionDiagnostics {
    /// Returns successful integral-cache lookups.
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }
    /// Returns integral preparations caused by cache misses.
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses
    }
    /// Returns the number of unique likelihood/tag integral records retained.
    pub fn cached_integrals(&self) -> usize {
        self.cached_integrals
    }
    /// Returns the summed prepared bytes reported by cached integral records.
    pub fn prepared_bytes(&self) -> usize {
        self.prepared_bytes
    }
}

#[derive(Clone)]
struct BinnedMeasurement {
    yields: Vec<f64>,
    exposures: Vec<f64>,
}

struct CombinedMemberValues {
    data: BinnedMeasurement,
    model: BinnedMeasurement,
    components: HashMap<String, BinnedMeasurement>,
}

struct CanonicalComponents {
    aliases: HashMap<String, CanonicalTags>,
    integrals: HashMap<CanonicalTags, CrossSectionIntegrals>,
}

struct CombinedMemberWorkspace {
    luminosity: f64,
    full: CrossSectionIntegrals,
    components: CanonicalComponents,
    data_bins: BinAssignments,
    data_weights: Vec<f64>,
    accepted_bins: BinAssignments,
    accepted_weights: Vec<f64>,
    generated_bins: BinAssignments,
    generated_weights: Vec<f64>,
}

impl CanonicalComponents {
    fn prepare(
        member: &CrossSection,
        likelihood: &Likelihood,
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<Self> {
        let aliases = components
            .iter()
            .map(|(name, tags)| (name.clone(), CanonicalTags::new(tags)))
            .collect::<HashMap<_, _>>();
        let mut integrals = HashMap::new();
        for tags in aliases.values() {
            if !integrals.contains_key(tags) {
                integrals.insert(
                    tags.clone(),
                    member.integrals_for(likelihood, Some(tags.as_slice()))?,
                );
            }
        }
        Ok(Self { aliases, integrals })
    }
}

struct BinAssignments {
    indices: Vec<Option<usize>>,
    count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProjectionKey(Vec<AxisKey>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AxisKey {
    root: usize,
    nodes: Vec<ExprNodeStructuralKey>,
    edges: Vec<u64>,
}

impl ProjectionKey {
    fn new(axes: &[Axis]) -> Self {
        Self(
            axes.iter()
                .map(|axis| {
                    let graph = axis.expression.to_graph();
                    AxisKey {
                        root: graph.root().index(),
                        nodes: graph
                            .nodes()
                            .iter()
                            .map(|node| node.structural_key())
                            .collect(),
                        edges: axis.edges.iter().map(|edge| edge.to_bits()).collect(),
                    }
                })
                .collect(),
        )
    }
}

struct PreparedProjection {
    name: String,
    request_axes: Vec<Axis>,
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    volumes: Vec<f64>,
    data_bins: BinAssignments,
    accepted_bins: BinAssignments,
    generated_bins: BinAssignments,
}

struct ProjectionReplica {
    bins: Vec<Option<BinAssignments>>,
    weights: Option<Vec<f64>>,
    total_data: f64,
}

impl BinAssignments {
    fn new(values: &[Vec<f64>], axes: &[Axis]) -> Self {
        let event_count = values.first().map_or(0, Vec::len);
        debug_assert!(
            values
                .iter()
                .all(|coordinates| coordinates.len() == event_count)
        );
        let indices = (0..event_count)
            .map(|event| {
                axes.iter()
                    .zip(values)
                    .try_fold(0, |flat, (axis, coordinates)| {
                        bin_index(coordinates[event], &axis.edges)
                            .map(|index| flat * axis.bins() + index)
                    })
            })
            .collect();
        Self {
            indices,
            count: axes.iter().map(Axis::bins).product(),
        }
    }

    fn accumulate(&self, weights: &[f64]) -> Vec<f64> {
        self.accumulate_products(weights, None)
    }

    fn accumulate_weighted_intensities(&self, weights: &[f64], intensities: &[f64]) -> Vec<f64> {
        self.accumulate_products(weights, Some(intensities))
    }

    fn accumulate_weighted_block(
        &self,
        offset: usize,
        weights: &[f64],
        intensities: &[f64],
        bins: &mut [f64],
    ) {
        debug_assert!(offset + intensities.len() <= self.indices.len());
        debug_assert_eq!(self.indices.len(), weights.len());
        debug_assert_eq!(self.count, bins.len());
        for (row, &intensity) in intensities.iter().enumerate() {
            let event = offset + row;
            if let Some(index) = self.indices[event] {
                bins[index] += weights[event] * intensity;
            }
        }
    }

    fn accumulate_products(&self, weights: &[f64], intensities: Option<&[f64]>) -> Vec<f64> {
        debug_assert_eq!(self.indices.len(), weights.len());
        debug_assert!(intensities.is_none_or(|values| values.len() == weights.len()));
        let mut bins = vec![0.0; self.count];
        for (event, (&index, &weight)) in self.indices.iter().zip(weights).enumerate() {
            if let Some(index) = index {
                let intensity = intensities.map_or(1.0, |values| values[event]);
                bins[index] += weight * intensity;
            }
        }
        bins
    }
}

impl CombinedMemberWorkspace {
    fn prepare(
        member: &CrossSection,
        axes: &[Axis],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<Self> {
        let execution = member.likelihood.execution();
        let (data, _) = member.likelihood.intensity_datasets(&member.term_name)?;
        let full = member.integrals_for(&member.likelihood, None)?;
        let component_integrals =
            CanonicalComponents::prepare(member, &member.likelihood, components)?;
        Ok(Self {
            luminosity: member.luminosity,
            data_bins: evaluate_bin_assignments(data, axes, execution)?,
            data_weights: dataset_weights(data)?,
            accepted_bins: evaluate_bin_assignments(full.accepted_mc_source(), axes, execution)?,
            accepted_weights: dataset_weights(full.accepted_mc_source())?,
            generated_bins: evaluate_bin_assignments(full.generated_mc_source(), axes, execution)?,
            generated_weights: dataset_weights(full.generated_mc_source())?,
            full,
            components: component_integrals,
        })
    }

    fn evaluate(
        &self,
        parameters: &[f64],
        data_bins: &BinAssignments,
        data_weights: &[f64],
        total_data: f64,
        factor: f64,
    ) -> LikelihoodResult<CombinedMemberValues> {
        let full_accepted_intensities = self.full.accepted_intensities(parameters)?;
        let full_generated_intensities = self.full.generated_intensities(parameters)?;
        let full_accepted_bins = self
            .accepted_bins
            .accumulate_weighted_intensities(&self.accepted_weights, &full_accepted_intensities);
        let full_generated_bins = self
            .generated_bins
            .accumulate_weighted_intensities(&self.generated_weights, &full_generated_intensities);
        let full_accepted = self.full.full_accepted_integral(parameters)?;
        let full_exposures = binned_exposures(
            self.luminosity * factor,
            &full_accepted_bins,
            &full_generated_bins,
        );
        let data = BinnedMeasurement {
            yields: data_bins.accumulate(data_weights),
            exposures: full_exposures.clone(),
        };
        let model = BinnedMeasurement {
            yields: full_accepted_bins
                .iter()
                .map(|accepted| total_data * accepted / full_accepted)
                .collect(),
            exposures: full_exposures,
        };
        let canonical_values = self
            .components
            .integrals
            .iter()
            .map(|(canonical_tags, selected)| {
                record_selection_intensity_evaluation();
                let accepted_intensities = selected.accepted_intensities(parameters)?;
                record_selection_intensity_evaluation();
                let generated_intensities = selected.generated_intensities(parameters)?;
                let accepted = self
                    .accepted_bins
                    .accumulate_weighted_intensities(&self.accepted_weights, &accepted_intensities);
                let generated = self.generated_bins.accumulate_weighted_intensities(
                    &self.generated_weights,
                    &generated_intensities,
                );
                Ok((
                    canonical_tags.clone(),
                    BinnedMeasurement {
                        yields: accepted
                            .iter()
                            .map(|value| total_data * value / full_accepted)
                            .collect(),
                        exposures: binned_exposures(
                            self.luminosity * factor,
                            &accepted,
                            &generated,
                        ),
                    },
                ))
            })
            .collect::<LikelihoodResult<HashMap<_, _>>>()?;
        let component_values = self
            .components
            .aliases
            .iter()
            .map(|(name, canonical_tags)| (name.clone(), canonical_values[canonical_tags].clone()))
            .collect();
        Ok(CombinedMemberValues {
            data,
            model,
            components: component_values,
        })
    }
}

impl std::fmt::Debug for CrossSection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CrossSection")
            .field("term_name", &self.term_name)
            .field("luminosity", &self.luminosity)
            .field("parameters", &self.parameters)
            .field("ensemble", &self.ensemble)
            .field(
                "members",
                &self.members.as_ref().map(|members| members.len()),
            )
            .finish_non_exhaustive()
    }
}

impl CrossSection {
    /// Constructs a central-value cross-section analysis.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or likelihood preparation failure.
    pub fn new(
        likelihood: Arc<Likelihood>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: f64,
        parameters: Vec<f64>,
    ) -> LikelihoodResult<Self> {
        Self::with_ensemble(
            likelihood,
            term_name,
            generated_mc,
            luminosity,
            parameters,
            None,
        )
    }

    /// Constructs an analysis with optional uncertainty draws.
    ///
    /// # Errors
    /// Returns an error for invalid inputs, mismatched parameters, or preparation failure.
    pub fn with_ensemble(
        likelihood: Arc<Likelihood>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: f64,
        parameters: Vec<f64>,
        ensemble: Option<Ensemble>,
    ) -> LikelihoodResult<Self> {
        if !luminosity.is_finite() || luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        likelihood.params().validate_free_values(&parameters)?;
        let term_name = term_name.into();
        if let Some(ensemble) = &ensemble {
            let names = likelihood
                .params()
                .free_params()
                .iter()
                .map(|id| likelihood.params().name(*id).map(str::to_owned))
                .collect::<Result<Vec<_>, _>>()?;
            if names != ensemble.parameter_names {
                return Err(invalid(
                    "ensemble parameter names do not match the likelihood",
                ));
            }
        }
        let full_integrals = likelihood.cross_section_integrals(&term_name, &generated_mc)?;
        let likelihood_key = Arc::as_ptr(&likelihood) as usize;
        let mut integral_cache = HashMap::new();
        integral_cache.insert((likelihood_key, None), full_integrals.clone());
        Ok(Self {
            likelihood,
            term_name,
            generated_mc,
            full_integrals,
            luminosity,
            parameters,
            ensemble,
            members: None,
            integral_cache: Arc::new(Mutex::new(integral_cache)),
            cache_hits: Default::default(),
            cache_misses: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Exposure-pools measurements of the same underlying cross section.
    ///
    /// # Errors
    /// Returns an error when no measurements are supplied.
    pub fn combine(members: Vec<CrossSection>) -> LikelihoodResult<Self> {
        let factors = (0..members.len())
            .map(|_| Estimate::central(1.0))
            .collect::<LikelihoodResult<Vec<_>>>()?;
        Self::combine_with_factors(members, factors)
    }

    /// Exposure-pools measurements with branching or other exposure factors.
    ///
    /// # Errors
    /// Returns an error for missing measurements or invalid factors.
    pub fn combine_with_factors(
        members: Vec<CrossSection>,
        factors: Vec<Estimate>,
    ) -> LikelihoodResult<Self> {
        if members.is_empty() {
            return Err(invalid("at least one CrossSection is required"));
        }
        if factors.len() != members.len()
            || factors.iter().any(|factor| {
                factor.central <= 0.0 || factor.draws.iter().any(|value| *value <= 0.0)
            })
        {
            return Err(invalid(
                "factors must contain one positive estimate per member",
            ));
        }
        let template = members[0].clone();
        Ok(Self {
            likelihood: Arc::clone(&template.likelihood),
            term_name: template.term_name,
            generated_mc: template.generated_mc,
            full_integrals: template.full_integrals,
            luminosity: template.luminosity,
            parameters: template.parameters,
            ensemble: None,
            members: Some(Arc::new(members.into_iter().zip(factors).collect())),
            integral_cache: template.integral_cache,
            cache_hits: template.cache_hits,
            cache_misses: template.cache_misses,
        })
    }

    /// Full-model observed-yield-normalized cross section.
    ///
    /// # Errors
    /// Returns an error when model integrals or ensemble evaluation fail.
    pub fn observed_total(&self) -> LikelihoodResult<Estimate> {
        self.observed_total_selected(None)
    }

    /// Returns integral-cache hit, miss, count, and retained-byte diagnostics.
    pub fn diagnostics(&self) -> CrossSectionDiagnostics {
        let cache = self
            .integral_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        CrossSectionDiagnostics {
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cached_integrals: cache.len(),
            prepared_bytes: cache
                .values()
                .map(CrossSectionIntegrals::resident_bytes)
                .sum(),
        }
    }

    /// Tag-narrowed observed-yield-normalized cross section.
    ///
    /// # Errors
    /// Returns an error when tag projection, integrals, or ensemble evaluation fail.
    pub fn observed_total_with_tags(&self, tags: &[String]) -> LikelihoodResult<Estimate> {
        self.observed_total_selected(Some(tags))
    }

    fn observed_total_selected(&self, tags: Option<&[String]>) -> LikelihoodResult<Estimate> {
        if self.members.is_some() {
            return self.combined_total(tags);
        }
        self.evaluate_estimate(tags, |integrals, parameters| {
            integrals.observed_cross_section(parameters, self.luminosity)
        })
    }

    /// Full-model fitted cross section from an absolute-rate likelihood term.
    ///
    /// # Errors
    /// Returns an error for shape-only terms, combined analyses, invalid
    /// luminosity, or failed integral or ensemble evaluation.
    pub fn fitted_total(&self) -> LikelihoodResult<Estimate> {
        self.fitted_total_selected(None)
    }

    /// Tag-narrowed fitted cross section from an absolute-rate likelihood term.
    ///
    /// # Errors
    /// Returns an error for shape-only terms, combined analyses, invalid
    /// luminosity, or failed tag projection, integral, or ensemble evaluation.
    pub fn fitted_total_with_tags(&self, tags: &[String]) -> LikelihoodResult<Estimate> {
        self.fitted_total_selected(Some(tags))
    }

    fn fitted_total_selected(&self, tags: Option<&[String]>) -> LikelihoodResult<Estimate> {
        self.evaluate_estimate(tags, |integrals, parameters| {
            integrals.fitted_cross_section(parameters, self.luminosity)
        })
    }

    /// Alias for [`Self::observed_total`].
    ///
    /// # Errors
    /// Returns an error when model integrals or ensemble evaluation fail.
    pub fn total(&self) -> LikelihoodResult<Estimate> {
        self.observed_total()
    }

    /// Alias for [`Self::observed_total_with_tags`].
    ///
    /// # Errors
    /// Returns an error when tag projection, integrals, or ensemble evaluation fail.
    pub fn total_with_tags(&self, tags: &[String]) -> LikelihoodResult<Estimate> {
        self.observed_total_with_tags(tags)
    }

    /// Full-model acceptance.
    ///
    /// # Errors
    /// Returns an error when model integrals or ensemble evaluation fail.
    pub fn acceptance(&self) -> LikelihoodResult<Estimate> {
        self.acceptance_selected(None)
    }

    /// Tag-narrowed model-weighted acceptance.
    ///
    /// # Errors
    /// Returns an error when tag projection, integrals, or ensemble evaluation fail.
    pub fn acceptance_with_tags(&self, tags: &[String]) -> LikelihoodResult<Estimate> {
        self.acceptance_selected(Some(tags))
    }

    fn acceptance_selected(&self, tags: Option<&[String]>) -> LikelihoodResult<Estimate> {
        self.evaluate_estimate(tags, CrossSectionIntegrals::acceptance)
    }

    /// Full-model acceptance-corrected yield.
    ///
    /// # Errors
    /// Returns an error when model integrals or ensemble evaluation fail.
    pub fn corrected_yield(&self) -> LikelihoodResult<Estimate> {
        self.corrected_yield_selected(None)
    }

    /// Tag-narrowed acceptance-corrected yield.
    ///
    /// # Errors
    /// Returns an error when tag projection, integrals, or ensemble evaluation fail.
    pub fn corrected_yield_with_tags(&self, tags: &[String]) -> LikelihoodResult<Estimate> {
        self.corrected_yield_selected(Some(tags))
    }

    fn corrected_yield_selected(&self, tags: Option<&[String]>) -> LikelihoodResult<Estimate> {
        self.evaluate_estimate(tags, |integrals, parameters| {
            let accepted_yield = if tags.is_some() {
                integrals.data_weight_sum() * integrals.accepted_integral(parameters)?
                    / integrals.full_accepted_integral(parameters)?
            } else {
                integrals.data_weight_sum()
            };
            integrals.acceptance_corrected_yield(parameters, accepted_yield)
        })
    }

    /// Computes an arbitrary-dimensional differential cross section.
    ///
    /// # Errors
    /// Returns an error for absent axes or failed expression/model evaluation.
    pub fn differential(
        &self,
        axes: &[Axis],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<DifferentialCrossSection> {
        if axes.is_empty() {
            return Err(invalid("at least one differential axis is required"));
        }
        if self.members.is_some() {
            return self.combined_differential(axes, components);
        }
        self.single_differential(axes, components)
    }

    /// Computes an ordered set of independent named differential cross sections.
    ///
    /// # Errors
    /// Returns an error for an empty request, duplicate names, combined cross
    /// sections, or failed expression/model evaluation.
    pub fn projection_set(
        &self,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<ProjectionSet> {
        if projections.is_empty() {
            return Err(invalid("at least one projection is required"));
        }
        if self.members.is_some() {
            return Err(invalid(
                "projection sets for combined CrossSection values are not supported",
            ));
        }
        let mut names = std::collections::HashSet::with_capacity(projections.len());
        for projection in projections {
            if !names.insert(projection.name()) {
                return Err(invalid(format!(
                    "duplicate projection name: {}",
                    projection.name()
                )));
            }
        }
        self.single_projection_set(projections, components)
    }

    fn integrals_for(
        &self,
        likelihood: &Likelihood,
        tags: Option<&[String]>,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let key_tags = tags.map(CanonicalTags::new);
        let key = (likelihood as *const Likelihood as usize, key_tags.clone());
        if let Some(integrals) = self
            .integral_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&key)
            .cloned()
        {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(integrals);
        }
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let integrals = match key_tags.as_ref() {
            Some(tags) => likelihood.cross_section_integrals_with_tags(
                &self.term_name,
                &self.generated_mc,
                tags.as_slice().iter().map(String::as_str),
            ),
            None => likelihood.cross_section_integrals(&self.term_name, &self.generated_mc),
        }?;
        self.integral_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(key, integrals.clone());
        Ok(integrals)
    }

    fn evaluate_estimate(
        &self,
        tags: Option<&[String]>,
        function: impl Fn(&CrossSectionIntegrals, &[f64]) -> LikelihoodResult<f64>,
    ) -> LikelihoodResult<Estimate> {
        if self.members.is_some() {
            return Err(invalid(
                "operation is not defined directly for a combined CrossSection",
            ));
        }
        let integrals = self.integrals_for(&self.likelihood, tags)?;
        let central = function(&integrals, &self.parameters)?;
        let draws = self
            .ensemble
            .as_ref()
            .map(|ensemble| {
                ensemble
                    .draws
                    .iter()
                    .enumerate()
                    .map(|(index, draw)| {
                        let replica_integrals = ensemble
                            .replicas
                            .get(index)
                            .map(|likelihood| self.integrals_for(likelihood, tags))
                            .transpose()?;
                        function(replica_integrals.as_ref().unwrap_or(&integrals), draw)
                    })
                    .collect::<LikelihoodResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Estimate::from_evaluation(
            central,
            draws,
            self.ensemble.as_ref().map(Ensemble::source_id),
        ))
    }

    fn selected_measurement_for(
        &self,
        likelihood: &Likelihood,
        parameters: &[f64],
        tags: Option<&[String]>,
        factor: f64,
    ) -> LikelihoodResult<(f64, f64)> {
        let full = self.integrals_for(likelihood, None)?;
        let selected = self.integrals_for(likelihood, tags)?;
        let full_accepted = full.full_accepted_integral(parameters)?;
        let accepted = selected.accepted_integral(parameters)?;
        let generated = selected.generated_integral(parameters)?;
        if full_accepted <= 0.0 || accepted <= 0.0 || generated <= 0.0 {
            return Err(invalid(
                "cross-section combination requires positive integrals",
            ));
        }
        Ok((
            selected.data_weight_sum() * accepted / full_accepted,
            self.luminosity * factor * accepted / generated,
        ))
    }

    fn combined_total(&self, tags: Option<&[String]>) -> LikelihoodResult<Estimate> {
        let members = self
            .members
            .as_ref()
            .ok_or_else(|| invalid("CrossSection is not combined"))?;
        let central = members.iter().try_fold(
            (0.0, 0.0),
            |(yield_sum, exposure_sum), (member, factor)| {
                let (yield_value, exposure) = member.selected_measurement_for(
                    &member.likelihood,
                    &member.parameters,
                    tags,
                    factor.central,
                )?;
                Ok::<_, LikelihoodError>((yield_sum + yield_value, exposure_sum + exposure))
            },
        )?;
        let draw_count = member_draw_count(members);
        let reference_source = member_reference_source(members);
        let mut draws = Vec::with_capacity(draw_count);
        for index in 0..draw_count {
            let mut yield_sum = 0.0;
            let mut exposure_sum = 0.0;
            for (position, (member, factor)) in members.iter().enumerate() {
                let draw_index = member.ensemble.as_ref().map(|ensemble| {
                    paired_draw_index(
                        index,
                        position,
                        ensemble.len(),
                        Some(ensemble.source_id),
                        reference_source,
                    )
                });
                let parameters = draw_index
                    .and_then(|draw_index| {
                        member
                            .ensemble
                            .as_ref()
                            .and_then(|ensemble| ensemble.draws.get(draw_index))
                    })
                    .map(Vec::as_slice)
                    .unwrap_or(&member.parameters);
                let likelihood = draw_index
                    .and_then(|draw_index| {
                        member
                            .ensemble
                            .as_ref()
                            .and_then(|ensemble| ensemble.replicas.get(draw_index))
                    })
                    .map(Arc::as_ref)
                    .unwrap_or(&member.likelihood);
                let factor_index = (!factor.draws.is_empty()).then(|| {
                    paired_draw_index(
                        index,
                        position,
                        factor.draws.len(),
                        factor.source_id,
                        reference_source,
                    )
                });
                let factor = factor_index
                    .and_then(|draw_index| factor.draws.get(draw_index))
                    .copied()
                    .unwrap_or(factor.central);
                let (yield_value, exposure) =
                    member.selected_measurement_for(likelihood, parameters, tags, factor)?;
                yield_sum += yield_value;
                exposure_sum += exposure;
            }
            draws.push(yield_sum / exposure_sum);
        }
        Ok(Estimate::from_evaluation(
            central.0 / central.1,
            draws,
            Some(next_uncertainty_source_id()),
        ))
    }

    fn single_differential(
        &self,
        axes: &[Axis],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<DifferentialCrossSection> {
        let projection = Projection {
            name: "differential".to_owned(),
            axes: axes.to_vec(),
        };
        let mut entries = self
            .single_projection_set(std::slice::from_ref(&projection), components)?
            .entries;
        Ok(entries.remove(0).1)
    }

    fn single_projection_set(
        &self,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<ProjectionSet> {
        let request_context = format!(
            "member `{}` projections [{}]",
            self.term_name,
            projections
                .iter()
                .map(|projection| projection.name())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let execution = self.likelihood.execution();
        let (data, _) = self.likelihood.intensity_datasets(&self.term_name)?;
        let full = self.integrals_for(&self.likelihood, None)?;
        let data_weights = dataset_weights(data)?;
        let accepted_weights = dataset_weights(full.accepted_mc_source())?;
        let generated_weights = dataset_weights(full.generated_mc_source())?;
        let component_integrals = CanonicalComponents::prepare(self, &self.likelihood, components)?;
        let mut plan_indexes = HashMap::with_capacity(projections.len());
        let mut plans = Vec::with_capacity(projections.len());
        let projection_plans = projections
            .iter()
            .map(|projection| {
                let key = ProjectionKey::new(projection.axes());
                if let Some(index) = plan_indexes.get(&key) {
                    return Ok(*index);
                }
                let index = plans.len();
                let prepare = || {
                    Ok(PreparedProjection {
                        name: projection.name().to_owned(),
                        request_axes: projection.axes().to_vec(),
                        axes: projection
                            .axes()
                            .iter()
                            .map(|axis| axis.edges.clone())
                            .collect(),
                        shape: projection.axes().iter().map(Axis::bins).collect(),
                        volumes: bin_volumes(projection.axes()),
                        data_bins: evaluate_bin_assignments(data, projection.axes(), execution)?,
                        accepted_bins: evaluate_bin_assignments(
                            full.accepted_mc_source(),
                            projection.axes(),
                            execution,
                        )?,
                        generated_bins: evaluate_bin_assignments(
                            full.generated_mc_source(),
                            projection.axes(),
                            execution,
                        )?,
                    })
                };
                plans.push(prepare().map_err(|error: LikelihoodError| {
                    invalid(format!(
                        "projection `{}` preparation failed: {error}",
                        projection.name()
                    ))
                })?);
                plan_indexes.insert(key, index);
                Ok(index)
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let parameter_sets = std::iter::once(self.parameters.as_slice())
            .chain(
                self.ensemble
                    .iter()
                    .flat_map(|ensemble| ensemble.draws.iter().map(Vec::as_slice)),
            )
            .collect::<Vec<_>>();
        let parameter_contexts = std::iter::once("central value".to_owned())
            .chain(
                (0..parameter_sets.len().saturating_sub(1))
                    .map(|index| format!("ensemble draw {index}")),
            )
            .collect::<Vec<_>>();
        let mut accepted_histograms = plans
            .iter()
            .map(|plan| vec![vec![0.0; plan.accepted_bins.count]; parameter_sets.len()])
            .collect::<Vec<_>>();
        record_prepared_intensity_evaluation();
        let full_accepted_integrals = full
            .visit_accepted_prepared_intensities_many(
                &parameter_sets,
                &parameter_contexts,
                |offset, parameter_index, intensities| {
                    for (plan, histograms) in plans.iter().zip(&mut accepted_histograms) {
                        plan.accepted_bins.accumulate_weighted_block(
                            offset,
                            &accepted_weights,
                            intensities,
                            &mut histograms[parameter_index],
                        );
                    }
                },
            )
            .map_err(|error| {
                invalid(format!(
                    "projection set {request_context} accepted MC intensity evaluation failed: {error}"
                ))
            })?;
        let mut generated_histograms = plans
            .iter()
            .map(|plan| vec![vec![0.0; plan.generated_bins.count]; parameter_sets.len()])
            .collect::<Vec<_>>();
        record_prepared_intensity_evaluation();
        full.visit_generated_prepared_intensities_many(
            &parameter_sets,
            &parameter_contexts,
            |offset, parameter_index, intensities| {
                for (plan, histograms) in plans.iter().zip(&mut generated_histograms) {
                    plan.generated_bins.accumulate_weighted_block(
                        offset,
                        &generated_weights,
                        intensities,
                        &mut histograms[parameter_index],
                    );
                }
            },
        )
        .map_err(|error| {
            invalid(format!(
                "projection set {request_context} generated MC intensity evaluation failed: {error}"
            ))
        })?;
        let mut component_histograms = HashMap::new();
        for (canonical_tags, selected) in &component_integrals.integrals {
            let mut histograms = plans
                .iter()
                .map(|plan| vec![vec![0.0; plan.generated_bins.count]; parameter_sets.len()])
                .collect::<Vec<_>>();
            record_selection_intensity_evaluation();
            record_prepared_intensity_evaluation();
            selected
                .visit_generated_prepared_intensities_many(
                    &parameter_sets,
                    &parameter_contexts,
                    |offset, parameter_index, intensities| {
                        for (plan, histograms) in plans.iter().zip(&mut histograms) {
                            plan.generated_bins.accumulate_weighted_block(
                                offset,
                                &generated_weights,
                                intensities,
                                &mut histograms[parameter_index],
                            );
                        }
                    },
                )
                .map_err(|error| {
                    invalid(format!(
                        "projection set {request_context} generated MC component {:?} intensity evaluation failed: {error}",
                        canonical_tags.as_slice()
                    ))
                })?;
            component_histograms.insert(canonical_tags.clone(), histograms);
        }
        let replicas = self
            .ensemble
            .as_ref()
            .map(|ensemble| {
                ensemble
                    .draws
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                let replica_data = ensemble
                    .replicas
                    .get(index)
                    .map(|likelihood| likelihood.intensity_datasets(&self.term_name))
                    .transpose()?
                    .map(|(data, _)| data);
                let replica_weights = replica_data.map(dataset_weights).transpose()?;
                let total_data = replica_weights
                    .as_ref()
                    .map(|weights| weights.iter().sum())
                    .unwrap_or_else(|| full.data_weight_sum());
                        let bins = plans
                            .iter()
                            .map(|plan| {
                                ensemble
                                    .replica_bin_assignments(
                                        replica_data,
                                        &plan.request_axes,
                                        execution,
                                    )
                                    .map_err(|error| {
                                        invalid(format!(
                                            "projection `{}` draw {index} bin preparation failed: {error}",
                                            plan.name
                                        ))
                                    })
                            })
                            .collect::<LikelihoodResult<Vec<_>>>()?;
                        Ok(ProjectionReplica {
                            bins,
                            weights: replica_weights,
                            total_data,
                        })
                    })
                    .collect::<LikelihoodResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        let unique_results = plans
            .iter()
            .enumerate()
            .map(|(plan_index, plan)| {
                let evaluate = |draw_index: usize,
                                draw_data_bins: &BinAssignments,
                                draw_data_weights: &[f64],
                                total_data: f64|
                 -> DifferentialValues {
                    let data_histogram = draw_data_bins.accumulate(draw_data_weights);
                    let accepted_histogram = &accepted_histograms[plan_index][draw_index];
                    let generated_histogram = &generated_histograms[plan_index][draw_index];
                    let full_accepted = full_accepted_integrals[draw_index];
                    let data_cross_section = data_histogram
                        .iter()
                        .zip(accepted_histogram)
                        .zip(generated_histogram)
                        .zip(&plan.volumes)
                        .map(|(((data, accepted), generated), volume)| {
                            if *accepted > 0.0 {
                                data * generated / accepted / self.luminosity / volume
                            } else {
                                f64::NAN
                            }
                        })
                        .collect();
                    let model = generated_histogram
                        .iter()
                        .zip(&plan.volumes)
                        .map(|(generated, volume)| {
                            total_data * generated / full_accepted / self.luminosity / volume
                        })
                        .collect();
                    let component_values = component_integrals
                        .aliases
                        .iter()
                        .map(|(name, canonical_tags)| {
                            let bins =
                                &component_histograms[canonical_tags][plan_index][draw_index];
                            (
                                name.clone(),
                                bins.iter()
                                    .zip(&plan.volumes)
                                    .map(|(generated, volume)| {
                                        total_data * generated
                                            / full_accepted
                                            / self.luminosity
                                            / volume
                                    })
                                    .collect(),
                            )
                        })
                        .collect();
                    (data_cross_section, model, component_values)
                };
                let (data_cross_section, model, component_values) =
                    evaluate(0, &plan.data_bins, &data_weights, full.data_weight_sum());
                let mut data_draws = Vec::with_capacity(replicas.len());
                let mut model_draws = Vec::with_capacity(replicas.len());
                let mut component_draws: HashMap<String, Vec<Vec<f64>>> = components
                    .keys()
                    .map(|name| (name.clone(), Vec::with_capacity(replicas.len())))
                    .collect();
                for (index, replica) in replicas.iter().enumerate() {
                    let draw_data_bins =
                        replica.bins[plan_index].as_ref().unwrap_or(&plan.data_bins);
                    let draw_data_weights = replica.weights.as_deref().unwrap_or(&data_weights);
                    let (data, model, values) = evaluate(
                        index + 1,
                        draw_data_bins,
                        draw_data_weights,
                        replica.total_data,
                    );
                    data_draws.push(data);
                    model_draws.push(model);
                    for (name, values) in values {
                        component_draws.entry(name).or_default().push(values);
                    }
                }
                Ok(DifferentialCrossSection {
                    axes: plan.axes.clone(),
                    shape: plan.shape.clone(),
                    data: BinnedEstimate::new(data_cross_section, data_draws),
                    model: BinnedEstimate::new(model, model_draws),
                    components: component_values
                        .into_iter()
                        .map(|(name, central)| {
                            let draws = component_draws.remove(&name).unwrap_or_default();
                            (name, BinnedEstimate::new(central, draws))
                        })
                        .collect(),
                })
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        Ok(ProjectionSet {
            entries: projections
                .iter()
                .zip(projection_plans)
                .map(|(projection, plan)| (projection.name.clone(), unique_results[plan].clone()))
                .collect(),
        })
    }

    fn combined_differential(
        &self,
        axes: &[Axis],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<DifferentialCrossSection> {
        let members = self
            .members
            .as_ref()
            .ok_or_else(|| invalid("CrossSection is not combined"))?;
        let volumes = bin_volumes(axes);
        let workspaces = members
            .iter()
            .map(|(member, _)| CombinedMemberWorkspace::prepare(member, axes, components))
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let central_values = members
            .iter()
            .zip(&workspaces)
            .map(|((member, factor), workspace)| {
                workspace.evaluate(
                    &member.parameters,
                    &workspace.data_bins,
                    &workspace.data_weights,
                    workspace.full.data_weight_sum(),
                    factor.central,
                )
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let data = pool_binned(central_values.iter().map(|values| &values.data), &volumes);
        let model = pool_binned(central_values.iter().map(|values| &values.model), &volumes);
        let component_central: HashMap<String, Vec<f64>> = components
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    pool_binned(
                        central_values.iter().map(|values| &values.components[name]),
                        &volumes,
                    ),
                )
            })
            .collect();
        let draw_count = member_draw_count(members);
        let reference_source = member_reference_source(members);
        let mut data_draws = Vec::with_capacity(draw_count);
        let mut model_draws = Vec::with_capacity(draw_count);
        let mut component_draws: HashMap<String, Vec<Vec<f64>>> = components
            .keys()
            .map(|name| (name.clone(), Vec::with_capacity(draw_count)))
            .collect();
        for index in 0..draw_count {
            let draw_values = members
                .iter()
                .zip(&workspaces)
                .enumerate()
                .map(|(position, ((member, factor), workspace))| {
                    let ensemble = member.ensemble.as_ref();
                    let draw_index = ensemble.map(|ensemble| {
                        paired_draw_index(
                            index,
                            position,
                            ensemble.len(),
                            Some(ensemble.source_id),
                            reference_source,
                        )
                    });
                    let parameters = draw_index
                        .and_then(|draw_index| {
                            ensemble.and_then(|ensemble| ensemble.draws.get(draw_index))
                        })
                        .map(Vec::as_slice)
                        .unwrap_or(&member.parameters);
                    let replica_data = draw_index
                        .and_then(|draw_index| {
                            ensemble.and_then(|ensemble| ensemble.replicas.get(draw_index))
                        })
                        .map(|likelihood| likelihood.intensity_datasets(&member.term_name))
                        .transpose()?
                        .map(|(data, _)| data);
                    let replica_bins = match ensemble {
                        Some(ensemble) => ensemble.replica_bin_assignments(
                            replica_data,
                            axes,
                            member.likelihood.execution(),
                        )?,
                        None => None,
                    };
                    let replica_weights = replica_data.map(dataset_weights).transpose()?;
                    let data_bins = replica_bins.as_ref().unwrap_or(&workspace.data_bins);
                    let data_weights = replica_weights
                        .as_deref()
                        .unwrap_or(&workspace.data_weights);
                    let total_data = replica_weights
                        .as_ref()
                        .map(|weights| weights.iter().sum())
                        .unwrap_or_else(|| workspace.full.data_weight_sum());
                    let factor_index = (!factor.draws.is_empty()).then(|| {
                        paired_draw_index(
                            index,
                            position,
                            factor.draws.len(),
                            factor.source_id,
                            reference_source,
                        )
                    });
                    let factor = factor_index
                        .and_then(|draw_index| factor.draws.get(draw_index))
                        .copied()
                        .unwrap_or(factor.central);
                    workspace.evaluate(parameters, data_bins, data_weights, total_data, factor)
                })
                .collect::<LikelihoodResult<Vec<_>>>()?;
            data_draws.push(pool_binned(
                draw_values.iter().map(|values| &values.data),
                &volumes,
            ));
            model_draws.push(pool_binned(
                draw_values.iter().map(|values| &values.model),
                &volumes,
            ));
            for name in components.keys() {
                component_draws
                    .entry(name.clone())
                    .or_default()
                    .push(pool_binned(
                        draw_values.iter().map(|values| &values.components[name]),
                        &volumes,
                    ));
            }
        }
        Ok(DifferentialCrossSection {
            axes: axes.iter().map(|axis| axis.edges.clone()).collect(),
            shape: axes.iter().map(Axis::bins).collect(),
            data: BinnedEstimate::new(data, data_draws),
            model: BinnedEstimate::new(model, model_draws),
            components: component_central
                .into_iter()
                .map(|(name, central)| {
                    let draws = component_draws.remove(&name).unwrap_or_default();
                    (name, BinnedEstimate::new(central, draws))
                })
                .collect(),
        })
    }
}

impl Likelihood {
    /// Prepares a central-value cross-section analysis from a shared likelihood.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or likelihood preparation failure.
    pub fn cross_section(
        self: &Arc<Self>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: f64,
        parameters: Vec<f64>,
    ) -> LikelihoodResult<CrossSection> {
        CrossSection::new(
            Arc::clone(self),
            term_name,
            generated_mc,
            luminosity,
            parameters,
        )
    }

    /// Prepares an ensemble-backed cross-section analysis.
    ///
    /// # Errors
    /// Returns an error for invalid inputs, mismatched draws, or preparation failure.
    pub fn cross_section_with_ensemble(
        self: &Arc<Self>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: f64,
        parameters: Vec<f64>,
        ensemble: Ensemble,
    ) -> LikelihoodResult<CrossSection> {
        CrossSection::with_ensemble(
            Arc::clone(self),
            term_name,
            generated_mc,
            luminosity,
            parameters,
            Some(ensemble),
        )
    }
}

fn member_draw_count(members: &[(CrossSection, Estimate)]) -> usize {
    members
        .iter()
        .flat_map(|(member, factor)| {
            [
                member.ensemble.as_ref().map(Ensemble::len),
                (!factor.draws.is_empty()).then_some(factor.draws.len()),
            ]
        })
        .flatten()
        .min()
        .unwrap_or(0)
}

fn member_reference_source(members: &[(CrossSection, Estimate)]) -> Option<u64> {
    members.iter().find_map(|(member, factor)| {
        member
            .ensemble
            .as_ref()
            .map(Ensemble::source_id)
            .or(factor.source_id)
    })
}

fn paired_draw_index(
    index: usize,
    position: usize,
    draw_count: usize,
    source_id: Option<u64>,
    reference_source: Option<u64>,
) -> usize {
    if source_id == reference_source {
        index % draw_count
    } else {
        (index.wrapping_mul(2 * position + 1) + position) % draw_count
    }
}

fn binned_exposures(luminosity: f64, accepted: &[f64], generated: &[f64]) -> Vec<f64> {
    accepted
        .iter()
        .zip(generated)
        .map(|(accepted, generated)| {
            if *generated > 0.0 {
                luminosity * accepted / generated
            } else {
                0.0
            }
        })
        .collect()
}

fn pool_binned<'a>(
    measurements: impl IntoIterator<Item = &'a BinnedMeasurement>,
    volumes: &[f64],
) -> Vec<f64> {
    let mut yields = vec![0.0; volumes.len()];
    let mut exposures = vec![0.0; volumes.len()];
    for measurement in measurements {
        for index in 0..volumes.len() {
            yields[index] += measurement.yields[index];
            exposures[index] += measurement.exposures[index];
        }
    }
    (0..volumes.len())
        .map(|index| {
            if exposures[index] > 0.0 {
                yields[index] / exposures[index] / volumes[index]
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn evaluate_coordinates(
    dataset: &Dataset,
    axes: &[Axis],
    execution: &Execution,
) -> LikelihoodResult<Vec<Vec<f64>>> {
    axes.iter()
        .map(|axis| {
            dataset
                .evaluate_real(&axis.expression, execution)
                .map_err(Into::into)
        })
        .collect()
}

fn evaluate_bin_assignments(
    dataset: &Dataset,
    axes: &[Axis],
    execution: &Execution,
) -> LikelihoodResult<BinAssignments> {
    record_bin_assignment_evaluation();
    Ok(BinAssignments::new(
        &evaluate_coordinates(dataset, axes, execution)?,
        axes,
    ))
}

fn dataset_weights(dataset: &Dataset) -> LikelihoodResult<Vec<f64>> {
    dataset
        .try_fold_events(Vec::new(), |mut weights, event| {
            weights.push(event.weight());
            Ok(weights)
        })
        .map_err(Into::into)
}

fn bin_volumes(axes: &[Axis]) -> Vec<f64> {
    axes.iter().fold(vec![1.0], |volumes, axis| {
        volumes
            .into_iter()
            .flat_map(|volume| {
                axis.edges
                    .windows(2)
                    .map(move |pair| volume * (pair[1] - pair[0]))
            })
            .collect()
    })
}

fn bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if !value.is_finite() || value < edges[0] || value >= *edges.last()? {
        return None;
    }
    edges.windows(2).position(|pair| value < pair[1])
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_data::{
        data::{EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{Expr, event_scalar, parameter};

    use super::*;

    fn weighted_dataset(values: &[(f64, f64)]) -> Dataset {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let batch = EventBatch::from_events(
            schema,
            values
                .iter()
                .map(|(x, weight)| OwnedEvent::weighted(vec![], vec![*x], *weight)),
        )
        .unwrap();
        Dataset::from_batches(vec![batch]).unwrap()
    }

    fn weighted_dataset_2d(values: &[(f64, f64, f64)]) -> Dataset {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], true).unwrap());
        let batch = EventBatch::from_events(
            schema,
            values
                .iter()
                .map(|(x, y, weight)| OwnedEvent::weighted(vec![], vec![*x, *y], *weight)),
        )
        .unwrap();
        Dataset::from_batches(vec![batch]).unwrap()
    }

    struct CanonicalSelectionFixture {
        likelihood: Arc<Likelihood>,
        generated: Dataset,
        axis: Axis,
        components: HashMap<String, Vec<String>>,
    }

    fn canonical_selection_fixture() -> CanonicalSelectionFixture {
        let x = event_scalar("x");
        let signal = (Expr::from(parameter!("a", initial: 1.5)) * x.clone()).tagged("signal");
        let background = Expr::from(parameter!("b", initial: 0.75)).tagged("background");
        let model = CompiledModel::from_expr(&(signal + background).norm_sqr()).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (0.75, 2.0), (1.25, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 1.0), (1.75, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        CanonicalSelectionFixture {
            likelihood,
            generated,
            axis: Axis::new(x, vec![0.0, 1.0, 2.0]).unwrap(),
            components: HashMap::from([
                ("ordered".into(), vec!["background".into(), "signal".into()]),
                (
                    "reordered".into(),
                    vec!["signal".into(), "background".into()],
                ),
                (
                    "repeated".into(),
                    vec!["signal".into(), "background".into(), "signal".into()],
                ),
            ]),
        }
    }

    fn assert_projection_close(
        actual: &DifferentialCrossSection,
        expected: &DifferentialCrossSection,
    ) {
        fn assert_estimate_close(actual: &BinnedEstimate, expected: &BinnedEstimate) {
            let rows = std::iter::once((actual.values(), expected.values())).chain(
                actual
                    .draws()
                    .iter()
                    .zip(expected.draws())
                    .map(|(actual, expected)| (actual.as_slice(), expected.as_slice())),
            );
            assert_eq!(actual.draws().len(), expected.draws().len());
            for (actual, expected) in rows {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_relative_eq!(
                            actual,
                            expected,
                            epsilon = 1e-10,
                            max_relative = 1e-10
                        );
                    }
                }
            }
        }

        assert_eq!(actual.axes(), expected.axes());
        assert_eq!(actual.shape(), expected.shape());
        assert_estimate_close(actual.data(), expected.data());
        assert_estimate_close(actual.model(), expected.model());
        assert_eq!(
            actual.components().keys().collect::<HashSet<_>>(),
            expected.components().keys().collect::<HashSet<_>>()
        );
        for (name, expected) in expected.components() {
            assert_estimate_close(&actual.components()[name], expected);
        }
    }

    #[test]
    fn estimate_arithmetic_preserves_scalar_provenance() {
        let estimate = Estimate::with_source_id(2.0, vec![1.0, 3.0], Some(17)).unwrap();
        let scaled = &estimate * 4.0;
        assert_eq!(scaled.value(), 8.0);
        assert_eq!(scaled.draws(), &[4.0, 12.0]);
        assert_eq!(scaled.source_id(), Some(17));
    }

    #[test]
    fn chain_adapter_discards_and_thins_each_walker() {
        let chain = vec![
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![vec![4.0], vec![5.0], vec![6.0], vec![7.0]],
        ];
        let ensemble = Ensemble::from_chain(vec!["x".into()], &chain, 1, 2).unwrap();
        assert_eq!(
            ensemble.draws(),
            &[vec![1.0], vec![3.0], vec![5.0], vec![7.0]]
        );
    }

    #[test]
    fn bin_lookup_uses_half_open_intervals() {
        assert_eq!(bin_index(0.0, &[0.0, 1.0, 2.0]), Some(0));
        assert_eq!(bin_index(1.0, &[0.0, 1.0, 2.0]), Some(1));
        assert_eq!(bin_index(2.0, &[0.0, 1.0, 2.0]), None);
    }

    #[test]
    fn joint_differential_preserves_bin_and_weight_semantics() {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0)).unwrap();
        let data = weighted_dataset_2d(&[
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 2.0),
            (1.0, 0.0, -3.0),
            (1.0, 1.0, 4.0),
            (2.0, 0.5, 8.0),
            (f64::NAN, 0.5, 16.0),
            (0.5, f64::INFINITY, 32.0),
            (0.5, f64::NEG_INFINITY, 64.0),
            (-0.5, 0.5, 128.0),
        ]);
        let accepted = weighted_dataset_2d(&[
            (0.25, 0.25, 1.0),
            (0.25, 1.25, 1.0),
            (1.25, 0.25, 1.0),
            (1.25, 1.25, -1.0),
        ]);
        let generated = weighted_dataset_2d(&[
            (0.25, 0.25, 1.0),
            (0.25, 1.25, 1.0),
            (1.25, 0.25, 1.0),
            (1.25, 1.25, 1.0),
        ]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let axes = [
            Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap(),
            Axis::new(event_scalar("y"), vec![0.0, 1.0, 2.0]).unwrap(),
        ];
        let differential = likelihood
            .cross_section("signal", generated, 1.0, Vec::new())
            .unwrap()
            .differential(&axes, &HashMap::new())
            .unwrap();

        assert_eq!(differential.shape(), &[2, 2]);
        assert_eq!(&differential.data().values()[..3], &[1.0, 2.0, -3.0]);
        assert!(differential.data().values()[3].is_nan());
    }

    #[test]
    fn rust_cross_section_api_covers_totals_differentials_and_bootstrap_pairing() {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0)).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (1.25, 2.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0), (1.75, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let cross_section = likelihood
            .cross_section("signal", generated, 10.0, Vec::new())
            .unwrap();
        assert!(cross_section.total().unwrap().value().is_finite());
        let before = cross_section.diagnostics();
        assert!(cross_section.total().unwrap().value().is_finite());
        let after = cross_section.diagnostics();
        assert_eq!(after.cache_hits(), before.cache_hits() + 1);
        assert_eq!(after.cache_misses(), 1);
        assert_eq!(after.cached_integrals(), 1);
        assert!(after.prepared_bytes() > 0);

        let axis = Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap();
        let differential = cross_section
            .differential(&[axis], &HashMap::new())
            .unwrap();
        assert_eq!(differential.shape(), &[2]);
        assert_eq!(differential.data().values().len(), 2);
        assert_eq!(differential.model().values().len(), 2);

        let ensemble = Ensemble::bootstrap_fit(&likelihood, 3, 42, |replica, _| {
            Ok::<_, std::convert::Infallible>(replica.default_params())
        })
        .unwrap();
        assert_eq!(ensemble.len(), 3);
        assert_eq!(ensemble.replicas().len(), 3);
    }

    #[test]
    fn rust_projection_set_preserves_order_lookup_and_differential_results() {
        let fixture = canonical_selection_fixture();
        let cross_section = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 2.0, vec![1.5, 0.75])
            .unwrap();
        let wide_axis = Axis::new(event_scalar("x"), vec![0.0, 2.0]).unwrap();
        let projections = vec![
            Projection::new("fine", vec![fixture.axis.clone()]).unwrap(),
            Projection::new("wide", vec![wide_axis.clone()]).unwrap(),
        ];

        let expected_fine = cross_section
            .differential(std::slice::from_ref(&fixture.axis), &fixture.components)
            .unwrap();
        let expected_wide = cross_section
            .differential(std::slice::from_ref(&wide_axis), &fixture.components)
            .unwrap();
        let actual = cross_section
            .projection_set(&projections, &fixture.components)
            .unwrap();

        assert_eq!(actual.len(), 2);
        assert_eq!(
            actual.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            vec!["fine", "wide"]
        );
        assert_projection_close(actual.get("fine").unwrap(), &expected_fine);
        assert_projection_close(actual.get("wide").unwrap(), &expected_wide);
        assert!(actual.get("missing").is_none());
    }

    #[test]
    fn projection_set_shares_intensities_and_identical_bin_assignments() {
        let fixture = canonical_selection_fixture();
        let cross_section = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 2.0, vec![1.5, 0.75])
            .unwrap();
        let wide_axis = Axis::new(event_scalar("x"), vec![0.0, 2.0]).unwrap();
        let projections = vec![
            Projection::new("first", vec![fixture.axis.clone()]).unwrap(),
            Projection::new("alias", vec![fixture.axis]).unwrap(),
            Projection::new("wide", vec![wide_axis]).unwrap(),
        ];

        reset_projection_evaluation_counts();
        let result = cross_section
            .projection_set(&projections, &fixture.components)
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(projection_evaluation_counts(), (3, 6));
        assert_projection_close(result.get("first").unwrap(), result.get("alias").unwrap());
    }

    #[test]
    fn projection_set_rejects_invalid_requests_before_evaluation() {
        let fixture = canonical_selection_fixture();
        let cross_section = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 2.0, vec![1.5, 0.75])
            .unwrap();

        assert!(Projection::new("", vec![fixture.axis.clone()]).is_err());
        assert!(Projection::new("empty", Vec::new()).is_err());
        assert!(
            cross_section
                .projection_set(&[], &fixture.components)
                .is_err()
        );
        let duplicate = vec![
            Projection::new("same", vec![fixture.axis.clone()]).unwrap(),
            Projection::new("same", vec![fixture.axis]).unwrap(),
        ];
        reset_projection_evaluation_counts();
        let error = cross_section
            .projection_set(&duplicate, &fixture.components)
            .expect_err("duplicate names must fail");

        assert!(
            error
                .to_string()
                .contains("duplicate projection name: same")
        );
        assert_eq!(projection_evaluation_counts(), (0, 0));
    }

    #[test]
    fn projection_set_execution_errors_report_dataset_and_draw_context() {
        let expression: Expr = parameter!("scale", initial: 1.0).into();
        let model = CompiledModel::from_expr(&expression).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let ensemble = Ensemble::new(vec!["scale".into()], vec![vec![-1.0]]).unwrap();
        let cross_section = likelihood
            .cross_section_with_ensemble("signal", generated, 1.0, vec![1.0], ensemble)
            .unwrap();
        let projections = [Projection::new(
            "x",
            vec![Axis::new(event_scalar("x"), vec![0.0, 1.0]).unwrap()],
        )
        .unwrap()];

        let error = cross_section
            .projection_set(&projections, &HashMap::new())
            .expect_err("a negative draw intensity must fail");
        let message = error.to_string();

        assert!(message.contains("accepted MC"), "{message}");
        assert!(message.contains("ensemble draw 0"), "{message}");
        assert!(message.contains("member `signal`"), "{message}");
        assert!(message.contains("projections [x]"), "{message}");
    }

    #[test]
    fn extended_nll_cross_section_distinguishes_observed_and_fitted_totals() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.25)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let cross_section = likelihood
            .cross_section("signal", generated, 10.0, likelihood.default_params())
            .unwrap();

        assert_relative_eq!(cross_section.observed_total().unwrap().value(), 0.3);
        assert_relative_eq!(cross_section.fitted_total().unwrap().value(), 0.15);
        assert_relative_eq!(
            cross_section.total().unwrap().value(),
            cross_section.observed_total().unwrap().value()
        );
    }

    #[test]
    fn differential_aliases_share_canonical_selection_evaluations() {
        let fixture = canonical_selection_fixture();
        let ensemble = Ensemble::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.6, 0.7], vec![1.4, 0.8]],
        )
        .unwrap();
        let cross_section = fixture
            .likelihood
            .cross_section_with_ensemble(
                "signal",
                fixture.generated,
                10.0,
                fixture.likelihood.default_params(),
                ensemble,
            )
            .unwrap();

        reset_selection_intensity_evaluation_count();
        let differential = cross_section
            .differential(std::slice::from_ref(&fixture.axis), &fixture.components)
            .unwrap();

        assert_eq!(selection_intensity_evaluation_count(), 1);
        assert_eq!(differential.components().len(), 3);
        assert_eq!(
            differential.components()["ordered"].values(),
            differential.components()["reordered"].values()
        );
        assert_eq!(
            differential.components()["ordered"].values(),
            differential.components()["repeated"].values()
        );
        assert_eq!(
            differential.components()["ordered"].draws(),
            differential.components()["reordered"].draws()
        );
        assert_eq!(
            differential.components()["ordered"].draws(),
            differential.components()["repeated"].draws()
        );
    }

    #[test]
    fn combined_differential_deduplicates_selections_per_member() {
        let fixture = canonical_selection_fixture();
        let members = [10.0, 15.0]
            .into_iter()
            .map(|luminosity| {
                fixture
                    .likelihood
                    .cross_section(
                        "signal",
                        fixture.generated.clone(),
                        luminosity,
                        fixture.likelihood.default_params(),
                    )
                    .unwrap()
            })
            .collect();
        let cross_section = CrossSection::combine(members).unwrap();

        reset_selection_intensity_evaluation_count();
        let differential = cross_section
            .differential(std::slice::from_ref(&fixture.axis), &fixture.components)
            .unwrap();

        assert_eq!(selection_intensity_evaluation_count(), 4);
        assert_eq!(differential.components().len(), 3);
        assert_eq!(
            differential.components()["ordered"].values(),
            differential.components()["reordered"].values()
        );
        assert_eq!(
            differential.components()["ordered"].values(),
            differential.components()["repeated"].values()
        );
    }

    #[test]
    fn optimized_bootstrap_differential_matches_individual_replica_evaluations() {
        let x = event_scalar("x");
        let selected = (Expr::from(parameter!("a", initial: 1.5)) * x.clone()).tagged("selected");
        let remainder = Expr::from(parameter!("b", initial: 0.75)).tagged("remainder");
        let model = CompiledModel::from_expr(&(selected + remainder).norm_sqr()).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (0.75, 2.0), (1.25, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 1.0), (1.75, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let ensemble = Ensemble::bootstrap_fit(&likelihood, 3, 73, |replica, index| {
            let mut parameters = replica.default_params();
            parameters[0] += index as f64 * 0.1;
            parameters[1] -= index as f64 * 0.05;
            Ok::<_, std::convert::Infallible>(parameters)
        })
        .unwrap();
        let axis = Axis::new(x, vec![0.0, 1.0, 2.0]).unwrap();
        let components = HashMap::from([("selected".into(), vec!["selected".into()])]);
        let propagated = likelihood
            .cross_section_with_ensemble(
                "signal",
                generated.clone(),
                10.0,
                likelihood.default_params(),
                ensemble.clone(),
            )
            .unwrap()
            .differential(std::slice::from_ref(&axis), &components)
            .unwrap();

        for (index, (replica, parameters)) in
            ensemble.replicas().iter().zip(ensemble.draws()).enumerate()
        {
            let individual = replica
                .cross_section("signal", generated.clone(), 10.0, parameters.clone())
                .unwrap()
                .differential(std::slice::from_ref(&axis), &components)
                .unwrap();
            assert_eq!(propagated.data().draws()[index], individual.data().values());
            assert_eq!(
                propagated.model().draws()[index],
                individual.model().values()
            );
            assert_eq!(
                propagated.components()["selected"].draws()[index],
                individual.components()["selected"].values()
            );
        }
    }

    #[test]
    fn arbitrary_replica_differentials_use_each_replicas_event_rows() {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0)).unwrap();
        let accepted = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let make_likelihood = |data: Dataset| {
            Arc::new(
                Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                    .unwrap(),
            )
        };
        let likelihood = make_likelihood(weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]));
        let replicas = vec![
            make_likelihood(weighted_dataset(&[(0.25, 2.0)])),
            make_likelihood(weighted_dataset(&[(1.25, 3.0)])),
        ];
        let ensemble =
            Ensemble::with_replicas(Vec::new(), vec![Vec::new(), Vec::new()], replicas.clone())
                .unwrap();
        let axis = Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap();
        let propagated = likelihood
            .cross_section_with_ensemble("signal", generated.clone(), 1.0, Vec::new(), ensemble)
            .unwrap()
            .differential(std::slice::from_ref(&axis), &HashMap::new())
            .unwrap();

        for (index, replica) in replicas.iter().enumerate() {
            let individual = replica
                .cross_section("signal", generated.clone(), 1.0, Vec::new())
                .unwrap()
                .differential(std::slice::from_ref(&axis), &HashMap::new())
                .unwrap();
            assert_eq!(propagated.data().draws()[index], individual.data().values());
        }
    }

    #[test]
    fn optimized_combined_differential_matches_explicit_draw_combinations() {
        let x = event_scalar("x");
        let selected = (Expr::from(parameter!("a", initial: 1.5)) * x.clone()).tagged("selected");
        let remainder = Expr::from(parameter!("b", initial: 0.75)).tagged("remainder");
        let model = CompiledModel::from_expr(&(selected + remainder).norm_sqr()).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (0.75, 2.0), (1.25, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 1.0), (1.75, 1.0)]);
        let data_b = weighted_dataset(&[(0.25, 2.0), (0.75, 1.0), (1.75, 2.0)]);
        let accepted_b = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0), (1.75, 1.0)]);
        let generated_b = weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 2.0), (1.75, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::NllTerm::new("period_a", &model, &data, &accepted).unwrap(),
                crate::NllTerm::new("period_b", &model, &data_b, &accepted_b).unwrap(),
            ])
            .unwrap(),
        );
        let ensemble = Ensemble::bootstrap_fit(&likelihood, 3, 91, |replica, index| {
            let mut parameters = replica.default_params();
            parameters[0] += index as f64 * 0.1;
            parameters[1] -= index as f64 * 0.05;
            Ok::<_, std::convert::Infallible>(parameters)
        })
        .unwrap();
        let member_inputs = [
            ("period_a", generated.clone(), 10.0),
            ("period_b", generated_b.clone(), 15.0),
        ];
        let members = member_inputs
            .iter()
            .map(|(name, generated, luminosity)| {
                likelihood
                    .cross_section_with_ensemble(
                        *name,
                        generated.clone(),
                        *luminosity,
                        likelihood.default_params(),
                        ensemble.clone(),
                    )
                    .unwrap()
            })
            .collect();
        let axis = Axis::new(x, vec![0.0, 1.0, 2.0]).unwrap();
        let components = HashMap::from([("selected".into(), vec!["selected".into()])]);
        let propagated = CrossSection::combine(members)
            .unwrap()
            .differential(std::slice::from_ref(&axis), &components)
            .unwrap();

        for (index, (replica, parameters)) in
            ensemble.replicas().iter().zip(ensemble.draws()).enumerate()
        {
            let explicit_members = member_inputs
                .iter()
                .map(|(name, generated, luminosity)| {
                    replica
                        .cross_section(*name, generated.clone(), *luminosity, parameters.clone())
                        .unwrap()
                })
                .collect();
            let explicit = CrossSection::combine(explicit_members)
                .unwrap()
                .differential(std::slice::from_ref(&axis), &components)
                .unwrap();
            assert_eq!(propagated.data().draws()[index], explicit.data().values());
            assert_eq!(propagated.model().draws()[index], explicit.model().values());
            assert_eq!(
                propagated.components()["selected"].draws()[index],
                explicit.components()["selected"].values()
            );
        }
    }
}
