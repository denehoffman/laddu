//! High-level cross-section analyses and uncertainty propagation.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use auto_ops::impl_op_ex;
use laddu_data::data::Dataset;
use laddu_expr::{Expr, ExprNodeStructuralKey};
#[cfg(test)]
use laddu_runtime::flat_bin_index_for_event;
use laddu_runtime::{BinningAxis, DatasetExprExt, Execution, FinalUpperEdge, checked_bin_count};
use rayon::prelude::*;

use crate::{CrossSectionIntegrals, Likelihood, LikelihoodError, LikelihoodResult, Yield};

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

    pub(crate) fn replicas_share_event_rows(&self) -> bool {
        self.replicas_share_event_rows
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
    binning: BinningAxis,
}

impl Axis {
    /// Constructs a differential axis.
    ///
    /// # Errors
    /// Returns an error unless edges are finite and strictly increasing.
    pub fn new(expression: Expr, edges: Vec<f64>) -> LikelihoodResult<Self> {
        let binning = BinningAxis::new(edges.iter().copied()).map_err(|_| {
            invalid("axis edges must contain at least two finite increasing values")
        })?;
        Ok(Self {
            expression,
            binning,
        })
    }

    /// Axis expression.
    pub fn expression(&self) -> &Expr {
        &self.expression
    }

    /// Bin edges.
    pub fn edges(&self) -> &[f64] {
        self.binning.edges()
    }

    /// Number of bins.
    pub fn bins(&self) -> usize {
        self.binning.bin_count()
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
        let bin_axes: Vec<_> = axes.iter().map(|axis| axis.binning.clone()).collect();
        if checked_bin_count(&bin_axes).is_none() {
            return Err(invalid(
                "projection axis shape exceeds addressable bin count",
            ));
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

/// Why a binned acceptance correction is undefined.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum YieldBinValidity {
    /// The bin has sufficient finite model support; zero selected yield is valid.
    Valid,
    /// No generated Monte Carlo event belongs to this bin.
    MissingGeneratedSupport,
    /// No accepted Monte Carlo event belongs to this bin.
    MissingAcceptedSupport,
    /// Accepted model support is zero or negative.
    NonPositiveAcceptedSupport,
    /// Generated model support is zero or negative.
    NonPositiveGeneratedSupport,
    /// Generated integration weights do not define positive local exposure.
    InvalidExposure,
    /// A weighted sum or model intensity was nonfinite.
    NonFiniteEvaluation,
}

/// Counts of coordinates excluded from a central yield projection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct YieldProjectionDiagnostics {
    /// Selected-data rows with nonfinite coordinates.
    pub selected_nonfinite: usize,
    /// Selected-data rows outside the joint axes.
    pub selected_out_of_range: usize,
    /// Accepted Monte Carlo rows with nonfinite coordinates.
    pub accepted_nonfinite: usize,
    /// Accepted Monte Carlo rows outside the joint axes.
    pub accepted_out_of_range: usize,
    /// Generated Monte Carlo rows with nonfinite coordinates.
    pub generated_nonfinite: usize,
    /// Generated Monte Carlo rows outside the joint axes.
    pub generated_out_of_range: usize,
}

/// Central row-major histogram values with their ordered geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct YieldHistogramView {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    values: Vec<f64>,
}

impl YieldHistogramView {
    /// Ordered bin edges.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }
    /// Bin count for each axis, with the last axis varying fastest.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Central row-major bin values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Coherent central selected and fitted yields over one joint set of axes.
#[derive(Clone, Debug)]
pub struct YieldProjection {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    selected: Vec<f64>,
    accepted: Vec<f64>,
    generated: Vec<f64>,
    acceptance: Vec<f64>,
    corrected: Vec<f64>,
    validity: Vec<YieldBinValidity>,
    diagnostics: YieldProjectionDiagnostics,
    has_absolute_rate: bool,
}

impl YieldProjection {
    /// Ordered bin edges.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }
    /// Row-major shape; the last axis varies fastest.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Selected data yield, D, including signed event weights.
    pub fn selected(&self) -> &[f64] {
        &self.selected
    }
    /// Accepted fitted yield, A, or NaN for a shape-only term.
    pub fn accepted(&self) -> &[f64] {
        &self.accepted
    }
    /// Generated fitted yield, G, or NaN for a shape-only term.
    pub fn generated(&self) -> &[f64] {
        &self.generated
    }
    /// Fitted per-bin acceptance, A/G, where defined.
    pub fn acceptance(&self) -> &[f64] {
        &self.acceptance
    }
    /// Fitted-acceptance-corrected selected yield, DG/A, where defined.
    pub fn corrected(&self) -> &[f64] {
        &self.corrected
    }
    /// Per-bin correction validity.
    pub fn validity(&self) -> &[YieldBinValidity] {
        &self.validity
    }
    /// Coordinate exclusion diagnostics for each sample.
    pub fn diagnostics(&self) -> &YieldProjectionDiagnostics {
        &self.diagnostics
    }
    /// Whether the source likelihood defines absolute fitted yields.
    pub fn has_absolute_rate(&self) -> bool {
        self.has_absolute_rate
    }
    /// Materialize a central selected-data histogram view.
    pub fn selected_histogram(&self) -> YieldHistogramView {
        self.histogram(&self.selected)
    }
    /// Materialize a central accepted-model histogram view.
    pub fn accepted_histogram(&self) -> YieldHistogramView {
        self.histogram(&self.accepted)
    }
    /// Materialize a central generated-model histogram view.
    pub fn generated_histogram(&self) -> YieldHistogramView {
        self.histogram(&self.generated)
    }
    /// Materialize a central corrected-yield histogram view.
    pub fn corrected_histogram(&self) -> YieldHistogramView {
        self.histogram(&self.corrected)
    }
    fn histogram(&self, values: &[f64]) -> YieldHistogramView {
        YieldHistogramView {
            axes: self.axes.clone(),
            shape: self.shape.clone(),
            values: values.to_vec(),
        }
    }
}

/// Named central yield projections in request order.
#[derive(Clone, Debug)]
pub struct YieldProjectionSet {
    entries: Vec<(String, YieldProjection)>,
}

impl YieldProjectionSet {
    /// Number of projections.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether the request produced no projections.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Lookup by request name.
    pub fn get(&self, name: &str) -> Option<&YieldProjection> {
        self.entries
            .iter()
            .find_map(|(candidate, result)| (candidate == name).then_some(result))
    }
    /// Iterate in request order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &YieldProjection)> {
        self.entries
            .iter()
            .map(|(name, result)| (name.as_str(), result))
    }
}

impl Yield {
    /// Evaluate one central joint yield projection on demand.
    ///
    /// # Errors
    /// Returns an error for invalid geometry, coordinates, or model evaluation.
    pub fn projection(&self, axes: &[Axis]) -> LikelihoodResult<YieldProjection> {
        let request = Projection::new("yield", axes.to_vec())?;
        Ok(self.projection_set(&[request])?.entries.remove(0).1)
    }

    /// Evaluate independent named central yield projections in request order.
    /// Identical axis groups share bin assignments and intensity traversal.
    ///
    /// # Errors
    /// Returns an error before event traversal for an empty request, repeated
    /// names, or invalid bin volumes. Failed model evaluation is atomic.
    pub fn projection_set(
        &self,
        projections: &[Projection],
    ) -> LikelihoodResult<YieldProjectionSet> {
        if projections.is_empty() {
            return Err(invalid("at least one projection is required"));
        }
        let mut names = HashSet::new();
        for projection in projections {
            if !names.insert(projection.name()) {
                return Err(invalid(format!(
                    "duplicate projection name: {}",
                    projection.name()
                )));
            }
            if !valid_bin_volumes(projection.axes()) {
                return Err(invalid(format!(
                    "projection `{}` has invalid bin volume",
                    projection.name()
                )));
            }
        }
        let execution = self.likelihood().execution();
        let mut seen_expressions = HashSet::new();
        let mut expressions = Vec::new();
        for projection in projections {
            for axis in projection.axes() {
                let graph = axis.expression.to_graph();
                let key = (
                    graph.root().index(),
                    graph
                        .nodes()
                        .iter()
                        .map(|node| node.structural_key())
                        .collect::<Vec<_>>(),
                );
                if seen_expressions.insert(key) {
                    expressions.push(axis.expression.clone());
                }
            }
        }
        self.observed_data()
            .validate_real_expressions(&expressions, execution)?;
        let integrals = self
            .likelihood()
            .cross_section_integrals(self.term_name(), self.generated_mc())?;
        let data = self.observed_data();
        let accepted = integrals.accepted_mc_source();
        let generated = integrals.generated_mc_source();
        let (unique, indexes) = deduplicate_projections(projections);
        let sample_events = [data, accepted, generated]
            .iter()
            .map(|sample| {
                usize::try_from(sample.stats()?.events())
                    .map_err(|_| invalid("projection event count exceeds addressable memory"))
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let event_total = sample_events
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let bins_total = unique
            .iter()
            .try_fold(0usize, |sum, projection| {
                let bins = checked_bin_count(
                    &projection
                        .axes()
                        .iter()
                        .map(|axis| axis.binning.clone())
                        .collect::<Vec<_>>(),
                )?;
                sum.checked_add(bins)
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let output_bins = projections
            .iter()
            .try_fold(0usize, |sum, projection| {
                let bins = checked_bin_count(
                    &projection
                        .axes()
                        .iter()
                        .map(|axis| axis.binning.clone())
                        .collect::<Vec<_>>(),
                )?;
                sum.checked_add(bins)
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let largest_sample = sample_events.iter().copied().max().unwrap_or(0);
        let distinct_axes = expressions.len();
        let workspace_bytes = event_total
            .checked_mul(unique.len())
            .and_then(|count| count.checked_mul(std::mem::size_of::<Option<usize>>() + 1))
            .and_then(|bytes| {
                bytes.checked_add(event_total.checked_mul(std::mem::size_of::<f64>())?)
            })
            .and_then(|bytes| {
                bytes.checked_add(bins_total.checked_mul(3 * std::mem::size_of::<f64>())?)
            })
            .and_then(|bytes| {
                bytes.checked_add(output_bins.checked_mul(7 * std::mem::size_of::<f64>())?)
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    distinct_axes
                        .checked_mul(largest_sample.min(8192))?
                        .checked_mul(32)?,
                )
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let workspace_bytes = u64::try_from(workspace_bytes)
            .map_err(|_| invalid("projection workspace size overflow"))?;
        let _workspace_lease =
            execution
                .host_memory()
                .reserve(workspace_bytes)
                .map_err(|error| {
                    LikelihoodError::Runtime(laddu_runtime::RuntimeError::Memory(error))
                })?;
        let data_weights = dataset_weights(data)?;
        let accepted_weights = dataset_weights(accepted)?;
        let generated_weights = dataset_weights(generated)?;
        let data_assignments = evaluate_bin_assignments_many(data, &unique, execution)?;
        let accepted_assignments = evaluate_bin_assignments_many(accepted, &unique, execution)?;
        let generated_assignments = evaluate_bin_assignments_many(generated, &unique, execution)?;
        let plans = unique
            .iter()
            .zip(data_assignments)
            .zip(accepted_assignments)
            .zip(generated_assignments)
            .map(
                |(((projection, data_bins), accepted_bins), generated_bins)| {
                    Ok(PreparedProjection {
                        name: projection.name().to_owned(),
                        request_axes: projection.axes().to_vec(),
                        axes: projection
                            .axes()
                            .iter()
                            .map(|axis| axis.edges().to_vec())
                            .collect(),
                        shape: projection.axes().iter().map(Axis::bins).collect(),
                        volumes: bin_volumes(projection.axes()),
                        data_bins,
                        accepted_bins,
                        generated_bins,
                    })
                },
            )
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let mut accepted_sums = plans
            .iter()
            .map(|plan| vec![0.0; plan.accepted_bins.count])
            .collect::<Vec<_>>();
        let mut generated_sums = plans
            .iter()
            .map(|plan| vec![0.0; plan.generated_bins.count])
            .collect::<Vec<_>>();
        let parameters = [self.parameters()];
        let contexts = ["central yield projection".to_owned()];
        integrals.visit_accepted_raw_prepared_intensities_many(
            &parameters,
            &contexts,
            |offset, _, intensities| {
                for (plan, sums) in plans.iter().zip(&mut accepted_sums) {
                    plan.accepted_bins.accumulate_weighted_block(
                        offset,
                        &accepted_weights,
                        intensities,
                        sums,
                    );
                }
            },
        )?;
        integrals.visit_generated_prepared_intensities_many(
            &parameters,
            &contexts,
            |offset, _, intensities| {
                for (plan, sums) in plans.iter().zip(&mut generated_sums) {
                    plan.generated_bins.accumulate_weighted_block(
                        offset,
                        &generated_weights,
                        intensities,
                        sums,
                    );
                }
            },
        )?;
        let unique_results = plans
            .iter()
            .enumerate()
            .map(|(plan_index, plan)| {
                let mut selected = plan.data_bins.accumulate_products(&data_weights, None);
                let accepted_raw = &accepted_sums[plan_index];
                let generated_raw = &generated_sums[plan_index];
                let accepted_counts = plan.accepted_bins.support_counts();
                let generated_counts = plan.generated_bins.support_counts();
                let generated_exposure = plan
                    .generated_bins
                    .accumulate_products(&generated_weights, None);
                let mut validity = Vec::with_capacity(selected.len());
                let mut acceptance = Vec::with_capacity(selected.len());
                let mut corrected = Vec::with_capacity(selected.len());
                for bin in 0..selected.len() {
                    let d = selected[bin];
                    let a = accepted_raw[bin];
                    let g = generated_raw[bin];
                    let status = if !d.is_finite() || !a.is_finite() || !g.is_finite() {
                        YieldBinValidity::NonFiniteEvaluation
                    } else if generated_counts[bin] == 0 {
                        YieldBinValidity::MissingGeneratedSupport
                    } else if !generated_exposure[bin].is_finite() || generated_exposure[bin] <= 0.0
                    {
                        YieldBinValidity::InvalidExposure
                    } else if accepted_counts[bin] == 0 {
                        YieldBinValidity::MissingAcceptedSupport
                    } else if a <= 0.0 {
                        YieldBinValidity::NonPositiveAcceptedSupport
                    } else if g <= 0.0 {
                        YieldBinValidity::NonPositiveGeneratedSupport
                    } else {
                        YieldBinValidity::Valid
                    };
                    validity.push(status);
                    if !d.is_finite() {
                        selected[bin] = f64::NAN;
                    }
                    if status == YieldBinValidity::Valid {
                        acceptance.push(a / g);
                        corrected.push(d * g / a);
                    } else {
                        acceptance.push(f64::NAN);
                        corrected.push(f64::NAN);
                    }
                }
                YieldProjection {
                    axes: plan.axes.clone(),
                    shape: plan.shape.clone(),
                    selected,
                    accepted: accepted_raw
                        .iter()
                        .enumerate()
                        .map(|(bin, &value)| {
                            if self.has_absolute_rate()
                                && accepted_counts[bin] > 0
                                && value.is_finite()
                                && value > 0.0
                            {
                                value
                            } else {
                                f64::NAN
                            }
                        })
                        .collect(),
                    generated: generated_raw
                        .iter()
                        .enumerate()
                        .map(|(bin, &value)| {
                            if self.has_absolute_rate()
                                && generated_counts[bin] > 0
                                && generated_exposure[bin].is_finite()
                                && generated_exposure[bin] > 0.0
                                && value.is_finite()
                                && value > 0.0
                            {
                                value
                            } else {
                                f64::NAN
                            }
                        })
                        .collect(),
                    acceptance,
                    corrected,
                    validity,
                    diagnostics: YieldProjectionDiagnostics {
                        selected_nonfinite: plan.data_bins.nonfinite_count,
                        selected_out_of_range: plan.data_bins.out_of_range_count,
                        accepted_nonfinite: plan.accepted_bins.nonfinite_count,
                        accepted_out_of_range: plan.accepted_bins.out_of_range_count,
                        generated_nonfinite: plan.generated_bins.nonfinite_count,
                        generated_out_of_range: plan.generated_bins.out_of_range_count,
                    },
                    has_absolute_rate: self.has_absolute_rate(),
                }
            })
            .collect::<Vec<_>>();
        Ok(YieldProjectionSet {
            entries: projections
                .iter()
                .zip(indexes)
                .map(|(request, index)| (request.name().to_owned(), unique_results[index].clone()))
                .collect(),
        })
    }
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

/// Full-model and named tagged scalar totals evaluated as one request.
#[derive(Clone, Debug)]
pub struct TotalSet {
    full: Estimate,
    components: HashMap<String, Estimate>,
}

impl TotalSet {
    /// Returns the full-model total.
    pub fn full(&self) -> &Estimate {
        &self.full
    }

    /// Returns every named component total.
    pub fn components(&self) -> &HashMap<String, Estimate> {
        &self.components
    }

    /// Looks up a named component total.
    pub fn get(&self, name: &str) -> Option<&Estimate> {
        self.components.get(name)
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

/// Policy controlling retention of prepared cross-section integrals.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum IntegralRetentionPolicy {
    /// Preserve the historical behavior and retain every preparation.
    #[default]
    Unbounded,
    /// Retain preparations up to the given aggregate resident-byte bound.
    Bounded {
        /// Maximum aggregate resident bytes retained by the integral cache.
        max_bytes: usize,
    },
    /// Evaluate preparations transiently without retaining cache entries.
    None,
}

#[derive(Clone)]
struct IntegralCacheEntry {
    integrals: CrossSectionIntegrals,
    last_used: u64,
}

#[derive(Default)]
struct IntegralCacheState {
    entries: HashMap<IntegralCacheKey, IntegralCacheEntry>,
    policy: IntegralRetentionPolicy,
    clock: u64,
    evictions: u64,
}

impl IntegralCacheState {
    fn retained_bytes(&self) -> usize {
        self.entries
            .values()
            .map(|entry| entry.integrals.resident_bytes())
            .fold(0, usize::saturating_add)
    }

    fn evict_to(&mut self, max_bytes: usize) {
        while self.retained_bytes() > max_bytes {
            if !self.evict_lru() {
                break;
            }
        }
    }

    fn evict_for_transient(&mut self, required_bytes: u64, pool: &laddu_runtime::MemoryPool) {
        while pool.report().remaining_bytes < required_bytes {
            if !self.evict_lru() {
                break;
            }
        }
    }

    fn evict_lru(&mut self) -> bool {
        let Some(key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
        else {
            return false;
        };
        self.entries.remove(&key);
        self.evictions = self.evictions.saturating_add(1);
        true
    }

    fn get(&mut self, key: &IntegralCacheKey) -> Option<CrossSectionIntegrals> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(key)?;
        entry.last_used = self.clock;
        Some(entry.integrals.clone())
    }

    fn insert(&mut self, key: IntegralCacheKey, integrals: CrossSectionIntegrals) {
        let max_bytes = match self.policy {
            IntegralRetentionPolicy::Unbounded => usize::MAX,
            IntegralRetentionPolicy::Bounded { max_bytes } => max_bytes,
            IntegralRetentionPolicy::None => return,
        };
        if integrals.resident_bytes() > max_bytes {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(
            key,
            IntegralCacheEntry {
                integrals,
                last_used: self.clock,
            },
        );
        self.evict_to(max_bytes);
    }
}

type IntegralCache = Arc<Mutex<IntegralCacheState>>;

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
    full_requests: Arc<AtomicU64>,
    tagged_requests: Arc<AtomicU64>,
    central_requests: Arc<AtomicU64>,
    shared_bootstrap_requests: Arc<AtomicU64>,
    arbitrary_replica_requests: Arc<AtomicU64>,
}

/// Integral-preparation cache statistics for a cross-section analysis.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CrossSectionDiagnostics {
    cache_hits: u64,
    cache_misses: u64,
    cached_integrals: usize,
    prepared_bytes: usize,
    cache_evictions: u64,
    reserved_bytes: u64,
    high_water_bytes: u64,
    estimated_prepared_bytes: usize,
    full_requests: u64,
    tagged_requests: u64,
    central_requests: u64,
    shared_bootstrap_requests: u64,
    arbitrary_replica_requests: u64,
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
    /// Returns estimated resident bytes of retained integral records.
    pub fn prepared_bytes(&self) -> usize {
        self.prepared_bytes
    }
    /// Number of integral records evicted by retention or memory pressure.
    pub fn cache_evictions(&self) -> u64 {
        self.cache_evictions
    }
    /// Current reservation in the analysis host-memory pool, including other users of that pool.
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }
    /// Highest concurrent reservation observed by that pool, including other users.
    pub fn high_water_bytes(&self) -> u64 {
        self.high_water_bytes
    }
    /// Estimated prepared bytes still owned by the analysis, including its baseline.
    pub fn estimated_prepared_bytes(&self) -> usize {
        self.estimated_prepared_bytes
    }
    /// Full-model integral requests.
    pub fn full_requests(&self) -> u64 {
        self.full_requests
    }
    /// Tagged integral requests.
    pub fn tagged_requests(&self) -> u64 {
        self.tagged_requests
    }
    /// Requests using the central likelihood.
    pub fn central_requests(&self) -> u64 {
        self.central_requests
    }
    /// Bootstrap replicas reusing central prepared rows.
    pub fn shared_bootstrap_requests(&self) -> u64 {
        self.shared_bootstrap_requests
    }
    /// Replica requests requiring their own integral preparation.
    pub fn arbitrary_replica_requests(&self) -> u64 {
        self.arbitrary_replica_requests
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
    projections: Vec<CombinedPreparedProjection>,
    data_weights: Vec<f64>,
    accepted_weights: Vec<f64>,
    generated_weights: Vec<f64>,
}

struct CombinedPreparedProjection {
    name: String,
    request_axes: Vec<Axis>,
    data_bins: BinAssignments,
    accepted_bins: BinAssignments,
    generated_bins: BinAssignments,
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
    nonfinite_count: usize,
    out_of_range_count: usize,
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
                        edges: axis
                            .binning
                            .edges()
                            .iter()
                            .map(|edge| edge.to_bits())
                            .collect(),
                    }
                })
                .collect(),
        )
    }
}

fn deduplicate_projections(projections: &[Projection]) -> (Vec<&Projection>, Vec<usize>) {
    let mut unique_indexes = HashMap::with_capacity(projections.len());
    let mut unique = Vec::with_capacity(projections.len());
    let indexes = projections
        .iter()
        .map(|projection| {
            let key = ProjectionKey::new(projection.axes());
            if let Some(index) = unique_indexes.get(&key) {
                return *index;
            }
            let index = unique.len();
            unique.push(projection);
            unique_indexes.insert(key, index);
            index
        })
        .collect();
    (unique, indexes)
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
    #[cfg(test)]
    fn new(values: &[Vec<f64>], axes: &[Axis]) -> Self {
        let event_count = values.first().map_or(0, Vec::len);
        let bin_axes: Vec<_> = axes.iter().map(|axis| axis.binning.clone()).collect();
        debug_assert!(
            values
                .iter()
                .all(|coordinates| coordinates.len() == event_count)
        );
        let mut nonfinite_count = 0;
        let mut out_of_range_count = 0;
        let indices = (0..event_count)
            .map(|event| {
                if values.iter().any(|axis| !axis[event].is_finite()) {
                    nonfinite_count += 1;
                    return None;
                }
                let index =
                    flat_bin_index_for_event(&bin_axes, values, event, FinalUpperEdge::Exclusive);
                if index.is_none() {
                    out_of_range_count += 1;
                }
                index
            })
            .collect();
        Self {
            indices,
            count: checked_bin_count(&bin_axes).expect("projection shape was validated"),
            nonfinite_count,
            out_of_range_count,
        }
    }

    fn support_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.count];
        for index in self.indices.iter().flatten() {
            counts[*index] += 1;
        }
        counts
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
        let worker_count = rayon::current_num_threads().min(intensities.len());
        if rayon::current_thread_index().is_none() || worker_count < 2 {
            for (row, &intensity) in intensities.iter().enumerate() {
                let event = offset + row;
                if let Some(index) = self.indices[event] {
                    bins[index] += weights[event] * intensity;
                }
            }
            return;
        }
        let chunk_size = intensities.len().div_ceil(worker_count);
        let chunk_count = intensities.len().div_ceil(chunk_size);
        let partials = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(intensities.len());
                let mut partial = vec![0.0; self.count];
                for (row, &intensity) in intensities[start..end].iter().enumerate() {
                    let event = offset + start + row;
                    if let Some(index) = self.indices[event] {
                        partial[index] += weights[event] * intensity;
                    }
                }
                partial
            })
            .collect::<Vec<_>>();
        for partial in partials {
            for (bin, value) in bins.iter_mut().zip(partial) {
                *bin += value;
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
        projections: &[&Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<Self> {
        let execution = member.likelihood.execution();
        let (data, _) = member.likelihood.intensity_datasets(&member.term_name)?;
        let full = member.integrals_for(&member.likelihood, None)?;
        let component_integrals =
            CanonicalComponents::prepare(member, &member.likelihood, components)?;
        let prepared_projections = projections
            .iter()
            .map(|projection| {
                Ok(CombinedPreparedProjection {
                    name: projection.name().to_owned(),
                    request_axes: projection.axes().to_vec(),
                    data_bins: evaluate_bin_assignments(data, projection.axes(), execution)
                        .map_err(|error| {
                            invalid(format!(
                                "projection `{}` data bin preparation failed: {error}",
                                projection.name()
                            ))
                        })?,
                    accepted_bins: evaluate_bin_assignments(
                        full.accepted_mc_source(),
                        projection.axes(),
                        execution,
                    )
                    .map_err(|error| {
                        invalid(format!(
                            "projection `{}` accepted MC bin preparation failed: {error}",
                            projection.name()
                        ))
                    })?,
                    generated_bins: evaluate_bin_assignments(
                        full.generated_mc_source(),
                        projection.axes(),
                        execution,
                    )
                    .map_err(|error| {
                        invalid(format!(
                            "projection `{}` generated MC bin preparation failed: {error}",
                            projection.name()
                        ))
                    })?,
                })
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        Ok(Self {
            luminosity: member.luminosity,
            projections: prepared_projections,
            data_weights: dataset_weights(data)
                .map_err(|error| invalid(format!("data weights: {error}")))?,
            accepted_weights: dataset_weights(full.accepted_mc_source())
                .map_err(|error| invalid(format!("accepted MC weights: {error}")))?,
            generated_weights: dataset_weights(full.generated_mc_source())
                .map_err(|error| invalid(format!("generated MC weights: {error}")))?,
            full,
            components: component_integrals,
        })
    }

    fn evaluate_projections_with_draws(
        &self,
        member: &CrossSection,
        factor: &Estimate,
        draw_count: usize,
        reference_source: Option<u64>,
        position: usize,
    ) -> LikelihoodResult<Vec<Vec<CombinedMemberValues>>> {
        let ensemble = member.ensemble.as_ref();
        let draw_indexes = (0..draw_count)
            .map(|index| {
                ensemble.map(|ensemble| {
                    paired_draw_index(
                        index,
                        position,
                        ensemble.len(),
                        Some(ensemble.source_id),
                        reference_source,
                    )
                })
            })
            .collect::<Vec<_>>();
        let parameter_sets = std::iter::once(member.parameters.as_slice())
            .chain(draw_indexes.iter().map(|draw_index| {
                draw_index
                    .and_then(|draw_index| ensemble.and_then(|value| value.draws.get(draw_index)))
                    .map(Vec::as_slice)
                    .unwrap_or(&member.parameters)
            }))
            .collect::<Vec<_>>();
        let parameter_contexts = std::iter::once("central value".to_owned())
            .chain((0..draw_count).map(|index| format!("ensemble draw {index}")))
            .collect::<Vec<_>>();

        let mut accepted_histograms = self
            .projections
            .iter()
            .map(|projection| vec![vec![0.0; projection.accepted_bins.count]; parameter_sets.len()])
            .collect::<Vec<_>>();
        record_prepared_intensity_evaluation();
        let full_accepted = self
            .full
            .visit_accepted_prepared_intensities_many(
                &parameter_sets,
                &parameter_contexts,
                |offset, parameter_index, intensities| {
                    for (projection, histograms) in
                        self.projections.iter().zip(&mut accepted_histograms)
                    {
                        projection.accepted_bins.accumulate_weighted_block(
                            offset,
                            &self.accepted_weights,
                            intensities,
                            &mut histograms[parameter_index],
                        );
                    }
                },
            )
            .map_err(|error| invalid(format!("accepted MC intensity evaluation: {error}")))?;
        let mut generated_histograms = self
            .projections
            .iter()
            .map(|projection| {
                vec![vec![0.0; projection.generated_bins.count]; parameter_sets.len()]
            })
            .collect::<Vec<_>>();
        record_prepared_intensity_evaluation();
        self.full
            .visit_generated_prepared_intensities_many(
                &parameter_sets,
                &parameter_contexts,
                |offset, parameter_index, intensities| {
                    for (projection, histograms) in
                        self.projections.iter().zip(&mut generated_histograms)
                    {
                        projection.generated_bins.accumulate_weighted_block(
                            offset,
                            &self.generated_weights,
                            intensities,
                            &mut histograms[parameter_index],
                        );
                    }
                },
            )
            .map_err(|error| invalid(format!("generated MC intensity evaluation: {error}")))?;

        let mut component_histograms = HashMap::new();
        for (canonical_tags, selected) in &self.components.integrals {
            let mut accepted = self
                .projections
                .iter()
                .map(|projection| {
                    vec![vec![0.0; projection.accepted_bins.count]; parameter_sets.len()]
                })
                .collect::<Vec<_>>();
            record_selection_intensity_evaluation();
            record_prepared_intensity_evaluation();
            selected
                .visit_accepted_prepared_intensities_many(
                    &parameter_sets,
                    &parameter_contexts,
                    |offset, parameter_index, intensities| {
                        for (projection, histograms) in self.projections.iter().zip(&mut accepted) {
                            projection.accepted_bins.accumulate_weighted_block(
                                offset,
                                &self.accepted_weights,
                                intensities,
                                &mut histograms[parameter_index],
                            );
                        }
                    },
                )
                .map_err(|error| {
                    invalid(format!(
                        "accepted MC component {:?} intensity evaluation: {error}",
                        canonical_tags.as_slice()
                    ))
                })?;
            let mut generated = self
                .projections
                .iter()
                .map(|projection| {
                    vec![vec![0.0; projection.generated_bins.count]; parameter_sets.len()]
                })
                .collect::<Vec<_>>();
            record_selection_intensity_evaluation();
            record_prepared_intensity_evaluation();
            selected
                .visit_generated_prepared_intensities_many(
                    &parameter_sets,
                    &parameter_contexts,
                    |offset, parameter_index, intensities| {
                        for (projection, histograms) in self.projections.iter().zip(&mut generated)
                        {
                            projection.generated_bins.accumulate_weighted_block(
                                offset,
                                &self.generated_weights,
                                intensities,
                                &mut histograms[parameter_index],
                            );
                        }
                    },
                )
                .map_err(|error| {
                    invalid(format!(
                        "generated MC component {:?} intensity evaluation: {error}",
                        canonical_tags.as_slice()
                    ))
                })?;
            component_histograms.insert(canonical_tags.clone(), (accepted, generated));
        }

        let draw_data = draw_indexes
            .iter()
            .enumerate()
            .map(|(index, draw_index)| {
                let prepare = || {
                    let replica_data = draw_index
                        .and_then(|draw_index| {
                            ensemble.and_then(|value| value.replicas.get(draw_index))
                        })
                        .map(|likelihood| likelihood.intensity_datasets(&member.term_name))
                        .transpose()
                        .map_err(|error| invalid(format!("replica data lookup: {error}")))?
                        .map(|(data, _)| data);
                    let weights = replica_data
                        .map(dataset_weights)
                        .transpose()
                        .map_err(|error| invalid(format!("replica data weights: {error}")))?;
                    let histograms = self
                        .projections
                        .iter()
                        .map(|projection| {
                            let bins = match ensemble {
                                Some(ensemble) => ensemble.replica_bin_assignments(
                                    replica_data,
                                    &projection.request_axes,
                                    member.likelihood.execution(),
                                ),
                                None => Ok(None),
                            }
                            .map_err(|error| {
                                invalid(format!(
                                    "projection `{}` replica data bin preparation: {error}",
                                    projection.name
                                ))
                            })?;
                            Ok(bins
                                .as_ref()
                                .unwrap_or(&projection.data_bins)
                                .accumulate_products(
                                    weights.as_deref().unwrap_or(&self.data_weights),
                                    None,
                                ))
                        })
                        .collect::<LikelihoodResult<Vec<_>>>()?;
                    Ok((
                        histograms,
                        weights
                            .as_ref()
                            .map(|values| values.iter().sum())
                            .unwrap_or_else(|| self.full.data_weight_sum()),
                    ))
                };
                prepare().map_err(|error: LikelihoodError| {
                    invalid(format!(
                        "member `{}` draw {index}: {error}",
                        member.term_name
                    ))
                })
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let factors = std::iter::once(factor.central)
            .chain((0..draw_count).map(|index| {
                if factor.draws.is_empty() {
                    factor.central
                } else {
                    let factor_index = paired_draw_index(
                        index,
                        position,
                        factor.draws.len(),
                        factor.source_id,
                        reference_source,
                    );
                    factor.draws[factor_index]
                }
            }))
            .collect::<Vec<_>>();

        Ok((0..parameter_sets.len())
            .map(|parameter_index| {
                self.projections
                    .iter()
                    .enumerate()
                    .map(|(projection_index, projection)| {
                        let (data_histogram, total_data) = if parameter_index == 0 {
                            (
                                projection
                                    .data_bins
                                    .accumulate_products(&self.data_weights, None),
                                self.full.data_weight_sum(),
                            )
                        } else {
                            let (histograms, total_data) = &draw_data[parameter_index - 1];
                            (histograms[projection_index].clone(), *total_data)
                        };
                        let accepted = &accepted_histograms[projection_index][parameter_index];
                        let generated = &generated_histograms[projection_index][parameter_index];
                        let exposures = binned_exposures(
                            self.luminosity * factors[parameter_index],
                            accepted,
                            generated,
                        );
                        let canonical_values = self
                            .components
                            .integrals
                            .keys()
                            .map(|tags| {
                                let (accepted_histograms, generated_histograms) =
                                    &component_histograms[tags];
                                let selected_accepted =
                                    &accepted_histograms[projection_index][parameter_index];
                                let selected_generated =
                                    &generated_histograms[projection_index][parameter_index];
                                (
                                    tags.clone(),
                                    BinnedMeasurement {
                                        yields: selected_accepted
                                            .iter()
                                            .map(|value| {
                                                total_data * value / full_accepted[parameter_index]
                                            })
                                            .collect(),
                                        exposures: binned_exposures(
                                            self.luminosity * factors[parameter_index],
                                            selected_accepted,
                                            selected_generated,
                                        ),
                                    },
                                )
                            })
                            .collect::<HashMap<_, _>>();
                        CombinedMemberValues {
                            data: BinnedMeasurement {
                                yields: data_histogram,
                                exposures: exposures.clone(),
                            },
                            model: BinnedMeasurement {
                                yields: accepted
                                    .iter()
                                    .map(|value| {
                                        total_data * value / full_accepted[parameter_index]
                                    })
                                    .collect(),
                                exposures,
                            },
                            components: self
                                .components
                                .aliases
                                .iter()
                                .map(|(name, tags)| (name.clone(), canonical_values[tags].clone()))
                                .collect(),
                        }
                    })
                    .collect()
            })
            .collect())
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
        let mut integral_cache = IntegralCacheState::default();
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
            full_requests: Arc::new(AtomicU64::new(1)),
            tagged_requests: Default::default(),
            central_requests: Arc::new(AtomicU64::new(1)),
            shared_bootstrap_requests: Default::default(),
            arbitrary_replica_requests: Default::default(),
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
            full_requests: template.full_requests,
            tagged_requests: template.tagged_requests,
            central_requests: template.central_requests,
            shared_bootstrap_requests: template.shared_bootstrap_requests,
            arbitrary_replica_requests: template.arbitrary_replica_requests,
        })
    }

    /// Full-model observed-yield-normalized cross section.
    ///
    /// # Errors
    /// Returns an error when model integrals or ensemble evaluation fail.
    pub fn observed_total(&self) -> LikelihoodResult<Estimate> {
        self.observed_total_selected(None)
    }

    /// Full-model observed-yield-normalized cross section without uncertainty draws.
    ///
    /// This operation does not prepare or evaluate ensemble replicas.
    ///
    /// # Errors
    /// Returns an error when model integrals or central evaluation fail.
    pub fn observed_total_central(&self) -> LikelihoodResult<Estimate> {
        self.observed_total_central_selected(None)
    }

    /// Returns integral-cache hit, miss, count, and retained-byte diagnostics.
    pub fn diagnostics(&self) -> CrossSectionDiagnostics {
        fn collect<'a>(section: &'a CrossSection, leaves: &mut Vec<&'a CrossSection>) {
            if let Some(members) = &section.members {
                for (member, _) in members.iter() {
                    collect(member, leaves);
                }
            } else {
                leaves.push(section);
            }
        }
        let mut leaves = Vec::new();
        collect(self, &mut leaves);
        let mut seen_caches = HashSet::new();
        let mut seen_pools = Vec::new();
        let mut total = CrossSectionDiagnostics::default();
        for section in leaves {
            if seen_caches.insert(Arc::as_ptr(&section.integral_cache)) {
                let cache = section
                    .integral_cache
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                total.cached_integrals = total.cached_integrals.saturating_add(cache.entries.len());
                let bytes = cache.retained_bytes();
                total.prepared_bytes = total.prepared_bytes.saturating_add(bytes);
                let baseline = section.full_integrals.resident_bytes();
                total.estimated_prepared_bytes = total.estimated_prepared_bytes.saturating_add(
                    if cache
                        .entries
                        .contains_key(&(Arc::as_ptr(&section.likelihood) as usize, None))
                    {
                        bytes
                    } else {
                        bytes.saturating_add(baseline)
                    },
                );
                total.cache_evictions = total.cache_evictions.saturating_add(cache.evictions);
                total.cache_hits = total
                    .cache_hits
                    .saturating_add(section.cache_hits.load(Ordering::Relaxed));
                total.cache_misses = total
                    .cache_misses
                    .saturating_add(section.cache_misses.load(Ordering::Relaxed));
                total.full_requests = total
                    .full_requests
                    .saturating_add(section.full_requests.load(Ordering::Relaxed));
                total.tagged_requests = total
                    .tagged_requests
                    .saturating_add(section.tagged_requests.load(Ordering::Relaxed));
                total.central_requests = total
                    .central_requests
                    .saturating_add(section.central_requests.load(Ordering::Relaxed));
                total.shared_bootstrap_requests = total
                    .shared_bootstrap_requests
                    .saturating_add(section.shared_bootstrap_requests.load(Ordering::Relaxed));
                total.arbitrary_replica_requests = total
                    .arbitrary_replica_requests
                    .saturating_add(section.arbitrary_replica_requests.load(Ordering::Relaxed));
            }
            let pool = section.likelihood.execution().host_memory();
            if !seen_pools
                .iter()
                .any(|seen: &laddu_runtime::MemoryPool| seen.shares_reservations_with(pool))
            {
                seen_pools.push(pool.clone());
                let report = pool.report();
                total.reserved_bytes = total.reserved_bytes.saturating_add(report.reserved_bytes);
                total.high_water_bytes = total
                    .high_water_bytes
                    .saturating_add(report.high_water_bytes);
            }
        }
        total
    }

    /// Changes the integral-retention policy and immediately enforces its bound.
    pub fn set_integral_retention(&self, policy: IntegralRetentionPolicy) {
        if let Some(members) = &self.members {
            let member_policy = match policy {
                IntegralRetentionPolicy::Bounded { max_bytes } => {
                    IntegralRetentionPolicy::Bounded {
                        max_bytes: max_bytes / members.len(),
                    }
                }
                other => other,
            };
            for (member, _) in members.iter() {
                member.set_integral_retention(member_policy);
            }
            return;
        }
        let mut cache = self
            .integral_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cache.policy = policy;
        match policy {
            IntegralRetentionPolicy::Unbounded => {}
            IntegralRetentionPolicy::Bounded { max_bytes } => cache.evict_to(max_bytes),
            IntegralRetentionPolicy::None => cache.entries.clear(),
        }
    }

    /// Drops all eligible retained integral preparations.
    ///
    /// The original full-model preparation remains owned by the cross section so
    /// the analysis remains usable; subsequent tagged or replica preparations
    /// follow the current retention policy.
    pub fn clear_integral_cache(&self) {
        if let Some(members) = &self.members {
            for (member, _) in members.iter() {
                member.clear_integral_cache();
            }
            return;
        }
        self.integral_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entries
            .clear();
    }

    /// Tag-narrowed observed-yield-normalized cross section.
    ///
    /// # Errors
    /// Returns an error when tag projection, integrals, or ensemble evaluation fail.
    pub fn observed_total_with_tags(&self, tags: &[String]) -> LikelihoodResult<Estimate> {
        self.observed_total_selected(Some(tags))
    }

    /// Tag-narrowed observed-yield-normalized cross section without uncertainty draws.
    ///
    /// This operation does not prepare or evaluate ensemble replicas.
    ///
    /// # Errors
    /// Returns an error when tag projection, integrals, or central evaluation fail.
    pub fn observed_total_central_with_tags(&self, tags: &[String]) -> LikelihoodResult<Estimate> {
        self.observed_total_central_selected(Some(tags))
    }

    fn observed_total_selected(&self, tags: Option<&[String]>) -> LikelihoodResult<Estimate> {
        if self.members.is_some() {
            return self.combined_total(tags);
        }
        self.evaluate_estimate(tags, |integrals, parameters| {
            integrals.observed_cross_section(parameters, self.luminosity)
        })
    }

    fn observed_total_central_selected(
        &self,
        tags: Option<&[String]>,
    ) -> LikelihoodResult<Estimate> {
        let value = if self.members.is_some() {
            self.combined_central_value(tags)?
        } else {
            let integrals = self.integrals_for(&self.likelihood, tags)?;
            integrals.observed_cross_section(&self.parameters, self.luminosity)?
        };
        Estimate::central(value)
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

    /// Evaluates the full-model total and named tagged component totals together.
    ///
    /// Reordered and repeated forms of the same tag selection are evaluated once.
    ///
    /// # Errors
    /// Returns an error when a component selection or its scalar evaluation fails.
    pub fn total_set(
        &self,
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<TotalSet> {
        if components.keys().any(String::is_empty) {
            return Err(invalid("component names must not be empty"));
        }
        if self.members.is_none()
            && self.ensemble.as_ref().is_none_or(|ensemble| {
                ensemble.replicas().is_empty() || ensemble.replicas_share_event_rows()
            })
        {
            return self.single_total_set(components);
        }
        self.fallback_total_set(components)
    }

    fn single_total_set(
        &self,
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<TotalSet> {
        let parameter_sets = std::iter::once(self.parameters.as_slice())
            .chain(
                self.ensemble
                    .as_ref()
                    .into_iter()
                    .flat_map(|ensemble| ensemble.draws().iter().map(Vec::as_slice)),
            )
            .collect::<Vec<_>>();
        let parameter_contexts = std::iter::once("central parameters".to_owned())
            .chain(
                (0..parameter_sets.len().saturating_sub(1))
                    .map(|index| format!("ensemble draw {index}")),
            )
            .collect::<Vec<_>>();
        let full_integrals = self.integrals_for(&self.likelihood, None)?;
        let full_accepted = if full_integrals.has_absolute_rate() {
            parameter_sets
                .iter()
                .map(|parameters| full_integrals.full_accepted_integral(parameters))
                .collect::<LikelihoodResult<Vec<_>>>()?
        } else {
            record_prepared_intensity_evaluation();
            full_integrals.visit_accepted_prepared_intensities_many(
                &parameter_sets,
                &parameter_contexts,
                |_, _, _| {},
            )?
        };
        let data_weight_sums = std::iter::once(Ok(full_integrals.data_weight_sum()))
            .chain(
                self.ensemble
                    .as_ref()
                    .into_iter()
                    .flat_map(|ensemble| ensemble.replicas().iter())
                    .map(|replica| replica.intensity_data_weight_sum(&self.term_name)),
            )
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let data_weight_sums = if data_weight_sums.len() == parameter_sets.len() {
            data_weight_sums
        } else {
            vec![full_integrals.data_weight_sum(); parameter_sets.len()]
        };
        let evaluate = |integrals: &CrossSectionIntegrals| -> LikelihoodResult<Estimate> {
            record_prepared_intensity_evaluation();
            if let Some(ensemble) = &self.ensemble {
                if ensemble.replicas_share_event_rows() {
                    self.shared_bootstrap_requests
                        .fetch_add(ensemble.replicas().len() as u64, Ordering::Relaxed);
                }
            }
            let generated =
                integrals.generated_integrals_many(&parameter_sets, &parameter_contexts)?;
            let values = generated
                .iter()
                .zip(&full_accepted)
                .zip(&data_weight_sums)
                .map(|((generated, accepted), data)| {
                    if *accepted <= 0.0 {
                        return Err(LikelihoodError::NonPositiveAcceptedIntegral(*accepted));
                    }
                    Ok(data * generated / accepted / self.luminosity)
                })
                .collect::<LikelihoodResult<Vec<_>>>()?;
            Ok(Estimate::from_evaluation(
                values[0],
                values[1..].to_vec(),
                self.ensemble.as_ref().map(Ensemble::source_id),
            ))
        };
        let full = evaluate(&full_integrals)?;
        self.assemble_total_set(full, components, |canonical| {
            let integrals = self.integrals_for(&self.likelihood, Some(canonical.as_slice()))?;
            evaluate(&integrals)
        })
    }

    fn fallback_total_set(
        &self,
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<TotalSet> {
        let full = self.observed_total_selected(None)?;
        self.assemble_total_set(full.clone(), components, |canonical| {
            let value = self.observed_total_selected(Some(canonical.as_slice()))?;
            Estimate::with_source_id(value.value(), value.draws().to_vec(), full.source_id())
        })
    }

    fn assemble_total_set(
        &self,
        full: Estimate,
        components: &HashMap<String, Vec<String>>,
        mut evaluate: impl FnMut(&CanonicalTags) -> LikelihoodResult<Estimate>,
    ) -> LikelihoodResult<TotalSet> {
        let mut canonical_values = HashMap::<CanonicalTags, Estimate>::new();
        let mut values = HashMap::with_capacity(components.len());
        for (name, tags) in components {
            let canonical = CanonicalTags::new(tags);
            let value = match canonical_values.get(&canonical) {
                Some(value) => value.clone(),
                None => {
                    let value = evaluate(&canonical)?;
                    canonical_values.insert(canonical, value.clone());
                    value
                }
            };
            values.insert(name.clone(), value);
        }
        Ok(TotalSet {
            full,
            components: values,
        })
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
        let projection = Projection::new("differential", axes.to_vec()).map_err(|error| {
            if axes.is_empty() {
                invalid("at least one differential axis is required")
            } else {
                error
            }
        })?;
        let mut entries = self
            .projection_set(std::slice::from_ref(&projection), components)?
            .entries;
        Ok(entries.remove(0).1)
    }

    /// Computes an ordered set of independent named differential cross sections.
    ///
    /// # Errors
    /// Returns an error for an empty request, duplicate names, or failed
    /// expression/model evaluation.
    pub fn projection_set(
        &self,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<ProjectionSet> {
        if projections.is_empty() {
            return Err(invalid("at least one projection is required"));
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
        if self.members.is_some() {
            return self.combined_projection_set(projections, components);
        }
        self.single_projection_set(projections, components)
    }

    fn integrals_for(
        &self,
        likelihood: &Likelihood,
        tags: Option<&[String]>,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        if tags.is_some() {
            self.tagged_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.full_requests.fetch_add(1, Ordering::Relaxed);
        }
        if std::ptr::eq(likelihood, self.likelihood.as_ref()) {
            self.central_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.arbitrary_replica_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        let key_tags = tags.map(CanonicalTags::new);
        let key = (likelihood as *const Likelihood as usize, key_tags.clone());
        if let Some(integrals) = self
            .integral_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&key)
        {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(integrals);
        }
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        if key_tags.is_none() && std::ptr::eq(likelihood, self.likelihood.as_ref()) {
            let integrals = self.full_integrals.clone();
            self.integral_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(key, integrals.clone());
            return Ok(integrals);
        }
        {
            let mut cache = self
                .integral_cache
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if matches!(cache.policy, IntegralRetentionPolicy::Bounded { .. }) {
                // Keep at least one baseline-sized preparation available. A larger
                // request still receives the normal pool budget error.
                cache.evict_for_transient(
                    self.full_integrals.resident_bytes() as u64,
                    likelihood.execution().host_memory(),
                );
            }
        }
        let integrals = loop {
            let prepared = match key_tags.as_ref() {
                Some(tags) => likelihood.cross_section_integrals_with_tags(
                    &self.term_name,
                    &self.generated_mc,
                    tags.as_slice().iter().map(String::as_str),
                ),
                None => likelihood.cross_section_integrals(&self.term_name, &self.generated_mc),
            };
            match prepared {
                Ok(integrals) => break integrals,
                Err(
                    error @ LikelihoodError::Runtime(laddu_runtime::RuntimeError::Memory(
                        laddu_runtime::MemoryError::BudgetExceeded { .. },
                    )),
                ) => {
                    let mut cache = self
                        .integral_cache
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    if !matches!(cache.policy, IntegralRetentionPolicy::Bounded { .. })
                        || !cache.evict_lru()
                    {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        };
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
                            .map(|likelihood| {
                                if ensemble.replicas_share_event_rows {
                                    self.shared_bootstrap_requests
                                        .fetch_add(1, Ordering::Relaxed);
                                    likelihood
                                        .intensity_data_weight_sum(&self.term_name)
                                        .map(|sum| integrals.with_data_weight_sum(sum))
                                } else {
                                    self.integrals_for(likelihood, tags)
                                }
                            })
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
        let central = self.combined_central_measurement(tags)?;
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

    fn combined_central_measurement(
        &self,
        tags: Option<&[String]>,
    ) -> LikelihoodResult<(f64, f64)> {
        let members = self
            .members
            .as_ref()
            .ok_or_else(|| invalid("CrossSection is not combined"))?;
        members
            .iter()
            .try_fold((0.0, 0.0), |(yield_sum, exposure_sum), (member, factor)| {
                let (yield_value, exposure) = member.selected_measurement_for(
                    &member.likelihood,
                    &member.parameters,
                    tags,
                    factor.central,
                )?;
                Ok((yield_sum + yield_value, exposure_sum + exposure))
            })
    }

    fn combined_central_value(&self, tags: Option<&[String]>) -> LikelihoodResult<f64> {
        let (yield_sum, exposure_sum) = self.combined_central_measurement(tags)?;
        Ok(yield_sum / exposure_sum)
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
        let (unique_projections, projection_plans) = deduplicate_projections(projections);
        let plans = unique_projections
            .iter()
            .map(|projection| {
                let prepare = || {
                    Ok(PreparedProjection {
                        name: projection.name().to_owned(),
                        request_axes: projection.axes().to_vec(),
                        axes: projection
                            .axes()
                            .iter()
                            .map(|axis| axis.binning.edges().to_vec())
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
                prepare().map_err(|error: LikelihoodError| {
                    invalid(format!(
                        "projection `{}` preparation failed: {error}",
                        projection.name()
                    ))
                })
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
                    let data_histogram =
                        draw_data_bins.accumulate_products(draw_data_weights, None);
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

    fn combined_projection_set(
        &self,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<ProjectionSet> {
        let members = self
            .members
            .as_ref()
            .ok_or_else(|| invalid("CrossSection is not combined"))?;
        let (unique_projections, projection_indexes) = deduplicate_projections(projections);
        let projection_names = projections
            .iter()
            .map(Projection::name)
            .collect::<Vec<_>>()
            .join(", ");
        let workspaces = members
            .iter()
            .map(|(member, _)| {
                CombinedMemberWorkspace::prepare(member, &unique_projections, components)
                    .map_err(|error| {
                        invalid(format!(
                            "projection set member `{}` projections [{projection_names}] preparation failed: {error}",
                            member.term_name
                        ))
                    })
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let draw_count = member_draw_count(members);
        let reference_source = member_reference_source(members);
        let member_values = members
            .iter()
            .zip(&workspaces)
            .enumerate()
            .map(|(position, ((member, factor), workspace))| {
                workspace
                    .evaluate_projections_with_draws(
                        member,
                        factor,
                        draw_count,
                        reference_source,
                        position,
                    )
                    .map_err(|error| {
                        invalid(format!(
                            "projection set member `{}` projections [{projection_names}] evaluation failed: {error}",
                            member.term_name
                        ))
                    })
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let unique_results = unique_projections
            .iter()
            .enumerate()
            .map(|(projection_index, projection)| {
                let volumes = bin_volumes(projection.axes());
                let central = member_values
                    .iter()
                    .map(|values| &values[0][projection_index])
                    .collect::<Vec<_>>();
                let data = pool_binned(central.iter().map(|values| &values.data), &volumes);
                let model = pool_binned(central.iter().map(|values| &values.model), &volumes);
                let component_central = components
                    .keys()
                    .map(|name| {
                        (
                            name.clone(),
                            pool_binned(
                                central.iter().map(|values| &values.components[name]),
                                &volumes,
                            ),
                        )
                    })
                    .collect::<HashMap<_, _>>();
                let mut data_draws = Vec::with_capacity(draw_count);
                let mut model_draws = Vec::with_capacity(draw_count);
                let mut component_draws = components
                    .keys()
                    .map(|name| (name.clone(), Vec::with_capacity(draw_count)))
                    .collect::<HashMap<_, _>>();
                for draw_index in 0..draw_count {
                    let draw = member_values
                        .iter()
                        .map(|values| &values[draw_index + 1][projection_index])
                        .collect::<Vec<_>>();
                    data_draws.push(pool_binned(
                        draw.iter().map(|values| &values.data),
                        &volumes,
                    ));
                    model_draws.push(pool_binned(
                        draw.iter().map(|values| &values.model),
                        &volumes,
                    ));
                    for name in components.keys() {
                        component_draws.get_mut(name).unwrap().push(pool_binned(
                            draw.iter().map(|values| &values.components[name]),
                            &volumes,
                        ));
                    }
                }
                DifferentialCrossSection {
                    axes: projection
                        .axes()
                        .iter()
                        .map(|axis| axis.binning.edges().to_vec())
                        .collect(),
                    shape: projection.axes().iter().map(Axis::bins).collect(),
                    data: BinnedEstimate::new(data, data_draws),
                    model: BinnedEstimate::new(model, model_draws),
                    components: component_central
                        .into_iter()
                        .map(|(name, central)| {
                            let draws = component_draws.remove(&name).unwrap_or_default();
                            (name, BinnedEstimate::new(central, draws))
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        Ok(ProjectionSet {
            entries: projections
                .iter()
                .zip(projection_indexes)
                .map(|(projection, index)| (projection.name.clone(), unique_results[index].clone()))
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

fn evaluate_bin_assignments_many(
    dataset: &Dataset,
    projections: &[&Projection],
    execution: &Execution,
) -> LikelihoodResult<Vec<BinAssignments>> {
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum CoordinateStatus {
        InRange,
        OutOfRange,
        Nonfinite,
    }
    record_bin_assignment_evaluation();
    let events = usize::try_from(dataset.stats()?.events())
        .map_err(|_| invalid("projection event count exceeds addressable memory"))?;
    let mut expression_indexes = HashMap::new();
    let mut expressions = Vec::new();
    let axes_by_projection = projections
        .iter()
        .map(|projection| {
            projection
                .axes()
                .iter()
                .map(|axis| {
                    let graph = axis.expression.to_graph();
                    let key = (
                        graph.root().index(),
                        graph
                            .nodes()
                            .iter()
                            .map(|node| node.structural_key())
                            .collect::<Vec<_>>(),
                    );
                    *expression_indexes.entry(key).or_insert_with(|| {
                        let index = expressions.len();
                        expressions.push(axis.expression.clone());
                        index
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut indices = projections
        .iter()
        .map(|_| vec![Some(0usize); events])
        .collect::<Vec<_>>();
    let mut statuses = projections
        .iter()
        .map(|_| vec![CoordinateStatus::InRange; events])
        .collect::<Vec<_>>();
    let mut visited = 0;
    dataset.visit_real_chunks(&expressions, execution, 8192, |offset, coordinates| {
        let chunk_len = coordinates.first().map_or(0, Vec::len);
        if coordinates.len() != expressions.len()
            || coordinates.iter().any(|values| {
                values.len() != chunk_len
                    || offset
                        .checked_add(values.len())
                        .is_none_or(|end| end > events)
            })
        {
            return Err(laddu_runtime::RuntimeError::InvalidShape {
                index: 0,
                message: "projection coordinates do not match the dataset shape".into(),
            });
        }
        visited = offset + chunk_len;
        for (projection_index, projection) in projections.iter().enumerate() {
            let assignments = &mut indices[projection_index];
            let status = &mut statuses[projection_index];
            for (axis, &coordinate_index) in projection
                .axes()
                .iter()
                .zip(&axes_by_projection[projection_index])
            {
                for (row, &value) in coordinates[coordinate_index].iter().enumerate() {
                    let event = offset + row;
                    if !value.is_finite() {
                        status[event] = CoordinateStatus::Nonfinite;
                        assignments[event] = None;
                    } else if status[event] == CoordinateStatus::InRange {
                        if let Some(bin) = axis.binning.index(value, FinalUpperEdge::Exclusive) {
                            assignments[event] = assignments[event]
                                .and_then(|index| index.checked_mul(axis.bins())?.checked_add(bin));
                        } else {
                            status[event] = CoordinateStatus::OutOfRange;
                            assignments[event] = None;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    if visited != events {
        return Err(invalid(
            "projection coordinate count does not match dataset events",
        ));
    }
    projections
        .iter()
        .zip(indices)
        .zip(statuses)
        .map(|((projection, indices), status)| {
            let count = checked_bin_count(
                &projection
                    .axes()
                    .iter()
                    .map(|axis| axis.binning.clone())
                    .collect::<Vec<_>>(),
            )
            .ok_or_else(|| invalid("projection axis shape exceeds addressable bin count"))?;
            Ok(BinAssignments {
                indices,
                count,
                nonfinite_count: status
                    .iter()
                    .filter(|&&value| value == CoordinateStatus::Nonfinite)
                    .count(),
                out_of_range_count: status
                    .iter()
                    .filter(|&&value| value == CoordinateStatus::OutOfRange)
                    .count(),
            })
        })
        .collect()
}

fn evaluate_bin_assignments(
    dataset: &Dataset,
    axes: &[Axis],
    execution: &Execution,
) -> LikelihoodResult<BinAssignments> {
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum CoordinateStatus {
        InRange,
        OutOfRange,
        Nonfinite,
    }
    record_bin_assignment_evaluation();
    let events = usize::try_from(dataset.stats()?.events())
        .map_err(|_| invalid("projection event count exceeds addressable memory"))?;
    let mut indices = vec![Some(0usize); events];
    let mut status = vec![CoordinateStatus::InRange; events];
    let expressions = axes
        .iter()
        .map(|axis| axis.expression.clone())
        .collect::<Vec<_>>();
    let mut visited = 0;
    dataset.visit_real_chunks(&expressions, execution, 8192, |offset, coordinates| {
        if coordinates.len() != axes.len()
            || coordinates.iter().any(|values| {
                values.len() != coordinates[0].len()
                    || offset
                        .checked_add(values.len())
                        .is_none_or(|end| end > events)
            })
        {
            return Err(laddu_runtime::RuntimeError::InvalidShape {
                index: 0,
                message: "projection coordinates do not match the dataset shape".into(),
            });
        }
        visited = offset + coordinates[0].len();
        for (axis, values) in axes.iter().zip(coordinates) {
            for (row, &value) in values.iter().enumerate() {
                let event = offset + row;
                if !value.is_finite() {
                    status[event] = CoordinateStatus::Nonfinite;
                    indices[event] = None;
                } else if status[event] == CoordinateStatus::InRange {
                    if let Some(bin) = axis.binning.index(value, FinalUpperEdge::Exclusive) {
                        indices[event] = indices[event]
                            .and_then(|index| index.checked_mul(axis.bins())?.checked_add(bin));
                    } else {
                        status[event] = CoordinateStatus::OutOfRange;
                        indices[event] = None;
                    }
                }
            }
        }
        Ok(())
    })?;
    if visited != events {
        return Err(invalid(
            "projection coordinate count does not match dataset events",
        ));
    }
    let count = checked_bin_count(
        &axes
            .iter()
            .map(|axis| axis.binning.clone())
            .collect::<Vec<_>>(),
    )
    .ok_or_else(|| invalid("projection axis shape exceeds addressable bin count"))?;
    Ok(BinAssignments {
        indices,
        count,
        nonfinite_count: status
            .iter()
            .filter(|&&value| value == CoordinateStatus::Nonfinite)
            .count(),
        out_of_range_count: status
            .iter()
            .filter(|&&value| value == CoordinateStatus::OutOfRange)
            .count(),
    })
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
                axis.binning
                    .edges()
                    .windows(2)
                    .map(move |pair| volume * (pair[1] - pair[0]))
            })
            .collect()
    })
}

fn valid_bin_volumes(axes: &[Axis]) -> bool {
    !axes.is_empty()
        && axes
            .iter()
            .try_fold((1.0_f64, 1.0_f64), |(min_volume, max_volume), axis| {
                let (min_width, max_width) = axis.binning.edges().windows(2).fold(
                    (f64::INFINITY, 0.0_f64),
                    |(min_width, max_width), pair| {
                        let width = pair[1] - pair[0];
                        (min_width.min(width), max_width.max(width))
                    },
                );
                let min_volume = min_volume * min_width;
                let max_volume = max_volume * max_width;
                (min_volume.is_finite()
                    && min_volume > 0.0
                    && max_volume.is_finite()
                    && max_volume > 0.0)
                    .then_some((min_volume, max_volume))
            })
            .is_some()
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

    fn bootstrap_total_fixture(samples: usize) -> (Arc<Likelihood>, Dataset, Ensemble) {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0)).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (1.25, 2.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0), (1.75, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let ensemble = Ensemble::bootstrap_fit(&likelihood, samples, 42, |replica, _| {
            Ok::<_, std::convert::Infallible>(replica.default_params())
        })
        .unwrap();
        (likelihood, generated, ensemble)
    }

    fn bootstrap_tagged_scalar_fixture(samples: usize) -> (Arc<Likelihood>, Dataset, Ensemble) {
        let signal =
            (Expr::from(parameter!("scale", initial: 0.5)) * event_scalar("x")).tagged("signal");
        let background = Expr::from(1.0).tagged("background");
        let model = CompiledModel::from_expr(&(signal + background).norm_sqr()).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (1.25, 2.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0), (1.75, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let ensemble = Ensemble::bootstrap_fit(&likelihood, samples, 42, |_replica, index| {
            Ok::<_, std::convert::Infallible>(vec![0.4 + index as f64 * 0.1])
        })
        .unwrap();
        (likelihood, generated, ensemble)
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
        let axis = BinningAxis::new([0.0, 1.0, 2.0]).unwrap();
        assert_eq!(axis.index(0.0, FinalUpperEdge::Exclusive), Some(0));
        assert_eq!(axis.index(1.0, FinalUpperEdge::Exclusive), Some(1));
        assert_eq!(axis.index(2.0, FinalUpperEdge::Exclusive), None);
    }

    #[test]
    fn projection_assignments_match_shared_one_and_multi_axis_contracts() {
        let x = Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap();
        let y = Axis::new(event_scalar("y"), vec![-1.0, 0.0, 2.0]).unwrap();
        let values = vec![vec![0.0, 1.5, 2.0, f64::NAN], vec![-0.5, 1.0, 0.0, 0.0]];

        let one = BinAssignments::new(&values[..1], std::slice::from_ref(&x));
        let expected_one: Vec<_> = values[0]
            .iter()
            .map(|value| x.binning.index(*value, FinalUpperEdge::Exclusive))
            .collect();
        assert_eq!(one.indices, expected_one);

        let axes = [x, y];
        let shared_axes: Vec<_> = axes.iter().map(|axis| axis.binning.clone()).collect();
        let multi = BinAssignments::new(&values, &axes);
        let expected_multi: Vec<_> = (0..values[0].len())
            .map(|event| {
                laddu_runtime::flat_bin_index(
                    &shared_axes,
                    &[values[0][event], values[1][event]],
                    FinalUpperEdge::Exclusive,
                )
            })
            .collect();
        assert_eq!(multi.indices, expected_multi);
    }

    #[test]
    fn projection_shape_overflow_is_rejected_during_specification() {
        let axes = (0..usize::BITS)
            .map(|_| Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap())
            .collect();
        let error = Projection::new("overflow", axes).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid cross-section analysis: projection axis shape exceeds addressable bin count"
        );
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

    #[cfg(feature = "jit")]
    #[test]
    fn projections_match_cpu_interpreter_and_jit_backends() {
        use laddu_runtime::{
            CpuOptions, Device, ExecutionOptions, JitPolicy, Precision, ThreadPolicy,
        };

        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0)).unwrap();
        let data = weighted_dataset_2d(&[(0.25, 0.25, 1.0), (1.25, 1.25, -2.0)]);
        let accepted = weighted_dataset_2d(&[(0.25, 0.25, 1.0), (1.25, 1.25, 1.0)]);
        let generated = weighted_dataset_2d(&[(0.25, 0.25, 1.0), (1.25, 1.25, 1.0)]);
        let axes = [
            Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap(),
            Axis::new(event_scalar("y"), vec![0.0, 1.0, 2.0]).unwrap(),
        ];
        let execution = |jit| {
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Serial,
                    jit,
                }),
                precision: Precision::F64,
                ..ExecutionOptions::default()
            })
            .unwrap()
        };
        let evaluate = |jit| {
            let likelihood = Arc::new(
                Likelihood::with_execution(
                    [crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()],
                    &execution(jit),
                )
                .unwrap(),
            );
            let yield_projection = Yield::with_ensemble(
                likelihood.clone(),
                "signal",
                generated.clone(),
                Vec::new(),
                None,
            )
            .unwrap()
            .projection(&axes)
            .unwrap();
            let differential = likelihood
                .cross_section("signal", generated.clone(), 1.0, Vec::new())
                .unwrap()
                .differential(&axes, &HashMap::new())
                .unwrap();
            (differential, yield_projection)
        };

        let (interpreted, interpreted_yield) = evaluate(JitPolicy::Disabled);
        let (compiled, compiled_yield) = evaluate(JitPolicy::Enabled);
        for (actual, expected) in compiled
            .data()
            .values()
            .iter()
            .zip(interpreted.data().values())
        {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
            }
        }
        assert_eq!(compiled_yield.shape(), interpreted_yield.shape());
        assert_eq!(compiled_yield.validity(), interpreted_yield.validity());
        for (actual, expected) in compiled_yield
            .corrected()
            .iter()
            .zip(interpreted_yield.corrected())
        {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
            }
        }
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
    fn native_bootstrap_totals_share_mc_preparation_and_preserve_paired_weights() {
        let (likelihood, generated, ensemble) = bootstrap_total_fixture(3);
        let expected_draws = ensemble
            .draws()
            .iter()
            .zip(ensemble.replicas())
            .map(|(parameters, replica)| {
                replica
                    .cross_section("signal", generated.clone(), 10.0, parameters.clone())
                    .unwrap()
                    .observed_total()
                    .unwrap()
                    .value()
            })
            .collect::<Vec<_>>();
        let cross_section = likelihood
            .cross_section_with_ensemble(
                "signal",
                generated,
                10.0,
                likelihood.default_params(),
                ensemble.clone(),
            )
            .unwrap();

        let total = cross_section.observed_total().unwrap();

        assert_eq!(total.draws(), expected_draws);
        assert_eq!(total.source_id(), Some(ensemble.source_id()));
        assert_eq!(cross_section.diagnostics().cached_integrals(), 1);
        assert_eq!(cross_section.diagnostics().shared_bootstrap_requests(), 3);
        assert_eq!(cross_section.diagnostics().arbitrary_replica_requests(), 0);
    }

    #[test]
    fn native_bootstrap_shares_preparation_across_every_scalar_operation() {
        let (likelihood, generated, ensemble) = bootstrap_tagged_scalar_fixture(3);
        let tags = vec!["signal".to_owned()];
        let explicit = ensemble
            .draws()
            .iter()
            .zip(ensemble.replicas())
            .map(|(parameters, replica)| {
                let cross_section = replica
                    .cross_section("signal", generated.clone(), 10.0, parameters.clone())
                    .unwrap();
                [
                    cross_section.fitted_total().unwrap().value(),
                    cross_section.acceptance().unwrap().value(),
                    cross_section.corrected_yield().unwrap().value(),
                    cross_section.fitted_total_with_tags(&tags).unwrap().value(),
                    cross_section.acceptance_with_tags(&tags).unwrap().value(),
                    cross_section
                        .corrected_yield_with_tags(&tags)
                        .unwrap()
                        .value(),
                ]
            })
            .collect::<Vec<_>>();
        let cross_section = likelihood
            .cross_section_with_ensemble(
                "signal",
                generated,
                10.0,
                likelihood.default_params(),
                ensemble,
            )
            .unwrap();
        let actual = [
            cross_section.fitted_total().unwrap(),
            cross_section.acceptance().unwrap(),
            cross_section.corrected_yield().unwrap(),
            cross_section.fitted_total_with_tags(&tags).unwrap(),
            cross_section.acceptance_with_tags(&tags).unwrap(),
            cross_section.corrected_yield_with_tags(&tags).unwrap(),
        ];

        for (quantity, quantity_index) in actual.iter().zip(0..) {
            let expected = explicit
                .iter()
                .map(|values| values[quantity_index])
                .collect::<Vec<_>>();
            assert_eq!(quantity.draws(), expected);
        }
        assert_eq!(cross_section.diagnostics().cached_integrals(), 2);
    }

    #[test]
    fn total_set_matches_separate_totals_and_canonicalizes_component_aliases() {
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
                2.0,
                fixture.likelihood.default_params(),
                ensemble,
            )
            .unwrap();
        let components = HashMap::from([
            ("ordered".into(), vec!["background".into(), "signal".into()]),
            (
                "reordered".into(),
                vec!["signal".into(), "background".into()],
            ),
            (
                "repeated".into(),
                vec!["signal".into(), "background".into(), "signal".into()],
            ),
        ]);
        let expected_full = cross_section.observed_total().unwrap();
        let expected_component = cross_section
            .observed_total_with_tags(&components["ordered"])
            .unwrap();

        let before = cross_section.diagnostics();
        reset_projection_evaluation_counts();
        let totals = cross_section.total_set(&components).unwrap();

        assert_relative_eq!(totals.full().value(), expected_full.value());
        for (actual, expected) in totals.full().draws().iter().zip(expected_full.draws()) {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        }
        for name in components.keys() {
            let actual = totals.get(name).unwrap();
            assert_relative_eq!(actual.value(), expected_component.value());
            for (actual, expected) in actual.draws().iter().zip(expected_component.draws()) {
                assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
            }
        }
        assert_eq!(cross_section.diagnostics().cached_integrals(), 2);
        assert_eq!(
            cross_section.diagnostics().full_requests(),
            before.full_requests() + 1
        );
        assert_eq!(
            cross_section.diagnostics().tagged_requests(),
            before.tagged_requests() + 1
        );
        assert_eq!(projection_evaluation_counts(), (3, 0));
    }

    #[test]
    fn combined_total_set_preserves_exposure_pooling_and_draw_pairing() {
        let fixture = canonical_selection_fixture();
        let ensemble_a = Ensemble::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.6, 0.7], vec![1.4, 0.8]],
        )
        .unwrap();
        let ensemble_b = Ensemble::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.7, 0.6], vec![1.3, 0.9]],
        )
        .unwrap();
        let factor_source = ensemble_a.source_id();
        let members = [(10.0, ensemble_a), (15.0, ensemble_b)]
            .into_iter()
            .map(|(luminosity, ensemble)| {
                fixture
                    .likelihood
                    .cross_section_with_ensemble(
                        "signal",
                        fixture.generated.clone(),
                        luminosity,
                        fixture.likelihood.default_params(),
                        ensemble,
                    )
                    .unwrap()
            })
            .collect();
        let combined = CrossSection::combine_with_factors(
            members,
            vec![
                Estimate::with_source_id(1.0, vec![1.1, 0.9], Some(factor_source)).unwrap(),
                Estimate::central(2.0).unwrap(),
            ],
        )
        .unwrap();
        let components = HashMap::from([("signal".into(), vec!["signal".into()])]);
        let expected_full = combined.observed_total().unwrap();
        let expected_signal = combined
            .observed_total_with_tags(&components["signal"])
            .unwrap();

        let totals = combined.total_set(&components).unwrap();

        assert_relative_eq!(totals.full().value(), expected_full.value());
        assert_eq!(totals.full().draws(), expected_full.draws());
        assert_relative_eq!(
            totals.get("signal").unwrap().value(),
            expected_signal.value()
        );
        assert_eq!(
            totals.get("signal").unwrap().draws(),
            expected_signal.draws()
        );
        assert_eq!(
            totals.get("signal").unwrap().source_id(),
            totals.full().source_id()
        );
    }

    #[test]
    fn absolute_rate_total_set_matches_separate_normalized_totals() {
        let (likelihood, generated, ensemble) = bootstrap_tagged_scalar_fixture(3);
        let cross_section = likelihood
            .cross_section_with_ensemble(
                "signal",
                generated,
                10.0,
                likelihood.default_params(),
                ensemble,
            )
            .unwrap();
        let components = HashMap::from([("signal".into(), vec!["signal".into()])]);
        let expected_full = cross_section.observed_total().unwrap();
        let expected_signal = cross_section
            .observed_total_with_tags(&components["signal"])
            .unwrap();

        let totals = cross_section.total_set(&components).unwrap();

        assert_relative_eq!(totals.full().value(), expected_full.value());
        assert_eq!(totals.full().draws(), expected_full.draws());
        assert_relative_eq!(
            totals.get("signal").unwrap().value(),
            expected_signal.value()
        );
        assert_eq!(
            totals.get("signal").unwrap().draws(),
            expected_signal.draws()
        );
    }

    #[test]
    fn unknown_replica_provenance_keeps_per_replica_preparation() {
        let (likelihood, generated, native) = bootstrap_total_fixture(2);
        let unknown = Ensemble::with_replicas(
            native.parameter_names().to_vec(),
            native.draws().to_vec(),
            native.replicas().to_vec(),
        )
        .unwrap();
        let cross_section = likelihood
            .cross_section_with_ensemble(
                "signal",
                generated,
                10.0,
                likelihood.default_params(),
                unknown,
            )
            .unwrap();

        cross_section.observed_total().unwrap();

        assert_eq!(cross_section.diagnostics().cached_integrals(), 3);
    }

    #[test]
    fn bounded_arbitrary_replica_totals_match_explicit_evaluation() {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0)).unwrap();
        let generated = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let make_likelihood = |data: Dataset, accepted: Dataset| {
            Arc::new(
                Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                    .unwrap(),
            )
        };
        let likelihood = make_likelihood(
            weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]),
            weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]),
        );
        let replicas = vec![
            make_likelihood(
                weighted_dataset(&[(0.25, 2.0)]),
                weighted_dataset(&[(0.25, 1.0), (0.75, 1.0), (1.25, 1.0)]),
            ),
            make_likelihood(
                weighted_dataset(&[(1.25, 3.0)]),
                weighted_dataset(&[(0.25, 1.0), (1.25, 1.0), (1.75, 1.0)]),
            ),
        ];
        let expected = replicas
            .iter()
            .map(|replica| {
                replica
                    .cross_section("signal", generated.clone(), 1.0, Vec::new())
                    .unwrap()
                    .observed_total()
                    .unwrap()
                    .value()
            })
            .collect::<Vec<_>>();
        let ensemble =
            Ensemble::with_replicas(Vec::new(), vec![Vec::new(), Vec::new()], replicas).unwrap();
        let cross_section = likelihood
            .cross_section_with_ensemble("signal", generated, 1.0, Vec::new(), ensemble)
            .unwrap();
        let max_bytes = cross_section.diagnostics().prepared_bytes();
        cross_section.set_integral_retention(IntegralRetentionPolicy::Bounded { max_bytes });

        assert_eq!(cross_section.observed_total().unwrap().draws(), expected);
        assert!(cross_section.diagnostics().prepared_bytes() <= max_bytes);
        assert!(cross_section.diagnostics().cached_integrals() <= 1);
        assert!(cross_section.diagnostics().arbitrary_replica_requests() >= 2);
    }

    #[test]
    fn integral_retention_can_be_disabled_cleared_and_bounded() {
        let fixture = canonical_selection_fixture();
        let cross_section = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 2.0, vec![1.5, 0.75])
            .unwrap();
        let tags = vec!["signal".to_owned()];

        cross_section.set_integral_retention(IntegralRetentionPolicy::None);
        cross_section.clear_integral_cache();
        assert_eq!(cross_section.diagnostics().cached_integrals(), 0);
        let expected = cross_section.observed_total_with_tags(&tags).unwrap();
        assert_eq!(cross_section.diagnostics().cached_integrals(), 0);

        let baseline_bytes = cross_section.full_integrals.resident_bytes();
        cross_section.set_integral_retention(IntegralRetentionPolicy::Bounded {
            max_bytes: baseline_bytes,
        });
        assert_relative_eq!(
            cross_section
                .observed_total_with_tags(&tags)
                .unwrap()
                .value(),
            expected.value()
        );
        let diagnostics = cross_section.diagnostics();
        assert!(diagnostics.prepared_bytes() <= baseline_bytes);

        cross_section.clear_integral_cache();
        assert_eq!(cross_section.diagnostics().cached_integrals(), 0);
        assert!(cross_section.observed_total().unwrap().value().is_finite());
    }

    #[test]
    fn clearing_retained_integrals_releases_pool_reservations() {
        let fixture = canonical_selection_fixture();
        let pool = fixture.likelihood.execution().host_memory().clone();
        let cross_section = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 2.0, vec![1.5, 0.75])
            .unwrap();
        let baseline = pool.report().reserved_bytes;
        cross_section
            .observed_total_with_tags(&["signal".to_owned()])
            .unwrap();
        let retained = pool.report().reserved_bytes;
        assert!(retained > baseline);

        cross_section.clear_integral_cache();

        assert!(pool.report().reserved_bytes < retained);
        assert_eq!(cross_section.diagnostics().cached_integrals(), 0);
    }

    #[test]
    fn combined_cache_policy_applies_to_each_member() {
        let fixture = canonical_selection_fixture();
        let first = fixture
            .likelihood
            .cross_section("signal", fixture.generated.clone(), 2.0, vec![1.5, 0.75])
            .unwrap();
        let second = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 3.0, vec![1.5, 0.75])
            .unwrap();
        let combined = CrossSection::combine(vec![first.clone(), second.clone()]).unwrap();

        combined.set_integral_retention(IntegralRetentionPolicy::None);
        combined.clear_integral_cache();
        assert!(
            combined
                .observed_total_with_tags(&["signal".to_owned()])
                .unwrap()
                .value()
                .is_finite()
        );

        assert_eq!(first.diagnostics().cached_integrals(), 0);
        assert_eq!(second.diagnostics().cached_integrals(), 0);
    }

    #[test]
    fn combined_integral_limit_and_diagnostics_cover_all_members() {
        let fixture = canonical_selection_fixture();
        let first = fixture
            .likelihood
            .cross_section("signal", fixture.generated.clone(), 2.0, vec![1.5, 0.75])
            .unwrap();
        let second = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 3.0, vec![1.5, 0.75])
            .unwrap();
        let combined = CrossSection::combine(vec![first.clone(), second.clone()]).unwrap();
        let tag = ["signal".to_owned()];
        first.observed_total_with_tags(&tag).unwrap();
        second.observed_total_with_tags(&tag).unwrap();
        assert_eq!(
            combined.diagnostics().prepared_bytes(),
            first.diagnostics().prepared_bytes() + second.diagnostics().prepared_bytes()
        );

        let max_bytes = first.diagnostics().prepared_bytes();
        combined.set_integral_retention(IntegralRetentionPolicy::Bounded { max_bytes });
        combined.observed_total_with_tags(&tag).unwrap();

        assert!(combined.diagnostics().prepared_bytes() <= max_bytes);
    }

    #[test]
    fn diagnostics_deduplicate_shared_cache_and_report_evaluation_paths() {
        let fixture = canonical_selection_fixture();
        let section = fixture
            .likelihood
            .cross_section("signal", fixture.generated, 2.0, vec![1.5, 0.75])
            .unwrap();
        let combined = CrossSection::combine(vec![section.clone(), section.clone()]).unwrap();
        let initial = section.diagnostics();
        assert_eq!(combined.diagnostics(), initial);
        assert_eq!(initial.full_requests(), 1);
        assert_eq!(initial.central_requests(), 1);
        assert!(initial.estimated_prepared_bytes() >= initial.prepared_bytes());

        section
            .observed_total_with_tags(&["signal".to_owned()])
            .unwrap();
        let after = section.diagnostics();
        assert_eq!(combined.diagnostics(), after);
        assert_eq!(after.tagged_requests(), 1);
        assert_eq!(after.central_requests(), 2);
        assert!(after.reserved_bytes() >= initial.reserved_bytes());
        assert!(after.high_water_bytes() >= after.reserved_bytes());

        section.set_integral_retention(IntegralRetentionPolicy::Bounded { max_bytes: 0 });
        assert_eq!(section.diagnostics().cached_integrals(), 0);
        assert!(section.diagnostics().cache_evictions() >= 1);
        assert!(section.diagnostics().estimated_prepared_bytes() > 0);
    }

    #[test]
    fn dropping_cross_section_releases_its_pool_reservations() {
        let fixture = canonical_selection_fixture();
        let pool = fixture.likelihood.execution().host_memory().clone();
        let baseline = pool.report().reserved_bytes;
        {
            let section = fixture
                .likelihood
                .cross_section("signal", fixture.generated, 2.0, vec![1.5, 0.75])
                .unwrap();
            section
                .observed_total_with_tags(&["signal".to_owned()])
                .unwrap();
            assert!(pool.report().reserved_bytes > baseline);
        }
        assert_eq!(pool.report().reserved_bytes, baseline);
    }

    #[test]
    fn central_yield_projection_keeps_joint_bins_and_invalid_support_visible() {
        let fixture = canonical_selection_fixture();
        let yield_context = Yield::with_ensemble(
            fixture.likelihood.clone(),
            "signal",
            fixture.generated,
            vec![1.5, 0.75],
            None,
        )
        .unwrap();
        let joint = yield_context
            .projection(&[fixture.axis.clone(), fixture.axis.clone()])
            .unwrap();
        assert_eq!(joint.shape(), &[2, 2]);
        assert_eq!(joint.selected(), &[3.0, 0.0, 0.0, 1.0]);
        assert_eq!(joint.selected_histogram().values(), joint.selected());
        assert_eq!(
            joint.validity(),
            &[
                YieldBinValidity::Valid,
                YieldBinValidity::MissingGeneratedSupport,
                YieldBinValidity::MissingGeneratedSupport,
                YieldBinValidity::Valid
            ]
        );
        assert!(joint.corrected()[1].is_nan());
        assert_eq!(joint.diagnostics().generated_out_of_range, 0);

        let projections = [
            Projection::new("joint", vec![fixture.axis.clone(), fixture.axis.clone()]).unwrap(),
            Projection::new("single", vec![fixture.axis]).unwrap(),
        ];
        reset_projection_evaluation_counts();
        let results = yield_context.projection_set(&projections).unwrap();
        assert_eq!(projection_evaluation_counts().1, 3);
        assert_eq!(
            results.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            vec!["joint", "single"]
        );
        assert_eq!(results.get("single").unwrap().selected(), &[3.0, 1.0]);
    }

    #[test]
    fn central_yield_projection_preserves_distinct_rate_quantities() {
        let x = event_scalar("x");
        let model =
            CompiledModel::from_expr(&(x.clone() * parameter!("scale", initial: 0.25))).unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 2.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let context =
            Yield::with_ensemble(likelihood, "signal", generated, vec![0.25], None).unwrap();
        let projection = context
            .projection(&[Axis::new(x, vec![0.0, 5.0, 10.0]).unwrap()])
            .unwrap();
        assert_eq!(projection.selected(), &[3.0, 0.0]);
        assert_relative_eq!(projection.accepted()[0], 1.0);
        assert_relative_eq!(projection.generated()[1], 1.5);
        assert_eq!(
            projection.validity(),
            &[
                YieldBinValidity::MissingGeneratedSupport,
                YieldBinValidity::MissingAcceptedSupport,
            ]
        );
        assert!(projection.corrected().iter().all(|value| value.is_nan()));

        let all = context
            .projection(&[Axis::new(event_scalar("x"), vec![0.0, 10.0]).unwrap()])
            .unwrap();
        assert_eq!(all.validity(), &[YieldBinValidity::Valid]);
        assert_relative_eq!(all.selected()[0], 3.0);
        assert_relative_eq!(all.accepted()[0], 1.0);
        assert_relative_eq!(all.generated()[0], 1.5);
        assert_relative_eq!(all.acceptance()[0], 2.0 / 3.0);
        assert_relative_eq!(all.corrected()[0], 4.5);
    }

    #[test]
    fn central_yield_projection_preserves_signed_and_empty_observations() {
        let x = event_scalar("x");
        let model = CompiledModel::from_expr(&Expr::from(1.0)).unwrap();
        let data = weighted_dataset(&[(0.25, -2.0), (0.25, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let generated = accepted.clone();
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let context = Yield::with_ensemble(likelihood, "signal", generated, vec![], None).unwrap();
        let projection = context
            .projection(&[Axis::new(x, vec![0.0, 1.0, 2.0]).unwrap()])
            .unwrap();
        assert_eq!(projection.selected(), &[-1.0, 0.0]);
        assert_eq!(
            projection.validity(),
            &[YieldBinValidity::Valid, YieldBinValidity::Valid]
        );
        assert_eq!(projection.corrected(), &[-1.0, 0.0]);
        assert!(projection.accepted().iter().all(|value| value.is_nan()));
    }

    #[test]
    fn central_yield_projection_reports_coordinate_exclusions_and_validates_volumes() {
        let model = CompiledModel::from_expr(&Expr::from(1.0)).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (f64::NAN, 1.0), (3.0, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0)]);
        let generated = accepted.clone();
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let context = Yield::with_ensemble(likelihood, "signal", generated, vec![], None).unwrap();
        let projection = context
            .projection(&[Axis::new(event_scalar("x"), vec![0.0, 1.0]).unwrap()])
            .unwrap();
        assert_eq!(projection.selected(), &[1.0]);
        assert_eq!(projection.diagnostics().selected_nonfinite, 1);
        assert_eq!(projection.diagnostics().selected_out_of_range, 1);
        assert_eq!(projection.validity(), &[YieldBinValidity::Valid]);
        let invalid = Axis::new(event_scalar("x"), vec![-f64::MAX, f64::MAX]).unwrap();
        assert!(context.projection(&[invalid]).is_err());
    }

    #[test]
    fn central_yield_projection_marks_nonpositive_local_acceptance() {
        let model = CompiledModel::from_expr(&Expr::from(1.0)).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, -1.0), (1.25, 2.0)]);
        let generated = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let context = Yield::with_ensemble(likelihood, "signal", generated, vec![], None).unwrap();
        let projection = context
            .projection(&[Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap()])
            .unwrap();
        assert_eq!(
            projection.validity(),
            &[
                YieldBinValidity::NonPositiveAcceptedSupport,
                YieldBinValidity::Valid,
            ]
        );
        assert!(projection.corrected()[0].is_nan());
    }

    #[test]
    fn central_yield_projection_marks_invalid_local_exposure() {
        let model = CompiledModel::from_expr(&Expr::from(1.0)).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let accepted = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let generated = weighted_dataset(&[(0.25, -1.0), (1.25, 2.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let context = Yield::with_ensemble(likelihood, "signal", generated, vec![], None).unwrap();
        let projection = context
            .projection(&[Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap()])
            .unwrap();
        assert_eq!(
            projection.validity(),
            &[YieldBinValidity::InvalidExposure, YieldBinValidity::Valid]
        );
        assert!(projection.corrected()[0].is_nan());
    }

    #[test]
    fn central_yield_projection_rejects_oversized_workspace_atomically() {
        use laddu_runtime::{ExecutionOptions, MemoryBudget, MemoryPlan};

        let model = CompiledModel::from_expr(&Expr::from(1.0)).unwrap();
        let data = weighted_dataset(&[(0.25, 1.0), (1.25, 1.0)]);
        let accepted = data.clone();
        let generated = data.clone();
        let execution = Execution::local(ExecutionOptions {
            memory: MemoryPlan::host_device(MemoryBudget::Bytes(128 * 1024), MemoryBudget::Auto),
            ..ExecutionOptions::default()
        })
        .unwrap();
        let likelihood = Arc::new(
            Likelihood::with_execution(
                [crate::NllTerm::new("signal", &model, &data, &accepted).unwrap()],
                &execution,
            )
            .unwrap(),
        );
        let context = Yield::with_ensemble(likelihood, "signal", generated, vec![], None).unwrap();
        let baseline = execution.host_memory().report().reserved_bytes;
        let requests = (0..1000)
            .map(|index| {
                Projection::new(
                    format!("projection-{index}"),
                    vec![
                        Axis::new(
                            event_scalar("x"),
                            vec![0.0, 1.0 + index as f64 * 1.0e-6, 2.0],
                        )
                        .unwrap(),
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(context.projection_set(&requests).is_err());
        assert_eq!(execution.host_memory().report().reserved_bytes, baseline);
        assert!(
            context
                .projection(&[Axis::new(event_scalar("x"), vec![0.0, 2.0]).unwrap()])
                .is_ok()
        );
    }

    #[test]
    fn central_only_total_skips_bootstrap_draw_preparation() {
        let (likelihood, generated, ensemble) = bootstrap_total_fixture(3);
        let cross_section = likelihood
            .cross_section_with_ensemble(
                "signal",
                generated,
                10.0,
                likelihood.default_params(),
                ensemble,
            )
            .unwrap();

        let total = cross_section.observed_total_central().unwrap();

        assert!(total.draws().is_empty());
        assert_eq!(total.source_id(), None);
        assert_eq!(cross_section.diagnostics().cached_integrals(), 1);
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

        let combined = CrossSection::combine(vec![cross_section.clone(), cross_section]).unwrap();
        let error = combined
            .projection_set(&projections, &HashMap::new())
            .expect_err("a combined negative draw intensity must fail");
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
    fn combined_projection_sets_match_independent_combined_differentials() {
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
        let wide_axis = Axis::new(event_scalar("x"), vec![0.0, 2.0]).unwrap();
        let projections = vec![
            Projection::new("fine", vec![fixture.axis.clone()]).unwrap(),
            Projection::new("fine_alias", vec![fixture.axis.clone()]).unwrap(),
            Projection::new("wide", vec![wide_axis.clone()]).unwrap(),
        ];

        reset_projection_evaluation_counts();
        let actual = cross_section
            .projection_set(&projections, &fixture.components)
            .unwrap();
        assert_eq!(projection_evaluation_counts(), (8, 12));
        assert_projection_close(
            actual.get("fine").unwrap(),
            actual.get("fine_alias").unwrap(),
        );

        for (name, axes) in [("fine", vec![fixture.axis]), ("wide", vec![wide_axis])] {
            let expected = cross_section
                .differential(&axes, &fixture.components)
                .unwrap();
            assert_projection_close(actual.get(name).unwrap(), &expected);
        }
    }

    #[test]
    fn combined_projection_sets_pair_distinct_ensemble_and_factor_sources() {
        let fixture = canonical_selection_fixture();
        let ensemble_a = Ensemble::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.6, 0.7], vec![1.4, 0.8]],
        )
        .unwrap();
        let ensemble_b = Ensemble::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.7, 0.6], vec![1.3, 0.9]],
        )
        .unwrap();
        let source_a = ensemble_a.source_id();
        let source_b = ensemble_b.source_id();
        let members = [ensemble_a.clone(), ensemble_b.clone()]
            .into_iter()
            .enumerate()
            .map(|(index, ensemble)| {
                fixture
                    .likelihood
                    .cross_section_with_ensemble(
                        "signal",
                        fixture.generated.clone(),
                        10.0 + index as f64 * 5.0,
                        fixture.likelihood.default_params(),
                        ensemble,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let factors = vec![
            Estimate::with_source_id(1.0, vec![1.1, 1.2], Some(source_a)).unwrap(),
            Estimate::with_source_id(2.0, vec![2.1, 2.2], Some(source_b)).unwrap(),
        ];
        let combined = CrossSection::combine_with_factors(members, factors.clone()).unwrap();
        let projection = Projection::new("x", vec![fixture.axis.clone()]).unwrap();
        let actual = combined
            .projection_set(std::slice::from_ref(&projection), &fixture.components)
            .unwrap();

        for index in 0..2 {
            let paired_b = paired_draw_index(index, 1, 2, Some(source_b), Some(source_a));
            let explicit_members = [
                fixture
                    .likelihood
                    .cross_section(
                        "signal",
                        fixture.generated.clone(),
                        10.0,
                        ensemble_a.draws()[index].clone(),
                    )
                    .unwrap(),
                fixture
                    .likelihood
                    .cross_section(
                        "signal",
                        fixture.generated.clone(),
                        15.0,
                        ensemble_b.draws()[paired_b].clone(),
                    )
                    .unwrap(),
            ];
            let expected = CrossSection::combine_with_factors(
                explicit_members.into(),
                vec![
                    Estimate::central(factors[0].draws()[index]).unwrap(),
                    Estimate::central(factors[1].draws()[paired_b]).unwrap(),
                ],
            )
            .unwrap()
            .differential(projection.axes(), &fixture.components)
            .unwrap();
            let actual = actual.get("x").unwrap();
            assert_eq!(actual.data().draws()[index], expected.data().values());
            assert_eq!(actual.model().draws()[index], expected.model().values());
            for name in fixture.components.keys() {
                assert_eq!(
                    actual.components()[name].draws()[index],
                    expected.components()[name].values()
                );
            }
        }
    }

    #[test]
    fn combined_projection_sets_use_arbitrary_replica_event_rows() {
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
        let members = [1.0, 2.0]
            .into_iter()
            .map(|luminosity| {
                likelihood
                    .cross_section_with_ensemble(
                        "signal",
                        generated.clone(),
                        luminosity,
                        Vec::new(),
                        ensemble.clone(),
                    )
                    .unwrap()
            })
            .collect();
        let axis = Axis::new(event_scalar("x"), vec![0.0, 1.0, 2.0]).unwrap();
        let projection = Projection::new("x", vec![axis.clone()]).unwrap();
        let actual = CrossSection::combine(members)
            .unwrap()
            .projection_set(std::slice::from_ref(&projection), &HashMap::new())
            .unwrap();

        for (index, replica) in replicas.iter().enumerate() {
            let explicit_members = [1.0, 2.0]
                .into_iter()
                .map(|luminosity| {
                    replica
                        .cross_section("signal", generated.clone(), luminosity, Vec::new())
                        .unwrap()
                })
                .collect();
            let expected = CrossSection::combine(explicit_members)
                .unwrap()
                .differential(std::slice::from_ref(&axis), &HashMap::new())
                .unwrap();
            assert_eq!(
                actual.get("x").unwrap().data().draws()[index],
                expected.data().values()
            );
        }
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
