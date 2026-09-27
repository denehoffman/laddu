//! Estimates, physical units, and uncertainty propagation for fitted rates.

use std::{
    collections::{BTreeMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use auto_ops::impl_op_ex;
use laddu_expr::Expr;
use laddu_runtime::BinningAxis;

use crate::{Likelihood, LikelihoodError, LikelihoodResult, YieldHistogramView};

static NEXT_SOURCE_ID: AtomicU64 = AtomicU64::new(1);
const GENERATED_SOURCE_NAMESPACE: u64 = 1 << 63;

/// Returns a process-local identifier for an independent uncertainty source.
/// Generated identifiers occupy the upper half of `u64`; caller-assigned IDs
/// should use the lower half, except when deliberately reusing a returned ID.
///
/// # Panics
/// Panics only after the process has allocated the entire generated ID space.
pub fn next_uncertainty_source_id() -> u64 {
    let sequence = NEXT_SOURCE_ID.fetch_add(1, Ordering::Relaxed);
    assert!(
        sequence < GENERATED_SOURCE_NAMESPACE,
        "uncertainty source IDs exhausted"
    );
    GENERATED_SOURCE_NAMESPACE | sequence
}

fn invalid(message: impl Into<String>) -> LikelihoodError {
    LikelihoodError::InvalidCrossSection(message.into())
}

/// Named parameter draws with optional paired bootstrap likelihood replicas.
#[derive(Clone)]
pub struct Ensemble {
    parameter_names: Vec<String>,
    draws: Vec<Vec<f64>>,
    source_id: u64,
    replicas: Vec<Arc<Likelihood>>,
    replicas_share_event_rows: bool,
    bootstrap_seed: Option<u64>,
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
            bootstrap_seed: None,
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
        Self::with_replicas_and_source_id(
            parameter_names,
            draws,
            replicas,
            next_uncertainty_source_id(),
        )
    }

    /// Constructs paired replicas while restoring an existing correlation identity.
    ///
    /// # Errors
    /// Returns an error when draw values or replica counts are invalid.
    pub fn with_replicas_and_source_id(
        parameter_names: Vec<String>,
        draws: Vec<Vec<f64>>,
        replicas: Vec<Arc<Likelihood>>,
        source_id: u64,
    ) -> LikelihoodResult<Self> {
        let mut ensemble = Self::with_source_id(parameter_names, draws, source_id)?;
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
        ensemble.bootstrap_seed = Some(seed);
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

    /// Base seed for deterministic Poisson bootstrap replicas, when applicable.
    pub const fn bootstrap_seed(&self) -> Option<u64> {
        self.bootstrap_seed
    }

    /// Whether paired replicas depend on external arbitrary event datasets.
    pub fn requires_external_replica_datasets(&self) -> bool {
        !self.replicas.is_empty() && self.bootstrap_seed.is_none()
    }

    pub(crate) fn replicas_share_event_rows(&self) -> bool {
        self.replicas_share_event_rows
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
///
/// Use `checked_add`, `checked_sub`, `checked_mul`, and `checked_div` when
/// combining independent results. Arithmetic operators panic on mismatched
/// nonempty draw counts; checked methods return a recoverable error.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    central: f64,
    draws: Vec<f64>,
    source_id: Option<u64>,
    standard_error: Option<f64>,
    variance_components: BTreeMap<ErrorComponent, f64>,
    variance_source_ids: BTreeMap<ErrorComponent, u64>,
    paired_bootstrap: bool,
}

/// A scalar marginal error and its complete selection diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarErrorView {
    error: f64,
    budget: ErrorBudget,
    included: Vec<ErrorComponent>,
    omitted: Vec<(ErrorComponent, &'static str)>,
    sources: Vec<(ErrorComponent, u64)>,
}

impl ScalarErrorView {
    /// Selected marginal standard error.
    pub fn error(&self) -> f64 {
        self.error
    }
    /// Exact requested budget.
    pub fn budget(&self) -> ErrorBudget {
        self.budget
    }
    /// Included variance sources.
    pub fn included(&self) -> &[ErrorComponent] {
        &self.included
    }
    /// Selected but unavailable sources with machine-readable reasons.
    pub fn omitted(&self) -> &[(ErrorComponent, &'static str)] {
        &self.omitted
    }
    /// Provenance of included sources.
    pub fn sources(&self) -> &[(ErrorComponent, u64)] {
        &self.sources
    }
}

impl Estimate {
    /// Add with explicit draw-count validation.
    ///
    /// # Errors
    /// Returns an error for mismatched draws or a nonfinite result.
    pub fn checked_add(&self, other: &Self) -> LikelihoodResult<Self> {
        self.checked_binary(other, |a, b| a + b, |_, _| (1.0, 1.0))
    }

    /// Subtract with explicit draw-count validation.
    ///
    /// # Errors
    /// Returns an error for mismatched draws or a nonfinite result.
    pub fn checked_sub(&self, other: &Self) -> LikelihoodResult<Self> {
        self.checked_binary(other, |a, b| a - b, |_, _| (1.0, -1.0))
    }

    /// Multiply with explicit draw-count validation.
    ///
    /// # Errors
    /// Returns an error for mismatched draws or a nonfinite result.
    pub fn checked_mul(&self, other: &Self) -> LikelihoodResult<Self> {
        self.checked_binary(other, |a, b| a * b, |a, b| (b, a))
    }

    /// Divide with explicit draw-count validation.
    ///
    /// # Errors
    /// Returns an error for mismatched draws or a nonfinite result.
    pub fn checked_div(&self, other: &Self) -> LikelihoodResult<Self> {
        self.checked_binary(other, |a, b| a / b, |a, b| (1.0 / b, -a / b.powi(2)))
    }

    fn checked_binary(
        &self,
        other: &Self,
        op: impl Fn(f64, f64) -> f64,
        derivatives: impl Fn(f64, f64) -> (f64, f64),
    ) -> LikelihoodResult<Self> {
        if !self.draws.is_empty()
            && !other.draws.is_empty()
            && self.draws.len() != other.draws.len()
        {
            return Err(invalid("estimate draw counts do not match"));
        }
        let result = self.binary(other, op, derivatives);
        if !result.central.is_finite()
            || result.draws.iter().any(|value| !value.is_finite())
            || result
                .standard_error
                .is_some_and(|value| !value.is_finite())
        {
            return Err(invalid("estimate arithmetic produced a nonfinite value"));
        }
        Ok(result)
    }
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
            standard_error: None,
            variance_components: BTreeMap::new(),
            variance_source_ids: BTreeMap::new(),
            paired_bootstrap: false,
        })
    }

    fn from_evaluation(central: f64, draws: Vec<f64>, source_id: Option<u64>) -> Self {
        Self {
            central,
            draws,
            source_id,
            standard_error: None,
            variance_components: BTreeMap::new(),
            variance_source_ids: BTreeMap::new(),
            paired_bootstrap: false,
        }
    }

    /// Attach a known standard uncertainty without synthesizing draws.
    ///
    /// # Errors
    /// Returns an error for an invalid uncertainty or existing draws.
    pub fn with_standard_error(mut self, error: f64) -> LikelihoodResult<Self> {
        if !error.is_finite() || error < 0.0 || !self.draws.is_empty() {
            return Err(invalid(
                "standard error must be finite and nonnegative and cannot accompany draws",
            ));
        }
        self.standard_error = Some(error);
        if self.source_id.is_none() {
            self.source_id = Some(next_uncertainty_source_id());
        }
        Ok(self)
    }

    /// Known standard uncertainty, when supplied directly.
    pub fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }

    pub(crate) fn with_fill_variance(
        mut self,
        component: ErrorComponent,
        variance: f64,
        source_id: u64,
    ) -> Self {
        self.variance_components.insert(component, variance);
        self.variance_source_ids.insert(component, source_id);
        self
    }

    pub(crate) fn with_paired_bootstrap(mut self, paired: bool) -> Self {
        self.paired_bootstrap = paired;
        self
    }

    pub(crate) fn divided_by_luminosity(&self, luminosity: &Luminosity) -> Self {
        let mut converted = self.clone();
        let scale = 1.0 / luminosity.value();
        converted.central *= scale;
        for draw in &mut converted.draws {
            *draw *= scale;
        }
        if let Some(error) = &mut converted.standard_error {
            *error *= scale;
        }
        for variance in converted.variance_components.values_mut() {
            *variance *= scale * scale;
        }
        if let (Some(relative), Some(source_id)) =
            (luminosity.relative_uncertainty(), luminosity.source_id())
        {
            let variance = (converted.central * relative).powi(2);
            converted =
                converted.with_fill_variance(ErrorComponent::Luminosity, variance, source_id);
        }
        converted
    }

    /// Materialize a scalar marginal error without changing draws or covariance.
    ///
    /// # Errors
    /// Returns an error when selected fill statistics duplicate paired bootstrap variation.
    pub fn error_with_budget(&self, budget: ErrorBudget) -> LikelihoodResult<ScalarErrorView> {
        if budget.ensemble
            && self.paired_bootstrap
            && (budget.data_fill || budget.accepted_mc_fill || budget.generated_mc_fill)
        {
            return Err(invalid(
                "paired bootstrap draws already contain fill-statistical variation; disable fill statistics when selecting draw variation",
            ));
        }
        let mut variance = 0.0;
        let mut included = Vec::new();
        let mut omitted = Vec::new();
        let mut sources = Vec::new();
        for component in budget.selected() {
            if component == ErrorComponent::Ensemble {
                if self.draws.len() < 2 {
                    omitted.push((component, "fewer_than_two_draws"));
                } else {
                    variance += self.std()?.powi(2);
                    included.push(component);
                    if let Some(source_id) = self.source_id {
                        sources.push((component, source_id));
                    }
                }
            } else if component == ErrorComponent::BranchingExposure
                && self.standard_error.is_some()
            {
                variance += self.standard_error.unwrap_or(0.0).powi(2);
                included.push(component);
                if let Some(source_id) = self.source_id {
                    sources.push((component, source_id));
                }
            } else if let Some(value) = self.variance_components.get(&component) {
                variance += value;
                included.push(component);
                if let Some(source_id) = self.variance_source_ids.get(&component) {
                    sources.push((component, *source_id));
                }
            } else {
                omitted.push((component, "unavailable"));
            }
        }
        let error = if variance.is_finite() && variance >= 0.0 {
            variance.sqrt()
        } else {
            f64::NAN
        };
        Ok(ScalarErrorView {
            error,
            budget,
            included,
            omitted,
            sources,
        })
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

    /// Known standard error, or draw standard deviation using Bessel's correction.
    ///
    /// # Errors
    /// Returns an error when no known error and fewer than two draws are available.
    pub fn std(&self) -> LikelihoodResult<f64> {
        if let Some(error) = self.standard_error {
            return Ok(error);
        }
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

    fn binary(
        &self,
        other: &Self,
        op: impl Fn(f64, f64) -> f64,
        derivatives: impl Fn(f64, f64) -> (f64, f64),
    ) -> Self {
        assert!(
            self.draws.is_empty()
                || other.draws.is_empty()
                || self.draws.len() == other.draws.len(),
            "estimate draw counts do not match; use checked arithmetic for a recoverable error"
        );
        let count = match (self.draws.len(), other.draws.len()) {
            (0, 0) => 0,
            (0, right) => right,
            (left, 0) => left,
            (left, right) => left.min(right),
        };
        let draws = (0..count)
            .map(|index| {
                let left = self.draws.get(index).copied().unwrap_or(self.central);
                let right_index = if self.source_id.is_some() && self.source_id == other.source_id {
                    index
                } else {
                    (index + 1) % other.draws.len().max(1)
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
            (false, false) if self.source_id.is_some() && self.source_id == other.source_id => {
                self.source_id
            }
            (false, false) => Some(next_uncertainty_source_id()),
            (true, true) => None,
        };
        let mut result = Self::from_evaluation(op(self.central, other.central), draws, source_id);
        let (left_derivative, right_derivative) = derivatives(self.central, other.central);
        let left = left_derivative * self.standard_error.unwrap_or(0.0);
        let right = right_derivative * other.standard_error.unwrap_or(0.0);
        if self.standard_error.is_some() || other.standard_error.is_some() {
            let same_source = self.source_id.is_some() && self.source_id == other.source_id;
            result.standard_error = Some(if same_source {
                (left + right).abs()
            } else {
                left.hypot(right)
            });
            if result.source_id.is_none() {
                result.source_id = match (self.standard_error, other.standard_error) {
                    (Some(_), None) => self.source_id,
                    (None, Some(_)) => other.source_id,
                    _ => Some(next_uncertainty_source_id()),
                };
            }
        }
        result
    }
}

impl_op_ex!(+ |left: &Estimate, right: &Estimate| -> Estimate {
    left.binary(right, |a, b| a + b, |_, _| (1.0, 1.0))
});
impl_op_ex!(-|left: &Estimate, right: &Estimate| -> Estimate {
    left.binary(right, |a, b| a - b, |_, _| (1.0, -1.0))
});
impl_op_ex!(*|left: &Estimate, right: &Estimate| -> Estimate {
    left.binary(right, |a, b| a * b, |a, b| (b, a))
});
impl_op_ex!(/ |left: &Estimate, right: &Estimate| -> Estimate {
    left.binary(right, |a, b| a / b, |a, b| (1.0 / b, -a / b.powi(2)))
});

impl_op_ex!(+ |left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a + b,
        |_, _| (1.0, 1.0),
    )
});
impl_op_ex!(-|left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a - b,
        |_, _| (1.0, -1.0),
    )
});
impl_op_ex!(*|left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a * b,
        |a, b| (b, a),
    )
});
impl_op_ex!(/ |left: &Estimate, right: &f64| -> Estimate {
    left.binary(
        &Estimate::from_evaluation(*right, Vec::new(), None),
        |a, b| a / b,
        |a, b| (1.0 / b, -a / b.powi(2)),
    )
});

/// An expression and monotonically increasing bin edges.
#[derive(Clone, Debug)]
pub struct Axis {
    pub(crate) expression: Expr,
    pub(crate) binning: BinningAxis,
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
    source_id: Option<u64>,
    source_ids: Vec<u64>,
    axes: Option<Vec<Vec<f64>>>,
    unit: BinnedEstimateUnit,
    variance_components: BTreeMap<ErrorComponent, Vec<f64>>,
    variance_source_ids: BTreeMap<ErrorComponent, u64>,
    omitted_components: BTreeMap<ErrorComponent, &'static str>,
    paired_bootstrap: bool,
}

/// Independent sources that may contribute to a marginal error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorComponent {
    /// Squared observed-event weights.
    DataFill,
    /// Squared accepted Monte Carlo contributions.
    AcceptedMcFill,
    /// Squared generated Monte Carlo contributions.
    GeneratedMcFill,
    /// Variation across paired parameter or bootstrap draws.
    Ensemble,
    /// Luminosity uncertainty, when supplied by a typed conversion.
    Luminosity,
    /// Branching or exposure-factor uncertainty.
    BranchingExposure,
}

/// Requested sources for marginal histogram errors. Fill statistics are on by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ErrorBudget {
    /// Include observed-event fill statistics.
    pub data_fill: bool,
    /// Include accepted Monte Carlo fill statistics.
    pub accepted_mc_fill: bool,
    /// Include generated Monte Carlo fill statistics.
    pub generated_mc_fill: bool,
    /// Include ensemble variation.
    pub ensemble: bool,
    /// Include luminosity variation.
    pub luminosity: bool,
    /// Include branching or exposure-factor variation.
    pub branching_exposure: bool,
}

impl Default for ErrorBudget {
    fn default() -> Self {
        Self {
            data_fill: true,
            accepted_mc_fill: true,
            generated_mc_fill: true,
            ensemble: false,
            luminosity: false,
            branching_exposure: false,
        }
    }
}

impl ErrorBudget {
    fn selected(self) -> impl Iterator<Item = ErrorComponent> {
        [
            (self.data_fill, ErrorComponent::DataFill),
            (self.accepted_mc_fill, ErrorComponent::AcceptedMcFill),
            (self.generated_mc_fill, ErrorComponent::GeneratedMcFill),
            (self.ensemble, ErrorComponent::Ensemble),
            (self.luminosity, ErrorComponent::Luminosity),
            (self.branching_exposure, ErrorComponent::BranchingExposure),
        ]
        .into_iter()
        .filter_map(|(enabled, component)| enabled.then_some(component))
    }
}

/// Physical interpretation of binned values.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinnedEstimateUnit {
    /// Dimensionless ratio or factor.
    Unitless,
    /// Weighted event yield.
    Yield,
    /// Cross section normalized by exposure.
    CrossSection,
}

/// Supported area prefixes for cross-section results and reciprocal luminosity.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AreaUnit {
    /// Barn.
    Barn,
    /// Millibarn.
    Millibarn,
    /// Microbarn.
    Microbarn,
    /// Nanobarn.
    Nanobarn,
    /// Picobarn.
    Picobarn,
    /// Femtobarn.
    Femtobarn,
}

impl AreaUnit {
    fn barns(self) -> f64 {
        match self {
            Self::Barn => 1.0,
            Self::Millibarn => 1.0e-3,
            Self::Microbarn => 1.0e-6,
            Self::Nanobarn => 1.0e-9,
            Self::Picobarn => 1.0e-12,
            Self::Femtobarn => 1.0e-15,
        }
    }
}

/// Positive integrated luminosity in inverse area units.
#[derive(Clone, Debug, PartialEq)]
pub struct Luminosity {
    value: f64,
    unit: AreaUnit,
    relative_uncertainty: Option<f64>,
    source_id: Option<u64>,
}

impl Luminosity {
    /// Construct a finite, positive integrated luminosity.
    ///
    /// # Errors
    /// Returns an error for a nonfinite or nonpositive value.
    pub fn new(value: f64, unit: AreaUnit) -> LikelihoodResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid("luminosity must be finite and positive"));
        }
        Ok(Self {
            value,
            unit,
            relative_uncertainty: None,
            source_id: None,
        })
    }

    /// Attach an independent relative standard uncertainty.
    ///
    /// # Errors
    /// Returns an error for a nonfinite or negative uncertainty.
    pub fn with_relative_uncertainty(
        mut self,
        uncertainty: f64,
        source_id: u64,
    ) -> LikelihoodResult<Self> {
        if !uncertainty.is_finite() || uncertainty < 0.0 {
            return Err(invalid(
                "luminosity relative uncertainty must be finite and nonnegative",
            ));
        }
        self.relative_uncertainty = Some(uncertainty);
        self.source_id = Some(source_id);
        Ok(self)
    }

    /// Numeric luminosity in the stored reciprocal area unit.
    pub fn value(&self) -> f64 {
        self.value
    }
    /// Area prefix reciprocal to the luminosity unit.
    pub fn unit(&self) -> AreaUnit {
        self.unit
    }
    /// Relative standard uncertainty, if supplied.
    pub fn relative_uncertainty(&self) -> Option<f64> {
        self.relative_uncertainty
    }
    /// Correlation identity of luminosity uncertainty.
    pub fn source_id(&self) -> Option<u64> {
        self.source_id
    }

    /// Convert the same luminosity to another reciprocal area prefix.
    ///
    /// # Errors
    /// Returns an error if the converted value is not finite and positive.
    pub fn to_unit(&self, unit: AreaUnit) -> LikelihoodResult<Self> {
        let mut converted = self.clone();
        converted.value = self.value * unit.barns() / self.unit.barns();
        if !converted.value.is_finite() || converted.value <= 0.0 {
            return Err(invalid(
                "luminosity unit conversion produced a nonfinite or nonpositive value",
            ));
        }
        converted.unit = unit;
        Ok(converted)
    }
}

pub(crate) fn validate_factor(factor: &Estimate) -> LikelihoodResult<()> {
    if factor.central <= 0.0 || factor.draws.iter().any(|value| *value <= 0.0) {
        return Err(invalid(
            "exposure factors must be positive in the central value and every draw",
        ));
    }
    Ok(())
}

pub(crate) fn validate_factor_covariance(
    factors: &[&Estimate],
    covariance: &[Vec<f64>],
) -> LikelihoodResult<()> {
    let n = factors.len();
    if covariance.len() != n || covariance.iter().any(|row| row.len() != n) {
        return Err(invalid(
            "factor covariance must have one row and column per member",
        ));
    }
    if factors
        .iter()
        .any(|factor| factor.standard_error.is_some() || !factor.draws.is_empty())
    {
        return Err(invalid(
            "factor covariance cannot accompany factor errors or draws",
        ));
    }
    let scale = covariance
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = scale.max(f64::MIN_POSITIVE) * 1e-12;
    for (i, row) in covariance.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            if !value.is_finite() || (*value - covariance[j][i]).abs() > tolerance {
                return Err(invalid("factor covariance must be finite and symmetric"));
            }
        }
    }
    // Cholesky with a zero-pivot tolerance accepts positive semidefinite matrices.
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let residual =
                covariance[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            if i == j {
                if residual < -tolerance {
                    return Err(invalid("factor covariance must be positive semidefinite"));
                }
                lower[i][j] = residual.max(0.0).sqrt();
            } else if lower[j][j] > tolerance.sqrt() {
                lower[i][j] = residual / lower[j][j];
            } else if residual.abs() > tolerance {
                return Err(invalid("factor covariance must be positive semidefinite"));
            }
        }
    }
    Ok(())
}

fn factor_variance(
    derivatives: &[f64],
    factors: &[&Estimate],
    covariance: Option<&[Vec<f64>]>,
) -> f64 {
    match covariance {
        Some(matrix) => derivatives
            .iter()
            .enumerate()
            .map(|(i, di)| {
                derivatives
                    .iter()
                    .enumerate()
                    .map(|(j, dj)| di * matrix[i][j] * dj)
                    .sum::<f64>()
            })
            .sum(),
        None => derivatives
            .iter()
            .zip(factors)
            .map(|(derivative, factor)| {
                derivative.powi(2) * factor.standard_error.unwrap_or(0.0).powi(2)
            })
            .sum(),
    }
}

pub(crate) fn pool_fitted_scalar(
    members: &[(&Estimate, &Luminosity, &Estimate)],
    covariance: Option<&[Vec<f64>]>,
    factor_source_id: Option<u64>,
) -> LikelihoodResult<Estimate> {
    if let Some(matrix) = covariance {
        validate_factor_covariance(
            &members.iter().map(|(_, _, f)| *f).collect::<Vec<_>>(),
            matrix,
        )?;
    }
    let exposure = members
        .iter()
        .map(|(_, luminosity, factor)| luminosity.value() * factor.value())
        .sum::<f64>();
    if !exposure.is_finite() || exposure <= 0.0 {
        return Err(invalid(
            "total effective exposure must be finite and positive",
        ));
    }
    let draw_count = members
        .iter()
        .flat_map(|(generated, _, factor)| [generated.draws.len(), factor.draws.len()])
        .max()
        .unwrap_or(0);
    if members.iter().any(|(generated, _, factor)| {
        [generated.draws.len(), factor.draws.len()]
            .into_iter()
            .any(|count| count != 0 && count != draw_count)
    }) {
        return Err(invalid("combined member draw counts do not match"));
    }
    let numerator = members
        .iter()
        .map(|(generated, _, _)| generated.central)
        .sum::<f64>();
    let central = numerator / exposure;
    let draws = (0..draw_count)
        .map(|draw| {
            let draw_exposure = members
                .iter()
                .map(|(_, luminosity, factor)| {
                    luminosity.value() * factor.draws.get(draw).copied().unwrap_or(factor.central)
                })
                .sum::<f64>();
            if !draw_exposure.is_finite() || draw_exposure <= 0.0 {
                return Err(invalid(
                    "effective exposure must be positive for every draw",
                ));
            }
            Ok(members
                .iter()
                .map(|(generated, _, _)| {
                    generated
                        .draws
                        .get(draw)
                        .copied()
                        .unwrap_or(generated.central)
                })
                .sum::<f64>()
                / draw_exposure)
        })
        .collect::<LikelihoodResult<Vec<_>>>()?;
    let sources = members
        .iter()
        .flat_map(|(generated, _, factor)| [generated.source_id, factor.source_id])
        .flatten()
        .collect::<HashSet<_>>();
    let source_id = match sources.len() {
        0 => None,
        1 => sources.iter().copied().next(),
        _ => Some(next_uncertainty_source_id()),
    };
    let mut result = Estimate::from_evaluation(central, draws, source_id);
    result.paired_bootstrap = members
        .iter()
        .any(|(generated, _, _)| generated.paired_bootstrap);
    for (generated, luminosity, factor) in members {
        for (&component, &variance) in &generated.variance_components {
            *result.variance_components.entry(component).or_default() +=
                variance / exposure.powi(2);
            if let Some(&source) = generated.variance_source_ids.get(&component) {
                result
                    .variance_source_ids
                    .entry(component)
                    .or_insert(source);
            }
        }
        if let (Some(relative), Some(source)) =
            (luminosity.relative_uncertainty(), luminosity.source_id())
        {
            let derivative = -numerator * factor.central / exposure.powi(2);
            *result
                .variance_components
                .entry(ErrorComponent::Luminosity)
                .or_default() += (derivative * luminosity.value() * relative).powi(2);
            result
                .variance_source_ids
                .insert(ErrorComponent::Luminosity, source);
        }
    }
    let derivatives = members
        .iter()
        .map(|(_, luminosity, _)| -numerator * luminosity.value() / exposure.powi(2))
        .collect::<Vec<_>>();
    let variance = factor_variance(
        &derivatives,
        &members.iter().map(|(_, _, f)| *f).collect::<Vec<_>>(),
        covariance,
    );
    if covariance.is_some() || members.iter().any(|(_, _, f)| f.standard_error.is_some()) {
        result
            .variance_components
            .insert(ErrorComponent::BranchingExposure, variance.max(0.0));
        if let Some(source_id) = factor_source_id {
            result
                .variance_source_ids
                .insert(ErrorComponent::BranchingExposure, source_id);
        }
    }
    Ok(result)
}

pub(crate) fn pool_fitted_binned(
    members: &[(&BinnedEstimate, &Luminosity, &Estimate)],
    covariance: Option<&[Vec<f64>]>,
    factor_source_id: Option<u64>,
) -> LikelihoodResult<BinnedEstimate> {
    if let Some(matrix) = covariance {
        validate_factor_covariance(
            &members.iter().map(|(_, _, f)| *f).collect::<Vec<_>>(),
            matrix,
        )?;
    }
    let Some((first, _, _)) = members.first() else {
        return Err(invalid("at least one period is required"));
    };
    if members
        .iter()
        .any(|(estimate, _, _)| estimate.axes != first.axes || estimate.unit != first.unit)
    {
        return Err(invalid(
            "combined projections must share geometry and units",
        ));
    }
    let exposure = members
        .iter()
        .map(|(_, luminosity, factor)| luminosity.value() * factor.central)
        .sum::<f64>();
    if !exposure.is_finite() || exposure <= 0.0 {
        return Err(invalid(
            "total effective exposure must be finite and positive",
        ));
    }
    let draw_count = members
        .iter()
        .flat_map(|(estimate, _, factor)| [estimate.draws.len(), factor.draws.len()])
        .max()
        .unwrap_or(0);
    if members.iter().any(|(estimate, _, factor)| {
        [estimate.draws.len(), factor.draws.len()]
            .into_iter()
            .any(|count| count != 0 && count != draw_count)
    }) {
        return Err(invalid("combined member draw counts do not match"));
    }
    let mut result = (*first).clone();
    let numerators = (0..first.central.len())
        .map(|bin| {
            members
                .iter()
                .map(|(estimate, luminosity, _)| estimate.central[bin] * luminosity.value())
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    result.central = numerators.iter().map(|value| value / exposure).collect();
    result.draws = (0..draw_count)
        .map(|draw| {
            let draw_exposure = members
                .iter()
                .map(|(_, luminosity, factor)| {
                    luminosity.value() * factor.draws.get(draw).copied().unwrap_or(factor.central)
                })
                .sum::<f64>();
            if !draw_exposure.is_finite() || draw_exposure <= 0.0 {
                return Err(invalid(
                    "effective exposure must be positive for every draw",
                ));
            }
            Ok((0..first.central.len())
                .map(|bin| {
                    members
                        .iter()
                        .map(|(estimate, luminosity, _)| {
                            luminosity.value()
                                * estimate.draws.get(draw).unwrap_or(&estimate.central)[bin]
                        })
                        .sum::<f64>()
                        / draw_exposure
                })
                .collect::<Vec<_>>())
        })
        .collect::<LikelihoodResult<Vec<_>>>()?;
    result.variance_components.clear();
    result.variance_source_ids.clear();
    result.paired_bootstrap = members
        .iter()
        .any(|(estimate, _, _)| estimate.paired_bootstrap);
    result.source_ids = members
        .iter()
        .flat_map(|(estimate, _, _)| estimate.source_ids.iter().copied())
        .collect();
    let sources = members
        .iter()
        .flat_map(|(estimate, _, factor)| [estimate.source_id, factor.source_id])
        .flatten()
        .collect::<HashSet<_>>();
    result.source_id = match sources.len() {
        0 => None,
        1 => sources.iter().copied().next(),
        _ => Some(next_uncertainty_source_id()),
    };
    for (estimate, luminosity, _) in members {
        let coefficient = luminosity.value() / exposure;
        for (&component, variances) in &estimate.variance_components {
            let output = result
                .variance_components
                .entry(component)
                .or_insert_with(|| vec![0.0; first.central.len()]);
            for (sum, variance) in output.iter_mut().zip(variances) {
                *sum += coefficient.powi(2) * variance;
            }
            if let Some(&source) = estimate.variance_source_ids.get(&component) {
                result
                    .variance_source_ids
                    .entry(component)
                    .or_insert(source);
            }
        }
    }
    if covariance.is_some() || members.iter().any(|(_, _, f)| f.standard_error.is_some()) {
        let factors = members.iter().map(|(_, _, f)| *f).collect::<Vec<_>>();
        let variances = numerators
            .iter()
            .map(|numerator| {
                let derivatives = members
                    .iter()
                    .map(|(_, luminosity, _)| -numerator * luminosity.value() / exposure.powi(2))
                    .collect::<Vec<_>>();
                factor_variance(&derivatives, &factors, covariance).max(0.0)
            })
            .collect();
        result
            .variance_components
            .insert(ErrorComponent::BranchingExposure, variances);
        if let Some(source_id) = factor_source_id {
            result
                .variance_source_ids
                .insert(ErrorComponent::BranchingExposure, source_id);
        }
    }
    Ok(result)
}

impl BinnedEstimate {
    pub(crate) fn projected(
        central: Vec<f64>,
        draws: Vec<Vec<f64>>,
        source_id: Option<u64>,
        axes: Vec<Vec<f64>>,
        unit: BinnedEstimateUnit,
    ) -> Self {
        Self {
            central,
            draws,
            source_id,
            source_ids: source_id.into_iter().collect(),
            axes: Some(axes),
            unit,
            variance_components: BTreeMap::new(),
            variance_source_ids: BTreeMap::new(),
            omitted_components: BTreeMap::new(),
            paired_bootstrap: false,
        }
    }

    pub(crate) fn with_variance(mut self, component: ErrorComponent, variances: Vec<f64>) -> Self {
        self.variance_components.insert(component, variances);
        self
    }

    pub(crate) fn with_variance_source(
        mut self,
        component: ErrorComponent,
        source_id: u64,
    ) -> Self {
        self.variance_source_ids.insert(component, source_id);
        self
    }

    pub(crate) fn with_paired_bootstrap(mut self, paired: bool) -> Self {
        self.paired_bootstrap = paired;
        self
    }

    pub(crate) fn divided_by_luminosity_and_measure(
        &self,
        luminosity: &Luminosity,
        measure: &[f64],
    ) -> LikelihoodResult<Self> {
        if measure.len() != self.central.len()
            || measure
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid(
                "bin measures must be finite, positive, and match the yield shape",
            ));
        }
        let scale = measure
            .iter()
            .map(|value| 1.0 / (luminosity.value() * value))
            .collect::<Vec<_>>();
        if scale
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid(
                "luminosity and bin measure produce a nonfinite cross-section scale",
            ));
        }
        let mut result = self.clone();
        result.unit = BinnedEstimateUnit::CrossSection;
        for (value, factor) in result.central.iter_mut().zip(&scale) {
            *value *= factor;
        }
        for draw in &mut result.draws {
            for (value, factor) in draw.iter_mut().zip(&scale) {
                *value *= factor;
            }
        }
        for variances in result.variance_components.values_mut() {
            for (variance, factor) in variances.iter_mut().zip(&scale) {
                *variance *= factor * factor;
            }
        }
        if let (Some(relative), Some(source_id)) =
            (luminosity.relative_uncertainty(), luminosity.source_id())
        {
            let variances = result
                .central
                .iter()
                .map(|value| (value * relative).powi(2))
                .collect();
            result = result
                .with_variance(ErrorComponent::Luminosity, variances)
                .with_variance_source(ErrorComponent::Luminosity, source_id);
        }
        Ok(result)
    }

    /// Physical interpretation of these values.
    pub fn unit(&self) -> BinnedEstimateUnit {
        self.unit
    }

    /// Correlation source for the ensemble draws, if any.
    pub fn source_id(&self) -> Option<u64> {
        self.source_id
    }

    /// Original uncertainty source IDs contributing to these draws.
    pub fn source_ids(&self) -> &[u64] {
        &self.source_ids
    }

    /// Ordered axes, when this estimate came from a yield projection.
    pub fn axes(&self) -> Option<&[Vec<f64>]> {
        self.axes.as_deref()
    }

    /// Add another estimate with compatible geometry and uncertainty draws.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, units, or draws.
    pub fn checked_add(&self, other: &Self) -> LikelihoodResult<Self> {
        self.checked_binary(
            other,
            self.same_unit(other)?,
            |left, right| left + right,
            |_, _| (1.0, 1.0),
        )
    }

    /// Subtract another estimate with compatible geometry and uncertainty draws.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, units, or draws.
    pub fn checked_sub(&self, other: &Self) -> LikelihoodResult<Self> {
        self.checked_binary(
            other,
            self.same_unit(other)?,
            |left, right| left - right,
            |_, _| (1.0, -1.0),
        )
    }

    /// Multiply by another estimate with compatible geometry and uncertainty draws.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, draws, or units.
    pub fn checked_mul(&self, other: &Self) -> LikelihoodResult<Self> {
        let unit = match (self.unit, other.unit) {
            (BinnedEstimateUnit::Unitless, unit) | (unit, BinnedEstimateUnit::Unitless) => unit,
            _ => return Err(invalid("multiplication requires a dimensionless operand")),
        };
        self.checked_binary(
            other,
            unit,
            |left, right| left * right,
            |left, right| (right, left),
        )
    }

    /// Divide by another estimate with compatible geometry and uncertainty draws.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, draws, or units.
    pub fn checked_div(&self, other: &Self) -> LikelihoodResult<Self> {
        let unit = if self.unit == other.unit {
            BinnedEstimateUnit::Unitless
        } else if other.unit == BinnedEstimateUnit::Unitless {
            self.unit
        } else {
            return Err(invalid("binned estimate division has incompatible units"));
        };
        self.checked_binary(
            other,
            unit,
            |left, right| left / right,
            |left, right| (1.0 / right, -left / (right * right)),
        )
    }

    fn same_unit(&self, other: &Self) -> LikelihoodResult<BinnedEstimateUnit> {
        if self.unit != other.unit {
            return Err(invalid("binned estimate units do not match"));
        }
        Ok(self.unit)
    }

    fn checked_binary(
        &self,
        other: &Self,
        unit: BinnedEstimateUnit,
        op: impl Fn(f64, f64) -> f64,
        derivatives: impl Fn(f64, f64) -> (f64, f64),
    ) -> LikelihoodResult<Self> {
        if self.central.len() != other.central.len() || self.axes != other.axes {
            return Err(invalid("binned estimate geometry does not match"));
        }
        if !self.draws.is_empty()
            && !other.draws.is_empty()
            && self.draws.len() != other.draws.len()
        {
            return Err(invalid("binned estimate draw counts do not match"));
        }
        let count = self.draws.len().max(other.draws.len());
        let draws = (0..count)
            .map(|index| {
                let left = self.draws.get(index).unwrap_or(&self.central);
                let right_index = if self.source_id.is_some() && self.source_id == other.source_id {
                    index
                } else {
                    (index + 1) % other.draws.len().max(1)
                };
                let right = other.draws.get(right_index).unwrap_or(&other.central);
                left.iter().zip(right).map(|(&a, &b)| op(a, b)).collect()
            })
            .collect();
        let source_id = match (self.draws.is_empty(), other.draws.is_empty()) {
            (true, true) => None,
            (false, true) => self.source_id,
            (true, false) => other.source_id,
            (false, false) if self.source_id.is_some() && self.source_id == other.source_id => {
                self.source_id
            }
            (false, false) => Some(next_uncertainty_source_id()),
        };
        let mut variance_components = BTreeMap::new();
        let mut variance_source_ids = BTreeMap::new();
        let mut omitted_components = self.omitted_components.clone();
        omitted_components.extend(
            other
                .omitted_components
                .iter()
                .map(|(&key, &reason)| (key, reason)),
        );
        for component in self
            .variance_components
            .keys()
            .chain(other.variance_components.keys())
        {
            if variance_components.contains_key(component)
                || omitted_components.contains_key(component)
            {
                continue;
            }
            let left = self.variance_components.get(component);
            let right = other.variance_components.get(component);
            if left.is_some() && right.is_some() {
                omitted_components.insert(*component, "covariance_unavailable");
                continue;
            }
            let variances = self
                .central
                .iter()
                .zip(&other.central)
                .enumerate()
                .map(|(bin, (&a, &b))| {
                    let (da, db) = derivatives(a, b);
                    left.map_or(0.0, |values| da * da * values[bin])
                        + right.map_or(0.0, |values| db * db * values[bin])
                })
                .collect();
            variance_components.insert(*component, variances);
            if let Some(source_id) = self
                .variance_source_ids
                .get(component)
                .or_else(|| other.variance_source_ids.get(component))
            {
                variance_source_ids.insert(*component, *source_id);
            }
        }
        Ok(Self {
            central: self
                .central
                .iter()
                .zip(&other.central)
                .map(|(&a, &b)| op(a, b))
                .collect(),
            draws,
            source_id,
            source_ids: {
                let mut sources = self.source_ids.clone();
                for source in &other.source_ids {
                    if !sources.contains(source) {
                        sources.push(*source);
                    }
                }
                sources
            },
            axes: self.axes.clone(),
            unit,
            variance_components,
            variance_source_ids,
            omitted_components,
            paired_bootstrap: self.paired_bootstrap || other.paired_bootstrap,
        })
    }

    /// Materializes marginal errors from the requested independent sources.
    /// Full draws and covariance remain on the estimate.
    ///
    /// # Errors
    /// Returns an error for missing axes or incompatible error sources.
    pub fn histogram_with_budget(
        &self,
        budget: ErrorBudget,
    ) -> LikelihoodResult<YieldHistogramView> {
        if budget.ensemble
            && self.paired_bootstrap
            && (budget.data_fill || budget.accepted_mc_fill || budget.generated_mc_fill)
        {
            return Err(invalid(
                "paired bootstrap draws already contain fill-statistical variation; disable fill statistics when selecting draw variation",
            ));
        }
        let axes = self
            .axes
            .clone()
            .ok_or_else(|| invalid("binned estimate has no histogram axes"))?;
        let shape = axes.iter().map(|axis| axis.len() - 1).collect();
        let mut variance = vec![0.0; self.central.len()];
        let mut included = Vec::new();
        let mut omitted = Vec::new();
        for component in budget.selected() {
            if component == ErrorComponent::Ensemble {
                if self.draws.len() < 2 {
                    omitted.push((component, "fewer_than_two_draws"));
                } else {
                    let covariance = self.covariance()?;
                    for (bin, row) in covariance.iter().enumerate() {
                        variance[bin] += row[bin];
                    }
                    included.push(component);
                }
            } else if let Some(&reason) = self.omitted_components.get(&component) {
                omitted.push((component, reason));
            } else if let Some(values) = self.variance_components.get(&component) {
                for (total, value) in variance.iter_mut().zip(values) {
                    *total += value;
                }
                included.push(component);
            } else {
                omitted.push((component, "unavailable"));
            }
        }
        let unresolved_covariance = omitted
            .iter()
            .any(|(_, reason)| *reason == "covariance_unavailable");
        let errors = variance
            .iter()
            .zip(&self.central)
            .map(|(&value, &central)| {
                if !unresolved_covariance
                    && central.is_finite()
                    && value.is_finite()
                    && value >= 0.0
                {
                    value.sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect();
        let sources = included
            .iter()
            .filter_map(|component| {
                if *component == ErrorComponent::Ensemble {
                    self.source_id.map(|id| (*component, id))
                } else {
                    self.variance_source_ids
                        .get(component)
                        .map(|id| (*component, *id))
                }
            })
            .collect();
        Ok(YieldHistogramView {
            axes,
            shape,
            values: self.central.clone(),
            errors,
            budget,
            included,
            omitted,
            sources,
        })
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
