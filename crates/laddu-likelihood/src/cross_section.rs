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
use laddu_expr::Expr;
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

/// Named parameter draws with optional paired bootstrap likelihood replicas.
#[derive(Clone)]
pub struct Ensemble {
    parameter_names: Vec<String>,
    draws: Vec<Vec<f64>>,
    source_id: u64,
    replicas: Vec<Arc<Likelihood>>,
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
        Self::with_replicas(parameter_names, draws, replicas).map_err(Into::into)
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

/// A prepared total, tagged, differential, and combinable cross-section analysis.
type IntegralCacheKey = (usize, Option<Vec<String>>);
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

struct BinnedMeasurement {
    yields: Vec<f64>,
    exposures: Vec<f64>,
}

struct CombinedMemberValues {
    data: BinnedMeasurement,
    model: BinnedMeasurement,
    components: HashMap<String, BinnedMeasurement>,
}

struct CombinedMemberWorkspace {
    luminosity: f64,
    full: CrossSectionIntegrals,
    components: HashMap<String, CrossSectionIntegrals>,
    data_values: Vec<Vec<f64>>,
    data_weights: Vec<f64>,
    accepted_values: Vec<Vec<f64>>,
    accepted_weights: Vec<f64>,
    generated_values: Vec<Vec<f64>>,
    generated_weights: Vec<f64>,
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
        let component_integrals = components
            .iter()
            .map(|(name, tags)| {
                Ok((
                    name.clone(),
                    member.integrals_for(&member.likelihood, Some(tags))?,
                ))
            })
            .collect::<LikelihoodResult<HashMap<_, _>>>()?;
        Ok(Self {
            luminosity: member.luminosity,
            data_values: evaluate_coordinates(data, axes, execution)?,
            data_weights: dataset_weights(data)?,
            accepted_values: evaluate_coordinates(full.accepted_mc_source(), axes, execution)?,
            accepted_weights: dataset_weights(full.accepted_mc_source())?,
            generated_values: evaluate_coordinates(full.generated_mc_source(), axes, execution)?,
            generated_weights: dataset_weights(full.generated_mc_source())?,
            full,
            components: component_integrals,
        })
    }

    fn evaluate(
        &self,
        axes: &[Axis],
        parameters: &[f64],
        data_values: &[Vec<f64>],
        data_weights: &[f64],
        total_data: f64,
        factor: f64,
    ) -> LikelihoodResult<CombinedMemberValues> {
        let full_accepted_intensities = self.full.accepted_intensities(parameters)?;
        let full_generated_intensities = self.full.generated_intensities(parameters)?;
        let full_accepted_bins = histogram_products_nd(
            &self.accepted_values,
            &self.accepted_weights,
            &full_accepted_intensities,
            axes,
        );
        let full_generated_bins = histogram_products_nd(
            &self.generated_values,
            &self.generated_weights,
            &full_generated_intensities,
            axes,
        );
        let full_accepted = self.full.full_accepted_integral(parameters)?;
        let full_exposures = binned_exposures(
            self.luminosity * factor,
            &full_accepted_bins,
            &full_generated_bins,
        );
        let data = BinnedMeasurement {
            yields: histogram_nd(data_values, data_weights, axes),
            exposures: full_exposures.clone(),
        };
        let model = BinnedMeasurement {
            yields: full_accepted_bins
                .iter()
                .map(|accepted| total_data * accepted / full_accepted)
                .collect(),
            exposures: full_exposures,
        };
        let component_values = self
            .components
            .iter()
            .map(|(name, selected)| {
                let accepted_intensities = selected.accepted_intensities(parameters)?;
                let generated_intensities = selected.generated_intensities(parameters)?;
                let accepted = histogram_products_nd(
                    &self.accepted_values,
                    &self.accepted_weights,
                    &accepted_intensities,
                    axes,
                );
                let generated = histogram_products_nd(
                    &self.generated_values,
                    &self.generated_weights,
                    &generated_intensities,
                    axes,
                );
                Ok((
                    name.clone(),
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

    fn integrals_for(
        &self,
        likelihood: &Likelihood,
        tags: Option<&[String]>,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let key_tags = tags.map(|tags| {
            let mut tags = tags.to_vec();
            tags.sort();
            tags.dedup();
            tags
        });
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
        let integrals = match key_tags.as_deref() {
            Some(tags) => likelihood.cross_section_integrals_with_tags(
                &self.term_name,
                &self.generated_mc,
                tags.iter().map(String::as_str),
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
        let execution = self.likelihood.execution();
        let (data, _) = self.likelihood.intensity_datasets(&self.term_name)?;
        let full = self.integrals_for(&self.likelihood, None)?;
        let data_values = evaluate_coordinates(data, axes, execution)?;
        let accepted_values = evaluate_coordinates(full.accepted_mc_source(), axes, execution)?;
        let generated_values = evaluate_coordinates(full.generated_mc_source(), axes, execution)?;
        let data_weights = dataset_weights(data)?;
        let accepted_weights = dataset_weights(full.accepted_mc_source())?;
        let generated_weights = dataset_weights(full.generated_mc_source())?;
        let volumes = bin_volumes(axes);
        let component_integrals = components
            .iter()
            .map(|(name, tags)| {
                Ok((
                    name.clone(),
                    self.integrals_for(&self.likelihood, Some(tags))?,
                ))
            })
            .collect::<LikelihoodResult<HashMap<_, _>>>()?;
        let evaluate = |parameters: &[f64],
                        draw_data_values: &[Vec<f64>],
                        draw_data_weights: &[f64],
                        total_data: f64|
         -> LikelihoodResult<DifferentialValues> {
            let accepted_intensities = full.accepted_intensities(parameters)?;
            let generated_intensities = full.generated_intensities(parameters)?;
            let data_bins = histogram_nd(draw_data_values, draw_data_weights, axes);
            let accepted_bins = histogram_products_nd(
                &accepted_values,
                &accepted_weights,
                &accepted_intensities,
                axes,
            );
            let generated_bins = histogram_products_nd(
                &generated_values,
                &generated_weights,
                &generated_intensities,
                axes,
            );
            let full_accepted = full.full_accepted_integral(parameters)?;
            let data_cross_section = data_bins
                .iter()
                .zip(&accepted_bins)
                .zip(&generated_bins)
                .zip(&volumes)
                .map(|(((data, accepted), generated), volume)| {
                    if *accepted > 0.0 {
                        data * generated / accepted / self.luminosity / volume
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            let model = generated_bins
                .iter()
                .zip(&volumes)
                .map(|(generated, volume)| {
                    total_data * generated / full_accepted / self.luminosity / volume
                })
                .collect();
            let component_values = component_integrals
                .iter()
                .map(|(name, selected)| {
                    let intensities = selected.generated_intensities(parameters)?;
                    let bins = histogram_products_nd(
                        &generated_values,
                        &generated_weights,
                        &intensities,
                        axes,
                    );
                    Ok((
                        name.clone(),
                        bins.iter()
                            .zip(&volumes)
                            .map(|(generated, volume)| {
                                total_data * generated / full_accepted / self.luminosity / volume
                            })
                            .collect(),
                    ))
                })
                .collect::<LikelihoodResult<HashMap<_, _>>>()?;
            Ok((data_cross_section, model, component_values))
        };
        let (data_cross_section, model, component_values) = evaluate(
            &self.parameters,
            &data_values,
            &data_weights,
            full.data_weight_sum(),
        )?;
        let mut data_draws = Vec::new();
        let mut model_draws = Vec::new();
        let mut component_draws: HashMap<String, Vec<Vec<f64>>> = components
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect();
        if let Some(ensemble) = &self.ensemble {
            for (index, parameters) in ensemble.draws.iter().enumerate() {
                let replica_data = ensemble
                    .replicas
                    .get(index)
                    .map(|likelihood| likelihood.intensity_datasets(&self.term_name))
                    .transpose()?
                    .map(|(data, _)| data);
                let replica_values = replica_data
                    .map(|data| evaluate_coordinates(data, axes, execution))
                    .transpose()?;
                let replica_weights = replica_data.map(dataset_weights).transpose()?;
                let draw_data_values = replica_values.as_deref().unwrap_or(&data_values);
                let draw_data_weights = replica_weights.as_deref().unwrap_or(&data_weights);
                let total_data = replica_weights
                    .as_ref()
                    .map(|weights| weights.iter().sum())
                    .unwrap_or_else(|| full.data_weight_sum());
                let (data, model, component_values) =
                    evaluate(parameters, draw_data_values, draw_data_weights, total_data)?;
                data_draws.push(data);
                model_draws.push(model);
                for (name, values) in component_values {
                    component_draws.entry(name).or_default().push(values);
                }
            }
        }
        Ok(DifferentialCrossSection {
            axes: axes.iter().map(|axis| axis.edges.clone()).collect(),
            shape: axes.iter().map(Axis::bins).collect(),
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
                    axes,
                    &member.parameters,
                    &workspace.data_values,
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
                    let replica_values = replica_data
                        .map(|data| evaluate_coordinates(data, axes, member.likelihood.execution()))
                        .transpose()?;
                    let replica_weights = replica_data.map(dataset_weights).transpose()?;
                    let data_values = replica_values.as_deref().unwrap_or(&workspace.data_values);
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
                    workspace.evaluate(
                        axes,
                        parameters,
                        data_values,
                        data_weights,
                        total_data,
                        factor,
                    )
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

fn dataset_weights(dataset: &Dataset) -> LikelihoodResult<Vec<f64>> {
    dataset
        .try_fold_events(Vec::new(), |mut weights, event| {
            weights.push(event.weight());
            Ok(weights)
        })
        .map_err(Into::into)
}

fn histogram_nd(values: &[Vec<f64>], weights: &[f64], axes: &[Axis]) -> Vec<f64> {
    let mut bins = vec![0.0; axes.iter().map(Axis::bins).product()];
    for (event, &weight) in weights.iter().enumerate() {
        let index = axes
            .iter()
            .zip(values)
            .try_fold(0, |flat, (axis, coordinates)| {
                bin_index(coordinates[event], &axis.edges).map(|index| flat * axis.bins() + index)
            });
        if let Some(index) = index {
            bins[index] += weight;
        }
    }
    bins
}

fn histogram_products_nd(
    values: &[Vec<f64>],
    weights: &[f64],
    intensities: &[f64],
    axes: &[Axis],
) -> Vec<f64> {
    let products = weights
        .iter()
        .zip(intensities)
        .map(|(weight, intensity)| weight * intensity)
        .collect::<Vec<_>>();
    histogram_nd(values, &products, axes)
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
