use std::{
    collections::HashSet,
    fmt::Debug,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{LikelihoodError, LikelihoodResult};
use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::data::Dataset;
#[cfg(test)]
use laddu_expr::parameters::ParamError;
use laddu_expr::parameters::{ParamId, ParamLayout, ParamRegistry, ParamValues};
use laddu_runtime::{Execution, PreparedDataset, PreparedModel, RuntimeError};

/// Role of a prepared dataset within a likelihood term.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DatasetRole {
    /// Observed events entering the data contribution.
    Observed,
    /// Accepted Monte Carlo entering the normalization contribution.
    AcceptedMc,
}

/// Preparation diagnostics for one likelihood dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetDiagnostics {
    term: String,
    role: DatasetRole,
    stats: laddu_runtime::PreparedDatasetStats,
    quadratic_normalization: bool,
    source_traversals: u64,
}

impl DatasetDiagnostics {
    /// Returns the owning likelihood-term name.
    pub fn term(&self) -> &str {
        &self.term
    }

    /// Returns the dataset's role within the term.
    pub fn role(&self) -> DatasetRole {
        self.role
    }

    /// Returns the runtime preparation statistics.
    pub fn stats(&self) -> &laddu_runtime::PreparedDatasetStats {
        &self.stats
    }

    /// Returns whether accepted normalization uses precomputed quadratic statistics.
    pub fn uses_quadratic_normalization(&self) -> bool {
        self.quadratic_normalization
    }

    /// Returns the number of source traversals opened through this dataset view.
    pub fn source_traversals(&self) -> u64 {
        self.source_traversals
    }
}

/// Snapshot of likelihood preparation and evaluation behavior.
#[derive(Clone, Debug, PartialEq)]
pub struct LikelihoodDiagnostics {
    datasets: Vec<DatasetDiagnostics>,
    objective_evaluations: u64,
    gradient_evaluations: u64,
    memory_decisions: Vec<laddu_runtime::MemoryDecision>,
}

impl LikelihoodDiagnostics {
    /// Returns prepared-dataset records in term and role order.
    pub fn datasets(&self) -> &[DatasetDiagnostics] {
        &self.datasets
    }

    /// Returns the number of value-only objective requests.
    pub fn objective_evaluations(&self) -> u64 {
        self.objective_evaluations
    }

    /// Returns the number of value-and-gradient objective requests.
    pub fn gradient_evaluations(&self) -> u64 {
        self.gradient_evaluations
    }

    /// Returns memory-planning decisions recorded by the execution.
    pub fn memory_decisions(&self) -> &[laddu_runtime::MemoryDecision] {
        &self.memory_decisions
    }
}

/// Stable, user-supplied name identifying a likelihood term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LikelihoodName(String);

impl LikelihoodName {
    /// Creates a likelihood name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A scalar likelihood value and its free-parameter gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct LikelihoodEvaluation {
    value: f64,
    gradient: Vec<f64>,
}

impl LikelihoodEvaluation {
    /// Creates an evaluation from a value and gradient.
    pub fn new(value: f64, gradient: Vec<f64>) -> Self {
        Self { value, gradient }
    }

    /// Returns the objective value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Returns derivatives in free-parameter order.
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    /// Consumes the evaluation and returns its value and gradient.
    pub fn into_parts(self) -> (f64, Vec<f64>) {
        (self.value, self.gradient)
    }
}

/// Backend-neutral differentiable objective over a stable free-parameter layout.
///
/// This object-safe interface is intended for downstream optimizers and
/// samplers. It deliberately does not prescribe parameter transforms,
/// minimization strategy, or result types.
pub trait Objective: Debug + Send + Sync {
    /// Returns the stable parameter layout used by objective vectors.
    fn parameter_layout(&self) -> &ParamLayout;
    /// Evaluates the objective at the supplied free-parameter vector.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the parameter vector is invalid or
    /// objective evaluation fails.
    fn value(&self, free_parameters: &[f64]) -> LikelihoodResult<f64>;
    /// Evaluates the objective and its gradient.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the parameter vector is invalid or
    /// objective or gradient evaluation fails.
    fn value_gradient(&self, free_parameters: &[f64]) -> LikelihoodResult<LikelihoodEvaluation>;
}

/// An objective that can provide an unbiased stochastic value and gradient.
///
/// The `seed` identifies one deterministic batch. Implementations must use the
/// same batch for the returned value and gradient.
pub trait StochasticObjective: Objective {
    /// Evaluates an unbiased stochastic objective and gradient on a deterministic batch.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the parameter vector or batch fraction
    /// is invalid, or stochastic evaluation fails.
    fn stochastic_value_gradient(
        &self,
        free_parameters: &[f64],
        fraction: f64,
        seed: u64,
    ) -> LikelihoodResult<LikelihoodEvaluation>;
}

/// A composable contribution to a negative log likelihood.
pub trait LikelihoodTerm: Debug + Send + Sync {
    /// Returns the unique term name.
    fn name(&self) -> &str;

    /// Appends preparation diagnostics owned by this term.
    fn append_diagnostics(&self, _diagnostics: &mut Vec<DatasetDiagnostics>) {}

    /// Returns whether [`Self::bootstrap_clone`] preserves resolved preparation state.
    fn bootstrap_clone_is_prepared(&self) -> bool {
        false
    }

    /// Clones this term while applying a deterministic Poisson bootstrap to
    /// its observed dataset.
    ///
    /// Terms without observed event data should return an ordinary clone.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the term cannot be cloned for a
    /// bootstrap replica.
    fn bootstrap_clone(&self, _seed: u64) -> LikelihoodResult<Box<dyn LikelihoodTerm>> {
        Err(LikelihoodError::Runtime(RuntimeError::InvalidShape {
            index: 0,
            message: format!("term `{}` does not support bootstrap cloning", self.name()),
        }))
    }

    /// Registers parameters required by this term.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameter definitions are invalid or
    /// conflict with previously registered definitions.
    fn register_params(&self, _registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        Ok(())
    }

    /// Resolves the term against a global parameter layout and execution context.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when required parameters are missing or
    /// incompatible, or runtime preparation fails.
    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        execution: &Execution,
    ) -> LikelihoodResult<()>;

    /// Evaluates this term's negative-log-likelihood contribution.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are incompatible or term
    /// evaluation fails.
    fn nll(&self, params: &ParamValues, execution: &Execution) -> LikelihoodResult<f64>;

    /// Adds this term's gradient to `gradient` and returns its objective contribution.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters or gradient length are
    /// incompatible, or value or derivative evaluation fails.
    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
    ) -> LikelihoodResult<f64> {
        let layout = params.layout();
        if gradient.len() != layout.n_free() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: layout.n_free(),
                actual: gradient.len(),
            });
        }

        let value = self.nll(params, execution)?;
        for (free_index, id) in layout.free_params().iter().copied().enumerate() {
            let parameter = layout.spec(id)?;
            let free_id = layout
                .free_id(id)?
                .ok_or(LikelihoodError::ParameterLayoutMismatch)?;
            let center = params.get(id)?;
            let scale = center.abs().max(1.0);
            let base_step = f64::EPSILON.cbrt() * scale;
            let bounds = parameter.bounds_spec();
            let left_room = bounds
                .min
                .map_or(f64::INFINITY, |min| (center - min).max(0.0));
            let right_room = bounds
                .max
                .map_or(f64::INFINITY, |max| (max - center).max(0.0));

            let derivative = if left_room > 0.0 && right_room > 0.0 {
                let step = base_step.min(left_room).min(right_room);
                let mut plus = params.clone();
                let mut minus = params.clone();
                plus.set_free(free_id, center + step)?;
                minus.set_free(free_id, center - step)?;
                (self.nll(&plus, execution)? - self.nll(&minus, execution)?) / (2.0 * step)
            } else if right_room > 0.0 {
                let step = base_step.min(right_room);
                let mut plus = params.clone();
                plus.set_free(free_id, center + step)?;
                (self.nll(&plus, execution)? - value) / step
            } else if left_room > 0.0 {
                let step = base_step.min(left_room);
                let mut minus = params.clone();
                minus.set_free(free_id, center - step)?;
                (value - self.nll(&minus, execution)?) / step
            } else {
                0.0
            };
            gradient[free_index] += derivative;
        }
        Ok(value)
    }

    /// Evaluate a stochastic term contribution. Non-data terms remain exact by
    /// default; intensity terms override this to batch only observed events.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters, gradient length, or batch
    /// fraction are invalid, or evaluation fails.
    fn stochastic_nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
        _fraction: f64,
        _seed: u64,
    ) -> LikelihoodResult<f64> {
        self.nll_with_gradient(params, gradient, execution)
    }

    /// Returns this term as an intensity term when supported.
    fn as_intensity(&self) -> Option<&NllTerm> {
        None
    }

    /// Reports whether this term determines an absolute expected event rate.
    ///
    /// Shape-only intensity terms leave their overall scale unconstrained.
    /// Extended intensity terms override this when their normalization is part
    /// of the objective.
    fn has_absolute_rate(&self) -> bool {
        false
    }

    /// Boxes this term for use in a heterogeneous [`Likelihood`].
    fn boxed(self) -> Box<dyn LikelihoodTerm>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

/// Accepted parameter representations for likelihood evaluation.
pub enum Parameters<'a> {
    /// Free values in the likelihood's parameter order.
    Slice(&'a [f64]),
    /// Fully resolved parameter values.
    ParamValues(&'a ParamValues),
}

impl<'a> From<&'a [f64]> for Parameters<'a> {
    fn from(val: &'a [f64]) -> Self {
        Self::Slice(val)
    }
}

impl<'a, const N: usize> From<&'a [f64; N]> for Parameters<'a> {
    fn from(val: &'a [f64; N]) -> Self {
        Self::Slice(val.as_slice())
    }
}

impl<'a> From<&'a Vec<f64>> for Parameters<'a> {
    fn from(value: &'a Vec<f64>) -> Self {
        Self::Slice(value.as_slice())
    }
}

impl<'a> From<&'a ParamValues> for Parameters<'a> {
    fn from(val: &'a ParamValues) -> Self {
        Self::ParamValues(val)
    }
}

/// A resolved collection of likelihood terms sharing one parameter layout.
#[derive(Debug)]
pub struct Likelihood {
    params: Arc<ParamLayout>,
    terms: Vec<Box<dyn LikelihoodTerm>>,
    execution: Execution,
    objective_evaluations: AtomicU64,
    gradient_evaluations: AtomicU64,
}

impl Likelihood {
    /// Construct a likelihood from terms of one concrete type.
    ///
    /// The terms are boxed internally, so the common case does not require a
    /// manual [`LikelihoodTerm::boxed`] call.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when term names or parameter definitions
    /// conflict, or a term cannot be resolved.
    pub fn new<T>(terms: impl IntoIterator<Item = T>) -> LikelihoodResult<Self>
    where
        T: LikelihoodTerm + 'static,
    {
        Self::with_execution(terms, &Execution::default())
    }

    /// Construct a likelihood containing heterogeneous, already boxed terms.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when term names or parameter definitions
    /// conflict, or a term cannot be resolved.
    pub fn new_boxed(
        terms: impl IntoIterator<Item = Box<dyn LikelihoodTerm>>,
    ) -> LikelihoodResult<Self> {
        Self::with_execution_boxed(terms, &Execution::default())
    }

    /// Constructs a likelihood with an explicit execution context.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when term names or parameter definitions
    /// conflict, or a term cannot be resolved for `execution`.
    pub fn with_execution<T>(
        terms: impl IntoIterator<Item = T>,
        execution: &Execution,
    ) -> LikelihoodResult<Self>
    where
        T: LikelihoodTerm + 'static,
    {
        Self::with_execution_boxed(
            terms
                .into_iter()
                .map(|term| Box::new(term) as Box<dyn LikelihoodTerm>),
            execution,
        )
    }

    /// Construct a heterogeneous likelihood using a borrowed execution setup.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when term names or parameter definitions
    /// conflict, or a term cannot be resolved for `execution`.
    pub fn with_execution_boxed(
        terms: impl IntoIterator<Item = Box<dyn LikelihoodTerm>>,
        execution: &Execution,
    ) -> LikelihoodResult<Self> {
        let mut terms: Vec<_> = terms.into_iter().collect();
        let mut names = HashSet::new();
        let mut registry = ParamRegistry::new();

        for term in &terms {
            if !names.insert(term.name().to_owned()) {
                return Err(LikelihoodError::DuplicateTermName(term.name().to_owned()));
            }
            term.register_params(&mut registry)?;
        }

        let params = Arc::new(registry.layout()?);
        for term in &mut terms {
            term.resolve(Arc::clone(&params), execution)?;
        }

        Ok(Self {
            params,
            terms,
            execution: execution.clone(),
            objective_evaluations: AtomicU64::new(0),
            gradient_evaluations: AtomicU64::new(0),
        })
    }

    /// Returns the global parameter layout.
    pub fn params(&self) -> &ParamLayout {
        &self.params
    }

    /// Return deterministic initial values in the order expected by [`Self::nll`].
    pub fn default_params(&self) -> Vec<f64> {
        self.params.initial_free_values()
    }

    /// Generate one value for each free parameter in objective-vector order.
    pub fn params_with(
        &self,
        value: impl FnMut(&laddu_expr::parameters::Parameter) -> f64,
    ) -> Vec<f64> {
        self.params.free_values_with(value)
    }

    /// Sample uniform initial ranges while preserving fixed and point-initialized parameters.
    pub fn sample_initial(&self, seed: u64) -> Vec<f64> {
        self.params.sample_initial(seed)
    }

    /// Rebuilds this likelihood with Poisson-bootstrapped observed datasets.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when a term cannot be bootstrap-cloned or
    /// the rebuilt likelihood cannot be prepared.
    pub fn bootstrap(&self, seed: u64) -> LikelihoodResult<Self> {
        let clones_are_prepared = self
            .terms
            .iter()
            .all(|term| term.bootstrap_clone_is_prepared());
        let terms = self
            .terms
            .iter()
            .enumerate()
            .map(|(index, term)| {
                term.bootstrap_clone(
                    seed.wrapping_add((index as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                )
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        if clones_are_prepared {
            Ok(Self {
                params: Arc::clone(&self.params),
                terms,
                execution: self.execution.clone(),
                objective_evaluations: AtomicU64::new(0),
                gradient_evaluations: AtomicU64::new(0),
            })
        } else {
            Self::with_execution_boxed(terms, &self.execution)
        }
    }

    /// Returns the resolved likelihood terms.
    pub fn terms(&self) -> &[Box<dyn LikelihoodTerm>] {
        &self.terms
    }

    /// Returns the execution context used by the likelihood.
    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    /// Returns a snapshot of preparation and objective-evaluation diagnostics.
    pub fn diagnostics(&self) -> LikelihoodDiagnostics {
        let mut datasets = Vec::new();
        for term in &self.terms {
            term.append_diagnostics(&mut datasets);
        }
        LikelihoodDiagnostics {
            datasets,
            objective_evaluations: self.objective_evaluations.load(Ordering::Relaxed),
            gradient_evaluations: self.gradient_evaluations.load(Ordering::Relaxed),
            memory_decisions: self.execution.memory_decisions(),
        }
    }

    /// Evaluate the objective from free values in [`Self::params`] order.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters have the wrong layout or
    /// length, violate bounds, or a term cannot be evaluated.
    pub fn nll<'a>(&self, parameters: impl Into<Parameters<'a>>) -> LikelihoodResult<f64> {
        let params = match parameters.into() {
            Parameters::Slice(free) => &self.params.values(free)?,
            Parameters::ParamValues(param_values) => param_values,
        };
        self.nll_values(params)
    }

    fn nll_values(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.objective_evaluations.fetch_add(1, Ordering::Relaxed);
        check_params(&self.params, params)?;
        self.terms.iter().try_fold(
            0.0,
            |sum, term| Ok(sum + term.nll(params, &self.execution)?),
        )
    }

    /// Evaluates the objective and gradient from free or resolved parameter values.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid or a term's
    /// value or gradient cannot be evaluated.
    pub fn nll_with_gradient<'a>(
        &self,
        parameters: impl Into<Parameters<'a>>,
    ) -> LikelihoodResult<LikelihoodEvaluation> {
        let params = match parameters.into() {
            Parameters::Slice(free) => &self.params.values(free)?,
            Parameters::ParamValues(param_values) => param_values,
        };
        self.nll_with_gradient_values(params)
    }

    fn nll_with_gradient_values(
        &self,
        params: &ParamValues,
    ) -> LikelihoodResult<LikelihoodEvaluation> {
        self.gradient_evaluations.fetch_add(1, Ordering::Relaxed);
        check_params(&self.params, params)?;
        let mut gradient = vec![0.0; self.params.n_free()];
        let value = self.terms.iter().try_fold(0.0, |sum, term| {
            Ok::<_, LikelihoodError>(
                sum + term.nll_with_gradient(params, &mut gradient, &self.execution)?,
            )
        })?;
        Ok(LikelihoodEvaluation { value, gradient })
    }

    /// Evaluates an unbiased stochastic objective and gradient.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when `fraction` is outside `(0, 1]`,
    /// parameters are invalid, or stochastic term evaluation fails.
    pub fn stochastic_nll_with_gradient(
        &self,
        free_parameters: &[f64],
        fraction: f64,
        seed: u64,
    ) -> LikelihoodResult<LikelihoodEvaluation> {
        self.gradient_evaluations.fetch_add(1, Ordering::Relaxed);
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(LikelihoodError::InvalidBatchFraction(fraction));
        }
        let params = self.params.values(free_parameters)?;
        let mut gradient = vec![0.0; self.params.n_free()];
        let value = self
            .terms
            .iter()
            .enumerate()
            .try_fold(0.0, |sum, (term_index, term)| {
                let term_seed =
                    seed.wrapping_add((term_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
                Ok::<_, LikelihoodError>(
                    sum + term.stochastic_nll_with_gradient(
                        &params,
                        &mut gradient,
                        &self.execution,
                        fraction,
                        term_seed,
                    )?,
                )
            })?;
        Ok(LikelihoodEvaluation::new(value, gradient))
    }

    /// Prepares accepted and generated Monte Carlo integrals for an intensity term.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when `term_name` is missing or not an
    /// intensity term, or dataset preparation fails.
    pub fn cross_section_integrals(
        &self,
        term_name: &str,
        generated_mc: &Dataset,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let Some(term) = self.terms.iter().find(|term| term.name() == term_name) else {
            return Err(LikelihoodError::MissingTerm(term_name.to_owned()));
        };
        let has_absolute_rate = term.has_absolute_rate();
        let Some(term) = term.as_intensity() else {
            return Err(LikelihoodError::NotIntensityTerm(term_name.to_owned()));
        };
        term.cross_section_integrals(generated_mc, &self.execution, has_absolute_rate)
    }

    /// Prepares tag-narrowed accepted and generated Monte Carlo integrals.
    ///
    /// The selected tags define the numerator contribution. Cross sections
    /// retain the full accepted-model normalization, matching
    /// [`Self::projection`].
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when `term_name` is missing or not an
    /// intensity term, graph projection fails, or dataset preparation fails.
    pub fn cross_section_integrals_with_tags<'a>(
        &self,
        term_name: &str,
        generated_mc: &Dataset,
        tags: impl IntoIterator<Item = &'a str>,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let Some(term) = self.terms.iter().find(|term| term.name() == term_name) else {
            return Err(LikelihoodError::MissingTerm(term_name.to_owned()));
        };
        let has_absolute_rate = term.has_absolute_rate();
        let Some(term) = term.as_intensity() else {
            return Err(LikelihoodError::NotIntensityTerm(term_name.to_owned()));
        };
        term.cross_section_integrals_with_tags(
            generated_mc,
            tags,
            &self.execution,
            has_absolute_rate,
        )
    }

    /// Returns the observed and accepted Monte Carlo sources for an intensity term.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when `term_name` is missing or is not an
    /// intensity term.
    pub fn intensity_datasets(&self, term_name: &str) -> LikelihoodResult<(&Dataset, &Dataset)> {
        let Some(term) = self.terms.iter().find(|term| term.name() == term_name) else {
            return Err(LikelihoodError::MissingTerm(term_name.to_owned()));
        };
        let Some(term) = term.as_intensity() else {
            return Err(LikelihoodError::NotIntensityTerm(term_name.to_owned()));
        };
        Ok((&term.data_source, &term.accepted_mc_source))
    }

    /// Projects an intensity term onto selected model tags over generated Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when `term_name` is missing or not an
    /// intensity term, graph projection fails, or dataset preparation fails.
    pub fn projection<'a>(
        &self,
        term_name: &str,
        generated_mc: &Dataset,
        tags: impl IntoIterator<Item = &'a str>,
    ) -> LikelihoodResult<LikelihoodProjection> {
        let Some(term) = self.terms.iter().find(|term| term.name() == term_name) else {
            return Err(LikelihoodError::MissingTerm(term_name.to_owned()));
        };
        let has_absolute_rate = term.has_absolute_rate();
        let Some(term) = term.as_intensity() else {
            return Err(LikelihoodError::NotIntensityTerm(term_name.to_owned()));
        };
        term.projection(generated_mc, tags, &self.execution, has_absolute_rate)
    }
}

impl Objective for Likelihood {
    fn parameter_layout(&self) -> &ParamLayout {
        self.params()
    }

    fn value(&self, free_parameters: &[f64]) -> LikelihoodResult<f64> {
        self.nll(free_parameters)
    }

    fn value_gradient(&self, free_parameters: &[f64]) -> LikelihoodResult<LikelihoodEvaluation> {
        self.nll_with_gradient(free_parameters)
    }
}

impl StochasticObjective for Likelihood {
    fn stochastic_value_gradient(
        &self,
        free_parameters: &[f64],
        fraction: f64,
        seed: u64,
    ) -> LikelihoodResult<LikelihoodEvaluation> {
        self.stochastic_nll_with_gradient(free_parameters, fraction, seed)
    }
}

/// A shape-normalized unbinned negative-log-likelihood term.
#[derive(Clone, Debug)]
struct QuadraticNormalization {
    constant: f64,
    linear: Vec<f64>,
    quadratic: Vec<f64>,
}

impl QuadraticNormalization {
    fn prepare(
        model: &CompiledModel,
        plan: &PreparedModel,
        dataset: &PreparedDataset,
        execution: &Execution,
    ) -> LikelihoodResult<Option<Self>> {
        if model
            .parameter_polynomial_degree()
            .is_none_or(|degree| degree > 2)
        {
            return Ok(None);
        }
        let count = model.params().n_free();
        let evaluate = |free: &[f64]| -> LikelihoodResult<f64> {
            let values = model.params().values(free)?;
            plan.reduce(execution, &values, dataset, ReductionPlan::weighted_real())
                .map_err(|error| map_reduction_error("accepted MC", error))
        };
        let zero = vec![0.0; count];
        let constant = evaluate(&zero)?;
        let mut linear = vec![0.0; count];
        let mut quadratic = vec![0.0; count * count];
        for index in 0..count {
            let mut positive = zero.clone();
            positive[index] = 1.0;
            let mut negative = zero.clone();
            negative[index] = -1.0;
            let positive = evaluate(&positive)?;
            let negative = evaluate(&negative)?;
            linear[index] = 0.5 * (positive - negative);
            quadratic[index * count + index] = 0.5 * (positive + negative) - constant;
        }
        for row in 0..count {
            for col in row + 1..count {
                let mut pair = zero.clone();
                pair[row] = 1.0;
                pair[col] = 1.0;
                let cross = 0.5
                    * (evaluate(&pair)?
                        - constant
                        - linear[row]
                        - linear[col]
                        - quadratic[row * count + row]
                        - quadratic[col * count + col]);
                quadratic[row * count + col] = cross;
                quadratic[col * count + row] = cross;
            }
        }
        Ok(Some(Self {
            constant,
            linear,
            quadratic,
        }))
    }

    fn evaluate(&self, free: &[f64]) -> (f64, Vec<f64>) {
        let count = self.linear.len();
        let mut value = self.constant;
        let mut gradient = self.linear.clone();
        for row in 0..count {
            value += self.linear[row] * free[row];
            for col in 0..count {
                value += free[row] * self.quadratic[row * count + col] * free[col];
                gradient[row] += 2.0 * self.quadratic[row * count + col] * free[col];
            }
        }
        (value, gradient)
    }
}

#[derive(Clone)]
/// A shape-normalized unbinned negative-log-likelihood term.
pub struct NllTerm {
    name: LikelihoodName,
    model: CompiledModel,
    plan: Option<PreparedModel>,
    local_params: Arc<ParamLayout>,
    projection: Option<ParamProjection>,
    data_source: Dataset,
    accepted_mc_source: Dataset,
    data: Option<PreparedDataset>,
    accepted_mc: Option<PreparedDataset>,
    data_weight_sum: Option<f64>,
    quadratic_normalization: Option<QuadraticNormalization>,
    execution: Option<Execution>,
}

impl std::fmt::Debug for NllTerm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NllTerm")
            .field("name", &self.name)
            .field("prepared", &self.data.is_some())
            .finish_non_exhaustive()
    }
}

impl NllTerm {
    fn bootstrap_term(&self, seed: u64) -> LikelihoodResult<Self> {
        let data_source = self.data_source.clone().bootstrap(seed);
        let (Some(plan), Some(projection), Some(accepted_mc), Some(execution)) = (
            &self.plan,
            &self.projection,
            &self.accepted_mc,
            &self.execution,
        ) else {
            return Self::new(
                self.name(),
                &self.model,
                &data_source,
                &self.accepted_mc_source,
            );
        };
        let data = plan.prepare_dataset(execution, &data_source)?;
        let data_weight_sum = data.stats().sum_weights();
        Ok(Self {
            name: self.name.clone(),
            model: self.model.clone(),
            plan: Some(plan.clone()),
            local_params: Arc::clone(&self.local_params),
            projection: Some(projection.clone()),
            data_source,
            accepted_mc_source: self.accepted_mc_source.clone(),
            data: Some(data),
            accepted_mc: Some(accepted_mc.clone()),
            data_weight_sum: Some(data_weight_sum),
            quadratic_normalization: self.quadratic_normalization.clone(),
            execution: Some(execution.clone()),
        })
    }
    fn projection<'a>(
        &self,
        generated_mc: &Dataset,
        tags: impl IntoIterator<Item = &'a str>,
        execution: &Execution,
        has_absolute_rate: bool,
    ) -> LikelihoodResult<LikelihoodProjection> {
        let projected_model =
            self.model
                .project_tags(tags)
                .map_err(|error| RuntimeError::InvalidShape {
                    index: 0,
                    message: error.to_string(),
                })?;
        let projected_plan = PreparedModel::prepare(&projected_model, execution)?;
        let projected_params = ParamProjection::new(
            Arc::clone(&self.resolved_projection()?.global_layout),
            projected_model.params(),
            self.name(),
        )?;
        Ok(LikelihoodProjection {
            name: self.name.clone(),
            full_plan: self.plan()?.clone(),
            full_projection: self.resolved_projection()?.clone(),
            full_accepted_mc: self.accepted_mc()?.clone(),
            projected_accepted_mc: projected_plan
                .prepare_dataset(execution, &self.accepted_mc_source)?,
            projected_generated_mc: projected_plan.prepare_dataset(execution, generated_mc)?,
            accepted_mc_source: self.accepted_mc_source.clone(),
            generated_mc_source: generated_mc.clone(),
            projected_plan,
            projected_params,
            data_weight_sum: self.data_weight_sum()?,
            has_absolute_rate,
            execution: execution.clone(),
        })
    }

    /// Creates an unresolved intensity term from data and accepted Monte Carlo datasets.
    ///
    /// # Errors
    ///
    /// This constructor currently succeeds for all inputs. The result type is
    /// retained for compatibility with other fallible likelihood constructors.
    pub fn new(
        name: impl Into<String>,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> LikelihoodResult<Self> {
        Ok(Self {
            name: LikelihoodName::new(name),
            model: model.clone(),
            plan: None,
            local_params: Arc::new(model.params().clone()),
            projection: None,
            data_source: data.clone(),
            accepted_mc_source: accepted_mc.clone(),
            data: None,
            accepted_mc: None,
            data_weight_sum: None,
            quadratic_normalization: None,
            execution: None,
        })
    }

    /// Returns the prepared observed dataset.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::UnresolvedTerm`] when this term has not been
    /// resolved as part of a [`Likelihood`].
    pub fn data(&self) -> LikelihoodResult<&PreparedDataset> {
        self.data
            .as_ref()
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    fn plan(&self) -> LikelihoodResult<&PreparedModel> {
        self.plan
            .as_ref()
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    /// Returns the prepared accepted Monte Carlo dataset.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::UnresolvedTerm`] when this term has not been
    /// resolved as part of a [`Likelihood`].
    pub fn accepted_mc(&self) -> LikelihoodResult<&PreparedDataset> {
        self.accepted_mc
            .as_ref()
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    /// Returns the observed dataset's total event weight.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::UnresolvedTerm`] when this term has not been
    /// resolved as part of a [`Likelihood`].
    pub fn data_weight_sum(&self) -> LikelihoodResult<f64> {
        self.data_weight_sum
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    /// Returns the weighted log-intensity sum over observed data.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the term is unresolved, parameters are
    /// invalid, runtime evaluation fails, or an intensity is not positive.
    pub fn data_log_intensity_sum(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.global_values(free)?;
        let local_params = self.local_values(&params)?;
        self.reduce(
            &local_params,
            self.data()?,
            ReductionPlan::weighted_log_positive_real(),
            "data",
        )
    }

    /// Returns the weighted intensity integral over accepted Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the term is unresolved, parameters are
    /// invalid, runtime evaluation fails, or an intensity is not positive.
    pub fn accepted_normalization(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.global_values(free)?;
        let local_params = self.local_values(&params)?;
        self.normalization_value(&local_params, self.resolved_execution()?)
    }

    fn cross_section_integrals(
        &self,
        generated_mc: &Dataset,
        execution: &Execution,
        has_absolute_rate: bool,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let plan = self.plan()?.clone();
        Ok(CrossSectionIntegrals {
            name: self.name.clone(),
            full_plan: plan.clone(),
            full_projection: self.resolved_projection()?.clone(),
            full_accepted_mc: self.accepted_mc()?.clone(),
            accepted_mc_source: self.accepted_mc_source.clone(),
            generated_mc_source: generated_mc.clone(),
            plan: plan.clone(),
            projection: self.resolved_projection()?.clone(),
            accepted_mc: self.accepted_mc()?.clone(),
            generated_mc: plan.prepare_dataset(execution, generated_mc)?,
            data_weight_sum: self.data_weight_sum()?,
            has_absolute_rate,
            execution: execution.clone(),
        })
    }

    fn cross_section_integrals_with_tags<'a>(
        &self,
        generated_mc: &Dataset,
        tags: impl IntoIterator<Item = &'a str>,
        execution: &Execution,
        has_absolute_rate: bool,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let projection = self.projection(generated_mc, tags, execution, has_absolute_rate)?;
        Ok(CrossSectionIntegrals {
            name: projection.name,
            full_plan: projection.full_plan,
            full_projection: projection.full_projection,
            full_accepted_mc: projection.full_accepted_mc,
            accepted_mc_source: projection.accepted_mc_source,
            generated_mc_source: projection.generated_mc_source,
            plan: projection.projected_plan,
            projection: projection.projected_params,
            accepted_mc: projection.projected_accepted_mc,
            generated_mc: projection.projected_generated_mc,
            data_weight_sum: projection.data_weight_sum,
            has_absolute_rate: projection.has_absolute_rate,
            execution: projection.execution,
        })
    }

    fn normalization_value(
        &self,
        params: &ParamValues,
        execution: &Execution,
    ) -> LikelihoodResult<f64> {
        if let Some(normalization) = &self.quadratic_normalization {
            return Ok(normalization.evaluate(&params.free_values()).0);
        }
        self.plan()?
            .reduce(
                execution,
                params,
                self.accepted_mc()?,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error("accepted MC", error))
    }

    fn normalization_with_gradient(
        &self,
        params: &ParamValues,
        execution: &Execution,
    ) -> LikelihoodResult<(f64, Vec<f64>)> {
        if let Some(normalization) = &self.quadratic_normalization {
            return Ok(normalization.evaluate(&params.free_values()));
        }
        Ok(self
            .plan()?
            .reduce_with_gradient(
                execution,
                params,
                self.accepted_mc()?,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error("accepted MC", error))?
            .into_parts())
    }

    fn reduce(
        &self,
        params: &ParamValues,
        dataset: &PreparedDataset,
        reduction: ReductionPlan,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        self.plan()?
            .reduce(self.resolved_execution()?, params, dataset, reduction)
            .map_err(|error| map_reduction_error(name, error))
    }

    fn stochastic_data_evaluation(
        &self,
        params: &ParamValues,
        execution: &Execution,
        fraction: f64,
        seed: u64,
    ) -> LikelihoodResult<(f64, Vec<f64>)> {
        let selected = self.data_source.clone().subsample(fraction, seed)?;
        let prepared = self.plan()?.prepare_dataset(execution, &selected)?;
        let evaluation = self
            .plan()?
            .reduce_with_gradient(
                execution,
                params,
                &prepared,
                ReductionPlan::weighted_log_positive_real(),
            )
            .map_err(|error| map_reduction_error("data batch", error))?;
        let (value, gradient) = evaluation.into_parts();
        Ok((
            value / fraction,
            gradient.into_iter().map(|value| value / fraction).collect(),
        ))
    }

    fn local_values(&self, params: &ParamValues) -> LikelihoodResult<ParamValues> {
        self.resolved_projection()?.project(params)
    }

    fn global_values(&self, free: &[f64]) -> LikelihoodResult<ParamValues> {
        Ok(self.resolved_projection()?.global_layout.values(free)?)
    }

    fn resolved_projection(&self) -> LikelihoodResult<&ParamProjection> {
        self.projection
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)
    }

    fn resolved_execution(&self) -> LikelihoodResult<&Execution> {
        self.execution
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)
    }
}

impl LikelihoodTerm for NllTerm {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn append_diagnostics(&self, diagnostics: &mut Vec<DatasetDiagnostics>) {
        if let Some(data) = &self.data {
            diagnostics.push(DatasetDiagnostics {
                term: self.name.as_str().to_owned(),
                role: DatasetRole::Observed,
                stats: *data.stats(),
                quadratic_normalization: false,
                source_traversals: self.data_source.source_traversals(),
            });
        }
        if let Some(accepted_mc) = &self.accepted_mc {
            diagnostics.push(DatasetDiagnostics {
                term: self.name.as_str().to_owned(),
                role: DatasetRole::AcceptedMc,
                stats: *accepted_mc.stats(),
                quadratic_normalization: self.quadratic_normalization.is_some(),
                source_traversals: self.accepted_mc_source.source_traversals(),
            });
        }
    }

    fn bootstrap_clone_is_prepared(&self) -> bool {
        self.plan.is_some() && self.data.is_some() && self.accepted_mc.is_some()
    }

    fn bootstrap_clone(&self, seed: u64) -> LikelihoodResult<Box<dyn LikelihoodTerm>> {
        Ok(Box::new(self.bootstrap_term(seed)?))
    }

    fn register_params(&self, registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        for spec in self.local_params.specs() {
            registry.register(spec.clone())?;
        }
        Ok(())
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        execution: &Execution,
    ) -> LikelihoodResult<()> {
        self.projection = Some(ParamProjection::new(
            global_params,
            &self.local_params,
            self.name(),
        )?);
        let plan = PreparedModel::prepare(&self.model, execution)?;
        let data = plan.prepare_dataset(execution, &self.data_source)?;
        let accepted_mc = plan.prepare_dataset(execution, &self.accepted_mc_source)?;
        let quadratic_normalization =
            QuadraticNormalization::prepare(&self.model, &plan, &accepted_mc, execution)?;
        self.data = Some(data);
        self.accepted_mc = Some(accepted_mc);
        self.plan = Some(plan);
        self.data_weight_sum = Some(self.data()?.stats().sum_weights());
        self.quadratic_normalization = quadratic_normalization;
        self.execution = Some(execution.clone());
        Ok(())
    }

    fn nll(&self, params: &ParamValues, execution: &Execution) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        let normalization = positive_integral(
            "accepted MC",
            self.normalization_value(&local_params, execution)?,
        )?;
        let data_log_sum = self
            .plan()?
            .reduce(
                execution,
                &local_params,
                self.data()?,
                ReductionPlan::weighted_log_positive_real(),
            )
            .map_err(|error| map_reduction_error("data", error))?;
        Ok(self.data_weight_sum()? * normalization.ln() - data_log_sum)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
    ) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        let (normalization, normalization_gradient) =
            self.normalization_with_gradient(&local_params, execution)?;
        let normalization = positive_integral("accepted MC", normalization)?;
        let data_evaluation = self
            .plan()?
            .reduce_with_gradient(
                execution,
                &local_params,
                self.data()?,
                ReductionPlan::weighted_log_positive_real(),
            )
            .map_err(|error| map_reduction_error("data", error))?;
        let (data_log_sum, data_log_gradient) = data_evaluation.into_parts();
        let data_weight_sum = self.data_weight_sum()?;
        let local_gradient = normalization_gradient
            .into_iter()
            .zip(data_log_gradient)
            .map(|(normalization_derivative, data_derivative)| {
                data_weight_sum * normalization_derivative / normalization - data_derivative
            })
            .collect::<Vec<_>>();
        self.resolved_projection()?
            .scatter_gradient(&local_gradient, gradient)?;
        Ok(data_weight_sum * normalization.ln() - data_log_sum)
    }

    fn stochastic_nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
        fraction: f64,
        seed: u64,
    ) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        let normalization_evaluation = self.plan()?.reduce_with_gradient(
            execution,
            &local_params,
            self.accepted_mc()?,
            ReductionPlan::weighted_positive_real(),
        )?;
        let (normalization, normalization_gradient) = normalization_evaluation.into_parts();
        let normalization = positive_integral("accepted MC", normalization)?;
        let (data_log_sum, data_log_gradient) =
            self.stochastic_data_evaluation(&local_params, execution, fraction, seed)?;
        let data_weight_sum = self.data_weight_sum()?;
        let local_gradient = normalization_gradient
            .into_iter()
            .zip(data_log_gradient)
            .map(|(normalization_derivative, data_derivative)| {
                data_weight_sum * normalization_derivative / normalization - data_derivative
            })
            .collect::<Vec<_>>();
        self.resolved_projection()?
            .scatter_gradient(&local_gradient, gradient)?;
        Ok(data_weight_sum * normalization.ln() - data_log_sum)
    }

    fn as_intensity(&self) -> Option<&NllTerm> {
        Some(self)
    }
}

/// An extended unbinned negative-log-likelihood term.
///
/// Unlike [`NllTerm`], the normalization is the expected event yield rather
/// than a shape-only normalization raised to the observed weighted yield.
#[derive(Clone)]
pub struct ExtendedNllTerm {
    inner: NllTerm,
}

impl std::fmt::Debug for ExtendedNllTerm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExtendedNllTerm")
            .field("name", &self.inner.name)
            .field("prepared", &self.inner.data.is_some())
            .finish_non_exhaustive()
    }
}

impl ExtendedNllTerm {
    /// Creates an unresolved extended intensity term.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] if construction of the underlying intensity
    /// term fails.
    pub fn new(
        name: impl Into<String>,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> LikelihoodResult<Self> {
        Ok(Self {
            inner: NllTerm::new(name, model, data, accepted_mc)?,
        })
    }

    /// Returns the prepared observed dataset.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::UnresolvedTerm`] when this term has not been
    /// resolved.
    pub fn data(&self) -> LikelihoodResult<&PreparedDataset> {
        self.inner.data()
    }

    /// Returns the prepared accepted Monte Carlo dataset.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::UnresolvedTerm`] when this term has not been
    /// resolved.
    pub fn accepted_mc(&self) -> LikelihoodResult<&PreparedDataset> {
        self.inner.accepted_mc()
    }

    /// Returns the observed dataset's total event weight.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::UnresolvedTerm`] when this term has not been
    /// resolved.
    pub fn data_weight_sum(&self) -> LikelihoodResult<f64> {
        self.inner.data_weight_sum()
    }

    /// Returns the weighted log-intensity sum over observed data.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the term is unresolved, parameters are
    /// invalid, runtime evaluation fails, or an intensity is not positive.
    pub fn data_log_intensity_sum(&self, free: &[f64]) -> LikelihoodResult<f64> {
        self.inner.data_log_intensity_sum(free)
    }

    /// Returns the weighted intensity integral over accepted Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the term is unresolved, parameters are
    /// invalid, runtime evaluation fails, or an intensity is not positive.
    pub fn accepted_normalization(&self, free: &[f64]) -> LikelihoodResult<f64> {
        self.inner.accepted_normalization(free)
    }
}

impl LikelihoodTerm for ExtendedNllTerm {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn append_diagnostics(&self, diagnostics: &mut Vec<DatasetDiagnostics>) {
        self.inner.append_diagnostics(diagnostics);
    }

    fn bootstrap_clone_is_prepared(&self) -> bool {
        self.inner.bootstrap_clone_is_prepared()
    }

    fn bootstrap_clone(&self, seed: u64) -> LikelihoodResult<Box<dyn LikelihoodTerm>> {
        Ok(Box::new(Self {
            inner: self.inner.bootstrap_term(seed)?,
        }))
    }

    fn register_params(&self, registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        self.inner.register_params(registry)
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        execution: &Execution,
    ) -> LikelihoodResult<()> {
        self.inner.resolve(global_params, execution)
    }

    fn nll(&self, params: &ParamValues, execution: &Execution) -> LikelihoodResult<f64> {
        let local_params = self.inner.local_values(params)?;
        let normalization = positive_integral(
            "accepted MC",
            self.inner.normalization_value(&local_params, execution)?,
        )?;
        let data_log_sum = self
            .inner
            .plan()?
            .reduce(
                execution,
                &local_params,
                self.inner.data()?,
                ReductionPlan::weighted_log_positive_real(),
            )
            .map_err(|error| map_reduction_error("data", error))?;
        Ok(normalization - data_log_sum)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
    ) -> LikelihoodResult<f64> {
        let local_params = self.inner.local_values(params)?;
        let (normalization, normalization_gradient) = self
            .inner
            .normalization_with_gradient(&local_params, execution)?;
        let normalization = positive_integral("accepted MC", normalization)?;
        let data_evaluation = self
            .inner
            .plan()?
            .reduce_with_gradient(
                execution,
                &local_params,
                self.inner.data()?,
                ReductionPlan::weighted_log_positive_real(),
            )
            .map_err(|error| map_reduction_error("data", error))?;
        let (data_log_sum, data_log_gradient) = data_evaluation.into_parts();
        let local_gradient = normalization_gradient
            .into_iter()
            .zip(data_log_gradient)
            .map(|(normalization_derivative, data_derivative)| {
                normalization_derivative - data_derivative
            })
            .collect::<Vec<_>>();
        self.inner
            .resolved_projection()?
            .scatter_gradient(&local_gradient, gradient)?;
        Ok(normalization - data_log_sum)
    }

    fn stochastic_nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
        fraction: f64,
        seed: u64,
    ) -> LikelihoodResult<f64> {
        let local_params = self.inner.local_values(params)?;
        let (normalization, normalization_gradient) = self
            .inner
            .normalization_with_gradient(&local_params, execution)?;
        let normalization = positive_integral("accepted MC", normalization)?;
        let (data_log_sum, data_log_gradient) =
            self.inner
                .stochastic_data_evaluation(&local_params, execution, fraction, seed)?;
        let local_gradient = normalization_gradient
            .into_iter()
            .zip(data_log_gradient)
            .map(|(normalization_derivative, data_derivative)| {
                normalization_derivative - data_derivative
            })
            .collect::<Vec<_>>();
        self.inner
            .resolved_projection()?
            .scatter_gradient(&local_gradient, gradient)?;
        Ok(normalization - data_log_sum)
    }

    fn as_intensity(&self) -> Option<&NllTerm> {
        Some(&self.inner)
    }

    fn has_absolute_rate(&self) -> bool {
        true
    }
}

/// Quadratic regularization over selected parameters.
#[derive(Clone, Debug)]
pub struct RidgePenalty {
    inner: CpuParameterPenalty,
}

impl RidgePenalty {
    /// Creates a ridge penalty with weight `lambda`.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the name or parameter list is empty or
    /// duplicated, or `lambda` is negative or non-finite.
    pub fn new(
        name: impl Into<String>,
        parameter_names: impl IntoIterator<Item = impl Into<String>>,
        lambda: f64,
    ) -> LikelihoodResult<Self> {
        Ok(Self {
            inner: CpuParameterPenalty::new(name, parameter_names, lambda, PenaltyKind::Ridge)?,
        })
    }
}

impl LikelihoodTerm for RidgePenalty {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn bootstrap_clone(&self, _seed: u64) -> LikelihoodResult<Box<dyn LikelihoodTerm>> {
        Ok(Box::new(self.clone()))
    }

    fn bootstrap_clone_is_prepared(&self) -> bool {
        self.inner.global_params.is_some()
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        _execution: &Execution,
    ) -> LikelihoodResult<()> {
        self.inner.resolve(global_params)
    }

    fn nll(&self, params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
        self.inner.nll(params)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        _execution: &Execution,
    ) -> LikelihoodResult<f64> {
        self.inner.nll_with_gradient(params, gradient)
    }
}

/// Absolute-value regularization over selected parameters.
#[derive(Clone, Debug)]
pub struct LassoPenalty {
    inner: CpuParameterPenalty,
}

impl LassoPenalty {
    /// Creates a lasso penalty with weight `lambda`.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when the name or parameter list is empty or
    /// duplicated, or `lambda` is negative or non-finite.
    pub fn new(
        name: impl Into<String>,
        parameter_names: impl IntoIterator<Item = impl Into<String>>,
        lambda: f64,
    ) -> LikelihoodResult<Self> {
        Ok(Self {
            inner: CpuParameterPenalty::new(name, parameter_names, lambda, PenaltyKind::Lasso)?,
        })
    }
}

impl LikelihoodTerm for LassoPenalty {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn bootstrap_clone(&self, _seed: u64) -> LikelihoodResult<Box<dyn LikelihoodTerm>> {
        Ok(Box::new(self.clone()))
    }

    fn bootstrap_clone_is_prepared(&self) -> bool {
        self.inner.global_params.is_some()
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        _execution: &Execution,
    ) -> LikelihoodResult<()> {
        self.inner.resolve(global_params)
    }

    fn nll(&self, params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
        self.inner.nll(params)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        _execution: &Execution,
    ) -> LikelihoodResult<f64> {
        self.inner.nll_with_gradient(params, gradient)
    }
}

#[derive(Clone, Debug)]
struct CpuParameterPenalty {
    name: LikelihoodName,
    parameter_names: Vec<String>,
    parameter_ids: Vec<ParamId>,
    global_params: Option<Arc<ParamLayout>>,
    lambda: f64,
    kind: PenaltyKind,
}

impl CpuParameterPenalty {
    fn new(
        name: impl Into<String>,
        parameter_names: impl IntoIterator<Item = impl Into<String>>,
        lambda: f64,
        kind: PenaltyKind,
    ) -> LikelihoodResult<Self> {
        let name = LikelihoodName::new(name);
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(LikelihoodError::InvalidPenaltyWeight {
                term: name.as_str().to_owned(),
                lambda,
            });
        }
        Ok(Self {
            name,
            parameter_names: parameter_names.into_iter().map(Into::into).collect(),
            parameter_ids: Vec::new(),
            global_params: None,
            lambda,
            kind,
        })
    }

    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()> {
        self.parameter_ids = self
            .parameter_names
            .iter()
            .map(|parameter| {
                global_params
                    .id(parameter)
                    .ok_or_else(|| LikelihoodError::MissingParameter {
                        term: self.name().to_owned(),
                        parameter: parameter.clone(),
                    })
            })
            .collect::<LikelihoodResult<_>>()?;
        self.global_params = Some(global_params);
        Ok(())
    }

    fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let global_params = self
            .global_params
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)?;
        check_params(global_params, params)?;
        let mut sum = 0.0;
        for id in &self.parameter_ids {
            let value = params.get(*id)?;
            sum += match self.kind {
                PenaltyKind::Ridge => value * value,
                PenaltyKind::Lasso => value.abs(),
            };
        }
        Ok(self.lambda * sum)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
    ) -> LikelihoodResult<f64> {
        let global_params = self
            .global_params
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)?;
        check_params(global_params, params)?;
        if gradient.len() != global_params.n_free() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: global_params.n_free(),
                actual: gradient.len(),
            });
        }
        let mut sum = 0.0;
        for id in &self.parameter_ids {
            let value = params.get(*id)?;
            let (penalty, derivative) = match self.kind {
                PenaltyKind::Ridge => (value * value, 2.0 * value),
                PenaltyKind::Lasso => {
                    (value.abs(), if value == 0.0 { 0.0 } else { value.signum() })
                }
            };
            sum += penalty;
            if let Some(free) = global_params.free_id(*id)? {
                gradient[free.index()] += self.lambda * derivative;
            }
        }
        Ok(self.lambda * sum)
    }
}

#[derive(Copy, Clone, Debug)]
enum PenaltyKind {
    Ridge,
    Lasso,
}

/// Prepared accepted and generated Monte Carlo integrals for an intensity model.
#[derive(Clone)]
pub struct CrossSectionIntegrals {
    name: LikelihoodName,
    full_plan: PreparedModel,
    full_projection: ParamProjection,
    full_accepted_mc: PreparedDataset,
    accepted_mc_source: Dataset,
    generated_mc_source: Dataset,
    plan: PreparedModel,
    projection: ParamProjection,
    accepted_mc: PreparedDataset,
    generated_mc: PreparedDataset,
    data_weight_sum: f64,
    has_absolute_rate: bool,
    execution: Execution,
}

impl std::fmt::Debug for CrossSectionIntegrals {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CrossSectionIntegrals")
            .field("name", &self.name)
            .field("data_weight_sum", &self.data_weight_sum)
            .finish_non_exhaustive()
    }
}

/// A tag-projected intensity model evaluated over generated Monte Carlo.
#[derive(Clone)]
pub struct LikelihoodProjection {
    name: LikelihoodName,
    full_plan: PreparedModel,
    full_projection: ParamProjection,
    full_accepted_mc: PreparedDataset,
    projected_plan: PreparedModel,
    projected_params: ParamProjection,
    projected_accepted_mc: PreparedDataset,
    projected_generated_mc: PreparedDataset,
    accepted_mc_source: Dataset,
    generated_mc_source: Dataset,
    data_weight_sum: f64,
    has_absolute_rate: bool,
    execution: Execution,
}

impl LikelihoodProjection {
    /// Returns the source likelihood term name.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the projected intensity integral over accepted Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid, runtime
    /// evaluation fails, or an intensity is not positive.
    pub fn accepted_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        self.projected_integral(free, &self.projected_accepted_mc, "accepted MC")
    }

    /// Returns the projected intensity integral over generated Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid, runtime
    /// evaluation fails, or an intensity is not positive.
    pub fn generated_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        self.projected_integral(free, &self.projected_generated_mc, "generated MC")
    }

    /// Returns the projected acceptance ratio.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when either accepted or generated integral
    /// cannot be evaluated or is not positive.
    pub fn acceptance(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let generated = positive_integral("generated MC", self.generated_integral(free)?)?;
        let accepted = positive_integral("accepted MC", self.accepted_integral(free)?)?;
        Ok(accepted / generated)
    }

    /// Returns the unprojected accepted Monte Carlo integral.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid, runtime
    /// evaluation fails, or an intensity is not positive.
    pub fn full_accepted_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let global = self.full_projection.global_layout.values(free)?;
        let local = self.full_projection.project(&global)?;
        self.full_plan
            .reduce(
                &self.execution,
                &local,
                &self.full_accepted_mc,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error("accepted MC", error))
    }

    /// Returns the projected, acceptance-corrected event yield.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when accepted or generated integrals cannot
    /// be evaluated or the accepted integral is not positive.
    pub fn acceptance_corrected_yield(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let accepted = positive_integral("accepted MC", self.full_accepted_integral(free)?)?;
        Ok(self.data_weight_sum * self.generated_integral(free)? / accepted)
    }

    /// Returns the observed-yield-normalized projected cross section.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when `luminosity` is not positive or the
    /// acceptance-corrected yield cannot be evaluated.
    pub fn observed_cross_section(&self, free: &[f64], luminosity: f64) -> LikelihoodResult<f64> {
        if !luminosity.is_finite() || luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        Ok(self.acceptance_corrected_yield(free)? / luminosity)
    }

    /// Returns the fitted projected cross section from an absolute-rate term.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::AbsoluteRateUnavailable`] for shape-only
    /// terms, or [`LikelihoodError`] when luminosity or integral evaluation
    /// fails.
    pub fn fitted_cross_section(&self, free: &[f64], luminosity: f64) -> LikelihoodResult<f64> {
        if !self.has_absolute_rate {
            return Err(LikelihoodError::AbsoluteRateUnavailable(
                self.name.as_str().to_owned(),
            ));
        }
        if !luminosity.is_finite() || luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        Ok(self.generated_integral(free)? / luminosity)
    }

    /// Alias for [`Self::observed_cross_section`].
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when luminosity or integral evaluation fails.
    pub fn cross_section(&self, free: &[f64], luminosity: f64) -> LikelihoodResult<f64> {
        self.observed_cross_section(free, luminosity)
    }

    /// Returns per-event projected weights over generated Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters or integrals are invalid,
    /// generated data cannot be read, or runtime evaluation fails.
    pub fn weights(&self, free: &[f64], acceptance_corrected: bool) -> LikelihoodResult<Vec<f64>> {
        let scale = if acceptance_corrected {
            self.data_weight_sum
                / positive_integral("accepted MC", self.full_accepted_integral(free)?)?
        } else {
            1.0
        };
        let intensities = self.intensities(free)?;
        let mut output = Vec::with_capacity(intensities.len());
        let mut offset = 0;
        for batch in self
            .generated_mc_source
            .batches()
            .map_err(|e| LikelihoodError::Runtime(RuntimeError::Data(e.to_string())))?
        {
            let batch =
                batch.map_err(|e| LikelihoodError::Runtime(RuntimeError::Data(e.to_string())))?;
            output.extend(
                (0..batch.len())
                    .map(|row| batch.weights_at(row) * intensities[offset + row] * scale),
            );
            offset += batch.len();
        }
        Ok(output)
    }

    /// Returns projected intensities over generated Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid, generated data
    /// cannot be read, or runtime evaluation fails.
    pub fn intensities(&self, free: &[f64]) -> LikelihoodResult<Vec<f64>> {
        let global = self.projected_params.global_layout.values(free)?;
        let local = self.projected_params.project(&global)?;
        let mut output = Vec::new();
        for batch in self
            .generated_mc_source
            .batches()
            .map_err(|e| LikelihoodError::Runtime(RuntimeError::Data(e.to_string())))?
        {
            let batch =
                batch.map_err(|e| LikelihoodError::Runtime(RuntimeError::Data(e.to_string())))?;
            output.extend(
                self.projected_plan
                    .evaluate_batch(&local, &batch)?
                    .into_iter()
                    .map(|value| value.re),
            );
        }
        Ok(output)
    }

    fn projected_integral(
        &self,
        free: &[f64],
        dataset: &PreparedDataset,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        let global = self.projected_params.global_layout.values(free)?;
        let local = self.projected_params.project(&global)?;
        self.projected_plan
            .reduce(
                &self.execution,
                &local,
                dataset,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error(name, error))
    }
}

impl CrossSectionIntegrals {
    /// Returns retained prepared-dataset bytes used by these integrals.
    pub fn resident_bytes(&self) -> usize {
        self.full_accepted_mc
            .stats()
            .resident_bytes()
            .saturating_add(self.accepted_mc.stats().resident_bytes())
            .saturating_add(self.generated_mc.stats().resident_bytes())
    }

    /// Returns the source likelihood term name.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the prepared accepted Monte Carlo dataset.
    pub fn accepted_mc(&self) -> &PreparedDataset {
        &self.accepted_mc
    }

    /// Returns the prepared generated Monte Carlo dataset.
    pub fn generated_mc(&self) -> &PreparedDataset {
        &self.generated_mc
    }

    /// Returns the accepted Monte Carlo source dataset.
    pub fn accepted_mc_source(&self) -> &Dataset {
        &self.accepted_mc_source
    }

    /// Returns the generated Monte Carlo source dataset.
    pub fn generated_mc_source(&self) -> &Dataset {
        &self.generated_mc_source
    }

    /// Returns the observed dataset's total event weight.
    pub fn data_weight_sum(&self) -> f64 {
        self.data_weight_sum
    }

    /// Returns the intensity integral over accepted Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid, runtime
    /// evaluation fails, or an intensity is not positive.
    pub fn accepted_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.projection.global_layout.values(free)?;
        let local_params = self.projection.project(&params)?;
        self.weighted_intensity_sum(&local_params, &self.accepted_mc, "accepted MC")
    }

    /// Returns the intensity integral over generated Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid, runtime
    /// evaluation fails, or an intensity is not positive.
    pub fn generated_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.projection.global_layout.values(free)?;
        let local_params = self.projection.project(&params)?;
        self.weighted_intensity_sum(&local_params, &self.generated_mc, "generated MC")
    }

    /// Returns selected intensities over accepted Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters or dataset evaluation fail.
    pub fn accepted_intensities(&self, free: &[f64]) -> LikelihoodResult<Vec<f64>> {
        self.intensities(free, &self.accepted_mc_source)
    }

    /// Returns selected intensities over generated Monte Carlo.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters or dataset evaluation fail.
    pub fn generated_intensities(&self, free: &[f64]) -> LikelihoodResult<Vec<f64>> {
        self.intensities(free, &self.generated_mc_source)
    }

    /// Returns the accepted-to-generated integral ratio.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when either integral cannot be evaluated or
    /// is not positive.
    pub fn acceptance(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let generated = positive_integral("generated MC", self.generated_integral(free)?)?;
        let accepted = positive_integral("accepted MC", self.accepted_integral(free)?)?;
        Ok(accepted / generated)
    }

    /// Corrects an accepted yield for finite acceptance.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when an integral cannot be evaluated or the
    /// accepted integral is not positive.
    pub fn acceptance_corrected_yield(
        &self,
        free: &[f64],
        accepted_yield: f64,
    ) -> LikelihoodResult<f64> {
        let accepted = self.accepted_integral(free)?;
        if accepted <= 0.0 {
            return Err(LikelihoodError::NonPositiveAcceptedIntegral(accepted));
        }
        Ok(accepted_yield * self.generated_integral(free)? / accepted)
    }

    /// Returns the observed-yield-normalized cross section.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when `luminosity` is not positive or the
    /// acceptance-corrected yield cannot be evaluated.
    pub fn observed_cross_section(&self, free: &[f64], luminosity: f64) -> LikelihoodResult<f64> {
        if !luminosity.is_finite() || luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        let full_accepted = positive_integral("accepted MC", self.full_accepted_integral(free)?)?;
        Ok(self.data_weight_sum * self.generated_integral(free)? / full_accepted / luminosity)
    }

    /// Returns the fitted cross section from an absolute-rate term.
    ///
    /// For a tagged evaluator this uses the selected generated intensity
    /// directly, without rescaling it to the observed yield.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError::AbsoluteRateUnavailable`] for shape-only
    /// terms, or [`LikelihoodError`] when luminosity or integral evaluation
    /// fails.
    pub fn fitted_cross_section(&self, free: &[f64], luminosity: f64) -> LikelihoodResult<f64> {
        if !self.has_absolute_rate {
            return Err(LikelihoodError::AbsoluteRateUnavailable(
                self.name.as_str().to_owned(),
            ));
        }
        if !luminosity.is_finite() || luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        Ok(self.generated_integral(free)? / luminosity)
    }

    /// Alias for [`Self::observed_cross_section`].
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when luminosity or integral evaluation fails.
    pub fn cross_section(&self, free: &[f64], luminosity: f64) -> LikelihoodResult<f64> {
        self.observed_cross_section(free, luminosity)
    }

    /// Returns the full-model accepted Monte Carlo integral.
    ///
    /// This differs from [`Self::accepted_integral`] only when the evaluator
    /// was narrowed by tags.
    ///
    /// # Errors
    ///
    /// Returns [`LikelihoodError`] when parameters are invalid, runtime
    /// evaluation fails, or an intensity is not positive.
    pub fn full_accepted_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.full_projection.global_layout.values(free)?;
        let local_params = self.full_projection.project(&params)?;
        self.full_plan
            .reduce(
                &self.execution,
                &local_params,
                &self.full_accepted_mc,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error("accepted MC", error))
    }

    fn weighted_intensity_sum(
        &self,
        params: &ParamValues,
        dataset: &PreparedDataset,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        self.plan
            .reduce(
                &self.execution,
                params,
                dataset,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error(name, error))
    }

    fn intensities(&self, free: &[f64], dataset: &Dataset) -> LikelihoodResult<Vec<f64>> {
        let global = self.projection.global_layout.values(free)?;
        let local = self.projection.project(&global)?;
        let mut output = Vec::new();
        for batch in dataset
            .batches()
            .map_err(|error| LikelihoodError::Runtime(RuntimeError::Data(error.to_string())))?
        {
            let batch = batch
                .map_err(|error| LikelihoodError::Runtime(RuntimeError::Data(error.to_string())))?;
            output.extend(
                self.plan
                    .evaluate_batch(&local, &batch)?
                    .into_iter()
                    .map(|value| value.re),
            );
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct ParamProjection {
    global_layout: Arc<ParamLayout>,
    local_layout: Arc<ParamLayout>,
    global_ids: Vec<ParamId>,
    local_free_to_global_free: Vec<usize>,
}

impl ParamProjection {
    fn new(
        global_layout: Arc<ParamLayout>,
        local_layout: &ParamLayout,
        term: &str,
    ) -> LikelihoodResult<Self> {
        let global_ids = local_layout
            .specs()
            .iter()
            .map(|spec| {
                global_layout
                    .id(spec.name())
                    .ok_or_else(|| LikelihoodError::MissingParameter {
                        term: term.to_owned(),
                        parameter: spec.name().to_owned(),
                    })
            })
            .collect::<LikelihoodResult<_>>()?;
        let local_free_to_global_free = local_layout
            .free_params()
            .iter()
            .map(|local_id| {
                let name = local_layout.name(*local_id)?;
                let global_id =
                    global_layout
                        .id(name)
                        .ok_or_else(|| LikelihoodError::MissingParameter {
                            term: term.to_owned(),
                            parameter: name.to_owned(),
                        })?;
                global_layout
                    .free_id(global_id)?
                    .map(|id| id.index())
                    .ok_or(LikelihoodError::ParameterLayoutMismatch)
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        Ok(Self {
            global_layout,
            local_layout: Arc::new(local_layout.clone()),
            global_ids,
            local_free_to_global_free,
        })
    }

    fn project(&self, params: &ParamValues) -> LikelihoodResult<ParamValues> {
        check_params(&self.global_layout, params)?;
        let free = self
            .local_layout
            .free_params()
            .iter()
            .map(|local_id| params.get(self.global_ids[local_id.index()]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.local_layout.values(&free)?)
    }

    fn scatter_gradient(&self, local: &[f64], global: &mut [f64]) -> LikelihoodResult<()> {
        if local.len() != self.local_free_to_global_free.len() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: self.local_free_to_global_free.len(),
                actual: local.len(),
            });
        }
        if global.len() != self.global_layout.n_free() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: self.global_layout.n_free(),
                actual: global.len(),
            });
        }
        for (derivative, target) in local.iter().zip(&self.local_free_to_global_free) {
            global[*target] += derivative;
        }
        Ok(())
    }
}

fn check_params(layout: &ParamLayout, params: &ParamValues) -> LikelihoodResult<()> {
    if params.layout().specs() == layout.specs() {
        Ok(())
    } else {
        Err(LikelihoodError::ParameterLayoutMismatch)
    }
}

fn map_reduction_error(
    dataset: &'static str,
    error: laddu_runtime::RuntimeError,
) -> LikelihoodError {
    match error {
        laddu_runtime::RuntimeError::Reduction(
            laddu_compile::ReductionError::NonPositiveValue { value, .. },
        ) => LikelihoodError::NonPositiveIntensity { dataset, value },
        error => error.into(),
    }
}

fn positive_integral(dataset: &'static str, value: f64) -> LikelihoodResult<f64> {
    if value > 0.0 {
        Ok(value)
    } else {
        Err(LikelihoodError::NonPositiveIntensity { dataset, value })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    #[cfg(feature = "wgpu")]
    use laddu_compile::CompileOptions;
    use laddu_compile::CompiledModel;
    use laddu_data::{
        data::{CacheStorage, Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{
        Expr, complex, event_scalar, matrix, parameter, parameters::Parameter, solve, vector,
    };
    #[cfg(feature = "wgpu")]
    use laddu_expr::{dot, matvec};
    use laddu_runtime::{CpuOptions, Device, ExecutionOptions, JitPolicy, Precision, ThreadPolicy};
    #[cfg(feature = "wgpu")]
    use laddu_runtime::{GpuBackend, GpuOptions, MemoryBudget, MemoryPlan};

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

    fn weighted_dataset_batches(values: &[(f64, f64)], ends: &[usize]) -> Dataset {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let mut start = 0;
        let batches = ends
            .iter()
            .map(|&end| {
                let batch = EventBatch::from_events(
                    Arc::clone(&schema),
                    values[start..end]
                        .iter()
                        .map(|(x, weight)| OwnedEvent::weighted(vec![], vec![*x], *weight)),
                )
                .unwrap();
                start = end;
                batch
            })
            .collect::<Vec<_>>();
        assert_eq!(start, values.len());
        Dataset::from_batches(batches)
            .unwrap()
            .chunked(values.len())
            .unwrap()
    }

    fn single_term_likelihood(
        name: &str,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> Likelihood {
        Likelihood::new([NllTerm::new(name, model, data, accepted_mc).unwrap()]).unwrap()
    }

    fn single_term_likelihood_with_execution(
        name: &str,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
        execution: Execution,
    ) -> Likelihood {
        Likelihood::with_execution(
            [NllTerm::new(name, model, data, accepted_mc).unwrap()],
            &execution,
        )
        .unwrap()
    }

    fn cpu_execution(precision: Precision, threads: ThreadPolicy, jit: JitPolicy) -> Execution {
        Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions { threads, jit }),
            precision,
            ..ExecutionOptions::default()
        })
        .unwrap()
    }

    #[cfg(feature = "wgpu")]
    fn wgpu_execution(memory_budget: Option<usize>) -> Execution {
        Execution::local(ExecutionOptions {
            device: Device::Gpu(GpuOptions {
                backend: GpuBackend::Wgpu,
                ..GpuOptions::default()
            }),
            memory: MemoryPlan {
                host: MemoryBudget::Auto,
                device: memory_budget.map(|bytes| MemoryBudget::Bytes(bytes as u64)),
            },
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap()
    }

    fn assert_evaluation_close(
        actual: &LikelihoodEvaluation,
        expected: &LikelihoodEvaluation,
        epsilon: f64,
    ) {
        assert_relative_eq!(actual.value(), expected.value(), epsilon = epsilon);
        assert_eq!(actual.gradient().len(), expected.gradient().len());
        for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
            assert_relative_eq!(actual, expected, epsilon = epsilon);
        }
    }

    fn finite_difference_nll(
        likelihood: &Likelihood,
        params: &[f64],
        free_parameter: usize,
    ) -> f64 {
        let center = params[free_parameter];
        let h = 1.0e-6;
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        plus[free_parameter] = center + h;
        minus[free_parameter] = center - h;
        (likelihood.nll(&plus).unwrap() - likelihood.nll(&minus).unwrap()) / (2.0 * h)
    }

    #[test]
    fn nll_uses_data_and_accepted_mc_reductions() {
        let expr = event_scalar("x") * parameter!("scale", initial: 0.5);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_mc = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = single_term_likelihood("data", &model, &data, &accepted_mc);
        let params = likelihood.default_params();

        let expected = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), expected);
    }

    #[test]
    fn extended_nll_uses_expected_yield_and_has_an_analytic_gradient() {
        let expr = event_scalar("x") * parameter!("scale", initial: 0.5);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_mc = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood =
            Likelihood::new([
                ExtendedNllTerm::new("extended", &model, &data, &accepted_mc).unwrap(),
            ])
            .unwrap();
        let params = likelihood.default_params();
        let evaluation = likelihood.nll_with_gradient(&params).unwrap();
        let expected = 2.0 - 1.0_f64.ln() - 1.5_f64.ln();

        assert_relative_eq!(evaluation.value(), expected);
        assert_relative_eq!(
            evaluation.gradient()[0],
            finite_difference_nll(&likelihood, &params, 0),
            epsilon = 1.0e-8
        );
    }

    #[test]
    fn extended_nll_exposes_observed_and_fitted_cross_sections() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.25)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_mc = weighted_dataset(&[(4.0, 1.0)]);
        let generated_mc = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood =
            Likelihood::new([
                ExtendedNllTerm::new("extended", &model, &data, &accepted_mc).unwrap(),
            ])
            .unwrap();
        let params = likelihood.default_params();
        let integrals = likelihood
            .cross_section_integrals("extended", &generated_mc)
            .unwrap();

        assert_relative_eq!(
            integrals.observed_cross_section(&params, 10.0).unwrap(),
            0.3
        );
        assert_relative_eq!(integrals.fitted_cross_section(&params, 10.0).unwrap(), 0.15);
        assert_relative_eq!(
            integrals.cross_section(&params, 10.0).unwrap(),
            integrals.observed_cross_section(&params, 10.0).unwrap()
        );
        assert_relative_eq!(
            likelihood
                .intensity_datasets("extended")
                .unwrap()
                .0
                .sum_weights()
                .unwrap(),
            data.sum_weights().unwrap()
        );
    }

    #[test]
    fn fitted_cross_section_rejects_shape_only_nll() {
        let model = CompiledModel::from_expr(&event_scalar("x")).unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted_mc = weighted_dataset(&[(4.0, 1.0)]);
        let generated_mc = weighted_dataset(&[(6.0, 1.0)]);
        let likelihood = single_term_likelihood("shape", &model, &data, &accepted_mc);
        let integrals = likelihood
            .cross_section_integrals("shape", &generated_mc)
            .unwrap();

        assert!(matches!(
            integrals.fitted_cross_section(&[], 10.0),
            Err(LikelihoodError::AbsoluteRateUnavailable(name)) if name == "shape"
        ));
    }

    #[test]
    fn likelihood_accepts_free_slices_and_generates_free_parameters() {
        let scale = laddu_expr::Expr::from(Parameter::free("scale").with_initial((0.25, 0.75)));
        let offset = laddu_expr::Expr::from(Parameter::fixed("offset", 1.0));
        let model = CompiledModel::from_expr(&(event_scalar("x") * scale + offset)).unwrap();
        let data = weighted_dataset(&[(1.0, 1.0), (2.0, 1.0)]);
        let accepted = weighted_dataset(&[(1.5, 1.0), (2.5, 1.0)]);
        let likelihood = single_term_likelihood("slice", &model, &data, &accepted);

        assert_eq!(likelihood.default_params(), vec![0.5]);
        assert_eq!(likelihood.sample_initial(0), vec![0.5513138035955086]);
        assert_eq!(
            likelihood.params_with(|parameter| parameter.name().len() as f64),
            vec![5.0]
        );
        assert!(likelihood.nll(&[0.5f64]).unwrap().is_finite());
        assert!(matches!(
            likelihood.nll(&[]),
            Err(LikelihoodError::Params(ParamError::FreeLengthMismatch {
                expected: 1,
                actual: 0
            }))
        ));
    }

    #[test]
    fn nll_gradient_matches_finite_difference() {
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.7));
        let expr = (event_scalar("x") + scale).powi(2);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let data = weighted_dataset(&[(0.3, 1.0), (1.1, 2.0)]);
        let accepted_mc = weighted_dataset(&[(0.5, 1.5), (1.7, 0.8)]);
        let likelihood = single_term_likelihood("data", &model, &data, &accepted_mc);
        let params = likelihood.default_params();
        let evaluation = likelihood.nll_with_gradient(&params).unwrap();

        assert_relative_eq!(evaluation.value(), likelihood.nll(&params).unwrap());
        assert_relative_eq!(
            evaluation.gradient()[0],
            finite_difference_nll(&likelihood, &params, 0),
            epsilon = 1.0e-8
        );
    }

    #[test]
    fn full_fraction_stochastic_evaluation_matches_exact_likelihood() {
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.7));
        let model = CompiledModel::from_expr(&(event_scalar("x") + scale).powi(2)).unwrap();
        let data = weighted_dataset(&[(0.3, 1.0), (1.1, 2.0), (1.8, 0.5)]);
        let accepted_mc = weighted_dataset(&[(0.5, 1.5), (1.7, 0.8)]);
        let likelihood = single_term_likelihood("data", &model, &data, &accepted_mc);
        let params = likelihood.default_params();
        let exact = likelihood.nll_with_gradient(&params).unwrap();
        let stochastic = likelihood
            .stochastic_nll_with_gradient(&params, 1.0, 42)
            .unwrap();

        assert_evaluation_close(&stochastic, &exact, 1.0e-12);
        assert!(matches!(
            likelihood.stochastic_nll_with_gradient(&params, 0.0, 42),
            Err(LikelihoodError::InvalidBatchFraction(0.0))
        ));
    }

    #[test]
    fn likelihood_is_invariant_under_dataset_batching() {
        let x = event_scalar("x");
        let coupling = laddu_expr::Expr::from(parameter!("coupling", initial: 0.35));
        let matrix = matrix([
            [x.clone() + 2.0, complex(coupling.clone(), 0.15)],
            [complex(-0.2, coupling), 3.5.into()],
        ]);
        let amplitude = solve(matrix, vector([x.sin() + 1.0, complex(x.cos(), 0.5)])).component(1);
        let model = CompiledModel::from_expr(&(amplitude.norm_sqr() + 0.25)).unwrap();
        let values = [
            (0.15, 0.7),
            (0.35, 1.2),
            (0.65, 0.5),
            (0.95, 1.8),
            (1.25, 0.9),
            (1.55, 1.1),
            (1.85, 0.6),
        ];
        let one_batch = weighted_dataset_batches(&values, &[values.len()]);
        let two_batches = weighted_dataset_batches(&values, &[3, values.len()]);
        let uneven_batches = weighted_dataset_batches(&values, &[1, 2, 6, values.len()]);
        let streaming = weighted_dataset_batches(&values, &[2, 5, values.len()]).streaming();

        let reference = single_term_likelihood("reference", &model, &one_batch, &one_batch);
        let two = single_term_likelihood("two", &model, &two_batches, &two_batches);
        let uneven = single_term_likelihood("uneven", &model, &uneven_batches, &uneven_batches);
        let serial = single_term_likelihood_with_execution(
            "serial",
            &model,
            &streaming,
            &streaming,
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Serial,
                    ..CpuOptions::default()
                }),
                ..ExecutionOptions::default()
            })
            .unwrap(),
        );
        let fixed = single_term_likelihood_with_execution(
            "fixed",
            &model,
            &two_batches,
            &streaming,
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(2),
                    ..CpuOptions::default()
                }),
                ..ExecutionOptions::default()
            })
            .unwrap(),
        );
        let expected = reference
            .nll_with_gradient(&reference.default_params())
            .unwrap();

        for actual in [
            two.nll_with_gradient(&two.default_params()).unwrap(),
            uneven.nll_with_gradient(&uneven.default_params()).unwrap(),
            serial.nll_with_gradient(&serial.default_params()).unwrap(),
            fixed.nll_with_gradient(&fixed.default_params()).unwrap(),
        ] {
            assert_relative_eq!(actual.value(), expected.value(), epsilon = 1.0e-12);
            assert_eq!(actual.gradient().len(), expected.gradient().len());
            for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
                assert_relative_eq!(actual, expected, epsilon = 1.0e-11);
            }
        }
        assert_eq!(
            reference.terms()[0]
                .as_intensity()
                .unwrap()
                .data()
                .unwrap()
                .stats()
                .storage(),
            CacheStorage::Resident
        );
        let streaming_stats = serial.terms()[0]
            .as_intensity()
            .unwrap()
            .data()
            .unwrap()
            .stats();
        assert_eq!(streaming_stats.storage(), CacheStorage::Streaming);
        assert_eq!(streaming_stats.resident_bytes(), 0);
        assert_eq!(streaming_stats.local_batches(), 1);
    }

    #[test]
    fn f32_cpu_likelihood_matches_across_resident_streaming_and_batches() {
        let x = event_scalar("x");
        let coupling = laddu_expr::Expr::from(parameter!("coupling", initial: 0.35));
        let matrix = matrix([
            [x.clone() + 2.0, complex(coupling.clone(), 0.15)],
            [complex(-0.2, coupling), 3.5.into()],
        ]);
        let amplitude = solve(matrix, vector([x.sin() + 1.0, complex(x.cos(), 0.5)])).component(1);
        let model = CompiledModel::from_expr(&(amplitude.norm_sqr() + 0.25)).unwrap();
        let values = [
            (0.15, 0.7),
            (0.35, 1.2),
            (0.65, 0.5),
            (0.95, 1.8),
            (1.25, 0.9),
            (1.55, 1.1),
            (1.85, 0.6),
        ];
        let one_batch = weighted_dataset_batches(&values, &[values.len()]);
        let two_batches = weighted_dataset_batches(&values, &[3, values.len()]);
        let streaming = weighted_dataset_batches(&values, &[2, 5, values.len()]).streaming();
        let interpreter = cpu_execution(Precision::F32, ThreadPolicy::Serial, JitPolicy::Disabled);
        let threaded = cpu_execution(Precision::F32, ThreadPolicy::Fixed(2), JitPolicy::Disabled);

        let reference = single_term_likelihood_with_execution(
            "reference",
            &model,
            &one_batch,
            &one_batch,
            interpreter.clone(),
        );
        let expected = reference
            .nll_with_gradient(&reference.default_params())
            .unwrap();

        let cases = [
            single_term_likelihood_with_execution(
                "resident",
                &model,
                &two_batches,
                &two_batches,
                interpreter.clone(),
            ),
            single_term_likelihood_with_execution(
                "streaming",
                &model,
                &streaming,
                &streaming,
                interpreter.clone(),
            ),
            single_term_likelihood_with_execution(
                "mixed",
                &model,
                &two_batches,
                &streaming,
                threaded,
            ),
        ];

        for likelihood in cases {
            let actual = likelihood
                .nll_with_gradient(&likelihood.default_params())
                .unwrap();
            assert_evaluation_close(&actual, &expected, 5.0e-5);
        }

        #[cfg(feature = "jit")]
        {
            let jit = cpu_execution(Precision::F32, ThreadPolicy::Fixed(2), JitPolicy::Enabled);
            for likelihood in [
                single_term_likelihood_with_execution(
                    "jit-resident",
                    &model,
                    &two_batches,
                    &two_batches,
                    jit.clone(),
                ),
                single_term_likelihood_with_execution(
                    "jit-streaming",
                    &model,
                    &streaming,
                    &streaming,
                    jit.clone(),
                ),
                single_term_likelihood_with_execution(
                    "jit-mixed",
                    &model,
                    &two_batches,
                    &streaming,
                    jit,
                ),
            ] {
                let actual = likelihood
                    .nll_with_gradient(&likelihood.default_params())
                    .unwrap();
                assert_evaluation_close(&actual, &expected, 5.0e-5);
            }
        }

        let streaming_likelihood = single_term_likelihood_with_execution(
            "streaming-stats",
            &model,
            &streaming,
            &streaming,
            interpreter,
        );
        let streaming_stats = streaming_likelihood.terms()[0]
            .as_intensity()
            .unwrap()
            .data()
            .unwrap()
            .stats();
        assert_eq!(streaming_stats.storage(), CacheStorage::Streaming);
        assert_eq!(streaming_stats.resident_bytes(), 0);
        assert_eq!(streaming_stats.local_batches(), 1);
    }

    #[test]
    fn likelihood_nll_uses_configured_f32_scalar_execution() {
        let dataset = weighted_dataset(&[(1.0, 1.0), (2.0, 1.0)]);
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.0));
        let model = CompiledModel::from_expr(&(event_scalar("x") + scale)).unwrap();
        let likelihood = single_term_likelihood_with_execution(
            "f32",
            &model,
            &dataset,
            &dataset,
            cpu_execution(Precision::F32, ThreadPolicy::Serial, JitPolicy::Auto),
        );

        let expected = 2.0 * 5.0_f64.ln() - 2.0_f32.ln() as f64 - 3.0_f32.ln() as f64;
        assert_eq!(
            likelihood.nll(&likelihood.default_params()).unwrap(),
            expected
        );
    }

    #[cfg(feature = "mpi")]
    #[mpi_test::mpi_test(np = [2, 3, 4])]
    fn mpi_likelihood_matches_local_reference_without_multiplying_penalties() {
        use mpi::traits::Communicator;

        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let values = [(0.4, 1.5), (1.2, 0.75)];
        let resident = weighted_dataset_batches(&values, &[1, values.len()]);
        let streaming = weighted_dataset_batches(&values, &[1, values.len()]).streaming();
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.6));
        let model = CompiledModel::from_expr(&(event_scalar("x") + scale).powi(2)).unwrap();

        let reference = Likelihood::new_boxed([
            NllTerm::new("data", &model, &resident, &streaming)
                .unwrap()
                .boxed(),
            RidgePenalty::new("ridge", ["scale"], 0.3).unwrap().boxed(),
        ])
        .unwrap();
        let distributed = Likelihood::with_execution_boxed(
            [
                NllTerm::new("data", &model, &resident, &streaming)
                    .unwrap()
                    .boxed(),
                RidgePenalty::new("ridge", ["scale"], 0.3).unwrap().boxed(),
            ],
            &Execution::distributed(
                ExecutionOptions {
                    device: Device::Cpu(CpuOptions {
                        threads: ThreadPolicy::Serial,
                        ..CpuOptions::default()
                    }),
                    partitioning: laddu_data::io::Partitioning::Contiguous,
                    ..ExecutionOptions::default()
                },
                &world,
            )
            .unwrap(),
        )
        .unwrap();

        let expected = reference
            .nll_with_gradient(&reference.default_params())
            .unwrap();
        let actual = distributed
            .nll_with_gradient(&distributed.default_params())
            .unwrap();
        assert_relative_eq!(actual.value(), expected.value(), epsilon = 1.0e-12);
        assert_relative_eq!(
            actual.gradient()[0],
            expected.gradient()[0],
            epsilon = 1.0e-11
        );
        let expected_term = reference.terms()[0].as_intensity().unwrap();
        let actual_term = distributed.terms()[0].as_intensity().unwrap();
        assert_relative_eq!(
            actual_term
                .accepted_normalization(&distributed.default_params())
                .unwrap(),
            expected_term
                .accepted_normalization(&reference.default_params())
                .unwrap(),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            actual_term
                .data_log_intensity_sum(&distributed.default_params())
                .unwrap(),
            expected_term
                .data_log_intensity_sum(&reference.default_params())
                .unwrap(),
            epsilon = 1.0e-12
        );

        let stats = distributed.terms()[0]
            .as_intensity()
            .unwrap()
            .data()
            .unwrap()
            .stats();
        assert_eq!(stats.global_events(), values.len());
        assert!(stats.local_events() <= 1 || world.size() <= 2);
    }

    #[cfg(all(feature = "mpi", feature = "wgpu"))]
    #[mpi_test::mpi_test(np = [2, 3])]
    fn mpi_wgpu_likelihood_matches_local_wgpu_reference_across_storage_modes() {
        use mpi::traits::Communicator;

        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let values = [
            (0.15, 0.7),
            (0.35, 1.2),
            (0.65, 0.5),
            (0.95, 1.8),
            (1.25, 0.9),
            (1.55, 1.1),
            (1.85, 0.6),
        ];
        let accepted_values = [
            (0.25, 0.5),
            (0.55, 1.0),
            (0.85, 1.5),
            (1.15, 0.75),
            (1.45, 1.25),
        ];
        let data = weighted_dataset_batches(&values, &[2, 5, values.len()]);
        let accepted = weighted_dataset_batches(&accepted_values, &[1, 3, accepted_values.len()]);
        let streaming_data = data.clone().streaming();
        let streaming_accepted = accepted.clone().streaming();
        let x = event_scalar("x");
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.5));
        let offset = laddu_expr::Expr::from(parameter!("offset", initial: 1.25));
        let model = CompiledModel::from_expr(&((x * scale + offset + 2.0).powi(2) + 0.1)).unwrap();

        let local = single_term_likelihood_with_execution(
            "local",
            &model,
            &data,
            &accepted,
            wgpu_execution(Some(256)),
        );
        let expected = local.nll_with_gradient(&local.default_params()).unwrap();

        let make_distributed = |name, data: &Dataset, accepted: &Dataset| {
            single_term_likelihood_with_execution(
                name,
                &model,
                data,
                accepted,
                Execution::distributed(
                    ExecutionOptions {
                        device: Device::Gpu(GpuOptions {
                            backend: GpuBackend::Wgpu,
                            ..GpuOptions::default()
                        }),
                        memory: MemoryPlan::host_device(
                            MemoryBudget::Auto,
                            MemoryBudget::Bytes(256),
                        ),
                        precision: Precision::F32,
                        partitioning: laddu_data::io::Partitioning::Contiguous,
                        ..ExecutionOptions::default()
                    },
                    &world,
                )
                .unwrap(),
            )
        };
        let resident = make_distributed("resident", &data, &accepted);
        let streaming = make_distributed("streaming", &streaming_data, &streaming_accepted);
        let mixed = make_distributed("mixed", &data, &streaming_accepted);

        for likelihood in [&resident, &streaming, &mixed] {
            let actual = likelihood
                .nll_with_gradient(&likelihood.default_params())
                .unwrap();
            assert_evaluation_close(&actual, &expected, 5.0e-4);
        }

        let stats = resident.terms()[0]
            .as_intensity()
            .unwrap()
            .data()
            .unwrap()
            .stats();
        assert_eq!(stats.global_events(), values.len());
        assert!(stats.local_events() <= values.len().div_ceil(world.size() as usize));
        assert_eq!(stats.storage(), CacheStorage::Resident);

        let streaming_stats = streaming.terms()[0]
            .as_intensity()
            .unwrap()
            .data()
            .unwrap()
            .stats();
        assert_eq!(streaming_stats.global_events(), values.len());
        assert_eq!(streaming_stats.storage(), CacheStorage::Streaming);
        assert_eq!(streaming_stats.resident_bytes(), 0);
    }

    #[cfg(feature = "mpi")]
    #[mpi_test::mpi_test(np = [2, 3])]
    fn mpi_likelihood_propagates_a_rank_local_error_without_deadlocking() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let data = weighted_dataset_batches(&[(-1.0, 1.0), (2.0, 1.0)], &[2]);
        let accepted_mc = weighted_dataset_batches(&[(1.0, 1.0), (2.0, 1.0)], &[2]);
        let model = CompiledModel::from_expr(&event_scalar("x")).unwrap();
        let likelihood = Likelihood::with_execution(
            [NllTerm::new("data", &model, &data, &accepted_mc).unwrap()],
            &Execution::distributed(ExecutionOptions::default(), &world).unwrap(),
        )
        .unwrap();

        assert!(likelihood.nll(&likelihood.default_params()).is_err());
    }

    #[test]
    fn shared_parameters_are_merged_across_independent_models() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let model_b =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data_a = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_a = weighted_dataset(&[(4.0, 1.0)]);
        let data_b = weighted_dataset(&[(5.0, 2.0)]);
        let accepted_b = weighted_dataset(&[(6.0, 3.0)]);
        let likelihood = Likelihood::new([
            NllTerm::new("KsKs", &model_a, &data_a, &accepted_a).unwrap(),
            NllTerm::new("eta_pi", &model_b, &data_b, &accepted_b).unwrap(),
        ])
        .unwrap();

        assert_eq!(likelihood.params().len(), 1);
        assert_eq!(likelihood.params().specs()[0].name(), "scale");

        let params = likelihood.default_params();
        let term_a = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        let term_b = 2.0 * 9.0_f64.ln() - 2.0 * 2.5_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), term_a + term_b);
    }

    #[test]
    fn changing_shared_parameter_affects_all_terms() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let model_b =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = Likelihood::new([
            NllTerm::new("a", &model_a, &data, &accepted).unwrap(),
            NllTerm::new("b", &model_b, &data, &accepted).unwrap(),
        ])
        .unwrap();
        let mut params = likelihood.default_params();
        params[0] = 1.0;

        let expected_term = 1.0 * 4.0_f64.ln() - 2.0_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), 2.0 * expected_term);
    }

    #[test]
    fn shared_and_channel_specific_gradients_scatter_into_global_layout() {
        let shared = laddu_expr::Expr::from(parameter!("shared", initial: 0.4));
        let model_a = CompiledModel::from_expr(
            &(event_scalar("x")
                + shared.clone()
                + laddu_expr::Expr::from(parameter!("only_a", initial: 0.2)))
            .powi(2),
        )
        .unwrap();
        let model_b = CompiledModel::from_expr(
            &(event_scalar("x")
                + shared
                + laddu_expr::Expr::from(parameter!("only_b", initial: -0.1)))
            .powi(2),
        )
        .unwrap();
        let data = weighted_dataset(&[(0.5, 1.0), (1.2, 0.7)]);
        let accepted = weighted_dataset(&[(0.8, 1.3), (1.5, 0.9)]);
        let likelihood = Likelihood::new([
            NllTerm::new("a", &model_a, &data, &accepted).unwrap(),
            NllTerm::new("b", &model_b, &data, &accepted).unwrap(),
        ])
        .unwrap();
        let params = likelihood.default_params();
        let evaluation = likelihood.nll_with_gradient(&params).unwrap();

        for (parameter, derivative) in evaluation.gradient().iter().enumerate() {
            assert_relative_eq!(
                *derivative,
                finite_difference_nll(&likelihood, &params, parameter),
                epsilon = 1.0e-8
            );
        }
    }

    #[test]
    fn incompatible_shared_parameter_specs_are_rejected() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let model_b =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 1.0)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let err = Likelihood::new([
            NllTerm::new("a", &model_a, &data, &accepted).unwrap(),
            NllTerm::new("b", &model_b, &data, &accepted).unwrap(),
        ])
        .unwrap_err();

        assert!(matches!(
            err,
            LikelihoodError::Params(ParamError::ParameterConflict { ref name, .. })
                if name == "scale"
        ));
    }

    #[test]
    fn unique_channel_parameters_remain_separate() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale_ksks", initial: 0.5)))
                .unwrap();
        let model_b = CompiledModel::from_expr(
            &(event_scalar("x") * parameter!("scale_eta_pi", initial: 0.5)),
        )
        .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = Likelihood::new([
            NllTerm::new("KsKs", &model_a, &data, &accepted).unwrap(),
            NllTerm::new("eta_pi", &model_b, &data, &accepted).unwrap(),
        ])
        .unwrap();

        assert_eq!(likelihood.params().len(), 2);
        assert!(likelihood.params().id("scale_ksks").is_some());
        assert!(likelihood.params().id("scale_eta_pi").is_some());
    }

    #[test]
    fn ridge_and_lasso_terms_add_penalties() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = Likelihood::new_boxed([
            NllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            RidgePenalty::new("ridge", ["scale"], 2.0).unwrap().boxed(),
            LassoPenalty::new("lasso", ["scale"], 3.0).unwrap().boxed(),
        ])
        .unwrap();
        let params = likelihood.default_params();

        let nll = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        let penalty = 2.0 * 0.5_f64.powi(2) + 3.0 * 0.5_f64.abs();
        let result = likelihood.nll_with_gradient(&params).unwrap();
        assert_relative_eq!(result.value(), nll + penalty);
        assert_relative_eq!(result.gradient()[0], 5.0);
    }

    #[test]
    fn penalty_terms_reject_missing_parameters() {
        let err =
            Likelihood::new([RidgePenalty::new("ridge", ["missing"], 1.0).unwrap()]).unwrap_err();

        assert!(matches!(
            err,
            LikelihoodError::MissingParameter { ref term, ref parameter }
                if term == "ridge" && parameter == "missing"
        ));
    }

    #[derive(Debug)]
    struct ConstantTerm {
        name: String,
        value: f64,
    }

    #[derive(Debug)]
    struct BoundedQuadraticTerm {
        parameter: Parameter,
        id: Option<ParamId>,
    }

    impl LikelihoodTerm for BoundedQuadraticTerm {
        fn name(&self) -> &str {
            "bounded-quadratic"
        }

        fn register_params(&self, registry: &mut ParamRegistry) -> LikelihoodResult<()> {
            registry.register(self.parameter.clone())?;
            Ok(())
        }

        fn resolve(
            &mut self,
            global_params: Arc<ParamLayout>,
            _execution: &Execution,
        ) -> LikelihoodResult<()> {
            self.id = global_params.id(self.parameter.name());
            Ok(())
        }

        fn nll(&self, params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
            let value = params.get(self.id.ok_or(LikelihoodError::ParameterLayoutMismatch)?)?;
            Ok((value - 2.0).powi(2))
        }
    }

    impl LikelihoodTerm for ConstantTerm {
        fn name(&self) -> &str {
            &self.name
        }

        fn resolve(
            &mut self,
            _global_params: Arc<ParamLayout>,
            _execution: &Execution,
        ) -> LikelihoodResult<()> {
            Ok(())
        }

        fn nll(&self, _params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
            Ok(self.value)
        }
    }

    #[test]
    fn custom_likelihood_term_can_be_user_defined() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = Likelihood::new_boxed([
            NllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            ConstantTerm {
                name: "constant".into(),
                value: 12.5,
            }
            .boxed(),
        ])
        .unwrap();
        let params = likelihood.default_params();

        let expected = 1.0 * 2.0_f64.ln() - 1.0_f64.ln() + 12.5;
        assert_relative_eq!(likelihood.nll(&params).unwrap(), expected);
    }

    #[test]
    fn custom_term_gradient_uses_bounded_finite_difference_fallback() {
        let likelihood = Likelihood::new([BoundedQuadraticTerm {
            parameter: Parameter::free("x")
                .with_initial(0.0)
                .with_bounds(Some(0.0), None),
            id: None,
        }])
        .unwrap();
        let params = likelihood.default_params();
        let evaluation = likelihood.nll_with_gradient(&params).unwrap();

        assert_relative_eq!(evaluation.value(), 4.0);
        assert_relative_eq!(evaluation.gradient()[0], -4.0, epsilon = 1.0e-5);
    }

    #[test]
    fn cross_section_integrals_use_named_intensity_term_and_global_params() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 2.0)))
                .unwrap();
        let data = weighted_dataset(&[(9.0, 4.0)]);
        let accepted_mc = weighted_dataset(&[(1.0, 2.0), (2.0, 3.0)]);
        let generated_mc = weighted_dataset(&[(4.0, 5.0), (5.0, 7.0)]);
        let likelihood = single_term_likelihood("KsKs", &model, &data, &accepted_mc);
        let params = likelihood.default_params();
        let integrals = likelihood
            .cross_section_integrals("KsKs", &generated_mc)
            .unwrap();

        let accepted = 2.0 * 2.0 + 3.0 * 4.0;
        let generated = 5.0 * 8.0 + 7.0 * 10.0;
        assert_eq!(integrals.name(), "KsKs");
        assert_relative_eq!(integrals.accepted_integral(&params).unwrap(), accepted);
        assert_relative_eq!(integrals.generated_integral(&params).unwrap(), generated);
        assert_relative_eq!(integrals.acceptance(&params).unwrap(), accepted / generated);
        assert_relative_eq!(
            integrals.acceptance_corrected_yield(&params, 20.0).unwrap(),
            20.0 * generated / accepted
        );
        assert_relative_eq!(
            integrals.cross_section(&params, 5.0).unwrap(),
            data.sum_weights().unwrap() * generated / accepted / 5.0
        );
        assert_eq!(integrals.accepted_intensities(&params).unwrap().len(), 2);
        assert_eq!(integrals.generated_intensities(&params).unwrap().len(), 2);
    }

    #[test]
    fn bootstrap_rebuilds_likelihood_with_deterministic_poisson_data_weights() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 2.0)))
                .unwrap();
        let data = weighted_dataset(&[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
        let accepted = weighted_dataset(&[(1.0, 1.0)]);
        let likelihood = single_term_likelihood("signal", &model, &data, &accepted);
        let first = likelihood.bootstrap(42).unwrap();
        let second = likelihood.bootstrap(42).unwrap();
        let first_sum = first
            .intensity_datasets("signal")
            .unwrap()
            .0
            .sum_weights()
            .unwrap();
        let second_sum = second
            .intensity_datasets("signal")
            .unwrap()
            .0
            .sum_weights()
            .unwrap();

        assert_eq!(first_sum, second_sum);
        assert_eq!(first.params().n_free(), likelihood.params().n_free());
    }

    #[test]
    fn coherent_quadratic_normalization_matches_general_event_reduction() {
        let coefficient = complex(
            parameter!("coefficient_re", initial: 0.7),
            parameter!("coefficient_im", initial: -0.2),
        );
        let basis = complex(event_scalar("x"), 0.5);
        let model = CompiledModel::from_expr(&(coefficient * basis).norm_sqr()).unwrap();
        let sample = weighted_dataset(&[(0.5, 1.0), (1.5, 2.0), (2.5, 0.75)]);
        let likelihood = single_term_likelihood("quadratic", &model, &sample, &sample);
        let term = likelihood.terms()[0].as_intensity().unwrap();
        let free = vec![0.4, -0.6];
        let global = term.global_values(&free).unwrap();
        let local = term.local_values(&global).unwrap();

        let optimized = term
            .normalization_with_gradient(&local, likelihood.execution())
            .unwrap();
        let general = term
            .plan()
            .unwrap()
            .reduce_with_gradient(
                likelihood.execution(),
                &local,
                term.accepted_mc().unwrap(),
                ReductionPlan::weighted_positive_real(),
            )
            .unwrap()
            .into_parts();
        assert!((optimized.0 - general.0).abs() < 1.0e-12);
        for (optimized, general) in optimized.1.iter().zip(general.1) {
            assert!((*optimized - general).abs() < 1.0e-12);
        }
        assert!(
            likelihood
                .diagnostics()
                .datasets()
                .iter()
                .any(DatasetDiagnostics::uses_quadratic_normalization)
        );
    }

    #[test]
    fn cross_section_integrals_reject_non_intensity_terms() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(5.0, 1.0)]);
        let likelihood = Likelihood::new_boxed([
            NllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            RidgePenalty::new("ridge", ["scale"], 1.0).unwrap().boxed(),
        ])
        .unwrap();
        let err = likelihood
            .cross_section_integrals("ridge", &generated)
            .unwrap_err();

        assert!(matches!(err, LikelihoodError::NotIntensityTerm(ref name) if name == "ridge"));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_scalar_likelihood_matches_cpu_across_storage_modes() {
        let x = event_scalar("x");
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.5));
        let offset = laddu_expr::Expr::from(parameter!("offset", initial: 1.25));
        let model = CompiledModel::from_expr(&(x * scale + offset + 2.0)).unwrap();
        let data = weighted_dataset_batches(
            &(0..70)
                .map(|index| (index as f64 * 0.01, 1.0 + index as f64 * 0.001))
                .collect::<Vec<_>>(),
            &[31, 70],
        );
        let accepted = weighted_dataset_batches(&[(0.25, 0.5), (0.75, 1.5), (1.25, 2.0)], &[1, 3]);
        let streaming_data = data.clone().streaming();
        let streaming_accepted = accepted.clone().streaming();
        let cpu = single_term_likelihood_with_execution(
            "scalar",
            &model,
            &data,
            &accepted,
            cpu_execution(Precision::F32, ThreadPolicy::Auto, JitPolicy::Disabled),
        );
        let params = cpu.default_params();
        let expected = cpu.nll_with_gradient(&params).unwrap();
        let gpu = single_term_likelihood_with_execution(
            "resident",
            &model,
            &data,
            &accepted,
            wgpu_execution(Some(256)),
        );
        let streaming_gpu = single_term_likelihood_with_execution(
            "streaming",
            &model,
            &streaming_data,
            &streaming_accepted,
            wgpu_execution(Some(256)),
        );
        let mixed_gpu = single_term_likelihood_with_execution(
            "mixed",
            &model,
            &data,
            &streaming_accepted,
            wgpu_execution(Some(256)),
        );
        let first = gpu.nll_with_gradient(&params).unwrap();
        let second = gpu.nll_with_gradient(&params).unwrap();

        assert_evaluation_close(&first, &expected, 5.0e-4);
        assert_eq!(second, first);
        let streaming_gradient = streaming_gpu.nll_with_gradient(&params).unwrap();
        let mixed_gradient = mixed_gpu.nll_with_gradient(&params).unwrap();
        assert_evaluation_close(&streaming_gradient, &expected, 5.0e-4);
        assert_evaluation_close(&mixed_gradient, &expected, 5.0e-4);
        assert_evaluation_close(&streaming_gradient, &first, 2.0e-5);
        assert_evaluation_close(&mixed_gradient, &first, 2.0e-5);

        let streaming_term = streaming_gpu.terms()[0].as_intensity().unwrap();
        let streaming_data_stats = streaming_term.data().unwrap().stats();
        let streaming_accepted_stats = streaming_term.accepted_mc().unwrap().stats();
        assert_eq!(streaming_data_stats.storage(), CacheStorage::Streaming);
        assert_eq!(streaming_data_stats.resident_bytes(), 0);
        assert_eq!(streaming_data_stats.local_batches(), 2);
        assert_eq!(streaming_accepted_stats.storage(), CacheStorage::Streaming);
        assert_eq!(streaming_accepted_stats.resident_bytes(), 0);
        assert_eq!(streaming_accepted_stats.local_batches(), 2);

        let cpu_f64 = single_term_likelihood_with_execution(
            "scalar",
            &model,
            &data,
            &accepted,
            cpu_execution(Precision::F64, ThreadPolicy::Auto, JitPolicy::Disabled),
        );
        let expected_gradient = cpu_f64
            .nll_with_gradient(&cpu_f64.default_params())
            .unwrap();
        assert_evaluation_close(&first, &expected_gradient, 5.0e-4);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires a WGPU-compatible hardware adapter"]
    fn wgpu_aggregate_likelihood_matches_cpu() {
        let expression = dot(
            matvec(
                matrix([[event_scalar("x"), event_scalar("x") + 1.0]]),
                vector([
                    parameter!("a", initial: 0.5),
                    parameter!("b", initial: 1.25),
                ]),
            ),
            vector([1.0]),
        )
        .norm_sqr();
        let model = CompiledModel::from_expr_with_options(
            &expression,
            &CompileOptions::without_optimizations(),
        )
        .unwrap();
        let data = weighted_dataset(&[(0.25, 0.5), (0.75, 1.5), (1.25, 2.0)]);
        let cpu = single_term_likelihood_with_execution(
            "aggregate",
            &model,
            &data,
            &data,
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions::default()),
                precision: Precision::F64,
                ..ExecutionOptions::default()
            })
            .unwrap(),
        );
        let gpu = single_term_likelihood_with_execution(
            "aggregate",
            &model,
            &data,
            &data,
            Execution::local(ExecutionOptions {
                device: Device::Gpu(GpuOptions {
                    backend: GpuBackend::Wgpu,
                    ..GpuOptions::default()
                }),
                memory: MemoryPlan::host_device(MemoryBudget::Auto, MemoryBudget::Bytes(256)),
                precision: Precision::F32,
                ..ExecutionOptions::default()
            })
            .unwrap(),
        );
        let params = cpu.default_params();

        assert_relative_eq!(
            gpu.nll(&params).unwrap(),
            cpu.nll(&params).unwrap(),
            epsilon = 2.0e-5
        );
    }

    #[test]
    fn tagged_projection_produces_partial_weights_and_cross_sections() {
        let x = event_scalar("x");
        let selected = (Expr::from(parameter!("a", initial: 2.0)) * x.clone()).tagged("selected");
        let removed = Expr::from(parameter!("b", initial: 1.0)).tagged("removed");
        let model = CompiledModel::from_expr(&(selected + removed).norm_sqr()).unwrap();
        let data = weighted_dataset(&[(1.0, 3.0)]);
        let accepted = weighted_dataset(&[(1.0, 1.0)]);
        let generated = weighted_dataset(&[(2.0, 1.0)]);
        let likelihood =
            Likelihood::new([NllTerm::new("waves", &model, &data, &accepted).unwrap()]).unwrap();
        let params = likelihood.default_params();
        let projection = likelihood
            .projection("waves", &generated, ["selected"])
            .unwrap();
        let integrals = likelihood
            .cross_section_integrals_with_tags("waves", &generated, ["selected"])
            .unwrap();

        assert_relative_eq!(projection.full_accepted_integral(&params).unwrap(), 9.0);
        assert_relative_eq!(projection.generated_integral(&params).unwrap(), 16.0);
        assert_relative_eq!(projection.acceptance(&params).unwrap(), 0.25);
        assert_relative_eq!(projection.intensities(&params).unwrap()[0], 16.0);
        assert_relative_eq!(
            projection.acceptance_corrected_yield(&params).unwrap(),
            16.0 / 3.0
        );
        assert_relative_eq!(projection.weights(&params, false).unwrap()[0], 16.0);
        assert_relative_eq!(projection.weights(&params, true).unwrap()[0], 16.0 / 3.0);
        assert_relative_eq!(integrals.accepted_integral(&params).unwrap(), 4.0);
        assert_relative_eq!(integrals.generated_integral(&params).unwrap(), 16.0);
        assert_relative_eq!(integrals.acceptance(&params).unwrap(), 0.25);
        assert_relative_eq!(integrals.full_accepted_integral(&params).unwrap(), 9.0);
        assert_relative_eq!(
            integrals.acceptance_corrected_yield(&params, 12.0).unwrap(),
            48.0
        );
        assert_relative_eq!(integrals.cross_section(&params, 2.0).unwrap(), 8.0 / 3.0);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires a WGPU-compatible hardware adapter"]
    fn explicit_wgpu_solve_likelihood_value_and_gradient_match_cpu() {
        let x = event_scalar("x");
        let amplitude = laddu_amplitudes::f_vector(
            x.clone() + 4.0,
            vector([1.0]),
            matrix([[x.clone() + 0.5, 0.25.into()], [0.1.into(), x + 1.0]]),
            vector([Expr::from(parameter!("scale", initial: 1.25)), 2.0.into()]),
            matrix([
                [complex(0.0, -0.2), 0.0.into()],
                [0.0.into(), complex(0.0, -0.1)],
            ]),
        )
        .unwrap()
        .component(0)
        .norm_sqr();
        let model = CompiledModel::from_expr(&amplitude).unwrap();
        let data = weighted_dataset(&[(0.5, 1.0), (1.0, 0.75), (1.5, 1.25)]);
        let accepted = weighted_dataset(&[(0.25, 0.5), (0.75, 1.0), (1.25, 1.5)]);
        let make = |device, precision| {
            single_term_likelihood_with_execution(
                "solve",
                &model,
                &data,
                &accepted,
                Execution::local(ExecutionOptions {
                    device,
                    precision,
                    ..ExecutionOptions::default()
                })
                .unwrap(),
            )
        };
        let cpu = make(Device::Cpu(CpuOptions::default()), Precision::F64);
        let gpu = make(
            Device::Gpu(GpuOptions {
                backend: GpuBackend::Wgpu,
                ..GpuOptions::default()
            }),
            Precision::F32,
        );
        let params = cpu.default_params();
        let expected = cpu.nll_with_gradient(&params).unwrap();
        let actual = gpu.nll_with_gradient(&params).unwrap();
        assert_relative_eq!(actual.value(), expected.value(), epsilon = 2.0e-4);
        assert_relative_eq!(
            actual.gradient()[0],
            expected.gradient()[0],
            epsilon = 5.0e-4
        );
    }
}
