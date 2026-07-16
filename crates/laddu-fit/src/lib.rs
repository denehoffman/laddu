//! ganesh-backed minimization and sampling adapters for laddu objectives.
//!
//! [`FitProblem`] implements ganesh's cost, gradient, and log-density traits
//! for both `f32` and `f64`. The adapter leaves algorithm choice and callbacks
//! fully exposed while centralizing laddu parameter names and transforms.

pub use ganesh;

use std::{
    marker::PhantomData,
    ops::ControlFlow,
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
};

use ganesh::algorithms::{
    gradient::{
        Adam, AdamConfig, ConjugateGradient, ConjugateGradientConfig, GradientStatus, LBFGSB,
        LBFGSBConfig, TrustRegion, TrustRegionConfig,
    },
    gradient_free::{
        CMAES, CMAESConfig, DifferentialEvolution, DifferentialEvolutionConfig, GradientFreeStatus,
        NelderMead, NelderMeadConfig, SimulatedAnnealing, SimulatedAnnealingConfig,
    },
    mcmc::{AIES, AIESConfig, AIESInit, ESS, ESSConfig, ESSInit},
    particles::{PSO, PSOConfig},
};
use ganesh::core::{Callbacks, MaxSteps};
use ganesh::core::{MCMCSummary, MinimizationSummary};
use ganesh::error::GaneshError;
use ganesh::traits::{
    Algorithm, Bounds as GaneshBounds, CostFunction, Gradient, LogDensity, PeriodicTransform,
    ScalarBound, ScaleTransform, Status, SupportsParameterNames, Terminator, Transform,
};
use ganesh::{LinearAlgebra, Matrix, RandomScalar, RealScalar, Vector};
use laddu_expr::parameters::{ParamLayout, Parameter};
use laddu_likelihood::{
    Likelihood, LikelihoodError, LikelihoodEvaluation, Objective, StochasticObjective,
};
use laddu_runtime::Precision;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors raised while adapting a laddu objective to ganesh.
#[derive(Debug, Error)]
pub enum FitError {
    #[error(transparent)]
    Likelihood(#[from] LikelihoodError),
    #[error(transparent)]
    Ganesh(#[from] GaneshError),
    #[error("ganesh scalar value cannot be represented as f64")]
    ScalarConversion,
    #[error("fit result does not contain parameter `{0}`")]
    MissingParameter(String),
}

pub type FitResult<T> = Result<T, FitError>;

/// laddu convenience view retaining ganesh's complete minimization summary.
#[derive(Clone, Serialize)]
#[serde(bound(serialize = "MinimizationSummary<T, B>: Serialize"))]
pub struct MinimizationResult<T: RealScalar = f64, B: LinearAlgebra<T> = ganesh::NalgebraProvider> {
    pub raw: MinimizationSummary<T, B>,
}

impl<T: RealScalar, B: LinearAlgebra<T>> MinimizationResult<T, B> {
    pub fn value(&self) -> T {
        self.raw.fx
    }

    pub fn parameters(&self) -> Vec<(String, T)> {
        (0..self.raw.x.len())
            .map(|index| {
                let name = self
                    .raw
                    .parameter_names
                    .as_ref()
                    .and_then(|names| names.get(index))
                    .cloned()
                    .unwrap_or_else(|| format!("x_{index}"));
                (name, self.raw.x.get(index))
            })
            .collect()
    }

    /// Return a fitted parameter by its user-facing name.
    pub fn parameter(&self, name: &str) -> FitResult<T> {
        self.parameters()
            .into_iter()
            .find_map(|(candidate, value)| (candidate == name).then_some(value))
            .ok_or_else(|| FitError::MissingParameter(name.to_owned()))
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> From<MinimizationSummary<T, B>>
    for MinimizationResult<T, B>
{
    fn from(raw: MinimizationSummary<T, B>) -> Self {
        Self { raw }
    }
}

/// laddu convenience view retaining ganesh's complete MCMC summary.
#[derive(Clone, Serialize)]
#[serde(bound(serialize = "MCMCSummary<T, B>: Serialize"))]
pub struct McmcResult<T: RealScalar = f64, B: LinearAlgebra<T> = ganesh::NalgebraProvider> {
    pub raw: MCMCSummary<T, B>,
}

impl<T: RealScalar, B: LinearAlgebra<T>> McmcResult<T, B> {
    pub fn chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<Vec<Vector<T, B>>> {
        self.raw.get_chain(burn, thin)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> From<MCMCSummary<T, B>> for McmcResult<T, B> {
    fn from(raw: MCMCSummary<T, B>) -> Self {
        Self { raw }
    }
}

/// Which metadata-derived transforms to use for minimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformOptions {
    pub scaling: bool,
    pub periodic: bool,
    pub bounds: bool,
}

/// Minimizers available through the low-configuration likelihood facade.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Minimizer {
    #[default]
    Lbfgsb,
    ConjugateGradient,
    TrustRegion,
    NelderMead,
    DifferentialEvolution,
    SimulatedAnnealing,
    Cmaes,
    Pso,
}

/// Ensemble samplers available through the likelihood facade.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Sampler {
    #[default]
    Aies,
    Ess,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinimizationProgress {
    pub step: usize,
    pub value: f64,
    pub parameters: Vec<(String, f64)>,
}

impl MinimizationProgress {
    pub fn parameter(&self, name: &str) -> Option<f64> {
        self.parameters
            .iter()
            .find_map(|(candidate, value)| (candidate == name).then_some(*value))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallbackControl {
    #[default]
    Continue,
    Stop,
}

pub type ProgressCallback =
    Arc<dyn Fn(&MinimizationProgress) -> CallbackControl + Send + Sync + 'static>;

/// Common, metadata-aware controls for deterministic minimization.
#[derive(Clone, Serialize, Deserialize)]
pub struct MinimizationOptions {
    pub algorithm: Minimizer,
    pub max_steps: usize,
    pub transforms: TransformOptions,
    #[serde(skip, default)]
    callback: Option<ProgressCallback>,
}

impl std::fmt::Debug for MinimizationOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MinimizationOptions")
            .field("algorithm", &self.algorithm)
            .field("max_steps", &self.max_steps)
            .field("transforms", &self.transforms)
            .field("has_callback", &self.callback.is_some())
            .finish()
    }
}

impl Default for MinimizationOptions {
    fn default() -> Self {
        Self {
            algorithm: Minimizer::Lbfgsb,
            max_steps: 4_000,
            transforms: TransformOptions::default(),
            callback: None,
        }
    }
}

impl MinimizationOptions {
    pub fn with_algorithm(mut self, algorithm: Minimizer) -> Self {
        self.algorithm = algorithm;
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&MinimizationProgress) -> CallbackControl + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(callback));
        self
    }
}

/// Adam controls for event-batched stochastic optimization.
#[derive(Clone, Serialize, Deserialize)]
pub struct StochasticMinimizationOptions {
    pub batch_fraction: f64,
    pub seed: u64,
    pub max_steps: usize,
    pub alpha: f64,
    pub beta_1: f64,
    pub beta_2: f64,
    pub epsilon: f64,
    pub transforms: TransformOptions,
    #[serde(skip, default)]
    callback: Option<ProgressCallback>,
}

impl std::fmt::Debug for StochasticMinimizationOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StochasticMinimizationOptions")
            .field("batch_fraction", &self.batch_fraction)
            .field("seed", &self.seed)
            .field("max_steps", &self.max_steps)
            .field("alpha", &self.alpha)
            .field("beta_1", &self.beta_1)
            .field("beta_2", &self.beta_2)
            .field("epsilon", &self.epsilon)
            .field("transforms", &self.transforms)
            .field("has_callback", &self.callback.is_some())
            .finish()
    }
}

impl Default for StochasticMinimizationOptions {
    fn default() -> Self {
        Self {
            batch_fraction: 0.1,
            seed: 0,
            max_steps: 4_000,
            alpha: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            transforms: TransformOptions::default(),
            callback: None,
        }
    }
}

impl StochasticMinimizationOptions {
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&MinimizationProgress) -> CallbackControl + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(callback));
        self
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SamplingOptions {
    pub sampler: Sampler,
    pub steps: usize,
    pub seed: u64,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            sampler: Sampler::Aies,
            steps: 4_000,
            seed: 0,
        }
    }
}

/// ganesh configurations which accept laddu's metadata-derived transform.
pub trait LadduTransformConfig<T, B>: SupportsParameterNames + Sized
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn with_laddu_transform(self, transform: Box<dyn Transform<T, B>>) -> FitResult<Self>;
}

/// Marker for ganesh optimization configurations.
pub trait LadduMinimizerConfig<T, B>: LadduTransformConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
}

/// Marker for ganesh MCMC configurations.
pub trait LadduSamplerConfig<T, B>: LadduTransformConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
}

macro_rules! infallible_transform_config {
    ($($config:ident),+ $(,)?) => {$ (
        impl<T, B> LadduTransformConfig<T, B> for $config<T, B>
        where
            T: RealScalar,
            B: LinearAlgebra<T>,
        {
            fn with_laddu_transform(
                self,
                transform: Box<dyn Transform<T, B>>,
            ) -> FitResult<Self> {
                Ok(self.with_transform(transform))
            }
        }
    )+ };
}

infallible_transform_config!(
    AdamConfig,
    ConjugateGradientConfig,
    TrustRegionConfig,
    NelderMeadConfig,
    DifferentialEvolutionConfig,
);

macro_rules! random_transform_config {
    ($($config:ident),+ $(,)?) => {$ (
        impl<T, B> LadduTransformConfig<T, B> for $config<T, B>
        where
            T: RandomScalar,
            B: LinearAlgebra<T>,
        {
            fn with_laddu_transform(
                self,
                transform: Box<dyn Transform<T, B>>,
            ) -> FitResult<Self> {
                Ok(self.with_transform(transform))
            }
        }
    )+ };
}

random_transform_config!(
    SimulatedAnnealingConfig,
    CMAESConfig,
    PSOConfig,
    AIESConfig,
    ESSConfig,
);

impl<T, L, B> LadduTransformConfig<T, B> for LBFGSBConfig<T, L, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn with_laddu_transform(self, transform: Box<dyn Transform<T, B>>) -> FitResult<Self> {
        Ok(self.with_transform(transform)?)
    }
}

macro_rules! real_minimizer_config {
    ($($config:ident),+ $(,)?) => {$ (
        impl<T, B> LadduMinimizerConfig<T, B> for $config<T, B>
        where
            T: RealScalar,
            B: LinearAlgebra<T>,
        {}
    )+ };
}

real_minimizer_config!(
    AdamConfig,
    ConjugateGradientConfig,
    TrustRegionConfig,
    NelderMeadConfig,
    DifferentialEvolutionConfig,
);

macro_rules! random_minimizer_config {
    ($($config:ident),+ $(,)?) => {$ (
        impl<T, B> LadduMinimizerConfig<T, B> for $config<T, B>
        where
            T: RandomScalar,
            B: LinearAlgebra<T>,
        {}
    )+ };
}

random_minimizer_config!(SimulatedAnnealingConfig, CMAESConfig, PSOConfig);

impl<T, L, B> LadduMinimizerConfig<T, B> for LBFGSBConfig<T, L, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
}

impl<T, B> LadduSamplerConfig<T, B> for AIESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
}

impl<T, B> LadduSamplerConfig<T, B> for ESSConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            scaling: true,
            periodic: true,
            bounds: true,
        }
    }
}

/// Scalar-generic ganesh view of a laddu objective.
#[derive(Clone, Copy, Debug)]
pub struct FitProblem<'a, O: ?Sized, T = f64, B = ganesh::NalgebraProvider> {
    objective: &'a O,
    _numeric: PhantomData<(T, B)>,
}

/// Adam-oriented adapter that draws a deterministic event batch per point.
///
/// ganesh asks for the value and gradient separately. This adapter caches the
/// paired stochastic evaluation so both requests see exactly the same rows.
#[derive(Debug)]
pub struct StochasticFitProblem<'a, O: ?Sized, T = f64, B = ganesh::NalgebraProvider> {
    objective: &'a O,
    fraction: f64,
    next_seed: AtomicU64,
    cache: Mutex<Option<(Vec<f64>, LikelihoodEvaluation)>>,
    _numeric: PhantomData<(T, B)>,
}

impl<'a, O: StochasticObjective + ?Sized, T, B> StochasticFitProblem<'a, O, T, B> {
    pub fn new(objective: &'a O, fraction: f64, seed: u64) -> FitResult<Self> {
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(LikelihoodError::InvalidBatchFraction(fraction).into());
        }
        Ok(Self {
            objective,
            fraction,
            next_seed: AtomicU64::new(seed),
            cache: Mutex::new(None),
            _numeric: PhantomData,
        })
    }

    pub fn initial(&self) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Vector::from_vec(
            self.objective
                .parameter_layout()
                .initial_free_values()
                .into_iter()
                .map(T::literal)
                .collect(),
        )
    }

    /// Configure ganesh Adam from the same laddu metadata used by deterministic
    /// minimizers.
    pub fn configure_adam(
        &self,
        config: AdamConfig<T, B>,
        options: TransformOptions,
    ) -> FitResult<AdamConfig<T, B>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).configure_minimizer(config, options)
    }

    pub fn run<A, S, I, C>(
        &self,
        algorithm: &mut A,
        init: I,
        config: A::Config,
        callbacks: C,
    ) -> FitResult<A::Summary>
    where
        S: Status,
        A: Algorithm<Self, S, (), FitError>,
        I: Into<A::Init>,
        C: Into<Callbacks<A, Self, S, (), FitError, A::Config>>,
    {
        algorithm.process(self, &(), init, config, callbacks)
    }

    pub fn minimize<A, S, I, C>(
        &self,
        algorithm: &mut A,
        init: I,
        config: A::Config,
        callbacks: C,
    ) -> FitResult<MinimizationResult<T, B>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
        S: Status,
        A: Algorithm<Self, S, (), FitError, Summary = MinimizationSummary<T, B>>,
        I: Into<A::Init>,
        C: Into<Callbacks<A, Self, S, (), FitError, A::Config>>,
    {
        self.run(algorithm, init, config, callbacks).map(Into::into)
    }

    fn external(&self, x: &Vector<T, B>) -> FitResult<Vec<f64>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        (0..x.len())
            .map(|index| x.get(index).to_f64().ok_or(FitError::ScalarConversion))
            .collect()
    }

    fn evaluation(&self, x: &Vector<T, B>) -> FitResult<LikelihoodEvaluation>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        let external = self.external(x)?;
        if let Some((_, evaluation)) = self
            .cache
            .lock()
            .as_ref()
            .filter(|(cached, _)| cached == &external)
        {
            return Ok(evaluation.clone());
        }
        let seed = self.next_seed.fetch_add(1, Ordering::Relaxed);
        let evaluation =
            self.objective
                .stochastic_value_gradient(&external, self.fraction, seed)?;
        *self.cache.lock() = Some((external, evaluation.clone()));
        Ok(evaluation)
    }
}

impl<O, T, B> CostFunction<T, B, (), FitError> for StochasticFitProblem<'_, O, T, B>
where
    O: StochasticObjective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> FitResult<T> {
        Ok(T::literal(self.evaluation(x)?.value()))
    }
}

impl<O, T, B> Gradient<T, B, (), FitError> for StochasticFitProblem<'_, O, T, B>
where
    O: StochasticObjective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<Vector<T, B>> {
        Ok(Vector::from_vec(
            self.evaluation(x)?
                .gradient()
                .iter()
                .copied()
                .map(T::literal)
                .collect(),
        ))
    }

    fn evaluate_with_gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<(T, Vector<T, B>)> {
        let evaluation = self.evaluation(x)?;
        Ok((
            T::literal(evaluation.value()),
            Vector::from_vec(
                evaluation
                    .gradient()
                    .iter()
                    .copied()
                    .map(T::literal)
                    .collect(),
            ),
        ))
    }
}

impl<'a, O: Objective + ?Sized, T, B> FitProblem<'a, O, T, B> {
    pub const fn new(objective: &'a O) -> Self {
        Self {
            objective,
            _numeric: PhantomData,
        }
    }

    pub const fn objective(&self) -> &'a O {
        self.objective
    }

    pub fn initial(&self) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Vector::from_vec(
            self.objective
                .parameter_layout()
                .initial_free_values()
                .into_iter()
                .map(T::literal)
                .collect(),
        )
    }

    pub fn parameter_names(&self) -> Vec<String> {
        free_parameters(self.objective.parameter_layout())
            .map(|parameter| parameter.name().to_owned())
            .collect()
    }

    /// Add parameter names plus scaling, periodicity, and smooth bounds to any
    /// ganesh minimizer configuration.
    pub fn configure_minimizer<C>(&self, config: C, options: TransformOptions) -> FitResult<C>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
        C: LadduMinimizerConfig<T, B>,
    {
        let config = config.with_parameter_names(self.parameter_names());
        match self.minimizer_transform(options)? {
            Some(transform) => config.with_laddu_transform(transform),
            None => Ok(config),
        }
    }

    /// Configure L-BFGS-B with native box constraints while still applying
    /// scaling and periodic metadata.
    ///
    /// Ganesh cannot map an unbounded native coordinate through a periodic
    /// transform. When bounded and periodic parameters coexist, this therefore
    /// falls back to Laddu's smooth bounds transform for the nonperiodic
    /// coordinates. Models without that combination retain native boxes.
    pub fn configure_lbfgsb<L>(
        &self,
        config: LBFGSBConfig<T, L, B>,
        mut options: TransformOptions,
    ) -> FitResult<LBFGSBConfig<T, L, B>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        let parameters = free_parameters(self.objective.parameter_layout()).collect::<Vec<_>>();
        let has_bounds = options.bounds
            && parameters.iter().any(|parameter| {
                !(options.periodic && parameter.is_periodic())
                    && (parameter.bounds_spec().min.is_some()
                        || parameter.bounds_spec().max.is_some())
            });
        let has_periodic =
            options.periodic && parameters.iter().any(|parameter| parameter.is_periodic());
        if has_bounds && has_periodic {
            return self.configure_minimizer(config, options);
        }
        let mut config = config.with_parameter_names(self.parameter_names());
        options.bounds = false;
        if let Some(transform) = self.minimizer_transform(options)? {
            config = config.with_transform(transform)?;
        }
        if has_bounds {
            config = config.with_bounds(parameters.iter().map(|parameter| {
                if options.periodic && parameter.is_periodic() {
                    (T::literal(f64::NEG_INFINITY), T::infinity())
                } else {
                    (
                        T::literal(parameter.bounds_spec().min.unwrap_or(f64::NEG_INFINITY)),
                        T::literal(parameter.bounds_spec().max.unwrap_or(f64::INFINITY)),
                    )
                }
            }))?;
        }
        Ok(config)
    }

    /// Add names and posterior-safe linear scaling to AIES or ESS configs.
    /// Parameter bounds are enforced as log-density support rather than via a
    /// nonlinear transform, and periodic coordinates use one canonical copy.
    pub fn configure_sampler<C>(&self, config: C) -> FitResult<C>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
        C: LadduSamplerConfig<T, B>,
    {
        let config = config.with_parameter_names(self.parameter_names());
        match self.sampler_transform()? {
            Some(transform) => config.with_laddu_transform(transform),
            None => Ok(config),
        }
    }

    /// Run any compatible ganesh algorithm with its native typed config and
    /// callbacks. The initialization payload is also algorithm-native: a
    /// vector for minimizers or an ensemble initializer for AIES/ESS.
    pub fn run<A, S, I, C>(
        &self,
        algorithm: &mut A,
        init: I,
        config: A::Config,
        callbacks: C,
    ) -> FitResult<A::Summary>
    where
        S: Status,
        A: Algorithm<Self, S, (), FitError>,
        I: Into<A::Init>,
        C: Into<Callbacks<A, Self, S, (), FitError, A::Config>>,
    {
        algorithm.process(self, &(), init, config, callbacks)
    }

    pub fn minimize<A, S, I, C>(
        &self,
        algorithm: &mut A,
        init: I,
        config: A::Config,
        callbacks: C,
    ) -> FitResult<MinimizationResult<T, B>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
        S: Status,
        A: Algorithm<Self, S, (), FitError, Summary = MinimizationSummary<T, B>>,
        I: Into<A::Init>,
        C: Into<Callbacks<A, Self, S, (), FitError, A::Config>>,
    {
        self.run(algorithm, init, config, callbacks).map(Into::into)
    }

    pub fn sample<A, S, I, C>(
        &self,
        algorithm: &mut A,
        init: I,
        config: A::Config,
        callbacks: C,
    ) -> FitResult<McmcResult<T, B>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
        S: Status,
        A: Algorithm<Self, S, (), FitError, Summary = MCMCSummary<T, B>>,
        I: Into<A::Init>,
        C: Into<Callbacks<A, Self, S, (), FitError, A::Config>>,
    {
        self.run(algorithm, init, config, callbacks).map(Into::into)
    }

    /// Build the default minimizer transform from parameter metadata.
    ///
    /// Scaling is applied first. Periodic coordinates use ganesh's repeated
    /// lift and are deliberately excluded from the smooth bounds transform.
    pub fn minimizer_transform(
        &self,
        options: TransformOptions,
    ) -> FitResult<Option<Box<dyn Transform<T, B>>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        minimizer_transform(self.objective.parameter_layout(), options)
    }

    /// Build the only metadata transform that is posterior-safe without a
    /// Jacobian correction: linear scaling (whose Jacobian is constant).
    /// Bounds are enforced by [`LogDensity::log_density`] as support, and
    /// periodic values remain in their single canonical domain.
    pub fn sampler_transform(&self) -> FitResult<Option<Box<dyn Transform<T, B>>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        scale_transform(self.objective.parameter_layout())
    }

    fn external(&self, x: &Vector<T, B>) -> FitResult<Vec<f64>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        (0..x.len())
            .map(|index| x.get(index).to_f64().ok_or(FitError::ScalarConversion))
            .collect()
    }
}

impl<O, T, B> CostFunction<T, B, (), FitError> for FitProblem<'_, O, T, B>
where
    O: Objective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> FitResult<T> {
        Ok(T::literal(self.objective.value(&self.external(x)?)?))
    }
}

impl<O, T, B> Gradient<T, B, (), FitError> for FitProblem<'_, O, T, B>
where
    O: Objective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<Vector<T, B>> {
        let evaluation = self.objective.value_gradient(&self.external(x)?)?;
        Ok(Vector::from_vec(
            evaluation
                .gradient()
                .iter()
                .copied()
                .map(T::literal)
                .collect(),
        ))
    }

    fn evaluate_with_gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<(T, Vector<T, B>)> {
        let evaluation = self.objective.value_gradient(&self.external(x)?)?;
        Ok((
            T::literal(evaluation.value()),
            Vector::from_vec(
                evaluation
                    .gradient()
                    .iter()
                    .copied()
                    .map(T::literal)
                    .collect(),
            ),
        ))
    }
}

impl<O, T, B> LogDensity<T, B, (), FitError> for FitProblem<'_, O, T, B>
where
    O: Objective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn log_density(&self, x: &Vector<T, B>, _args: &()) -> FitResult<T> {
        let external = self.external(x)?;
        if self
            .objective
            .parameter_layout()
            .validate_free_values(&external)
            .is_err()
        {
            return Ok(-T::infinity());
        }
        Ok(-T::literal(self.objective.value(&external)?))
    }
}

fn free_parameters(layout: &ParamLayout) -> impl Iterator<Item = &Parameter> {
    layout.free_params().iter().map(|id| {
        layout
            .spec(*id)
            .expect("layout owns every free parameter id")
    })
}

fn append_transform<T, B, X>(
    current: Option<Box<dyn Transform<T, B>>>,
    next: X,
) -> Option<Box<dyn Transform<T, B>>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    X: Transform<T, B> + 'static,
{
    Some(match current {
        Some(current) => Box::new(current.then(next)),
        None => Box::new(next),
    })
}

fn scale_transform<T, B>(layout: &ParamLayout) -> FitResult<Option<Box<dyn Transform<T, B>>>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    let parameters = free_parameters(layout).collect::<Vec<_>>();
    if parameters
        .iter()
        .all(|parameter| parameter.scale().is_none())
    {
        return Ok(None);
    }
    let scales = parameters
        .into_iter()
        .map(|parameter| T::literal(parameter.scale().unwrap_or(1.0)));
    Ok(Some(Box::new(
        ScaleTransform::<T, B>::from_parameter_scales(scales)?,
    )))
}

fn minimizer_transform<T, B>(
    layout: &ParamLayout,
    options: TransformOptions,
) -> FitResult<Option<Box<dyn Transform<T, B>>>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    let parameters = free_parameters(layout).collect::<Vec<_>>();
    let mut transform = if options.scaling {
        scale_transform(layout)?
    } else {
        None
    };

    if options.periodic && parameters.iter().any(|parameter| parameter.is_periodic()) {
        let intervals = parameters.iter().map(|parameter| {
            parameter
                .periodic_bounds()
                .map(|(min, max)| (T::literal(min), T::literal(max)))
        });
        transform = append_transform(transform, PeriodicTransform::<T, B>::new(intervals)?);
    }

    if options.bounds
        && parameters.iter().any(|parameter| {
            !(options.periodic && parameter.is_periodic())
                && (parameter.bounds_spec().min.is_some() || parameter.bounds_spec().max.is_some())
        })
    {
        let bounds = parameters.iter().map(|parameter| {
            if options.periodic && parameter.is_periodic() {
                (T::literal(f64::NEG_INFINITY), T::infinity())
            } else {
                (
                    T::literal(parameter.bounds_spec().min.unwrap_or(f64::NEG_INFINITY)),
                    T::literal(parameter.bounds_spec().max.unwrap_or(f64::INFINITY)),
                )
            }
        });
        transform = append_transform(transform, GaneshBounds::<T, B>::new(bounds)?);
    }
    Ok(transform)
}

#[derive(Clone)]
struct ProgressTerminator {
    names: Vec<String>,
    callback: ProgressCallback,
}

fn progress_snapshot<T, B>(
    step: usize,
    value: T,
    x: &Vector<T, B>,
    names: &[String],
) -> MinimizationProgress
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    MinimizationProgress {
        step,
        value: value.to_f64().unwrap_or(f64::NAN),
        parameters: (0..x.len())
            .map(|index| {
                (
                    names
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| format!("x_{index}")),
                    x.get(index).to_f64().unwrap_or(f64::NAN),
                )
            })
            .collect(),
    }
}

impl<A, P, T, B, E, C> Terminator<A, P, GradientStatus<T, B>, (), E, C> for ProgressTerminator
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    A: Algorithm<P, GradientStatus<T, B>, (), E, Config = C>,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut GradientStatus<T, B>,
        _args: &(),
        _config: &C,
    ) -> ControlFlow<()> {
        match (self.callback)(&progress_snapshot(
            current_step,
            status.fx,
            &status.x,
            &self.names,
        )) {
            CallbackControl::Continue => ControlFlow::Continue(()),
            CallbackControl::Stop => ControlFlow::Break(()),
        }
    }
}

impl<A, P, T, B, E, C> Terminator<A, P, GradientFreeStatus<T, B>, (), E, C> for ProgressTerminator
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    A: Algorithm<P, GradientFreeStatus<T, B>, (), E, Config = C>,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut GradientFreeStatus<T, B>,
        _args: &(),
        _config: &C,
    ) -> ControlFlow<()> {
        match (self.callback)(&progress_snapshot(
            current_step,
            status.fx,
            &status.x,
            &self.names,
        )) {
            CallbackControl::Continue => ControlFlow::Continue(()),
            CallbackControl::Stop => ControlFlow::Break(()),
        }
    }
}

fn execute_minimizer<'a, O, T, A, S>(
    problem: &FitProblem<'a, O, T>,
    mut algorithm: A,
    initial: Vector<T>,
    config: A::Config,
    mut callbacks: Callbacks<A, FitProblem<'a, O, T>, S, (), FitError, A::Config>,
    max_steps: usize,
    callback: Option<ProgressCallback>,
) -> FitResult<MinimizationResult<T>>
where
    O: Objective + ?Sized,
    T: RandomScalar,
    ganesh::NalgebraProvider: LinearAlgebra<T>,
    S: Status,
    A: Algorithm<
            FitProblem<'a, O, T>,
            S,
            (),
            FitError,
            Summary = MinimizationSummary<T>,
            Init = Vector<T>,
        >,
    MaxSteps: Terminator<A, FitProblem<'a, O, T>, S, (), FitError, A::Config>,
    ProgressTerminator: Terminator<A, FitProblem<'a, O, T>, S, (), FitError, A::Config>,
{
    callbacks = callbacks.with_terminator(MaxSteps(max_steps));
    if let Some(callback) = callback {
        callbacks = callbacks.with_terminator(ProgressTerminator {
            names: problem.parameter_names(),
            callback,
        });
    }
    problem.minimize(&mut algorithm, initial, config, callbacks)
}

macro_rules! define_minimize_with_precision {
    ($name:ident, $scalar:ty) => {
        fn $name(
            likelihood: &Likelihood,
            initial: &[f64],
            options: MinimizationOptions,
        ) -> FitResult<MinimizationResult<$scalar>> {
            type T = $scalar;
            let problem = FitProblem::<_, T>::new(likelihood);
            let initial = Vector::from_vec(initial.iter().copied().map(T::literal).collect());
            let transforms = options.transforms;
            let max_steps = options.max_steps;
            let callback = options.callback;
            match options.algorithm {
                Minimizer::Lbfgsb => {
                    let config =
                        problem.configure_lbfgsb(LBFGSBConfig::<T>::default(), transforms)?;
                    execute_minimizer(
                        &problem,
                        LBFGSB::<T>::default(),
                        initial,
                        config,
                        LBFGSB::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
                Minimizer::ConjugateGradient => {
                    let config = problem
                        .configure_minimizer(ConjugateGradientConfig::<T>::default(), transforms)?;
                    execute_minimizer(
                        &problem,
                        ConjugateGradient::<T>::default(),
                        initial,
                        config,
                        ConjugateGradient::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
                Minimizer::TrustRegion => {
                    let config = problem
                        .configure_minimizer(TrustRegionConfig::<T>::default(), transforms)?;
                    execute_minimizer(
                        &problem,
                        TrustRegion::<T>::default(),
                        initial,
                        config,
                        TrustRegion::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
                Minimizer::NelderMead => {
                    let config = problem
                        .configure_minimizer(NelderMeadConfig::<T>::default(), transforms)?;
                    execute_minimizer(
                        &problem,
                        NelderMead::<T>::default(),
                        initial,
                        config,
                        NelderMead::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
                Minimizer::DifferentialEvolution => {
                    let config = problem.configure_minimizer(
                        DifferentialEvolutionConfig::<T>::default(),
                        transforms,
                    )?;
                    execute_minimizer(
                        &problem,
                        DifferentialEvolution::<T>::default(),
                        initial,
                        config,
                        DifferentialEvolution::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
                Minimizer::SimulatedAnnealing => {
                    let config = problem.configure_minimizer(
                        SimulatedAnnealingConfig::<T>::default(),
                        transforms,
                    )?;
                    execute_minimizer(
                        &problem,
                        SimulatedAnnealing::<T>::default(),
                        initial,
                        config,
                        SimulatedAnnealing::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
                Minimizer::Cmaes => {
                    let config =
                        problem.configure_minimizer(CMAESConfig::<T>::default(), transforms)?;
                    execute_minimizer(
                        &problem,
                        CMAES::<T>::default(),
                        initial,
                        config,
                        CMAES::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
                Minimizer::Pso => {
                    let config =
                        problem.configure_minimizer(PSOConfig::<T>::default(), transforms)?;
                    execute_minimizer(
                        &problem,
                        PSO::<T>::default(),
                        initial,
                        config,
                        PSO::<T>::default_callbacks(),
                        max_steps,
                        callback,
                    )
                }
            }
        }
    };
}

define_minimize_with_precision!(minimize_f32, f32);
define_minimize_with_precision!(minimize_f64, f64);

fn minimization_to_f64<T>(result: MinimizationResult<T>) -> FitResult<MinimizationResult>
where
    T: RealScalar,
    ganesh::NalgebraProvider: LinearAlgebra<T>,
{
    let raw = result.raw;
    let convert_vector = |vector: Vector<T>| -> FitResult<Vector<f64>> {
        Ok(Vector::from_vec(
            vector
                .to_vec()
                .into_iter()
                .map(|value| value.to_f64().ok_or(FitError::ScalarConversion))
                .collect::<FitResult<Vec<_>>>()?,
        ))
    };
    let convert_bound = |bound: ScalarBound<T>| -> FitResult<ScalarBound<f64>> {
        Ok(match bound {
            ScalarBound::Unbounded => ScalarBound::Unbounded,
            ScalarBound::Lower(value) => {
                ScalarBound::Lower(value.to_f64().ok_or(FitError::ScalarConversion)?)
            }
            ScalarBound::Upper(value) => {
                ScalarBound::Upper(value.to_f64().ok_or(FitError::ScalarConversion)?)
            }
            ScalarBound::Both(lower, upper) => ScalarBound::Both(
                lower.to_f64().ok_or(FitError::ScalarConversion)?,
                upper.to_f64().ok_or(FitError::ScalarConversion)?,
            ),
        })
    };
    let covariance = Matrix::from_vec(
        raw.covariance.rows(),
        raw.covariance.cols(),
        (0..raw.covariance.rows())
            .flat_map(|row| (0..raw.covariance.cols()).map(move |col| (row, col)))
            .map(|(row, col)| {
                raw.covariance
                    .get(row, col)
                    .to_f64()
                    .ok_or(FitError::ScalarConversion)
            })
            .collect::<FitResult<Vec<_>>>()?,
    );
    Ok(MinimizationResult {
        raw: MinimizationSummary {
            bounds: raw
                .bounds
                .map(|bounds| bounds.into_iter().map(convert_bound).collect())
                .transpose()?,
            parameter_names: raw.parameter_names,
            message: raw.message,
            x0: convert_vector(raw.x0)?,
            x: convert_vector(raw.x)?,
            std: convert_vector(raw.std)?,
            fx: raw.fx.to_f64().ok_or(FitError::ScalarConversion)?,
            evals: raw.evals,
            covariance,
        },
    })
}

fn stochastic_minimize_f64(
    likelihood: &Likelihood,
    initial: &[f64],
    options: StochasticMinimizationOptions,
) -> FitResult<MinimizationResult> {
    let problem =
        StochasticFitProblem::<_, f64>::new(likelihood, options.batch_fraction, options.seed)?;
    let config = problem.configure_adam(
        AdamConfig::<f64>::default()
            .with_alpha(options.alpha)?
            .with_beta_1(options.beta_1)?
            .with_beta_2(options.beta_2)?
            .with_epsilon(options.epsilon)?,
        options.transforms,
    )?;
    let mut callbacks =
        Adam::<f64>::default_callbacks().with_terminator(MaxSteps(options.max_steps));
    if let Some(callback) = options.callback {
        callbacks = callbacks.with_terminator(ProgressTerminator {
            names: free_parameters(likelihood.params())
                .map(|parameter| parameter.name().to_owned())
                .collect(),
            callback,
        });
    }
    problem.minimize(
        &mut Adam::<f64>::default(),
        Vector::from_vec(initial.to_vec()),
        config,
        callbacks,
    )
}

fn stochastic_minimize_f32(
    likelihood: &Likelihood,
    initial: &[f64],
    options: StochasticMinimizationOptions,
) -> FitResult<MinimizationResult> {
    let problem =
        StochasticFitProblem::<_, f32>::new(likelihood, options.batch_fraction, options.seed)?;
    let config = problem.configure_adam(
        AdamConfig::<f32>::default()
            .with_alpha(options.alpha as f32)?
            .with_beta_1(options.beta_1 as f32)?
            .with_beta_2(options.beta_2 as f32)?
            .with_epsilon(options.epsilon as f32)?,
        options.transforms,
    )?;
    let mut callbacks =
        Adam::<f32>::default_callbacks().with_terminator(MaxSteps(options.max_steps));
    if let Some(callback) = options.callback {
        callbacks = callbacks.with_terminator(ProgressTerminator {
            names: free_parameters(likelihood.params())
                .map(|parameter| parameter.name().to_owned())
                .collect(),
            callback,
        });
    }
    minimization_to_f64(problem.minimize(
        &mut Adam::<f32>::default(),
        Vector::from_vec(initial.iter().map(|value| *value as f32).collect()),
        config,
        callbacks,
    )?)
}

fn sample_f64(
    likelihood: &Likelihood,
    walkers: &[Vec<f64>],
    options: SamplingOptions,
) -> FitResult<McmcResult> {
    let problem = FitProblem::<_, f64>::new(likelihood);
    let walkers = walkers
        .iter()
        .cloned()
        .map(Vector::from_vec)
        .collect::<Vec<_>>();
    match options.sampler {
        Sampler::Aies => {
            let config = problem.configure_sampler(AIESConfig::<f64>::default())?;
            problem.sample(
                &mut AIES::<f64>::new(Some(options.seed)),
                AIESInit::new(walkers)?,
                config,
                AIES::<f64>::default_callbacks().with_terminator(MaxSteps(options.steps)),
            )
        }
        Sampler::Ess => {
            let config = problem.configure_sampler(ESSConfig::<f64>::default())?;
            problem.sample(
                &mut ESS::<f64>::new(Some(options.seed)),
                ESSInit::new(walkers)?,
                config,
                ESS::<f64>::default_callbacks().with_terminator(MaxSteps(options.steps)),
            )
        }
    }
}

/// High-level fitting and sampling methods for prepared likelihoods.
pub trait LikelihoodFitExt {
    fn minimize(
        &self,
        initial: &[f64],
        options: Option<MinimizationOptions>,
    ) -> FitResult<MinimizationResult>;

    fn minimize_stochastic(
        &self,
        initial: &[f64],
        options: Option<StochasticMinimizationOptions>,
    ) -> FitResult<MinimizationResult>;

    fn sample(
        &self,
        walkers: &[Vec<f64>],
        options: Option<SamplingOptions>,
    ) -> FitResult<McmcResult>;
}

impl LikelihoodFitExt for Likelihood {
    fn minimize(
        &self,
        initial: &[f64],
        options: Option<MinimizationOptions>,
    ) -> FitResult<MinimizationResult> {
        match self.execution().precision() {
            Precision::F32 => {
                minimization_to_f64(minimize_f32(self, initial, options.unwrap_or_default())?)
            }
            Precision::Auto | Precision::F64 => {
                minimize_f64(self, initial, options.unwrap_or_default())
            }
        }
    }

    fn minimize_stochastic(
        &self,
        initial: &[f64],
        options: Option<StochasticMinimizationOptions>,
    ) -> FitResult<MinimizationResult> {
        match self.execution().precision() {
            Precision::F32 => stochastic_minimize_f32(self, initial, options.unwrap_or_default()),
            Precision::Auto | Precision::F64 => {
                stochastic_minimize_f64(self, initial, options.unwrap_or_default())
            }
        }
    }

    fn sample(
        &self,
        walkers: &[Vec<f64>],
        options: Option<SamplingOptions>,
    ) -> FitResult<McmcResult> {
        // Sampling remains f64 because posterior chains are a reporting boundary;
        // likelihood evaluation still follows its configured execution backend.
        sample_f64(self, walkers, options.unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laddu_expr::parameters::Parameter;
    use laddu_likelihood::{LikelihoodEvaluation, LikelihoodResult};
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug)]
    struct Quadratic {
        layout: ParamLayout,
    }

    impl Objective for Quadratic {
        fn parameter_layout(&self) -> &ParamLayout {
            &self.layout
        }

        fn value(&self, parameters: &[f64]) -> LikelihoodResult<f64> {
            Ok(parameters.iter().map(|value| value * value).sum())
        }

        fn value_gradient(&self, parameters: &[f64]) -> LikelihoodResult<LikelihoodEvaluation> {
            let value = self.value(parameters)?;
            let gradient = parameters.iter().map(|value| 2.0 * value).collect();
            Ok(LikelihoodEvaluation::new(value, gradient))
        }
    }

    #[test]
    fn adapter_is_native_in_both_precisions() {
        let objective = Quadratic {
            layout: ParamLayout::new([Parameter::free("x").with_initial(2.0)]).unwrap(),
        };
        let f64_problem = FitProblem::<_, f64>::new(&objective);
        let f32_problem = FitProblem::<_, f32>::new(&objective);
        assert_eq!(
            f64_problem.evaluate(&f64_problem.initial(), &()).unwrap(),
            4.0
        );
        assert_eq!(
            f32_problem.evaluate(&f32_problem.initial(), &()).unwrap(),
            4.0
        );
    }

    #[test]
    fn lbfgsb_accepts_mixed_bounded_and_periodic_parameters() {
        let objective = Quadratic {
            layout: ParamLayout::new([
                Parameter::free("magnitude")
                    .with_initial(0.5)
                    .with_bounds(0.0, 2.0),
                Parameter::free("phase")
                    .with_initial(0.0)
                    .with_bounds(-std::f64::consts::PI, std::f64::consts::PI)
                    .with_periodic(),
            ])
            .unwrap(),
        };
        let problem = FitProblem::<_, f64>::new(&objective);

        problem
            .configure_lbfgsb(LBFGSBConfig::<f64>::default(), TransformOptions::default())
            .expect("mixed bounds and periodic metadata should produce a valid transform");
    }

    #[derive(Debug)]
    struct StochasticQuadratic {
        inner: Quadratic,
        calls: AtomicUsize,
    }

    impl Objective for StochasticQuadratic {
        fn parameter_layout(&self) -> &ParamLayout {
            self.inner.parameter_layout()
        }

        fn value(&self, parameters: &[f64]) -> LikelihoodResult<f64> {
            self.inner.value(parameters)
        }

        fn value_gradient(&self, parameters: &[f64]) -> LikelihoodResult<LikelihoodEvaluation> {
            self.inner.value_gradient(parameters)
        }
    }

    impl StochasticObjective for StochasticQuadratic {
        fn stochastic_value_gradient(
            &self,
            parameters: &[f64],
            _fraction: f64,
            _seed: u64,
        ) -> LikelihoodResult<LikelihoodEvaluation> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.value_gradient(parameters)
        }
    }

    #[test]
    fn stochastic_adapter_pairs_value_and_gradient_on_one_batch() {
        let objective = StochasticQuadratic {
            inner: Quadratic {
                layout: ParamLayout::new([Parameter::free("x").with_initial(2.0)]).unwrap(),
            },
            calls: AtomicUsize::new(0),
        };
        let problem = StochasticFitProblem::<_, f64>::new(&objective, 0.5, 7).unwrap();
        let initial = problem.initial();
        assert_eq!(problem.evaluate(&initial, &()).unwrap(), 4.0);
        assert_eq!(problem.gradient(&initial, &()).unwrap().get(0), 4.0);
        assert_eq!(objective.calls.load(Ordering::Relaxed), 1);
    }
}
