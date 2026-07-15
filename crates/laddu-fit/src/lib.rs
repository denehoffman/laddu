//! ganesh-backed minimization and sampling adapters for laddu objectives.
//!
//! [`FitProblem`] implements ganesh's cost, gradient, and log-density traits
//! for both `f32` and `f64`. The adapter leaves algorithm choice and callbacks
//! fully exposed while centralizing laddu parameter names and transforms.

pub use ganesh;

use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use ganesh::algorithms::{
    gradient::{AdamConfig, ConjugateGradientConfig, LBFGSBConfig, TrustRegionConfig},
    gradient_free::{
        CMAESConfig, DifferentialEvolutionConfig, NelderMeadConfig, SimulatedAnnealingConfig,
    },
    mcmc::{AIESConfig, ESSConfig},
    particles::PSOConfig,
};
use ganesh::core::Callbacks;
use ganesh::core::{MCMCSummary, MinimizationSummary};
use ganesh::error::GaneshError;
use ganesh::traits::{
    Algorithm, Bounds as GaneshBounds, CostFunction, Gradient, LogDensity, PeriodicTransform,
    ScaleTransform, Status, SupportsParameterNames, Transform,
};
use ganesh::{LinearAlgebra, RandomScalar, RealScalar, Vector};
use laddu_expr::parameters::{ParamLayout, Parameter};
use laddu_likelihood::{LikelihoodError, LikelihoodEvaluation, Objective, StochasticObjective};
use parking_lot::Mutex;
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
}

pub type FitResult<T> = Result<T, FitError>;

/// laddu convenience view retaining ganesh's complete minimization summary.
#[derive(Clone)]
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
}

impl<T: RealScalar, B: LinearAlgebra<T>> From<MinimizationSummary<T, B>>
    for MinimizationResult<T, B>
{
    fn from(raw: MinimizationSummary<T, B>) -> Self {
        Self { raw }
    }
}

/// laddu convenience view retaining ganesh's complete MCMC summary.
#[derive(Clone)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformOptions {
    pub scaling: bool,
    pub periodic: bool,
    pub bounds: bool,
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
        let has_bounds = parameters.iter().any(|parameter| {
            parameter.periodic_domain().is_none()
                && (parameter.bounds_spec().min.is_some() || parameter.bounds_spec().max.is_some())
        });
        let has_periodic = options.periodic
            && parameters
                .iter()
                .any(|parameter| parameter.periodic_domain().is_some());
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
                if parameter.periodic_domain().is_some() {
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

    if options.periodic
        && parameters
            .iter()
            .any(|parameter| parameter.periodic_domain().is_some())
    {
        let intervals = parameters.iter().map(|parameter| {
            parameter
                .periodic_domain()
                .map(|domain| (T::literal(domain.min()), T::literal(domain.max())))
        });
        transform = append_transform(transform, PeriodicTransform::<T, B>::new(intervals)?);
    }

    if options.bounds
        && parameters.iter().any(|parameter| {
            parameter.periodic_domain().is_none()
                && (parameter.bounds_spec().min.is_some() || parameter.bounds_spec().max.is_some())
        })
    {
        let bounds = parameters.iter().map(|parameter| {
            if parameter.periodic_domain().is_some() {
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
                    .with_periodic(-std::f64::consts::PI, std::f64::consts::PI),
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
