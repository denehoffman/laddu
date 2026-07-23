//! ganesh-backed minimization and sampling adapters for laddu objectives.
//!
//! [`FitProblem`] implements ganesh's cost, gradient, and log-density traits
//! for both `f32` and `f64`. The adapter leaves algorithm choice and callbacks
//! fully exposed while centralizing laddu parameter conversion, metadata, and
//! stochastic batching.

pub use ganesh;

use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use ganesh::error::GaneshError;
use ganesh::traits::{
    Bounds as GaneshBounds, CostFunction, Gradient, IdentityTransform, LogDensity,
    PeriodicTransform, ScaleTransform, Transform,
};
use ganesh::{LinearAlgebra, RealScalar, Vector};
use laddu_expr::parameters::{ParamLayout, Parameter};
use laddu_likelihood::{LikelihoodError, LikelihoodEvaluation, Objective, StochasticObjective};
use parking_lot::Mutex;
use thiserror::Error;

/// Errors raised while adapting a laddu objective to ganesh.
#[derive(Debug, Error)]
pub enum FitError {
    /// The likelihood could not be evaluated or prepared.
    #[error(transparent)]
    Likelihood(#[from] LikelihoodError),
    /// The underlying ganesh optimizer or sampler failed.
    #[error(transparent)]
    Optimizer(#[from] GaneshError),
    /// A backend scalar could not be represented as an `f64`.
    #[error("ganesh scalar value cannot be represented as f64")]
    ScalarConversion,
}

/// A result produced by a laddu fitting operation.
pub type FitResult<T> = Result<T, FitError>;

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
    /// Create a stochastic adapter that samples `fraction` of events per evaluation.
    ///
    /// `seed` initializes the deterministic sequence of batch seeds.
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

    /// Return the adapted stochastic objective.
    pub const fn objective(&self) -> &'a O {
        self.objective
    }

    /// Return free parameter names in optimizer-vector order.
    pub fn parameter_names(&self) -> Vec<String> {
        FitProblem::<O, T, B>::new(self.objective).parameter_names()
    }

    /// Convert user-facing `f64` values to the backend vector type.
    pub fn vector(&self, values: &[f64]) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).vector(values)
    }

    /// Build the parameter transform for minimizers without native bounds.
    pub fn minimizer_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).minimizer_transform()
    }

    /// Build the parameter transform for minimizers that enforce native bounds.
    pub fn native_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).native_transform()
    }

    /// Return native optimizer bounds in transformed coordinates.
    pub fn native_bounds(&self) -> Vec<(T, T)>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).native_bounds()
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
    /// Adapt an objective to ganesh's scalar-generic optimization traits.
    pub const fn new(objective: &'a O) -> Self {
        Self {
            objective,
            _numeric: PhantomData,
        }
    }

    /// Return the adapted objective.
    pub const fn objective(&self) -> &'a O {
        self.objective
    }

    /// Convert user-facing f64 parameter values to this problem's ganesh scalar.
    pub fn vector(&self, values: &[f64]) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Vector::from_vec(values.iter().copied().map(T::literal).collect())
    }

    /// Return free parameter names in optimizer-vector order.
    pub fn parameter_names(&self) -> Vec<String> {
        free_parameters(self.objective.parameter_layout())
            .map(|parameter| parameter.name().to_owned())
            .collect()
    }

    /// Build the metadata transform for minimizers without native bounds.
    /// Scaling, periodic wrapping, and smooth bounds are applied automatically.
    pub fn minimizer_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Ok(
            minimizer_transform(self.objective.parameter_layout(), true)?
                .unwrap_or_else(|| Box::new(IdentityTransform)),
        )
    }

    /// Build the metadata transform for algorithms with native bounds, such as
    /// L-BFGS-B. Bounds themselves are returned by [`Self::native_bounds`].
    pub fn native_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Ok(
            minimizer_transform(self.objective.parameter_layout(), false)?
                .unwrap_or_else(|| Box::new(IdentityTransform)),
        )
    }

    /// Native optimizer bounds in the coordinates produced by
    /// [`Self::native_transform`]. Periodic parameters are deliberately
    /// unbounded in optimizer space.
    pub fn native_bounds(&self) -> Vec<(T, T)>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        free_parameters(self.objective.parameter_layout())
            .map(|parameter| {
                if parameter.is_periodic() {
                    (T::literal(f64::NEG_INFINITY), T::infinity())
                } else {
                    let scale = parameter.scale().unwrap_or(1.0);
                    (
                        T::literal(
                            parameter.bounds_spec().min.unwrap_or(f64::NEG_INFINITY) / scale,
                        ),
                        T::literal(parameter.bounds_spec().max.unwrap_or(f64::INFINITY) / scale),
                    )
                }
            })
            .collect()
    }

    /// Build the only metadata transform that is posterior-safe without a
    /// Jacobian correction: linear scaling (whose Jacobian is constant).
    /// Bounds are enforced by [`LogDensity::log_density`] as support, and
    /// periodic values remain in their single canonical domain.
    pub fn sampler_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Ok(scale_transform(self.objective.parameter_layout())?
            .unwrap_or_else(|| Box::new(IdentityTransform)))
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
    include_bounds: bool,
) -> FitResult<Option<Box<dyn Transform<T, B>>>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    let parameters = free_parameters(layout).collect::<Vec<_>>();
    let mut transform = scale_transform(layout)?;

    if parameters.iter().any(|parameter| parameter.is_periodic()) {
        let intervals = parameters.iter().map(|parameter| {
            parameter
                .periodic_bounds()
                .map(|(min, max)| (T::literal(min), T::literal(max)))
        });
        transform = append_transform(transform, PeriodicTransform::<T, B>::new(intervals)?);
    }

    if include_bounds
        && parameters.iter().any(|parameter| {
            !parameter.is_periodic()
                && (parameter.bounds_spec().min.is_some() || parameter.bounds_spec().max.is_some())
        })
    {
        let bounds = parameters.iter().map(|parameter| {
            if parameter.is_periodic() {
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
    use ganesh::{
        algorithms::{
            gradient::{Adam, AdamConfig, ConjugateGradientConfig, LBFGSB, LBFGSBConfig},
            mcmc::{AIES, AIESConfig, AIESInit, ESS, ESSConfig, ESSInit},
        },
        core::{Callbacks, MaxSteps},
        traits::{Algorithm, SupportsParameterNames},
    };
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
            f64_problem
                .evaluate(&f64_problem.vector(&[2.0]), &())
                .unwrap(),
            4.0
        );
        assert_eq!(
            f32_problem
                .evaluate(&f32_problem.vector(&[2.0]), &())
                .unwrap(),
            4.0
        );
    }

    #[test]
    fn ensemble_samplers_are_native_in_both_precisions() {
        let objective = Quadratic {
            layout: ParamLayout::new([Parameter::free("x").with_initial(0.0)]).unwrap(),
        };

        let f32_problem = FitProblem::<_, f32>::new(&objective);
        let f32_walkers = [-0.3_f32, -0.1, 0.1, 0.3]
            .into_iter()
            .map(|value| Vector::from_vec(vec![value]))
            .collect::<Vec<_>>();
        let aies_config = AIESConfig::<f32>::default()
            .with_parameter_names(f32_problem.parameter_names())
            .with_transform(f32_problem.sampler_transform().unwrap());
        let aies = AIES::<f32>::new(Some(7))
            .process(
                &f32_problem,
                &(),
                AIESInit::new(f32_walkers.clone()).unwrap(),
                aies_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(aies.dimension, (4, 5, 1));

        let ess_config = ESSConfig::<f32>::default()
            .with_parameter_names(f32_problem.parameter_names())
            .with_transform(f32_problem.sampler_transform().unwrap());
        let ess = ESS::<f32>::new(Some(11))
            .process(
                &f32_problem,
                &(),
                ESSInit::new(f32_walkers).unwrap(),
                ess_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(ess.dimension, (4, 5, 1));

        let f64_problem = FitProblem::<_, f64>::new(&objective);
        let f64_walkers = [-0.3_f64, -0.1, 0.1, 0.3]
            .into_iter()
            .map(|value| Vector::from_vec(vec![value]))
            .collect::<Vec<_>>();
        let aies_config = AIESConfig::<f64>::default()
            .with_parameter_names(f64_problem.parameter_names())
            .with_transform(f64_problem.sampler_transform().unwrap());
        let aies = AIES::<f64>::new(Some(13))
            .process(
                &f64_problem,
                &(),
                AIESInit::new(f64_walkers.clone()).unwrap(),
                aies_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(aies.dimension, (4, 5, 1));

        let ess_config = ESSConfig::<f64>::default()
            .with_parameter_names(f64_problem.parameter_names())
            .with_transform(f64_problem.sampler_transform().unwrap());
        let ess = ESS::<f64>::new(Some(17))
            .process(
                &f64_problem,
                &(),
                ESSInit::new(f64_walkers).unwrap(),
                ess_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(ess.dimension, (4, 5, 1));
    }

    #[test]
    fn lbfgsb_accepts_mixed_bounded_and_periodic_parameters() {
        let objective = Quadratic {
            layout: ParamLayout::new([
                Parameter::free("magnitude")
                    .with_initial(0.5)
                    .with_scale(0.25)
                    .with_bounds(0.0, 2.0),
                Parameter::free("phase")
                    .with_initial(0.0)
                    .with_bounds(-std::f64::consts::PI, std::f64::consts::PI)
                    .with_periodic(),
            ])
            .unwrap(),
        };
        let problem = FitProblem::<_, f64>::new(&objective);

        let config = LBFGSBConfig::<f64>::default()
            .with_parameter_names(problem.parameter_names())
            .with_transform(problem.native_transform().unwrap())
            .unwrap()
            .with_bounds(problem.native_bounds())
            .unwrap();
        let result = LBFGSB::<f64>::default()
            .process(
                &problem,
                &(),
                problem.vector(&[0.5, 0.0]),
                config,
                LBFGSB::<f64>::default_callbacks().with_terminator(MaxSteps(20)),
            )
            .unwrap();
        assert!((0.0..=2.0).contains(&result.x.get(0)));
        assert!(result.x.get(1).abs() <= std::f64::consts::PI);
    }

    #[test]
    fn typed_configuration_preserves_custom_line_searches() {
        use ganesh::algorithms::line_search::HagerZhangLineSearch;

        let objective = Quadratic {
            layout: ParamLayout::new([Parameter::free("x").with_initial(1.0)]).unwrap(),
        };
        let problem = FitProblem::<_, f64>::new(&objective);
        let _config = ConjugateGradientConfig::<f64>::default()
            .with_line_search(HagerZhangLineSearch::<f64>::default())
            .with_parameter_names(problem.parameter_names())
            .with_transform(problem.minimizer_transform().unwrap());
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
        let initial = problem.vector(&[2.0]);
        assert_eq!(problem.evaluate(&initial, &()).unwrap(), 4.0);
        assert_eq!(problem.gradient(&initial, &()).unwrap().get(0), 4.0);
        assert_eq!(objective.calls.load(Ordering::Relaxed), 1);

        let config = AdamConfig::<f64>::default()
            .with_parameter_names(problem.parameter_names())
            .with_transform(problem.minimizer_transform().unwrap());
        let result = Adam::<f64>::default()
            .process(
                &problem,
                &(),
                initial,
                config,
                Adam::<f64>::default_callbacks().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert!(result.fx.is_finite());
    }
}
