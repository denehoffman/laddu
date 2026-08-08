use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use laddu_fit::{
    FitError, FitProblem,
    ganesh::{
        NalgebraProvider, Vector,
        algorithms::{
            gradient::{Adam, AdamConfig, GradientStatus, LBFGSB, LBFGSBConfig},
            gradient_free::{GradientFreeStatus, NelderMead},
            mcmc::{AIES, AIESConfig, EnsembleStatus},
        },
        core::{Callbacks, DebugObserver, MaxSteps, ProgressObserver},
        python::{
            PyAIESConfig, PyAIESInit, PyAdamConfig, PyDebugObserver, PyLBFGSBConfig, PyMCMCSummary,
            PyMaxSteps, PyMinimizationSummary, PyNelderMeadConfig, PyProgressObserver,
            PyVectorInit, PythonCallbackBundle, process_with_python_callbacks,
        },
        traits::{Algorithm, CostFunction, Gradient, LogDensity, SupportsParameterNames},
    },
};
use laddu_likelihood::{
    BootstrapFitError, Ensemble, Likelihood, LikelihoodEvaluation, Objective, StochasticObjective,
};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyAny};

use super::{
    cross_section::PyEnsemble,
    error::to_py_err,
    likelihood::{PyLikelihood, free_values},
};

#[derive(Clone)]
struct OwnedProblem {
    objective: Arc<Likelihood>,
}

impl OwnedProblem {
    fn new(objective: Arc<Likelihood>) -> Self {
        Self { objective }
    }
    fn metadata(&self) -> FitProblem<'_, Likelihood, f64> {
        FitProblem::new(&self.objective)
    }
    fn external(x: &Vector) -> Vec<f64> {
        (0..x.len()).map(|index| x.get(index)).collect()
    }
}

impl CostFunction<f64, NalgebraProvider, (), FitError> for OwnedProblem {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, FitError> {
        Ok(self.objective.value(&Self::external(x))?)
    }
}

impl Gradient<f64, NalgebraProvider, (), FitError> for OwnedProblem {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, FitError> {
        Ok(Vector::from_vec(
            self.objective
                .value_gradient(&Self::external(x))?
                .gradient()
                .to_vec(),
        ))
    }

    fn evaluate_with_gradient(&self, x: &Vector, _: &()) -> Result<(f64, Vector), FitError> {
        let evaluation = self.objective.value_gradient(&Self::external(x))?;
        Ok((
            evaluation.value(),
            Vector::from_vec(evaluation.gradient().to_vec()),
        ))
    }
}

impl LogDensity<f64, NalgebraProvider, (), FitError> for OwnedProblem {
    fn log_density(&self, x: &Vector, _: &()) -> Result<f64, FitError> {
        let external = Self::external(x);
        if self
            .objective
            .parameter_layout()
            .validate_free_values(&external)
            .is_err()
        {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(-self.objective.value(&external)?)
    }
}

struct OwnedStochasticProblem {
    objective: Arc<Likelihood>,
    fraction: f64,
    next_seed: AtomicU64,
}

impl OwnedStochasticProblem {
    fn evaluate(&self, x: &Vector) -> Result<LikelihoodEvaluation, FitError> {
        let values = OwnedProblem::external(x);
        let seed = self.next_seed.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .objective
            .stochastic_value_gradient(&values, self.fraction, seed)?)
    }
}

impl CostFunction<f64, NalgebraProvider, (), FitError> for OwnedStochasticProblem {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, FitError> {
        Ok(self.evaluate(x)?.value())
    }
}

impl Gradient<f64, NalgebraProvider, (), FitError> for OwnedStochasticProblem {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, FitError> {
        Ok(Vector::from_vec(self.evaluate(x)?.gradient().to_vec()))
    }

    fn evaluate_with_gradient(&self, x: &Vector, _: &()) -> Result<(f64, Vector), FitError> {
        let evaluation = self.evaluate(x)?;
        Ok((
            evaluation.value(),
            Vector::from_vec(evaluation.gradient().to_vec()),
        ))
    }
}

fn initial_vector(likelihood: &Likelihood, initial: Option<&Bound<'_, PyAny>>) -> PyResult<Vector> {
    let values = match initial {
        None => likelihood.default_params(),
        Some(value) if value.extract::<PyRef<'_, PyVectorInit>>().is_ok() => value
            .extract::<PyRef<'_, PyVectorInit>>()?
            .to_rust()
            .to_vec(),
        Some(value) => free_values(likelihood, value)?,
    };
    Ok(Vector::from_vec(values))
}

fn callback_bundle<A, P, S, C>(
    py: Python<'_>,
    callbacks: Callbacks<A, P, S, (), FitError, C>,
    terminators: Vec<Py<PyAny>>,
    observers: Vec<Py<PyAny>>,
) -> PyResult<PythonCallbackBundle<A, P, S, (), FitError, C>>
where
    A: Algorithm<P, S, (), FitError, Config = C> + 'static,
    P: 'static,
    S: laddu_fit::ganesh::traits::Status
        + laddu_fit::ganesh::traits::ProgressStatus
        + std::fmt::Debug
        + 'static,
    C: 'static,
{
    let mut bundle = PythonCallbackBundle::new(callbacks);
    for terminator in terminators {
        if let Ok(value) = terminator.bind(py).extract::<PyRef<'_, PyMaxSteps>>() {
            bundle = bundle.with_terminator(MaxSteps::from(*value));
        } else if terminator.bind(py).is_callable() {
            bundle = bundle.with_python_terminator(terminator);
        } else {
            return Err(PyTypeError::new_err(
                "terminators must be ganesh.MaxSteps objects or Python callables",
            ));
        }
    }
    for observer in observers {
        if let Ok(value) = observer.bind(py).extract::<PyRef<'_, PyProgressObserver>>() {
            bundle = bundle.with_observer(ProgressObserver::from(*value));
        } else if let Ok(value) = observer.bind(py).extract::<PyRef<'_, PyDebugObserver>>() {
            bundle = bundle.with_observer(DebugObserver::from(*value));
        } else if observer.bind(py).is_callable() {
            bundle = bundle.with_python_observer(observer);
        } else {
            return Err(PyTypeError::new_err(
                "observers must be ganesh observer objects or Python callables",
            ));
        }
    }
    Ok(bundle)
}

#[pymethods]
impl PyLikelihood {
    #[pyo3(signature = (
        center: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        n_walkers,
        *,
        scale=1.0e-3,
        seed=0
    ) -> "Sequence[Sequence[float]]")]
    /// Generate walker positions in a small cloud around a fitted point.
    ///
    /// The returned NumPy array has shape ``(n_walkers, n_parameters)``.
    /// Per-parameter scales declared on model parameters are used when
    /// available; otherwise the characteristic scale is
    /// ``max(abs(center), 1)``. Periodic coordinates are wrapped and bounded
    /// proposals are resampled.
    ///
    /// Parameters
    /// ----------
    /// center : sequence, numpy.ndarray, or dict[str, float]
    ///     Center in free-parameter coordinates.
    /// n_walkers : int
    ///     Number of walkers. AIES requires at least two.
    /// scale : float, default=1e-3
    ///     Fractional half-width of the uniform cloud.
    /// seed : int, default=0
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Initial walker matrix suitable for ``ganesh.AIESInit``.
    fn walker_positions<'py>(
        &self,
        py: Python<'py>,
        center: &Bound<'_, PyAny>,
        n_walkers: usize,
        scale: f64,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if n_walkers < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_walkers must be at least 2",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "scale must be finite and positive",
            ));
        }
        let center = free_values(&self.inner, center)?;
        self.inner
            .params()
            .validate_free_values(&center)
            .map_err(to_py_err)?;
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut rows = Vec::with_capacity(n_walkers);
        for _ in 0..n_walkers {
            let mut row = center.clone();
            for (index, id) in self.inner.params().free_params().iter().enumerate() {
                let parameter = self.inner.params().spec(*id).map_err(to_py_err)?;
                let width = scale
                    * parameter
                        .scale()
                        .unwrap_or_else(|| center[index].abs().max(1.0));
                let mut accepted = None;
                for _ in 0..128 {
                    let candidate = center[index] + width * (2.0 * rng.f64() - 1.0);
                    if parameter.bounds_spec().contains(candidate) {
                        accepted = Some(candidate);
                        break;
                    }
                }
                row[index] = accepted.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "could not generate a bounded walker coordinate for `{}`; reduce scale or move the center away from the boundary",
                        parameter.name()
                    ))
                })?;
            }
            row = self
                .inner
                .params()
                .wrap_periodic_free_values(&row)
                .map_err(to_py_err)?;
            rows.push(row);
        }
        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    #[pyo3(signature = (
        initial: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        *,
        config: "object | None" = None,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new()
    ))]
    /// Minimize this likelihood with a deterministic optimizer.
    ///
    /// Parameters
    /// ----------
    /// initial : sequence, numpy.ndarray, or dict[str, float], optional
    ///     Initial free-parameter values. A mapping may specify only the names
    ///     to override; remaining parameters use their defaults.
    /// config : ganesh.NelderMeadConfig or ganesh.LBFGSBConfig, optional
    ///     Optimizer configuration. ``None`` uses the default L-BFGS-B
    ///     configuration. L-BFGS-B uses analytic gradients and the bounds
    ///     declared by model parameters.
    /// terminators : sequence, optional
    ///     ganesh termination callbacks, such as ``ganesh.MaxSteps``.
    /// observers : sequence, optional
    ///     ganesh observers or Python callables invoked during minimization.
    ///
    /// Returns
    /// -------
    /// ganesh.MinimizationSummary
    ///     Final parameter vector, objective value, and convergence metadata.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the configuration, initial values, or callbacks are unsupported.
    /// LadduError
    ///     If parameter transformation or likelihood evaluation fails.
    fn fit(
        &self,
        py: Python<'_>,
        initial: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
    ) -> PyResult<PyMinimizationSummary> {
        let initial = initial_vector(&self.inner, initial)?;
        let problem = OwnedProblem::new(Arc::clone(&self.inner));
        if let Some(config) = config
            && let Ok(config) = config.extract::<PyRef<'_, PyNelderMeadConfig>>()
        {
            let metadata = problem.metadata();
            let config = config
                .to_rust()
                .map_err(to_py_err)?
                .with_parameter_names(metadata.parameter_names())
                .with_transform(metadata.minimizer_transform().map_err(to_py_err)?);
            let callbacks = callback_bundle::<NelderMead, _, GradientFreeStatus, _>(
                py,
                NelderMead::default_callbacks(),
                terminators,
                observers,
            )?;
            let summary = process_with_python_callbacks(
                &mut NelderMead::default(),
                &problem,
                &(),
                initial,
                config,
                callbacks,
                to_py_err,
            )?;
            return Ok(summary.into());
        }
        let config = match config {
            Some(config) => config
                .extract::<PyRef<'_, PyLBFGSBConfig>>()
                .map_err(|_| {
                    PyTypeError::new_err(
                        "fit config must be ganesh.NelderMeadConfig, ganesh.LBFGSBConfig, or None",
                    )
                })?
                .to_rust()
                .map_err(to_py_err)?,
            None => LBFGSBConfig::default(),
        };
        let metadata = problem.metadata();
        let config = config
            .with_parameter_names(metadata.parameter_names())
            .with_transform(metadata.native_transform().map_err(to_py_err)?)
            .map_err(to_py_err)?
            .with_bounds(metadata.native_bounds())
            .map_err(to_py_err)?;
        let callbacks = callback_bundle::<LBFGSB, _, GradientStatus, _>(
            py,
            LBFGSB::default_callbacks(),
            terminators,
            observers,
        )?;
        let summary = process_with_python_callbacks(
            &mut LBFGSB::default(),
            &problem,
            &(),
            initial,
            config,
            callbacks,
            to_py_err,
        )?;
        Ok(summary.into())
    }

    #[pyo3(signature = (
        initial: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        *,
        config=None,
        fraction=0.1,
        seed=0,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    /// Minimize this likelihood with stochastic Adam updates.
    ///
    /// Parameters
    /// ----------
    /// initial : sequence, numpy.ndarray, or dict[str, float], optional
    ///     Initial free-parameter values.
    /// config : ganesh.AdamConfig, optional
    ///     Adam optimizer configuration. ``None`` uses the default
    ///     configuration.
    /// fraction : float, default=0.1
    ///     Fraction of events sampled for each stochastic evaluation. Must lie
    ///     in ``(0, 1]``.
    /// seed : int, default=0
    ///     Seed for reproducible event subsampling.
    /// terminators, observers : sequence, optional
    ///     ganesh callbacks applied during optimization.
    ///
    /// Returns
    /// -------
    /// ganesh.MinimizationSummary
    ///     Final optimizer state and convergence metadata.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `fraction` is outside ``(0, 1]``.
    /// LadduError
    ///     If preparation or an objective evaluation fails.
    fn fit_stochastic(
        &self,
        py: Python<'_>,
        initial: Option<&Bound<'_, PyAny>>,
        config: Option<&PyAdamConfig>,
        fraction: f64,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
    ) -> PyResult<PyMinimizationSummary> {
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fraction must be in (0, 1]",
            ));
        }
        let initial = initial_vector(&self.inner, initial)?;
        let metadata = FitProblem::<_, f64>::new(&*self.inner);
        let config = match config {
            Some(config) => config.to_rust().map_err(to_py_err)?,
            None => AdamConfig::default(),
        }
        .with_parameter_names(metadata.parameter_names())
        .with_transform(metadata.minimizer_transform().map_err(to_py_err)?);
        let problem = OwnedStochasticProblem {
            objective: Arc::clone(&self.inner),
            fraction,
            next_seed: AtomicU64::new(seed),
        };
        let callbacks = callback_bundle::<Adam, _, GradientStatus, _>(
            py,
            Adam::default_callbacks(),
            terminators,
            observers,
        )?;
        let summary = process_with_python_callbacks(
            &mut Adam::default(),
            &problem,
            &(),
            initial,
            config,
            callbacks,
            to_py_err,
        )?;
        Ok(summary.into())
    }

    #[pyo3(signature = (
        samples,
        initial: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        *,
        config: "object | None" = None,
        seed=0,
        terminators: "Sequence[object]" = Vec::new()
    ))]
    /// Poisson-bootstrap observed datasets, refit each replica, and retain the
    /// paired likelihood and parameter draws for cross-section propagation.
    fn bootstrap_fit(
        &self,
        py: Python<'_>,
        samples: usize,
        initial: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
    ) -> PyResult<PyEnsemble> {
        let inner = Ensemble::bootstrap_fit(&self.inner, samples, seed, |replica, _| {
            let replica_python = PyLikelihood {
                inner: Arc::clone(replica),
            };
            let callbacks = terminators
                .iter()
                .map(|callback| callback.clone_ref(py))
                .collect();
            let summary = replica_python.fit(py, initial, config, callbacks, Vec::new())?;
            let summary = Py::new(py, summary)?;
            Ok(summary
                .bind(py)
                .getattr("x")?
                .extract::<PyReadonlyArray1<'_, f64>>()?
                .as_array()
                .to_vec())
        })
        .map_err(|error| match error {
            BootstrapFitError::Likelihood(error) => to_py_err(error),
            BootstrapFitError::Fit { source, .. } => source,
        })?;
        Ok(PyEnsemble { inner })
    }

    #[pyo3(signature = (
        init,
        *,
        config=None,
        seed=0,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new()
    ))]
    /// Sample the likelihood with the affine-invariant ensemble sampler.
    ///
    /// Parameters
    /// ----------
    /// init : ganesh.AIESInit
    ///     Initial walker ensemble.
    /// config : ganesh.AIESConfig, optional
    ///     Ensemble sampler configuration. ``None`` uses the default
    ///     configuration.
    /// seed : int, default=0
    ///     Random seed for proposal generation.
    /// terminators, observers : sequence, optional
    ///     ganesh callbacks applied during sampling.
    ///
    /// Returns
    /// -------
    /// ganesh.MCMCSummary
    ///     Samples, acceptance diagnostics, and final ensemble state.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If parameter transformation, initialization, or sampling fails.
    fn sample(
        &self,
        py: Python<'_>,
        init: &PyAIESInit,
        config: Option<&PyAIESConfig>,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
    ) -> PyResult<PyMCMCSummary> {
        let problem = OwnedProblem::new(Arc::clone(&self.inner));
        let metadata = problem.metadata();
        let config = match config {
            Some(config) => config.to_rust().map_err(to_py_err)?,
            None => AIESConfig::default(),
        }
        .with_parameter_names(metadata.parameter_names())
        .with_transform(metadata.sampler_transform().map_err(to_py_err)?);
        let callbacks = callback_bundle::<AIES, _, EnsembleStatus, _>(
            py,
            AIES::default_callbacks(),
            terminators,
            observers,
        )?;
        let summary = process_with_python_callbacks(
            &mut AIES::new(Some(seed)),
            &problem,
            &(),
            init.to_rust().map_err(to_py_err)?,
            config,
            callbacks,
            to_py_err,
        )?;
        Ok(summary.into())
    }
}
