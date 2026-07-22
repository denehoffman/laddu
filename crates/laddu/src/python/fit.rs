use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use laddu_fit::{
    FitError, FitProblem,
    ganesh::{
        NalgebraProvider, Vector,
        algorithms::{
            gradient::{Adam, GradientStatus, LBFGSB},
            gradient_free::{GradientFreeStatus, NelderMead},
            mcmc::{AIES, EnsembleStatus},
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
use laddu_likelihood::{Likelihood, LikelihoodEvaluation, Objective, StochasticObjective};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyAny};

use super::{
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
                "observers must be Ganesh observer objects or Python callables",
            ));
        }
    }
    Ok(bundle)
}

#[pymethods]
impl PyLikelihood {
    #[pyo3(signature = (config, *, initial=None, terminators=Vec::new(), observers=Vec::new()))]
    fn fit(
        &self,
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        initial: Option<&Bound<'_, PyAny>>,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
    ) -> PyResult<PyMinimizationSummary> {
        let initial = initial_vector(&self.inner, initial)?;
        let problem = OwnedProblem::new(Arc::clone(&self.inner));
        if let Ok(config) = config.extract::<PyRef<'_, PyNelderMeadConfig>>() {
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
        if let Ok(config) = config.extract::<PyRef<'_, PyLBFGSBConfig>>() {
            let metadata = problem.metadata();
            let config = config
                .to_rust()
                .map_err(to_py_err)?
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
            return Ok(summary.into());
        }
        Err(PyTypeError::new_err(
            "fit config must be ganesh.NelderMeadConfig or ganesh.LBFGSBConfig",
        ))
    }

    #[pyo3(signature = (config, *, initial=None, fraction=0.1, seed=0, terminators=Vec::new(), observers=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn fit_stochastic(
        &self,
        py: Python<'_>,
        config: &PyAdamConfig,
        initial: Option<&Bound<'_, PyAny>>,
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
        let config = config
            .to_rust()
            .map_err(to_py_err)?
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

    #[pyo3(signature = (config, init, *, seed=0, terminators=Vec::new(), observers=Vec::new()))]
    fn sample(
        &self,
        py: Python<'_>,
        config: &PyAIESConfig,
        init: &PyAIESInit,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
    ) -> PyResult<PyMCMCSummary> {
        let problem = OwnedProblem::new(Arc::clone(&self.inner));
        let metadata = problem.metadata();
        let config = config
            .to_rust()
            .map_err(to_py_err)?
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
