use crate::{
    likelihoods::{LikelihoodTerm, StochasticNLL},
    LikelihoodEvaluator, NLL,
};
use ganesh::traits::{CostFunction, Gradient, LogDensity};
#[cfg(feature = "python")]
use ganesh::{
    algorithms::{
        gradient::{
            Adam, AdamConfig, ConjugateGradient, ConjugateGradientConfig, GradientStatus,
            LBFGSBConfig, TrustRegion, TrustRegionConfig, LBFGSB,
        },
        gradient_free::{
            nelder_mead::NelderMeadInit, GradientFreeStatus, NelderMead, NelderMeadConfig,
        },
        mcmc::{aies::AIESInit, ess::ESSInit, AIESConfig, ESSConfig, EnsembleStatus, AIES, ESS},
        particles::{PSOConfig, Swarm, SwarmStatus, PSO},
    },
    core::{summary::HasParameterNames, Callbacks, MCMCSummary, MinimizationSummary},
    traits::{Algorithm, Observer, Status},
};
use laddu_core::{LadduError, LadduResult, ThreadPoolManager};
use nalgebra::DVector;

/// A wrapper for the requested thread-count policy used by optimization callbacks.
#[derive(Clone, Copy, Debug)]
pub struct MaybeThreadPool {
    requested_threads: Option<usize>,
}

impl MaybeThreadPool {
    #[cfg(any(feature = "python", test))]
    fn new(num_threads: usize) -> Self {
        Self {
            requested_threads: Some(num_threads),
        }
    }

    fn install<R: Send>(&self, op: impl FnOnce() -> LadduResult<R> + Send) -> LadduResult<R> {
        ThreadPoolManager::shared().install(self.requested_threads, op)?
    }
}

#[cfg(feature = "python")]
#[derive(Copy, Clone)]
struct LikelihoodTermObserver;
#[cfg(feature = "python")]
impl<A, P, S, U, E, C> Observer<A, P, S, U, E, C> for LikelihoodTermObserver
where
    A: Algorithm<P, S, U, E, Config = C>,
    P: LikelihoodTerm,
    S: Status,
{
    fn observe(
        &mut self,
        _current_step: usize,
        _algorithm: &A,
        problem: &P,
        _status: &S,
        _args: &U,
        _config: &C,
    ) {
        problem.update();
    }
}

#[cfg(feature = "python")]
fn run_minimizer<A, P, S>(
    problem: &P,
    parameter_names: &[String],
    num_threads: usize,
    init: A::Init,
    config: A::Config,
    callbacks: Callbacks<A, P, S, MaybeThreadPool, LadduError, A::Config>,
) -> LadduResult<MinimizationSummary>
where
    A: Algorithm<P, S, MaybeThreadPool, LadduError, Summary = MinimizationSummary> + Default,
    P: LikelihoodTerm,
    S: Status,
{
    let mtp = MaybeThreadPool::new(num_threads);
    Ok(A::default()
        .process(
            problem,
            &mtp,
            init,
            config,
            callbacks.with_observer(LikelihoodTermObserver),
        )?
        .with_parameter_names(parameter_names.to_vec()))
}

#[cfg(feature = "python")]
fn run_mcmc_algorithm<A, P>(
    problem: &P,
    parameter_names: &[String],
    num_threads: usize,
    init: A::Init,
    config: A::Config,
    callbacks: Callbacks<A, P, EnsembleStatus, MaybeThreadPool, LadduError, A::Config>,
) -> LadduResult<MCMCSummary>
where
    A: Algorithm<P, EnsembleStatus, MaybeThreadPool, LadduError, Summary = MCMCSummary> + Default,
    P: LikelihoodTerm,
{
    let mtp = MaybeThreadPool::new(num_threads);
    Ok(A::default()
        .process(
            problem,
            &mtp,
            init,
            config,
            callbacks.with_observer(LikelihoodTermObserver),
        )?
        .with_parameter_names(parameter_names.to_vec()))
}

impl CostFunction<MaybeThreadPool, LadduError> for NLL {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))
    }
}
impl Gradient<MaybeThreadPool, LadduError> for NLL {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        args.install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for NLL {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        Ok(-args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
    }
}

impl CostFunction<MaybeThreadPool, LadduError> for StochasticNLL {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))
    }
}
impl Gradient<MaybeThreadPool, LadduError> for StochasticNLL {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        args.install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for StochasticNLL {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        Ok(-args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
    }
}

impl CostFunction<MaybeThreadPool, LadduError> for LikelihoodEvaluator {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))
    }
}
impl Gradient<MaybeThreadPool, LadduError> for LikelihoodEvaluator {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        args.install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for LikelihoodEvaluator {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        Ok(-args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
    }
}

/// Python bindings for the [`ganesh`] crate
#[cfg(feature = "python")]
pub mod py_ganesh {
    use std::{ops::ControlFlow, sync::Arc};

    use super::*;

    use ganesh::{
        algorithms::{
            mcmc::{integrated_autocorrelation_times, Walker},
            particles::SwarmParticle,
        },
        core::CtrlCAbortSignal,
        python::{
            PyAIESOptions as GaneshPyAIESOptions, PyAdamOptions as GaneshPyAdamOptions,
            PyConjugateGradientOptions as GaneshPyConjugateGradientOptions,
            PyESSOptions as GaneshPyESSOptions, PyLBFGSBOptions as GaneshPyLBFGSBOptions,
            PyNelderMeadOptions as GaneshPyNelderMeadOptions, PyPSOOptions as GaneshPyPSOOptions,
            PyTrustRegionOptions as GaneshPyTrustRegionOptions,
        },
        traits::{Observer, Status, Terminator},
    };
    use laddu_core::{f64, validate_free_parameter_len, LadduError};
    use nalgebra::DMatrix;
    use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
    use parking_lot::Mutex;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::PyList,
        Borrowed, PyErr,
    };

    fn validate_mcmc_parameter_len(walkers: &[Vec<f64>], expected_len: usize) -> PyResult<()> {
        for walker in walkers {
            validate_free_parameter_len(walker.len(), expected_len)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        Ok(())
    }

    fn normalize_method_name(method: &str) -> String {
        method
            .to_lowercase()
            .trim()
            .replace("-", "")
            .replace(" ", "")
    }

    fn extract_minimization_observers(
        observers: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<MinimizationObserver>> {
        if let Some(observers) = observers {
            if let Ok(observers) = observers.cast::<PyList>() {
                observers
                    .into_iter()
                    .map(|observer| {
                        observer.extract::<MinimizationObserver>().map_err(|_| {
                            PyValueError::new_err(
                                "The observers must be either a single MinimizationObserver or a list of MinimizationObservers.",
                            )
                        })
                    })
                    .collect()
            } else if let Ok(observer) = observers.extract::<MinimizationObserver>() {
                Ok(vec![observer])
            } else {
                Err(PyValueError::new_err(
                    "The observers must be either a single MinimizationObserver or a list of MinimizationObservers.",
                ))
            }
        } else {
            Ok(vec![])
        }
    }

    fn extract_minimization_terminators(
        terminators: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<MinimizationTerminator>> {
        if let Some(terminators) = terminators {
            if let Ok(terminators) = terminators.cast::<PyList>() {
                terminators
                    .into_iter()
                    .map(|terminator| {
                        terminator.extract::<MinimizationTerminator>().map_err(|_| {
                            PyValueError::new_err(
                                "The terminators must be either a single MinimizationTerminator or a list of MinimizationTerminators.",
                            )
                        })
                    })
                    .collect()
            } else if let Ok(terminator) = terminators.extract::<MinimizationTerminator>() {
                Ok(vec![terminator])
            } else {
                Err(PyValueError::new_err(
                    "The terminators must be either a single MinimizationTerminator or a list of MinimizationTerminators.",
                ))
            }
        } else {
            Ok(vec![])
        }
    }

    fn extract_mcmc_observers(observers: Option<Bound<'_, PyAny>>) -> PyResult<Vec<MCMCObserver>> {
        if let Some(observers) = observers {
            if let Ok(observers) = observers.cast::<PyList>() {
                observers
                    .into_iter()
                    .map(|observer| {
                        observer.extract::<MCMCObserver>().map_err(|_| {
                            PyValueError::new_err(
                                "The observers must be either a single MCMCObserver or a list of MCMCObservers.",
                            )
                        })
                    })
                    .collect()
            } else if let Ok(observer) = observers.extract::<MCMCObserver>() {
                Ok(vec![observer])
            } else {
                Err(PyValueError::new_err(
                    "The observers must be either a single MCMCObserver or a list of MCMCObservers.",
                ))
            }
        } else {
            Ok(vec![])
        }
    }

    fn extract_mcmc_terminators(
        terminators: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<MCMCTerminator>> {
        if let Some(terminators) = terminators {
            if let Ok(terminators) = terminators.cast::<PyList>() {
                terminators
                    .into_iter()
                    .map(|terminator| terminator.extract::<MCMCTerminator>().map_err(|_| {
                        PyValueError::new_err(
                            "The terminators must be either a single MCMCTerminator or a list of MCMCTerminators.",
                        )
                    }))
                    .collect()
            } else if let Ok(terminator) = terminators.extract::<MCMCTerminator>() {
                Ok(vec![terminator])
            } else {
                Err(PyValueError::new_err(
                    "The terminators must be either a single MCMCTerminator or a list of MCMCTerminators.",
                ))
            }
        } else {
            Ok(vec![])
        }
    }

    fn extract_python_like<T>(value: &Bound<'_, PyAny>, error_message: &str) -> PyResult<T>
    where
        T: for<'a, 'py> FromPyObject<'a, 'py, Error = PyErr>,
    {
        value
            .extract::<T>()
            .map_err(|_| PyTypeError::new_err(error_message.to_string()))
    }

    fn extract_optional_python_like<T>(
        value: Option<&Bound<'_, PyAny>>,
        error_message: &str,
    ) -> PyResult<T>
    where
        T: Default + for<'a, 'py> FromPyObject<'a, 'py, Error = PyErr>,
    {
        match value {
            Some(value) => extract_python_like(value, error_message),
            None => Ok(T::default()),
        }
    }

    pub(crate) fn minimize_from_python<P>(
        problem: &P,
        p0: &Bound<'_, PyAny>,
        n_free: usize,
        parameter_names: &[String],
        method: String,
        config: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<MinimizationSummary>
    where
        P: Gradient<MaybeThreadPool, LadduError>
            + CostFunction<MaybeThreadPool, LadduError>
            + LikelihoodTerm,
    {
        let observers = extract_minimization_observers(observers)?;
        let terminators = extract_minimization_terminators(terminators)?;
        let method = normalize_method_name(&method);

        match method.as_str() {
            "lbfgsb" => {
                let init = p0.extract::<Vec<f64>>()?;
                validate_free_parameter_len(init.len(), n_free)?;
                let config = extract_optional_python_like::<LBFGSBConfig>(
                    config,
                    "config for method 'lbfgsb' must be LBFGSBConfig-compatible or None",
                )?;
                let parsed_options = extract_optional_python_like::<GaneshPyLBFGSBOptions>(
                    options,
                    "options for method 'lbfgsb' must be LBFGSBOptions-compatible or None",
                )?;
                let mut callbacks = parsed_options
                    .build_callbacks()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<LBFGSB, _, GradientStatus>(
                    problem,
                    parameter_names,
                    threads,
                    DVector::from_vec(init),
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            "adam" => {
                let init = p0.extract::<Vec<f64>>()?;
                validate_free_parameter_len(init.len(), n_free)?;
                let config = extract_optional_python_like::<AdamConfig>(
                    config,
                    "config for method 'adam' must be AdamConfig-compatible or None",
                )?;
                let parsed_options = extract_optional_python_like::<GaneshPyAdamOptions>(
                    options,
                    "options for method 'adam' must be AdamOptions-compatible or None",
                )?;
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<Adam, _, GradientStatus>(
                    problem,
                    parameter_names,
                    threads,
                    DVector::from_vec(init),
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            "conjugategradient" => {
                let init = p0.extract::<Vec<f64>>()?;
                validate_free_parameter_len(init.len(), n_free)?;
                let config = extract_optional_python_like::<ConjugateGradientConfig>(
                    config,
                    "config for method 'conjugate-gradient' must be ConjugateGradientConfig-compatible or None",
                )?;
                let parsed_options =
                    extract_optional_python_like::<GaneshPyConjugateGradientOptions>(
                        options,
                        "options for method 'conjugate-gradient' must be ConjugateGradientOptions-compatible or None",
                    )?;
                let mut callbacks = parsed_options.build_callbacks()?;
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<ConjugateGradient, _, GradientStatus>(
                    problem,
                    parameter_names,
                    threads,
                    DVector::from_vec(init),
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            "trustregion" => {
                let init = p0.extract::<Vec<f64>>()?;
                validate_free_parameter_len(init.len(), n_free)?;
                let config = extract_optional_python_like::<TrustRegionConfig>(
                    config,
                    "config for method 'trust-region' must be TrustRegionConfig-compatible or None",
                )?;
                let parsed_options = extract_optional_python_like::<GaneshPyTrustRegionOptions>(
                    options,
                    "options for method 'trust-region' must be TrustRegionOptions-compatible or None",
                )?;
                let mut callbacks = parsed_options.build_callbacks()?;
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<TrustRegion, _, GradientStatus>(
                    problem,
                    parameter_names,
                    threads,
                    DVector::from_vec(init),
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            "neldermead" => {
                let init = if let Ok(init) = p0.extract::<NelderMeadInit>() {
                    init
                } else if let Ok(init) = p0.extract::<Vec<f64>>() {
                    validate_free_parameter_len(init.len(), n_free)?;
                    NelderMeadInit::new(init)
                } else {
                    return Err(PyTypeError::new_err(
                        "p0 for method 'nelder-mead' must be NelderMeadInit-compatible or a point",
                    ));
                };
                let config = extract_optional_python_like::<NelderMeadConfig>(
                    config,
                    "config for method 'nelder-mead' must be NelderMeadConfig-compatible or None",
                )?;
                let parsed_options = extract_optional_python_like::<GaneshPyNelderMeadOptions>(
                    options,
                    "options for method 'nelder-mead' must be NelderMeadOptions-compatible or None",
                )?;
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<NelderMead, _, GradientFreeStatus>(
                    problem,
                    parameter_names,
                    threads,
                    init,
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            "pso" => {
                let init = if let Ok(init) = p0.extract::<ganesh::algorithms::particles::Swarm>() {
                    init
                } else if let Ok(positions) = p0.extract::<Vec<Vec<f64>>>() {
                    validate_mcmc_parameter_len(&positions, n_free)?;
                    ganesh::algorithms::particles::Swarm::new(
                        ganesh::algorithms::particles::SwarmPositionInitializer::Custom(
                            positions.into_iter().map(DVector::from_vec).collect(),
                        ),
                    )
                } else {
                    return Err(PyTypeError::new_err(
                        "p0 for method 'pso' must be PSOInit-compatible or a position matrix",
                    ));
                };
                let config = extract_optional_python_like::<PSOConfig>(
                    config,
                    "config for method 'pso' must be PSOConfig-compatible or None",
                )?;
                let parsed_options = extract_optional_python_like::<GaneshPyPSOOptions>(
                    options,
                    "options for method 'pso' must be PSOOptions-compatible or None",
                )?;
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<PSO, _, SwarmStatus>(
                    problem,
                    parameter_names,
                    threads,
                    init,
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid minimizer: {method}"
            ))),
        }
    }

    pub(crate) fn mcmc_from_python<P>(
        problem: &P,
        p0: &Bound<'_, PyAny>,
        n_free: usize,
        parameter_names: &[String],
        method: String,
        config: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<MCMCSummary>
    where
        P: LogDensity<MaybeThreadPool, LadduError> + LikelihoodTerm,
    {
        let observers = extract_mcmc_observers(observers)?;
        let terminators = extract_mcmc_terminators(terminators)?;
        let method = normalize_method_name(&method);

        match method.as_str() {
            "aies" => {
                let config = extract_optional_python_like::<AIESConfig>(
                    config,
                    "config for method 'aies' must be AIESConfig-compatible or None",
                )?;
                let parsed_options = extract_optional_python_like::<GaneshPyAIESOptions>(
                    options,
                    "options for method 'aies' must be AIESOptions-compatible or None",
                )?;
                let init = if let Ok(init) = p0.extract::<AIESInit>() {
                    init
                } else if let Ok(walkers) = p0.extract::<Vec<Vec<f64>>>() {
                    validate_mcmc_parameter_len(&walkers, n_free)?;
                    AIESInit::new(walkers.into_iter().map(DVector::from_vec).collect())
                        .map_err(|err| PyValueError::new_err(err.to_string()))?
                } else {
                    return Err(PyTypeError::new_err(
                        "p0 for method 'aies' must be AIESInit-compatible or a walker matrix",
                    ));
                };
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_mcmc_algorithm::<AIES, _>(
                    problem,
                    parameter_names,
                    threads,
                    init,
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            "ess" => {
                let config = extract_optional_python_like::<ESSConfig>(
                    config,
                    "config for method 'ess' must be ESSConfig-compatible or None",
                )?;
                let parsed_options = extract_optional_python_like::<GaneshPyESSOptions>(
                    options,
                    "options for method 'ess' must be ESSOptions-compatible or None",
                )?;
                let init = if let Ok(init) = p0.extract::<ESSInit>() {
                    init
                } else if let Ok(walkers) = p0.extract::<Vec<Vec<f64>>>() {
                    validate_mcmc_parameter_len(&walkers, n_free)?;
                    ESSInit::new(walkers.into_iter().map(DVector::from_vec).collect())
                        .map_err(|err| PyValueError::new_err(err.to_string()))?
                } else {
                    return Err(PyTypeError::new_err(
                        "p0 for method 'ess' must be ESSInit-compatible or a walker matrix",
                    ));
                };
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_mcmc_algorithm::<ESS, _>(
                    problem,
                    parameter_names,
                    threads,
                    init,
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid MCMC algorithm: {method}"
            ))),
        }
    }

    /// A swarm of particles used in particle swarm optimization.
    ///
    #[pyclass(name = "Swarm", module = "laddu")]
    pub struct PySwarm(Swarm);

    #[pymethods]
    impl PySwarm {
        /// The particles in the swarm.
        ///
        /// Returns
        /// -------
        /// list of SwarmParticle
        ///
        #[getter]
        fn particles(&self) -> Vec<PySwarmParticle> {
            self.0
                .get_particles()
                .into_iter()
                .map(PySwarmParticle)
                .collect()
        }
    }

    /// A particle in a swarm used in particle swarm optimization.
    ///
    #[pyclass(name = "SwarmParticle", module = "laddu")]
    pub struct PySwarmParticle(SwarmParticle);

    #[pymethods]
    impl PySwarmParticle {
        /// The position of the particle.
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.position.x.as_slice().to_pyarray(py)
        }
        /// The evaluation of the objective function at the particle's position.
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn fx(&self) -> f64 {
            self.0.position.fx.unwrap_or(f64::NAN)
        }
        /// The best position found by the particle.
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x_best<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.best.x.as_slice().to_pyarray(py)
        }
        /// The evaluation of the objective function at the particle's best position.
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn fx_best(&self) -> f64 {
            self.0.best.fx.unwrap_or(f64::NAN)
        }
        /// The velocity vector of the particle.
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.velocity.as_slice().to_pyarray(py)
        }
    }

    /// Gradient-based minimization status passed to Python observers and terminators.
    #[pyclass(name = "GradientStatus", module = "laddu")]
    pub struct PyGradientStatus(Arc<Mutex<GradientStatus>>);

    #[pymethods]
    impl PyGradientStatus {
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.lock().x.as_slice().to_pyarray(py)
        }
        #[getter]
        fn fx(&self) -> f64 {
            self.0.lock().fx
        }
        #[getter]
        fn message(&self) -> String {
            self.0.lock().message().to_string()
        }
        #[getter]
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
            self.0
                .lock()
                .err
                .clone()
                .map(|err| err.as_slice().to_pyarray(py))
        }
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.lock().n_f_evals
        }
        #[getter]
        fn n_g_evals(&self) -> usize {
            self.0.lock().n_g_evals
        }
        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            self.0.lock().cov.clone().map(|cov| cov.to_pyarray(py))
        }
        #[getter]
        fn hess<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            self.0.lock().hess.clone().map(|hess| hess.to_pyarray(py))
        }
        #[getter]
        fn converged(&self) -> bool {
            self.0.lock().success()
        }
    }

    /// Gradient-free minimization status passed to Python observers and terminators.
    #[pyclass(name = "GradientFreeStatus", module = "laddu")]
    pub struct PyGradientFreeStatus(Arc<Mutex<GradientFreeStatus>>);

    #[pymethods]
    impl PyGradientFreeStatus {
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.lock().x.as_slice().to_pyarray(py)
        }
        #[getter]
        fn fx(&self) -> f64 {
            self.0.lock().fx
        }
        #[getter]
        fn message(&self) -> String {
            self.0.lock().message().to_string()
        }
        #[getter]
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
            self.0
                .lock()
                .err
                .clone()
                .map(|err| err.as_slice().to_pyarray(py))
        }
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.lock().n_f_evals
        }
        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            self.0.lock().cov.clone().map(|cov| cov.to_pyarray(py))
        }
        #[getter]
        fn hess<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            self.0.lock().hess.clone().map(|hess| hess.to_pyarray(py))
        }
        #[getter]
        fn converged(&self) -> bool {
            self.0.lock().success()
        }
    }

    /// Particle-swarm minimization status passed to Python observers and terminators.
    #[pyclass(name = "SwarmStatus", module = "laddu")]
    pub struct PySwarmStatus(Arc<Mutex<SwarmStatus>>);

    #[pymethods]
    impl PySwarmStatus {
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.lock().gbest.x.as_slice().to_pyarray(py)
        }
        #[getter]
        fn fx(&self) -> f64 {
            self.0.lock().gbest.fx.unwrap_or(f64::NAN)
        }
        #[getter]
        fn message(&self) -> String {
            self.0.lock().message().to_string()
        }
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.lock().n_f_evals
        }
        #[getter]
        fn converged(&self) -> bool {
            self.0.lock().success()
        }
        #[getter]
        fn swarm(&self) -> PySwarm {
            PySwarm(self.0.lock().swarm.clone())
        }
    }

    /// An enum used by a terminator to continue or stop an algorithm.
    ///
    #[pyclass(eq, eq_int, name = "ControlFlow", module = "laddu", from_py_object)]
    #[derive(PartialEq, Clone)]
    pub enum PyControlFlow {
        /// Continue running the algorithm.
        Continue = 0,
        /// Terminate the algorithm.
        Break = 1,
    }

    impl From<PyControlFlow> for ControlFlow<()> {
        fn from(v: PyControlFlow) -> Self {
            match v {
                PyControlFlow::Continue => ControlFlow::Continue(()),
                PyControlFlow::Break => ControlFlow::Break(()),
            }
        }
    }

    /// An [`Observer`] which can be used to monitor the progress of a minimization.
    ///
    /// This should be paired with a Python object which has an `observe` method
    /// that takes the current step and a method-specific minimization status object.
    #[derive(Clone)]
    pub struct MinimizationObserver(Arc<Py<PyAny>>);
    impl<'a, 'py> FromPyObject<'a, 'py> for MinimizationObserver {
        type Error = PyErr;
        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(MinimizationObserver(Arc::new(ob.to_owned().unbind())))
        }
    }
    impl<A, P, C> Observer<A, P, GradientStatus, MaybeThreadPool, LadduError, C>
        for MinimizationObserver
    where
        A: Algorithm<P, GradientStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn observe(
            &mut self,
            current_step: usize,
            _algorithm: &A,
            _problem: &P,
            status: &GradientStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) {
            Python::attach(|py| {
                if let Err(err) = self.0.bind(py).call_method1(
                    "observe",
                    (
                        current_step,
                        PyGradientStatus(Arc::new(Mutex::new(status.clone()))),
                    ),
                ) {
                    err.print(py);
                }
            })
        }
    }
    impl<A, P, C> Observer<A, P, GradientFreeStatus, MaybeThreadPool, LadduError, C>
        for MinimizationObserver
    where
        A: Algorithm<P, GradientFreeStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn observe(
            &mut self,
            current_step: usize,
            _algorithm: &A,
            _problem: &P,
            status: &GradientFreeStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) {
            Python::attach(|py| {
                if let Err(err) = self.0.bind(py).call_method1(
                    "observe",
                    (
                        current_step,
                        PyGradientFreeStatus(Arc::new(Mutex::new(status.clone()))),
                    ),
                ) {
                    err.print(py);
                }
            })
        }
    }
    impl<A, P, C> Observer<A, P, SwarmStatus, MaybeThreadPool, LadduError, C> for MinimizationObserver
    where
        A: Algorithm<P, SwarmStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn observe(
            &mut self,
            current_step: usize,
            _algorithm: &A,
            _problem: &P,
            status: &SwarmStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) {
            Python::attach(|py| {
                if let Err(err) = self.0.bind(py).call_method1(
                    "observe",
                    (
                        current_step,
                        PySwarmStatus(Arc::new(Mutex::new(status.clone()))),
                    ),
                ) {
                    err.print(py);
                }
            })
        }
    }

    /// An [`Terminator`] which can be used to monitor the progress of a minimization.
    ///
    /// This should be paired with a Python object which has an `check_for_termination` method
    /// that takes the current step and a method-specific minimization status object and returns a
    /// [`PyControlFlow`].
    #[derive(Clone)]
    pub struct MinimizationTerminator(Arc<Py<PyAny>>);

    impl<'a, 'py> FromPyObject<'a, 'py> for MinimizationTerminator {
        type Error = PyErr;
        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(MinimizationTerminator(Arc::new(ob.to_owned().unbind())))
        }
    }

    impl<A, P, C> Terminator<A, P, GradientStatus, MaybeThreadPool, LadduError, C>
        for MinimizationTerminator
    where
        A: Algorithm<P, GradientStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn check_for_termination(
            &mut self,
            current_step: usize,
            _algorithm: &mut A,
            _problem: &P,
            status: &mut GradientStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) -> ControlFlow<()> {
            Python::attach(|py| -> PyResult<ControlFlow<()>> {
                let wrapped_status = Arc::new(Mutex::new(std::mem::take(status)));
                let py_status = Py::new(py, PyGradientStatus(wrapped_status.clone()))?;
                let call_result = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status));
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
                let ret = call_result?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .unwrap_or(ControlFlow::Continue(()))
        }
    }
    impl<A, P, C> Terminator<A, P, GradientFreeStatus, MaybeThreadPool, LadduError, C>
        for MinimizationTerminator
    where
        A: Algorithm<P, GradientFreeStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn check_for_termination(
            &mut self,
            current_step: usize,
            _algorithm: &mut A,
            _problem: &P,
            status: &mut GradientFreeStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) -> ControlFlow<()> {
            Python::attach(|py| -> PyResult<ControlFlow<()>> {
                let wrapped_status = Arc::new(Mutex::new(std::mem::take(status)));
                let py_status = Py::new(py, PyGradientFreeStatus(wrapped_status.clone()))?;
                let call_result = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status));
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
                let ret = call_result?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .unwrap_or(ControlFlow::Continue(()))
        }
    }
    impl<A, P, C> Terminator<A, P, SwarmStatus, MaybeThreadPool, LadduError, C>
        for MinimizationTerminator
    where
        A: Algorithm<P, SwarmStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn check_for_termination(
            &mut self,
            current_step: usize,
            _algorithm: &mut A,
            _problem: &P,
            status: &mut SwarmStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) -> ControlFlow<()> {
            Python::attach(|py| -> PyResult<ControlFlow<()>> {
                let wrapped_status = Arc::new(Mutex::new(std::mem::take(status)));
                let py_status = Py::new(py, PySwarmStatus(wrapped_status.clone()))?;
                let call_result = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status));
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
                let ret = call_result?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .unwrap_or(ControlFlow::Continue(()))
        }
    }

    /// A walker in an MCMC ensemble.
    ///
    #[pyclass(name = "Walker", module = "laddu")]
    pub struct PyWalker(pub Walker);

    #[pymethods]
    impl PyWalker {
        /// The dimension of the walker's space (n_steps, n_variables)
        ///
        /// Returns
        /// -------
        /// tuple of int
        #[getter]
        fn dimension(&self) -> (usize, usize) {
            self.0.dimension()
        }
        /// Retrieve the latest point and the latest objective value of the Walker.
        ///
        fn get_latest<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray1<f64>>, f64) {
            let point = self.0.get_latest();
            (
                point.x.clone().as_slice().to_pyarray(py),
                point.fx_checked(),
            )
        }
    }

    /// The intermediate status used to inform the user of the current state of an MCMC algorithm.
    ///
    #[pyclass(name = "EnsembleStatus", module = "laddu")]
    pub struct PyEnsembleStatus(Arc<Mutex<EnsembleStatus>>);

    #[pymethods]
    impl PyEnsembleStatus {
        /// A message indicating the current state of the minimization.
        ///
        /// Returns
        /// -------
        /// str
        ///
        #[getter]
        fn message(&self) -> String {
            self.0.lock().message().to_string()
        }
        /// The number of objective function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.lock().n_f_evals
        }
        /// The number of gradient function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_g_evals(&self) -> usize {
            self.0.lock().n_g_evals
        }
        /// The walkers in the ensemble.
        ///
        /// Returns
        /// -------
        /// list of Walker
        ///
        #[getter]
        fn walkers(&self) -> Vec<PyWalker> {
            self.0
                .lock()
                .walkers
                .iter()
                .map(|w| PyWalker(w.clone()))
                .collect()
        }
        /// The dimension of the ensemble `(n_walkers, n_steps, n_variables)`.
        ///
        /// Returns
        /// -------
        /// tuple of int
        ///
        #[getter]
        fn dimension(&self) -> (usize, usize, usize) {
            self.0.lock().dimension()
        }

        /// Retrieve the chain of the MCMC sampling.
        ///
        /// Parameters
        /// ----------
        /// burn : int, optional
        ///     The number of steps to discard from the beginning of the chain.
        /// thin : int, optional
        ///     The number of steps to skip between samples.
        ///
        /// Returns
        /// -------
        /// chain : array of shape (n_steps, n_variables, n_walkers)
        ///
        #[pyo3(signature = (*, burn = None, thin = None))]
        fn get_chain<'py>(
            &self,
            py: Python<'py>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray3<f64>>> {
            let vec_chain: Vec<Vec<Vec<f64>>> = self
                .0
                .lock()
                .get_chain(burn, thin)
                .iter()
                .map(|steps| steps.iter().map(|p| p.as_slice().to_vec()).collect())
                .collect();
            Ok(PyArray3::from_vec3(py, &vec_chain)?)
        }

        /// Retrieve the chain of the MCMC sampling, flattened over walkers.
        ///
        /// Parameters
        /// ----------
        /// burn : int, optional
        ///     The number of steps to discard from the beginning of the chain.
        /// thin : int, optional
        ///     The number of steps to skip between samples.
        ///
        /// Returns
        /// -------
        /// flat_chain : array of shape (n_steps * n_walkers, n_variables)
        ///
        #[pyo3(signature = (*, burn = None, thin = None))]
        fn get_flat_chain<'py>(
            &self,
            py: Python<'py>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> Bound<'py, PyArray2<f64>> {
            DMatrix::from_columns(&self.0.lock().get_flat_chain(burn, thin))
                .transpose()
                .to_pyarray(py)
        }
    }

    /// An [`Observer`] which can be used to monitor the progress of an MCMC algorithm.
    ///
    /// This should be paired with a Python object which has an `observe` method
    /// that takes the current step and a [`PyEnsembleStatus`] as arguments.
    #[derive(Clone)]
    pub struct MCMCObserver(Arc<Py<PyAny>>);

    impl<'a, 'py> FromPyObject<'a, 'py> for MCMCObserver {
        type Error = PyErr;
        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(MCMCObserver(Arc::new(ob.to_owned().unbind())))
        }
    }

    impl<A, P, C> Observer<A, P, EnsembleStatus, MaybeThreadPool, LadduError, C> for MCMCObserver
    where
        A: Algorithm<P, EnsembleStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn observe(
            &mut self,
            current_step: usize,
            _algorithm: &A,
            _problem: &P,
            status: &EnsembleStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) {
            Python::attach(|py| {
                if let Err(err) = self.0.bind(py).call_method1(
                    "observe",
                    (
                        current_step,
                        PyEnsembleStatus(Arc::new(Mutex::new(status.clone()))),
                    ),
                ) {
                    err.print(py);
                }
            })
        }
    }

    /// A [`Terminator`] which can be used to monitor the progress of an MCMC algorithm.
    ///
    /// This should be paired with a Python object which has an `check_for_termination` method
    /// that takes the current step and a [`PyEnsembleStatus`] as arguments and returns a
    /// [`PyControlFlow`].
    #[derive(Clone)]
    pub struct MCMCTerminator(Arc<Py<PyAny>>);

    impl<'a, 'py> FromPyObject<'a, 'py> for MCMCTerminator {
        type Error = PyErr;
        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(MCMCTerminator(Arc::new(ob.to_owned().unbind())))
        }
    }

    impl<A, P, C> Terminator<A, P, EnsembleStatus, MaybeThreadPool, LadduError, C> for MCMCTerminator
    where
        A: Algorithm<P, EnsembleStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn check_for_termination(
            &mut self,
            current_step: usize,
            _algorithm: &mut A,
            _problem: &P,
            status: &mut EnsembleStatus,
            _args: &MaybeThreadPool,
            _config: &C,
        ) -> ControlFlow<()> {
            Python::attach(|py| -> PyResult<ControlFlow<()>> {
                let wrapped_status = Arc::new(Mutex::new(std::mem::take(status)));
                let py_status = Py::new(py, PyEnsembleStatus(wrapped_status.clone()))?;
                let call_result = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status));
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
                let ret = call_result?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .unwrap_or(ControlFlow::Continue(()))
        }
    }

    /// Calculate the integrated autocorrelation time for each parameter according to
    /// Karamanis & Beutler (2021).
    ///
    /// Parameters
    /// ----------
    /// samples : array_like
    ///     An array of dimension ``(n_walkers, n_steps, n_parameters)``.
    /// c : float, default = 7.0
    ///     The time window for Sokal's autowindowing function. If omitted, the
    ///     default window size of 7.0 is used.
    ///
    /// Returns
    /// -------
    /// array of shape (n_parameters,)
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> from laddu import integrated_autocorrelation_times
    /// >>> samples = np.random.randn(4, 16, 2).tolist()
    /// >>> integrated_autocorrelation_times(samples).shape
    /// (2,)
    ///
    /// References
    /// ----------
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat. Comput. 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
    ///
    /// Sokal, A. (1997). *Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms*. NATO ASI Series, 131–192. <https://doi.org/10.1007/978-1-4899-0319-8_6>
    #[pyfunction(name = "integrated_autocorrelation_times")]
    #[pyo3(signature = (samples, *, c=None))]
    pub fn py_integrated_autocorrelation_times<'py>(
        py: Python<'py>,
        samples: Vec<Vec<Vec<f64>>>,
        c: Option<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let samples: Vec<Vec<DVector<f64>>> = samples
            .into_iter()
            .map(|walker| walker.into_iter().map(DVector::from_vec).collect())
            .collect();
        integrated_autocorrelation_times(samples, c)
            .as_slice()
            .to_pyarray(py)
    }
}

#[cfg(test)]
mod tests {
    use super::MaybeThreadPool;

    #[test]
    fn maybe_thread_pool_handles_repeated_short_installs() {
        let pool = MaybeThreadPool::new(2);
        let total = (0usize..64)
            .map(|index| {
                pool.install(|| Ok(index + 1))
                    .expect("repeated install should succeed")
            })
            .sum::<usize>();
        assert_eq!(total, (1usize..=64).sum::<usize>());
    }
}
