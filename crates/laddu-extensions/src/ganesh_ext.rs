use ganesh::traits::{Algorithm, CostFunction, Gradient, LogDensity, Observer, Status};
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
        particles::{PSOConfig, SwarmStatus, PSO},
    },
    core::{Callbacks, MCMCSummary, MinimizationSummary},
};
use laddu_core::{LadduError, LadduResult, ThreadPoolManager};
use nalgebra::DVector;

use crate::{
    likelihoods::{LikelihoodTerm, StochasticNLL},
    LikelihoodExpression, NLL,
};

/// A wrapper for the requested thread-count policy used by optimization callbacks.
#[derive(Clone, Copy, Debug)]
pub struct MaybeThreadPool {
    requested_threads: Option<usize>,
}

/// An observer which calls [`LikelihoodTerm::update`] on each step of the algorithm.
///
/// This should generally be used with any algorithm, but it mostly impacts [`StochasticNLL`] terms
/// which need to update random state at each step.
#[derive(Copy, Clone)]
pub struct LikelihoodTermObserver;
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

impl CostFunction<MaybeThreadPool, LadduError> for LikelihoodExpression {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))
    }
}
impl Gradient<MaybeThreadPool, LadduError> for LikelihoodExpression {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        args.install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for LikelihoodExpression {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        Ok(-args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
    }
}

/// Python bindings for the [`ganesh`] crate
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(feature = "python")]
pub mod py_ganesh {
    use std::{ops::ControlFlow, sync::Arc};

    use ganesh::{
        algorithms::{
            gradient_free::{
                CMAESConfig, CMAESInit, DifferentialEvolution, DifferentialEvolutionConfig,
                DifferentialEvolutionInit, CMAES,
            },
            mcmc::integrated_autocorrelation_times,
        },
        core::CtrlCAbortSignal,
        python::{
            PyAIESOptions as GaneshPyAIESOptions, PyAdamOptions as GaneshPyAdamOptions,
            PyCMAESOptions as GaneshPyCMAESOptions,
            PyConjugateGradientOptions as GaneshPyConjugateGradientOptions,
            PyDifferentialEvolutionOptions as GaneshPyDifferentialEvolutionOptions,
            PyESSOptions as GaneshPyESSOptions, PyEnsembleStatus as GaneshPyEnsembleStatus,
            PyGradientFreeStatus as GaneshPyGradientFreeStatus,
            PyGradientStatus as GaneshPyGradientStatus, PyLBFGSBOptions as GaneshPyLBFGSBOptions,
            PyNelderMeadOptions as GaneshPyNelderMeadOptions, PyPSOOptions as GaneshPyPSOOptions,
            PySwarmStatus as GaneshPySwarmStatus,
            PyTrustRegionOptions as GaneshPyTrustRegionOptions,
        },
        traits::{Observer, SupportsParameterNames, Terminator},
    };
    use laddu_core::{f64, validate_free_parameter_len, LadduError};
    use numpy::{PyArray1, ToPyArray};
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::PyList,
        Borrowed, PyErr,
    };

    use super::*;

    #[cfg(feature = "python")]
    fn run_minimizer<A, P, S>(
        problem: &P,
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
        Ok(A::default().process(
            problem,
            &mtp,
            init,
            config,
            callbacks.with_observer(LikelihoodTermObserver),
        )?)
    }

    #[cfg(feature = "python")]
    fn run_mcmc_algorithm<A, P>(
        problem: &P,
        num_threads: usize,
        init: A::Init,
        config: A::Config,
        callbacks: Callbacks<A, P, EnsembleStatus, MaybeThreadPool, LadduError, A::Config>,
    ) -> LadduResult<MCMCSummary>
    where
        A: Algorithm<P, EnsembleStatus, MaybeThreadPool, LadduError, Summary = MCMCSummary>
            + Default,
        P: LikelihoodTerm,
    {
        let mtp = MaybeThreadPool::new(num_threads);
        Ok(A::default().process(
            problem,
            &mtp,
            init,
            config,
            callbacks.with_observer(LikelihoodTermObserver),
        )?)
    }

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
                let mut config = config
                    .map(|c| c.extract::<LBFGSBConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyLBFGSBOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
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
                let mut config = config
                    .map(|c| c.extract::<AdamConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyAdamOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
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
                let mut config = config
                    .map(|c| c.extract::<ConjugateGradientConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyConjugateGradientOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
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
                let mut config = config
                    .map(|c| c.extract::<TrustRegionConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyTrustRegionOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
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
                    threads,
                    DVector::from_vec(init),
                    config,
                    callbacks,
                )
                .map_err(PyErr::from)
            }
            "cmaes" => {
                let init = p0.extract::<CMAESInit>()?;
                let mut config = config
                    .map(|c| c.extract::<CMAESConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyCMAESOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<CMAES, _, GradientFreeStatus>(
                    problem, threads, init, config, callbacks,
                )
                .map_err(PyErr::from)
            }
            "differential-evolution" => {
                let init = p0.extract::<DifferentialEvolutionInit>()?;
                let mut config = config
                    .map(|c| c.extract::<DifferentialEvolutionConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyDifferentialEvolutionOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<DifferentialEvolution, _, GradientFreeStatus>(
                    problem, threads, init, config, callbacks,
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
                let mut config = config
                    .map(|c| c.extract::<NelderMeadConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyNelderMeadOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<NelderMead, _, GradientFreeStatus>(
                    problem, threads, init, config, callbacks,
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
                let mut config = config
                    .map(|c| c.extract::<PSOConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyPSOOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                let mut callbacks = parsed_options.build_callbacks();
                for observer in observers {
                    callbacks = callbacks.with_observer(observer);
                }
                for terminator in terminators {
                    callbacks = callbacks.with_terminator(terminator);
                }
                callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                run_minimizer::<PSO, _, SwarmStatus>(problem, threads, init, config, callbacks)
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
                let mut config = config
                    .map(|c| c.extract::<AIESConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyAIESOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
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
                run_mcmc_algorithm::<AIES, _>(problem, threads, init, config, callbacks)
                    .map_err(PyErr::from)
            }
            "ess" => {
                let mut config = config
                    .map(|c| c.extract::<ESSConfig>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
                if config.get_parameter_names_mut().is_none() {
                    config = config.with_parameter_names(parameter_names);
                }
                let parsed_options = options
                    .map(|opt| opt.extract::<GaneshPyESSOptions>())
                    .transpose()
                    .map_err(|err| PyTypeError::new_err(err.to_string()))?
                    .unwrap_or_default();
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
                run_mcmc_algorithm::<ESS, _>(problem, threads, init, config, callbacks)
                    .map_err(PyErr::from)
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid MCMC algorithm: {method}"
            ))),
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
                        Py::new(py, GaneshPyGradientStatus::from(status.clone()))
                            .expect("ganesh gradient status should construct"),
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
                        Py::new(py, GaneshPyGradientFreeStatus::from(status.clone()))
                            .expect("ganesh gradient-free status should construct"),
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
            Python::attach(|py| -> PyResult<()> {
                if let Err(err) = self.0.bind(py).call_method1(
                    "observe",
                    (
                        current_step,
                        Py::new(py, GaneshPySwarmStatus::from(status.clone()))?,
                    ),
                ) {
                    err.print(py);
                }
                Ok(())
            })
            .expect("call to 'observe' has failed!")
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
                let py_status = Py::new(py, GaneshPyGradientStatus::from(status.clone()))?;
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .expect("call to 'check_for_termination' has failed!")
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
                let py_status = Py::new(py, GaneshPyGradientFreeStatus::from(status.clone()))?;
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .expect("call to 'check_for_termination' has failed!")
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
                let py_status = Py::new(py, GaneshPySwarmStatus::from(status.clone()))?;
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .expect("call to 'check_for_termination' has failed!")
        }
    }

    /// An [`Observer`] which can be used to monitor the progress of an MCMC algorithm.
    ///
    /// This should be paired with a Python object which has an `observe` method
    /// that takes the current step and a `ganesh.EnsembleStatus` object.
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
            Python::attach(|py| -> PyResult<()> {
                if let Err(err) = self.0.bind(py).call_method1(
                    "observe",
                    (
                        current_step,
                        Py::new(py, GaneshPyEnsembleStatus::from(status.clone()))?,
                    ),
                ) {
                    err.print(py);
                }
                Ok(())
            })
            .expect("call to 'observe' has failed!")
        }
    }

    /// A [`Terminator`] which can be used to monitor the progress of an MCMC algorithm.
    ///
    /// This should be paired with a Python object which has an `check_for_termination` method
    /// that takes the current step and a `ganesh.EnsembleStatus` object and returns a
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
                let py_status = Py::new(py, GaneshPyEnsembleStatus::from(status.clone()))?;
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))?;
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .expect("call to 'check_for_termination' has failed!")
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
