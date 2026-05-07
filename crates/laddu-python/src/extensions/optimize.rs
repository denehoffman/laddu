use ganesh::{
    algorithms::{
        gradient::{
            Adam, AdamConfig, ConjugateGradient, ConjugateGradientConfig, GradientStatus,
            LBFGSBConfig, TrustRegion, TrustRegionConfig, LBFGSB,
        },
        gradient_free::{
            nelder_mead::NelderMeadInit, CMAESConfig, CMAESInit, DifferentialEvolution,
            DifferentialEvolutionConfig, DifferentialEvolutionInit, GradientFreeStatus, NelderMead,
            NelderMeadConfig, CMAES,
        },
        mcmc::{aies::AIESInit, ess::ESSInit, AIESConfig, ESSConfig, EnsembleStatus, AIES, ESS},
        particles::{PSOConfig, SwarmStatus, PSO},
    },
    core::{Callbacks, CtrlCAbortSignal, MCMCSummary, MinimizationSummary},
    python::{
        PyAIESOptions, PyAdamOptions, PyCMAESOptions, PyConjugateGradientOptions,
        PyDifferentialEvolutionOptions, PyESSOptions, PyLBFGSBOptions, PyNelderMeadOptions,
        PyPSOOptions, PyTrustRegionOptions,
    },
    traits::{Algorithm, CostFunction, Gradient, LogDensity, Status, SupportsParameterNames},
    DVector,
};
use laddu_core::{f64, validate_free_parameter_len, LadduError, LadduResult};
use laddu_extensions::{
    likelihood::LikelihoodTerm, optimize::MaybeThreadPool, LikelihoodTermObserver,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyList,
    PyErr,
};

use crate::extensions::callbacks::{
    MCMCObserver, MCMCTerminator, MinimizationObserver, MinimizationTerminator,
};

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
    A::default().process(
        problem,
        &mtp,
        init,
        config,
        callbacks.with_observer(LikelihoodTermObserver),
    )
}

fn run_mcmc_algorithm<A, P>(
    problem: &P,
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
    A::default().process(
        problem,
        &mtp,
        init,
        config,
        callbacks.with_observer(LikelihoodTermObserver),
    )
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

#[allow(clippy::too_many_arguments)]
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
                .map(|opt| opt.extract::<PyLBFGSBOptions>())
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
                .map(|opt| opt.extract::<PyAdamOptions>())
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
                .map(|opt| opt.extract::<PyConjugateGradientOptions>())
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
                .map(|opt| opt.extract::<PyTrustRegionOptions>())
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
                .map(|opt| opt.extract::<PyCMAESOptions>())
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
            run_minimizer::<CMAES, _, GradientFreeStatus>(problem, threads, init, config, callbacks)
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
                .map(|opt| opt.extract::<PyDifferentialEvolutionOptions>())
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
                .map(|opt| opt.extract::<PyNelderMeadOptions>())
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
                .map(|opt| opt.extract::<PyPSOOptions>())
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

#[allow(clippy::too_many_arguments)]
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
                .map(|opt| opt.extract::<PyAIESOptions>())
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
                .map(|opt| opt.extract::<PyESSOptions>())
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
