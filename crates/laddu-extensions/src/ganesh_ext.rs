use crate::{
    likelihoods::{LikelihoodTerm, StochasticNLL},
    LikelihoodEvaluator, NLL,
};
use ganesh::{
    algorithms::{
        gradient::{Adam, AdamConfig, GradientStatus, LBFGSBConfig, LBFGSB},
        gradient_free::{GradientFreeStatus, NelderMead, NelderMeadConfig},
        mcmc::{AIESConfig, ESSConfig, EnsembleStatus, AIES, ESS},
        particles::{PSOConfig, SwarmStatus, PSO},
    },
    core::{summary::HasParameterNames, Callbacks, MCMCSummary, MinimizationSummary},
    traits::{Algorithm, CostFunction, Gradient, LogDensity, Observer, Status},
};
use laddu_core::{LadduError, LadduResult};
use nalgebra::DVector;
#[cfg(feature = "rayon")]
use rayon::{ThreadPool, ThreadPoolBuilder};

/// A settings wrapper for minimization algorithms
pub enum MinimizationSettings<P> {
    /// Settings for the L-BFGS-B algorithm
    LBFGSB {
        /// The configuration struct
        config: LBFGSBConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<LBFGSB, P, GradientStatus, MaybeThreadPool, LadduError, LBFGSBConfig>,
        /// The number of threads to use (0 will run in single-threaded mode)
        num_threads: usize,
    },
    /// Settings for the Adam algorithm
    Adam {
        /// The configuration struct
        config: AdamConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<Adam, P, GradientStatus, MaybeThreadPool, LadduError, AdamConfig>,
        /// The number of threads to use (0 will run in single-threaded mode)
        num_threads: usize,
    },
    /// Settings for the Nelder-Mead algorithm
    NelderMead {
        /// The configuration struct
        config: NelderMeadConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<
            NelderMead,
            P,
            GradientFreeStatus,
            MaybeThreadPool,
            LadduError,
            NelderMeadConfig,
        >,
        /// The number of threads to use (0 will run in single-threaded mode)
        num_threads: usize,
    },
    /// Settings for the Particle Swarm Optimimization algorithm
    PSO {
        /// The configuration struct
        config: PSOConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<PSO, P, SwarmStatus, MaybeThreadPool, LadduError, PSOConfig>,
        /// The number of threads to use (0 will run in single-threaded mode)
        num_threads: usize,
    },
}

/// A settings wrapper for MCMC algorithms
pub enum MCMCSettings<P> {
    /// Settings for the Affine Invariant Ensemble Sampler
    AIES {
        /// The configuration struct
        config: AIESConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<AIES, P, EnsembleStatus, MaybeThreadPool, LadduError, AIESConfig>,
        /// The number of threads to use (0 will run in single-threaded mode)
        num_threads: usize,
    },
    /// Settings for the Ensemble Slice Sampler
    ESS {
        /// The configuration struct
        config: ESSConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<ESS, P, EnsembleStatus, MaybeThreadPool, LadduError, ESSConfig>,
        /// The number of threads to use (0 will run in single-threaded mode)
        num_threads: usize,
    },
}

/// A wrapper struct which conditionally contains a thread pool if the rayon feature is enabled.
#[derive(Debug)]
pub struct MaybeThreadPool {
    #[cfg(feature = "rayon")]
    /// The underlying [`ThreadPool`]
    pub thread_pool: ThreadPool,
}
/// A trait for objects which can be used in multithreaded contexts
pub trait Threadable {
    /// Returns a [`MaybeThreadPool`] associated with the object.
    ///
    /// # Errors
    ///
    /// This will return an error if there is any problem constructing the underlying thread pool.
    fn get_pool(&self) -> Result<MaybeThreadPool, LadduError>;
}
impl<P> Threadable for MinimizationSettings<P> {
    fn get_pool(&self) -> Result<MaybeThreadPool, LadduError> {
        #[cfg(feature = "rayon")]
        {
            Ok(MaybeThreadPool {
                thread_pool: ThreadPoolBuilder::new()
                    .num_threads(match self {
                        Self::LBFGSB {
                            config: _,
                            callbacks: _,
                            num_threads,
                        }
                        | Self::Adam {
                            config: _,
                            callbacks: _,
                            num_threads,
                        }
                        | Self::NelderMead {
                            config: _,
                            callbacks: _,
                            num_threads,
                        }
                        | Self::PSO {
                            config: _,
                            callbacks: _,
                            num_threads,
                        } => *num_threads,
                    })
                    .build()
                    .map_err(LadduError::from)?,
            })
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(MaybeThreadPool {})
        }
    }
}

impl<P> Threadable for MCMCSettings<P> {
    fn get_pool(&self) -> Result<MaybeThreadPool, LadduError> {
        #[cfg(feature = "rayon")]
        {
            Ok(MaybeThreadPool {
                thread_pool: ThreadPoolBuilder::new()
                    .num_threads(match self {
                        Self::AIES {
                            config: _,
                            callbacks: _,
                            num_threads,
                        }
                        | Self::ESS {
                            config: _,
                            callbacks: _,
                            num_threads,
                        } => *num_threads,
                    })
                    .build()
                    .map_err(LadduError::from)?,
            })
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(MaybeThreadPool {})
        }
    }
}

#[derive(Copy, Clone)]
struct LikelihoodTermObserver;
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

impl CostFunction<MaybeThreadPool, LadduError> for NLL {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        #[cfg(feature = "rayon")]
        {
            args.thread_pool
                .install(|| LikelihoodTerm::evaluate(self, parameters.into()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            LikelihoodTerm::evaluate(self, parameters.into())
        }
    }
}
impl Gradient<MaybeThreadPool, LadduError> for NLL {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        #[cfg(feature = "rayon")]
        {
            args.thread_pool
                .install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            LikelihoodTerm::evaluate_gradient(self, parameters.into())
        }
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for NLL {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        #[cfg(feature = "rayon")]
        {
            Ok(-args
                .thread_pool
                .install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(-LikelihoodTerm::evaluate(self, parameters.into())?)
        }
    }
}

impl NLL {
    /// Minimize the [`NLL`] with the algorithm given in the [`MinimizationSettings`].
    ///
    /// # Errors
    ///
    /// This method may return an error if there was any problem constructing thread pools or
    /// evaluating the underlying model.
    pub fn minimize(
        &self,
        settings: MinimizationSettings<Self>,
    ) -> Result<MinimizationSummary, LadduError> {
        let mtp = settings.get_pool()?;
        Ok(match settings {
            MinimizationSettings::LBFGSB {
                config,
                callbacks,
                num_threads: _,
            } => LBFGSB::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::Adam {
                config,
                callbacks,
                num_threads: _,
            } => Adam::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::NelderMead {
                config,
                callbacks,
                num_threads: _,
            } => NelderMead::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::PSO {
                config,
                callbacks,
                num_threads: _,
            } => PSO::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
        }?
        .with_parameter_names(self.parameters()))
    }

    /// Run an MCMC sampling algorithm over the [`NLL`] with the given [`MCMCSettings`].
    ///
    /// # Errors
    ///
    /// This method may return an error if there was any problem constructing thread pools or
    /// evaluating the underlying model.
    pub fn mcmc(&self, settings: MCMCSettings<Self>) -> Result<MCMCSummary, LadduError> {
        let mtp = settings.get_pool()?;
        Ok(match settings {
            MCMCSettings::AIES {
                config,
                callbacks,
                num_threads: _,
            } => AIES::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MCMCSettings::ESS {
                config,
                callbacks,
                num_threads: _,
            } => ESS::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
        }?
        .with_parameter_names(self.parameters()))
    }
}

impl CostFunction<MaybeThreadPool, LadduError> for StochasticNLL {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        #[cfg(feature = "rayon")]
        {
            args.thread_pool
                .install(|| LikelihoodTerm::evaluate(self, parameters.into()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            LikelihoodTerm::evaluate(self, parameters.into())
        }
    }
}
impl Gradient<MaybeThreadPool, LadduError> for StochasticNLL {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        #[cfg(feature = "rayon")]
        {
            args.thread_pool
                .install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            LikelihoodTerm::evaluate_gradient(self, parameters.into())
        }
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for StochasticNLL {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        #[cfg(feature = "rayon")]
        {
            Ok(-args
                .thread_pool
                .install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(-LikelihoodTerm::evaluate(self, parameters.into())?)
        }
    }
}

impl StochasticNLL {
    /// Minimize the [`StochasticNLL`] with the algorithm given in the [`MinimizationSettings`].
    ///
    /// # Errors
    ///
    /// This method may return an error if there was any problem constructing thread pools or
    /// evaluating the underlying model.
    pub fn minimize(
        &self,
        settings: MinimizationSettings<Self>,
    ) -> LadduResult<MinimizationSummary> {
        let mtp = settings.get_pool()?;
        Ok(match settings {
            MinimizationSettings::LBFGSB {
                config,
                callbacks,
                num_threads: _,
            } => LBFGSB::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::Adam {
                config,
                callbacks,
                num_threads: _,
            } => Adam::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::NelderMead {
                config,
                callbacks,
                num_threads: _,
            } => NelderMead::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::PSO {
                config,
                callbacks,
                num_threads: _,
            } => PSO::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
        }?
        .with_parameter_names(self.parameters()))
    }

    /// Run an MCMC sampling algorithm over the [`StochasticNLL`] with the given [`MCMCSettings`].
    ///
    /// # Errors
    ///
    /// This method may return an error if there was any problem constructing thread pools or
    /// evaluating the underlying model.
    pub fn mcmc(&self, settings: MCMCSettings<Self>) -> LadduResult<MCMCSummary> {
        let mtp = settings.get_pool()?;
        Ok(match settings {
            MCMCSettings::AIES {
                config,
                callbacks,
                num_threads: _,
            } => AIES::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MCMCSettings::ESS {
                config,
                callbacks,
                num_threads: _,
            } => ESS::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
        }?
        .with_parameter_names(self.parameters()))
    }
}

impl CostFunction<MaybeThreadPool, LadduError> for LikelihoodEvaluator {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        #[cfg(feature = "rayon")]
        {
            args.thread_pool
                .install(|| LikelihoodTerm::evaluate(self, parameters.into()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            LikelihoodTerm::evaluate(self, parameters.into())
        }
    }
}
impl Gradient<MaybeThreadPool, LadduError> for LikelihoodEvaluator {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        #[cfg(feature = "rayon")]
        {
            args.thread_pool
                .install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            LikelihoodTerm::evaluate_gradient(self, parameters.into())
        }
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for LikelihoodEvaluator {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        #[cfg(feature = "rayon")]
        {
            Ok(-args
                .thread_pool
                .install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(-LikelihoodTerm::evaluate(self, parameters.into())?)
        }
    }
}

impl LikelihoodEvaluator {
    /// Minimize the [`LikelihoodEvaluator`] with the algorithm given in the [`MinimizationSettings`].
    ///
    /// # Errors
    ///
    /// This method may return an error if there was any problem constructing thread pools or
    /// evaluating the underlying model.
    pub fn minimize(
        &self,
        settings: MinimizationSettings<Self>,
    ) -> Result<MinimizationSummary, LadduError> {
        let mtp = settings.get_pool()?;
        match settings {
            MinimizationSettings::LBFGSB {
                config,
                callbacks,
                num_threads: _,
            } => LBFGSB::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::Adam {
                config,
                callbacks,
                num_threads: _,
            } => Adam::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::NelderMead {
                config,
                callbacks,
                num_threads: _,
            } => NelderMead::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MinimizationSettings::PSO {
                config,
                callbacks,
                num_threads: _,
            } => PSO::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
        }
    }

    /// Run an MCMC sampling algorithm over the [`LikelihoodEvaluator`] with the given [`MCMCSettings`].
    ///
    /// # Errors
    ///
    /// This method may return an error if there was any problem constructing thread pools or
    /// evaluating the underlying model.
    pub fn mcmc(&self, settings: MCMCSettings<Self>) -> Result<MCMCSummary, LadduError> {
        let mtp = settings.get_pool()?;
        match settings {
            MCMCSettings::AIES {
                config,
                callbacks,
                num_threads: _,
            } => AIES::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
            MCMCSettings::ESS {
                config,
                callbacks,
                num_threads: _,
            } => ESS::default().process(
                self,
                &mtp,
                config,
                callbacks.with_observer(LikelihoodTermObserver),
            ),
        }
    }
}

/// Python bindings for the [`ganesh`] crate
#[cfg(feature = "python")]
pub mod py_ganesh {
    use std::{ops::ControlFlow, sync::Arc};

    use super::*;

    use ganesh::{
        algorithms::{
            gradient::{
                adam::AdamEMATerminator,
                lbfgsb::{
                    LBFGSBErrorMode, LBFGSBFTerminator, LBFGSBGTerminator, LBFGSBInfNormGTerminator,
                },
            },
            gradient_free::nelder_mead::{
                NelderMeadFTerminator, NelderMeadXTerminator, SimplexConstructionMethod,
                SimplexExpansionMethod,
            },
            line_search::{HagerZhangLineSearch, MoreThuenteLineSearch, StrongWolfeLineSearch},
            mcmc::{
                integrated_autocorrelation_times, AIESMove, AutocorrelationTerminator, ESSMove,
                Walker,
            },
            particles::{
                Swarm, SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmTopology,
                SwarmUpdateMethod, SwarmVelocityInitializer,
            },
        },
        core::{Bounds, CtrlCAbortSignal, DebugObserver, MaxSteps},
        traits::{Observer, Status, SupportsBounds, SupportsTransform, Terminator},
    };
    use laddu_core::{LadduError, ReadWrite};
    use nalgebra::DMatrix;
    use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
    use parking_lot::Mutex;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{PyBytes, PyDict, PyList},
    };

    /// A helper trait for parsing Python arguments.
    pub trait FromPyArgs<A = ()>: Sized {
        /// Convert the given Python arguments into a [`Self`].
        fn from_pyargs(args: &A, d: &Bound<PyDict>) -> PyResult<Self>;
    }
    impl FromPyArgs<Vec<f64>> for LBFGSBConfig {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            let mut config = LBFGSBConfig::new(args);
            if let Some(m) = d.get_item("m")? {
                let m_int = m.extract()?;
                config = config.with_memory_limit(m_int);
            }
            if let Some(flag) = d.get_item("skip_hessian")? {
                if flag.extract()? {
                    config = config.with_error_mode(LBFGSBErrorMode::Skip);
                }
            }
            if let Some(linesearch_dict) = d.get_item("line_search")? {
                config = config.with_line_search(StrongWolfeLineSearch::from_pyargs(
                    &(),
                    &linesearch_dict.extract()?,
                )?);
            }
            Ok(config)
        }
    }
    impl FromPyArgs for StrongWolfeLineSearch {
        fn from_pyargs(_args: &(), d: &Bound<PyDict>) -> PyResult<Self> {
            if let Some(method) = d.get_item("method")? {
                match method
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "morethuente" => {
                        let mut line_search = MoreThuenteLineSearch::default();
                        if let Some(max_iterations) = d.get_item("max_iterations")? {
                            line_search =
                                line_search.with_max_iterations(max_iterations.extract()?);
                        }
                        if let Some(max_zoom) = d.get_item("max_zoom")? {
                            line_search = line_search.with_max_zoom(max_zoom.extract()?);
                        }
                        match (d.get_item("c1")?, d.get_item("c2")?) {
                            (Some(c1), Some(c2)) => {
                                line_search = line_search.with_c1_c2(c1.extract()?, c2.extract()?);
                            }
                            (Some(c1), None) => {
                                line_search = line_search.with_c1(c1.extract()?);
                            }
                            (None, Some(c2)) => {
                                line_search = line_search.with_c2(c2.extract()?);
                            }
                            (None, None) => {}
                        }
                        Ok(StrongWolfeLineSearch::MoreThuente(line_search))
                    }
                    "hagerzhang" => {
                        let mut line_search = HagerZhangLineSearch::default();
                        if let Some(max_iterations) = d.get_item("max_iterations")? {
                            line_search =
                                line_search.with_max_iterations(max_iterations.extract()?);
                        }
                        if let Some(max_bisects) = d.get_item("max_bisects")? {
                            line_search = line_search.with_max_bisects(max_bisects.extract()?);
                        }
                        match (d.get_item("delta")?, d.get_item("sigma")?) {
                            (Some(delta), Some(sigma)) => {
                                line_search = line_search
                                    .with_delta_sigma(delta.extract()?, sigma.extract()?);
                            }
                            (Some(delta), None) => {
                                line_search = line_search.with_delta(delta.extract()?);
                            }
                            (None, Some(sigma)) => {
                                line_search = line_search.with_sigma(sigma.extract()?);
                            }
                            (None, None) => {}
                        }
                        if let Some(epsilon) = d.get_item("epsilon")? {
                            line_search = line_search.with_epsilon(epsilon.extract()?);
                        }
                        if let Some(theta) = d.get_item("theta")? {
                            line_search = line_search.with_theta(theta.extract()?);
                        }
                        if let Some(gamma) = d.get_item("gamma")? {
                            line_search = line_search.with_gamma(gamma.extract()?);
                        }
                        Ok(StrongWolfeLineSearch::HagerZhang(line_search))
                    }
                    _ => Err(PyTypeError::new_err(format!(
                        "Invalid line search method: {}",
                        method
                    ))),
                }
            } else {
                Err(PyTypeError::new_err("Line search method not specified"))
            }
        }
    }
    impl<P> FromPyArgs
        for Callbacks<LBFGSB, P, GradientStatus, MaybeThreadPool, LadduError, LBFGSBConfig>
    where
        P: Gradient<MaybeThreadPool, LadduError>,
    {
        fn from_pyargs(_args: &(), d: &Bound<PyDict>) -> PyResult<Self> {
            let mut callbacks = Callbacks::empty();
            if let Some(eps_f) = d.get_item("eps_f")? {
                if let Some(eps_abs) = eps_f.extract()? {
                    callbacks = callbacks.with_terminator(LBFGSBFTerminator { eps_abs });
                } else {
                    callbacks = callbacks.with_terminator(LBFGSBFTerminator::default());
                }
            } else {
                callbacks = callbacks.with_terminator(LBFGSBFTerminator::default());
            }
            if let Some(eps_g) = d.get_item("eps_g")? {
                if let Some(eps_abs) = eps_g.extract()? {
                    callbacks = callbacks.with_terminator(LBFGSBGTerminator { eps_abs });
                } else {
                    callbacks = callbacks.with_terminator(LBFGSBGTerminator::default());
                }
            } else {
                callbacks = callbacks.with_terminator(LBFGSBGTerminator::default());
            }
            if let Some(eps_norm_g) = d.get_item("eps_norm_g")? {
                if let Some(eps_abs) = eps_norm_g.extract()? {
                    callbacks = callbacks.with_terminator(LBFGSBInfNormGTerminator { eps_abs });
                } else {
                    callbacks = callbacks.with_terminator(LBFGSBInfNormGTerminator::default());
                }
            } else {
                callbacks = callbacks.with_terminator(LBFGSBInfNormGTerminator::default());
            }
            Ok(callbacks)
        }
    }
    impl FromPyArgs<Vec<f64>> for AdamConfig {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            let mut config = AdamConfig::new(args);
            if let Some(alpha) = d.get_item("alpha")? {
                config = config.with_alpha(alpha.extract()?);
            }
            if let Some(beta_1) = d.get_item("beta_1")? {
                config = config.with_beta_1(beta_1.extract()?);
            }
            if let Some(beta_2) = d.get_item("beta_2")? {
                config = config.with_beta_2(beta_2.extract()?);
            }
            if let Some(epsilon) = d.get_item("epsilon")? {
                config = config.with_epsilon(epsilon.extract()?);
            }
            Ok(config)
        }
    }
    impl<P> FromPyArgs for Callbacks<Adam, P, GradientStatus, MaybeThreadPool, LadduError, AdamConfig>
    where
        P: Gradient<MaybeThreadPool, LadduError>,
    {
        fn from_pyargs(_args: &(), d: &Bound<PyDict>) -> PyResult<Self> {
            let mut callbacks = Callbacks::empty();
            let mut term = AdamEMATerminator::default();
            if let Some(beta_c) = d.get_item("beta_c")? {
                term.beta_c = beta_c.extract()?;
            }
            if let Some(eps_loss) = d.get_item("eps_loss")? {
                term.eps_loss = eps_loss.extract()?;
            }
            if let Some(patience) = d.get_item("patience")? {
                term.patience = patience.extract()?;
            }
            callbacks = callbacks.with_terminator(term);
            Ok(callbacks)
        }
    }
    impl FromPyArgs<Vec<f64>> for NelderMeadConfig {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            let construction_method = SimplexConstructionMethod::from_pyargs(args, d)?;
            let mut config = NelderMeadConfig::new_with_method(construction_method);
            if let Some(alpha) = d.get_item("alpha")? {
                config = config.with_alpha(alpha.extract()?);
            }
            if let Some(beta) = d.get_item("beta")? {
                config = config.with_beta(beta.extract()?);
            }
            if let Some(gamma) = d.get_item("gamma")? {
                config = config.with_gamma(gamma.extract()?);
            }
            if let Some(delta) = d.get_item("delta")? {
                config = config.with_delta(delta.extract()?);
            }
            if let Some(adaptive) = d.get_item("adaptive")? {
                if adaptive.extract()? {
                    config = config.with_adaptive(args.len());
                }
            }
            if let Some(expansion_method) = d.get_item("expansion_method")? {
                match expansion_method
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "greedyminimization" => {
                        config = config
                            .with_expansion_method(SimplexExpansionMethod::GreedyMinimization);
                        Ok(())
                    }
                    "greedyexpansion" => {
                        config = config
                            .with_expansion_method(SimplexExpansionMethod::GreedyMinimization);
                        Ok(())
                    }
                    _ => Err(PyValueError::new_err(format!(
                        "Invalid expansion method: {}",
                        expansion_method
                    ))),
                }?
            }
            Ok(config)
        }
    }
    impl FromPyArgs<Vec<f64>> for SimplexConstructionMethod {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            if let Some(simplex_construction_method) = d.get_item("simplex_construction_method")? {
                match simplex_construction_method
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "scaledorthogonal" => {
                        let orthogonal_multiplier = d
                            .get_item("orthogonal_multiplier")?
                            .map(|v| v.extract())
                            .transpose()?
                            .unwrap_or(1.05);
                        let orthogonal_zero_step = d
                            .get_item("orthogonal_zero_step")?
                            .map(|v| v.extract())
                            .transpose()?
                            .unwrap_or(0.00025);
                        return Ok(SimplexConstructionMethod::custom_scaled_orthogonal(
                            args,
                            orthogonal_multiplier,
                            orthogonal_zero_step,
                        ));
                    }
                    "orthogonal" => {
                        let simplex_size = d
                            .get_item("simplex_size")?
                            .map(|v| v.extract())
                            .transpose()?
                            .unwrap_or(1.0);
                        return Ok(SimplexConstructionMethod::custom_orthogonal(
                            args,
                            simplex_size,
                        ));
                    }
                    "custom" => {
                        if let Some(other_simplex_points) = d.get_item("simplex")? {
                            let mut simplex = Vec::with_capacity(args.len() + 1);
                            simplex[0] = DVector::from_vec(args.clone());
                            let others = other_simplex_points.extract::<Vec<Vec<f64>>>()?; // TODO: numpy arrays
                            if others.len() != args.len() {
                                return Err(PyValueError::new_err(format!(
                                    "Expected {} additional simplex points, got {}.",
                                    args.len(),
                                    others.len()
                                )));
                            }
                            simplex.extend(others.iter().map(|x| DVector::from_vec(x.clone())));
                            return Ok(SimplexConstructionMethod::custom(simplex));
                        } else {
                            return Err(PyValueError::new_err("Simplex must be specified when using the 'custom' simplex_construction_method."));
                        }
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid simplex_construction_method: {}",
                            simplex_construction_method
                        )))
                    }
                }
            } else {
                Ok(SimplexConstructionMethod::scaled_orthogonal(args))
            }
        }
    }
    impl<P> FromPyArgs
        for Callbacks<
            NelderMead,
            P,
            GradientFreeStatus,
            MaybeThreadPool,
            LadduError,
            NelderMeadConfig,
        >
    where
        P: CostFunction<MaybeThreadPool, LadduError>,
    {
        fn from_pyargs(_args: &(), d: &Bound<PyDict>) -> PyResult<Self> {
            let mut callbacks = Callbacks::empty();
            let eps_f = if let Some(eps_f) = d.get_item("eps_f")? {
                eps_f.extract()?
            } else {
                f64::EPSILON.powf(0.25)
            };
            if let Some(f_term) = d.get_item("f_terminator")? {
                match f_term
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "amoeba" => {
                        callbacks = callbacks
                            .with_terminator(NelderMeadFTerminator::Amoeba { eps_rel: eps_f });
                    }
                    "absolute" => {
                        callbacks = callbacks
                            .with_terminator(NelderMeadFTerminator::Absolute { eps_abs: eps_f });
                    }
                    "stddev" => {
                        callbacks = callbacks
                            .with_terminator(NelderMeadFTerminator::StdDev { eps_abs: eps_f });
                    }
                    _ => Err(PyValueError::new_err(format!(
                        "Invalid f_terminator: {}",
                        f_term
                    )))?,
                }
            } else {
                callbacks =
                    callbacks.with_terminator(NelderMeadFTerminator::StdDev { eps_abs: eps_f });
            }
            let eps_x = if let Some(eps_x) = d.get_item("eps_x")? {
                eps_x.extract()?
            } else {
                f64::EPSILON.powf(0.25)
            };
            if let Some(x_term) = d.get_item("x_terminator")? {
                match x_term
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "diameter" => {
                        callbacks = callbacks
                            .with_terminator(NelderMeadXTerminator::Diameter { eps_abs: eps_x });
                    }
                    "higham" => {
                        callbacks = callbacks
                            .with_terminator(NelderMeadXTerminator::Higham { eps_rel: eps_x });
                    }
                    "rowan" => {
                        callbacks = callbacks
                            .with_terminator(NelderMeadXTerminator::Rowan { eps_rel: eps_x });
                    }
                    "singer" => {
                        callbacks = callbacks
                            .with_terminator(NelderMeadXTerminator::Singer { eps_rel: eps_x });
                    }
                    _ => Err(PyValueError::new_err(format!(
                        "Invalid x_terminator: {}",
                        x_term
                    )))?,
                }
            } else {
                callbacks =
                    callbacks.with_terminator(NelderMeadXTerminator::Singer { eps_rel: eps_x });
            }
            Ok(callbacks)
        }
    }
    impl FromPyArgs<Vec<f64>> for SwarmPositionInitializer {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            if let Some(swarm_position_initializer) = d.get_item("swarm_position_initializer")? {
                match swarm_position_initializer
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "randominlimits" => {
                        if let (Some(swarm_position_bounds), Some(swarm_size)) = (
                            d.get_item("swarm_position_bounds")?,
                            d.get_item("swarm_size")?,
                        ) {
                            return Ok(SwarmPositionInitializer::RandomInLimits {
                                bounds: swarm_position_bounds.extract()?,
                                n_particles: swarm_size.extract()?,
                            });
                        } else {
                            return Err(PyValueError::new_err("The swarm_position_bounds and swarm_size must be specified when using the 'randominlimits' swarm_position_initializer."));
                        }
                    }
                    "latinhypercube" => {
                        if let (Some(swarm_position_bounds), Some(swarm_size)) = (
                            d.get_item("swarm_position_bounds")?,
                            d.get_item("swarm_size")?,
                        ) {
                            return Ok(SwarmPositionInitializer::LatinHypercube {
                                bounds: swarm_position_bounds.extract()?,
                                n_particles: swarm_size.extract()?,
                            });
                        } else {
                            return Err(PyValueError::new_err("The swarm_position_bounds and swarm_size must be specified when using the 'latinhypercube' swarm_position_initializer."));
                        }
                    }
                    "custom" => {
                        if let Some(swarm) = d.get_item("swarm")? {
                            return Ok(SwarmPositionInitializer::Custom(
                                swarm
                                    .extract::<Vec<Vec<f64>>>()?
                                    .iter()
                                    .chain(vec![args].into_iter())
                                    .map(|x| DVector::from_vec(x.clone()))
                                    .collect(), // TODO: numpy arrays?
                            ));
                        } else {
                            return Err(PyValueError::new_err("The swarm must be specified when using the 'custom' swarm_position_initializer."));
                        }
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid swarm_position_initializer: {}",
                            swarm_position_initializer
                        )));
                    }
                }
            } else {
                return Err(PyValueError::new_err(
                    "The swarm_position_initializer must be specified for the PSO algorithm.",
                ));
            }
        }
    }
    impl FromPyArgs<Vec<f64>> for Swarm {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            let swarm_position_initializer = SwarmPositionInitializer::from_pyargs(args, d)?;
            let mut swarm = Swarm::new(swarm_position_initializer);
            if let Some(swarm_topology_str) = d.get_item("swarm_topology")? {
                match swarm_topology_str
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "global" => {
                        swarm = swarm.with_topology(SwarmTopology::Global);
                    }
                    "ring" => {
                        swarm = swarm.with_topology(SwarmTopology::Ring);
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid swarm_topology: {}",
                            swarm_topology_str
                        )))
                    }
                }
            }
            if let Some(swarm_update_method_str) = d.get_item("swarm_update_method")? {
                match swarm_update_method_str
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "sync" | "synchronous" => {
                        swarm = swarm.with_update_method(SwarmUpdateMethod::Synchronous);
                    }
                    "async" | "asynchronous" => {
                        swarm = swarm.with_update_method(SwarmUpdateMethod::Asynchronous);
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid swarm_update_method: {}",
                            swarm_update_method_str
                        )))
                    }
                }
            }
            if let Some(swarm_boundary_method_str) = d.get_item("swarm_boundary_method")? {
                match swarm_boundary_method_str
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "inf" => {
                        swarm = swarm.with_boundary_method(SwarmBoundaryMethod::Inf);
                    }
                    "shr" => {
                        swarm = swarm.with_boundary_method(SwarmBoundaryMethod::Shr);
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid swarm_boundary_method: {}",
                            swarm_boundary_method_str
                        )))
                    }
                }
            }
            if let Some(swarm_velocity_bounds) = d.get_item("swarm_velocity_bounds")? {
                swarm = swarm.with_velocity_initializer(SwarmVelocityInitializer::RandomInLimits(
                    swarm_velocity_bounds.extract()?,
                ));
            }
            Ok(swarm)
        }
    }
    impl FromPyArgs<Vec<f64>> for PSOConfig {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            let swarm = Swarm::from_pyargs(args, d)?;
            let mut config = PSOConfig::new(swarm);
            if let Some(omega) = d.get_item("omega")? {
                config = config.with_omega(omega.extract()?);
            }
            if let Some(c1) = d.get_item("c1")? {
                config = config.with_c1(c1.extract()?);
            }
            if let Some(c2) = d.get_item("c2")? {
                config = config.with_c2(c2.extract()?);
            }
            Ok(config)
        }
    }
    impl<P> FromPyArgs<Vec<f64>> for MinimizationSettings<P>
    where
        P: Gradient<MaybeThreadPool, LadduError>,
    {
        fn from_pyargs(args: &Vec<f64>, d: &Bound<PyDict>) -> PyResult<Self> {
            let bounds: Option<Vec<ganesh::traits::boundlike::Bound>> = d
                .get_item("bounds")?
                .map(|bounds| bounds.extract::<Vec<(Option<f64>, Option<f64>)>>())
                .transpose()?
                .map(|bounds| {
                    bounds
                        .into_iter()
                        .map(ganesh::traits::boundlike::Bound::from)
                        .collect()
                });
            let num_threads = d
                .get_item("threads")?
                .map(|t| t.extract())
                .transpose()?
                .unwrap_or(0);
            let add_debug = d
                .get_item("debug")?
                .map(|d| d.extract())
                .transpose()?
                .unwrap_or(false);
            let observers = if let Some(observers) = d.get_item("observers")? {
                if let Ok(observers) = observers.downcast::<PyList>() {
                    observers.into_iter().map(|observer| {
                        if let Ok(observer) = observer.extract::<MinimizationObserver>() {
                            Ok(observer)
                        } else {
                            Err(PyValueError::new_err("The observers must be either a single MinimizationObserver or a list of MinimizationObservers."))
                        }
                    }).collect::<PyResult<Vec<MinimizationObserver>>>()?
                } else if let Ok(observer) = observers.extract::<MinimizationObserver>() {
                    vec![observer]
                } else {
                    return Err(PyValueError::new_err("The observers must be either a single MinimizationObserver or a list of MinimizationObservers."));
                }
            } else {
                vec![]
            };
            let terminators = if let Some(terminators) = d.get_item("terminators")? {
                if let Ok(terminators) = terminators.downcast::<PyList>() {
                    terminators.into_iter().map(|terminator| {
                    if let Ok(terminator) = terminator.extract::<MinimizationTerminator>() {
                        Ok(terminator)
                    } else {
                        Err(PyValueError::new_err("The terminators must be either a single MinimizationTerminator or a list of MinimizationTerminators."))
                        }
                    }).collect::<PyResult<Vec<MinimizationTerminator>>>()?
                } else if let Ok(terminator) = terminators.extract::<MinimizationTerminator>() {
                    vec![terminator]
                } else {
                    return Err(PyValueError::new_err("The terminators must be either a single MinimizationTerminator or a list of MinimizationTerminators."));
                }
            } else {
                vec![]
            };
            let max_steps: Option<usize> = d
                .get_item("max_steps")?
                .map(|ms| ms.extract())
                .transpose()?;
            let settings: Bound<PyDict> = d
                .get_item("settings")?
                .map(|settings| settings.extract())
                .transpose()?
                .unwrap_or_else(|| PyDict::new(d.py()));
            if let Some(method) = d.get_item("method")? {
                match method
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "lbfgsb" => {
                        let mut config = LBFGSBConfig::from_pyargs(args, &settings)?;
                        if let Some(bounds) = bounds {
                            config = config.with_bounds(bounds);
                        }
                        let mut callbacks = Callbacks::from_pyargs(&(), &settings)?;
                        if add_debug {
                            callbacks = callbacks.with_observer(DebugObserver);
                        }
                        if let Some(max_steps) = max_steps {
                            callbacks = callbacks.with_terminator(MaxSteps(max_steps));
                        }
                        for observer in observers {
                            callbacks = callbacks.with_observer(observer);
                        }
                        for terminator in terminators {
                            callbacks = callbacks.with_terminator(terminator);
                        }
                        callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                        Ok(MinimizationSettings::LBFGSB {
                            config,
                            callbacks,
                            num_threads,
                        })
                    }
                    "adam" => {
                        let mut config = AdamConfig::from_pyargs(args, &settings)?;
                        if let Some(bounds) = bounds {
                            config = config.with_transform(&Bounds::from(bounds))
                        }
                        let mut callbacks = Callbacks::from_pyargs(&(), &settings)?;
                        if add_debug {
                            callbacks = callbacks.with_observer(DebugObserver);
                        }
                        if let Some(max_steps) = max_steps {
                            callbacks = callbacks.with_terminator(MaxSteps(max_steps));
                        }
                        for observer in observers {
                            callbacks = callbacks.with_observer(observer);
                        }
                        for terminator in terminators {
                            callbacks = callbacks.with_terminator(terminator);
                        }
                        callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                        Ok(MinimizationSettings::Adam {
                            config,
                            callbacks,
                            num_threads,
                        })
                    }
                    "neldermead" => {
                        let mut config = NelderMeadConfig::from_pyargs(args, &settings)?;
                        if let Some(bounds) = bounds {
                            config = config.with_bounds(bounds);
                        }
                        let mut callbacks = Callbacks::from_pyargs(&(), &settings)?;
                        if add_debug {
                            callbacks = callbacks.with_observer(DebugObserver);
                        }
                        if let Some(max_steps) = max_steps {
                            callbacks = callbacks.with_terminator(MaxSteps(max_steps));
                        }
                        for observer in observers {
                            callbacks = callbacks.with_observer(observer);
                        }
                        for terminator in terminators {
                            callbacks = callbacks.with_terminator(terminator);
                        }
                        callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                        Ok(MinimizationSettings::NelderMead {
                            config,
                            callbacks,
                            num_threads,
                        })
                    }
                    "pso" => {
                        let mut config = PSOConfig::from_pyargs(args, &settings)?;
                        if let Some(bounds) = bounds {
                            if let Some(use_transform) = settings.get_item("use_transform")? {
                                if use_transform.extract()? {
                                    config = config.with_transform(&Bounds::from(bounds))
                                } else {
                                    config = config.with_bounds(bounds)
                                }
                            } else {
                                config = config.with_bounds(bounds)
                            }
                        }
                        let mut callbacks = Callbacks::empty();
                        if add_debug {
                            return Err(PyValueError::new_err(
                                "The debug setting is not yet supported for PSO",
                            ));
                            // callbacks = callbacks.with_observer(DebugObserver);
                            // TODO: SwarmStatus needs to impl Debug
                        }
                        if let Some(max_steps) = max_steps {
                            callbacks = callbacks.with_terminator(MaxSteps(max_steps));
                        }
                        for observer in observers {
                            callbacks = callbacks.with_observer(observer);
                        }
                        for terminator in terminators {
                            callbacks = callbacks.with_terminator(terminator);
                        }
                        callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                        Ok(MinimizationSettings::PSO {
                            config,
                            callbacks,
                            num_threads,
                        })
                    }
                    _ => Err(PyValueError::new_err(format!(
                        "Invalid minimizer: {}",
                        method
                    ))),
                }
            } else {
                Err(PyValueError::new_err("No method specified"))
            }
        }
    }

    impl FromPyArgs<Vec<DVector<f64>>> for AIESConfig {
        fn from_pyargs(args: &Vec<DVector<f64>>, d: &Bound<PyDict>) -> PyResult<Self> {
            let mut config = AIESConfig::new(args.to_vec());
            if let Some(moves) = d.get_item("moves")? {
                let moves_list = moves.downcast::<PyList>()?;
                let mut aies_moves = vec![];
                for mcmc_move in moves_list {
                    if let Ok(default_move) = mcmc_move.extract::<(String, f64)>() {
                        match default_move
                            .0
                            .to_lowercase()
                            .trim()
                            .replace("-", "")
                            .replace(" ", "")
                            .as_str()
                        {
                            "stretch" => aies_moves.push(AIESMove::stretch(default_move.1)),
                            "walk" => aies_moves.push(AIESMove::walk(default_move.1)),
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "Invalid AIES move: {}",
                                    default_move.0
                                )))
                            }
                        }
                    } else if let Ok(custom_move) =
                        mcmc_move.extract::<(String, Bound<PyDict>, f64)>()
                    {
                        match custom_move
                            .0
                            .to_lowercase()
                            .trim()
                            .replace("-", "")
                            .replace(" ", "")
                            .as_str()
                        {
                            "stretch" => aies_moves.push((
                                AIESMove::Stretch {
                                    a: custom_move
                                        .1
                                        .get_item("a")?
                                        .map(|val| val.extract())
                                        .transpose()?
                                        .unwrap_or(2.0),
                                },
                                custom_move.2,
                            )),
                            "walk" => aies_moves.push(AIESMove::walk(custom_move.2)),
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "Invalid AIES move: {}",
                                    custom_move.0
                                )))
                            }
                        }
                    } else {
                        return Err(PyValueError::new_err("The 'moves' argument must be a list of (str, float) or (str, dict, float) tuples!"));
                    }
                }
                config = config.with_moves(aies_moves);
            }
            Ok(config)
        }
    }

    impl FromPyArgs<Vec<DVector<f64>>> for ESSConfig {
        fn from_pyargs(args: &Vec<DVector<f64>>, d: &Bound<PyDict>) -> PyResult<Self> {
            let mut config = ESSConfig::new(args.to_vec());
            if let Some(moves) = d.get_item("moves")? {
                let moves_list = moves.downcast::<PyList>()?;
                let mut ess_moves = vec![];
                for mcmc_move in moves_list {
                    if let Ok(default_move) = mcmc_move.extract::<(String, f64)>() {
                        match default_move
                            .0
                            .to_lowercase()
                            .trim()
                            .replace("-", "")
                            .replace(" ", "")
                            .as_str()
                        {
                            "differential" => ess_moves.push(ESSMove::differential(default_move.1)),
                            "gaussian" => ess_moves.push(ESSMove::gaussian(default_move.1)),
                            "global" => {
                                ess_moves.push(ESSMove::global(default_move.1, None, None, None))
                            }
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "Invalid ESS move: {}",
                                    default_move.0
                                )))
                            }
                        }
                    } else if let Ok(custom_move) =
                        mcmc_move.extract::<(String, Bound<PyDict>, f64)>()
                    {
                        match custom_move
                            .0
                            .to_lowercase()
                            .trim()
                            .replace("-", "")
                            .replace(" ", "")
                            .as_str()
                        {
                            "differential" => ess_moves.push(ESSMove::differential(custom_move.2)),
                            "gaussian" => ess_moves.push(ESSMove::gaussian(custom_move.2)),
                            "global" => ess_moves.push(ESSMove::global(
                                custom_move.2,
                                custom_move
                                    .1
                                    .get_item("scale")?
                                    .map(|value| value.extract())
                                    .transpose()?,
                                custom_move
                                    .1
                                    .get_item("rescale_cov")?
                                    .map(|value| value.extract())
                                    .transpose()?,
                                custom_move
                                    .1
                                    .get_item("n_components")?
                                    .map(|value| value.extract())
                                    .transpose()?,
                            )),
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "Invalid ESS move: {}",
                                    custom_move.0
                                )))
                            }
                        }
                    } else {
                        return Err(PyValueError::new_err("The 'moves' argument must be a list of (str, float) or (str, dict, float) tuples!"));
                    }
                }
                config = config.with_moves(ess_moves)
            }
            if let Some(n_adaptive) = d.get_item("n_adaptive")? {
                config = config.with_n_adaptive(n_adaptive.extract()?);
            }
            if let Some(mu) = d.get_item("mu")? {
                config = config.with_mu(mu.extract()?);
            }
            if let Some(max_steps) = d.get_item("max_steps")? {
                config = config.with_max_steps(max_steps.extract()?);
            }
            Ok(config)
        }
    }

    impl<P> FromPyArgs<Vec<DVector<f64>>> for MCMCSettings<P>
    where
        P: LogDensity<MaybeThreadPool, LadduError>,
    {
        fn from_pyargs(args: &Vec<DVector<f64>>, d: &Bound<PyDict>) -> PyResult<Self> {
            let bounds: Option<Vec<ganesh::traits::boundlike::Bound>> = d
                .get_item("bounds")?
                .map(|bounds| bounds.extract::<Vec<(Option<f64>, Option<f64>)>>())
                .transpose()?
                .map(|bounds| {
                    bounds
                        .into_iter()
                        .map(ganesh::traits::boundlike::Bound::from)
                        .collect()
                });
            let num_threads = d
                .get_item("threads")?
                .map(|t| t.extract())
                .transpose()?
                .unwrap_or(0);
            let add_debug = d
                .get_item("debug")?
                .map(|d| d.extract())
                .transpose()?
                .unwrap_or(false);
            let observers = if let Some(observers) = d.get_item("observers")? {
                if let Ok(observers) = observers.downcast::<PyList>() {
                    observers.into_iter().map(|observer| {
                        if let Ok(observer) = observer.extract::<MCMCObserver>() {
                            Ok(observer)
                        } else {
                            Err(PyValueError::new_err("The observers must be either a single MCMCObserver or a list of MCMCObservers."))
                        }
                    }).collect::<PyResult<Vec<MCMCObserver>>>()?
                } else if let Ok(observer) = observers.extract::<MCMCObserver>() {
                    vec![observer]
                } else {
                    return Err(PyValueError::new_err("The observers must be either a single MCMCObserver or a list of MCMCObservers."));
                }
            } else {
                vec![]
            };
            let terminators = if let Some(terminators) = d.get_item("terminators")? {
                if let Ok(terminators) = terminators.downcast::<PyList>() {
                    terminators
                        .into_iter()
                        .map(|terminator| {
                            if let Ok(terminator) =
                                terminator.extract::<PyAutocorrelationTerminator>()
                            {
                                Ok(PythonMCMCTerminator::Autocorrelation(terminator))
                            }
                            else if let Ok(terminator) = terminator.extract::<MCMCTerminator>() {
                                Ok(PythonMCMCTerminator::UserDefined(terminator))
                            } else {
                                Err(PyValueError::new_err("The terminators must be either a single MCMCTerminator or a list of MCMCTerminators."))
                            }
                        })
                        .collect::<PyResult<Vec<PythonMCMCTerminator>>>()?
                } else if let Ok(terminator) = terminators.extract::<PyAutocorrelationTerminator>()
                {
                    vec![PythonMCMCTerminator::Autocorrelation(terminator)]
                } else if let Ok(terminator) = terminators.extract::<MCMCTerminator>() {
                    vec![PythonMCMCTerminator::UserDefined(terminator)]
                } else {
                    return Err(PyValueError::new_err("The terminators must be either a single MCMCTerminator or a list of MCMCTerminators."));
                }
            } else {
                vec![]
            };
            let max_steps: Option<usize> = d
                .get_item("max_steps")?
                .map(|ms| ms.extract())
                .transpose()?;
            let settings: Bound<PyDict> = d
                .get_item("settings")?
                .map(|settings| settings.extract())
                .transpose()?
                .unwrap_or_else(|| PyDict::new(d.py()));
            if let Some(method) = d.get_item("method")? {
                match method
                    .extract::<String>()?
                    .to_lowercase()
                    .trim()
                    .replace("-", "")
                    .replace(" ", "")
                    .as_str()
                {
                    "aies" => {
                        let mut config = AIESConfig::from_pyargs(args, &settings)?;
                        if let Some(bounds) = bounds {
                            config = config.with_transform(&Bounds::from(bounds))
                        }
                        let mut callbacks = Callbacks::empty();
                        if add_debug {
                            callbacks = callbacks.with_observer(DebugObserver);
                        }
                        if let Some(max_steps) = max_steps {
                            callbacks = callbacks.with_terminator(MaxSteps(max_steps));
                        }
                        for observer in observers {
                            callbacks = callbacks.with_observer(observer);
                        }
                        for terminator in terminators {
                            callbacks = callbacks.with_terminator(terminator);
                        }
                        callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                        Ok(MCMCSettings::AIES {
                            config,
                            callbacks,
                            num_threads,
                        })
                    }
                    "ess" => {
                        let mut config = ESSConfig::from_pyargs(args, &settings)?;
                        if let Some(bounds) = bounds {
                            config = config.with_transform(&Bounds::from(bounds))
                        }
                        let mut callbacks = Callbacks::empty();
                        if add_debug {
                            callbacks = callbacks.with_observer(DebugObserver);
                        }
                        if let Some(max_steps) = max_steps {
                            callbacks = callbacks.with_terminator(MaxSteps(max_steps));
                        }
                        for observer in observers {
                            callbacks = callbacks.with_observer(observer);
                        }
                        for terminator in terminators {
                            callbacks = callbacks.with_terminator(terminator);
                        }
                        callbacks = callbacks.with_terminator(CtrlCAbortSignal::new());
                        Ok(MCMCSettings::ESS {
                            config,
                            callbacks,
                            num_threads,
                        })
                    }
                    _ => Err(PyValueError::new_err(format!(
                        "Invalid MCMC algorithm: {}",
                        method
                    ))),
                }
            } else {
                Err(PyValueError::new_err("No method specified"))
            }
        }
    }

    enum MinimizationStatus {
        GradientStatus(Arc<Mutex<GradientStatus>>),
        GradientFreeStatus(Arc<Mutex<GradientFreeStatus>>),
        SwarmStatus(Arc<Mutex<SwarmStatus>>),
    }
    impl MinimizationStatus {
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            match self {
                Self::GradientStatus(gradient_status) => {
                    gradient_status.lock().x.as_slice().to_pyarray(py)
                }
                Self::GradientFreeStatus(gradient_free_status) => {
                    gradient_free_status.lock().x.as_slice().to_pyarray(py)
                }
                Self::SwarmStatus(swarm_status) => {
                    swarm_status.lock().gbest.x.as_slice().to_pyarray(py)
                }
            }
        }
        fn fx(&self) -> f64 {
            match self {
                Self::GradientStatus(gradient_status) => gradient_status.lock().fx,
                Self::GradientFreeStatus(gradient_free_status) => gradient_free_status.lock().fx,
                Self::SwarmStatus(swarm_status) => swarm_status.lock().gbest.fx.unwrap(),
            }
        }
        fn message(&self) -> String {
            match self {
                Self::GradientStatus(gradient_status) => {
                    gradient_status.lock().message().to_string()
                }
                Self::GradientFreeStatus(gradient_free_status) => {
                    gradient_free_status.lock().message().to_string()
                }
                Self::SwarmStatus(swarm_status) => swarm_status.lock().message().to_string(),
            }
        }
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
            match self {
                Self::GradientStatus(gradient_status) => gradient_status.lock().err.clone(),
                Self::GradientFreeStatus(gradient_free_status) => {
                    gradient_free_status.lock().err.clone()
                }
                Self::SwarmStatus(_) => None,
            }
            .map(|e| e.as_slice().to_pyarray(py))
        }
        fn n_f_evals(&self) -> usize {
            match self {
                Self::GradientStatus(gradient_status) => gradient_status.lock().n_f_evals,
                Self::GradientFreeStatus(gradient_free_status) => {
                    gradient_free_status.lock().n_f_evals
                }
                Self::SwarmStatus(swarm_status) => swarm_status.lock().n_f_evals,
            }
        }
        fn n_g_evals(&self) -> usize {
            match self {
                Self::GradientStatus(gradient_status) => gradient_status.lock().n_g_evals,
                Self::GradientFreeStatus(_) => 0,
                Self::SwarmStatus(_) => 0,
            }
        }
        fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            match self {
                Self::GradientStatus(gradient_status) => gradient_status.lock().cov.clone(),
                Self::GradientFreeStatus(gradient_free_status) => {
                    gradient_free_status.lock().cov.clone()
                }
                Self::SwarmStatus(_) => None,
            }
            .map(|cov| cov.to_pyarray(py))
        }
        fn hess<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            match self {
                Self::GradientStatus(gradient_status) => gradient_status.lock().hess.clone(),
                Self::GradientFreeStatus(gradient_free_status) => {
                    gradient_free_status.lock().hess.clone()
                }
                Self::SwarmStatus(_) => None,
            }
            .map(|hess| hess.to_pyarray(py))
        }
        fn converged(&self) -> bool {
            match self {
                Self::GradientStatus(gradient_status) => gradient_status.lock().converged(),
                Self::GradientFreeStatus(gradient_free_status) => {
                    gradient_free_status.lock().converged()
                }
                Self::SwarmStatus(swarm_status) => swarm_status.lock().converged(),
            }
        }
        fn swarm(&self) -> Option<PySwarm> {
            match self {
                Self::GradientStatus(_) | Self::GradientFreeStatus(_) => None,
                Self::SwarmStatus(swarm_status) => Some(PySwarm(swarm_status.lock().swarm.clone())),
            }
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
            self.0.position.fx.unwrap()
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
            self.0.best.fx.unwrap()
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

    /// The intermediate status used to inform the user of the current state of a minimization algorithm.
    ///
    #[pyclass(name = "MinimizationStatus", module = "laddu")]
    pub struct PyMinimizationStatus(MinimizationStatus);

    #[pymethods]
    impl PyMinimizationStatus {
        /// The current best position of the minimizer.
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.x(py)
        }
        /// The current value of the objective function at the best position.
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn fx(&self) -> f64 {
            self.0.fx()
        }
        /// A message indicating the current state of the minimization.
        ///
        /// Returns
        /// -------
        /// str
        ///
        #[getter]
        fn message(&self) -> String {
            self.0.message()
        }
        /// The current error estimate at the best position. May be None for algorithms that do not estimate errors.
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        #[getter]
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
            self.0.err(py)
        }
        /// The number of objective function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.n_f_evals()
        }
        /// The number of gradient function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_g_evals(&self) -> usize {
            self.0.n_g_evals()
        }
        /// The covariance matrix of the best position. May be None for algorithms that do not estimate errors.
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            self.0.cov(py)
        }
        /// The Hessian matrix of the best position. May be None for algorithms that do not estimate errors.
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        #[getter]
        fn hess<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            self.0.hess(py)
        }
        #[getter]
        fn converged(&self) -> bool {
            self.0.converged()
        }
        /// The swarm of particles used in swarm-based optimization algorithms. May be None for algorithms that do not use a swarm.
        ///
        /// Returns
        /// -------
        /// Swarm or None
        ///
        #[getter]
        fn swarm(&self) -> Option<PySwarm> {
            self.0.swarm()
        }
    }

    /// A summary of the results of a minimization.
    ///
    #[pyclass(name = "MinimizationSummary", module = "laddu")]
    #[derive(Clone)]
    pub struct PyMinimizationSummary(pub MinimizationSummary);

    #[pymethods]
    impl PyMinimizationSummary {
        /// Bounds which were used during the minimization.
        ///
        /// Returns
        /// -------
        /// list of tuple of floats or None
        ///
        #[getter]
        fn bounds(&self) -> Option<Vec<(f64, f64)>> {
            self.0
                .clone()
                .bounds
                .map(|bs| bs.iter().map(|b| b.0.as_floats()).collect())
        }
        /// Names of each parameter used in the minimization.
        ///
        /// Returns
        /// -------
        /// list of str
        ///
        #[getter]
        fn parameter_names(&self) -> Vec<String> {
            self.0.parameter_names.clone().unwrap_or_default()
        }
        /// The status at the end of the minimization.
        ///
        /// Returns
        /// -------
        /// str
        ///
        #[getter]
        fn message(&self) -> String {
            self.0.message.clone()
        }
        /// The starting position of the minimizer.
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.x0.as_slice().to_pyarray(py)
        }
        /// The best position found by the minimizer.
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.x.as_slice().to_pyarray(py)
        }
        /// The uncertainty associated with each parameter (may be zeros if no uncertainty was estimated or nan if the covariance matrix was not positive definite).
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn std<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.std.as_slice().to_pyarray(py)
        }
        /// The value of the objective function at the best position found by the minimizer.
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn fx(&self) -> f64 {
            self.0.fx
        }
        /// The number of objective function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn cost_evals(&self) -> usize {
            self.0.cost_evals
        }
        /// The number of gradient function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn gradient_evals(&self) -> usize {
            self.0.gradient_evals
        }
        /// True if the minimization algorithm has converged.
        ///
        /// Returns
        /// -------
        /// bool
        ///
        #[getter]
        fn converged(&self) -> bool {
            self.0.converged
        }
        /// The covariance matrix of the best position (may contain zeros if the algorithm did not estimate a Hessian).
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            self.0.covariance.to_pyarray(py)
        }
        fn __str__(&self) -> String {
            self.0.to_string()
        }
        #[new]
        fn new() -> Self {
            Self(MinimizationSummary::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(
                py,
                bincode::serde::encode_to_vec(&self.0, bincode::config::standard())
                    .map_err(LadduError::EncodeError)?
                    .as_slice(),
            ))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = Self(
                bincode::serde::decode_from_slice(state.as_bytes(), bincode::config::standard())
                    .map_err(LadduError::DecodeError)?
                    .0,
            );
            Ok(())
        }
    }

    /// An enum used by a terminator to continue or stop an algorithm.
    ///
    #[pyclass(eq, eq_int, name = "ControlFlow", module = "laddu")]
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
    /// that takes the current step and a [`PyMinimizationStatus`] as arguments.
    #[derive(Clone)]
    pub struct MinimizationObserver(Arc<Py<PyAny>>);
    impl FromPyObject<'_> for MinimizationObserver {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(MinimizationObserver(Arc::new(ob.clone().unbind())))
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
            Python::with_gil(|py| {
                self.0
                    .bind(py)
                    .call_method1(
                        "observe",
                        (
                            current_step,
                            PyMinimizationStatus(MinimizationStatus::GradientStatus(Arc::new(
                                Mutex::new(status.clone()),
                            ))),
                        ),
                    )
                    .expect("Error calling observe");
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
            Python::with_gil(|py| {
                self.0
                    .bind(py)
                    .call_method1(
                        "observe",
                        (
                            current_step,
                            PyMinimizationStatus(MinimizationStatus::GradientFreeStatus(Arc::new(
                                Mutex::new(status.clone()),
                            ))),
                        ),
                    )
                    .expect("Error calling observe");
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
            Python::with_gil(|py| {
                self.0
                    .bind(py)
                    .call_method1(
                        "observe",
                        (
                            current_step,
                            PyMinimizationStatus(MinimizationStatus::SwarmStatus(Arc::new(
                                Mutex::new(status.clone()),
                            ))),
                        ),
                    )
                    .expect("Error calling observe");
            })
        }
    }

    /// An [`Terminator`] which can be used to monitor the progress of a minimization.
    ///
    /// This should be paired with a Python object which has an `check_for_termination` method
    /// that takes the current step and a [`PyMinimizationStatus`] as arguments and returns a
    /// [`PyControlFlow`].
    #[derive(Clone)]
    pub struct MinimizationTerminator(Arc<Py<PyAny>>);

    impl FromPyObject<'_> for MinimizationTerminator {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(MinimizationTerminator(Arc::new(ob.clone().unbind())))
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
            Python::with_gil(|py| -> PyResult<ControlFlow<()>> {
                let wrapped_status = Arc::new(Mutex::new(std::mem::take(status)));
                let py_status = Py::new(
                    py,
                    PyMinimizationStatus(MinimizationStatus::GradientStatus(
                        wrapped_status.clone(),
                    )),
                )?;
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))
                    .expect("Error calling check_for_termination");
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
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
            Python::with_gil(|py| -> PyResult<ControlFlow<()>> {
                let wrapped_status = Arc::new(Mutex::new(std::mem::take(status)));
                let py_status = Py::new(
                    py,
                    PyMinimizationStatus(MinimizationStatus::GradientFreeStatus(
                        wrapped_status.clone(),
                    )),
                )?;
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))
                    .expect("Error calling check_for_termination");
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
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
            Python::with_gil(|py| -> PyResult<ControlFlow<()>> {
                let wrapped_status = Arc::new(Mutex::new(std::mem::take(status)));
                let py_status = Py::new(
                    py,
                    PyMinimizationStatus(MinimizationStatus::SwarmStatus(wrapped_status.clone())),
                )?;
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))
                    .expect("Error calling check_for_termination");
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
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
            let lock = point.read();
            (lock.x.clone().as_slice().to_pyarray(py), lock.fx_checked())
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

    /// A summary of the results of an MCMC sampling.
    ///
    #[pyclass(name = "MCMCSummary", module = "laddu")]
    pub struct PyMCMCSummary(pub MCMCSummary);

    #[pymethods]
    impl PyMCMCSummary {
        /// Bounds which were used during the MCMC sampling.
        ///
        /// Returns
        /// -------
        /// list of tuple of floats or None
        ///
        #[getter]
        fn bounds(&self) -> Option<Vec<(f64, f64)>> {
            self.0
                .clone()
                .bounds
                .map(|bs| bs.iter().map(|b| b.0.as_floats()).collect())
        }
        /// Names of each parameter used in the MCMC sampling.
        ///
        /// Returns
        /// -------
        /// list of str
        ///
        #[getter]
        fn parameter_names(&self) -> Vec<String> {
            self.0.parameter_names.clone().unwrap_or_default()
        }
        /// The status at the end of the MCMC sampling.
        ///
        /// Returns
        /// -------
        /// str
        ///
        #[getter]
        fn message(&self) -> String {
            self.0.message.clone()
        }
        /// The number of objective function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn cost_evals(&self) -> usize {
            self.0.cost_evals
        }
        /// The number of gradient function evaluations performed.
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn gradient_evals(&self) -> usize {
            self.0.gradient_evals
        }
        /// True if the MCMC algorithm has converged.
        ///
        /// Returns
        /// -------
        /// bool
        ///
        #[getter]
        fn converged(&self) -> bool {
            self.0.converged
        }
        /// The dimension of the ensemble `(n_walkers, n_steps, n_variables)`.
        ///
        /// Returns
        /// -------
        /// tuple of int
        ///
        #[getter]
        fn dimension(&self) -> (usize, usize, usize) {
            self.0.dimension
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
            DMatrix::from_columns(&self.0.get_flat_chain(burn, thin))
                .transpose()
                .to_pyarray(py)
        }

        #[new]
        fn new() -> Self {
            Self(MCMCSummary::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(
                py,
                bincode::serde::encode_to_vec(&self.0, bincode::config::standard())
                    .map_err(LadduError::EncodeError)?
                    .as_slice(),
            ))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = Self(
                bincode::serde::decode_from_slice(state.as_bytes(), bincode::config::standard())
                    .map_err(LadduError::DecodeError)?
                    .0,
            );
            Ok(())
        }
    }

    /// An [`Observer`] which can be used to monitor the progress of an MCMC algorithm.
    ///
    /// This should be paired with a Python object which has an `observe` method
    /// that takes the current step and a [`PyEnsembleStatus`] as arguments.
    #[derive(Clone)]
    pub struct MCMCObserver(Arc<Py<PyAny>>);

    impl FromPyObject<'_> for MCMCObserver {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(MCMCObserver(Arc::new(ob.clone().unbind())))
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
                self.0
                    .bind(py)
                    .call_method1(
                        "observe",
                        (
                            current_step,
                            PyEnsembleStatus(Arc::new(Mutex::new(status.clone()))),
                        ),
                    )
                    .expect("Error calling observe");
            })
        }
    }

    #[derive(Clone)]
    enum PythonMCMCTerminator {
        UserDefined(MCMCTerminator),
        Autocorrelation(PyAutocorrelationTerminator),
    }

    impl<A, P, C> Terminator<A, P, EnsembleStatus, MaybeThreadPool, LadduError, C>
        for PythonMCMCTerminator
    where
        A: Algorithm<P, EnsembleStatus, MaybeThreadPool, LadduError, Config = C>,
    {
        fn check_for_termination(
            &mut self,
            current_step: usize,
            algorithm: &mut A,
            problem: &P,
            status: &mut EnsembleStatus,
            args: &MaybeThreadPool,
            config: &C,
        ) -> ControlFlow<()> {
            match self {
                Self::UserDefined(mcmcterminator) => mcmcterminator.check_for_termination(
                    current_step,
                    algorithm,
                    problem,
                    status,
                    args,
                    config,
                ),
                Self::Autocorrelation(py_autocorrelation_terminator) => {
                    py_autocorrelation_terminator.0.check_for_termination(
                        current_step,
                        algorithm,
                        problem,
                        status,
                        args,
                        config,
                    )
                }
            }
        }
    }
    /// A [`Terminator`] which can be used to monitor the progress of an MCMC algorithm.
    ///
    /// This should be paired with a Python object which has an `check_for_termination` method
    /// that takes the current step and a [`PyEnsembleStatus`] as arguments and returns a
    /// [`PyControlFlow`].
    #[derive(Clone)]
    pub struct MCMCTerminator(Arc<Py<PyAny>>);

    impl FromPyObject<'_> for MCMCTerminator {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(MCMCTerminator(Arc::new(ob.clone().unbind())))
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
                let ret = self
                    .0
                    .bind(py)
                    .call_method1("check_for_termination", (current_step, py_status))
                    .expect("Error calling check_for_termination");
                {
                    let mut guard = wrapped_status.lock();
                    std::mem::swap(status, &mut *guard);
                }
                let cf: PyControlFlow = ret.extract()?;
                Ok(cf.into())
            })
            .unwrap_or(ControlFlow::Continue(()))
        }
    }

    /// Calculate the integrated autocorrelation time for each parameter according to
    /// [Karamanis]_
    ///
    /// Parameters
    /// ----------
    /// x : array_like
    ///     An array of dimension ``(n_walkers, n_steps, n_parameters)``
    /// c : float, default = 7.0
    ///     Set the time window for Sokal's autowindowing function[Sokal]_. If None, the default window
    ///     size of 7.0 is used.
    ///
    /// .. rubric:: References
    ///
    /// .. [Karamanis] Karamanis, M., & Beutler, F. (2021). Ensemble slice sampling. Statistics and Computing, 31(5). https://doi.org/10.1007/s11222-021-10038-2
    /// .. [Sokal] Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In NATO ASI Series (pp. 131–192). Springer US. https://doi.org/10.1007/978-1-4899-0319-8_6
    ///
    #[pyfunction(name = "integrated_autocorrelation_times")]
    #[pyo3(signature = (samples, *, c=None))]
    pub fn py_integrated_autocorrelation_times<'py>(
        py: Python<'py>,
        samples: Vec<Vec<Vec<f64>>>,
        c: Option<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let samples: Vec<Vec<DVector<f64>>> = samples
            .into_iter()
            .map(|v| v.into_iter().map(|p| DVector::from_vec(p)).collect())
            .collect();
        integrated_autocorrelation_times(samples, c)
            .as_slice()
            .to_pyarray(py)
    }

    /// A terminator for MCMC algorithms that monitors autocorrelation according to [Karamanis]_.
    ///
    /// Parameters
    /// ----------
    /// n_check : int, default=50
    ///     Number of steps to take between autocorrelation checks.
    /// n_taus_threshold : int, default=50
    ///     Convergence may be achieved if the number of steps exceeds this value times the current
    ///     mean autocorrelation time.
    /// dtau_threshold : float, default=0.01
    ///     The minimum change in mean autocorrelation time required to consider convergence.
    /// discard : float, default=0.5
    ///     The fraction of the chain to discard when calculating autocorrelation times.
    /// terminate : bool, default=True
    ///     If set to False, the terminator will act like an observer and only store
    ///     autocorrelation times.
    /// sokal_window : float, default=None
    ///     Set the time window for Sokal's autowindowing function[Sokal]_. If None, the default window
    ///     size of 7.0 is used.
    /// verbose : bool, default=False
    ///     Print autocorrelation information at each check step.
    ///
    #[pyclass(name = "AutocorrelationTerminator", module = "laddu")]
    #[derive(Clone)]
    pub struct PyAutocorrelationTerminator(Arc<Mutex<AutocorrelationTerminator>>);

    #[pymethods]
    impl PyAutocorrelationTerminator {
        #[new]
        #[pyo3(signature = (*, n_check = 50, n_taus_threshold = 50, dtau_threshold = 0.01, discard = 0.5, terminate = true, sokal_window = None, verbose = false))]
        fn new(
            n_check: usize,
            n_taus_threshold: usize,
            dtau_threshold: f64,
            discard: f64,
            terminate: bool,
            sokal_window: Option<f64>,
            verbose: bool,
        ) -> Self {
            let mut act = AutocorrelationTerminator::default()
                .with_n_check(n_check)
                .with_n_taus_threshold(n_taus_threshold)
                .with_dtau_threshold(dtau_threshold)
                .with_discard(discard)
                .with_terminate(terminate)
                .with_verbose(verbose);
            if let Some(sokal_window) = sokal_window {
                act = act.with_sokal_window(sokal_window)
            }
            Self(act.build())
        }

        /// A list of autocorrelation times for each parameter.
        ///
        /// Returns
        /// -------
        /// taus : array_like
        ///
        #[getter]
        fn taus<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            self.0.lock().taus.to_pyarray(py)
        }
    }
}
