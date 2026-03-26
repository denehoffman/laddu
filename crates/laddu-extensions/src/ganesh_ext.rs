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
use laddu_core::{LadduError, LadduResult, ThreadPoolManager};
use nalgebra::DVector;

/// A settings wrapper for minimization algorithms
pub enum MinimizationSettings<P> {
    /// Settings for the L-BFGS-B algorithm
    LBFGSB {
        /// The configuration struct
        config: LBFGSBConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<LBFGSB, P, GradientStatus, MaybeThreadPool, LadduError, LBFGSBConfig>,
        /// The number of threads to use (`0` uses the global default thread count)
        num_threads: usize,
    },
    /// Settings for the Adam algorithm
    Adam {
        /// The configuration struct
        config: AdamConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<Adam, P, GradientStatus, MaybeThreadPool, LadduError, AdamConfig>,
        /// The number of threads to use (`0` uses the global default thread count)
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
        /// The number of threads to use (`0` uses the global default thread count)
        num_threads: usize,
    },
    /// Settings for the Particle Swarm Optimimization algorithm
    PSO {
        /// The configuration struct
        config: PSOConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<PSO, P, SwarmStatus, MaybeThreadPool, LadduError, PSOConfig>,
        /// The number of threads to use (`0` uses the global default thread count)
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
        /// The number of threads to use (`0` uses the global default thread count)
        num_threads: usize,
    },
    /// Settings for the Ensemble Slice Sampler
    ESS {
        /// The configuration struct
        config: ESSConfig,
        /// Callbacks to apply to the algorithm
        callbacks: Callbacks<ESS, P, EnsembleStatus, MaybeThreadPool, LadduError, ESSConfig>,
        /// The number of threads to use (`0` uses the global default thread count)
        num_threads: usize,
    },
}

#[allow(dead_code)]
mod allowed_keys {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct AllowedKeysByMethod {
        pub(crate) method: &'static str,
        pub(crate) keys: &'static [&'static str],
    }

    pub(crate) const MINIMIZATION_TOP_LEVEL: &[&str] = &[
        "bounds",
        "method",
        "observers",
        "terminators",
        "max_steps",
        "debug",
        "threads",
        "settings",
    ];

    pub(crate) const MCMC_TOP_LEVEL: &[&str] = &[
        "bounds",
        "method",
        "observers",
        "terminators",
        "max_steps",
        "debug",
        "threads",
        "settings",
    ];

    pub(crate) const LBFGSB_SETTINGS: &[&str] = &[
        "m",
        "skip_hessian",
        "line_search",
        "eps_f",
        "eps_g",
        "eps_norm_g",
    ];

    pub(crate) const ADAM_SETTINGS: &[&str] = &[
        "alpha", "beta_1", "beta_2", "epsilon", "beta_c", "eps_loss", "patience",
    ];

    pub(crate) const NELDER_MEAD_SETTINGS: &[&str] = &[
        "simplex_construction_method",
        "orthogonal_multiplier",
        "orthogonal_zero_step",
        "simplex_size",
        "simplex",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "adaptive",
        "expansion_method",
        "eps_f",
        "f_terminator",
        "eps_x",
        "x_terminator",
    ];

    pub(crate) const PSO_SETTINGS: &[&str] = &[
        "swarm_position_initializer",
        "swarm_position_bounds",
        "swarm_size",
        "swarm",
        "swarm_topology",
        "swarm_update_method",
        "swarm_boundary_method",
        "swarm_velocity_bounds",
        "omega",
        "c1",
        "c2",
        "use_transform",
    ];

    pub(crate) const AIES_SETTINGS: &[&str] = &["moves"];

    pub(crate) const ESS_SETTINGS: &[&str] = &["moves", "n_adaptive", "mu", "max_steps"];

    pub(crate) const MINIMIZATION_SETTINGS: &[AllowedKeysByMethod] = &[
        AllowedKeysByMethod {
            method: "lbfgsb",
            keys: LBFGSB_SETTINGS,
        },
        AllowedKeysByMethod {
            method: "adam",
            keys: ADAM_SETTINGS,
        },
        AllowedKeysByMethod {
            method: "neldermead",
            keys: NELDER_MEAD_SETTINGS,
        },
        AllowedKeysByMethod {
            method: "pso",
            keys: PSO_SETTINGS,
        },
    ];

    pub(crate) const MCMC_SETTINGS: &[AllowedKeysByMethod] = &[
        AllowedKeysByMethod {
            method: "aies",
            keys: AIES_SETTINGS,
        },
        AllowedKeysByMethod {
            method: "ess",
            keys: ESS_SETTINGS,
        },
    ];

    pub(crate) const LINE_SEARCH_SETTINGS: &[AllowedKeysByMethod] = &[
        AllowedKeysByMethod {
            method: "morethuente",
            keys: &["method", "max_iterations", "max_zoom", "c1", "c2"],
        },
        AllowedKeysByMethod {
            method: "hagerzhang",
            keys: &[
                "method",
                "max_iterations",
                "max_bisects",
                "delta",
                "sigma",
                "epsilon",
                "theta",
                "gamma",
            ],
        },
    ];

    pub(crate) const SIMPLEX_CONSTRUCTION_SETTINGS: &[AllowedKeysByMethod] = &[
        AllowedKeysByMethod {
            method: "scaledorthogonal",
            keys: &[
                "simplex_construction_method",
                "orthogonal_multiplier",
                "orthogonal_zero_step",
            ],
        },
        AllowedKeysByMethod {
            method: "orthogonal",
            keys: &["simplex_construction_method", "simplex_size"],
        },
        AllowedKeysByMethod {
            method: "custom",
            keys: &["simplex_construction_method", "simplex"],
        },
    ];

    pub(crate) const SWARM_POSITION_INITIALIZER_SETTINGS: &[AllowedKeysByMethod] = &[
        AllowedKeysByMethod {
            method: "randominlimits",
            keys: &[
                "swarm_position_initializer",
                "swarm_position_bounds",
                "swarm_size",
            ],
        },
        AllowedKeysByMethod {
            method: "latinhypercube",
            keys: &[
                "swarm_position_initializer",
                "swarm_position_bounds",
                "swarm_size",
            ],
        },
        AllowedKeysByMethod {
            method: "custom",
            keys: &["swarm_position_initializer", "swarm"],
        },
    ];

    pub(crate) const AIES_MOVE_SETTINGS: &[AllowedKeysByMethod] = &[
        AllowedKeysByMethod {
            method: "stretch",
            keys: &["a"],
        },
        AllowedKeysByMethod {
            method: "walk",
            keys: &[],
        },
    ];

    pub(crate) const ESS_MOVE_SETTINGS: &[AllowedKeysByMethod] = &[
        AllowedKeysByMethod {
            method: "differential",
            keys: &[],
        },
        AllowedKeysByMethod {
            method: "gaussian",
            keys: &[],
        },
        AllowedKeysByMethod {
            method: "global",
            keys: &["scale", "rescale_cov", "n_components"],
        },
    ];

    fn keys_for_method(
        allowed_keys_by_method: &'static [AllowedKeysByMethod],
        method: &str,
    ) -> Option<&'static [&'static str]> {
        allowed_keys_by_method
            .iter()
            .find(|allowed| allowed.method == method)
            .map(|allowed| allowed.keys)
    }

    pub(crate) fn minimization_settings(method: &str) -> Option<&'static [&'static str]> {
        keys_for_method(MINIMIZATION_SETTINGS, method)
    }

    pub(crate) fn mcmc_settings(method: &str) -> Option<&'static [&'static str]> {
        keys_for_method(MCMC_SETTINGS, method)
    }

    pub(crate) fn line_search_settings(method: &str) -> Option<&'static [&'static str]> {
        keys_for_method(LINE_SEARCH_SETTINGS, method)
    }

    pub(crate) fn simplex_construction_settings(method: &str) -> Option<&'static [&'static str]> {
        keys_for_method(SIMPLEX_CONSTRUCTION_SETTINGS, method)
    }

    pub(crate) fn swarm_position_initializer_settings(
        method: &str,
    ) -> Option<&'static [&'static str]> {
        keys_for_method(SWARM_POSITION_INITIALIZER_SETTINGS, method)
    }

    pub(crate) fn aies_move_settings(method: &str) -> Option<&'static [&'static str]> {
        keys_for_method(AIES_MOVE_SETTINGS, method)
    }

    pub(crate) fn ess_move_settings(method: &str) -> Option<&'static [&'static str]> {
        keys_for_method(ESS_MOVE_SETTINGS, method)
    }
}

/// A wrapper for the requested thread-count policy used by optimization callbacks.
#[derive(Clone, Copy, Debug)]
pub struct MaybeThreadPool {
    requested_threads: Option<usize>,
}

impl MaybeThreadPool {
    fn new(num_threads: usize) -> Self {
        Self {
            requested_threads: Some(num_threads),
        }
    }

    fn install<R: Send>(&self, op: impl FnOnce() -> LadduResult<R> + Send) -> LadduResult<R> {
        ThreadPoolManager::shared().install(self.requested_threads, op)?
    }
}
/// A trait for objects which can be used in multithreaded contexts
pub trait Threadable {
    /// Returns a [`MaybeThreadPool`] associated with the object.
    ///
    /// # Errors
    ///
    /// This will return an error if there is any problem constructing the underlying thread pool
    /// on first use.
    fn get_pool(&self) -> LadduResult<MaybeThreadPool>;
}

impl<P> MinimizationSettings<P> {
    fn num_threads(&self) -> usize {
        match self {
            Self::LBFGSB { num_threads, .. }
            | Self::Adam { num_threads, .. }
            | Self::NelderMead { num_threads, .. }
            | Self::PSO { num_threads, .. } => *num_threads,
        }
    }
}

impl<P> Threadable for MinimizationSettings<P> {
    fn get_pool(&self) -> LadduResult<MaybeThreadPool> {
        Ok(MaybeThreadPool::new(self.num_threads()))
    }
}

impl<P> MCMCSettings<P> {
    fn num_threads(&self) -> usize {
        match self {
            Self::AIES { num_threads, .. } | Self::ESS { num_threads, .. } => *num_threads,
        }
    }
}

impl<P> Threadable for MCMCSettings<P> {
    fn get_pool(&self) -> LadduResult<MaybeThreadPool> {
        Ok(MaybeThreadPool::new(self.num_threads()))
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
        .with_parameter_names(self.free_parameters()))
    }

    /// Run an MCMC sampling algorithm over the [`NLL`] with the given [`MCMCSettings`].
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
        .with_parameter_names(self.free_parameters()))
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
        .with_parameter_names(self.free_parameters()))
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
        .with_parameter_names(self.free_parameters()))
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
        .with_parameter_names(self.free_parameters()))
    }

    /// Run an MCMC sampling algorithm over the [`LikelihoodEvaluator`] with the given [`MCMCSettings`].
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
        .with_parameter_names(self.free_parameters()))
    }
}

/// Python bindings for the [`ganesh`] crate
#[cfg(feature = "python")]
pub mod py_ganesh {
    use std::{ops::ControlFlow, sync::Arc};

    use super::*;
    use allowed_keys::{
        aies_move_settings, ess_move_settings, line_search_settings, mcmc_settings,
        minimization_settings, MCMC_TOP_LEVEL, MINIMIZATION_TOP_LEVEL,
    };

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
    use laddu_core::{f64, LadduError, ReadWrite};
    use nalgebra::DMatrix;
    use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
    use parking_lot::Mutex;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{PyBytes, PyDict, PyList},
        Borrowed, PyClass,
    };

    #[derive(Clone)]
    enum LineSearchSpec {
        MoreThuente {
            max_iterations: Option<usize>,
            max_zoom: Option<usize>,
            c1: Option<f64>,
            c2: Option<f64>,
        },
        HagerZhang {
            max_iterations: Option<usize>,
            max_bisects: Option<usize>,
            delta: Option<f64>,
            sigma: Option<f64>,
            epsilon: Option<f64>,
            theta: Option<f64>,
            gamma: Option<f64>,
        },
    }

    #[derive(Clone)]
    enum SimplexSpec {
        ScaledOrthogonal {
            orthogonal_multiplier: f64,
            orthogonal_zero_step: f64,
        },
        Orthogonal {
            simplex_size: f64,
        },
        Custom {
            simplex: Vec<Vec<f64>>,
        },
    }

    #[derive(Clone)]
    enum ExpansionMethodSpec {
        GreedyMinimization,
        GreedyExpansion,
    }

    #[derive(Clone)]
    enum FTerminatorSpec {
        Amoeba,
        Absolute,
        StdDev,
    }

    #[derive(Clone)]
    enum XTerminatorSpec {
        Diameter,
        Higham,
        Rowan,
        Singer,
    }

    #[derive(Clone)]
    enum SwarmInitializerSpec {
        RandomInLimits {
            bounds: Vec<(f64, f64)>,
            n_particles: usize,
        },
        LatinHypercube {
            bounds: Vec<(f64, f64)>,
            n_particles: usize,
        },
        Custom {
            swarm: Vec<Vec<f64>>,
        },
    }

    #[derive(Clone)]
    enum AIESMoveSpec {
        Stretch { a: f64, weight: f64 },
        Walk { weight: f64 },
    }

    #[derive(Clone)]
    enum ESSMoveSpec {
        Differential {
            weight: f64,
        },
        Gaussian {
            weight: f64,
        },
        Global {
            weight: f64,
            scale: Option<f64>,
            rescale_cov: Option<f64>,
            n_components: Option<usize>,
        },
    }

    /// Typed line-search configuration used by `LBFGSBSettings`.
    #[pyclass(name = "LineSearchConfig", module = "laddu")]
    #[derive(Clone)]
    pub struct PyLineSearchConfig(LineSearchSpec);

    #[pymethods]
    impl PyLineSearchConfig {
        #[staticmethod]
        #[pyo3(signature = (*, max_iterations=None, max_zoom=None, c1=None, c2=None))]
        fn morethuente(
            max_iterations: Option<usize>,
            max_zoom: Option<usize>,
            c1: Option<f64>,
            c2: Option<f64>,
        ) -> Self {
            Self(LineSearchSpec::MoreThuente {
                max_iterations,
                max_zoom,
                c1,
                c2,
            })
        }

        #[staticmethod]
        #[pyo3(signature = (*, max_iterations=None, max_bisects=None, delta=None, sigma=None, epsilon=None, theta=None, gamma=None))]
        fn hagerzhang(
            max_iterations: Option<usize>,
            max_bisects: Option<usize>,
            delta: Option<f64>,
            sigma: Option<f64>,
            epsilon: Option<f64>,
            theta: Option<f64>,
            gamma: Option<f64>,
        ) -> Self {
            Self(LineSearchSpec::HagerZhang {
                max_iterations,
                max_bisects,
                delta,
                sigma,
                epsilon,
                theta,
                gamma,
            })
        }
    }

    /// Typed simplex-construction configuration used by `NelderMeadSettings`.
    #[pyclass(name = "SimplexConfig", module = "laddu")]
    #[derive(Clone)]
    pub struct PySimplexConfig(SimplexSpec);

    #[pymethods]
    impl PySimplexConfig {
        #[staticmethod]
        #[pyo3(signature = (*, orthogonal_multiplier=1.05, orthogonal_zero_step=0.00025))]
        fn scaled_orthogonal(orthogonal_multiplier: f64, orthogonal_zero_step: f64) -> Self {
            Self(SimplexSpec::ScaledOrthogonal {
                orthogonal_multiplier,
                orthogonal_zero_step,
            })
        }

        #[staticmethod]
        #[pyo3(signature = (*, simplex_size=1.0))]
        fn orthogonal(simplex_size: f64) -> Self {
            Self(SimplexSpec::Orthogonal { simplex_size })
        }

        #[staticmethod]
        fn custom(simplex: Vec<Vec<f64>>) -> Self {
            Self(SimplexSpec::Custom { simplex })
        }
    }

    /// Typed swarm-initializer configuration used by `PSOSettings`.
    #[pyclass(name = "SwarmInitializerConfig", module = "laddu")]
    #[derive(Clone)]
    pub struct PySwarmInitializerConfig(SwarmInitializerSpec);

    #[pymethods]
    impl PySwarmInitializerConfig {
        #[staticmethod]
        fn random_in_limits(bounds: Vec<(f64, f64)>, n_particles: usize) -> Self {
            Self(SwarmInitializerSpec::RandomInLimits {
                bounds,
                n_particles,
            })
        }

        #[staticmethod]
        fn latin_hypercube(bounds: Vec<(f64, f64)>, n_particles: usize) -> Self {
            Self(SwarmInitializerSpec::LatinHypercube {
                bounds,
                n_particles,
            })
        }

        #[staticmethod]
        fn custom(swarm: Vec<Vec<f64>>) -> Self {
            Self(SwarmInitializerSpec::Custom { swarm })
        }
    }

    /// Typed AIES move configuration used by `AIESSettings`.
    #[pyclass(name = "AIESMoveConfig", module = "laddu")]
    #[derive(Clone)]
    pub struct PyAIESMoveConfig(AIESMoveSpec);

    #[pymethods]
    impl PyAIESMoveConfig {
        #[staticmethod]
        #[pyo3(signature = (weight, *, a=2.0))]
        fn stretch(weight: f64, a: f64) -> Self {
            Self(AIESMoveSpec::Stretch { a, weight })
        }

        #[staticmethod]
        fn walk(weight: f64) -> Self {
            Self(AIESMoveSpec::Walk { weight })
        }
    }

    /// Typed ESS move configuration used by `ESSSettings`.
    #[pyclass(name = "ESSMoveConfig", module = "laddu")]
    #[derive(Clone)]
    pub struct PyESSMoveConfig(ESSMoveSpec);

    #[pymethods]
    impl PyESSMoveConfig {
        #[staticmethod]
        fn differential(weight: f64) -> Self {
            Self(ESSMoveSpec::Differential { weight })
        }

        #[staticmethod]
        fn gaussian(weight: f64) -> Self {
            Self(ESSMoveSpec::Gaussian { weight })
        }

        #[staticmethod]
        #[pyo3(signature = (weight, *, scale=None, rescale_cov=None, n_components=None))]
        fn global_(
            weight: f64,
            scale: Option<f64>,
            rescale_cov: Option<f64>,
            n_components: Option<usize>,
        ) -> Self {
            Self(ESSMoveSpec::Global {
                weight,
                scale,
                rescale_cov,
                n_components,
            })
        }
    }

    /// Typed settings for the `lbfgsb` minimizer.
    #[pyclass(name = "LBFGSBSettings", module = "laddu")]
    #[derive(Clone, Default)]
    pub struct PyLBFGSBSettings {
        m: Option<usize>,
        skip_hessian: bool,
        line_search: Option<LineSearchSpec>,
        eps_f: Option<f64>,
        eps_g: Option<f64>,
        eps_norm_g: Option<f64>,
    }

    #[pymethods]
    impl PyLBFGSBSettings {
        #[new]
        #[pyo3(signature = (*, m=None, skip_hessian=false, line_search=None, eps_f=None, eps_g=None, eps_norm_g=None))]
        fn new(
            m: Option<usize>,
            skip_hessian: bool,
            line_search: Option<PyRef<'_, PyLineSearchConfig>>,
            eps_f: Option<f64>,
            eps_g: Option<f64>,
            eps_norm_g: Option<f64>,
        ) -> Self {
            Self {
                m,
                skip_hessian,
                line_search: line_search.map(|line_search| line_search.0.clone()),
                eps_f,
                eps_g,
                eps_norm_g,
            }
        }
    }

    /// Typed settings for the `adam` minimizer.
    #[pyclass(name = "AdamSettings", module = "laddu")]
    #[derive(Clone, Default)]
    pub struct PyAdamSettings {
        alpha: Option<f64>,
        beta_1: Option<f64>,
        beta_2: Option<f64>,
        epsilon: Option<f64>,
        beta_c: Option<f64>,
        eps_loss: Option<f64>,
        patience: Option<usize>,
    }

    #[pymethods]
    impl PyAdamSettings {
        #[new]
        #[pyo3(signature = (*, alpha=None, beta_1=None, beta_2=None, epsilon=None, beta_c=None, eps_loss=None, patience=None))]
        fn new(
            alpha: Option<f64>,
            beta_1: Option<f64>,
            beta_2: Option<f64>,
            epsilon: Option<f64>,
            beta_c: Option<f64>,
            eps_loss: Option<f64>,
            patience: Option<usize>,
        ) -> Self {
            Self {
                alpha,
                beta_1,
                beta_2,
                epsilon,
                beta_c,
                eps_loss,
                patience,
            }
        }
    }

    /// Typed settings for the `nelder-mead` minimizer.
    #[pyclass(name = "NelderMeadSettings", module = "laddu")]
    #[derive(Clone, Default)]
    pub struct PyNelderMeadSettings {
        simplex: Option<SimplexSpec>,
        alpha: Option<f64>,
        beta: Option<f64>,
        gamma: Option<f64>,
        delta: Option<f64>,
        adaptive: bool,
        expansion_method: Option<ExpansionMethodSpec>,
        eps_f: Option<f64>,
        f_terminator: Option<FTerminatorSpec>,
        eps_x: Option<f64>,
        x_terminator: Option<XTerminatorSpec>,
    }

    #[pymethods]
    impl PyNelderMeadSettings {
        #[new]
        #[pyo3(signature = (*, simplex=None, alpha=None, beta=None, gamma=None, delta=None, adaptive=false, expansion_method=None, eps_f=None, f_terminator=None, eps_x=None, x_terminator=None))]
        fn new(
            simplex: Option<PyRef<'_, PySimplexConfig>>,
            alpha: Option<f64>,
            beta: Option<f64>,
            gamma: Option<f64>,
            delta: Option<f64>,
            adaptive: bool,
            expansion_method: Option<String>,
            eps_f: Option<f64>,
            f_terminator: Option<String>,
            eps_x: Option<f64>,
            x_terminator: Option<String>,
        ) -> PyResult<Self> {
            Ok(Self {
                simplex: simplex.map(|simplex| simplex.0.clone()),
                alpha,
                beta,
                gamma,
                delta,
                adaptive,
                expansion_method: expansion_method
                    .map(|expansion_method| parse_expansion_method(&expansion_method))
                    .transpose()?,
                eps_f,
                f_terminator: f_terminator
                    .map(|f_terminator| parse_f_terminator(&f_terminator))
                    .transpose()?,
                eps_x,
                x_terminator: x_terminator
                    .map(|x_terminator| parse_x_terminator(&x_terminator))
                    .transpose()?,
            })
        }
    }

    /// Typed settings for the `pso` minimizer.
    #[pyclass(name = "PSOSettings", module = "laddu")]
    #[derive(Clone)]
    pub struct PyPSOSettings {
        initializer: SwarmInitializerSpec,
        topology: SwarmTopology,
        update_method: SwarmUpdateMethod,
        boundary_method: SwarmBoundaryMethod,
        use_transform: bool,
        swarm_velocity_bounds: Option<Vec<(f64, f64)>>,
        omega: Option<f64>,
        c1: Option<f64>,
        c2: Option<f64>,
    }

    #[pymethods]
    impl PyPSOSettings {
        #[new]
        #[pyo3(signature = (initializer, *, swarm_topology="global".to_string(), swarm_update_method="sync".to_string(), swarm_boundary_method="inf".to_string(), use_transform=false, swarm_velocity_bounds=None, omega=None, c1=None, c2=None))]
        fn new(
            initializer: PyRef<'_, PySwarmInitializerConfig>,
            swarm_topology: String,
            swarm_update_method: String,
            swarm_boundary_method: String,
            use_transform: bool,
            swarm_velocity_bounds: Option<Vec<(f64, f64)>>,
            omega: Option<f64>,
            c1: Option<f64>,
            c2: Option<f64>,
        ) -> PyResult<Self> {
            Ok(Self {
                initializer: initializer.0.clone(),
                topology: parse_swarm_topology(&swarm_topology)?,
                update_method: parse_swarm_update_method(&swarm_update_method)?,
                boundary_method: parse_swarm_boundary_method(&swarm_boundary_method)?,
                use_transform,
                swarm_velocity_bounds,
                omega,
                c1,
                c2,
            })
        }
    }

    /// Typed settings for the `aies` sampler.
    #[pyclass(name = "AIESSettings", module = "laddu")]
    #[derive(Clone, Default)]
    pub struct PyAIESSettings {
        moves: Vec<AIESMoveSpec>,
    }

    #[pymethods]
    impl PyAIESSettings {
        #[new]
        #[pyo3(signature = (*, moves=None))]
        fn new(moves: Option<Vec<Py<PyAIESMoveConfig>>>, py: Python<'_>) -> PyResult<Self> {
            let moves = moves
                .unwrap_or_default()
                .into_iter()
                .map(|move_config| Ok(move_config.bind(py).borrow().0.clone()))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Self { moves })
        }
    }

    /// Typed settings for the `ess` sampler.
    #[pyclass(name = "ESSSettings", module = "laddu")]
    #[derive(Clone, Default)]
    pub struct PyESSSettings {
        moves: Vec<ESSMoveSpec>,
        n_adaptive: Option<usize>,
        mu: Option<f64>,
        max_steps: Option<usize>,
    }

    #[pymethods]
    impl PyESSSettings {
        #[new]
        #[pyo3(signature = (*, moves=None, n_adaptive=None, mu=None, max_steps=None))]
        fn new(
            moves: Option<Vec<Py<PyESSMoveConfig>>>,
            n_adaptive: Option<usize>,
            mu: Option<f64>,
            max_steps: Option<usize>,
            py: Python<'_>,
        ) -> PyResult<Self> {
            let moves = moves
                .unwrap_or_default()
                .into_iter()
                .map(|move_config| Ok(move_config.bind(py).borrow().0.clone()))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Self {
                moves,
                n_adaptive,
                mu,
                max_steps,
            })
        }
    }

    fn normalize_method_name(method: &str) -> String {
        method
            .to_lowercase()
            .trim()
            .replace("-", "")
            .replace(" ", "")
    }

    fn normalize_key_for_match(key: &str) -> String {
        key.to_ascii_lowercase()
    }

    fn levenshtein_distance(left: &str, right: &str) -> usize {
        let right_len = right.chars().count();
        let mut previous: Vec<usize> = (0..=right_len).collect();
        let mut current = vec![0; right_len + 1];

        for (left_index, left_char) in left.chars().enumerate() {
            current[0] = left_index + 1;
            for (right_index, right_char) in right.chars().enumerate() {
                let substitution_cost = usize::from(left_char != right_char);
                current[right_index + 1] = (current[right_index] + 1)
                    .min(previous[right_index + 1] + 1)
                    .min(previous[right_index] + substitution_cost);
            }
            std::mem::swap(&mut previous, &mut current);
        }

        previous[right_len]
    }

    fn max_suggestion_distance(key_len: usize) -> usize {
        match key_len {
            0..=4 => 1,
            5..=10 => 2,
            _ => 3,
        }
    }

    fn best_allowed_key_match<'a>(key: &str, allowed: &'a [&str]) -> Option<&'a str> {
        let normalized_key = normalize_key_for_match(key);
        let mut best_match = None;
        let mut best_distance = usize::MAX;
        let mut is_tied = false;

        for candidate in allowed {
            let distance =
                levenshtein_distance(&normalized_key, &normalize_key_for_match(candidate));
            if distance < best_distance {
                best_distance = distance;
                best_match = Some(*candidate);
                is_tied = false;
            } else if distance == best_distance {
                is_tied = true;
            }
        }

        if is_tied || best_distance > max_suggestion_distance(normalized_key.chars().count()) {
            None
        } else {
            best_match
        }
    }

    fn allowed_keys_message(allowed: &[&str]) -> String {
        allowed
            .iter()
            .map(|key| format!("'{key}'"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn unknown_key_error(context: &str, key: &str, allowed: &[&str]) -> PyErr {
        let suggestion = best_allowed_key_match(key, allowed)
            .map(|candidate| format!(" Did you mean '{candidate}'?"))
            .unwrap_or_default();
        PyValueError::new_err(format!(
            "Unknown key '{key}' in {context}.{suggestion} Allowed keys: {}",
            allowed_keys_message(allowed)
        ))
    }

    fn validate_allowed_dict_keys(
        dict: &Bound<PyDict>,
        context: &str,
        allowed: &[&str],
    ) -> PyResult<()> {
        for (key, _) in dict.iter() {
            let key: String = key.extract().map_err(|_| {
                PyTypeError::new_err(format!("All keys in {context} must be strings."))
            })?;
            if !allowed.contains(&key.as_str()) {
                return Err(unknown_key_error(context, &key, allowed));
            }
        }
        Ok(())
    }

    fn parse_expansion_method(method: &str) -> PyResult<ExpansionMethodSpec> {
        match normalize_method_name(method).as_str() {
            "greedyminimization" => Ok(ExpansionMethodSpec::GreedyMinimization),
            "greedyexpansion" => Ok(ExpansionMethodSpec::GreedyExpansion),
            _ => Err(PyValueError::new_err(format!(
                "Invalid expansion method: {method}"
            ))),
        }
    }

    fn parse_f_terminator(method: &str) -> PyResult<FTerminatorSpec> {
        match normalize_method_name(method).as_str() {
            "amoeba" => Ok(FTerminatorSpec::Amoeba),
            "absolute" => Ok(FTerminatorSpec::Absolute),
            "stddev" => Ok(FTerminatorSpec::StdDev),
            _ => Err(PyValueError::new_err(format!(
                "Invalid f_terminator: {method}"
            ))),
        }
    }

    fn parse_x_terminator(method: &str) -> PyResult<XTerminatorSpec> {
        match normalize_method_name(method).as_str() {
            "diameter" => Ok(XTerminatorSpec::Diameter),
            "higham" => Ok(XTerminatorSpec::Higham),
            "rowan" => Ok(XTerminatorSpec::Rowan),
            "singer" => Ok(XTerminatorSpec::Singer),
            _ => Err(PyValueError::new_err(format!(
                "Invalid x_terminator: {method}"
            ))),
        }
    }

    fn parse_swarm_topology(topology: &str) -> PyResult<SwarmTopology> {
        match normalize_method_name(topology).as_str() {
            "global" => Ok(SwarmTopology::Global),
            "ring" => Ok(SwarmTopology::Ring),
            _ => Err(PyValueError::new_err(format!(
                "Invalid swarm_topology: {topology}"
            ))),
        }
    }

    fn parse_swarm_update_method(method: &str) -> PyResult<SwarmUpdateMethod> {
        match normalize_method_name(method).as_str() {
            "sync" | "synchronous" => Ok(SwarmUpdateMethod::Synchronous),
            "async" | "asynchronous" => Ok(SwarmUpdateMethod::Asynchronous),
            _ => Err(PyValueError::new_err(format!(
                "Invalid swarm_update_method: {method}"
            ))),
        }
    }

    fn parse_swarm_boundary_method(method: &str) -> PyResult<SwarmBoundaryMethod> {
        match normalize_method_name(method).as_str() {
            "inf" => Ok(SwarmBoundaryMethod::Inf),
            "shr" => Ok(SwarmBoundaryMethod::Shr),
            _ => Err(PyValueError::new_err(format!(
                "Invalid swarm_boundary_method: {method}"
            ))),
        }
    }

    fn build_line_search(spec: &LineSearchSpec) -> StrongWolfeLineSearch {
        match spec {
            LineSearchSpec::MoreThuente {
                max_iterations,
                max_zoom,
                c1,
                c2,
            } => {
                let mut line_search = MoreThuenteLineSearch::default();
                if let Some(max_iterations) = max_iterations {
                    line_search = line_search.with_max_iterations(*max_iterations);
                }
                if let Some(max_zoom) = max_zoom {
                    line_search = line_search.with_max_zoom(*max_zoom);
                }
                match (c1, c2) {
                    (Some(c1), Some(c2)) => {
                        line_search = line_search.with_c1_c2(*c1, *c2);
                    }
                    (Some(c1), None) => {
                        line_search = line_search.with_c1(*c1);
                    }
                    (None, Some(c2)) => {
                        line_search = line_search.with_c2(*c2);
                    }
                    (None, None) => {}
                }
                StrongWolfeLineSearch::MoreThuente(line_search)
            }
            LineSearchSpec::HagerZhang {
                max_iterations,
                max_bisects,
                delta,
                sigma,
                epsilon,
                theta,
                gamma,
            } => {
                let mut line_search = HagerZhangLineSearch::default();
                if let Some(max_iterations) = max_iterations {
                    line_search = line_search.with_max_iterations(*max_iterations);
                }
                if let Some(max_bisects) = max_bisects {
                    line_search = line_search.with_max_bisects(*max_bisects);
                }
                match (delta, sigma) {
                    (Some(delta), Some(sigma)) => {
                        line_search = line_search.with_delta_sigma(*delta, *sigma);
                    }
                    (Some(delta), None) => {
                        line_search = line_search.with_delta(*delta);
                    }
                    (None, Some(sigma)) => {
                        line_search = line_search.with_sigma(*sigma);
                    }
                    (None, None) => {}
                }
                if let Some(epsilon) = epsilon {
                    line_search = line_search.with_epsilon(*epsilon);
                }
                if let Some(theta) = theta {
                    line_search = line_search.with_theta(*theta);
                }
                if let Some(gamma) = gamma {
                    line_search = line_search.with_gamma(*gamma);
                }
                StrongWolfeLineSearch::HagerZhang(line_search)
            }
        }
    }

    fn build_simplex(args: &[f64], spec: &SimplexSpec) -> PyResult<SimplexConstructionMethod> {
        match spec {
            SimplexSpec::ScaledOrthogonal {
                orthogonal_multiplier,
                orthogonal_zero_step,
            } => Ok(SimplexConstructionMethod::custom_scaled_orthogonal(
                &args.to_vec(),
                *orthogonal_multiplier,
                *orthogonal_zero_step,
            )),
            SimplexSpec::Orthogonal { simplex_size } => Ok(
                SimplexConstructionMethod::custom_orthogonal(&args.to_vec(), *simplex_size),
            ),
            SimplexSpec::Custom { simplex } => {
                if simplex.len() != args.len() {
                    return Err(PyValueError::new_err(format!(
                        "Expected {} additional simplex points, got {}.",
                        args.len(),
                        simplex.len()
                    )));
                }
                let simplex: Vec<DVector<f64>> = std::iter::once(DVector::from_vec(args.to_vec()))
                    .chain(simplex.iter().cloned().map(DVector::from_vec))
                    .collect();
                Ok(SimplexConstructionMethod::custom(simplex))
            }
        }
    }

    fn build_swarm_initializer(
        args: &[f64],
        spec: &SwarmInitializerSpec,
    ) -> SwarmPositionInitializer {
        match spec {
            SwarmInitializerSpec::RandomInLimits {
                bounds,
                n_particles,
            } => SwarmPositionInitializer::RandomInLimits {
                bounds: bounds.clone(),
                n_particles: *n_particles,
            },
            SwarmInitializerSpec::LatinHypercube {
                bounds,
                n_particles,
            } => SwarmPositionInitializer::LatinHypercube {
                bounds: bounds.clone(),
                n_particles: *n_particles,
            },
            SwarmInitializerSpec::Custom { swarm } => SwarmPositionInitializer::Custom(
                swarm
                    .iter()
                    .cloned()
                    .chain(std::iter::once(args.to_vec()))
                    .map(DVector::from_vec)
                    .collect(),
            ),
        }
    }

    fn build_aies_moves(moves: &[AIESMoveSpec]) -> Vec<(AIESMove, f64)> {
        moves
            .iter()
            .map(|move_spec| match move_spec {
                AIESMoveSpec::Stretch { a, weight } => (AIESMove::Stretch { a: *a }, *weight),
                AIESMoveSpec::Walk { weight } => AIESMove::walk(*weight),
            })
            .collect()
    }

    fn build_ess_moves(moves: &[ESSMoveSpec]) -> Vec<(ESSMove, f64)> {
        moves
            .iter()
            .map(|move_spec| match move_spec {
                ESSMoveSpec::Differential { weight } => ESSMove::differential(*weight),
                ESSMoveSpec::Gaussian { weight } => ESSMove::gaussian(*weight),
                ESSMoveSpec::Global {
                    weight,
                    scale,
                    rescale_cov,
                    n_components,
                } => ESSMove::global(*weight, *scale, *rescale_cov, *n_components),
            })
            .collect()
    }

    fn extract_optional_typed_settings<T>(
        settings: Option<&Bound<'_, PyAny>>,
        method: &str,
        expected_type: &str,
    ) -> PyResult<Option<T>>
    where
        T: PyClass + Clone,
    {
        let Some(settings) = settings else {
            return Ok(None);
        };
        let settings_ref: PyRef<'_, T> = settings.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "settings for method '{method}' must be {expected_type} or None"
            ))
        })?;
        Ok(Some((*settings_ref).clone()))
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
    ) -> PyResult<Vec<PythonMCMCTerminator>> {
        if let Some(terminators) = terminators {
            if let Ok(terminators) = terminators.cast::<PyList>() {
                terminators
                    .into_iter()
                    .map(|terminator| {
                        if let Ok(terminator) = terminator.extract::<PyAutocorrelationTerminator>()
                        {
                            Ok(PythonMCMCTerminator::Autocorrelation(terminator))
                        } else if let Ok(terminator) = terminator.extract::<MCMCTerminator>() {
                            Ok(PythonMCMCTerminator::UserDefined(terminator))
                        } else {
                            Err(PyValueError::new_err(
                                "The terminators must be either a single MCMCTerminator or a list of MCMCTerminators.",
                            ))
                        }
                    })
                    .collect()
            } else if let Ok(terminator) = terminators.extract::<PyAutocorrelationTerminator>() {
                Ok(vec![PythonMCMCTerminator::Autocorrelation(terminator)])
            } else if let Ok(terminator) = terminators.extract::<MCMCTerminator>() {
                Ok(vec![PythonMCMCTerminator::UserDefined(terminator)])
            } else {
                Err(PyValueError::new_err(
                    "The terminators must be either a single MCMCTerminator or a list of MCMCTerminators.",
                ))
            }
        } else {
            Ok(vec![])
        }
    }

    pub(crate) fn minimization_settings_from_python<P>(
        args: &Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<&Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<MinimizationSettings<P>>
    where
        P: Gradient<MaybeThreadPool, LadduError> + CostFunction<MaybeThreadPool, LadduError>,
    {
        let bounds = bounds.map(|bounds| {
            bounds
                .into_iter()
                .map(ganesh::traits::boundlike::Bound::from)
                .collect::<Vec<_>>()
        });
        let observers = extract_minimization_observers(observers)?;
        let terminators = extract_minimization_terminators(terminators)?;
        let method = normalize_method_name(&method);

        match method.as_str() {
            "lbfgsb" => {
                let settings = extract_optional_typed_settings::<PyLBFGSBSettings>(
                    settings,
                    &method,
                    "LBFGSBSettings",
                )?
                .unwrap_or_default();
                let mut config = LBFGSBConfig::new(args);
                if let Some(m) = settings.m {
                    config = config.with_memory_limit(m);
                }
                if settings.skip_hessian {
                    config = config.with_error_mode(LBFGSBErrorMode::Skip);
                }
                if let Some(line_search) = settings.line_search.as_ref() {
                    config = config.with_line_search(build_line_search(line_search));
                }
                if let Some(bounds) = bounds {
                    config = config.with_bounds(bounds);
                }
                let mut callbacks = Callbacks::empty()
                    .with_terminator(LBFGSBFTerminator {
                        eps_abs: settings.eps_f.unwrap_or(f64::EPSILON.powf(0.5)),
                    })
                    .with_terminator(LBFGSBGTerminator {
                        eps_abs: settings.eps_g.unwrap_or(f64::EPSILON.powf(1.0 / 3.0)),
                    })
                    .with_terminator(LBFGSBInfNormGTerminator {
                        eps_abs: settings.eps_norm_g.unwrap_or(1e-5),
                    });
                if debug {
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
                    num_threads: threads,
                })
            }
            "adam" => {
                let settings = extract_optional_typed_settings::<PyAdamSettings>(
                    settings,
                    &method,
                    "AdamSettings",
                )?
                .unwrap_or_default();
                let mut config = AdamConfig::new(args);
                if let Some(alpha) = settings.alpha {
                    config = config.with_alpha(alpha);
                }
                if let Some(beta_1) = settings.beta_1 {
                    config = config.with_beta_1(beta_1);
                }
                if let Some(beta_2) = settings.beta_2 {
                    config = config.with_beta_2(beta_2);
                }
                if let Some(epsilon) = settings.epsilon {
                    config = config.with_epsilon(epsilon);
                }
                if let Some(bounds) = bounds {
                    config = config.with_transform(&Bounds::from(bounds))
                }
                let mut term = AdamEMATerminator::default();
                if let Some(beta_c) = settings.beta_c {
                    term.beta_c = beta_c;
                }
                if let Some(eps_loss) = settings.eps_loss {
                    term.eps_loss = eps_loss;
                }
                if let Some(patience) = settings.patience {
                    term.patience = patience;
                }
                let mut callbacks = Callbacks::empty().with_terminator(term);
                if debug {
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
                    num_threads: threads,
                })
            }
            "neldermead" => {
                let settings = extract_optional_typed_settings::<PyNelderMeadSettings>(
                    settings,
                    &method,
                    "NelderMeadSettings",
                )?
                .unwrap_or_default();
                let construction_method = if let Some(simplex) = settings.simplex.as_ref() {
                    build_simplex(args, simplex)?
                } else {
                    SimplexConstructionMethod::scaled_orthogonal(args)
                };
                let mut config = NelderMeadConfig::new_with_method(construction_method);
                if let Some(alpha) = settings.alpha {
                    config = config.with_alpha(alpha);
                }
                if let Some(beta) = settings.beta {
                    config = config.with_beta(beta);
                }
                if let Some(gamma) = settings.gamma {
                    config = config.with_gamma(gamma);
                }
                if let Some(delta) = settings.delta {
                    config = config.with_delta(delta);
                }
                if settings.adaptive {
                    config = config.with_adaptive(args.len());
                }
                if let Some(expansion_method) = settings.expansion_method.as_ref() {
                    let expansion_method = match expansion_method {
                        ExpansionMethodSpec::GreedyMinimization => {
                            SimplexExpansionMethod::GreedyMinimization
                        }
                        ExpansionMethodSpec::GreedyExpansion => {
                            SimplexExpansionMethod::GreedyMinimization
                        }
                    };
                    config = config.with_expansion_method(expansion_method);
                }
                if let Some(bounds) = bounds {
                    config = config.with_bounds(bounds);
                }
                let eps_f = settings.eps_f.unwrap_or(f64::EPSILON.powf(0.25));
                let f_terminator = match settings
                    .f_terminator
                    .as_ref()
                    .unwrap_or(&FTerminatorSpec::StdDev)
                {
                    FTerminatorSpec::Amoeba => NelderMeadFTerminator::Amoeba { eps_rel: eps_f },
                    FTerminatorSpec::Absolute => NelderMeadFTerminator::Absolute { eps_abs: eps_f },
                    FTerminatorSpec::StdDev => NelderMeadFTerminator::StdDev { eps_abs: eps_f },
                };
                let eps_x = settings.eps_x.unwrap_or(f64::EPSILON.powf(0.25));
                let x_terminator = match settings
                    .x_terminator
                    .as_ref()
                    .unwrap_or(&XTerminatorSpec::Singer)
                {
                    XTerminatorSpec::Diameter => NelderMeadXTerminator::Diameter { eps_abs: eps_x },
                    XTerminatorSpec::Higham => NelderMeadXTerminator::Higham { eps_rel: eps_x },
                    XTerminatorSpec::Rowan => NelderMeadXTerminator::Rowan { eps_rel: eps_x },
                    XTerminatorSpec::Singer => NelderMeadXTerminator::Singer { eps_rel: eps_x },
                };
                let mut callbacks = Callbacks::empty()
                    .with_terminator(f_terminator)
                    .with_terminator(x_terminator);
                if debug {
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
                    num_threads: threads,
                })
            }
            "pso" => {
                let settings = extract_optional_typed_settings::<PyPSOSettings>(
                    settings,
                    &method,
                    "PSOSettings",
                )?
                .ok_or_else(|| {
                    PyTypeError::new_err("settings for method 'pso' must be PSOSettings or None")
                })?;
                let mut swarm = Swarm::new(build_swarm_initializer(args, &settings.initializer))
                    .with_topology(settings.topology)
                    .with_update_method(settings.update_method)
                    .with_boundary_method(settings.boundary_method);
                if let Some(swarm_velocity_bounds) = settings.swarm_velocity_bounds.clone() {
                    swarm = swarm.with_velocity_initializer(
                        SwarmVelocityInitializer::RandomInLimits(swarm_velocity_bounds),
                    );
                }
                let mut config = PSOConfig::new(swarm);
                if let Some(omega) = settings.omega {
                    config = config.with_omega(omega);
                }
                if let Some(c1) = settings.c1 {
                    config = config.with_c1(c1);
                }
                if let Some(c2) = settings.c2 {
                    config = config.with_c2(c2);
                }
                if let Some(bounds) = bounds {
                    if settings.use_transform {
                        config = config.with_transform(&Bounds::from(bounds));
                    } else {
                        config = config.with_bounds(bounds);
                    }
                }
                let mut callbacks = Callbacks::empty();
                if debug {
                    return Err(PyValueError::new_err(
                        "The debug setting is not yet supported for PSO",
                    ));
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
                    num_threads: threads,
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid minimizer: {method}"
            ))),
        }
    }

    pub(crate) fn mcmc_settings_from_python<P>(
        args: &Vec<DVector<f64>>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<&Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<MCMCSettings<P>>
    where
        P: LogDensity<MaybeThreadPool, LadduError>,
    {
        let bounds = bounds.map(|bounds| {
            bounds
                .into_iter()
                .map(ganesh::traits::boundlike::Bound::from)
                .collect::<Vec<_>>()
        });
        let observers = extract_mcmc_observers(observers)?;
        let terminators = extract_mcmc_terminators(terminators)?;
        let method = normalize_method_name(&method);

        match method.as_str() {
            "aies" => {
                let settings = extract_optional_typed_settings::<PyAIESSettings>(
                    settings,
                    &method,
                    "AIESSettings",
                )?
                .unwrap_or_default();
                let mut config = AIESConfig::new(args.to_vec());
                if !settings.moves.is_empty() {
                    config = config.with_moves(build_aies_moves(&settings.moves));
                }
                if let Some(bounds) = bounds {
                    config = config.with_transform(&Bounds::from(bounds))
                }
                let mut callbacks = Callbacks::empty();
                if debug {
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
                    num_threads: threads,
                })
            }
            "ess" => {
                let settings = extract_optional_typed_settings::<PyESSSettings>(
                    settings,
                    &method,
                    "ESSSettings",
                )?
                .unwrap_or_default();
                let mut config = ESSConfig::new(args.to_vec());
                if !settings.moves.is_empty() {
                    config = config.with_moves(build_ess_moves(&settings.moves));
                }
                if let Some(n_adaptive) = settings.n_adaptive {
                    config = config.with_n_adaptive(n_adaptive);
                }
                if let Some(mu) = settings.mu {
                    config = config.with_mu(mu);
                }
                if let Some(max_steps) = settings.max_steps {
                    config = config.with_max_steps(max_steps);
                }
                if let Some(bounds) = bounds {
                    config = config.with_transform(&Bounds::from(bounds))
                }
                let mut callbacks = Callbacks::empty();
                if debug {
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
                    num_threads: threads,
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid MCMC algorithm: {method}"
            ))),
        }
    }

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
                let method = normalize_method_name(&method.extract::<String>()?);
                let allowed = line_search_settings(&method).ok_or_else(|| {
                    PyTypeError::new_err(format!("Invalid line search method: {}", method))
                })?;
                validate_allowed_dict_keys(d, &format!("{method} line_search settings"), allowed)?;
                match method.as_str() {
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
                        Ok(SimplexConstructionMethod::custom_scaled_orthogonal(
                            args,
                            orthogonal_multiplier,
                            orthogonal_zero_step,
                        ))
                    }
                    "orthogonal" => {
                        let simplex_size = d
                            .get_item("simplex_size")?
                            .map(|v| v.extract())
                            .transpose()?
                            .unwrap_or(1.0);
                        Ok(SimplexConstructionMethod::custom_orthogonal(
                            args,
                            simplex_size,
                        ))
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
                            Ok(SimplexConstructionMethod::custom(simplex))
                        } else {
                            Err(PyValueError::new_err("Simplex must be specified when using the 'custom' simplex_construction_method."))
                        }
                    }
                    _ => Err(PyValueError::new_err(format!(
                        "Invalid simplex_construction_method: {}",
                        simplex_construction_method
                    ))),
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
                            Ok(SwarmPositionInitializer::RandomInLimits {
                                bounds: swarm_position_bounds.extract()?,
                                n_particles: swarm_size.extract()?,
                            })
                        } else {
                            Err(PyValueError::new_err("The swarm_position_bounds and swarm_size must be specified when using the 'randominlimits' swarm_position_initializer."))
                        }
                    }
                    "latinhypercube" => {
                        if let (Some(swarm_position_bounds), Some(swarm_size)) = (
                            d.get_item("swarm_position_bounds")?,
                            d.get_item("swarm_size")?,
                        ) {
                            Ok(SwarmPositionInitializer::LatinHypercube {
                                bounds: swarm_position_bounds.extract()?,
                                n_particles: swarm_size.extract()?,
                            })
                        } else {
                            Err(PyValueError::new_err("The swarm_position_bounds and swarm_size must be specified when using the 'latinhypercube' swarm_position_initializer."))
                        }
                    }
                    "custom" => {
                        if let Some(swarm) = d.get_item("swarm")? {
                            Ok(SwarmPositionInitializer::Custom(
                                swarm
                                    .extract::<Vec<Vec<f64>>>()?
                                    .iter()
                                    .chain(vec![args])
                                    .map(|x| DVector::from_vec(x.clone()))
                                    .collect(), // TODO: numpy arrays?
                            ))
                        } else {
                            Err(PyValueError::new_err("The swarm must be specified when using the 'custom' swarm_position_initializer."))
                        }
                    }
                    _ => Err(PyValueError::new_err(format!(
                        "Invalid swarm_position_initializer: {}",
                        swarm_position_initializer
                    ))),
                }
            } else {
                Err(PyValueError::new_err(
                    "The swarm_position_initializer must be specified for the PSO algorithm.",
                ))
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
            validate_allowed_dict_keys(d, "minimization arguments", MINIMIZATION_TOP_LEVEL)?;
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
                if let Ok(observers) = observers.cast::<PyList>() {
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
                if let Ok(terminators) = terminators.cast::<PyList>() {
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
            let settings: Bound<PyDict> = if let Some(settings) = d.get_item("settings")? {
                settings.extract()?
            } else {
                PyDict::new(d.py())
            };
            if let Some(method) = d.get_item("method")? {
                let method = normalize_method_name(&method.extract::<String>()?);
                let allowed = minimization_settings(&method).ok_or_else(|| {
                    PyValueError::new_err(format!("Invalid minimizer: {}", method))
                })?;
                validate_allowed_dict_keys(&settings, &format!("{method} settings"), allowed)?;
                match method.as_str() {
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
                let moves_list = moves.cast::<PyList>()?;
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
                        let method = normalize_method_name(&custom_move.0);
                        let allowed = aies_move_settings(&method).ok_or_else(|| {
                            PyValueError::new_err(format!("Invalid AIES move: {}", custom_move.0))
                        })?;
                        validate_allowed_dict_keys(
                            &custom_move.1,
                            &format!("{method} AIES move settings"),
                            allowed,
                        )?;
                        match method.as_str() {
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
                let moves_list = moves.cast::<PyList>()?;
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
                        let method = normalize_method_name(&custom_move.0);
                        let allowed = ess_move_settings(&method).ok_or_else(|| {
                            PyValueError::new_err(format!("Invalid ESS move: {}", custom_move.0))
                        })?;
                        validate_allowed_dict_keys(
                            &custom_move.1,
                            &format!("{method} ESS move settings"),
                            allowed,
                        )?;
                        match method.as_str() {
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
            validate_allowed_dict_keys(d, "MCMC arguments", MCMC_TOP_LEVEL)?;
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
                if let Ok(observers) = observers.cast::<PyList>() {
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
                if let Ok(terminators) = terminators.cast::<PyList>() {
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
            let settings: Bound<PyDict> = if let Some(settings) = d.get_item("settings")? {
                settings.extract()?
            } else {
                PyDict::new(d.py())
            };
            if let Some(method) = d.get_item("method")? {
                let method = normalize_method_name(&method.extract::<String>()?);
                let allowed = mcmc_settings(&method).ok_or_else(|| {
                    PyValueError::new_err(format!("Invalid MCMC algorithm: {}", method))
                })?;
                validate_allowed_dict_keys(&settings, &format!("{method} settings"), allowed)?;
                match method.as_str() {
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
        Gradient(Arc<Mutex<GradientStatus>>),
        GradientFree(Arc<Mutex<GradientFreeStatus>>),
        Swarm(Arc<Mutex<SwarmStatus>>),
    }
    impl MinimizationStatus {
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            match self {
                Self::Gradient(gradient_status) => {
                    gradient_status.lock().x.as_slice().to_pyarray(py)
                }
                Self::GradientFree(gradient_free_status) => {
                    gradient_free_status.lock().x.as_slice().to_pyarray(py)
                }
                Self::Swarm(swarm_status) => swarm_status.lock().gbest.x.as_slice().to_pyarray(py),
            }
        }
        fn fx(&self) -> f64 {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().fx,
                Self::GradientFree(gradient_free_status) => gradient_free_status.lock().fx,
                Self::Swarm(swarm_status) => swarm_status.lock().gbest.fx.unwrap_or(f64::NAN),
            }
        }
        fn message(&self) -> String {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().message().to_string(),
                Self::GradientFree(gradient_free_status) => {
                    gradient_free_status.lock().message().to_string()
                }
                Self::Swarm(swarm_status) => swarm_status.lock().message().to_string(),
            }
        }
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().err.clone(),
                Self::GradientFree(gradient_free_status) => gradient_free_status.lock().err.clone(),
                Self::Swarm(_) => None,
            }
            .map(|e| e.as_slice().to_pyarray(py))
        }
        fn n_f_evals(&self) -> usize {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().n_f_evals,
                Self::GradientFree(gradient_free_status) => gradient_free_status.lock().n_f_evals,
                Self::Swarm(swarm_status) => swarm_status.lock().n_f_evals,
            }
        }
        fn n_g_evals(&self) -> usize {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().n_g_evals,
                Self::GradientFree(_) => 0,
                Self::Swarm(_) => 0,
            }
        }
        fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().cov.clone(),
                Self::GradientFree(gradient_free_status) => gradient_free_status.lock().cov.clone(),
                Self::Swarm(_) => None,
            }
            .map(|cov| cov.to_pyarray(py))
        }
        fn hess<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().hess.clone(),
                Self::GradientFree(gradient_free_status) => {
                    gradient_free_status.lock().hess.clone()
                }
                Self::Swarm(_) => None,
            }
            .map(|hess| hess.to_pyarray(py))
        }
        fn converged(&self) -> bool {
            match self {
                Self::Gradient(gradient_status) => gradient_status.lock().converged(),
                Self::GradientFree(gradient_free_status) => gradient_free_status.lock().converged(),
                Self::Swarm(swarm_status) => swarm_status.lock().converged(),
            }
        }
        fn swarm(&self) -> Option<PySwarm> {
            match self {
                Self::Gradient(_) | Self::GradientFree(_) => None,
                Self::Swarm(swarm_status) => Some(PySwarm(swarm_status.lock().swarm.clone())),
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
                        PyMinimizationStatus(MinimizationStatus::Gradient(Arc::new(Mutex::new(
                            status.clone(),
                        )))),
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
                        PyMinimizationStatus(MinimizationStatus::GradientFree(Arc::new(
                            Mutex::new(status.clone()),
                        ))),
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
                        PyMinimizationStatus(MinimizationStatus::Swarm(Arc::new(Mutex::new(
                            status.clone(),
                        )))),
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
    /// that takes the current step and a [`PyMinimizationStatus`] as arguments and returns a
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
                let py_status = Py::new(
                    py,
                    PyMinimizationStatus(MinimizationStatus::Gradient(wrapped_status.clone())),
                )?;
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
                let py_status = Py::new(
                    py,
                    PyMinimizationStatus(MinimizationStatus::GradientFree(wrapped_status.clone())),
                )?;
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
                let py_status = Py::new(
                    py,
                    PyMinimizationStatus(MinimizationStatus::Swarm(wrapped_status.clone())),
                )?;
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
    /// Karamanis & Beutler (2021)
    ///
    /// Parameters
    /// ----------
    /// x : array_like
    ///     An array of dimension ``(n_walkers, n_steps, n_parameters)``
    /// c : float, default = 7.0
    ///     Set the time window for Sokal's autowindowing function (Sokal, 1997). If None, the default window
    ///     size of 7.0 is used.
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
            .map(|v| v.into_iter().map(DVector::from_vec).collect())
            .collect();
        integrated_autocorrelation_times(samples, c)
            .as_slice()
            .to_pyarray(py)
    }

    #[cfg_attr(doctest, doc = "```ignore")]
    /// A terminator for MCMC algorithms that monitors autocorrelation according to Karamanis & Beutler (2021).
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
    ///     Set the time window for Sokal's autowindowing function (Sokal, 1997). If None, the default window
    ///     size of 7.0 is used.
    /// verbose : bool, default=False
    ///     Print autocorrelation information at each check step.
    ///
    /// Examples
    /// --------
    /// .. code-block:: python
    ///
    ///     terminator = AutocorrelationTerminator(n_check=25, discard=0.3)
    ///     summary = evaluator.mcmc(p0, terminators=[terminator])
    ///     print(terminator.taus)
    ///
    #[cfg_attr(doctest, doc = "```")]
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

#[cfg(test)]
mod tests {
    use super::{
        allowed_keys::{
            aies_move_settings, ess_move_settings, line_search_settings, mcmc_settings,
            minimization_settings, simplex_construction_settings,
            swarm_position_initializer_settings, MCMC_TOP_LEVEL, MINIMIZATION_TOP_LEVEL,
        },
        MaybeThreadPool,
    };

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

    #[test]
    fn top_level_allowed_key_sets_match_public_entry_points() {
        assert_eq!(
            MINIMIZATION_TOP_LEVEL,
            &[
                "bounds",
                "method",
                "observers",
                "terminators",
                "max_steps",
                "debug",
                "threads",
                "settings",
            ]
        );
        assert_eq!(MCMC_TOP_LEVEL, MINIMIZATION_TOP_LEVEL);
    }

    #[test]
    fn algorithm_setting_key_sets_match_parser_usage() {
        assert_eq!(
            minimization_settings("lbfgsb"),
            Some(
                &[
                    "m",
                    "skip_hessian",
                    "line_search",
                    "eps_f",
                    "eps_g",
                    "eps_norm_g",
                ][..]
            )
        );
        assert_eq!(
            minimization_settings("adam"),
            Some(&["alpha", "beta_1", "beta_2", "epsilon", "beta_c", "eps_loss", "patience",][..])
        );
        assert_eq!(
            minimization_settings("neldermead"),
            Some(
                &[
                    "simplex_construction_method",
                    "orthogonal_multiplier",
                    "orthogonal_zero_step",
                    "simplex_size",
                    "simplex",
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "adaptive",
                    "expansion_method",
                    "eps_f",
                    "f_terminator",
                    "eps_x",
                    "x_terminator",
                ][..]
            )
        );
        assert_eq!(
            minimization_settings("pso"),
            Some(
                &[
                    "swarm_position_initializer",
                    "swarm_position_bounds",
                    "swarm_size",
                    "swarm",
                    "swarm_topology",
                    "swarm_update_method",
                    "swarm_boundary_method",
                    "swarm_velocity_bounds",
                    "omega",
                    "c1",
                    "c2",
                    "use_transform",
                ][..]
            )
        );
        assert_eq!(mcmc_settings("aies"), Some(&["moves"][..]));
        assert_eq!(
            mcmc_settings("ess"),
            Some(&["moves", "n_adaptive", "mu", "max_steps"][..])
        );
    }

    #[test]
    fn nested_setting_key_sets_match_parser_usage() {
        assert_eq!(
            line_search_settings("morethuente"),
            Some(&["method", "max_iterations", "max_zoom", "c1", "c2"][..])
        );
        assert_eq!(
            line_search_settings("hagerzhang"),
            Some(
                &[
                    "method",
                    "max_iterations",
                    "max_bisects",
                    "delta",
                    "sigma",
                    "epsilon",
                    "theta",
                    "gamma",
                ][..]
            )
        );

        assert_eq!(
            simplex_construction_settings("scaledorthogonal"),
            Some(
                &[
                    "simplex_construction_method",
                    "orthogonal_multiplier",
                    "orthogonal_zero_step",
                ][..]
            )
        );
        assert_eq!(
            simplex_construction_settings("orthogonal"),
            Some(&["simplex_construction_method", "simplex_size"][..])
        );
        assert_eq!(
            simplex_construction_settings("custom"),
            Some(&["simplex_construction_method", "simplex"][..])
        );

        assert_eq!(
            swarm_position_initializer_settings("randominlimits"),
            Some(
                &[
                    "swarm_position_initializer",
                    "swarm_position_bounds",
                    "swarm_size",
                ][..]
            )
        );
        assert_eq!(
            swarm_position_initializer_settings("latinhypercube"),
            Some(
                &[
                    "swarm_position_initializer",
                    "swarm_position_bounds",
                    "swarm_size",
                ][..]
            )
        );
        assert_eq!(
            swarm_position_initializer_settings("custom"),
            Some(&["swarm_position_initializer", "swarm"][..])
        );

        assert_eq!(aies_move_settings("stretch"), Some(&["a"][..]));
        assert_eq!(aies_move_settings("walk"), Some(&[][..]));

        assert_eq!(ess_move_settings("differential"), Some(&[][..]));
        assert_eq!(ess_move_settings("gaussian"), Some(&[][..]));
        assert_eq!(
            ess_move_settings("global"),
            Some(&["scale", "rescale_cov", "n_components"][..])
        );
    }
}
