use std::sync::Arc;

use fastrand::Rng;
use ganesh::{
    algorithms::LBFGSB,
    observers::{
        DebugMCMCObserver, DebugObserver, DebugSwarmObserver, MCMCObserver, Observer, SwarmObserver,
    },
    samplers::{ESSMove, ESS},
    swarms::PSO,
    traits::{Algorithm, MCMCAlgorithm, SwarmAlgorithm},
    Status, Swarm,
};
use laddu_core::{Ensemble, LadduError};
use parking_lot::RwLock;
#[cfg(feature = "rayon")]
use rayon::ThreadPool;

struct VerboseObserver {
    show_step: bool,
    show_x: bool,
    show_fx: bool,
}
impl VerboseObserver {
    fn build(self) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(self))
    }
}

/// A set of options that are used when minimizations are performed.
pub struct MinimizerOptions {
    #[cfg(feature = "rayon")]
    pub(crate) algorithm: Box<dyn Algorithm<ThreadPool, LadduError>>,
    #[cfg(not(feature = "rayon"))]
    pub(crate) algorithm: Box<dyn Algorithm<(), LadduError>>,
    #[cfg(feature = "rayon")]
    pub(crate) observers: Vec<Arc<RwLock<dyn Observer<ThreadPool>>>>,
    #[cfg(not(feature = "rayon"))]
    pub(crate) observers: Vec<Arc<RwLock<dyn Observer<()>>>>,
    pub(crate) max_steps: usize,
    #[cfg(feature = "rayon")]
    pub(crate) threads: usize,
}

impl Default for MinimizerOptions {
    fn default() -> Self {
        Self {
            algorithm: Box::new(LBFGSB::default()),
            observers: Default::default(),
            max_steps: 4000,
            #[cfg(all(feature = "rayon", feature = "num_cpus"))]
            threads: num_cpus::get(),
            #[cfg(all(feature = "rayon", not(feature = "num_cpus")))]
            threads: 0,
        }
    }
}

impl MinimizerOptions {
    /// Adds the [`DebugObserver`] to the minimization.
    pub fn debug(self) -> Self {
        let mut observers = self.observers;
        observers.push(DebugObserver::build());
        Self {
            algorithm: self.algorithm,
            observers,
            max_steps: self.max_steps,
            #[cfg(feature = "rayon")]
            threads: self.threads,
        }
    }
    /// Adds a customizable `VerboseObserver` to the minimization.
    pub fn verbose(self, show_step: bool, show_x: bool, show_fx: bool) -> Self {
        let mut observers = self.observers;
        observers.push(
            VerboseObserver {
                show_step,
                show_x,
                show_fx,
            }
            .build(),
        );
        Self {
            algorithm: self.algorithm,
            observers,
            max_steps: self.max_steps,
            #[cfg(feature = "rayon")]
            threads: self.threads,
        }
    }
    /// Set the [`Algorithm`] to be used in the minimization (default: [`LBFGSB`] with default
    /// settings).
    #[cfg(feature = "rayon")]
    pub fn with_algorithm<A: Algorithm<ThreadPool, LadduError> + 'static>(
        self,
        algorithm: A,
    ) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: self.observers,
            max_steps: self.max_steps,
            threads: self.threads,
        }
    }

    /// Set the [`Algorithm`] to be used in the minimization (default: [`LBFGSB`] with default
    /// settings).
    #[cfg(not(feature = "rayon"))]
    pub fn with_algorithm<A: Algorithm<(), LadduError> + 'static>(self, algorithm: A) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: self.observers,
            max_steps: self.max_steps,
        }
    }
    /// Add an [`Observer`] to the list of [`Observer`]s used in the minimization.
    #[cfg(feature = "rayon")]
    pub fn with_observer(self, observer: Arc<RwLock<dyn Observer<ThreadPool>>>) -> Self {
        let mut observers = self.observers;
        observers.push(observer.clone());
        Self {
            algorithm: self.algorithm,
            observers,
            max_steps: self.max_steps,
            threads: self.threads,
        }
    }
    /// Add an [`Observer`] to the list of [`Observer`]s used in the minimization.
    #[cfg(not(feature = "rayon"))]
    pub fn with_observer(self, observer: Arc<RwLock<dyn Observer<()>>>) -> Self {
        let mut observers = self.observers;
        observers.push(observer.clone());
        Self {
            algorithm: self.algorithm,
            observers,
            max_steps: self.max_steps,
        }
    }

    /// Set the maximum number of [`Algorithm`] steps for the minimization (default: 4000).
    pub fn with_max_steps(self, max_steps: usize) -> Self {
        Self {
            algorithm: self.algorithm,
            observers: self.observers,
            max_steps,
            #[cfg(feature = "rayon")]
            threads: self.threads,
        }
    }

    /// Set the number of threads to use.
    #[cfg(feature = "rayon")]
    pub fn with_threads(self, threads: usize) -> Self {
        Self {
            algorithm: self.algorithm,
            observers: self.observers,
            max_steps: self.max_steps,
            threads,
        }
    }
}

#[cfg(feature = "rayon")]
impl Observer<ThreadPool> for VerboseObserver {
    fn callback(&mut self, step: usize, status: &mut Status, _user_data: &mut ThreadPool) -> bool {
        if self.show_step {
            println!("Step: {}", step);
        }
        if self.show_x {
            println!("Current Best Position: {}", status.x.transpose());
        }
        if self.show_fx {
            println!("Current Best Value: {}", status.fx);
        }
        false
    }
}

impl Observer<()> for VerboseObserver {
    fn callback(&mut self, step: usize, status: &mut Status, _user_data: &mut ()) -> bool {
        if self.show_step {
            println!("Step: {}", step);
        }
        if self.show_x {
            println!("Current Best Position: {}", status.x.transpose());
        }
        if self.show_fx {
            println!("Current Best Value: {}", status.fx);
        }
        false
    }
}

struct VerboseMCMCObserver;
impl VerboseMCMCObserver {
    fn build() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self))
    }
}

#[cfg(feature = "rayon")]
impl MCMCObserver<ThreadPool> for VerboseMCMCObserver {
    fn callback(
        &mut self,
        step: usize,
        _ensemble: &mut Ensemble,
        _thread_pool: &mut ThreadPool,
    ) -> bool {
        println!("Step: {}", step);
        false
    }
}

impl MCMCObserver<()> for VerboseMCMCObserver {
    fn callback(&mut self, step: usize, _ensemble: &mut Ensemble, _user_data: &mut ()) -> bool {
        println!("Step: {}", step);
        false
    }
}

/// A set of options that are used when Markov Chain Monte Carlo samplings are performed.
pub struct MCMCOptions {
    #[cfg(feature = "rayon")]
    pub(crate) algorithm: Box<dyn MCMCAlgorithm<ThreadPool, LadduError>>,
    #[cfg(not(feature = "rayon"))]
    pub(crate) algorithm: Box<dyn MCMCAlgorithm<(), LadduError>>,
    #[cfg(feature = "rayon")]
    pub(crate) observers: Vec<Arc<RwLock<dyn MCMCObserver<ThreadPool>>>>,
    #[cfg(not(feature = "rayon"))]
    pub(crate) observers: Vec<Arc<RwLock<dyn MCMCObserver<()>>>>,
    #[cfg(feature = "rayon")]
    pub(crate) threads: usize,
}

impl MCMCOptions {
    /// Create the default set of [`MCMCOptions`], which cannot be made with a typical [`Default`]
    /// implementation because it requires an [`Rng`] to be provided.
    pub fn default_with_rng(rng: Rng) -> Self {
        let default_ess_moves = [ESSMove::differential(0.9), ESSMove::gaussian(0.1)];
        Self {
            algorithm: Box::new(ESS::new(default_ess_moves, rng).with_n_adaptive(100)),
            observers: Default::default(),
            #[cfg(all(feature = "rayon", feature = "num_cpus"))]
            threads: num_cpus::get(),
            #[cfg(all(feature = "rayon", not(feature = "num_cpus")))]
            threads: 0,
        }
    }
    /// Adds the [`DebugMCMCObserver`] to the minimization.
    pub fn debug(self) -> Self {
        let mut observers = self.observers;
        observers.push(DebugMCMCObserver::build());
        Self {
            algorithm: self.algorithm,
            observers,
            #[cfg(feature = "rayon")]
            threads: self.threads,
        }
    }
    /// Adds a customizable `VerboseObserver` to the minimization.
    pub fn verbose(self) -> Self {
        let mut observers = self.observers;
        observers.push(VerboseMCMCObserver::build());
        Self {
            algorithm: self.algorithm,
            observers,
            #[cfg(feature = "rayon")]
            threads: self.threads,
        }
    }
    /// Set the [`MCMCAlgorithm`] to be used in the minimization.
    #[cfg(feature = "rayon")]
    pub fn from_algorithm<A: MCMCAlgorithm<ThreadPool, LadduError> + 'static>(
        algorithm: A,
    ) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: Default::default(),
            #[cfg(all(feature = "rayon", feature = "num_cpus"))]
            threads: num_cpus::get(),
            #[cfg(all(feature = "rayon", not(feature = "num_cpus")))]
            threads: 0,
        }
    }
    /// Set the [`MCMCAlgorithm`] to be used in the minimization.
    #[cfg(not(feature = "rayon"))]
    pub fn from_algorithm<A: MCMCAlgorithm<(), LadduError> + 'static>(self, algorithm: A) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: Default::default(),
        }
    }
    #[cfg(feature = "rayon")]
    /// Add an [`MCMCObserver`] to the list of [`MCMCObserver`]s used in the minimization.
    pub fn with_observer(self, observer: Arc<RwLock<dyn MCMCObserver<ThreadPool>>>) -> Self {
        let mut observers = self.observers;
        observers.push(observer.clone());
        Self {
            algorithm: self.algorithm,
            observers,
            threads: self.threads,
        }
    }
    #[cfg(not(feature = "rayon"))]
    /// Add an [`MCMCObserver`] to the list of [`MCMCObserver`]s used in the minimization.
    pub fn with_observer(self, observer: Arc<RwLock<dyn MCMCObserver<()>>>) -> Self {
        let mut observers = self.observers;
        observers.push(observer.clone());
        Self {
            algorithm: self.algorithm,
            observers,
        }
    }

    /// Set the number of threads to use.
    #[cfg(feature = "rayon")]
    pub fn with_threads(self, threads: usize) -> Self {
        Self {
            algorithm: self.algorithm,
            observers: self.observers,
            threads,
        }
    }
}

struct VerboseSwarmObserver;
impl VerboseSwarmObserver {
    fn build() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self))
    }
}

#[cfg(feature = "rayon")]
impl SwarmObserver<ThreadPool> for VerboseSwarmObserver {
    fn callback(&mut self, step: usize, _swarm: &mut Swarm, _thread_pool: &mut ThreadPool) -> bool {
        println!("Step: {}", step);
        false
    }
}

impl SwarmObserver<()> for VerboseSwarmObserver {
    fn callback(&mut self, step: usize, _swarm: &mut Swarm, _user_data: &mut ()) -> bool {
        println!("Step: {}", step);
        false
    }
}

/// A set of options that are used when Markov Chain Monte Carlo samplings are performed.
pub struct SwarmOptions {
    #[cfg(feature = "rayon")]
    pub(crate) algorithm: Box<dyn SwarmAlgorithm<ThreadPool, LadduError>>,
    #[cfg(not(feature = "rayon"))]
    pub(crate) algorithm: Box<dyn SwarmAlgorithm<(), LadduError>>,
    #[cfg(feature = "rayon")]
    pub(crate) observers: Vec<Arc<RwLock<dyn SwarmObserver<ThreadPool>>>>,
    #[cfg(not(feature = "rayon"))]
    pub(crate) observers: Vec<Arc<RwLock<dyn SwarmObserver<()>>>>,
    #[cfg(feature = "rayon")]
    pub(crate) threads: usize,
}

impl SwarmOptions {
    /// Create the default set of [`SwarmOptions`], which cannot be made with a typical [`Default`]
    /// implementation because it requires an [`Rng`] to be provided.
    pub fn default_with_rng(rng: Rng) -> Self {
        Self {
            algorithm: Box::new(PSO::new(rng)),
            observers: Default::default(),
            #[cfg(all(feature = "rayon", feature = "num_cpus"))]
            threads: num_cpus::get(),
            #[cfg(all(feature = "rayon", not(feature = "num_cpus")))]
            threads: 0,
        }
    }
    /// Adds the [`DebugSwarmObserver`] to the minimization.
    pub fn debug(self) -> Self {
        let mut observers = self.observers;
        observers.push(DebugSwarmObserver::build());
        Self {
            algorithm: self.algorithm,
            observers,
            #[cfg(feature = "rayon")]
            threads: self.threads,
        }
    }
    /// Adds a customizable `VerboseObserver` to the minimization.
    pub fn verbose(self) -> Self {
        let mut observers = self.observers;
        observers.push(VerboseSwarmObserver::build());
        Self {
            algorithm: self.algorithm,
            observers,
            #[cfg(feature = "rayon")]
            threads: self.threads,
        }
    }
    /// Set the [`SwarmAlgorithm`] to be used in the minimization.
    #[cfg(feature = "rayon")]
    pub fn from_algorithm<A: SwarmAlgorithm<ThreadPool, LadduError> + 'static>(
        algorithm: A,
    ) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: Default::default(),
            #[cfg(all(feature = "rayon", feature = "num_cpus"))]
            threads: num_cpus::get(),
            #[cfg(all(feature = "rayon", not(feature = "num_cpus")))]
            threads: 0,
        }
    }
    /// Set the [`SwarmAlgorithm`] to be used in the minimization.
    #[cfg(not(feature = "rayon"))]
    pub fn from_algorithm<A: SwarmAlgorithm<(), LadduError> + 'static>(self, algorithm: A) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: Default::default(),
        }
    }
    #[cfg(feature = "rayon")]
    /// Add an [`SwarmObserver`] to the list of [`SwarmObserver`]s used in the minimization.
    pub fn with_observer(self, observer: Arc<RwLock<dyn SwarmObserver<ThreadPool>>>) -> Self {
        let mut observers = self.observers;
        observers.push(observer.clone());
        Self {
            algorithm: self.algorithm,
            observers,
            threads: self.threads,
        }
    }
    #[cfg(not(feature = "rayon"))]
    /// Add an [`SwarmObserver`] to the list of [`SwarmObserver`]s used in the minimization.
    pub fn with_observer(self, observer: Arc<RwLock<dyn SwarmObserver<()>>>) -> Self {
        let mut observers = self.observers;
        observers.push(observer.clone());
        Self {
            algorithm: self.algorithm,
            observers,
        }
    }

    /// Set the number of threads to use.
    #[cfg(feature = "rayon")]
    pub fn with_threads(self, threads: usize) -> Self {
        Self {
            algorithm: self.algorithm,
            observers: self.observers,
            threads,
        }
    }
}

/// Python bindings for the [`ganesh`] crate
#[cfg(feature = "python")]
pub mod py_ganesh {
    use super::*;
    use std::sync::Arc;

    use bincode::{deserialize, serialize};
    use fastrand::Rng;
    use ganesh::{
        algorithms::{
            lbfgsb::LBFGSBErrorMode,
            nelder_mead::{
                NelderMeadFTerminator, NelderMeadXTerminator, SimplexConstructionMethod,
                SimplexExpansionMethod,
            },
            NelderMead, LBFGSB,
        },
        observers::{AutocorrelationObserver, MCMCObserver, Observer, SwarmObserver},
        samplers::{
            aies::WeightedAIESMove, ess::WeightedESSMove, integrated_autocorrelation_times,
            AIESMove, ESSMove, AIES, ESS,
        },
        swarms::{
            BoundaryMethod, Particle, SwarmPositionInitializer, SwarmVelocityInitializer, Topology,
            UpdateMethod, PSO,
        },
        Point, Status, Swarm,
    };
    use laddu_core::{DVector, Ensemble, Float, LadduError, ReadWrite};
    use numpy::{PyArray1, PyArray2, PyArray3};
    use parking_lot::RwLock;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{PyBytes, PyDict, PyList, PyTuple},
    };

    /// The L-BFGS-B Minimizer
    ///
    /// Parameters
    /// ----------
    /// eps_f_abs : float, optional
    ///     Set the absolute tolerance on the function value for the termination criteria
    /// eps_g_abs : float, optional
    ///     Set the absolute tolerance on the gradient value for the termination criteria
    /// tol_g_abs : float, optional
    ///     Set the absolute tolerance on the gradient value for the termination criteria // TODO:
    ///
    #[pyclass(name = "LBFGSB", module = "laddu")]
    #[derive(Clone)]
    pub struct PyLBFGSB {
        eps_f_abs: Option<Float>,
        eps_g_abs: Option<Float>,
        tol_g_abs: Option<Float>,
    }
    #[pymethods]
    impl PyLBFGSB {
        #[new]
        #[pyo3(signature=(*, eps_f_abs=None, eps_g_abs=None, tol_g_abs=None))]
        fn new(
            eps_f_abs: Option<Float>,
            eps_g_abs: Option<Float>,
            tol_g_abs: Option<Float>,
        ) -> Self {
            PyLBFGSB {
                eps_f_abs,
                eps_g_abs,
                tol_g_abs,
            }
        }
    }
    impl PyLBFGSB {
        fn get_algorithm<U>(&self, skip_hessian: bool) -> LBFGSB<U, LadduError> {
            let mut lbfgsb = LBFGSB::default();
            if let Some(eps_f_abs) = self.eps_f_abs {
                lbfgsb = lbfgsb.with_eps_f_abs(eps_f_abs);
            }
            if let Some(eps_g_abs) = self.eps_g_abs {
                lbfgsb = lbfgsb.with_eps_g_abs(eps_g_abs);
            }
            if let Some(tol_g_abs) = self.tol_g_abs {
                lbfgsb = lbfgsb.with_tol_g_abs(tol_g_abs);
            }
            if skip_hessian {
                lbfgsb = lbfgsb.with_error_mode(LBFGSBErrorMode::Skip);
            }
            lbfgsb
        }
    }

    /// Methods to initialize the ``NelderMead`` minimizer's simplex
    ///
    #[pyclass(name = "SimplexConstructionMethod", module = "laddu")]
    #[derive(Clone)]
    pub struct PySimplexConstructionMethod(SimplexConstructionMethod);
    #[pymethods]
    impl PySimplexConstructionMethod {
        /// Create an orthogonal simplex structure with a given spacing
        ///
        /// Parameters
        /// ----------
        /// simplex_size : float
        ///     The spacing between the vertices of the simplex
        ///
        /// Returns
        /// -------
        /// SimplexConstructionMethod
        ///
        /// Notes
        /// -----
        /// This will initialize a ``NelderMead`` simplex with a vertex at the initial guess of the
        /// minimizer in parameter space along with other vertices a distance of ``simplex_size``
        /// from the initial guess in each positive dimension of the space.
        ///
        #[staticmethod]
        fn orthogonal(simplex_size: Float) -> Self {
            Self(SimplexConstructionMethod::Orthogonal { simplex_size })
        }
        /// Create an simplex structure with the given vertices
        ///
        /// Parameters
        /// ----------
        /// simplex : list of list of float
        ///     A list of vertices in the parameter space
        ///
        /// Returns
        /// -------
        /// SimplexConstructionMethod
        ///
        /// Notes
        /// -----
        /// This construction method will ignore the initial guess input to the minimizer and will
        /// instead use the vertices provided. The list of vertices must have a length od ``D+1``
        /// where ``D`` is the dimensionality of the parameter space.
        ///
        #[staticmethod]
        fn custom(simplex: Vec<Vec<Float>>) -> Self {
            Self(SimplexConstructionMethod::Custom { simplex })
        }
    }

    /// The Nelder-Mead simplex minimizer
    ///
    /// The Nedler-Mead algorithm is a gradient-free minimizer which uses a set of evaluations to
    /// determine the next step to take.
    ///
    /// Parameters
    /// ----------
    /// eps_x_rel : float, optional
    ///     The relative tolerance on the parameter space to determine convergence
    /// eps_x_abs : float, optional
    ///     The absolute tolerance on the parameter space to determine convergence
    /// eps_f_rel : float, optional
    ///     The relative tolerance on the function value to determine convergence
    /// eps_f_abs : float, optional
    ///     The absolute tolerance on the function value to determine convergence
    /// alpha : float, optional
    ///     The reflection coefficient (default = ``1``, must be greater than zero)
    /// beta : float, optional
    ///     The expansion coefficient (default = ``2``, must be greater than one and
    ///     greater than ``alpha``)
    /// gamma : float, optional
    ///     The contraction coefficient (default = ``0.5``, must be strictly between zero and one)
    /// delta : float, optional
    ///     The shrink coefficient (default = ``0.5``, must be between zero and one)
    /// adaptive : int, optional
    ///     Override any of the coefficients ``alpha``, ``beta``, ``gamma``, and ``delta`` with
    ///     adaptive versions which scale with the number of parameters (this value should be set
    ///     to the number of free parameters if used, and setting any coefficient manually will
    ///     override that value)
    /// construction_method : SimplexConstructionMethod, optional
    ///     Specify how to initialize the simplex
    /// simplex_expansion_method : {"greedy minimization", "greedy expansion"}, optional
    ///     Specify whether the expansion method should favor minimization or exploration
    /// terminator_f : {"StdDev", "Amoeba", "Absolute", "None"}, optional
    ///     Set the termination criteria for the function value
    /// terminator_x : {"Singer", "Diameter", "Higham", "Rowan", "None"}, optional
    ///     Set the termination criteria for the simplex positions
    ///
    #[pyclass(name = "NelderMead", module = "laddu")]
    #[derive(Clone)]
    pub struct PyNelderMead {
        eps_x_rel: Option<Float>,
        eps_x_abs: Option<Float>,
        eps_f_rel: Option<Float>,
        eps_f_abs: Option<Float>,
        alpha: Option<Float>,
        beta: Option<Float>,
        gamma: Option<Float>,
        delta: Option<Float>,
        adaptive: Option<usize>,
        construction_method: Option<SimplexConstructionMethod>,
        simplex_expansion_method: Option<SimplexExpansionMethod>,
        terminator_f: Option<NelderMeadFTerminator>,
        terminator_x: Option<NelderMeadXTerminator>,
    }
    #[pymethods]
    impl PyNelderMead {
        #[new]
        #[pyo3(signature=(*, eps_x_rel=None, eps_x_abs=None, eps_f_rel=None, eps_f_abs=None, alpha=None, beta=None, gamma=None, delta=None, adaptive=None, construction_method=None, simplex_expansion_method=None, terminator_f=None, terminator_x=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            eps_x_rel: Option<Float>,
            eps_x_abs: Option<Float>,
            eps_f_rel: Option<Float>,
            eps_f_abs: Option<Float>,
            alpha: Option<Float>,
            beta: Option<Float>,
            gamma: Option<Float>,
            delta: Option<Float>,
            adaptive: Option<usize>,
            construction_method: Option<PySimplexConstructionMethod>,
            simplex_expansion_method: Option<String>,
            terminator_f: Option<String>,
            terminator_x: Option<String>,
        ) -> PyResult<Self> {
            let construction_method = construction_method.map(|cm| cm.0);
            let simplex_expansion_method = simplex_expansion_method.map(|sem| {match sem.to_lowercase().as_str() {
                "greedy minimization" => Ok(SimplexExpansionMethod::GreedyMinimization),
                "greedy expansion" => Ok(SimplexExpansionMethod::GreedyExpansion),
                _ => Err(PyTypeError::new_err("Invalid simplex_expansion_method! Valid options are \"greedy minimization\" (default) or \"greedy expansion\".")),
            }}).transpose()?;
            let terminator_f=terminator_f.map(|tf| {match tf.to_lowercase().as_str() {
                "amoeba" => Ok(NelderMeadFTerminator::Amoeba),
                "absolute" => Ok(NelderMeadFTerminator::Absolute),
                "stddev" => Ok(NelderMeadFTerminator::StdDev),
                "none" => Ok(NelderMeadFTerminator::None),
                _ => Err(PyTypeError::new_err("Invalid terminator_f! Valid options are \"stddev\" (default), \"amoeba\", \"absolute\", or \"none\".")),
            }}).transpose()?;
            let terminator_x=terminator_x.map(|tx| {match tx.to_lowercase().as_str() {
                "diameter" => Ok(NelderMeadXTerminator::Diameter),
                "higham" => Ok(NelderMeadXTerminator::Higham),
                "rowan" => Ok(NelderMeadXTerminator::Rowan),
                "singer" => Ok(NelderMeadXTerminator::Singer),
                "none" => Ok(NelderMeadXTerminator::None),
                _ => Err(PyTypeError::new_err("Invalid terminator_x! Valid options are \"singer\" (default), \"diameter\", \"higham\", \"rowan\", or \"none\".")),
            }}).transpose()?;
            Ok(PyNelderMead {
                eps_x_rel,
                eps_x_abs,
                eps_f_rel,
                eps_f_abs,
                alpha,
                beta,
                gamma,
                delta,
                adaptive,
                construction_method,
                simplex_expansion_method,
                terminator_f,
                terminator_x,
            })
        }
    }
    impl PyNelderMead {
        fn get_algorithm(&self, skip_hessian: bool) -> NelderMead {
            let mut nm = NelderMead::default();
            if let Some(eps_x_rel) = self.eps_x_rel {
                nm = nm.with_eps_x_rel(eps_x_rel);
            }
            if let Some(eps_x_abs) = self.eps_x_abs {
                nm = nm.with_eps_x_abs(eps_x_abs);
            }
            if let Some(eps_f_rel) = self.eps_f_rel {
                nm = nm.with_eps_f_rel(eps_f_rel);
            }
            if let Some(eps_f_abs) = self.eps_f_abs {
                nm = nm.with_eps_f_abs(eps_f_abs);
            }
            if let Some(adaptive) = self.adaptive {
                nm = nm.with_adaptive(adaptive);
            }
            if let Some(alpha) = self.alpha {
                nm = nm.with_alpha(alpha);
            }
            if let Some(beta) = self.beta {
                nm = nm.with_beta(beta);
            }
            if let Some(gamma) = self.gamma {
                nm = nm.with_gamma(gamma);
            }
            if let Some(delta) = self.delta {
                nm = nm.with_delta(delta);
            }
            if let Some(construction_method) = self.construction_method.clone() {
                nm = nm.with_construction_method(construction_method);
            }
            if let Some(simplex_expansion_method) = self.simplex_expansion_method.clone() {
                nm = nm.with_expansion_method(simplex_expansion_method);
            }
            if let Some(terminator_f) = self.terminator_f.clone() {
                nm = nm.with_terminator_f(terminator_f);
            }
            if let Some(terminator_x) = self.terminator_x.clone() {
                nm = nm.with_terminator_x(terminator_x);
            }
            if skip_hessian {
                nm = nm.with_no_error_calculation();
            }
            nm
        }
    }

    /// A weighted AIES move.
    ///
    #[pyclass(name = "AIESMove", module = "laddu")]
    #[derive(Clone)]
    pub struct PyAIESMove(WeightedAIESMove);
    #[pymethods]
    impl PyAIESMove {
        /// Construct a stretch move.
        ///
        /// Parameters
        /// ----------
        /// weight : float, default=1.0
        ///     The relative frequency this move should be chosen
        /// a : float, default=2.0
        ///     A scaling factor (higher values encourage exploration)
        ///
        #[staticmethod]
        #[pyo3(signature = (weight=1.0, *, a=None))]
        fn stretch(weight: Option<Float>, a: Option<Float>) -> Self {
            let weight = weight.unwrap_or(1.0);
            if let Some(a) = a {
                Self((AIESMove::Stretch { a }, weight))
            } else {
                Self(AIESMove::stretch(weight))
            }
        }
        /// Construct a walk move.
        ///
        /// Parameters
        /// ----------
        /// weight : float, default=1.0
        ///     The relative frequency this move should be chosen
        ///
        #[staticmethod]
        #[pyo3(signature = (weight=1.0))]
        fn walk(weight: Option<Float>) -> Self {
            let weight = weight.unwrap_or(1.0);
            Self(AIESMove::walk(weight))
        }
    }

    /// Construct an Affine Invariant Ensemble Sampler (AIES).
    ///
    /// Parameters
    /// ----------
    /// moves : list of AIESMove
    ///     The set of moves to randomly draw from at each step
    ///
    #[pyclass(name = "AIES", module = "laddu")]
    #[derive(Clone)]
    pub struct PyAIES(Vec<PyAIESMove>);
    #[pymethods]
    impl PyAIES {
        #[new]
        fn new(moves: Vec<PyAIESMove>) -> Self {
            Self(moves)
        }
    }
    impl PyAIES {
        fn get_algorithm(&self, rng: Rng) -> AIES {
            AIES::new(
                self.0
                    .iter()
                    .map(|m| m.0)
                    .collect::<Vec<WeightedAIESMove>>(),
                rng,
            )
        }
    }

    /// A weighted ESS move.
    ///
    #[pyclass(name = "ESSMove", module = "laddu")]
    #[derive(Clone)]
    pub struct PyESSMove(WeightedESSMove);
    #[pymethods]
    impl PyESSMove {
        /// Construct a differential move.
        ///
        /// Parameters
        /// ----------
        /// weight : float, default=1.0
        ///     The relative frequency this move should be chosen
        ///
        #[staticmethod]
        #[pyo3(signature = (weight=1.0))]
        fn differential(weight: Option<Float>) -> Self {
            let weight = weight.unwrap_or(1.0);
            Self(ESSMove::differential(weight))
        }
        /// Construct a Gaussian move.
        ///
        /// Parameters
        /// ----------
        /// weight : float, default=1.0
        ///     The relative frequency this move should be chosen
        ///
        #[staticmethod]
        #[pyo3(signature = (weight=1.0))]
        fn gaussian(weight: Option<Float>) -> Self {
            let weight = weight.unwrap_or(1.0);
            Self(ESSMove::gaussian(weight))
        }
        /// Construct a global move.
        ///
        /// Parameters
        /// ----------
        /// weight : float, default=1.0
        ///     The relative frequency this move should be chosen
        /// scale : float, optional
        ///     The rescaling factor to apply for jumps within the same mode (larger values promote
        ///     larger jumps, the default is ``1.0``)
        /// rescale_cov : float, optional
        ///     The rescaling factor to apply to the covariance matrix for jumps between modes (larger values promote
        ///     jumping, the default is ``0.001``)
        /// n_components : int, optional
        ///     The number of components to use in the Dirichlet Process Bayesian Gaussian Mixture
        ///     model (defaults to ``5``)
        ///
        #[staticmethod]
        #[pyo3(signature = (weight=1.0, *, scale=None, rescale_cov=None, n_components=None))]
        fn global_move(
            weight: Option<Float>,
            scale: Option<Float>,
            rescale_cov: Option<Float>,
            n_components: Option<usize>,
        ) -> Self {
            let weight = weight.unwrap_or(1.0);
            Self(ESSMove::global(weight, scale, rescale_cov, n_components))
        }
    }

    /// Construct an Ensemble Slice Sampler (ESS).
    ///
    /// Parameters
    /// ----------
    /// moves : list of ESSMove
    ///     The set of moves to randomly draw from at each step
    /// n_adaptive : int, optional
    ///     The number of adaptive moves to perform at the start of sampling (defaults to ``0``)
    /// max_steps : int, optional
    ///     The maximum number of expansions/contractions to perform at each step (defaults to
    ///     ``10000``)
    /// mu : float, optional
    ///     The adaptive scaling parameter (defaults to ``1.0``)
    ///
    #[pyclass(name = "ESS", module = "laddu")]
    #[derive(Clone)]
    pub struct PyESS {
        moves: Vec<PyESSMove>,
        n_adaptive: Option<usize>,
        max_steps: Option<usize>,
        mu: Option<Float>,
    }
    #[pymethods]
    impl PyESS {
        #[new]
        #[pyo3(signature = (moves, *, n_adaptive=None, max_steps=None, mu=None))]
        fn new(
            moves: Vec<PyESSMove>,
            n_adaptive: Option<usize>,
            max_steps: Option<usize>,
            mu: Option<Float>,
        ) -> Self {
            Self {
                moves,
                n_adaptive,
                max_steps,
                mu,
            }
        }
    }
    impl PyESS {
        fn get_algorithm(&self, rng: Rng) -> ESS {
            let mut ess = ESS::new(
                self.moves
                    .iter()
                    .map(|m| m.0)
                    .collect::<Vec<WeightedESSMove>>(),
                rng,
            );
            if let Some(n_adaptive) = self.n_adaptive {
                ess = ess.with_n_adaptive(n_adaptive)
            }
            if let Some(max_steps) = self.max_steps {
                ess = ess.with_max_steps(max_steps)
            }
            if let Some(mu) = self.mu {
                ess = ess.with_mu(mu)
            }
            ess
        }
    }

    /// A class which defines the initial position of a Swarm
    ///
    #[pyclass(name = "SwarmPositionInitializer", module = "laddu")]
    #[derive(Clone)]
    pub struct PySwarmPositionInitializer(SwarmPositionInitializer);
    #[pymethods]
    impl PySwarmPositionInitializer {
        /// Construct a swarm with all particles at the origin.
        ///
        /// Parameters
        /// ----------
        /// n_particles : int
        ///     The number of particles to create
        /// n_dimensions : int
        ///     The dimension of the parameter space
        ///
        /// Returns
        /// -------
        /// SwarmPositionInitializer
        ///
        #[staticmethod]
        fn zero(n_particles: usize, n_dimensions: usize) -> Self {
            Self(SwarmPositionInitializer::Zero {
                n_particles,
                n_dimensions,
            })
        }
        /// Construct a swarm of random particles in the given limits.
        ///
        /// Parameters
        /// ----------
        /// n_particles : int
        ///     The number of particles to create
        /// limits: list of tuple of float
        ///     A list of lower and upper limit pairs for each dimension of the parameter space
        ///
        /// Returns
        /// -------
        /// SwarmPositionInitializer
        ///
        #[staticmethod]
        fn random_in_limits(n_particles: usize, limits: Vec<(Float, Float)>) -> Self {
            Self(SwarmPositionInitializer::RandomInLimits {
                n_particles,
                limits,
            })
        }
        /// Construct a swarm of particles at the given positions.
        ///
        /// Parameters
        /// ----------
        /// positions: array_like or list of list of float
        ///     A list of particle positions
        ///
        /// Returns
        /// -------
        /// SwarmPositionInitializer
        ///
        #[staticmethod]
        fn custom(positions: Vec<Vec<Float>>) -> Self {
            Self(SwarmPositionInitializer::Custom(
                positions.into_iter().map(DVector::from_vec).collect(),
            ))
        }
        /// Construct a swarm of random particles in the given limits, using Latin Hypercube
        /// sampling to distribute the particles.
        ///
        /// Parameters
        /// ----------
        /// n_particles : int
        ///     The number of particles to create
        /// limits: list of tuple of float
        ///     A list of lower and upper limit pairs for each dimension of the parameter space
        ///
        /// Returns
        /// -------
        /// SwarmPositionInitializer
        ///
        #[staticmethod]
        fn latin_hypercube(n_particles: usize, limits: Vec<(Float, Float)>) -> Self {
            Self(SwarmPositionInitializer::LatinHypercube {
                n_particles,
                limits,
            })
        }
    }

    /// A class which defines the initial velocity of a Swarm
    ///
    #[pyclass(name = "SwarmVelocityInitializer", module = "laddu")]
    #[derive(Clone)]
    pub struct PySwarmVelocityInitializer(SwarmVelocityInitializer);
    #[pymethods]
    impl PySwarmVelocityInitializer {
        /// Initialize all particle velocities to zero.
        ///
        /// Returns
        /// -------
        /// SwarmVelocityInitializer
        ///
        #[staticmethod]
        fn zero() -> Self {
            Self(SwarmVelocityInitializer::Zero)
        }
        /// Initialize particle velocities to random values in the given limits.
        ///
        /// Parameters
        /// ----------
        /// limits: list of tuple of float
        ///     A list of lower and upper limit pairs for each dimension of the parameter space
        ///
        /// Returns
        /// -------
        /// SwarmVelocityInitializer
        ///
        #[staticmethod]
        fn random_in_limits(limits: Vec<(Float, Float)>) -> Self {
            Self(SwarmVelocityInitializer::RandomInLimits(limits))
        }
    }

    /// A standard Particle Swarm Optimization (PSO) algorithm.
    ///
    /// Parameters
    /// ----------
    /// omega : float, optional
    ///     The inertial weight (defaults to ``0.8``)
    /// c1 : float, optional
    ///     The cognitive weight (defaults to ``0.1``)
    /// c2 : float, optional
    ///     The social weight (defaults to ``0.1``)
    /// topology : {"global", "ring"}, optional
    ///     The swarm topology to use (defaults to ``"global"``)
    /// update_method : {"sync", "async"}, optional
    ///     The update method to use (defaults to ``"sync"``)
    ///
    /// Returns
    /// -------
    /// PSO
    ///
    #[pyclass(name = "PSO", module = "laddu")]
    #[derive(Clone)]
    pub struct PyPSO {
        omega: Option<Float>,
        c1: Option<Float>,
        c2: Option<Float>,
        topology: Option<Topology>,
        update_method: Option<UpdateMethod>,
    }
    #[pymethods]
    impl PyPSO {
        #[new]
        #[pyo3(signature=(*, omega=None, c1=None, c2=None, topology=None, update_method=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            omega: Option<Float>,
            c1: Option<Float>,
            c2: Option<Float>,
            topology: Option<String>,
            update_method: Option<String>,
        ) -> PyResult<Self> {
            Ok(Self {
                omega,
                c1,
                c2,
                topology: topology
                    .map(|t| match t.to_lowercase().as_str() {
                        "global" => Ok(Topology::Global),
                        "ring" => Ok(Topology::Ring),
                        _ => Err(PyTypeError::new_err(
                            "Invalid topology! Valid options are \"global\" (default) or \"ring\".",
                        )),
                    })
                    .transpose()?,
                update_method: update_method
                    .map(|u| match u.to_lowercase().as_str() {
                        "sync" => Ok(UpdateMethod::Synchronous),
                        "async" => Ok(UpdateMethod::Asynchronous),
                        _ => Err(PyTypeError::new_err(
                            "Invalid update_method! Valid options are \"sync\" (default), or \"async\".",
                        )),
                    }).transpose()?,
            })
        }
    }
    impl PyPSO {
        fn get_algorithm(&self, rng: Rng) -> PSO {
            let mut pso = PSO::new(rng);
            if let Some(omega) = self.omega {
                pso = pso.with_omega(omega);
            }
            if let Some(c1) = self.c1 {
                pso = pso.with_c1(c1);
            }
            if let Some(c2) = self.c2 {
                pso = pso.with_c2(c2);
            }
            if let Some(topology) = self.topology {
                pso = pso.with_topology(topology);
            }
            if let Some(update_method) = self.update_method {
                pso = pso.with_update_method(update_method);
            }
            pso
        }
    }

    /// A user implementation of [`Observer`](`crate::ganesh::observers::Observer`) from Python
    #[pyclass]
    #[pyo3(name = "Observer")]
    pub struct PyObserver(Py<PyAny>);

    #[pymethods]
    impl PyObserver {
        #[new]
        fn new(observer: Py<PyAny>) -> Self {
            Self(observer)
        }
    }

    /// A user implementation of [`MCMCObserver`](`crate::ganesh::observers::MCMCObserver`) from Python
    #[pyclass]
    #[pyo3(name = "MCMCObserver")]
    pub struct PyMCMCObserver(Py<PyAny>);

    #[pymethods]
    impl PyMCMCObserver {
        #[new]
        fn new(observer: Py<PyAny>) -> Self {
            Self(observer)
        }
    }

    /// A user implementation of [`SwarmObserver`](`crate::ganesh::observers::SwarmObserver`) from Python
    #[pyclass]
    #[pyo3(name = "SwarmObserver")]
    pub struct PySwarmObserver(Py<PyAny>);

    #[pymethods]
    impl PySwarmObserver {
        #[new]
        fn new(observer: Py<PyAny>) -> Self {
            Self(observer)
        }
    }

    /// The status/result of a minimization
    ///
    #[pyclass(name = "Status", module = "laddu")]
    #[derive(Clone)]
    pub struct PyStatus(pub Status);
    #[pymethods]
    impl PyStatus {
        /// The current best position in parameter space
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(py, self.0.x.as_slice())
        }
        /// The uncertainty on each parameter (``None`` if it wasn't calculated)
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        #[getter]
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<Float>>> {
            self.0
                .err
                .clone()
                .map(|err| PyArray1::from_slice(py, err.as_slice()))
        }
        /// The initial position at the start of the minimization
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(py, self.0.x0.as_slice())
        }
        /// The optimized value of the objective function
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn fx(&self) -> Float {
            self.0.fx
        }
        /// The covariance matrix (``None`` if it wasn't calculated)
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        /// Raises
        /// ------
        /// Exception
        ///     If there was a problem creating the resulting ``numpy`` array
        ///
        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<Float>>>> {
            self.0
                .cov
                .clone()
                .map(|cov| {
                    Ok(PyArray2::from_vec2(
                        py,
                        &cov.row_iter()
                            .map(|row| row.iter().cloned().collect())
                            .collect::<Vec<Vec<Float>>>(),
                    )
                    .map_err(LadduError::NumpyError)?)
                })
                .transpose()
        }
        /// The Hessian matrix (``None`` if it wasn't calculated)
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        /// Raises
        /// ------
        /// Exception
        ///     If there was a problem creating the resulting ``numpy`` array
        ///
        #[getter]
        fn hess<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<Float>>>> {
            self.0
                .hess
                .clone()
                .map(|hess| {
                    Ok(PyArray2::from_vec2(
                        py,
                        &hess
                            .row_iter()
                            .map(|row| row.iter().cloned().collect())
                            .collect::<Vec<Vec<Float>>>(),
                    )
                    .map_err(LadduError::NumpyError)?)
                })
                .transpose()
        }
        /// A status message from the optimizer at the end of the algorithm
        ///
        /// Returns
        /// -------
        /// str
        ///
        #[getter]
        fn message(&self) -> String {
            self.0.message.clone()
        }
        /// The state of the optimizer's convergence conditions
        ///
        /// Returns
        /// -------
        /// bool
        ///
        #[getter]
        fn converged(&self) -> bool {
            self.0.converged
        }
        /// Parameter bounds which were applied to the fitting algorithm
        ///
        /// Returns
        /// -------
        /// list of Bound or None
        ///
        #[getter]
        fn bounds(&self) -> Option<Vec<PyBound>> {
            self.0
                .bounds
                .clone()
                .map(|bounds| bounds.iter().map(|bound| PyBound(*bound)).collect())
        }
        /// The number of times the objective function was evaluated
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.n_f_evals
        }
        /// The number of times the gradient of the objective function was evaluated
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_g_evals(&self) -> usize {
            self.0.n_g_evals
        }
        fn __str__(&self) -> String {
            self.0.to_string()
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
        /// Save the fit result to a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the new file (overwrites if the file exists!)
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to write the file
        ///
        fn save_as(&self, path: &str) -> PyResult<()> {
            self.0.save_as(path)?;
            Ok(())
        }
        /// Load a fit result from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file
        ///
        /// Returns
        /// -------
        /// Status
        ///     The fit result contained in the file
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to read the file
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(PyStatus(Status::load_from(path)?))
        }
        #[new]
        fn new() -> Self {
            PyStatus(Status::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(
                py,
                serialize(&self.0)
                    .map_err(LadduError::SerdeError)?
                    .as_slice(),
            ))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = PyStatus(deserialize(state.as_bytes()).map_err(LadduError::SerdeError)?);
            Ok(())
        }
        /// Converts a Status into a Python dictionary
        ///
        /// Returns
        /// -------
        /// dict
        ///
        /// Raises
        /// ------
        /// Exception
        ///     If there was a problem creating the resulting ``numpy`` array
        ///
        fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let dict = PyDict::new(py);
            dict.set_item("x", self.x(py))?;
            dict.set_item("err", self.err(py))?;
            dict.set_item("x0", self.x0(py))?;
            dict.set_item("fx", self.fx())?;
            dict.set_item("cov", self.cov(py)?)?;
            dict.set_item("hess", self.hess(py)?)?;
            dict.set_item("message", self.message())?;
            dict.set_item("converged", self.converged())?;
            dict.set_item("bounds", self.bounds())?;
            dict.set_item("n_f_evals", self.n_f_evals())?;
            dict.set_item("n_g_evals", self.n_g_evals())?;
            Ok(dict)
        }
    }

    /// An ensemble of MCMC walkers
    ///
    #[pyclass(name = "Ensemble", module = "laddu")]
    #[derive(Clone)]
    pub struct PyEnsemble(pub Ensemble);
    #[pymethods]
    impl PyEnsemble {
        /// The dimension of the Ensemble ``(n_walkers, n_steps, n_variables)``
        #[getter]
        fn dimension(&self) -> (usize, usize, usize) {
            self.0.dimension()
        }
        /// Get the contents of the Ensemble
        ///
        /// Parameters
        /// ----------
        /// burn: int, default = 0
        ///     The number of steps to burn from the beginning of each walker's history
        /// thin: int, default = 1
        ///     The number of steps to discard after burn-in (``1`` corresponds to no thinning,
        ///     ``2`` discards every other step, ``3`` discards every third, and so on)
        ///
        /// Returns
        /// -------
        /// array_like
        ///     An array with dimension ``(n_walkers, n_steps, n_parameters)``
        ///
        /// Raises
        /// ------
        /// Exception
        ///     If there was a problem creating the resulting ``numpy`` array
        ///
        #[pyo3(signature = (*, burn = 0, thin = 1))]
        fn get_chain<'py>(
            &self,
            py: Python<'py>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray3<Float>>> {
            let chain = self.0.get_chain(burn, thin);
            Ok(PyArray3::from_vec3(
                py,
                &chain
                    .iter()
                    .map(|walker| {
                        walker
                            .iter()
                            .map(|step| step.data.as_vec().to_vec())
                            .collect()
                    })
                    .collect::<Vec<_>>(),
            )
            .map_err(LadduError::NumpyError)?)
        }
        /// Get the contents of the Ensemble, flattened over walkers
        ///
        /// Parameters
        /// ----------
        /// burn: int, default = 0
        ///     The number of steps to burn from the beginning of each walker's history
        /// thin: int, default = 1
        ///     The number of steps to discard after burn-in (``1`` corresponds to no thinning,
        ///     ``2`` discards every other step, ``3`` discards every third, and so on)
        ///
        /// Returns
        /// -------
        /// array_like
        ///     An array with dimension ``(n_steps, n_parameters)``
        ///
        /// Raises
        /// ------
        /// Exception
        ///     If there was a problem creating the resulting ``numpy`` array
        ///
        #[pyo3(signature = (*, burn = 0, thin = 1))]
        fn get_flat_chain<'py>(
            &self,
            py: Python<'py>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray2<Float>>> {
            let chain = self.0.get_flat_chain(burn, thin);
            Ok(PyArray2::from_vec2(
                py,
                &chain
                    .iter()
                    .map(|step| step.data.as_vec().to_vec())
                    .collect::<Vec<_>>(),
            )
            .map_err(LadduError::NumpyError)?)
        }
        /// Save the ensemble to a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file (overwrites if the file exists!)
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to write the file
        ///
        fn save_as(&self, path: &str) -> PyResult<()> {
            self.0.save_as(path)?;
            Ok(())
        }
        /// Load an ensemble from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file
        ///
        /// Returns
        /// -------
        /// Ensemble
        ///     The ensemble contained in the file
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to read the file
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(PyEnsemble(Ensemble::load_from(path)?))
        }
        #[new]
        fn new() -> Self {
            PyEnsemble(Ensemble::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(
                py,
                serialize(&self.0)
                    .map_err(LadduError::SerdeError)?
                    .as_slice(),
            ))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = PyEnsemble(deserialize(state.as_bytes()).map_err(LadduError::SerdeError)?);
            Ok(())
        }
        /// Calculate the integrated autocorrelation time for each parameter according to
        /// [Karamanis]_
        ///
        /// Parameters
        /// ----------
        /// c : float, default = 7.0
        ///     The size of the window used in the autowindowing algorithm by [Sokal]_
        /// burn: int, default = 0
        ///     The number of steps to burn from the beginning of each walker's history
        /// thin: int, default = 1
        ///     The number of steps to discard after burn-in (``1`` corresponds to no thinning,
        ///     ``2`` discards every other step, ``3`` discards every third, and so on)
        ///
        #[pyo3(signature = (*, c=7.0, burn=0, thin=1))]
        fn get_integrated_autocorrelation_times<'py>(
            &self,
            py: Python<'py>,
            c: Option<Float>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(
                py,
                self.0
                    .get_integrated_autocorrelation_times(c, burn, thin)
                    .as_slice(),
            )
        }
    }

    /// A point in parameter space with a position and value.
    ///
    #[pyclass(name = "Point", module = "laddu")]
    #[derive(Clone)]
    pub struct PyPoint(pub Point);
    #[pymethods]
    impl PyPoint {
        /// The position of the point in parameter space
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(py, self.0.get_x().as_slice())
        }
        /// The evaluation of the point
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn fx(&self) -> Float {
            self.0.get_fx()
        }
        /// Save the Point to a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file (overwrites if the file exists!)
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to write the file
        ///
        fn save_as(&self, path: &str) -> PyResult<()> {
            self.0.save_as(path)?;
            Ok(())
        }
        /// Load a Point from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file
        ///
        /// Returns
        /// -------
        /// Point
        ///     The Point contained in the file
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to read the file
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(PyPoint(Point::load_from(path)?))
        }
        #[new]
        fn new() -> Self {
            PyPoint(Point::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(
                py,
                serialize(&self.0)
                    .map_err(LadduError::SerdeError)?
                    .as_slice(),
            ))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = PyPoint(deserialize(state.as_bytes()).map_err(LadduError::SerdeError)?);
            Ok(())
        }
    }

    /// A particle in parameter space with a position, velocity, and best position.
    ///
    #[pyclass(name = "Particle", module = "laddu")]
    #[derive(Clone)]
    pub struct PyParticle(pub Particle);
    #[pymethods]
    impl PyParticle {
        /// The position of the particle
        ///
        /// Returns
        /// -------
        /// Point
        ///
        #[getter]
        fn position(&self) -> PyPoint {
            PyPoint(self.0.position.clone())
        }
        /// The velocity of the particle
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(py, self.0.velocity.as_slice())
        }
        /// The best position the particle has found
        ///
        /// Returns
        /// -------
        /// Point
        ///
        #[getter]
        fn best(&self) -> PyPoint {
            PyPoint(self.0.best.clone())
        }
        /// Save the Particle to a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file (overwrites if the file exists!)
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to write the file
        ///
        fn save_as(&self, path: &str) -> PyResult<()> {
            self.0.save_as(path)?;
            Ok(())
        }
        /// Load a Particle from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file
        ///
        /// Returns
        /// -------
        /// Particle
        ///     The Particle contained in the file
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to read the file
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(PyParticle(Particle::load_from(path)?))
        }
        #[new]
        fn new() -> Self {
            PyParticle(Particle::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(
                py,
                serialize(&self.0)
                    .map_err(LadduError::SerdeError)?
                    .as_slice(),
            ))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = PyParticle(deserialize(state.as_bytes()).map_err(LadduError::SerdeError)?);
            Ok(())
        }
    }

    /// A particle swarm used in particle-swarm-optimization-like algorithms
    ///
    /// Parameters
    /// ----------
    /// position_initializer : PositionInitializer
    ///     The method for setting the initial position of the swarm
    /// velocity_initializer : VelocityInitializer, optional
    ///     The method for setting the initial velocity of the swarm (defaults to setting all
    ///     velocities to zero)
    /// boundary_method : {"inf", "shr", "transform"}, optional
    ///     Specifies the boundary method to use if bounds are specified (defaults to "inf")
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the boundary method given is not a valid method
    ///
    #[pyclass(name = "Swarm", module = "laddu")]
    #[derive(Clone)]
    pub struct PySwarm(pub Swarm);
    #[pymethods]
    impl PySwarm {
        #[new]
        #[pyo3(signature = (position_initializer, *, velocity_initializer=None, boundary_method=None))]
        fn new(
            position_initializer: PySwarmPositionInitializer,
            velocity_initializer: Option<PySwarmVelocityInitializer>,
            boundary_method: Option<String>,
        ) -> PyResult<Self> {
            let mut swarm = Swarm::new(position_initializer.0);
            if let Some(velocity_initializer) = velocity_initializer {
                swarm = swarm.with_velocity_initializer(velocity_initializer.0);
            }
            if let Some(boundary_method) = boundary_method {
                swarm = swarm.with_boundary_method(match boundary_method.to_lowercase().as_str() {
                        "inf" => Ok(BoundaryMethod::Inf),
                        "shr" => Ok(BoundaryMethod::Shr),
                        "transform" => Ok(BoundaryMethod::Transform),
                        _ => Err(PyTypeError::new_err(
                            "Invalid boundary_method! Valid options are \"inf\" (default), \"shr\", or \"transform\".",
                        )),
                    }?);
            }
            Ok(PySwarm(swarm))
        }
        /// The dimension of the parameter space
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn dimension(&self) -> usize {
            self.0.dimension
        }
        /// A list of the particles in the swarm
        ///
        /// Returns
        /// -------
        /// list of laddu.Particle
        ///
        #[getter]
        fn particles(&self) -> Vec<PyParticle> {
            self.0
                .particles
                .iter()
                .map(|p| PyParticle(p.clone()))
                .collect()
        }
        /// The global best position found by the Swarm
        ///
        /// Returns
        /// -------
        /// laddu.Point
        #[getter]
        fn gbest(&self) -> PyPoint {
            PyPoint(self.0.gbest.clone())
        }
        /// A status message from the optimizer at the end of the algorithm
        ///
        /// Returns
        /// -------
        /// str
        ///
        #[getter]
        fn message(&self) -> String {
            self.0.message.clone()
        }
        /// The state of the optimizer's convergence conditions
        ///
        /// Returns
        /// -------
        /// bool
        ///
        #[getter]
        fn converged(&self) -> bool {
            self.0.converged
        }
        /// Parameter bounds which were applied to the swarm algorithm
        ///
        /// Returns
        /// -------
        /// list of Bound or None
        ///
        #[getter]
        fn bounds(&self) -> Option<Vec<PyBound>> {
            self.0
                .bounds
                .clone()
                .map(|bounds| bounds.iter().map(|bound| PyBound(*bound)).collect())
        }
        fn __str__(&self) -> String {
            self.0.to_string()
        }
        fn __repr__(&self) -> String {
            format!("Swarm({} particles)", self.0.particles.len())
        }
        /// Save the Swarm to a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the file (overwrites if the file exists!)
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to write the file
        ///
        fn save_as(&self, path: &str) -> PyResult<()> {
            self.0.save_as(path)?;
            Ok(())
        }
        /// Load a Swarm from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the existing fit file
        ///
        /// Returns
        /// -------
        /// Swarm
        ///     The fit result contained in the file
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to read the file
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(PySwarm(Swarm::load_from(path)?))
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(
                py,
                serialize(&self.0)
                    .map_err(LadduError::SerdeError)?
                    .as_slice(),
            ))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = PySwarm(deserialize(state.as_bytes()).map_err(LadduError::SerdeError)?);
            Ok(())
        }
        /// Converts a Swarm into a Python dictionary
        ///
        /// Returns
        /// -------
        /// dict
        ///
        /// Raises
        /// ------
        /// Exception
        ///     If there was a problem creating the resulting ``numpy`` array
        ///
        fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let dict = PyDict::new(py);
            dict.set_item("dimension", self.dimension())?;
            dict.set_item("particles", self.particles())?;
            dict.set_item("gbest", self.gbest())?;
            dict.set_item("message", self.message())?;
            dict.set_item("converged", self.converged())?;
            dict.set_item("bounds", self.bounds())?;
            Ok(dict)
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
    ///     The size of the window used in the autowindowing algorithm by [Sokal]_
    ///
    ///
    /// .. rubric:: References
    ///
    /// .. [Karamanis] Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002. 06212.
    ///
    /// .. [Sokal] Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
    ///
    #[pyfunction(name = "integrated_autocorrelation_times")]
    #[pyo3(signature = (x, *, c=7.0))]
    pub fn py_integrated_autocorrelation_times(
        py: Python<'_>,
        x: Vec<Vec<Vec<Float>>>,
        c: Option<Float>,
    ) -> Bound<'_, PyArray1<Float>> {
        let x: Vec<Vec<DVector<Float>>> = x
            .into_iter()
            .map(|y| y.into_iter().map(DVector::from_vec).collect())
            .collect();
        PyArray1::from_slice(py, integrated_autocorrelation_times(x, c).as_slice())
    }

    /// An obsever which can check the integrated autocorrelation time of the ensemble and
    /// terminate if convergence conditions are met
    ///
    /// Parameters
    /// ----------
    /// n_check : int, default = 50
    ///     How often (in number of steps) to check this observer
    /// n_tau_threshold : int, default = 50
    ///     The number of mean integrated autocorrelation times needed to terminate
    /// dtau_threshold : float, default = 0.01
    ///     The threshold for the absolute change in integrated autocorrelation time (Δτ/τ)
    /// discard : float, default = 0.5
    ///     The fraction of steps to discard from the beginning of the chain before analysis
    /// terminate : bool, default = True
    ///     Set to ``False`` to forego termination even if the chains converge
    /// c : float, default = 7.0
    ///     The size of the window used in the autowindowing algorithm by [Sokal]_
    /// verbose : bool, default = False
    ///     Set to ``True`` to print out details at each check
    ///
    #[pyclass(name = "AutocorrelationObserver", module = "laddu")]
    pub struct PyAutocorrelationObserver(Arc<RwLock<AutocorrelationObserver>>);

    #[pymethods]
    impl PyAutocorrelationObserver {
        #[new]
        #[pyo3(signature = (*, n_check=50, n_taus_threshold=50, dtau_threshold=0.01, discard=0.5, terminate=true, c=7.0, verbose=false))]
        fn new(
            n_check: usize,
            n_taus_threshold: usize,
            dtau_threshold: Float,
            discard: Float,
            terminate: bool,
            c: Float,
            verbose: bool,
        ) -> Self {
            Self(
                AutocorrelationObserver::default()
                    .with_n_check(n_check)
                    .with_n_taus_threshold(n_taus_threshold)
                    .with_dtau_threshold(dtau_threshold)
                    .with_discard(discard)
                    .with_terminate(terminate)
                    .with_sokal_window(c)
                    .with_verbose(verbose)
                    .build(),
            )
        }
        /// The integrated autocorrelation times observed at each checking step
        ///
        #[getter]
        fn taus<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            let taus = self.0.read().taus.clone();
            PyArray1::from_vec(py, taus)
        }
    }

    /// A class representing a lower and upper bound on a free parameter
    ///
    #[pyclass]
    #[derive(Clone)]
    #[pyo3(name = "Bound")]
    pub struct PyBound(laddu_core::Bound);
    #[pymethods]
    impl PyBound {
        /// The lower bound
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn lower(&self) -> Float {
            self.0.lower()
        }
        /// The upper bound
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn upper(&self) -> Float {
            self.0.upper()
        }
    }

    impl Observer<()> for PyObserver {
        fn callback(&mut self, step: usize, status: &mut Status, _user_data: &mut ()) -> bool {
            let (new_status, result) = Python::with_gil(|py| {
                let res = self
                    .0
                    .bind(py)
                    .call_method("callback", (step, PyStatus(status.clone())), None)
                    .unwrap_or_else(|err| {
                        err.print(py);
                        panic!("Python error encountered!");
                    });
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!");
                let new_status = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
                    .extract::<PyStatus>()
                    .expect("The first item returned from \"callback\" must be a \"laddu.Status\"!")
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
                    .extract::<bool>()
                    .expect("The second item returned from \"callback\" must be a \"bool\"!");
                (new_status, result)
            });
            *status = new_status;
            result
        }
    }

    #[cfg(feature = "rayon")]
    impl Observer<ThreadPool> for PyObserver {
        fn callback(
            &mut self,
            step: usize,
            status: &mut Status,
            _thread_pool: &mut ThreadPool,
        ) -> bool {
            let (new_status, result) = Python::with_gil(|py| {
                let res = self
                    .0
                    .bind(py)
                    .call_method("callback", (step, PyStatus(status.clone())), None)
                    .unwrap_or_else(|err| {
                        err.print(py);
                        panic!("Python error encountered!");
                    });
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!");
                let new_status = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
                    .extract::<PyStatus>()
                    .expect("The first item returned from \"callback\" must be a \"laddu.Status\"!")
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
                    .extract::<bool>()
                    .expect("The second item returned from \"callback\" must be a \"bool\"!");
                (new_status, result)
            });
            *status = new_status;
            result
        }
    }
    impl FromPyObject<'_> for PyObserver {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(PyObserver(ob.clone().into()))
        }
    }
    impl MCMCObserver<()> for PyMCMCObserver {
        fn callback(&mut self, step: usize, ensemble: &mut Ensemble, _user_data: &mut ()) -> bool {
            let (new_ensemble, result) = Python::with_gil(|py| {
                let res = self
                    .0
                    .bind(py)
                    .call_method("callback", (step, PyEnsemble(ensemble.clone())), None)
                    .unwrap_or_else(|err| {
                        err.print(py);
                        panic!("Python error encountered!");
                    });
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[Ensemble, bool]\"!");
                let new_status = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[Ensemble, bool]\"!")
                    .extract::<PyEnsemble>()
                    .expect("The first item returned from \"callback\" must be a \"Ensemble\"!")
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[Ensemble, bool]\"!")
                    .extract::<bool>()
                    .expect("The second item returned from \"callback\" must be a \"bool\"!");
                (new_status, result)
            });
            *ensemble = new_ensemble;
            result
        }
    }
    #[cfg(feature = "rayon")]
    impl MCMCObserver<ThreadPool> for PyMCMCObserver {
        fn callback(
            &mut self,
            step: usize,
            ensemble: &mut Ensemble,
            _thread_pool: &mut ThreadPool,
        ) -> bool {
            let (new_ensemble, result) = Python::with_gil(|py| {
                let res = self
                    .0
                    .bind(py)
                    .call_method("callback", (step, PyEnsemble(ensemble.clone())), None)
                    .unwrap_or_else(|err| {
                        err.print(py);
                        panic!("Python error encountered!");
                    });
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[Ensemble, bool]\"!");
                let new_status = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[Ensemble, bool]\"!")
                    .extract::<PyEnsemble>()
                    .expect("The first item returned from \"callback\" must be a \"Ensemble\"!")
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[Ensemble, bool]\"!")
                    .extract::<bool>()
                    .expect("The second item returned from \"callback\" must be a \"bool\"!");
                (new_status, result)
            });
            *ensemble = new_ensemble;
            result
        }
    }

    impl FromPyObject<'_> for PyMCMCObserver {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(PyMCMCObserver(ob.clone().into()))
        }
    }

    impl SwarmObserver<()> for PySwarmObserver {
        fn callback(&mut self, step: usize, swarm: &mut Swarm, _user_data: &mut ()) -> bool {
            let (new_swarm, result) = Python::with_gil(|py| {
                let res = self
                    .0
                    .bind(py)
                    .call_method("callback", (step, PySwarm(swarm.clone())), None)
                    .unwrap_or_else(|err| {
                        err.print(py);
                        panic!("Python error encountered!");
                    });
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[Swarm, bool]\"!");
                let new_swarm = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[Swarm, bool]\"!")
                    .extract::<PySwarm>()
                    .expect("The first item returned from \"callback\" must be a \"Swarm\"!")
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[Swarm, bool]\"!")
                    .extract::<bool>()
                    .expect("The second item returned from \"callback\" must be a \"bool\"!");
                (new_swarm, result)
            });
            *swarm = new_swarm;
            result
        }
    }

    #[cfg(feature = "rayon")]
    impl SwarmObserver<ThreadPool> for PySwarmObserver {
        fn callback(
            &mut self,
            step: usize,
            swarm: &mut Swarm,
            _thread_pool: &mut ThreadPool,
        ) -> bool {
            let (new_swarm, result) = Python::with_gil(|py| {
                let res = self
                    .0
                    .bind(py)
                    .call_method("callback", (step, PySwarm(swarm.clone())), None)
                    .unwrap_or_else(|err| {
                        err.print(py);
                        panic!("Python error encountered!");
                    });
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[Swarm, bool]\"!");
                let new_warm = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[Swarm, bool]\"!")
                    .extract::<PySwarm>()
                    .expect("The first item returned from \"callback\" must be a \"Swarm\"!")
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[Swarm, bool]\"!")
                    .extract::<bool>()
                    .expect("The second item returned from \"callback\" must be a \"bool\"!");
                (new_warm, result)
            });
            *swarm = new_swarm;
            result
        }
    }
    impl FromPyObject<'_> for PySwarmObserver {
        fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(PySwarmObserver(ob.clone().into()))
        }
    }

    #[cfg(feature = "python")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn py_parse_minimizer_options(
        opt_method: Option<Bound<'_, PyAny>>,
        opt_observers: Option<Bound<'_, PyAny>>,
        max_steps: usize,
        debug: bool,
        verbose: bool,
        show_step: bool,
        show_x: bool,
        show_fx: bool,
        skip_hessian: bool,
        opt_threads: Option<usize>,
    ) -> PyResult<MinimizerOptions> {
        let mut options = MinimizerOptions::default();
        let mut observers: Vec<Arc<RwLock<PyObserver>>> = Vec::default();
        if let Some(pyany_observers) = opt_observers {
            if let Ok(observer_list) = pyany_observers.downcast::<PyList>() {
                for item in observer_list.iter() {
                    let observer = item.extract::<PyObserver>()?;
                    observers.push(Arc::new(RwLock::new(observer)));
                }
            } else if let Ok(single_observer) = pyany_observers.extract::<PyObserver>() {
                observers.push(Arc::new(RwLock::new(single_observer)));
            } else {
                return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single Observer or a list of Observers!"));
            }
            for observer in observers {
                options = options.with_observer(observer);
            }
        }
        if let Some(method) = opt_method {
            if let Ok(algorithm) = method.extract::<PyLBFGSB>() {
                options = options.with_algorithm(algorithm.get_algorithm(skip_hessian))
            } else if let Ok(algorithm) = method.extract::<PyNelderMead>() {
                options = options.with_algorithm(algorithm.get_algorithm(skip_hessian))
            } else {
                return Err(PyValueError::new_err(
                    "Invalid \"method\": Valid methods include 'LBFGSB', 'NelderMead'.".to_string(),
                ));
            }
        }
        #[cfg(feature = "rayon")]
        {
            options = options.with_threads(opt_threads.unwrap_or_else(num_cpus::get));
        }
        if debug {
            options = options.debug();
        }
        if verbose {
            options = options.verbose(show_step, show_x, show_fx);
        }
        options = options.with_max_steps(max_steps);
        Ok(options)
    }

    #[cfg(feature = "python")]
    pub(crate) fn py_parse_mcmc_options(
        opt_method: Option<Bound<'_, PyAny>>,
        opt_observers: Option<Bound<'_, PyAny>>,
        debug: bool,
        verbose: bool,
        opt_threads: Option<usize>,
        rng: Rng,
    ) -> PyResult<MCMCOptions> {
        let mut options = if let Some(method) = opt_method {
            if let Ok(algorithm) = method.extract::<PyAIES>() {
                MCMCOptions::from_algorithm(algorithm.get_algorithm(rng))
            } else if let Ok(algorithm) = method.extract::<PyESS>() {
                MCMCOptions::from_algorithm(algorithm.get_algorithm(rng))
            } else {
                return Err(PyValueError::new_err(
                    "Invalid \"method\": Valid methods include 'ESS', and 'AIES'.".to_string(),
                ));
            }
        } else {
            MCMCOptions::default_with_rng(rng)
        };
        let mut observers: Vec<Arc<RwLock<PyMCMCObserver>>> = Vec::default();
        if let Some(pyany_observers) = opt_observers {
            if let Ok(observer_list) = pyany_observers.downcast::<PyList>() {
                for item in observer_list.iter() {
                    let observer = item.extract::<PyMCMCObserver>()?;
                    observers.push(Arc::new(RwLock::new(observer)));
                }
            } else if let Ok(single_observer) = pyany_observers.extract::<PyMCMCObserver>() {
                observers.push(Arc::new(RwLock::new(single_observer)));
            } else {
                return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single MCMCObserver or a list of MCMCObservers!"));
            }
            for observer in observers {
                options = options.with_observer(observer);
            }
        }
        #[cfg(feature = "rayon")]
        {
            options = options.with_threads(opt_threads.unwrap_or_else(num_cpus::get));
        }
        if debug {
            options = options.debug();
        }
        if verbose {
            options = options.verbose();
        }
        Ok(options)
    }

    #[cfg(feature = "python")]
    pub(crate) fn py_parse_swarm_options(
        opt_method: Option<Bound<'_, PyAny>>,
        opt_observers: Option<Bound<'_, PyAny>>,
        debug: bool,
        verbose: bool,
        opt_threads: Option<usize>,
        rng: Rng,
    ) -> PyResult<SwarmOptions> {
        let mut options = if let Some(method) = opt_method {
            if let Ok(algorithm) = method.extract::<PyPSO>() {
                SwarmOptions::from_algorithm(algorithm.get_algorithm(rng))
            } else {
                return Err(PyValueError::new_err(
                    "Invalid \"method\": Valid methods include 'PSO'.".to_string(),
                ));
            }
        } else {
            SwarmOptions::default_with_rng(rng)
        };
        let mut observers: Vec<Arc<RwLock<PySwarmObserver>>> = Vec::default();
        if let Some(pyany_observers) = opt_observers {
            if let Ok(observer_list) = pyany_observers.downcast::<PyList>() {
                for item in observer_list.iter() {
                    let observer = item.extract::<PySwarmObserver>()?;
                    observers.push(Arc::new(RwLock::new(observer)));
                }
            } else if let Ok(single_observer) = pyany_observers.extract::<PySwarmObserver>() {
                observers.push(Arc::new(RwLock::new(single_observer)));
            } else {
                return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single SwarmObserver or a list of SwarmObservers!"));
            }
            for observer in observers {
                options = options.with_observer(observer);
            }
        }
        #[cfg(feature = "rayon")]
        {
            options = options.with_threads(opt_threads.unwrap_or_else(num_cpus::get));
        }
        if debug {
            options = options.debug();
        }
        if verbose {
            options = options.verbose();
        }
        Ok(options)
    }
}
