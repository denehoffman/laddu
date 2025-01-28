use std::sync::Arc;

use fastrand::Rng;
use ganesh::{
    algorithms::LBFGSB,
    mcmc::{aies::WeightedAIESMove, ess::WeightedESSMove, MCMCAlgorithm, MCMCObserver, AIES, ESS},
    observers::{DebugMCMCObserver, DebugObserver},
    Algorithm, Observer, Status,
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
    pub(crate) algorithm: Box<dyn ganesh::Algorithm<ThreadPool, LadduError>>,
    #[cfg(not(feature = "rayon"))]
    pub(crate) algorithm: Box<dyn ganesh::Algorithm<(), LadduError>>,
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
    /// Use the [`ESS`] algorithm with `100` adaptive steps.
    pub fn new_ess<T: AsRef<[WeightedESSMove]>>(moves: T, rng: Rng) -> Self {
        Self {
            algorithm: Box::new(ESS::new(moves, rng).with_n_adaptive(100)),
            observers: Default::default(),
            #[cfg(all(feature = "rayon", feature = "num_cpus"))]
            threads: num_cpus::get(),
            #[cfg(all(feature = "rayon", not(feature = "num_cpus")))]
            threads: 0,
        }
    }
    /// Use the [`AIES`] algorithm.
    pub fn new_aies<T: AsRef<[WeightedAIESMove]>>(moves: T, rng: Rng) -> Self {
        Self {
            algorithm: Box::new(AIES::new(moves, rng)),
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
    pub fn with_algorithm<A: MCMCAlgorithm<ThreadPool, LadduError> + 'static>(
        self,
        algorithm: A,
    ) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: self.observers,
            threads: self.threads,
        }
    }
    /// Set the [`MCMCAlgorithm`] to be used in the minimization.
    #[cfg(not(feature = "rayon"))]
    pub fn with_algorithm<A: MCMCAlgorithm<(), LadduError> + 'static>(self, algorithm: A) -> Self {
        Self {
            algorithm: Box::new(algorithm),
            observers: self.observers,
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

#[cfg(feature = "python")]
pub mod py_ganesh {
    use super::*;
    use std::sync::Arc;

    use bincode::{deserialize, serialize};
    use fastrand::Rng;
    use ganesh::{
        algorithms::{
            lbfgsb::{LBFGSBFTerminator, LBFGSBGTerminator},
            nelder_mead::{NelderMeadFTerminator, NelderMeadXTerminator, SimplexExpansionMethod},
            NelderMead, LBFGSB,
        },
        mcmc::{
            aies::WeightedAIESMove, ess::WeightedESSMove, integrated_autocorrelation_times,
            AIESMove, ESSMove, MCMCObserver, AIES, ESS,
        },
        observers::AutocorrelationObserver,
        Observer, Status,
    };
    use laddu_core::{DVector, Ensemble, Float, LadduError, ReadWrite};
    use laddu_python::GetStrExtractObj;
    use numpy::{PyArray1, PyArray2, PyArray3};
    use parking_lot::RwLock;
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{PyBytes, PyDict, PyList, PyTuple},
    };

    #[pyclass]
    #[pyo3(name = "Observer")]
    pub struct PyObserver(Py<PyAny>);

    #[pymethods]
    impl PyObserver {
        #[new]
        pub fn new(observer: Py<PyAny>) -> Self {
            Self(observer)
        }
    }

    #[pyclass]
    #[pyo3(name = "MCMCObserver")]
    pub struct PyMCMCObserver(Py<PyAny>);

    #[pymethods]
    impl PyMCMCObserver {
        #[new]
        pub fn new(observer: Py<PyAny>) -> Self {
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
        ///     The path of the existing fit file
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
        /// Load an ensemble from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the existing fit file
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
    /// .. [Karamanis] Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002. 06212.
    /// .. [Sokal] Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
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
                .call_method(
                    "callback",
                    (step, PyStatus(status.clone())),
                    None,
                )
                .expect("Observer does not have a \"callback(step: int, status: laddu.Status) -> tuple[laddu.Status, bool]\" method!");
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
                .call_method(
                    "callback",
                    (step, PyStatus(status.clone())),
                    None,
                )
                .expect("Observer does not have a \"callback(step: int, status: laddu.Status) -> tuple[laddu.Status, bool]\" method!");
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
                .call_method(
                    "callback",
                    (step, PyEnsemble(ensemble.clone())),
                    None,
                )
                .expect("MCMCObserver does not have a \"callback(step: int, status: laddu.Ensemble) -> tuple[laddu.Ensemble, bool]\" method!");
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!");
                let new_status = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
                    .extract::<PyEnsemble>()
                    .expect(
                        "The first item returned from \"callback\" must be a \"laddu.Ensemble\"!",
                    )
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
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
                .call_method(
                    "callback",
                    (step, PyEnsemble(ensemble.clone())),
                    None,
                )
                .expect("MCMCObserver does not have a \"callback(step: int, status: laddu.Ensemble) -> tuple[laddu.Ensemble, bool]\" method!");
                let res_tuple = res
                    .downcast::<PyTuple>()
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!");
                let new_status = res_tuple
                    .get_item(0)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
                    .extract::<PyEnsemble>()
                    .expect(
                        "The first item returned from \"callback\" must be a \"laddu.Ensemble\"!",
                    )
                    .0;
                let result = res_tuple
                    .get_item(1)
                    .expect("\"callback\" method should return a \"tuple[laddu.Status, bool]\"!")
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

    #[cfg(feature = "python")]
    pub(crate) fn py_parse_minimizer_options(
        n_parameters: usize,
        method: &str,
        max_steps: usize,
        debug: bool,
        verbose: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<MinimizerOptions> {
        let mut options = MinimizerOptions::default();
        let mut show_step = true;
        let mut show_x = true;
        let mut show_fx = true;
        if let Some(kwargs) = kwargs {
            show_step = kwargs.get_extract::<bool>("show_step")?.unwrap_or(true);
            show_x = kwargs.get_extract::<bool>("show_x")?.unwrap_or(true);
            show_fx = kwargs.get_extract::<bool>("show_fx")?.unwrap_or(true);
            let tol_x_rel = kwargs
                .get_extract::<Float>("tol_x_rel")?
                .unwrap_or(Float::EPSILON);
            let tol_x_abs = kwargs
                .get_extract::<Float>("tol_x_abs")?
                .unwrap_or(Float::EPSILON);
            let tol_f_rel = kwargs
                .get_extract::<Float>("tol_f_rel")?
                .unwrap_or(Float::EPSILON);
            let tol_f_abs = kwargs
                .get_extract::<Float>("tol_f_abs")?
                .unwrap_or(Float::EPSILON);
            let tol_g_abs = kwargs
                .get_extract::<Float>("tol_g_abs")?
                .unwrap_or(Float::cbrt(Float::EPSILON));
            let g_tolerance = kwargs.get_extract::<Float>("g_tolerance")?.unwrap_or(1e-5);
            let skip_hessian = kwargs.get_extract::<bool>("skip_hessian")?.unwrap_or(false);
            let adaptive = kwargs.get_extract::<bool>("adaptive")?.unwrap_or(false);
            let alpha = kwargs.get_extract::<Float>("alpha")?;
            let beta = kwargs.get_extract::<Float>("beta")?;
            let gamma = kwargs.get_extract::<Float>("gamma")?;
            let delta = kwargs.get_extract::<Float>("delta")?;
            let simplex_expansion_method = kwargs
                .get_extract::<String>("simplex_expansion_method")?
                .unwrap_or("greedy minimization".into());
            let nelder_mead_f_terminator = kwargs
                .get_extract::<String>("nelder_mead_f_terminator")?
                .unwrap_or("stddev".into());
            let nelder_mead_x_terminator = kwargs
                .get_extract::<String>("nelder_mead_x_terminator")?
                .unwrap_or("singer".into());
            #[cfg(feature = "rayon")]
            let threads = kwargs
                .get_extract::<usize>("threads")
                .unwrap_or(None)
                .unwrap_or_else(num_cpus::get);
            let mut observers: Vec<Arc<RwLock<PyObserver>>> = Vec::default();
            if let Ok(Some(observer_arg)) = kwargs.get_item("observers") {
                if let Ok(observer_list) = observer_arg.downcast::<PyList>() {
                    for item in observer_list.iter() {
                        let observer = item.extract::<PyObserver>()?;
                        observers.push(Arc::new(RwLock::new(observer)));
                    }
                } else if let Ok(single_observer) = observer_arg.extract::<PyObserver>() {
                    observers.push(Arc::new(RwLock::new(single_observer)));
                } else {
                    return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single Observer or a list of Observers!"));
                }
            }
            for observer in observers {
                options = options.with_observer(observer);
            }
            match method {
                "lbfgsb" => {
                    let mut lbfgsb = LBFGSB::default()
                        .with_terminator_f(LBFGSBFTerminator { tol_f_abs })
                        .with_terminator_g(LBFGSBGTerminator { tol_g_abs })
                        .with_g_tolerance(g_tolerance);
                    if skip_hessian {
                        lbfgsb.with_error_mode(LBFGSBGTerminator::Skip);
                    }
                    options = options.with_algorithm(lbfgsb)
                }
                "nelder_mead" => {
                    let terminator_f = match nelder_mead_f_terminator.as_str() {
                        "amoeba" => NelderMeadFTerminator::Amoeba { tol_f_rel },
                        "absolute" => NelderMeadFTerminator::Absolute { tol_f_abs },
                        "stddev" => NelderMeadFTerminator::StdDev { tol_f_abs },
                        "none" => NelderMeadFTerminator::None,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"nelder_mead_f_terminator\": \"{}\"",
                                nelder_mead_f_terminator
                            )))
                        }
                    };
                    let terminator_x = match nelder_mead_x_terminator.as_str() {
                        "diameter" => NelderMeadXTerminator::Diameter { tol_x_abs },
                        "higham" => NelderMeadXTerminator::Higham { tol_x_rel },
                        "rowan" => NelderMeadXTerminator::Rowan { tol_x_rel },
                        "singer" => NelderMeadXTerminator::Singer { tol_x_rel },
                        "none" => NelderMeadXTerminator::None,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"nelder_mead_x_terminator\": \"{}\"",
                                nelder_mead_x_terminator
                            )))
                        }
                    };
                    let simplex_expansion_method = match simplex_expansion_method.as_str() {
                        "greedy minimization" => SimplexExpansionMethod::GreedyMinimization,
                        "greedy expansion" => SimplexExpansionMethod::GreedyExpansion,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"simplex_expansion_method\": \"{}\"",
                                simplex_expansion_method
                            )))
                        }
                    };
                    let mut nelder_mead = NelderMead::default()
                        .with_terminator_f(terminator_f)
                        .with_terminator_x(terminator_x)
                        .with_expansion_method(simplex_expansion_method);
                    if adaptive {
                        nelder_mead = nelder_mead.with_adaptive(n_parameters);
                    }
                    if let Some(alpha) = alpha {
                        nelder_mead = nelder_mead.with_alpha(alpha);
                    }
                    if let Some(beta) = beta {
                        nelder_mead = nelder_mead.with_beta(beta);
                    }
                    if let Some(gamma) = gamma {
                        nelder_mead = nelder_mead.with_gamma(gamma);
                    }
                    if let Some(delta) = delta {
                        nelder_mead = nelder_mead.with_delta(delta);
                    }
                    if skip_hessian {
                        nelder_mead = nelder_mead.with_no_error_calculation();
                    }
                    options = options.with_algorithm(nelder_mead)
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid \"method\": \"{}\"",
                        method
                    )))
                }
            }
            #[cfg(feature = "rayon")]
            {
                options = options.with_threads(threads);
            }
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
        method: &str,
        debug: bool,
        verbose: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
        rng: Rng,
    ) -> PyResult<MCMCOptions> {
        let default_ess_moves = [ESSMove::differential(0.9), ESSMove::gaussian(0.1)];
        let default_aies_moves = [AIESMove::stretch(0.9), AIESMove::walk(0.1)];
        let mut options = MCMCOptions::new_ess(default_ess_moves, rng.clone());
        if let Some(kwargs) = kwargs {
            let n_adaptive = kwargs.get_extract::<usize>("n_adaptive")?.unwrap_or(100);
            let mu = kwargs.get_extract::<Float>("mu")?.unwrap_or(1.0);
            let max_ess_steps = kwargs
                .get_extract::<usize>("max_ess_steps")?
                .unwrap_or(10000);
            let mut ess_moves: Vec<WeightedESSMove> = Vec::default();
            if let Ok(Some(ess_move_list_arg)) = kwargs.get_item("ess_moves") {
                if let Ok(ess_move_list) = ess_move_list_arg.downcast::<PyList>() {
                    for item in ess_move_list.iter() {
                        let item_tuple = item.downcast::<PyTuple>()?;
                        let move_name = item_tuple.get_item(0)?.extract::<String>()?;
                        let move_weight = item_tuple.get_item(1)?.extract::<Float>()?;
                        match move_name.to_lowercase().as_ref() {
                            "differential" => ess_moves.push(ESSMove::differential(move_weight)),
                            "gaussian" => ess_moves.push(ESSMove::gaussian(move_weight)),
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "Unknown ESS move type: {}",
                                    move_name
                                )))
                            }
                        }
                    }
                }
            }
            if ess_moves.is_empty() {
                ess_moves = default_ess_moves.to_vec();
            }
            let mut aies_moves: Vec<WeightedAIESMove> = Vec::default();
            if let Ok(Some(aies_move_list_arg)) = kwargs.get_item("aies_moves") {
                if let Ok(aies_move_list) = aies_move_list_arg.downcast::<PyList>() {
                    for item in aies_move_list.iter() {
                        let item_tuple = item.downcast::<PyTuple>()?;
                        if let Ok(move_name) = item_tuple.get_item(0)?.extract::<String>() {
                            let move_weight = item_tuple.get_item(1)?.extract::<Float>()?;
                            match move_name.to_lowercase().as_ref() {
                                "stretch" => aies_moves.push(AIESMove::stretch(move_weight)),
                                "walk" => aies_moves.push(AIESMove::walk(move_weight)),
                                _ => {
                                    return Err(PyValueError::new_err(format!(
                                        "Unknown AIES move type: {}",
                                        move_name
                                    )))
                                }
                            }
                        } else if let Ok(move_spec) = item_tuple.get_item(0)?.downcast::<PyTuple>()
                        {
                            let move_name = move_spec.get_item(0)?.extract::<String>()?;
                            let move_weight = item_tuple.get_item(1)?.extract::<Float>()?;
                            if move_name.to_lowercase() == "stretch" {
                                let a = move_spec.get_item(1)?.extract::<Float>()?;
                                aies_moves.push((AIESMove::Stretch { a }, move_weight))
                            } else {
                                return Err(PyValueError::new_err(
                                    "Only the 'stretch' move has a hyperparameter",
                                ));
                            }
                        }
                    }
                }
            }
            if aies_moves.is_empty() {
                aies_moves = default_aies_moves.to_vec();
            }
            #[cfg(feature = "rayon")]
            let threads = kwargs
                .get_extract::<usize>("threads")
                .unwrap_or(None)
                .unwrap_or_else(num_cpus::get);
            #[cfg(feature = "rayon")]
            let mut observers: Vec<
                Arc<RwLock<dyn ganesh::mcmc::MCMCObserver<ThreadPool>>>,
            > = Vec::default();
            #[cfg(not(feature = "rayon"))]
            let mut observers: Vec<Arc<RwLock<dyn ganesh::mcmc::MCMCObserver<()>>>> =
                Vec::default();
            if let Ok(Some(observer_arg)) = kwargs.get_item("observers") {
                if let Ok(observer_list) = observer_arg.downcast::<PyList>() {
                    for item in observer_list.iter() {
                        if let Ok(observer) = item.downcast::<PyAutocorrelationObserver>() {
                            observers.push(observer.borrow().0.clone());
                        } else if let Ok(observer) = item.extract::<PyMCMCObserver>() {
                            observers.push(Arc::new(RwLock::new(observer)));
                        }
                    }
                } else if let Ok(single_observer) =
                    observer_arg.downcast::<PyAutocorrelationObserver>()
                {
                    observers.push(single_observer.borrow().0.clone());
                } else if let Ok(single_observer) = observer_arg.extract::<PyMCMCObserver>() {
                    observers.push(Arc::new(RwLock::new(single_observer)));
                } else {
                    return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single MCMCObserver or a list of MCMCObservers!"));
                }
            }
            for observer in observers {
                options = options.with_observer(observer.clone());
            }
            match method.to_lowercase().as_ref() {
                "ess" => {
                    options = options.with_algorithm(
                        ESS::new(ess_moves, rng)
                            .with_mu(mu)
                            .with_n_adaptive(n_adaptive)
                            .with_max_steps(max_ess_steps),
                    )
                }
                "aies" => options = options.with_algorithm(AIES::new(aies_moves, rng)),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid \"method\": \"{}\"",
                        method
                    )))
                }
            }
            #[cfg(feature = "rayon")]
            {
                options = options.with_threads(threads);
            }
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
