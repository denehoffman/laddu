use std::{ops::ControlFlow, sync::Arc};

use ganesh::{
    algorithms::{
        gradient::GradientStatus,
        gradient_free::GradientFreeStatus,
        mcmc::{integrated_autocorrelation_times, EnsembleStatus},
        particles::SwarmStatus,
    },
    python::{PyEnsembleStatus, PyGradientFreeStatus, PyGradientStatus, PySwarmStatus},
    traits::{Algorithm, Observer, Terminator},
    DVector,
};
use laddu_core::LadduError;
use laddu_extensions::optimize::MaybeThreadPool;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
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
                    Py::new(py, PyGradientStatus::from(status.clone()))
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
                    Py::new(py, PyGradientFreeStatus::from(status.clone()))
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
                    Py::new(py, PySwarmStatus::from(status.clone()))?,
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
            let py_status = Py::new(py, PyGradientStatus::from(status.clone()))?;
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
            let py_status = Py::new(py, PyGradientFreeStatus::from(status.clone()))?;
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
            let py_status = Py::new(py, PySwarmStatus::from(status.clone()))?;
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
                    Py::new(py, PyEnsembleStatus::from(status.clone()))?,
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
            let py_status = Py::new(py, PyEnsembleStatus::from(status.clone()))?;
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
