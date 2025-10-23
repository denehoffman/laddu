use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::RngSubsetExtension;
use accurate::{sum::Klein, traits::*};
use auto_ops::*;
use dyn_clone::DynClone;
use fastrand::Rng;
use laddu_core::{
    amplitudes::{central_difference, AmplitudeValues, Evaluator, GradientValues, Model},
    data::Dataset,
    resources::Parameters,
    LadduError,
};
use nalgebra::DVector;
use num::complex::Complex64;

#[cfg(feature = "mpi")]
use laddu_core::mpi::LadduMPI;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};
use parking_lot::Mutex;

#[cfg(feature = "python")]
use crate::ganesh_ext::{
    py_ganesh::{FromPyArgs, PyMCMCSummary, PyMinimizationSummary},
    MCMCSettings, MinimizationSettings,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyEvaluator, PyModel},
    data::PyDataset,
};
#[cfg(feature = "python")]
use numpy::PyArray1;
#[cfg(feature = "python")]
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyList},
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(feature = "python", feature = "rayon"))]
use rayon::ThreadPoolBuilder;

/// A trait which describes a term that can be used like a likelihood (more correctly, a negative
/// log-likelihood) in a minimization.
pub trait LikelihoodTerm: DynClone + Send + Sync {
    /// Evaluate the term given some input parameters.
    fn evaluate(&self, parameters: &[f64]) -> f64;
    /// Evaluate the gradient of the term given some input parameters.
    fn evaluate_gradient(&self, parameters: &[f64]) -> DVector<f64> {
        central_difference(parameters, |parameters: &[f64]| self.evaluate(parameters))
    }
    /// The list of names of the input parameters for [`LikelihoodTerm::evaluate`].
    fn parameters(&self) -> Vec<String>;
    /// A method called every step of any minimization/MCMC algorithm.
    fn update(&self) {}
}

dyn_clone::clone_trait_object!(LikelihoodTerm);

/// A term in an expression with multiple likelihood components
///
/// See Also
/// --------
/// NLL.to_term
///
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodTerm", module = "laddu")]
#[derive(Clone)]
pub struct PyLikelihoodTerm(pub Box<dyn LikelihoodTerm>);

/// An extended, unbinned negative log-likelihood evaluator.
#[derive(Clone)]
pub struct NLL {
    /// The internal [`Evaluator`] for data
    pub data_evaluator: Evaluator,
    /// The internal [`Evaluator`] for accepted Monte Carlo
    pub accmc_evaluator: Evaluator,
}

impl NLL {
    /// Construct an [`NLL`] from a [`Model`] and two [`Dataset`]s (data and Monte Carlo). This is the equivalent of the [`Model::load`] method,
    /// but for two [`Dataset`]s and a different method of evaluation.
    pub fn new(model: &Model, ds_data: &Arc<Dataset>, ds_accmc: &Arc<Dataset>) -> Box<Self> {
        Self {
            data_evaluator: model.load(ds_data),
            accmc_evaluator: model.load(ds_accmc),
        }
        .into()
    }
    /// Create a new [`StochasticNLL`] from this [`NLL`].
    pub fn to_stochastic(&self, batch_size: usize, seed: Option<usize>) -> StochasticNLL {
        StochasticNLL::new(self.clone(), batch_size, seed)
    }
    /// Activate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name.
    pub fn activate<T: AsRef<str>>(&self, name: T) -> Result<(), LadduError> {
        self.data_evaluator.activate(&name)?;
        self.accmc_evaluator.activate(&name)
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) -> Result<(), LadduError> {
        self.data_evaluator.activate_many(names)?;
        self.accmc_evaluator.activate_many(names)
    }
    /// Activate all registered [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s.
    pub fn activate_all(&self) {
        self.data_evaluator.activate_all();
        self.accmc_evaluator.activate_all();
    }
    /// Dectivate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) -> Result<(), LadduError> {
        self.data_evaluator.deactivate(&name)?;
        self.accmc_evaluator.deactivate(&name)
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) -> Result<(), LadduError> {
        self.data_evaluator.deactivate_many(names)?;
        self.accmc_evaluator.deactivate_many(names)
    }
    /// Deactivate all registered [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s.
    pub fn deactivate_all(&self) {
        self.data_evaluator.deactivate_all();
        self.accmc_evaluator.deactivate_all();
    }
    /// Isolate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name (deactivate the rest).
    pub fn isolate<T: AsRef<str>>(&self, name: T) -> Result<(), LadduError> {
        self.data_evaluator.isolate(&name)?;
        self.accmc_evaluator.isolate(&name)
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name (deactivate the rest).
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) -> Result<(), LadduError> {
        self.data_evaluator.isolate_many(names)?;
        self.accmc_evaluator.isolate_many(names)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each
    /// Monte-Carlo event (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project`] instead.
    pub fn project_local(&self, parameters: &[f64], mc_evaluator: Option<Evaluator>) -> Vec<f64> {
        let (mc_dataset, result) = if let Some(mc_evaluator) = mc_evaluator {
            (
                mc_evaluator.dataset.clone(),
                mc_evaluator.evaluate_local(parameters),
            )
        } else {
            (
                self.accmc_evaluator.dataset.clone(),
                self.accmc_evaluator.evaluate_local(parameters),
            )
        };
        let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
        #[cfg(feature = "rayon")]
        let output: Vec<f64> = result
            .par_iter()
            .zip(mc_dataset.events.par_iter())
            .map(|(l, e)| e.weight * l.re / n_mc)
            .collect();

        #[cfg(not(feature = "rayon"))]
        let output: Vec<f64> = result
            .iter()
            .zip(mc_dataset.events.iter())
            .map(|(l, e)| e.weight * l.re / n_mc)
            .collect();
        output
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each
    /// Monte-Carlo event (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_mpi(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> Vec<f64> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projection = self.project_local(parameters, mc_evaluator);
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }
        buffer
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each
    /// Monte-Carlo event. This method takes the real part of the given expression (discarding
    /// the imaginary part entirely, which does not matter if expressions are coherent sums
    /// wrapped in [`Expression::norm_sqr`](`laddu_core::Expression::norm_sqr`).
    /// Event weights are determined by the following formula:
    ///
    /// ```math
    /// \text{weight}(\vec{p}; e) = \text{weight}(e) \mathcal{L}(e) / N_{\text{MC}}
    /// ```
    ///
    /// Note that $`N_{\text{MC}}`$ will always be the number of accepted Monte Carlo events,
    /// regardless of the `mc_evaluator`.
    pub fn project(&self, parameters: &[f64], mc_evaluator: Option<Evaluator>) -> Vec<f64> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_mpi(parameters, mc_evaluator, &world);
            }
        }
        self.project_local(parameters, mc_evaluator)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_gradient`] instead.
    pub fn project_gradient_local(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
    ) -> (Vec<f64>, Vec<DVector<f64>>) {
        let (mc_dataset, result, result_gradient) = if let Some(mc_evaluator) = mc_evaluator {
            (
                mc_evaluator.dataset.clone(),
                mc_evaluator.evaluate_local(parameters),
                mc_evaluator.evaluate_gradient_local(parameters),
            )
        } else {
            (
                self.accmc_evaluator.dataset.clone(),
                self.accmc_evaluator.evaluate_local(parameters),
                self.accmc_evaluator.evaluate_gradient_local(parameters),
            )
        };
        let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
        #[cfg(feature = "rayon")]
        {
            (
                result
                    .par_iter()
                    .zip(mc_dataset.events.par_iter())
                    .map(|(l, e)| e.weight * l.re / n_mc)
                    .collect(),
                result_gradient
                    .par_iter()
                    .zip(mc_dataset.events.par_iter())
                    .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                    .collect(),
            )
        }
        #[cfg(not(feature = "rayon"))]
        {
            (
                result
                    .iter()
                    .zip(mc_dataset.events.iter())
                    .map(|(l, e)| e.weight * l.re / n_mc)
                    .collect(),
                result_gradient
                    .iter()
                    .zip(mc_dataset.events.iter())
                    .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                    .collect(),
            )
        }
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_gradient`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_gradient_mpi(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> (Vec<f64>, Vec<DVector<f64>>) {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let (local_projection, local_gradient_projection) =
            self.project_gradient_local(parameters, mc_evaluator);
        let mut projection_result: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut projection_result, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }

        let flattened_local_gradient_projection = local_gradient_projection
            .iter()
            .flat_map(|g| g.data.as_vec().to_vec())
            .collect::<Vec<f64>>();
        let (counts, displs) = world.get_flattened_counts_displs(n_events, parameters.len());
        let mut flattened_result_buffer = vec![0.0; n_events * parameters.len()];
        let mut partitioned_flattened_result_buffer =
            PartitionMut::new(&mut flattened_result_buffer, counts, displs);
        world.all_gather_varcount_into(
            &flattened_local_gradient_projection,
            &mut partitioned_flattened_result_buffer,
        );
        let gradient_projection_result = flattened_result_buffer
            .chunks(parameters.len())
            .map(DVector::from_row_slice)
            .collect();
        (projection_result, gradient_projection_result)
    }
    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event. This method takes the real part of the given
    /// expression (discarding the imaginary part entirely, which does not matter if expressions
    /// are coherent sums wrapped in [`Expression::norm_sqr`](`laddu_core::Expression::norm_sqr`).
    /// Event weights are determined by the following formula:
    ///
    /// ```math
    /// \text{weight}(\vec{p}; e) = \text{weight}(e) \mathcal{L}(e) / N_{\text{MC}}
    /// ```
    ///
    /// Note that $`N_{\text{MC}}`$ will always be the number of accepted Monte Carlo events,
    /// regardless of the `mc_evaluator`.
    pub fn project_gradient(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
    ) -> (Vec<f64>, Vec<DVector<f64>>) {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_gradient_mpi(parameters, mc_evaluator, &world);
            }
        }
        self.project_gradient_local(parameters, mc_evaluator)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_with`] instead.
    pub fn project_with_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<Vec<f64>, LadduError> {
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.resources.read().active.clone();
            mc_evaluator.isolate_many(names)?;
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_local(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let output: Vec<f64> = result
                .par_iter()
                .zip(mc_dataset.events.par_iter())
                .map(|(l, e)| e.weight * l.re / n_mc)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<f64> = result
                .iter()
                .zip(mc_dataset.events.iter())
                .map(|(l, e)| e.weight * l.re / n_mc)
                .collect();
            mc_evaluator.resources.write().active = current_active_mc;
            Ok(output)
        } else {
            let current_active_data = self.data_evaluator.resources.read().active.clone();
            let current_active_accmc = self.accmc_evaluator.resources.read().active.clone();
            self.isolate_many(names)?;
            let mc_dataset = &self.accmc_evaluator.dataset;
            let result = self.accmc_evaluator.evaluate_local(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let output: Vec<f64> = result
                .par_iter()
                .zip(mc_dataset.events.par_iter())
                .map(|(l, e)| e.weight * l.re / n_mc)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<f64> = result
                .iter()
                .zip(mc_dataset.events.iter())
                .map(|(l, e)| e.weight * l.re / n_mc)
                .collect();
            self.data_evaluator.resources.write().active = current_active_data;
            self.accmc_evaluator.resources.write().active = current_active_accmc;
            Ok(output)
        }
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_with`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_with_mpi<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> Result<Vec<f64>, LadduError> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projection = self.project_with_local(parameters, names, mc_evaluator)?;
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }
        Ok(buffer)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation.
    ///
    /// This method takes the real part of the given expression (discarding
    /// the imaginary part entirely, which does not matter if expressions are coherent sums
    /// wrapped in [`Expression::norm_sqr`](`laddu_core::Expression::norm_sqr`).
    /// Event weights are determined by the following formula:
    ///
    /// ```math
    /// \text{weight}(\vec{p}; e) = \text{weight}(e) \mathcal{L}(e) / N_{\text{MC}}
    /// ```
    ///
    /// Note that $`N_{\text{MC}}`$ will always be the number of accepted Monte Carlo events,
    /// regardless of the `mc_evaluator`.
    pub fn project_with<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<Vec<f64>, LadduError> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_with_mpi(parameters, names, mc_evaluator, &world);
            }
        }
        self.project_with_local(parameters, names, mc_evaluator)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_gradient`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_with`] instead.
    pub fn project_gradient_with_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<(Vec<f64>, Vec<DVector<f64>>), LadduError> {
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.resources.read().active.clone();
            mc_evaluator.isolate_many(names)?;
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_local(parameters);
            let result_gradient = mc_evaluator.evaluate_gradient(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let (res, res_gradient) = {
                (
                    result
                        .par_iter()
                        .zip(mc_dataset.events.par_iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .par_iter()
                        .zip(mc_dataset.events.par_iter())
                        .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                        .collect(),
                )
            };
            #[cfg(not(feature = "rayon"))]
            let (res, res_gradient) = {
                (
                    result
                        .iter()
                        .zip(mc_dataset.events.iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .iter()
                        .zip(mc_dataset.events.iter())
                        .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                        .collect(),
                )
            };
            mc_evaluator.resources.write().active = current_active_mc;
            Ok((res, res_gradient))
        } else {
            let current_active_data = self.data_evaluator.resources.read().active.clone();
            let current_active_accmc = self.accmc_evaluator.resources.read().active.clone();
            self.isolate_many(names)?;
            let mc_dataset = &self.accmc_evaluator.dataset;
            let result = self.accmc_evaluator.evaluate_local(parameters);
            let result_gradient = self.accmc_evaluator.evaluate_gradient(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let (res, res_gradient) = {
                (
                    result
                        .par_iter()
                        .zip(mc_dataset.events.par_iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .par_iter()
                        .zip(mc_dataset.events.par_iter())
                        .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                        .collect(),
                )
            };
            #[cfg(not(feature = "rayon"))]
            let (res, res_gradient) = {
                (
                    result
                        .iter()
                        .zip(mc_dataset.events.iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .iter()
                        .zip(mc_dataset.events.iter())
                        .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                        .collect(),
                )
            };
            self.data_evaluator.resources.write().active = current_active_data;
            self.accmc_evaluator.resources.write().active = current_active_accmc;
            Ok((res, res_gradient))
        }
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_gradient`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_with`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_gradient_with_mpi<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> Result<(Vec<f64>, Vec<DVector<f64>>), LadduError> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let (local_projection, local_gradient_projection) =
            self.project_gradient_with_local(parameters, names, mc_evaluator)?;
        let mut projection_result: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut projection_result, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }

        let flattened_local_gradient_projection = local_gradient_projection
            .iter()
            .flat_map(|g| g.data.as_vec().to_vec())
            .collect::<Vec<f64>>();
        let (counts, displs) = world.get_flattened_counts_displs(n_events, parameters.len());
        let mut flattened_result_buffer = vec![0.0; n_events * parameters.len()];
        let mut partitioned_flattened_result_buffer =
            PartitionMut::new(&mut flattened_result_buffer, counts, displs);
        world.all_gather_varcount_into(
            &flattened_local_gradient_projection,
            &mut partitioned_flattened_result_buffer,
        );
        let gradient_projection_result = flattened_result_buffer
            .chunks(parameters.len())
            .map(DVector::from_row_slice)
            .collect();
        Ok((projection_result, gradient_projection_result))
    }
    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each
    /// Monte-Carlo event. This method differs from the standard [`NLL::project_gradient`] in that it first
    /// isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name, but returns
    /// the [`NLL`] to its prior state after calculation.
    ///
    /// This method takes the real part of the given expression (discarding
    /// the imaginary part entirely, which does not matter if expressions are coherent sums
    /// wrapped in [`Expression::norm_sqr`](`laddu_core::Expression::norm_sqr`).
    /// Event weights are determined by the following formula:
    ///
    /// ```math
    /// \text{weight}(\vec{p}; e) = \text{weight}(e) \mathcal{L}(e) / N_{\text{MC}}
    /// ```
    ///
    /// Note that $`N_{\text{MC}}`$ will always be the number of accepted Monte Carlo events,
    /// regardless of the `mc_evaluator`.
    pub fn project_gradient_with<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<(Vec<f64>, Vec<DVector<f64>>), LadduError> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_gradient_with_mpi(parameters, names, mc_evaluator, &world);
            }
        }
        self.project_gradient_with_local(parameters, names, mc_evaluator)
    }

    fn evaluate_local(&self, parameters: &[f64]) -> f64 {
        let data_result = self.data_evaluator.evaluate_local(parameters);
        let mc_result = self.accmc_evaluator.evaluate_local(parameters);
        let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
        #[cfg(feature = "rayon")]
        let data_term: f64 = data_result
            .par_iter()
            .zip(self.data_evaluator.dataset.events.par_iter())
            .map(|(l, e)| e.weight * f64::ln(l.re))
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(feature = "rayon")]
        let mc_term: f64 = mc_result
            .par_iter()
            .zip(self.accmc_evaluator.dataset.events.par_iter())
            .map(|(l, e)| e.weight * l.re)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let data_term: f64 = data_result
            .iter()
            .zip(self.data_evaluator.dataset.events.iter())
            .map(|(l, e)| e.weight * f64::ln(l.re))
            .sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let mc_term: f64 = mc_result
            .iter()
            .zip(self.accmc_evaluator.dataset.events.iter())
            .map(|(l, e)| e.weight * l.re)
            .sum_with_accumulator::<Klein<f64>>();
        -2.0 * (data_term - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_mpi(&self, parameters: &[f64], world: &SimpleCommunicator) -> f64 {
        let local_evaluation = self.evaluate_local(parameters);
        let mut buffer: Vec<f64> = vec![0.0; world.size() as usize];
        world.all_gather_into(&local_evaluation, &mut buffer);
        buffer.iter().sum()
    }

    fn evaluate_gradient_local(&self, parameters: &[f64]) -> DVector<f64> {
        let data_resources = self.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let mc_resources = self.accmc_evaluator.resources.read();
        let mc_parameters = Parameters::new(parameters, &mc_resources.constants);
        let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
        #[cfg(feature = "rayon")]
        let data_term: DVector<f64> = self
            .data_evaluator
            .dataset
            .events
            .par_iter()
            .zip(data_resources.caches.par_iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.data_evaluator.amplitudes.len()];
                self.data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&data_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.data_evaluator
                            .amplitudes
                            .iter()
                            .zip(data_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&data_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.data_evaluator.expression.evaluate(&amp_vals),
                    self.data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .collect::<Vec<DVector<f64>>>()
            .iter()
            .sum(); // TODO: replace with custom implementation of accurate crate's trait
        #[cfg(feature = "rayon")]
        let mc_term: DVector<f64> = self
            .accmc_evaluator
            .dataset
            .events
            .par_iter()
            .zip(mc_resources.caches.par_iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.accmc_evaluator.amplitudes.len()];
                self.accmc_evaluator
                    .amplitudes
                    .iter()
                    .zip(mc_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&mc_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.accmc_evaluator
                            .amplitudes
                            .iter()
                            .zip(mc_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&mc_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.accmc_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .collect::<Vec<DVector<f64>>>()
            .iter()
            .sum();
        #[cfg(not(feature = "rayon"))]
        let data_term: DVector<f64> = self
            .data_evaluator
            .dataset
            .events
            .iter()
            .zip(data_resources.caches.iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.data_evaluator.amplitudes.len()];
                self.data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&data_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.data_evaluator
                            .amplitudes
                            .iter()
                            .zip(data_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&data_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.data_evaluator.expression.evaluate(&amp_vals),
                    self.data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .sum();
        #[cfg(not(feature = "rayon"))]
        let mc_term: DVector<f64> = self
            .accmc_evaluator
            .dataset
            .events
            .iter()
            .zip(mc_resources.caches.iter())
            .map(|(event, cache)| {
                let mut gradient_values =
                    vec![DVector::zeros(parameters.len()); self.accmc_evaluator.amplitudes.len()];
                self.accmc_evaluator
                    .amplitudes
                    .iter()
                    .zip(mc_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&mc_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.accmc_evaluator
                            .amplitudes
                            .iter()
                            .zip(mc_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&mc_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.accmc_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .sum();
        -2.0 * (data_term - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> DVector<f64> {
        let local_evaluation_vec = self
            .evaluate_gradient_local(parameters)
            .data
            .as_vec()
            .to_vec();
        let mut flattened_result_buffer = vec![0.0; world.size() as usize * parameters.len()];
        world.all_gather_into(&local_evaluation_vec, &mut flattened_result_buffer);
        flattened_result_buffer
            .chunks(parameters.len())
            .map(DVector::from_row_slice)
            .sum::<DVector<f64>>()
    }
}

impl LikelihoodTerm for NLL {
    /// Get the list of parameter names in the order they appear in the [`NLL::evaluate`]
    /// method.
    fn parameters(&self) -> Vec<String> {
        self.data_evaluator
            .resources
            .read()
            .parameters
            .iter()
            .cloned()
            .collect()
    }

    /// Evaluate the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in [`Expression::norm_sqr`](`laddu_core::Expression::norm_sqr`). The
    /// result is given by the following formula:
    ///
    /// ```math
    /// NLL(\vec{p}) = -2 \left(\sum_{e \in \text{Data}} \text{weight}(e) \ln(\mathcal{L}(e)) - \frac{1}{N_{\text{MC}_A}} \sum_{e \in \text{MC}_A} \text{weight}(e) \mathcal{L}(e) \right)
    /// ```
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_mpi(parameters, &world);
            }
        }
        self.evaluate_local(parameters)
    }

    /// Evaluate the gradient of the stored [`Model`] over the events in the [`Dataset`]
    /// stored by the [`Evaluator`] with the given values for free parameters. This method takes the
    /// real part of the given expression (discarding the imaginary part entirely, which
    /// does not matter if expressions are coherent sums wrapped in
    /// [`Expression::norm_sqr`](`laddu_core::Expression::norm_sqr`).
    fn evaluate_gradient(&self, parameters: &[f64]) -> DVector<f64> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_gradient_mpi(parameters, &world);
            }
        }
        self.evaluate_gradient_local(parameters)
    }
}

/// A stochastic [`NLL`] term.
///
/// While a regular [`NLL`] will operate over the entire dataset, this term will only operate over
/// a random subset of the data, determined by the `batch_size` parameter. This will make the
/// objective function faster to evaluate at the cost of adding random noise to the likelihood.
#[derive(Clone)]
pub struct StochasticNLL {
    /// A handle to the original [`NLL`] term.
    pub nll: NLL,
    n: usize,
    batch_size: usize,
    batch_indices: Arc<Mutex<Vec<usize>>>,
    rng: Arc<Mutex<Rng>>,
}

impl LikelihoodTerm for StochasticNLL {
    fn parameters(&self) -> Vec<String> {
        self.nll.parameters()
    }
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_mpi(parameters, &self.batch_indices.lock(), &world);
            }
        }
        #[cfg(feature = "rayon")]
        let n_data_batch_local = self
            .batch_indices
            .lock()
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = self
            .batch_indices
            .lock()
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .sum_with_accumulator::<Klein<f64>>();
        self.evaluate_local(parameters, &self.batch_indices.lock(), n_data_batch_local)
    }

    fn evaluate_gradient(&self, parameters: &[f64]) -> DVector<f64> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_gradient_mpi(parameters, &self.batch_indices.lock(), &world);
            }
        }
        #[cfg(feature = "rayon")]
        let n_data_batch_local = self
            .batch_indices
            .lock()
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = self
            .batch_indices
            .lock()
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .sum_with_accumulator::<Klein<f64>>();
        self.evaluate_gradient_local(parameters, &self.batch_indices.lock(), n_data_batch_local)
    }
    fn update(&self) {
        self.resample();
    }
}

impl StochasticNLL {
    /// Generate a new [`StochasticNLL`] with the given [`NLL`], batch size, and optional random seed
    ///
    /// # See Also
    ///
    /// [`NLL::to_stochastic`]
    pub fn new(nll: NLL, batch_size: usize, seed: Option<usize>) -> Self {
        let mut rng = seed.map_or_else(Rng::new, |seed| Rng::with_seed(seed as u64));
        let n = nll.data_evaluator.dataset.n_events();
        assert!(batch_size > 0 && batch_size <= n);
        let batch_indices = rng.subset(batch_size, n);
        Self {
            nll,
            n,
            batch_size,
            batch_indices: Arc::new(Mutex::new(batch_indices)),
            rng: Arc::new(Mutex::new(rng)),
        }
    }
    /// Resample the batch indices used in evaluation
    pub fn resample(&self) {
        let mut rng = self.rng.lock();
        *self.batch_indices.lock() = rng.subset(self.batch_size, self.n);
    }
    fn evaluate_local(&self, parameters: &[f64], indices: &[usize], n_data_batch: f64) -> f64 {
        let data_result = self
            .nll
            .data_evaluator
            .evaluate_batch_local(parameters, indices);
        let mc_result = self.nll.accmc_evaluator.evaluate_local(parameters);
        let n_mc = self.nll.accmc_evaluator.dataset.n_events_weighted();
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        #[cfg(feature = "rayon")]
        {
            let data_term: f64 = indices
                .par_iter()
                .map(|&i| {
                    let e = &self.nll.data_evaluator.dataset.events[i];
                    let l = data_result[i];
                    e.weight * l.re.ln()
                })
                .parallel_sum_with_accumulator::<Klein<f64>>();
            let mc_term: f64 = mc_result
                .par_iter()
                .zip(self.nll.accmc_evaluator.dataset.events.par_iter())
                .map(|(l, e)| e.weight * l.re)
                .parallel_sum_with_accumulator::<Klein<f64>>();
            -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let data_term: f64 = indices
                .iter()
                .map(|&i| {
                    let e = &self.nll.data_evaluator.dataset.events[i];
                    let l = data_result[i];
                    e.weight * l.re.ln()
                })
                .sum_with_accumulator::<Klein<f64>>();
            let mc_term: f64 = mc_result
                .iter()
                .zip(self.nll.accmc_evaluator.dataset.events.iter())
                .map(|(l, e)| e.weight * l.re)
                .sum_with_accumulator::<Klein<f64>>();
            -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
        }
    }

    #[cfg(feature = "mpi")]
    fn evaluate_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> f64 {
        let (_, _, locals) = self
            .nll
            .data_evaluator
            .dataset
            .get_counts_displs_locals_from_indices(indices, world);
        let mut n_data_batch_partitioned: Vec<f64> = vec![0.0; world.size() as usize];
        #[cfg(feature = "rayon")]
        let n_data_batch_local = indices
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .sum_with_accumulator::<Klein<f64>>();
        world.all_gather_into(&n_data_batch_local, &mut n_data_batch_partitioned);
        #[cfg(feature = "rayon")]
        let n_data_batch = n_data_batch_partitioned
            .into_par_iter()
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch = n_data_batch_partitioned
            .into_iter()
            .sum_with_accumulator::<Klein<f64>>();
        let local_evaluation = self.evaluate_local(parameters, &locals, n_data_batch);
        let mut buffer: Vec<f64> = vec![0.0; world.size() as usize];
        world.all_gather_into(&local_evaluation, &mut buffer);
        buffer.iter().sum()
    }

    fn evaluate_gradient_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
        n_data_batch: f64,
    ) -> DVector<f64> {
        let data_resources = self.nll.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let mc_resources = self.nll.accmc_evaluator.resources.read();
        let mc_parameters = Parameters::new(parameters, &mc_resources.constants);
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let n_mc = self.nll.accmc_evaluator.dataset.n_events_weighted();
        #[cfg(feature = "rayon")]
        let data_term: DVector<f64> = self
            .nll
            .data_evaluator
            .dataset
            .events
            .par_iter()
            .zip(data_resources.caches.par_iter())
            .enumerate()
            .filter_map(|(i, (event, cache))| {
                if indices.contains(&i) {
                    Some((event, cache))
                } else {
                    None
                }
            })
            .map(|(event, cache)| {
                let mut gradient_values = vec![
                    DVector::zeros(parameters.len());
                    self.nll.data_evaluator.amplitudes.len()
                ];
                self.nll
                    .data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&data_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.nll
                            .data_evaluator
                            .amplitudes
                            .iter()
                            .zip(data_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&data_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.nll.data_evaluator.expression.evaluate(&amp_vals),
                    self.nll
                        .data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .collect::<Vec<DVector<f64>>>()
            .iter()
            .sum(); // TODO: replace with custom implementation of accurate crate's trait
        #[cfg(feature = "rayon")]
        let mc_term: DVector<f64> = self
            .nll
            .accmc_evaluator
            .dataset
            .events
            .par_iter()
            .zip(mc_resources.caches.par_iter())
            .map(|(event, cache)| {
                let mut gradient_values = vec![
                    DVector::zeros(parameters.len());
                    self.nll.accmc_evaluator.amplitudes.len()
                ];
                self.nll
                    .accmc_evaluator
                    .amplitudes
                    .iter()
                    .zip(mc_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&mc_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.nll
                            .accmc_evaluator
                            .amplitudes
                            .iter()
                            .zip(mc_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&mc_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.nll
                        .accmc_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .collect::<Vec<DVector<f64>>>()
            .iter()
            .sum();
        #[cfg(not(feature = "rayon"))]
        let data_term: DVector<f64> = self
            .nll
            .data_evaluator
            .dataset
            .events
            .iter()
            .zip(data_resources.caches.iter())
            .enumerate()
            .filter_map(|(i, (event, cache))| {
                if indices.contains(&i) {
                    Some((event, cache))
                } else {
                    None
                }
            })
            .map(|(event, cache)| {
                let mut gradient_values = vec![
                    DVector::zeros(parameters.len());
                    self.nll.data_evaluator.amplitudes.len()
                ];
                self.nll
                    .data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&data_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.nll
                            .data_evaluator
                            .amplitudes
                            .iter()
                            .zip(data_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&data_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.nll.data_evaluator.expression.evaluate(&amp_vals),
                    self.nll
                        .data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .sum();
        #[cfg(not(feature = "rayon"))]
        let mc_term: DVector<f64> = self
            .nll
            .accmc_evaluator
            .dataset
            .events
            .iter()
            .zip(mc_resources.caches.iter())
            .map(|(event, cache)| {
                let mut gradient_values = vec![
                    DVector::zeros(parameters.len());
                    self.nll.accmc_evaluator.amplitudes.len()
                ];
                self.nll
                    .accmc_evaluator
                    .amplitudes
                    .iter()
                    .zip(mc_resources.active.iter())
                    .zip(gradient_values.iter_mut())
                    .for_each(|((amp, active), grad)| {
                        if *active {
                            amp.compute_gradient(&mc_parameters, event, cache, grad)
                        }
                    });
                (
                    event.weight,
                    AmplitudeValues(
                        self.nll
                            .accmc_evaluator
                            .amplitudes
                            .iter()
                            .zip(mc_resources.active.iter())
                            .map(|(amp, active)| {
                                if *active {
                                    amp.compute(&mc_parameters, event, cache)
                                } else {
                                    Complex64::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(parameters.len(), gradient_values),
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.nll
                        .accmc_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .sum();
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> DVector<f64> {
        let (_, _, locals) = self
            .nll
            .data_evaluator
            .dataset
            .get_counts_displs_locals_from_indices(indices, world);
        let mut n_data_batch_partitioned: Vec<f64> = vec![0.0; world.size() as usize];
        #[cfg(feature = "rayon")]
        let n_data_batch_local = indices
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.events[i].weight)
            .sum_with_accumulator::<Klein<f64>>();
        world.all_gather_into(&n_data_batch_local, &mut n_data_batch_partitioned);
        #[cfg(feature = "rayon")]
        let n_data_batch = n_data_batch_partitioned
            .into_par_iter()
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch = n_data_batch_partitioned
            .into_iter()
            .sum_with_accumulator::<Klein<f64>>();
        let local_evaluation_vec = self
            .evaluate_gradient_local(parameters, &locals, n_data_batch)
            .data
            .as_vec()
            .to_vec();
        let mut flattened_result_buffer = vec![0.0; world.size() as usize * parameters.len()];
        world.all_gather_into(&local_evaluation_vec, &mut flattened_result_buffer);
        flattened_result_buffer
            .chunks(parameters.len())
            .map(DVector::from_row_slice)
            .sum::<DVector<f64>>()
    }
}

/// A (extended) negative log-likelihood evaluator
///
/// Parameters
/// ----------
/// model: Model
///     The Model to evaluate
/// ds_data : Dataset
///     A Dataset representing true signal data
/// ds_accmc : Dataset
///     A Dataset of physically flat accepted Monte Carlo data used for normalization
///
#[cfg(feature = "python")]
#[pyclass(name = "NLL", module = "laddu")]
#[derive(Clone)]
pub struct PyNLL(pub Box<NLL>);

#[cfg(feature = "python")]
#[pymethods]
impl PyNLL {
    #[new]
    #[pyo3(signature = (model, ds_data, ds_accmc))]
    fn new(model: &PyModel, ds_data: &PyDataset, ds_accmc: &PyDataset) -> Self {
        Self(NLL::new(&model.0, &ds_data.0, &ds_accmc.0))
    }
    /// The underlying signal dataset used in calculating the NLL
    ///
    /// Returns
    /// -------
    /// Dataset
    ///
    #[getter]
    fn data(&self) -> PyDataset {
        PyDataset(self.0.data_evaluator.dataset.clone())
    }
    /// The underlying accepted Monte Carlo dataset used in calculating the NLL
    ///
    /// Returns
    /// -------
    /// Dataset
    ///
    #[getter]
    fn accmc(&self) -> PyDataset {
        PyDataset(self.0.accmc_evaluator.dataset.clone())
    }
    /// Turn an ``NLL`` into a ``StochasticNLL``
    ///
    /// Parameters
    /// ----------
    /// batch_size : int
    ///     The batch size for the data
    /// seed : int, default=None
    ///
    /// Returns
    /// -------
    /// StochasticNLL
    ///
    #[pyo3(signature = (batch_size, *, seed=None))]
    fn to_stochastic(&self, batch_size: usize, seed: Option<usize>) -> PyStochasticNLL {
        PyStochasticNLL(self.0.to_stochastic(batch_size, seed))
    }
    /// Turn an ``NLL`` into a term that can be used by a ``LikelihoodManager``
    ///
    /// Returns
    /// -------
    /// term : LikelihoodTerm
    ///     The isolated NLL which can be used to build more complex models
    ///
    ///
    fn to_term(&self) -> PyLikelihoodTerm {
        PyLikelihoodTerm(self.0.clone())
    }
    /// The names of the free parameters used to evaluate the NLL
    ///
    /// Returns
    /// -------
    /// parameters : list of str
    ///
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }
    /// Activates Amplitudes in the NLL by name
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names of Amplitudes to be activated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    ///
    fn activate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            self.0.activate(&string_arg)?;
        } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            self.0.activate_many(&vec)?;
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        }
        Ok(())
    }
    /// Activates all Amplitudes in the NLL
    ///
    fn activate_all(&self) {
        self.0.activate_all();
    }
    /// Deactivates Amplitudes in the NLL by name
    ///
    /// Deactivated Amplitudes act as zeros in the NLL
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names of Amplitudes to be deactivated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    ///
    fn deactivate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            self.0.deactivate(&string_arg)?;
        } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            self.0.deactivate_many(&vec)?;
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        }
        Ok(())
    }
    /// Deactivates all Amplitudes in the NLL
    ///
    fn deactivate_all(&self) {
        self.0.deactivate_all();
    }
    /// Isolates Amplitudes in the NLL by name
    ///
    /// Activates the Amplitudes given in `arg` and deactivates the rest
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names of Amplitudes to be isolated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    ///
    fn isolate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            self.0.isolate(&string_arg)?;
        } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            self.0.isolate_many(&vec)?;
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        }
        Ok(())
    }
    /// Evaluate the extended negative log-likelihood over the stored Datasets
    ///
    /// This is defined as
    ///
    /// .. math:: NLL(\vec{p}; D, MC) = -2 \left( \sum_{e \in D} (e_w \log(\mathcal{L}(e))) - \frac{1}{N_{MC}} \sum_{e \in MC} (e_w \mathcal{L}(e)) \right)
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// result : float
    ///     The total negative log-likelihood
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate(&self, parameters: Vec<f64>, threads: Option<usize>) -> PyResult<f64> {
        #[cfg(feature = "rayon")]
        {
            Ok(ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| LikelihoodTerm::evaluate(self.0.as_ref(), &parameters)))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(LikelihoodTerm::evaluate(self.0.as_ref(), &parameters))
        }
    }
    /// Evaluate the gradient of the negative log-likelihood over the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` array of representing the gradient of the negative log-likelihood over each parameter
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool or problem creating the resulting
    ///     ``numpy`` array
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or(0))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| self.0.evaluate_gradient(&parameters))
                    .as_slice(),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray1::from_slice(
                py,
                self.0.evaluate_gradient(&parameters).as_slice(),
            ))
        }
    }
    /// Project the model over the Monte Carlo dataset with the given parameter values
    ///
    /// This is defined as
    ///
    /// .. math:: e_w(\vec{p}) = \frac{e_w}{N_{MC}} \mathcal{L}(e)
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// mc_evaluator: Evaluator, optional
    ///     Project using the given Evaluator or use the stored ``accmc`` if None
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     Weights for every Monte Carlo event which represent the fit to data
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool or problem creating the resulting
    ///     ``numpy`` array
    ///
    #[pyo3(signature = (parameters, *, mc_evaluator = None, threads=None))]
    fn project<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or(0))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| {
                        self.0
                            .project(&parameters, mc_evaluator.map(|pyeval| pyeval.0.clone()))
                    })
                    .as_slice(),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray1::from_slice(
                py,
                self.0
                    .project(&parameters, mc_evaluator.map(|pyeval| pyeval.0.clone()))
                    .as_slice(),
            ))
        }
    }

    /// Project the model over the Monte Carlo dataset with the given parameter values, first
    /// isolating the given terms by name. The NLL is then reset to its previous state of
    /// activation.
    ///
    /// This is defined as
    ///
    /// .. math:: e_w(\vec{p}) = \frac{e_w}{N_{MC}} \mathcal{L}(e)
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// arg : str or list of str
    ///     Names of Amplitudes to be isolated
    /// mc_evaluator: Evaluator, optional
    ///     Project using the given Evaluator or use the stored ``accmc`` if None
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     Weights for every Monte Carlo event which represent the fit to data
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool or problem creating the resulting
    ///     ``numpy`` array
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    ///
    #[pyo3(signature = (parameters, arg, *, mc_evaluator = None, threads=None))]
    fn project_with<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        arg: &Bound<'_, PyAny>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let names = if let Ok(string_arg) = arg.extract::<String>() {
            vec![string_arg]
        } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            vec
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        };
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or(0))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| {
                        self.0.project_with(
                            &parameters,
                            &names,
                            mc_evaluator.map(|pyeval| pyeval.0.clone()),
                        )
                    })?
                    .as_slice(),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray1::from_slice(
                py,
                self.0
                    .project_with(
                        &parameters,
                        &names,
                        mc_evaluator.map(|pyeval| pyeval.0.clone()),
                    )?
                    .as_slice(),
            ))
        }
    }

    /// Minimize the NLL with respect to the free parameters in the model
    ///
    /// This method "runs the fit". Given an initial position `p0`, this
    /// method performs a minimization over the negative log-likelihood, optimizing the model
    /// over the stored signal data and Monte Carlo.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial parameters at the start of optimization
    /// bounds : list of tuple of float or None, optional
    ///     Optional bounds on each parameter (use None or an infinity for no bound)
    /// method : {'lbfgsb', 'nelder-mead', 'adam', 'pso'}
    ///     The minimization algorithm to use
    /// settings : dict, optional
    ///     Settings for the minimization algorithm (see notes)
    /// observers : MinimizerObserver or list of MinimizerObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MinimizerTerminator or list of MinimizerTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to 0 will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// MinimizationSummary
    ///     The status of the minimization algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the minimization algorithm as keyword arguments. Each
    /// algorithm has different settings:
    ///
    /// L-BFGS-B
    /// ========
    /// m : int, default=10
    ///     The number of saved corrections to the approximated Hessian.
    /// skip_hessian : bool, default=False
    ///     If True, the exact Hessian will not be calculated.
    /// line_search : dict
    ///     Settings for the line search (see next section).
    /// eps_f : float, default=`MACH_EPS^(1/2)`
    ///     The tolerance for stopping based on the change in function value.
    /// eps_g : float, default=`MACH_EPS^(1/3)`
    ///     The tolerance for stopping based on the change in function gradient.
    /// eps_norm_g : float, default=1e-5
    ///     The tolerance for stopping based on the change in the infinity-norm of the function gradient.
    ///
    /// Line Search
    /// ===========
    /// method : {"morethuente", "hagerzhang"}
    ///     The line search method to use.
    /// max_iterations : int, default=100
    ///     The maximum number of line search iterations.
    /// c1 : float, default=1e-4
    ///     The first Wolfe condition constant (More-Thuente only).
    /// c2 : float, default=0.9
    ///     The second Wolfe condition constant (More-Thuente only).
    /// max_zoom : int, default=100
    ///     The maximum number of zoom steps (More-Thuente only).
    /// delta : float, default=0.1
    ///     The first Wolfe condition constant (Hager-Zhang only).
    /// sigma : float, default=0.9
    ///     The second Wolfe condition constant (Hager-Zhang only).
    /// epsilon : float, default=`MACH_EPS^(1/3)`
    ///     The tolerance parameter on approximate Wolfe termination (Hager-Zhang only).
    /// theta : float, default=0.5
    ///     The split ratio for interval updates (defaults to bisection) (Hager-Zhang only).
    /// gamma : float, default=0.66
    ///     A parameter which determines when a bisection is performed (Hager-Zhang only).
    /// max_bisects : int, default=50
    ///     The maximum number of allowed bisections (Hager-Zhang only).
    ///
    /// Adam
    /// ====
    /// beta_c : float, default=0.9
    ///     The slope of the exponential moving average used to terminate the algorithm.
    /// eps_loss : float, default=`MACH_EPS^(1/2)`
    ///     The minimum change in exponential moving average loss which will increase the patience counter.
    /// patience : int, default=1
    ///     The number of allowed iterations with no improvement in the loss (according to an exponential moving average) before the algorithm terminates.
    ///
    /// Nelder-Mead
    /// ===========
    /// alpha : float, default=1.0
    ///     The reflection coefficient.
    /// beta : float, default=2.0
    ///     The expansion coefficient.
    /// gamma : float, default=0.5
    ///     The contraction coefficient.
    /// delta : float, default=0.5
    ///     The shrink coefficient.
    /// adaptive : bool, default=False
    ///     Use adaptive hyperparameters according to Gao and Han[Gao]_.
    /// expansion_method : {"greedyminimization", "greedyexpansion"}
    ///     Greedy minimization will favor points which minimize faster, but greedy expansion may explore a space more efficiently. See [Lagarias]_ for details.
    /// simplex_construction_method : {"scaledorthogonal", "orthogonal", "custom"}
    ///     The method used to generate the initial simplex.
    /// orthogonal_multiplier : float, default=1.05
    ///     Multiplier used on nonzero coordinates of the initial vertex in simplex generation (scaled orthogonal method).
    /// orthogonal_zero_step : float, default=0.00025
    ///     Value to use for coordinates of the initial vertex which are zero in simplex generation (scaled orthogonal method).
    /// simplex_size : float, default=1.0
    ///     The step in each orthogonal direction from the initial vertex in simplex generation (orthogonal method).
    /// simplex : list of list of floats
    ///     Specify the initial simplex directly. Each entry in the list must be a unique point in the parameter space. The initial vertex is also included, so this argument must specify as many vertices as there are dimensions in the parameter space. This must be specified if simplex_construction_method is set to "custom".
    /// f_terminator : {"stddev", "absolute", "amoeba"} or None, default="stddev"
    ///     Set the method to terminate the algorithm based on the function values over the simplex. See [Singer]_ for details. Set to None to skip this check.
    /// eps_f : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the f_terminator method.
    /// x_terminator : {"singer", "diameter", "higham", "rowan"} or None, default="singer"
    ///     Set the method to terminate the algorithm based on the position of simplex vertices. See [Singer]_ for details. Set to None to skip this check.
    /// eps_x : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the x_terminator method.
    ///
    /// Particle Swarm Optimization (PSO)
    /// =================================
    /// swarm_position_initializer : {"randominlimits", "latinhypercube", "custom"}
    ///     The method used to initialize the swarm position. The "randominlimits" and "latinhypercube" methods require swarm_position_bounds and swarm_size to be specified, and they ignore the initial position given when constructing the swarm (this behavior may change in the future). The "custom" method requires swarm to be specified and does include the initial position.
    /// swarm_position_bounds : list of tuple of floats or None
    ///     Bounds used when randomly generating a swarm with either the "randominlimits" or "latinhypercube" swarm_position_initializer.
    /// swarm_size : int
    ///     The number of particles in the swarm when using the "randominlimits" or "latinhypercube" swarm_position_initializer.
    /// swarm : list of list of floats
    ///     A list of positions of each particle in the swarm. This argument is required when using the "custom" swarm_position_initializer.
    /// swarm_topology : {"global", "ring"}
    ///     The topology connecting particles in the swarm.
    /// swarm_update_method : {"sync", "synchronous", "async", "asynchronous"}
    ///     Synchronous updates update positions and targets in separate loops (slower but sometimes more stable) while asynchronous updates them in the same loop (faster but sometimes less stable).
    /// swarm_boundary_method : {"inf", "shr"}
    ///     The boundary method used for the swarm. "inf" sets infeasable values to +inf while "shr" shrinks the velocity vector to place the particle on the boundary where it would cross.
    /// use_transform : bool, default=False
    ///     If True, the algorithm will ignore the swarm_boundary_method and instead perform a coordinate transformation on the swarm to ensure the swarm is within bounds.
    /// swarm_velocity_bounds : list of tuple of floats or None, optional
    ///     Bounds used when randomly generating the initial velocity of each particle in the swarm. If not specified, initial velocities are set to zero.
    /// omega : float, default=0.8
    ///     The inertial weight.
    /// c1 : float, default = 0.1
    ///     The cognitive weight.
    /// c2 : float, default = 0.1
    ///     The social weight.
    ///
    /// .. rubric:: References
    ///
    /// .. [Gao] F. Gao and L. Han, “Implementing the Nelder-Mead simplex algorithm with adaptive parameters,” Comput Optim Appl, vol. 51, no. 1, pp. 259–277, May 2010, doi: 10.1007/s10589-010-9329-3.
    /// .. [Lagarias] J. C. Lagarias, J. A. Reeds, M. H. Wright, and P. E. Wright, “Convergence Properties of the Nelder--Mead Simplex Method in Low Dimensions,” SIAM J. Optim., vol. 9, no. 1, pp. 112–147, Jan. 1998, doi: 10.1137/s1052623496303470.
    /// .. [Singer] S. Singer and S. Singer, “Efficient Implementation of the Nelder–Mead Search Algorithm,” Appl Numer Analy &amp; Comput, vol. 1, no. 2, pp. 524–534, Dec. 2004, doi: 10.1002/anac.200410015.
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="lbfgsb".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        p0: Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'py, PyDict>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMinimizationSummary> {
        let full_settings = PyDict::new(py);
        if let Some(bounds) = bounds {
            full_settings.set_item("bounds", bounds)?;
        }
        full_settings.set_item("method", method)?;
        if let Some(observers) = observers {
            full_settings.set_item("observers", observers)?;
        }
        if let Some(terminators) = terminators {
            full_settings.set_item("terminators", terminators)?;
        }
        if let Some(max_steps) = max_steps {
            full_settings.set_item("max_steps", max_steps)?;
        }
        full_settings.set_item("debug", debug)?;
        full_settings.set_item("threads", threads)?;
        if let Some(settings) = settings {
            full_settings.set_item("settings", settings)?;
        }
        let result = self
            .0
            .minimize(MinimizationSettings::from_pyargs(&p0, &full_settings)?)?;
        Ok(PyMinimizationSummary(result))
    }

    /// Run an MCMC algorithm on the free parameters of the NLL's model
    ///
    /// This method can be used to sample the underlying log-likelihood given an initial
    /// position for each walker `p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial positions of each walker with dimension (n_walkers, n_parameters)
    /// bounds : list of tuple of float or None, optional
    ///     Optional bounds on each parameter (use None or an infinity for no bound)
    /// method : {'aies', 'ess'}
    ///     The MCMC algorithm to use
    /// settings : dict, optional
    ///     Settings for the MCMC algorithm (see notes)
    /// observers : MCMCObserver or list of MCMCObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MCMCTerminator or list of MCMCTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to 0 will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// MCMCSummary
    ///     The status of the MCMC algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the MCMC algorithm as keyword arguments. Each
    /// algorithm has different settings.
    ///
    /// AIES (Affine-Invariant Ensemble Sampler) [Goodman]_
    /// =============================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('stretch', {'a': 2.0}, 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('stretch' or 'walk') and the last is the frequency that move should be used relative
    ///     to the sum of all frequencies given across all moves. An optional middle dictionary
    ///     parameter may be provided to specify properties of moves which support it. For the AIES
    ///     algorithm, the stretch move may use the 'a' parameter to specify the scaling parameter
    ///     (2.0 by default).
    ///
    /// ESS (Ensemble Slice Sampler) [Karamanis]_
    /// =================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('differential', 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('differential', 'gaussian', or 'global') and the last is the frequency that move
    ///     should be used relative to the sum of all frequencies given across all moves. An
    ///     optional middle dictionary parameter may be provided to specify properties of moves
    ///     which support it. For the ESS algorithm, the global move may use a 'scale' parameter
    ///     (1.0 by default) which rescales steps within a local cluster, a 'rescale_cov' parameter
    ///     (0.001 by default) which rescales the covariance matrix of clusters, and an
    ///     'n_components' parameter (5 by default) which represents the number of mixture
    ///     components to use for clustering (should be larger than the expected number of modes).
    /// n_adaptive : int, default=0
    ///     The number of adaptive moves to perform at the start of sampling
    /// max_steps : int, default=10000
    ///     The maximum number of expansions/contractions to perform at each step in the algorithm
    /// mu : float, default = 1.0
    ///     The adaptive scaling parameter (only applies if 'n_adaptive' is greater than zero)
    ///
    /// .. rubric:: References
    ///
    /// .. [Goodman] J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” CAMCoS, vol. 5, no. 1, pp. 65–80, Jan. 2010, doi: 10.2140/camcos.2010.5.65.
    /// .. [Karamanis] M. Karamanis and F. Beutler, “Ensemble slice sampling,” Stat Comput, vol. 31, no. 5, Aug. 2021, doi: 10.1007/s11222-021-10038-2.
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="aies".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc<'py>(
        &self,
        py: Python<'py>,
        p0: Vec<Vec<f64>>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'py, PyDict>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMCMCSummary> {
        let full_settings = PyDict::new(py);
        if let Some(bounds) = bounds {
            full_settings.set_item("bounds", bounds)?;
        }
        full_settings.set_item("method", method)?;
        if let Some(observers) = observers {
            full_settings.set_item("observers", observers)?;
        }
        if let Some(terminators) = terminators {
            full_settings.set_item("terminators", terminators)?;
        }
        if let Some(max_steps) = max_steps {
            full_settings.set_item("max_steps", max_steps)?;
        }
        full_settings.set_item("debug", debug)?;
        full_settings.set_item("threads", threads)?;
        if let Some(settings) = settings {
            full_settings.set_item("settings", settings)?;
        }
        let result = self.0.mcmc(MCMCSettings::from_pyargs(
            &p0.into_iter().map(|p| DVector::from_vec(p)).collect(),
            &full_settings,
        )?)?;
        Ok(PyMCMCSummary(result))
    }
}

/// A stochastic (extended) negative log-likelihood evaluator
///
/// This evaluator operates on a subset of the data, which may improve performance for large
/// datasets at the cost of adding noise to the likelihood.
///
/// Notes
/// -----
/// See the `NLL.to_stochastic` method for details.
#[cfg(feature = "python")]
#[pyclass(name = "StochasticNLL", module = "laddu")]
#[derive(Clone)]
pub struct PyStochasticNLL(pub StochasticNLL);

#[cfg(feature = "python")]
#[pymethods]
impl PyStochasticNLL {
    /// The NLL term containing the underlying model and evaluators
    ///
    /// Returns
    /// -------
    /// NLL
    ///
    #[getter]
    fn nll(&self) -> PyNLL {
        PyNLL(Box::new(self.0.nll.clone()))
    }
    /// Minimize the StochasticNLL with respect to the free parameters in the model
    ///
    /// This method "runs the fit". Given an initial position `p0`, this
    /// method performs a minimization over the negative log-likelihood, optimizing the model
    /// over the stored signal data and Monte Carlo.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial parameters at the start of optimization
    /// bounds : list of tuple of float or None, optional
    ///     Optional bounds on each parameter (use None or an infinity for no bound)
    /// method : {'lbfgsb', 'nelder-mead', 'adam', 'pso'}
    ///     The minimization algorithm to use
    /// settings : dict, optional
    ///     Settings for the minimization algorithm (see notes)
    /// observers : MinimizerObserver or list of MinimizerObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MinimizerTerminator or list of MinimizerTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to 0 will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// MinimizationSummary
    ///     The status of the minimization algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the minimization algorithm as keyword arguments. Each
    /// algorithm has different settings:
    ///
    /// L-BFGS-B
    /// ========
    /// m : int, default=10
    ///     The number of saved corrections to the approximated Hessian.
    /// skip_hessian : bool, default=False
    ///     If True, the exact Hessian will not be calculated.
    /// line_search : dict
    ///     Settings for the line search (see next section).
    /// eps_f : float, default=`MACH_EPS^(1/2)`
    ///     The tolerance for stopping based on the change in function value.
    /// eps_g : float, default=`MACH_EPS^(1/3)`
    ///     The tolerance for stopping based on the change in function gradient.
    /// eps_norm_g : float, default=1e-5
    ///     The tolerance for stopping based on the change in the infinity-norm of the function gradient.
    ///
    /// Line Search
    /// ===========
    /// method : {"morethuente", "hagerzhang"}
    ///     The line search method to use.
    /// max_iterations : int, default=100
    ///     The maximum number of line search iterations.
    /// c1 : float, default=1e-4
    ///     The first Wolfe condition constant (More-Thuente only).
    /// c2 : float, default=0.9
    ///     The second Wolfe condition constant (More-Thuente only).
    /// max_zoom : int, default=100
    ///     The maximum number of zoom steps (More-Thuente only).
    /// delta : float, default=0.1
    ///     The first Wolfe condition constant (Hager-Zhang only).
    /// sigma : float, default=0.9
    ///     The second Wolfe condition constant (Hager-Zhang only).
    /// epsilon : float, default=`MACH_EPS^(1/3)`
    ///     The tolerance parameter on approximate Wolfe termination (Hager-Zhang only).
    /// theta : float, default=0.5
    ///     The split ratio for interval updates (defaults to bisection) (Hager-Zhang only).
    /// gamma : float, default=0.66
    ///     A parameter which determines when a bisection is performed (Hager-Zhang only).
    /// max_bisects : int, default=50
    ///     The maximum number of allowed bisections (Hager-Zhang only).
    ///
    /// Adam
    /// ====
    /// beta_c : float, default=0.9
    ///     The slope of the exponential moving average used to terminate the algorithm.
    /// eps_loss : float, default=`MACH_EPS^(1/2)`
    ///     The minimum change in exponential moving average loss which will increase the patience counter.
    /// patience : int, default=1
    ///     The number of allowed iterations with no improvement in the loss (according to an exponential moving average) before the algorithm terminates.
    ///
    /// Nelder-Mead
    /// ===========
    /// alpha : float, default=1.0
    ///     The reflection coefficient.
    /// beta : float, default=2.0
    ///     The expansion coefficient.
    /// gamma : float, default=0.5
    ///     The contraction coefficient.
    /// delta : float, default=0.5
    ///     The shrink coefficient.
    /// adaptive : bool, default=False
    ///     Use adaptive hyperparameters according to Gao and Han[Gao]_.
    /// expansion_method : {"greedyminimization", "greedyexpansion"}
    ///     Greedy minimization will favor points which minimize faster, but greedy expansion may explore a space more efficiently. See [Lagarias]_ for details.
    /// simplex_construction_method : {"scaledorthogonal", "orthogonal", "custom"}
    ///     The method used to generate the initial simplex.
    /// orthogonal_multiplier : float, default=1.05
    ///     Multiplier used on nonzero coordinates of the initial vertex in simplex generation (scaled orthogonal method).
    /// orthogonal_zero_step : float, default=0.00025
    ///     Value to use for coordinates of the initial vertex which are zero in simplex generation (scaled orthogonal method).
    /// simplex_size : float, default=1.0
    ///     The step in each orthogonal direction from the initial vertex in simplex generation (orthogonal method).
    /// simplex : list of list of floats
    ///     Specify the initial simplex directly. Each entry in the list must be a unique point in the parameter space. The initial vertex is also included, so this argument must specify as many vertices as there are dimensions in the parameter space. This must be specified if simplex_construction_method is set to "custom".
    /// f_terminator : {"stddev", "absolute", "amoeba"} or None, default="stddev"
    ///     Set the method to terminate the algorithm based on the function values over the simplex. See [Singer]_ for details. Set to None to skip this check.
    /// eps_f : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the f_terminator method.
    /// x_terminator : {"singer", "diameter", "higham", "rowan"} or None, default="singer"
    ///     Set the method to terminate the algorithm based on the position of simplex vertices. See [Singer]_ for details. Set to None to skip this check.
    /// eps_x : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the x_terminator method.
    ///
    /// Particle Swarm Optimization (PSO)
    /// =================================
    /// swarm_position_initializer : {"randominlimits", "latinhypercube", "custom"}
    ///     The method used to initialize the swarm position. The "randominlimits" and "latinhypercube" methods require swarm_position_bounds and swarm_size to be specified, and they ignore the initial position given when constructing the swarm (this behavior may change in the future). The "custom" method requires swarm to be specified and does include the initial position.
    /// swarm_position_bounds : list of tuple of floats or None
    ///     Bounds used when randomly generating a swarm with either the "randominlimits" or "latinhypercube" swarm_position_initializer.
    /// swarm_size : int
    ///     The number of particles in the swarm when using the "randominlimits" or "latinhypercube" swarm_position_initializer.
    /// swarm : list of list of floats
    ///     A list of positions of each particle in the swarm. This argument is required when using the "custom" swarm_position_initializer.
    /// swarm_topology : {"global", "ring"}
    ///     The topology connecting particles in the swarm.
    /// swarm_update_method : {"sync", "synchronous", "async", "asynchronous"}
    ///     Synchronous updates update positions and targets in separate loops (slower but sometimes more stable) while asynchronous updates them in the same loop (faster but sometimes less stable).
    /// swarm_boundary_method : {"inf", "shr"}
    ///     The boundary method used for the swarm. "inf" sets infeasable values to +inf while "shr" shrinks the velocity vector to place the particle on the boundary where it would cross.
    /// use_transform : bool, default=False
    ///     If True, the algorithm will ignore the swarm_boundary_method and instead perform a coordinate transformation on the swarm to ensure the swarm is within bounds.
    /// swarm_velocity_bounds : list of tuple of floats or None, optional
    ///     Bounds used when randomly generating the initial velocity of each particle in the swarm. If not specified, initial velocities are set to zero.
    /// omega : float, default=0.8
    ///     The inertial weight.
    /// c1 : float, default = 0.1
    ///     The cognitive weight.
    /// c2 : float, default = 0.1
    ///     The social weight.
    ///
    /// .. rubric:: References
    ///
    /// .. [Gao] F. Gao and L. Han, “Implementing the Nelder-Mead simplex algorithm with adaptive parameters,” Comput Optim Appl, vol. 51, no. 1, pp. 259–277, May 2010, doi: 10.1007/s10589-010-9329-3.
    /// .. [Lagarias] J. C. Lagarias, J. A. Reeds, M. H. Wright, and P. E. Wright, “Convergence Properties of the Nelder--Mead Simplex Method in Low Dimensions,” SIAM J. Optim., vol. 9, no. 1, pp. 112–147, Jan. 1998, doi: 10.1137/s1052623496303470.
    /// .. [Singer] S. Singer and S. Singer, “Efficient Implementation of the Nelder–Mead Search Algorithm,” Appl Numer Analy &amp; Comput, vol. 1, no. 2, pp. 524–534, Dec. 2004, doi: 10.1002/anac.200410015.
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="lbfgsb".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        p0: Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'py, PyDict>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMinimizationSummary> {
        let full_settings = PyDict::new(py);
        if let Some(bounds) = bounds {
            full_settings.set_item("bounds", bounds)?;
        }
        full_settings.set_item("method", method)?;
        if let Some(observers) = observers {
            full_settings.set_item("observers", observers)?;
        }
        if let Some(terminators) = terminators {
            full_settings.set_item("terminators", terminators)?;
        }
        if let Some(max_steps) = max_steps {
            full_settings.set_item("max_steps", max_steps)?;
        }
        full_settings.set_item("debug", debug)?;
        full_settings.set_item("threads", threads)?;
        if let Some(settings) = settings {
            full_settings.set_item("settings", settings)?;
        }
        let result = self
            .0
            .minimize(MinimizationSettings::from_pyargs(&p0, &full_settings)?)?;
        Ok(PyMinimizationSummary(result))
    }
    /// Run an MCMC algorithm on the free parameters of the StochasticNLL's model
    ///
    /// This method can be used to sample the underlying log-likelihood given an initial
    /// position for each walker `p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial positions of each walker with dimension (n_walkers, n_parameters)
    /// bounds : list of tuple of float or None, optional
    ///     Optional bounds on each parameter (use None or an infinity for no bound)
    /// method : {'aies', 'ess'}
    ///     The MCMC algorithm to use
    /// settings : dict, optional
    ///     Settings for the MCMC algorithm (see notes)
    /// observers : MCMCObserver or list of MCMCObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MCMCTerminator or list of MCMCTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to 0 will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// MCMCSummary
    ///     The status of the MCMC algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the MCMC algorithm as keyword arguments. Each
    /// algorithm has different settings.
    ///
    /// AIES (Affine-Invariant Ensemble Sampler) [Goodman]_
    /// =============================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('stretch', {'a': 2.0}, 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('stretch' or 'walk') and the last is the frequency that move should be used relative
    ///     to the sum of all frequencies given across all moves. An optional middle dictionary
    ///     parameter may be provided to specify properties of moves which support it. For the AIES
    ///     algorithm, the stretch move may use the 'a' parameter to specify the scaling parameter
    ///     (2.0 by default).
    ///
    /// ESS (Ensemble Slice Sampler) [Karamanis]_
    /// =================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('differential', 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('differential', 'gaussian', or 'global') and the last is the frequency that move
    ///     should be used relative to the sum of all frequencies given across all moves. An
    ///     optional middle dictionary parameter may be provided to specify properties of moves
    ///     which support it. For the ESS algorithm, the global move may use a 'scale' parameter
    ///     (1.0 by default) which rescales steps within a local cluster, a 'rescale_cov' parameter
    ///     (0.001 by default) which rescales the covariance matrix of clusters, and an
    ///     'n_components' parameter (5 by default) which represents the number of mixture
    ///     components to use for clustering (should be larger than the expected number of modes).
    /// n_adaptive : int, default=0
    ///     The number of adaptive moves to perform at the start of sampling
    /// max_steps : int, default=10000
    ///     The maximum number of expansions/contractions to perform at each step in the algorithm
    /// mu : float, default = 1.0
    ///     The adaptive scaling parameter (only applies if 'n_adaptive' is greater than zero)
    ///
    /// .. rubric:: References
    ///
    /// .. [Goodman] J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” CAMCoS, vol. 5, no. 1, pp. 65–80, Jan. 2010, doi: 10.2140/camcos.2010.5.65.
    /// .. [Karamanis] M. Karamanis and F. Beutler, “Ensemble slice sampling,” Stat Comput, vol. 31, no. 5, Aug. 2021, doi: 10.1007/s11222-021-10038-2.
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="aies".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc<'py>(
        &self,
        py: Python<'py>,
        p0: Vec<Vec<f64>>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'py, PyDict>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMCMCSummary> {
        let full_settings = PyDict::new(py);
        if let Some(bounds) = bounds {
            full_settings.set_item("bounds", bounds)?;
        }
        full_settings.set_item("method", method)?;
        if let Some(observers) = observers {
            full_settings.set_item("observers", observers)?;
        }
        if let Some(terminators) = terminators {
            full_settings.set_item("terminators", terminators)?;
        }
        if let Some(max_steps) = max_steps {
            full_settings.set_item("max_steps", max_steps)?;
        }
        full_settings.set_item("debug", debug)?;
        full_settings.set_item("threads", threads)?;
        if let Some(settings) = settings {
            full_settings.set_item("settings", settings)?;
        }
        let result = self.0.mcmc(MCMCSettings::from_pyargs(
            &p0.into_iter().map(|p| DVector::from_vec(p)).collect(),
            &full_settings,
        )?)?;
        Ok(PyMCMCSummary(result))
    }
}

/// An identifier that can be used like an [`AmplitudeID`](`laddu_core::amplitudes::AmplitudeID`) to combine registered
/// [`LikelihoodTerm`]s.
#[derive(Clone, Debug)]
pub struct LikelihoodID(usize, Option<String>);

impl Display for LikelihoodID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = &self.1 {
            write!(f, "{}({})", name, self.0)
        } else {
            write!(f, "({})", self.0)
        }
    }
}

/// An object which holds a registered ``LikelihoodTerm``
///
/// See Also
/// --------
/// laddu.LikelihoodManager.register
///
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodID", module = "laddu")]
#[derive(Clone)]
pub struct PyLikelihoodID(LikelihoodID);

#[cfg(feature = "python")]
#[pymethods]
impl PyLikelihoodID {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(self.0.clone() + other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() + other_expr.0.clone(),
            ))
        } else if let Ok(int) = other.extract::<usize>() {
            if int == 0 {
                Ok(PyLikelihoodExpression(LikelihoodExpression::Term(
                    self.0.clone(),
                )))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(other_aid.0.clone() + self.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                other_expr.0.clone() + self.0.clone(),
            ))
        } else if let Ok(int) = other.extract::<usize>() {
            if int == 0 {
                Ok(PyLikelihoodExpression(LikelihoodExpression::Term(
                    self.0.clone(),
                )))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(self.0.clone() * other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() * other_expr.0.clone(),
            ))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(other_aid.0.clone() * self.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                other_expr.0.clone() * self.0.clone(),
            ))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A [`Manager`](`laddu_core::Manager`) but for [`LikelihoodTerm`]s.
#[derive(Default, Clone)]
pub struct LikelihoodManager {
    terms: Vec<Box<dyn LikelihoodTerm>>,
    param_name_to_index: HashMap<String, usize>,
    param_names: Vec<String>,
    param_layouts: Vec<Vec<usize>>,
    param_counts: Vec<usize>,
}

impl LikelihoodManager {
    /// Register a [`LikelihoodTerm`] to get a [`LikelihoodID`] which can be combined with others
    /// to form [`LikelihoodExpression`]s which can be minimized.
    pub fn register(&mut self, term: Box<dyn LikelihoodTerm>) -> LikelihoodID {
        let term_idx = self.terms.len();
        for param_name in term.parameters() {
            if !self.param_name_to_index.contains_key(&param_name) {
                self.param_name_to_index
                    .insert(param_name.clone(), self.param_name_to_index.len());
                self.param_names.push(param_name.clone());
            }
        }
        let param_layout: Vec<usize> = term
            .parameters()
            .iter()
            .map(|name| self.param_name_to_index[name])
            .collect();
        let param_count = term.parameters().len();
        self.param_layouts.push(param_layout);
        self.param_counts.push(param_count);
        self.terms.push(term.clone());

        LikelihoodID(term_idx, None)
    }

    /// Register (and name) a [`LikelihoodTerm`] to get a [`LikelihoodID`] which can be combined with others
    /// to form [`LikelihoodExpression`]s which can be minimized.
    pub fn register_with_name<T: AsRef<str>>(
        &mut self,
        term: Box<dyn LikelihoodTerm>,
        name: T,
    ) -> LikelihoodID {
        let term_idx = self.terms.len();
        for param_name in term.parameters() {
            if !self.param_name_to_index.contains_key(&param_name) {
                self.param_name_to_index
                    .insert(param_name.clone(), self.param_name_to_index.len());
                self.param_names.push(param_name.clone());
            }
        }
        let param_layout: Vec<usize> = term
            .parameters()
            .iter()
            .map(|name| self.param_name_to_index[name])
            .collect();
        let param_count = term.parameters().len();
        self.param_layouts.push(param_layout);
        self.param_counts.push(param_count);
        self.terms.push(term.clone());

        LikelihoodID(term_idx, Some(name.as_ref().to_string().clone()))
    }

    /// Return all of the parameter names of registered [`LikelihoodTerm`]s in order. This only
    /// returns the unique names in the order they should be input when evaluated with a
    /// [`LikelihoodEvaluator`].
    pub fn parameters(&self) -> Vec<String> {
        self.param_names.clone()
    }

    /// Load a [`LikelihoodExpression`] to generate a [`LikelihoodEvaluator`] that can be
    /// minimized.
    pub fn load(&self, likelihood_expression: &LikelihoodExpression) -> LikelihoodEvaluator {
        LikelihoodEvaluator {
            likelihood_manager: self.clone(),
            likelihood_expression: likelihood_expression.clone(),
        }
    }
}

/// A class which can be used to register LikelihoodTerms and store precalculated data
///
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodManager", module = "laddu")]
#[derive(Clone)]
pub struct PyLikelihoodManager(LikelihoodManager);

#[cfg(feature = "python")]
#[pymethods]
impl PyLikelihoodManager {
    #[new]
    fn new() -> Self {
        Self(LikelihoodManager::default())
    }
    /// Register a LikelihoodTerm with the LikelihoodManager
    ///
    /// Parameters
    /// ----------
    /// term : LikelihoodTerm
    ///     The LikelihoodTerm to register
    ///
    /// Returns
    /// -------
    /// LikelihoodID
    ///     A reference to the registered ``likelihood`` that can be used to form complex
    ///     LikelihoodExpressions
    ///
    #[pyo3(signature = (likelihood_term, *, name = None))]
    fn register(
        &mut self,
        likelihood_term: &PyLikelihoodTerm,
        name: Option<String>,
    ) -> PyLikelihoodID {
        if let Some(name) = name {
            PyLikelihoodID(self.0.register_with_name(likelihood_term.0.clone(), name))
        } else {
            PyLikelihoodID(self.0.register(likelihood_term.0.clone()))
        }
    }
    /// The free parameters used by all terms in the LikelihoodManager
    ///
    /// Returns
    /// -------
    /// parameters : list of str
    ///     The list of parameter names
    ///
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }
    /// Load a LikelihoodExpression by precalculating each term over their internal Datasets
    ///
    /// Parameters
    /// ----------
    /// likelihood_expression : LikelihoodExpression
    ///     The expression to use in precalculation
    ///
    /// Returns
    /// -------
    /// LikelihoodEvaluator
    ///     An object that can be used to evaluate the `likelihood_expression` over all managed
    ///     terms
    ///
    /// Notes
    /// -----
    /// While the given `likelihood_expression` will be the one evaluated in the end, all registered
    /// Amplitudes will be loaded, and all of their parameters will be included in the final
    /// expression. These parameters will have no effect on evaluation, but they must be
    /// included in function calls.
    ///
    /// See Also
    /// --------
    /// LikelihoodManager.parameters
    ///
    fn load(&self, likelihood_expression: &PyLikelihoodExpression) -> PyLikelihoodEvaluator {
        PyLikelihoodEvaluator(self.0.load(&likelihood_expression.0))
    }
}

#[derive(Debug)]
struct LikelihoodValues(Vec<f64>);

#[derive(Debug)]
struct LikelihoodGradients(Vec<DVector<f64>>);

/// A combination of [`LikelihoodTerm`]s as well as sums and products of them.
#[derive(Clone, Default)]
pub enum LikelihoodExpression {
    /// A expression equal to zero.
    #[default]
    Zero,
    /// A expression equal to one.
    One,
    /// A registered [`LikelihoodTerm`] referenced by an [`LikelihoodID`].
    Term(LikelihoodID),
    /// The sum of two [`LikelihoodExpression`]s.
    Add(Box<LikelihoodExpression>, Box<LikelihoodExpression>),
    /// The product of two [`LikelihoodExpression`]s.
    Mul(Box<LikelihoodExpression>, Box<LikelihoodExpression>),
}

impl Debug for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(f, "", "", "")
    }
}

impl Display for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LikelihoodExpression {
    fn evaluate(&self, likelihood_values: &LikelihoodValues) -> f64 {
        match self {
            LikelihoodExpression::Zero => 0.0,
            LikelihoodExpression::One => 1.0,
            LikelihoodExpression::Term(lid) => likelihood_values.0[lid.0],
            LikelihoodExpression::Add(a, b) => {
                a.evaluate(likelihood_values) + b.evaluate(likelihood_values)
            }
            LikelihoodExpression::Mul(a, b) => {
                a.evaluate(likelihood_values) * b.evaluate(likelihood_values)
            }
        }
    }
    fn evaluate_gradient(
        &self,
        likelihood_values: &LikelihoodValues,
        likelihood_gradients: &LikelihoodGradients,
    ) -> DVector<f64> {
        match self {
            LikelihoodExpression::Zero => DVector::zeros(0),
            LikelihoodExpression::One => DVector::zeros(0),
            LikelihoodExpression::Term(lid) => likelihood_gradients.0[lid.0].clone(),
            LikelihoodExpression::Add(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
            }
            LikelihoodExpression::Mul(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    * b.evaluate(likelihood_values)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
                        * a.evaluate(likelihood_values)
            }
        }
    }
    /// Credit to Daniel Janus: <https://blog.danieljanus.pl/2023/07/20/iterating-trees/>
    fn write_tree(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        parent_prefix: &str,
        immediate_prefix: &str,
        parent_suffix: &str,
    ) -> std::fmt::Result {
        let display_string = match self {
            Self::Zero => "0".to_string(),
            Self::One => "1".to_string(),
            Self::Term(lid) => {
                format!("{}", lid.0)
            }
            Self::Add(_, _) => "+".to_string(),
            Self::Mul(_, _) => "*".to_string(),
        };
        writeln!(f, "{}{}{}", parent_prefix, immediate_prefix, display_string)?;
        match self {
            Self::Term(_) | Self::Zero | Self::One => {}
            Self::Add(a, b) | Self::Mul(a, b) => {
                let terms = [a, b];
                let mut it = terms.iter().peekable();
                let child_prefix = format!("{}{}", parent_prefix, parent_suffix);
                while let Some(child) = it.next() {
                    match it.peek() {
                        Some(_) => child.write_tree(f, &child_prefix, "├─ ", "│  "),
                        None => child.write_tree(f, &child_prefix, "└─ ", "   "),
                    }?;
                }
            }
        }
        Ok(())
    }
}

impl_op_ex!(+ |a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression { LikelihoodExpression::Add(Box::new(a.clone()), Box::new(b.clone()))});
impl_op_ex!(
    *|a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression {
        LikelihoodExpression::Mul(Box::new(a.clone()), Box::new(b.clone()))
    }
);
impl_op_ex_commutative!(+ |a: &LikelihoodID, b: &LikelihoodExpression| -> LikelihoodExpression { LikelihoodExpression::Add(Box::new(LikelihoodExpression::Term(a.clone())), Box::new(b.clone()))});
impl_op_ex_commutative!(
    *|a: &LikelihoodID, b: &LikelihoodExpression| -> LikelihoodExpression {
        LikelihoodExpression::Mul(
            Box::new(LikelihoodExpression::Term(a.clone())),
            Box::new(b.clone()),
        )
    }
);
impl_op_ex!(+ |a: &LikelihoodID, b: &LikelihoodID| -> LikelihoodExpression { LikelihoodExpression::Add(Box::new(LikelihoodExpression::Term(a.clone())), Box::new(LikelihoodExpression::Term(b.clone())))});
impl_op_ex!(
    *|a: &LikelihoodID, b: &LikelihoodID| -> LikelihoodExpression {
        LikelihoodExpression::Mul(
            Box::new(LikelihoodExpression::Term(a.clone())),
            Box::new(LikelihoodExpression::Term(b.clone())),
        )
    }
);

/// A mathematical expression formed from LikelihoodIDs
///
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodExpression", module = "laddu")]
#[derive(Clone)]
pub struct PyLikelihoodExpression(LikelihoodExpression);

/// A convenience method to sum sequences of LikelihoodExpressions
///
#[cfg(feature = "python")]
#[pyfunction(name = "likelihood_sum")]
pub fn py_likelihood_sum(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyLikelihoodExpression> {
    if terms.is_empty() {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::Zero));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyLikelihoodExpression>() {
            return Ok(expression);
        }
        if let Ok(py_amplitude_id) = term.extract::<PyLikelihoodID>() {
            return Ok(PyLikelihoodExpression(LikelihoodExpression::Term(
                py_amplitude_id.0,
            )));
        }
        return Err(PyTypeError::new_err(
            "Item is neither a PyLikelihoodExpression nor a PyLikelihoodID",
        ));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::Zero));
    };
    if let Ok(first_expression) = first_term.extract::<PyLikelihoodExpression>() {
        let mut summation = first_expression.clone();
        for term in iter {
            summation = summation.__add__(term)?;
        }
        return Ok(summation);
    }
    if let Ok(first_likelihood_id) = first_term.extract::<PyLikelihoodID>() {
        let mut summation =
            PyLikelihoodExpression(LikelihoodExpression::Term(first_likelihood_id.0));
        for term in iter {
            summation = summation.__add__(term)?;
        }
        return Ok(summation);
    }
    Err(PyTypeError::new_err(
        "Elements must be PyLikelihoodExpression or PyLikelihoodID",
    ))
}

/// A convenience method to multiply sequences of LikelihoodExpressions
///
#[cfg(feature = "python")]
#[pyfunction(name = "likelihood_product")]
pub fn py_likelihood_product(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyLikelihoodExpression> {
    if terms.is_empty() {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::One));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyLikelihoodExpression>() {
            return Ok(expression);
        }
        if let Ok(py_amplitude_id) = term.extract::<PyLikelihoodID>() {
            return Ok(PyLikelihoodExpression(LikelihoodExpression::Term(
                py_amplitude_id.0,
            )));
        }
        return Err(PyTypeError::new_err(
            "Item is neither a PyLikelihoodExpression nor a PyLikelihoodID",
        ));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::One));
    };
    if let Ok(first_expression) = first_term.extract::<PyLikelihoodExpression>() {
        let mut product = first_expression.clone();
        for term in iter {
            product = product.__mul__(term)?;
        }
        return Ok(product);
    }
    if let Ok(first_likelihood_id) = first_term.extract::<PyLikelihoodID>() {
        let mut product = PyLikelihoodExpression(LikelihoodExpression::Term(first_likelihood_id.0));
        for term in iter {
            product = product.__mul__(term)?;
        }
        return Ok(product);
    }
    Err(PyTypeError::new_err(
        "Elements must be PyLikelihoodExpression or PyLikelihoodID",
    ))
}

/// A convenience class representing a zero-valued LikelihoodExpression
///
#[cfg(feature = "python")]
#[pyfunction(name = "LikelihoodZero")]
pub fn py_likelihood_zero() -> PyLikelihoodExpression {
    PyLikelihoodExpression(LikelihoodExpression::Zero)
}

/// A convenience class representing a unit-valued LikelihoodExpression
///
#[cfg(feature = "python")]
#[pyfunction(name = "LikelihoodOne")]
pub fn py_likelihood_one() -> PyLikelihoodExpression {
    PyLikelihoodExpression(LikelihoodExpression::One)
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLikelihoodExpression {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(self.0.clone() + other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() + other_expr.0.clone(),
            ))
        } else if let Ok(int) = other.extract::<usize>() {
            if int == 0 {
                Ok(PyLikelihoodExpression(self.0.clone()))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(other_aid.0.clone() + self.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                other_expr.0.clone() + self.0.clone(),
            ))
        } else if let Ok(int) = other.extract::<usize>() {
            if int == 0 {
                Ok(PyLikelihoodExpression(self.0.clone()))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(self.0.clone() * other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() * other_expr.0.clone(),
            ))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyLikelihoodID>>() {
            Ok(PyLikelihoodExpression(self.0.clone() * other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() * other_expr.0.clone(),
            ))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A structure to evaluate and minimize combinations of [`LikelihoodTerm`]s.
#[derive(Clone)]
pub struct LikelihoodEvaluator {
    likelihood_manager: LikelihoodManager,
    likelihood_expression: LikelihoodExpression,
}

impl LikelihoodTerm for LikelihoodEvaluator {
    fn update(&self) {
        self.likelihood_manager
            .terms
            .iter()
            .for_each(|term| term.update())
    }
    /// The parameter names used in [`LikelihoodEvaluator::evaluate`]'s input in order.
    fn parameters(&self) -> Vec<String> {
        self.likelihood_manager.parameters()
    }
    /// A function that can be called to evaluate the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        let mut param_buffers: Vec<Vec<f64>> = self
            .likelihood_manager
            .param_counts
            .iter()
            .map(|&count| vec![0.0; count])
            .collect();
        for (layout, buffer) in self
            .likelihood_manager
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.likelihood_manager
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect(),
        );
        self.likelihood_expression.evaluate(&likelihood_values)
    }

    /// Evaluate the gradient of the stored [`LikelihoodExpression`] over the events in the [`Dataset`]
    /// stored by the [`LikelihoodEvaluator`] with the given values for free parameters.
    fn evaluate_gradient(&self, parameters: &[f64]) -> DVector<f64> {
        let mut param_buffers: Vec<Vec<f64>> = self
            .likelihood_manager
            .param_counts
            .iter()
            .map(|&count| vec![0.0; count])
            .collect();
        for (layout, buffer) in self
            .likelihood_manager
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.likelihood_manager
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect(),
        );
        let mut gradient_buffers: Vec<DVector<f64>> = (0..self.likelihood_manager.terms.len())
            .map(|_| DVector::zeros(self.likelihood_manager.param_names.len()))
            .collect();
        for (((term, param_buffer), gradient_buffer), layout) in self
            .likelihood_manager
            .terms
            .iter()
            .zip(param_buffers.iter())
            .zip(gradient_buffers.iter_mut())
            .zip(self.likelihood_manager.param_layouts.iter())
        {
            let term_gradient = term.evaluate_gradient(param_buffer); // This has a local layout
            for (term_idx, &buffer_idx) in layout.iter().enumerate() {
                gradient_buffer[buffer_idx] = term_gradient[term_idx] // This has a global layout
            }
        }
        let likelihood_gradients = LikelihoodGradients(gradient_buffers);
        self.likelihood_expression
            .evaluate_gradient(&likelihood_values, &likelihood_gradients)
    }
}

/// A class which can be used to evaluate a collection of LikelihoodTerms managed by a
/// LikelihoodManager
///
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodEvaluator", module = "laddu")]
pub struct PyLikelihoodEvaluator(LikelihoodEvaluator);

#[cfg(feature = "python")]
#[pymethods]
impl PyLikelihoodEvaluator {
    /// A list of the names of the free parameters across all terms in all models
    ///
    /// Returns
    /// -------
    /// parameters : list of str
    ///
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }
    /// Evaluate the sum of all terms in the evaluator
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// result : float
    ///     The total negative log-likelihood summed over all terms
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate(&self, parameters: Vec<f64>, threads: Option<usize>) -> PyResult<f64> {
        #[cfg(feature = "rayon")]
        {
            Ok(ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| self.0.evaluate(&parameters)))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(self.0.evaluate(&parameters))
        }
    }
    /// Evaluate the gradient of the sum of all terms in the evaluator
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` array of representing the gradient of the sum of all terms in the
    ///     evaluator
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool or problem creating the resulting
    ///     ``numpy`` array
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or(0))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| self.0.evaluate_gradient(&parameters))
                    .as_slice(),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray1::from_slice(
                py,
                self.0.evaluate_gradient(&parameters).as_slice(),
            ))
        }
    }
    /// Minimize the LikelihoodTerm with respect to the free parameters in the model
    ///
    /// This method "runs the fit". Given an initial position `p0`, this
    /// method performs a minimization over the likelihood term, optimizing the model
    /// over the stored signal data and Monte Carlo.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial parameters at the start of optimization
    /// bounds : list of tuple of float or None, optional
    ///     Optional bounds on each parameter (use None or an infinity for no bound)
    /// method : {'lbfgsb', 'nelder-mead', 'adam', 'pso'}
    ///     The minimization algorithm to use
    /// settings : dict, optional
    ///     Settings for the minimization algorithm (see notes)
    /// observers : MinimizerObserver or list of MinimizerObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MinimizerTerminator or list of MinimizerTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to 0 will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// MinimizationSummary
    ///     The status of the minimization algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the minimization algorithm as keyword arguments. Each
    /// algorithm has different settings:
    ///
    /// L-BFGS-B
    /// ========
    /// m : int, default=10
    ///     The number of saved corrections to the approximated Hessian.
    /// skip_hessian : bool, default=False
    ///     If True, the exact Hessian will not be calculated.
    /// line_search : dict
    ///     Settings for the line search (see next section).
    /// eps_f : float,  default=`MACH_EPS^(1/2)`
    ///     The tolerance for stopping based on the change in function value.
    /// eps_g : float, default=`MACH_EPS^(1/3)`
    ///     The tolerance for stopping based on the change in function gradient.
    /// eps_norm_g : float, default=1e-5
    ///     The tolerance for stopping based on the change in the infinity-norm of the function gradient.
    ///
    /// Line Search
    /// ===========
    /// method : {"morethuente", "hagerzhang"}
    ///     The line search method to use.
    /// max_iterations : int, default=100
    ///     The maximum number of line search iterations.
    /// c1 : float, default=1e-4
    ///     The first Wolfe condition constant (More-Thuente only).
    /// c2 : float, default=0.9
    ///     The second Wolfe condition constant (More-Thuente only).
    /// max_zoom : int, default=100
    ///     The maximum number of zoom steps (More-Thuente only).
    /// delta : float, default=0.1
    ///     The first Wolfe condition constant (Hager-Zhang only).
    /// sigma : float, default=0.9
    ///     The second Wolfe condition constant (Hager-Zhang only).
    /// epsilon : float, default=`MACH_EPS^(1/3)`
    ///     The tolerance parameter on approximate Wolfe termination (Hager-Zhang only).
    /// theta : float, default=0.5
    ///     The split ratio for interval updates (defaults to bisection) (Hager-Zhang only).
    /// gamma : float, default=0.66
    ///     A parameter which determines when a bisection is performed (Hager-Zhang only).
    /// max_bisects : int, default=50
    ///     The maximum number of allowed bisections (Hager-Zhang only).
    ///
    /// Adam
    /// ====
    /// beta_c : float, default=0.9
    ///     The slope of the exponential moving average used to terminate the algorithm.
    /// eps_loss : float, default=`MACH_EPS^(1/2)`
    ///     The minimum change in exponential moving average loss which will increase the patience counter.
    /// patience : int, default=1
    ///     The number of allowed iterations with no improvement in the loss (according to an exponential moving average) before the algorithm terminates.
    ///
    /// Nelder-Mead
    /// ===========
    /// alpha : float, default=1.0
    ///     The reflection coefficient.
    /// beta : float, default=2.0
    ///     The expansion coefficient.
    /// gamma : float, default=0.5
    ///     The contraction coefficient.
    /// delta : float, default=0.5
    ///     The shrink coefficient.
    /// adaptive : bool, default=False
    ///     Use adaptive hyperparameters according to Gao and Han[Gao]_.
    /// expansion_method : {"greedyminimization", "greedyexpansion"}
    ///     Greedy minimization will favor points which minimize faster, but greedy expansion may explore a space more efficiently. See [Lagarias]_ for details.
    /// simplex_construction_method : {"scaledorthogonal", "orthogonal", "custom"}
    ///     The method used to generate the initial simplex.
    /// orthogonal_multiplier : float, default=1.05
    ///     Multiplier used on nonzero coordinates of the initial vertex in simplex generation (scaled orthogonal method).
    /// orthogonal_zero_step : float, default=0.00025
    ///     Value to use for coordinates of the initial vertex which are zero in simplex generation (scaled orthogonal method).
    /// simplex_size : float, default=1.0
    ///     The step in each orthogonal direction from the initial vertex in simplex generation (orthogonal method).
    /// simplex : list of list of floats
    ///     Specify the initial simplex directly. Each entry in the list must be a unique point in the parameter space. The initial vertex is also included, so this argument must specify as many vertices as there are dimensions in the parameter space. This must be specified if simplex_construction_method is set to "custom".
    /// f_terminator : {"stddev", "absolute", "amoeba"} or None, default="stddev"
    ///     Set the method to terminate the algorithm based on the function values over the simplex. See [Singer]_ for details. Set to None to skip this check.
    /// eps_f : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the f_terminator method.
    /// x_terminator : {"singer", "diameter", "higham", "rowan"} or None, default="singer"
    ///     Set the method to terminate the algorithm based on the position of simplex vertices. See [Singer]_ for details. Set to None to skip this check.
    /// eps_x : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the x_terminator method.
    ///
    /// Particle Swarm Optimization (PSO)
    /// =================================
    /// swarm_position_initializer : {"randominlimits", "latinhypercube", "custom"}
    ///     The method used to initialize the swarm position. The "randominlimits" and "latinhypercube" methods require swarm_position_bounds and swarm_size to be specified, and they ignore the initial position given when constructing the swarm (this behavior may change in the future). The "custom" method requires swarm to be specified and does include the initial position.
    /// swarm_position_bounds : list of tuple of floats or None
    ///     Bounds used when randomly generating a swarm with either the "randominlimits" or "latinhypercube" swarm_position_initializer.
    /// swarm_size : int
    ///     The number of particles in the swarm when using the "randominlimits" or "latinhypercube" swarm_position_initializer.
    /// swarm : list of list of floats
    ///     A list of positions of each particle in the swarm. This argument is required when using the "custom" swarm_position_initializer.
    /// swarm_topology : {"global", "ring"}
    ///     The topology connecting particles in the swarm.
    /// swarm_update_method : {"sync", "synchronous", "async", "asynchronous"}
    ///     Synchronous updates update positions and targets in separate loops (slower but sometimes more stable) while asynchronous updates them in the same loop (faster but sometimes less stable).
    /// swarm_boundary_method : {"inf", "shr"}
    ///     The boundary method used for the swarm. "inf" sets infeasable values to +inf while "shr" shrinks the velocity vector to place the particle on the boundary where it would cross.
    /// use_transform : bool, default=False
    ///     If True, the algorithm will ignore the swarm_boundary_method and instead perform a coordinate transformation on the swarm to ensure the swarm is within bounds.
    /// swarm_velocity_bounds : list of tuple of floats or None, optional
    ///     Bounds used when randomly generating the initial velocity of each particle in the swarm. If not specified, initial velocities are set to zero.
    /// omega : float, default=0.8
    ///     The inertial weight.
    /// c1 : float, default = 0.1
    ///     The cognitive weight.
    /// c2 : float, default = 0.1
    ///     The social weight.
    ///
    /// .. rubric:: References
    ///
    /// .. [Gao] F. Gao and L. Han, “Implementing the Nelder-Mead simplex algorithm with adaptive parameters,” Comput Optim Appl, vol. 51, no. 1, pp. 259–277, May 2010, doi: 10.1007/s10589-010-9329-3.
    /// .. [Lagarias] J. C. Lagarias, J. A. Reeds, M. H. Wright, and P. E. Wright, “Convergence Properties of the Nelder--Mead Simplex Method in Low Dimensions,” SIAM J. Optim., vol. 9, no. 1, pp. 112–147, Jan. 1998, doi: 10.1137/s1052623496303470.
    /// .. [Singer] S. Singer and S. Singer, “Efficient Implementation of the Nelder–Mead Search Algorithm,” Appl Numer Analy &amp; Comput, vol. 1, no. 2, pp. 524–534, Dec. 2004, doi: 10.1002/anac.200410015.
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="lbfgsb".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        p0: Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'py, PyDict>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMinimizationSummary> {
        let full_settings = PyDict::new(py);
        if let Some(bounds) = bounds {
            full_settings.set_item("bounds", bounds)?;
        }
        full_settings.set_item("method", method)?;
        if let Some(observers) = observers {
            full_settings.set_item("observers", observers)?;
        }
        if let Some(terminators) = terminators {
            full_settings.set_item("terminators", terminators)?;
        }
        if let Some(max_steps) = max_steps {
            full_settings.set_item("max_steps", max_steps)?;
        }
        full_settings.set_item("debug", debug)?;
        full_settings.set_item("threads", threads)?;
        if let Some(settings) = settings {
            full_settings.set_item("settings", settings)?;
        }
        let result = self
            .0
            .minimize(MinimizationSettings::from_pyargs(&p0, &full_settings)?)?;
        Ok(PyMinimizationSummary(result))
    }
    /// Run an MCMC algorithm on the free parameters of the LikelihoodTerm's model
    ///
    /// This method can be used to sample the underlying likelihood term given an initial
    /// position for each walker `p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial positions of each walker with dimension (n_walkers, n_parameters)
    /// bounds : list of tuple of float or None, optional
    ///     Optional bounds on each parameter (use None or an infinity for no bound)
    /// method : {'aies', 'ess'}
    ///     The MCMC algorithm to use
    /// settings : dict, optional
    ///     Settings for the MCMC algorithm (see notes)
    /// observers : MCMCObserver or list of MCMCObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MCMCTerminator or list of MCMCTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to 0 will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// MCMCSummary
    ///     The status of the MCMC algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the MCMC algorithm as keyword arguments. Each
    /// algorithm has different settings.
    ///
    /// AIES (Affine-Invariant Ensemble Sampler) [Goodman]_
    /// =============================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('stretch', {'a': 2.0}, 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('stretch' or 'walk') and the last is the frequency that move should be used relative
    ///     to the sum of all frequencies given across all moves. An optional middle dictionary
    ///     parameter may be provided to specify properties of moves which support it. For the AIES
    ///     algorithm, the stretch move may use the 'a' parameter to specify the scaling parameter
    ///     (2.0 by default).
    ///
    /// ESS (Ensemble Slice Sampler) [Karamanis]_
    /// =================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('differential', 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('differential', 'gaussian', or 'global') and the last is the frequency that move
    ///     should be used relative to the sum of all frequencies given across all moves. An
    ///     optional middle dictionary parameter may be provided to specify properties of moves
    ///     which support it. For the ESS algorithm, the global move may use a 'scale' parameter
    ///     (1.0 by default) which rescales steps within a local cluster, a 'rescale_cov' parameter
    ///     (0.001 by default) which rescales the covariance matrix of clusters, and an
    ///     'n_components' parameter (5 by default) which represents the number of mixture
    ///     components to use for clustering (should be larger than the expected number of modes).
    /// n_adaptive : int, default=0
    ///     The number of adaptive moves to perform at the start of sampling
    /// max_steps : int, default=10000
    ///     The maximum number of expansions/contractions to perform at each step in the algorithm
    /// mu : float, default = 1.0
    ///     The adaptive scaling parameter (only applies if 'n_adaptive' is greater than zero)
    ///
    /// .. [Goodman] J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” CAMCoS, vol. 5, no. 1, pp. 65–80, Jan. 2010, doi: 10.2140/camcos.2010.5.65.
    /// .. [Karamanis] M. Karamanis and F. Beutler, “Ensemble slice sampling,” Stat Comput, vol. 31, no. 5, Aug. 2021, doi: 10.1007/s11222-021-10038-2.
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="aies".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc<'py>(
        &self,
        py: Python<'py>,
        p0: Vec<Vec<f64>>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'py, PyDict>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMCMCSummary> {
        let full_settings = PyDict::new(py);
        if let Some(bounds) = bounds {
            full_settings.set_item("bounds", bounds)?;
        }
        full_settings.set_item("method", method)?;
        if let Some(observers) = observers {
            full_settings.set_item("observers", observers)?;
        }
        if let Some(terminators) = terminators {
            full_settings.set_item("terminators", terminators)?;
        }
        if let Some(max_steps) = max_steps {
            full_settings.set_item("max_steps", max_steps)?;
        }
        full_settings.set_item("debug", debug)?;
        full_settings.set_item("threads", threads)?;
        if let Some(settings) = settings {
            full_settings.set_item("settings", settings)?;
        }
        let result = self.0.mcmc(MCMCSettings::from_pyargs(
            &p0.into_iter().map(|p| DVector::from_vec(p)).collect(),
            &full_settings,
        )?)?;
        Ok(PyMCMCSummary(result))
    }
}

/// A [`LikelihoodTerm`] which represents a single scaling parameter.
#[derive(Clone)]
pub struct LikelihoodScalar(String);

impl LikelihoodScalar {
    /// Create a new [`LikelihoodScalar`] with a parameter with the given name.
    pub fn new<T: AsRef<str>>(name: T) -> Box<Self> {
        Self(name.as_ref().into()).into()
    }
}

impl LikelihoodTerm for LikelihoodScalar {
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        parameters[0]
    }

    fn evaluate_gradient(&self, _parameters: &[f64]) -> DVector<f64> {
        DVector::from_vec(vec![1.0])
    }

    fn parameters(&self) -> Vec<String> {
        vec![self.0.clone()]
    }
}

/// A parameterized scalar term which can be added to a LikelihoodManager
///
/// Parameters
/// ----------
/// name : str
///     The name of the new scalar parameter
///
/// Returns
/// -------
/// LikelihoodTerm
///
#[cfg(feature = "python")]
#[pyfunction(name = "LikelihoodScalar")]
pub fn py_likelihood_scalar(name: String) -> PyLikelihoodTerm {
    PyLikelihoodTerm(LikelihoodScalar::new(name))
}
