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
    amplitudes::{Evaluator, Expression},
    data::Dataset,
    parameter_manager::ParameterManager,
    resources::Parameters,
    validate_free_parameter_len, LadduError, LadduResult,
};
use nalgebra::DVector;
use num::complex::Complex64;

#[cfg(feature = "mpi")]
use laddu_core::mpi::LadduMPI;

#[cfg(feature = "mpi")]
use mpi::{
    collective::SystemOperation, datatype::PartitionMut, topology::SimpleCommunicator, traits::*,
};
use parking_lot::Mutex;

#[cfg(feature = "python")]
use crate::ganesh_ext::{
    py_ganesh::{FromPyArgs, PyMCMCSummary, PyMinimizationSummary},
    MCMCSettings, MinimizationSettings,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyEvaluator, PyExpression},
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

fn validate_stochastic_batch_size(batch_size: usize, n_events: usize) -> LadduResult<()> {
    if n_events == 0 {
        return Err(LadduError::Custom(
            "stochastic batch_size requires a non-empty dataset".to_string(),
        ));
    }
    if batch_size == 0 || batch_size > n_events {
        return Err(LadduError::LengthMismatch {
            context: format!("stochastic batch_size (valid range: 1..={n_events})"),
            expected: n_events,
            actual: batch_size,
        });
    }
    Ok(())
}

#[cfg(feature = "python")]
fn validate_mcmc_parameter_len(walkers: &[Vec<f64>], expected_len: usize) -> LadduResult<()> {
    for walker in walkers {
        validate_free_parameter_len(walker.len(), expected_len)?;
    }
    Ok(())
}

#[cfg(feature = "mpi")]
fn reduce_scalar(world: &SimpleCommunicator, value: f64) -> f64 {
    let mut reduced = 0.0;
    world.all_reduce_into(&value, &mut reduced, &SystemOperation::sum());
    reduced
}

#[cfg(feature = "mpi")]
fn reduce_gradient(world: &SimpleCommunicator, gradient: &DVector<f64>) -> DVector<f64> {
    let mut reduced = vec![0.0; gradient.len()];
    world.all_reduce_into(gradient.as_slice(), &mut reduced, &SystemOperation::sum());
    DVector::from_vec(reduced)
}

/// A trait which describes a term that can be used like a likelihood (more correctly, a negative
/// log-likelihood) in a minimization.
pub trait LikelihoodTerm: DynClone + Send + Sync {
    /// Evaluate the term given some input parameters.
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64>;
    /// Evaluate the gradient of the term given some input parameters.
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>>;
    /// The list of names of the input parameters for [`LikelihoodTerm::evaluate`].
    fn parameters(&self) -> Vec<String>;
    /// Access the parameter manager describing free/fixed state.
    fn parameter_manager(&self) -> &ParameterManager;
    /// A method called every step of any minimization/MCMC algorithm.
    fn update(&self) {}

    /// Convenience helper to wrap a likelihood term into a [`LikelihoodExpression`].
    ///
    /// This allows term constructors to return expressions without exposing the manager
    /// machinery that previously performed registration.
    fn into_expression(self) -> LikelihoodExpression
    where
        Self: Sized + 'static,
    {
        LikelihoodExpression::from_term(Box::new(self))
    }
}

dyn_clone::clone_trait_object!(LikelihoodTerm);

/// An extended, unbinned negative log-likelihood evaluator.
#[derive(Clone)]
pub struct NLL {
    /// The internal [`Evaluator`] for data
    pub data_evaluator: Evaluator,
    /// The internal [`Evaluator`] for accepted Monte Carlo
    pub accmc_evaluator: Evaluator,
    parameter_manager: ParameterManager,
}

impl NLL {
    /// Construct an [`NLL`] from an [`Expression`] and two [`Dataset`]s (data and Monte Carlo). This mirrors loading a model but starts from
    /// the expression directly.
    pub fn new(
        expression: &Expression,
        ds_data: &Arc<Dataset>,
        ds_accmc: &Arc<Dataset>,
    ) -> LadduResult<Box<Self>> {
        let data_evaluator = expression.load(ds_data)?;
        let accmc_evaluator = expression.load(ds_accmc)?;
        Ok(Self {
            parameter_manager: data_evaluator.parameter_manager().clone(),
            data_evaluator,
            accmc_evaluator,
        }
        .into())
    }

    /// The parameter names for this NLL.
    pub fn parameters(&self) -> Vec<String> {
        self.parameter_manager.parameters()
    }

    /// The free parameter names for this NLL.
    pub fn free_parameters(&self) -> Vec<String> {
        self.parameter_manager.free_parameters()
    }

    /// The fixed parameter names for this NLL.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.parameter_manager.fixed_parameters()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.parameter_manager.n_free_parameters()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.parameter_manager.n_fixed_parameters()
    }

    /// Total number of parameters.
    pub fn n_parameters(&self) -> usize {
        self.parameter_manager.n_parameters()
    }

    /// Return a new [`NLL`] with the given parameter fixed to a value.
    pub fn fix(&self, name: &str, value: f64) -> LadduResult<Box<Self>> {
        let data_evaluator = self.data_evaluator.fix(name, value)?;
        let accmc_evaluator = self.accmc_evaluator.fix(name, value)?;
        Ok(Self {
            parameter_manager: self.parameter_manager.fix(name, value)?,
            data_evaluator,
            accmc_evaluator,
        }
        .into())
    }

    /// Return a new [`NLL`] with the given parameter freed.
    pub fn free(&self, name: &str) -> LadduResult<Box<Self>> {
        let data_evaluator = self.data_evaluator.free(name)?;
        let accmc_evaluator = self.accmc_evaluator.free(name)?;
        Ok(Self {
            parameter_manager: self.parameter_manager.free(name)?,
            data_evaluator,
            accmc_evaluator,
        }
        .into())
    }

    /// Return a new [`NLL`] with a single parameter renamed.
    pub fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<Box<Self>> {
        let data_evaluator = self.data_evaluator.rename_parameter(old, new)?;
        let accmc_evaluator = self.accmc_evaluator.rename_parameter(old, new)?;
        Ok(Self {
            parameter_manager: self.parameter_manager.rename(old, new)?,
            data_evaluator,
            accmc_evaluator,
        }
        .into())
    }

    /// Return a new [`NLL`] with several parameters renamed.
    pub fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<Box<Self>> {
        let data_evaluator = self.data_evaluator.rename_parameters(mapping)?;
        let accmc_evaluator = self.accmc_evaluator.rename_parameters(mapping)?;
        Ok(Self {
            parameter_manager: self.parameter_manager.rename_parameters(mapping)?,
            data_evaluator,
            accmc_evaluator,
        }
        .into())
    }
    /// Create a new [`StochasticNLL`] from this [`NLL`].
    pub fn to_stochastic(
        &self,
        batch_size: usize,
        seed: Option<usize>,
    ) -> LadduResult<StochasticNLL> {
        StochasticNLL::new(self.clone(), batch_size, seed)
    }
    /// Activate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name, skipping missing entries.
    pub fn activate<T: AsRef<str>>(&self, name: T) {
        self.data_evaluator.activate(&name);
        self.accmc_evaluator.activate(name);
    }
    /// Activate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name and return an error if it is missing.
    pub fn activate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.data_evaluator.activate_strict(&name)?;
        self.accmc_evaluator.activate_strict(name)?;
        Ok(())
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name, skipping missing entries.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.activate_many(names);
        self.accmc_evaluator.activate_many(names);
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name and return an error if any are missing.
    pub fn activate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.data_evaluator.activate_many_strict(names)?;
        self.accmc_evaluator.activate_many_strict(names)?;
        Ok(())
    }
    /// Activate all registered [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s.
    pub fn activate_all(&self) {
        self.data_evaluator.activate_all();
        self.accmc_evaluator.activate_all();
    }
    /// Dectivate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name, skipping missing entries.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.data_evaluator.deactivate(&name);
        self.accmc_evaluator.deactivate(name);
    }
    /// Dectivate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name and return an error if it is missing.
    pub fn deactivate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.data_evaluator.deactivate_strict(&name)?;
        self.accmc_evaluator.deactivate_strict(name)?;
        Ok(())
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name, skipping missing entries.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.deactivate_many(names);
        self.accmc_evaluator.deactivate_many(names);
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name and return an error if any are missing.
    pub fn deactivate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.data_evaluator.deactivate_many_strict(names)?;
        self.accmc_evaluator.deactivate_many_strict(names)?;
        Ok(())
    }
    /// Deactivate all registered [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s.
    pub fn deactivate_all(&self) {
        self.data_evaluator.deactivate_all();
        self.accmc_evaluator.deactivate_all();
    }
    /// Isolate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name (deactivate the rest), skipping missing entries.
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.data_evaluator.isolate(&name);
        self.accmc_evaluator.isolate(name);
    }
    /// Isolate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name (deactivate the rest) and return an error if it is missing.
    pub fn isolate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.data_evaluator.isolate_strict(&name)?;
        self.accmc_evaluator.isolate_strict(name)?;
        Ok(())
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name (deactivate the rest), skipping missing entries.
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.isolate_many(names);
        self.accmc_evaluator.isolate_many(names);
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name (deactivate the rest) and return an error if any are missing.
    pub fn isolate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.data_evaluator.isolate_many_strict(names)?;
        self.accmc_evaluator.isolate_many_strict(names)?;
        Ok(())
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each
    /// Monte-Carlo event (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project`] instead.
    pub fn project_local(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
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
            .zip(mc_dataset.events_local().par_iter())
            .map(|(l, e)| e.weight * l.re / n_mc)
            .collect();

        #[cfg(not(feature = "rayon"))]
        let output: Vec<f64> = result
            .iter()
            .zip(mc_dataset.events_local().iter())
            .map(|(l, e)| e.weight * l.re / n_mc)
            .collect();
        Ok(output)
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
    ) -> LadduResult<Vec<f64>> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projection = self.project_local(parameters, mc_evaluator)?;
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }
        Ok(buffer)
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
    pub fn project(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
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
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
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
            Ok((
                result
                    .par_iter()
                    .zip(mc_dataset.events_local().par_iter())
                    .map(|(l, e)| e.weight * l.re / n_mc)
                    .collect(),
                result_gradient
                    .par_iter()
                    .zip(mc_dataset.events_local().par_iter())
                    .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                    .collect(),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok((
                result
                    .iter()
                    .zip(mc_dataset.events_local().iter())
                    .map(|(l, e)| e.weight * l.re / n_mc)
                    .collect(),
                result_gradient
                    .iter()
                    .zip(mc_dataset.events_local().iter())
                    .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                    .collect(),
            ))
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
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let (local_projection, local_gradient_projection) =
            self.project_gradient_local(parameters, mc_evaluator)?;
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
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
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
    ) -> LadduResult<Vec<f64>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.resources.read().active.clone();
            mc_evaluator.isolate_many_strict(names)?;
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_local(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let output: Vec<f64> = result
                .par_iter()
                .zip(mc_dataset.events_local().par_iter())
                .map(|(l, e)| e.weight * l.re / n_mc)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<f64> = result
                .iter()
                .zip(mc_dataset.events_local().iter())
                .map(|(l, e)| e.weight * l.re / n_mc)
                .collect();
            mc_evaluator.resources.write().active = current_active_mc;
            Ok(output)
        } else {
            let current_active_data = self.data_evaluator.resources.read().active.clone();
            let current_active_accmc = self.accmc_evaluator.resources.read().active.clone();
            self.isolate_many_strict(names)?;
            let mc_dataset = &self.accmc_evaluator.dataset;
            let result = self.accmc_evaluator.evaluate_local(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let output: Vec<f64> = result
                .par_iter()
                .zip(mc_dataset.events_local().par_iter())
                .map(|(l, e)| e.weight * l.re / n_mc)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<f64> = result
                .iter()
                .zip(mc_dataset.events_local().iter())
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
    ) -> LadduResult<Vec<f64>> {
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
    ) -> LadduResult<Vec<f64>> {
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
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.resources.read().active.clone();
            mc_evaluator.isolate_many_strict(names)?;
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_local(parameters);
            let result_gradient = mc_evaluator.evaluate_gradient(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let (res, res_gradient) = {
                (
                    result
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                        .collect(),
                )
            };
            #[cfg(not(feature = "rayon"))]
            let (res, res_gradient) = {
                (
                    result
                        .iter()
                        .zip(mc_dataset.events_local().iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .iter()
                        .zip(mc_dataset.events_local().iter())
                        .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                        .collect(),
                )
            };
            mc_evaluator.resources.write().active = current_active_mc;
            Ok((res, res_gradient))
        } else {
            let current_active_data = self.data_evaluator.resources.read().active.clone();
            let current_active_accmc = self.accmc_evaluator.resources.read().active.clone();
            self.isolate_many_strict(names)?;
            let mc_dataset = &self.accmc_evaluator.dataset;
            let result = self.accmc_evaluator.evaluate_local(parameters);
            let result_gradient = self.accmc_evaluator.evaluate_gradient(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
            #[cfg(feature = "rayon")]
            let (res, res_gradient) = {
                (
                    result
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|(grad_l, e)| grad_l.map(|g| g.re).scale(e.weight / n_mc))
                        .collect(),
                )
            };
            #[cfg(not(feature = "rayon"))]
            let (res, res_gradient) = {
                (
                    result
                        .iter()
                        .zip(mc_dataset.events_local().iter())
                        .map(|(l, e)| e.weight * l.re / n_mc)
                        .collect(),
                    result_gradient
                        .iter()
                        .zip(mc_dataset.events_local().iter())
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
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
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
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_gradient_with_mpi(parameters, names, mc_evaluator, &world);
            }
        }
        self.project_gradient_with_local(parameters, names, mc_evaluator)
    }

    fn evaluate_terms_local(&self, parameters: &[f64]) -> (f64, f64) {
        let data_result = self.data_evaluator.evaluate_local(parameters);
        let mc_result = self.accmc_evaluator.evaluate_local(parameters);
        #[cfg(feature = "rayon")]
        let data_term: f64 = data_result
            .par_iter()
            .zip(self.data_evaluator.dataset.events_local().par_iter())
            .map(|(l, e)| e.weight * f64::ln(l.re))
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(feature = "rayon")]
        let mc_term: f64 = mc_result
            .par_iter()
            .zip(self.accmc_evaluator.dataset.events_local().par_iter())
            .map(|(l, e)| e.weight * l.re)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let data_term: f64 = data_result
            .iter()
            .zip(self.data_evaluator.dataset.events_local().iter())
            .map(|(l, e)| e.weight * f64::ln(l.re))
            .sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let mc_term: f64 = mc_result
            .iter()
            .zip(self.accmc_evaluator.dataset.events_local().iter())
            .map(|(l, e)| e.weight * l.re)
            .sum_with_accumulator::<Klein<f64>>();
        (data_term, mc_term)
    }

    fn evaluate_local(&self, parameters: &[f64]) -> f64 {
        let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
        let (data_term, mc_term) = self.evaluate_terms_local(parameters);
        -2.0 * (data_term - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_mpi(&self, parameters: &[f64], world: &SimpleCommunicator) -> f64 {
        let (data_term_local, mc_term_local) = self.evaluate_terms_local(parameters);
        let n_mc_local = self.accmc_evaluator.dataset.n_events_weighted_local();
        let data_term = reduce_scalar(world, data_term_local);
        let mc_term = reduce_scalar(world, mc_term_local);
        let n_mc = reduce_scalar(world, n_mc_local);
        -2.0 * (data_term - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    /// Evaluate the NLL value across all ranks.
    pub fn evaluate_mpi_value(&self, parameters: &[f64], world: &SimpleCommunicator) -> f64 {
        self.evaluate_mpi(parameters, world)
    }

    fn evaluate_gradient_terms_local(&self, parameters: &[f64]) -> (DVector<f64>, DVector<f64>) {
        let data_resources = self.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let mc_resources = self.accmc_evaluator.resources.read();
        let mc_parameters = Parameters::new(parameters, &mc_resources.constants);
        #[cfg(feature = "rayon")]
        let data_term: DVector<f64> = self
            .data_evaluator
            .dataset
            .events_local()
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
                            amp.compute_gradient(&data_parameters, cache, grad)
                        }
                    });
                (
                    event.weight,
                    self.data_evaluator
                        .amplitudes
                        .iter()
                        .zip(data_resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&data_parameters, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect::<Vec<_>>(),
                    gradient_values,
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.data_evaluator.evaluate_expression_value(&amp_vals),
                    self.data_evaluator
                        .evaluate_expression_gradient(&amp_vals, &grad_vals),
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
            .events_local()
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
                            amp.compute_gradient(&mc_parameters, cache, grad)
                        }
                    });
                (
                    event.weight,
                    self.accmc_evaluator
                        .amplitudes
                        .iter()
                        .zip(mc_resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&mc_parameters, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect::<Vec<_>>(),
                    gradient_values,
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.accmc_evaluator
                        .evaluate_expression_gradient(&amp_vals, &grad_vals),
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
            .events_local()
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
                            amp.compute_gradient(&data_parameters, cache, grad)
                        }
                    });
                (
                    event.weight,
                    self.data_evaluator
                        .amplitudes
                        .iter()
                        .zip(data_resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&data_parameters, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect::<Vec<_>>(),
                    gradient_values,
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.data_evaluator.evaluate_expression_value(&amp_vals),
                    self.data_evaluator
                        .evaluate_expression_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .sum();
        #[cfg(not(feature = "rayon"))]
        let mc_term: DVector<f64> = self
            .accmc_evaluator
            .dataset
            .events_local()
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
                            amp.compute_gradient(&mc_parameters, cache, grad)
                        }
                    });
                (
                    event.weight,
                    self.accmc_evaluator
                        .amplitudes
                        .iter()
                        .zip(mc_resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&mc_parameters, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect::<Vec<_>>(),
                    gradient_values,
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.accmc_evaluator
                        .evaluate_expression_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .sum();
        (data_term, mc_term)
    }

    fn evaluate_gradient_local(&self, parameters: &[f64]) -> DVector<f64> {
        let n_mc = self.accmc_evaluator.dataset.n_events_weighted();
        let (data_term, mc_term) = self.evaluate_gradient_terms_local(parameters);
        -2.0 * (data_term - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> DVector<f64> {
        let n_mc_local = self.accmc_evaluator.dataset.n_events_weighted_local();
        let (data_term_local, mc_term_local) = self.evaluate_gradient_terms_local(parameters);
        let data_term = reduce_gradient(world, &data_term_local);
        let mc_term = reduce_gradient(world, &mc_term_local);
        let n_mc = reduce_scalar(world, n_mc_local);
        -2.0 * (data_term - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    /// Evaluate the gradient across all ranks.
    pub fn evaluate_mpi_gradient(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> DVector<f64> {
        self.evaluate_gradient_mpi(parameters, world)
    }
}

impl LikelihoodTerm for NLL {
    /// Get the list of parameter names in the order they appear in the [`NLL::evaluate`]
    /// method.
    fn parameters(&self) -> Vec<String> {
        self.parameters()
    }
    fn parameter_manager(&self) -> &ParameterManager {
        &self.parameter_manager
    }
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return Ok(self.evaluate_mpi(parameters, &world));
            }
        }
        Ok(self.evaluate_local(parameters))
    }
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return Ok(self.evaluate_gradient_mpi(parameters, &world));
            }
        }
        Ok(self.evaluate_gradient_local(parameters))
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
    fn parameter_manager(&self) -> &ParameterManager {
        self.nll.parameter_manager()
    }
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        validate_free_parameter_len(parameters.len(), self.nll.n_free())?;
        let indices = self.batch_indices.lock();
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return Ok(self.evaluate_mpi(parameters, &indices, &world));
            }
        }
        #[cfg(feature = "rayon")]
        let n_data_batch_local = indices
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.events_local()[i].weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.events_local()[i].weight)
            .sum_with_accumulator::<Klein<f64>>();
        Ok(self.evaluate_local(parameters, &indices, n_data_batch_local))
    }
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        validate_free_parameter_len(parameters.len(), self.nll.n_free())?;
        let indices = self.batch_indices.lock();
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return Ok(self.evaluate_gradient_mpi(parameters, &indices, &world));
            }
        }
        #[cfg(feature = "rayon")]
        let n_data_batch_local = indices
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.events_local()[i].weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.events_local()[i].weight)
            .sum_with_accumulator::<Klein<f64>>();
        Ok(self.evaluate_gradient_local(parameters, &indices, n_data_batch_local))
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
    pub fn new(nll: NLL, batch_size: usize, seed: Option<usize>) -> LadduResult<Self> {
        let mut rng = seed.map_or_else(Rng::new, |seed| Rng::with_seed(seed as u64));
        let n = nll.data_evaluator.dataset.n_events();
        validate_stochastic_batch_size(batch_size, n)?;
        let batch_indices = rng.subset(batch_size, n);
        Ok(Self {
            nll,
            n,
            batch_size,
            batch_indices: Arc::new(Mutex::new(batch_indices)),
            rng: Arc::new(Mutex::new(rng)),
        })
    }
    /// Resample the batch indices used in evaluation
    pub fn resample(&self) {
        let mut rng = self.rng.lock();
        *self.batch_indices.lock() = rng.subset(self.batch_size, self.n);
    }

    /// The parameter names for this stochastic NLL.
    pub fn parameters(&self) -> Vec<String> {
        self.nll.parameters()
    }

    /// The free parameter names for this stochastic NLL.
    pub fn free_parameters(&self) -> Vec<String> {
        self.nll.free_parameters()
    }

    /// The fixed parameter names for this stochastic NLL.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.nll.fixed_parameters()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.nll.n_free()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.nll.n_fixed()
    }

    /// Total number of parameters.
    pub fn n_parameters(&self) -> usize {
        self.nll.n_parameters()
    }
    #[cfg(feature = "mpi")]
    fn data_batch_weight_local(&self, indices: &[usize]) -> f64 {
        #[cfg(feature = "rayon")]
        return indices
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.events_local()[i].weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        return indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.events_local()[i].weight)
            .sum_with_accumulator::<Klein<f64>>();
    }

    fn evaluate_terms_local(&self, parameters: &[f64], indices: &[usize]) -> (f64, f64) {
        let data_result = self
            .nll
            .data_evaluator
            .evaluate_batch_local(parameters, indices);
        let mc_result = self.nll.accmc_evaluator.evaluate_local(parameters);
        #[cfg(feature = "rayon")]
        {
            let data_term: f64 = indices
                .par_iter()
                .zip(data_result.par_iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.events_local()[i];
                    e.weight * l.re.ln()
                })
                .parallel_sum_with_accumulator::<Klein<f64>>();
            let mc_term: f64 = mc_result
                .par_iter()
                .zip(self.nll.accmc_evaluator.dataset.events_local().par_iter())
                .map(|(l, e)| e.weight * l.re)
                .parallel_sum_with_accumulator::<Klein<f64>>();
            (data_term, mc_term)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let data_term: f64 = indices
                .iter()
                .zip(data_result.iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.events_local()[i];
                    e.weight * l.re.ln()
                })
                .sum_with_accumulator::<Klein<f64>>();
            let mc_term: f64 = mc_result
                .iter()
                .zip(self.nll.accmc_evaluator.dataset.events_local().iter())
                .map(|(l, e)| e.weight * l.re)
                .sum_with_accumulator::<Klein<f64>>();
            (data_term, mc_term)
        }
    }

    fn evaluate_local(&self, parameters: &[f64], indices: &[usize], n_data_batch: f64) -> f64 {
        let n_mc = self.nll.accmc_evaluator.dataset.n_events_weighted_local();
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted_local();
        let (data_term, mc_term) = self.evaluate_terms_local(parameters, indices);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> f64 {
        let total = self.nll.data_evaluator.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let n_data_batch_local = self.data_batch_weight_local(&locals);
        let n_data_total_local = self.nll.data_evaluator.dataset.n_events_weighted_local();
        let n_mc_local = self.nll.accmc_evaluator.dataset.n_events_weighted_local();
        let (data_term_local, mc_term_local) = self.evaluate_terms_local(parameters, &locals);
        let n_data_batch = reduce_scalar(world, n_data_batch_local);
        let n_data_total = reduce_scalar(world, n_data_total_local);
        let n_mc = reduce_scalar(world, n_mc_local);
        let data_term = reduce_scalar(world, data_term_local);
        let mc_term = reduce_scalar(world, mc_term_local);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
    }

    fn evaluate_gradient_terms_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> (DVector<f64>, DVector<f64>) {
        let data_resources = self.nll.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let mc_resources = self.nll.accmc_evaluator.resources.read();
        let mc_parameters = Parameters::new(parameters, &mc_resources.constants);
        #[cfg(feature = "rayon")]
        let data_term: DVector<f64> = indices
            .par_iter()
            .map(|&idx| {
                let event = &self.nll.data_evaluator.dataset.events_local()[idx];
                let cache = &data_resources.caches[idx];
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
                            amp.compute_gradient(&data_parameters, cache, grad)
                        }
                    });
                let amp_vals: Vec<_> = self
                    .nll
                    .data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .map(|(amp, active)| {
                        if *active {
                            amp.compute(&data_parameters, cache)
                        } else {
                            Complex64::ZERO
                        }
                    })
                    .collect();
                (
                    event.weight,
                    self.nll.data_evaluator.evaluate_expression_value(&amp_vals),
                    self.nll
                        .data_evaluator
                        .evaluate_expression_gradient(&amp_vals, &gradient_values),
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
            .events_local()
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
                            amp.compute_gradient(&mc_parameters, cache, grad)
                        }
                    });
                (
                    event.weight,
                    self.nll
                        .accmc_evaluator
                        .amplitudes
                        .iter()
                        .zip(mc_resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&mc_parameters, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect::<Vec<_>>(),
                    gradient_values,
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.nll
                        .accmc_evaluator
                        .evaluate_expression_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .collect::<Vec<DVector<f64>>>()
            .iter()
            .sum();
        #[cfg(not(feature = "rayon"))]
        let data_term: DVector<f64> = indices
            .iter()
            .map(|&idx| {
                let event = &self.nll.data_evaluator.dataset.events_local()[idx];
                let cache = &data_resources.caches[idx];
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
                            amp.compute_gradient(&data_parameters, cache, grad)
                        }
                    });
                let amp_vals: Vec<_> = self
                    .nll
                    .data_evaluator
                    .amplitudes
                    .iter()
                    .zip(data_resources.active.iter())
                    .map(|(amp, active)| {
                        if *active {
                            amp.compute(&data_parameters, cache)
                        } else {
                            Complex64::ZERO
                        }
                    })
                    .collect();
                (
                    event.weight,
                    self.nll.data_evaluator.evaluate_expression_value(&amp_vals),
                    self.nll
                        .data_evaluator
                        .evaluate_expression_gradient(&amp_vals, &gradient_values),
                )
            })
            .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
            .sum();
        #[cfg(not(feature = "rayon"))]
        let mc_term: DVector<f64> = self
            .nll
            .accmc_evaluator
            .dataset
            .events_local()
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
                            amp.compute_gradient(&mc_parameters, cache, grad)
                        }
                    });
                (
                    event.weight,
                    self.nll
                        .accmc_evaluator
                        .amplitudes
                        .iter()
                        .zip(mc_resources.active.iter())
                        .map(|(amp, active)| {
                            if *active {
                                amp.compute(&mc_parameters, cache)
                            } else {
                                Complex64::ZERO
                            }
                        })
                        .collect::<Vec<_>>(),
                    gradient_values,
                )
            })
            .map(|(weight, amp_vals, grad_vals)| {
                (
                    weight,
                    self.nll
                        .accmc_evaluator
                        .evaluate_expression_gradient(&amp_vals, &grad_vals),
                )
            })
            .map(|(w, g)| w * g.map(|gi| gi.re))
            .sum();
        (data_term, mc_term)
    }

    fn evaluate_gradient_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
        n_data_batch: f64,
    ) -> DVector<f64> {
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted_local();
        let n_mc = self.nll.accmc_evaluator.dataset.n_events_weighted_local();
        let (data_term, mc_term) = self.evaluate_gradient_terms_local(parameters, indices);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> DVector<f64> {
        let total = self.nll.data_evaluator.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let n_data_batch_local = self.data_batch_weight_local(&locals);
        let n_data_total_local = self.nll.data_evaluator.dataset.n_events_weighted_local();
        let n_mc_local = self.nll.accmc_evaluator.dataset.n_events_weighted_local();
        let (data_term_local, mc_term_local) =
            self.evaluate_gradient_terms_local(parameters, &locals);
        let n_data_batch = reduce_scalar(world, n_data_batch_local);
        let n_data_total = reduce_scalar(world, n_data_total_local);
        let n_mc = reduce_scalar(world, n_mc_local);
        let data_term = reduce_gradient(world, &data_term_local);
        let mc_term = reduce_gradient(world, &mc_term_local);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
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
    #[pyo3(signature = (expression, ds_data, ds_accmc))]
    fn new(expression: &PyExpression, ds_data: &PyDataset, ds_accmc: &PyDataset) -> PyResult<Self> {
        Ok(Self(NLL::new(&expression.0, &ds_data.0, &ds_accmc.0)?))
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
    fn to_stochastic(&self, batch_size: usize, seed: Option<usize>) -> PyResult<PyStochasticNLL> {
        Ok(PyStochasticNLL(self.0.to_stochastic(batch_size, seed)?))
    }
    /// Turn an ``NLL`` into a likelihood expression that can be combined with other terms.
    fn to_expression(&self) -> PyLikelihoodExpression {
        PyLikelihoodExpression(self.0.clone().into_expression())
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
    /// The free parameters used by the NLL
    #[getter]
    fn free_parameters(&self) -> Vec<String> {
        self.0.free_parameters()
    }
    /// The fixed parameters used by the NLL
    #[getter]
    fn fixed_parameters(&self) -> Vec<String> {
        self.0.fixed_parameters()
    }
    /// Number of free parameters
    #[getter]
    fn n_free(&self) -> usize {
        self.0.n_free()
    }
    /// Number of fixed parameters
    #[getter]
    fn n_fixed(&self) -> usize {
        self.0.n_fixed()
    }
    /// Total number of parameters
    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }
    /// Return a new NLL with the given parameter fixed
    fn fix(&self, name: &str, value: f64) -> PyResult<PyNLL> {
        Ok(PyNLL(self.0.fix(name, value)?))
    }
    /// Return a new NLL with the given parameter freed
    fn free(&self, name: &str) -> PyResult<PyNLL> {
        Ok(PyNLL(self.0.free(name)?))
    }
    /// Return a new NLL with a single parameter renamed
    fn rename_parameter(&self, old: &str, new: &str) -> PyResult<PyNLL> {
        Ok(PyNLL(self.0.rename_parameter(old, new)?))
    }
    /// Return a new NLL with several parameters renamed
    fn rename_parameters(&self, mapping: HashMap<String, String>) -> PyResult<PyNLL> {
        Ok(PyNLL(self.0.rename_parameters(&mapping)?))
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
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any amplitude is missing. When ``False``,
    ///     silently skip missing amplitudes.
    #[pyo3(signature = (arg, *, strict=true))]
    fn activate(&self, arg: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            if strict {
                self.0.activate_strict(&string_arg)?;
            } else {
                self.0.activate(&string_arg);
            }
        } else if let Ok(list_arg) = arg.cast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            if strict {
                self.0.activate_many_strict(&vec)?;
            } else {
                self.0.activate_many(&vec);
            }
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
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any amplitude is missing. When ``False``,
    ///     silently skip missing amplitudes.
    #[pyo3(signature = (arg, *, strict=true))]
    fn deactivate(&self, arg: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            if strict {
                self.0.deactivate_strict(&string_arg)?;
            } else {
                self.0.deactivate(&string_arg);
            }
        } else if let Ok(list_arg) = arg.cast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            if strict {
                self.0.deactivate_many_strict(&vec)?;
            } else {
                self.0.deactivate_many(&vec);
            }
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
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any amplitude is missing. When ``False``,
    ///     silently skip missing amplitudes.
    #[pyo3(signature = (arg, *, strict=true))]
    fn isolate(&self, arg: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            if strict {
                self.0.isolate_strict(&string_arg)?;
            } else {
                self.0.isolate(&string_arg);
            }
        } else if let Ok(list_arg) = arg.cast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            if strict {
                self.0.isolate_many_strict(&vec)?;
            } else {
                self.0.isolate_many(&vec);
            }
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
    #[cfg_attr(not(feature = "rayon"), allow(unused_variables))]
    fn evaluate(&self, parameters: Vec<f64>, threads: Option<usize>) -> PyResult<f64> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        #[cfg(feature = "rayon")]
        {
            ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| LikelihoodTerm::evaluate(self.0.as_ref(), &parameters))
                .map_err(PyErr::from)
        }
        #[cfg(not(feature = "rayon"))]
        {
            LikelihoodTerm::evaluate(self.0.as_ref(), &parameters).map_err(PyErr::from)
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
    #[cfg_attr(not(feature = "rayon"), allow(unused_variables))]
    fn evaluate_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        #[cfg(feature = "rayon")]
        {
            let gradient = ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| LikelihoodTerm::evaluate_gradient(self.0.as_ref(), &parameters))
                .map_err(PyErr::from)?;
            Ok(PyArray1::from_slice(py, gradient.as_slice()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            let gradient = LikelihoodTerm::evaluate_gradient(self.0.as_ref(), &parameters)?;
            Ok(PyArray1::from_slice(py, gradient.as_slice()))
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
    #[cfg_attr(not(feature = "rayon"), allow(unused_variables))]
    fn project<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        #[cfg(feature = "rayon")]
        {
            let projection = ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| {
                    self.0
                        .project(&parameters, mc_evaluator.map(|pyeval| pyeval.0.clone()))
                })
                .map_err(PyErr::from)?;
            Ok(PyArray1::from_slice(py, projection.as_slice()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            let projection = self
                .0
                .project(&parameters, mc_evaluator.map(|pyeval| pyeval.0.clone()))?;
            Ok(PyArray1::from_slice(py, projection.as_slice()))
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
    #[cfg_attr(not(feature = "rayon"), allow(unused_variables))]
    fn project_with<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        arg: &Bound<'_, PyAny>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        let names = if let Ok(string_arg) = arg.extract::<String>() {
            vec![string_arg]
        } else if let Ok(list_arg) = arg.cast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            vec
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        };
        #[cfg(feature = "rayon")]
        {
            let projection = ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| {
                    self.0.project_with(
                        &parameters,
                        &names,
                        mc_evaluator.map(|pyeval| pyeval.0.clone()),
                    )
                })
                .map_err(PyErr::from)?;
            Ok(PyArray1::from_slice(py, projection.as_slice()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            let projection = self.0.project_with(
                &parameters,
                &names,
                mc_evaluator.map(|pyeval| pyeval.0.clone()),
            )?;
            Ok(PyArray1::from_slice(py, projection.as_slice()))
        }
    }

    #[cfg_attr(doctest, doc = "```ignore")]
    /// Minimize the NLL with respect to the free parameters in the model
    ///
    /// This method "runs the fit". Given an initial position `p0`, this
    /// method performs a minimization over the negative log-likelihood, optimizing the model
    /// over the stored signal data and Monte Carlo.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial values for the free parameters (length ``n_free``)
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
    ///
    /// Notes
    /// -----
    /// The `settings` dict is forwarded to the solver as keyword arguments. Each algorithm
    /// recognises the following keys:
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
    ///     Use adaptive hyperparameters according to Gao and Han (2010).
    /// expansion_method : {"greedyminimization", "greedyexpansion"}
    ///     Greedy minimization will favor points which minimize faster, but greedy expansion may explore a space more efficiently. See Lagarias et al. (1998) for details.
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
    ///     Set the method to terminate the algorithm based on the function values over the simplex. See Singer & Singer (2004) for details. Set to None to skip this check.
    /// eps_f : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the f_terminator method.
    /// x_terminator : {"singer", "diameter", "higham", "rowan"} or None, default="singer"
    ///     Set the method to terminate the algorithm based on the position of simplex vertices. See Singer & Singer (2004) for details. Set to None to skip this check.
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
    /// References
    /// ----------
    /// Gao, F. & Han, L. (2010). *Implementing the Nelder-Mead simplex algorithm with adaptive
    /// parameters*. Comput. Optim. Appl. 51(1), 259–277. <https://doi.org/10.1007/s10589-010-9329-3>
    ///
    /// Lagarias, J. C., Reeds, J. A., Wright, M. H., & Wright, P. E. (1998). *Convergence Properties
    /// of the Nelder–Mead Simplex Method in Low Dimensions*. SIAM J. Optim. 9(1), 112–147.
    /// <https://doi.org/10.1137/S1052623496303470>
    ///
    /// Singer, S. & Singer, S. (2004). *Efficient Implementation of the Nelder–Mead Search Algorithm*.
    /// Appl. Numer. Anal. & Comput. 1(2), 524–534. <https://doi.org/10.1002/anac.200410015>
    ///
    #[cfg_attr(doctest, doc = "```")]
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
        validate_free_parameter_len(p0.len(), self.0.n_free())?;
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
    /// See Also
    /// --------
    /// NLL.minimize
    /// StochasticNLL.mcmc
    ///
    /// Examples
    /// --------
    /// >>> nll.mcmc([[0.0, 0.5]], method='ess', max_steps=512)  # doctest: +SKIP
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the MCMC algorithm as keyword arguments. Each
    /// algorithm exposes different keys:
    ///
    /// AIES (Affine-Invariant Ensemble Sampler)
    /// ========================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('stretch', {'a': 2.0}, 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('stretch' or 'walk') and the last is the frequency that move should be used relative
    ///     to the sum of all frequencies given across all moves. An optional middle dictionary
    ///     parameter may be provided to specify properties of moves which support it. For the AIES
    ///     algorithm, the stretch move may use the 'a' parameter to specify the scaling parameter
    ///     (2.0 by default). See Goodman & Weare (2010).
    ///
    /// ESS (Ensemble Slice Sampler)
    /// ============================
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
    ///     See Karamanis & Beutler (2021).
    /// n_adaptive : int, default=0
    ///     The number of adaptive moves to perform at the start of sampling
    /// max_steps : int, default=10000
    ///     The maximum number of expansions/contractions to perform at each step in the algorithm
    /// mu : float, default = 1.0
    ///     The adaptive scaling parameter (only applies if 'n_adaptive' is greater than zero)
    ///
    /// References
    /// ----------
    /// Goodman, J. & Weare, J. (2010). *Ensemble samplers with affine invariance*. CAMCoS 5(1), 65–80. <https://doi.org/10.2140/camcos.2010.5.65>
    ///
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat Comput 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
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
        validate_mcmc_parameter_len(&p0, self.0.n_free())?;
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
    #[cfg_attr(doctest, doc = "```ignore")]
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
    ///
    /// Notes
    /// -----
    /// The `settings` dict is forwarded to the solver as keyword arguments. Each algorithm
    /// recognises the following keys:
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
    ///     Use adaptive hyperparameters according to Gao and Han (2010).
    /// expansion_method : {"greedyminimization", "greedyexpansion"}
    ///     Greedy minimization will favor points which minimize faster, but greedy expansion may explore a space more efficiently. See Lagarias et al. (1998) for details.
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
    ///     Set the method to terminate the algorithm based on the function values over the simplex. See Singer & Singer (2004) for details. Set to None to skip this check.
    /// eps_f : float, default=`MACH_EPS^(1/4)`
    ///     The tolerance for the f_terminator method.
    /// x_terminator : {"singer", "diameter", "higham", "rowan"} or None, default="singer"
    ///     Set the method to terminate the algorithm based on the position of simplex vertices. See Singer & Singer (2004) for details. Set to None to skip this check.
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
    /// References
    /// ----------
    /// Gao, F. & Han, L. (2010). *Implementing the Nelder-Mead simplex algorithm with adaptive
    /// parameters*. Comput. Optim. Appl. 51(1), 259–277. <https://doi.org/10.1007/s10589-010-9329-3>
    ///
    /// Lagarias, J. C., Reeds, J. A., Wright, M. H., & Wright, P. E. (1998). *Convergence Properties
    /// of the Nelder–Mead Simplex Method in Low Dimensions*. SIAM J. Optim. 9(1), 112–147.
    /// <https://doi.org/10.1137/S1052623496303470>
    ///
    /// Singer, S. & Singer, S. (2004). *Efficient Implementation of the Nelder–Mead Search Algorithm*.
    /// Appl. Numer. Anal. & Comput. 1(2), 524–534. <https://doi.org/10.1002/anac.200410015>
    ///
    #[cfg_attr(doctest, doc = "```")]
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
        validate_free_parameter_len(p0.len(), self.0.n_free())?;
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
    /// See Also
    /// --------
    /// StochasticNLL.minimize
    /// NLL.mcmc
    ///
    /// Examples
    /// --------
    /// >>> s_nll = nll.to_stochastic(batch_size=2048, seed=1234)  # doctest: +SKIP
    /// >>> s_nll.mcmc([[0.0, 0.5]], max_steps=1024)  # doctest: +SKIP
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the MCMC algorithm as keyword arguments. Each
    /// algorithm exposes different keys:
    ///
    /// AIES (Affine-Invariant Ensemble Sampler)
    /// ========================================
    /// moves : list of tuple of (str, float) or (str, dict, float), default = [('stretch', {'a': 2.0}, 1.0)]
    ///     The list of moves to use. The first element of the tuple is the name of the move
    ///     ('stretch' or 'walk') and the last is the frequency that move should be used relative
    ///     to the sum of all frequencies given across all moves. An optional middle dictionary
    ///     parameter may be provided to specify properties of moves which support it. For the AIES
    ///     algorithm, the stretch move may use the 'a' parameter to specify the scaling parameter
    ///     (2.0 by default). See Goodman & Weare (2010).
    ///
    /// ESS (Ensemble Slice Sampler)
    /// ============================
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
    ///     See Karamanis & Beutler (2021).
    /// n_adaptive : int, default=0
    ///     The number of adaptive moves to perform at the start of sampling
    /// max_steps : int, default=10000
    ///     The maximum number of expansions/contractions to perform at each step in the algorithm
    /// mu : float, default = 1.0
    ///     The adaptive scaling parameter (only applies if 'n_adaptive' is greater than zero)
    ///
    /// References
    /// ----------
    /// Goodman, J. & Weare, J. (2010). *Ensemble samplers with affine invariance*. CAMCoS 5(1), 65–80. <https://doi.org/10.2140/camcos.2010.5.65>
    ///
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat Comput 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
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
        validate_mcmc_parameter_len(&p0, self.0.n_free())?;
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

#[derive(Debug)]
struct LikelihoodValues(Vec<f64>);

#[derive(Debug)]
struct LikelihoodGradients(Vec<DVector<f64>>);

#[derive(Clone, Default)]
enum LikelihoodNode {
    #[default]
    Zero,
    One,
    Term(usize),
    Add(Box<LikelihoodNode>, Box<LikelihoodNode>),
    Mul(Box<LikelihoodNode>, Box<LikelihoodNode>),
}

impl LikelihoodNode {
    fn remap(&self, mapping: &[usize]) -> Self {
        match self {
            Self::Term(idx) => Self::Term(mapping[*idx]),
            Self::Add(a, b) => Self::Add(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Mul(a, b) => Self::Mul(Box::new(a.remap(mapping)), Box::new(b.remap(mapping))),
            Self::Zero => Self::Zero,
            Self::One => Self::One,
        }
    }

    fn evaluate(&self, likelihood_values: &LikelihoodValues) -> f64 {
        match self {
            LikelihoodNode::Zero => 0.0,
            LikelihoodNode::One => 1.0,
            LikelihoodNode::Term(idx) => likelihood_values.0[*idx],
            LikelihoodNode::Add(a, b) => {
                a.evaluate(likelihood_values) + b.evaluate(likelihood_values)
            }
            LikelihoodNode::Mul(a, b) => {
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
            LikelihoodNode::Zero => DVector::zeros(0),
            LikelihoodNode::One => DVector::zeros(0),
            LikelihoodNode::Term(idx) => likelihood_gradients.0[*idx].clone(),
            LikelihoodNode::Add(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
            }
            LikelihoodNode::Mul(a, b) => {
                a.evaluate_gradient(likelihood_values, likelihood_gradients)
                    * b.evaluate(likelihood_values)
                    + b.evaluate_gradient(likelihood_values, likelihood_gradients)
                        * a.evaluate(likelihood_values)
            }
        }
    }

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
            Self::Term(idx) => format!("term({idx})"),
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
                        Some(_) => child.write_tree(f, &child_prefix, "├─ ", "│  ")?,
                        None => child.write_tree(f, &child_prefix, "└─ ", "   ")?,
                    }
                }
            }
        }
        Ok(())
    }
}

/// A combination of [`LikelihoodTerm`]s as well as sums and products of them.
///
/// # Notes
/// When multiple terms provide parameters with the same name, the term earliest in the expression
/// (or argument list) defines the fixed/free status and default value.
#[derive(Clone, Default)]
pub struct LikelihoodExpression {
    registry: LikelihoodRegistry,
    tree: LikelihoodNode,
    parameter_manager: ParameterManager,
}

impl LikelihoodExpression {
    /// Build a [`LikelihoodExpression`] from a single [`LikelihoodTerm`].
    pub fn from_term(term: Box<dyn LikelihoodTerm>) -> Self {
        let registry = LikelihoodRegistry::singleton(term);
        let parameter_manager = registry.parameter_manager.clone();
        Self {
            registry,
            tree: LikelihoodNode::Term(0),
            parameter_manager,
        }
    }

    /// Create an expression representing zero, the additive identity.
    pub fn zero() -> Self {
        Self {
            registry: LikelihoodRegistry::default(),
            tree: LikelihoodNode::Zero,
            parameter_manager: ParameterManager::default(),
        }
    }

    /// Create an expression representing one, the multiplicative identity.
    pub fn one() -> Self {
        Self {
            registry: LikelihoodRegistry::default(),
            tree: LikelihoodNode::One,
            parameter_manager: ParameterManager::default(),
        }
    }

    fn binary_op(
        a: &LikelihoodExpression,
        b: &LikelihoodExpression,
        build: impl Fn(Box<LikelihoodNode>, Box<LikelihoodNode>) -> LikelihoodNode,
    ) -> LikelihoodExpression {
        let (registry, left_map, right_map) = a.registry.merge(&b.registry);
        let left_tree = a.tree.remap(&left_map);
        let right_tree = b.tree.remap(&right_map);
        let (parameter_manager, _, _) = a.parameter_manager.merge(&b.parameter_manager);
        LikelihoodExpression {
            registry,
            tree: build(Box::new(left_tree), Box::new(right_tree)),
            parameter_manager,
        }
    }

    fn write_tree(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        parent_prefix: &str,
        immediate_prefix: &str,
        parent_suffix: &str,
    ) -> std::fmt::Result {
        self.tree
            .write_tree(f, parent_prefix, immediate_prefix, parent_suffix)
    }

    /// The parameter names referenced across all terms in this expression.
    pub fn parameters(&self) -> Vec<String> {
        self.parameter_manager.parameters()
    }

    /// The free parameter names which require user-provided values.
    pub fn free_parameters(&self) -> Vec<String> {
        self.parameter_manager.free_parameters()
    }

    /// The names of parameters with constant (fixed) values.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.parameter_manager.fixed_parameters()
    }

    fn assert_parameter_exists(&self, name: &str) -> LadduResult<()> {
        if self.parameter_manager.contains(name) {
            Ok(())
        } else {
            Err(LadduError::UnregisteredParameter {
                name: name.to_string(),
                reason: "parameter not found".into(),
            })
        }
    }

    /// Return a new [`LikelihoodExpression`] with the given parameter fixed to a value.
    pub fn fix(&self, name: &str, value: f64) -> LadduResult<Self> {
        self.assert_parameter_exists(name)?;
        let mut output = self.clone();
        output.parameter_manager = self.parameter_manager.fix(name, value)?;
        Ok(output)
    }

    /// Return a new [`LikelihoodExpression`] with the given parameter freed.
    pub fn free(&self, name: &str) -> LadduResult<Self> {
        self.assert_parameter_exists(name)?;
        let mut output = self.clone();
        output.parameter_manager = self.parameter_manager.free(name)?;
        Ok(output)
    }

    /// Return a new [`LikelihoodExpression`] with the given parameter renamed.
    pub fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<Self> {
        self.assert_parameter_exists(old)?;
        if old == new {
            return Ok(self.clone());
        }
        let mut output = self.clone();
        output.parameter_manager = self.parameter_manager.rename(old, new)?;
        Ok(output)
    }

    /// Return a new [`LikelihoodExpression`] with several parameters renamed.
    pub fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<Self> {
        for old in mapping.keys() {
            self.assert_parameter_exists(old)?;
        }
        let mut output = self.clone();
        output.parameter_manager = self.parameter_manager.rename_parameters(mapping)?;
        Ok(output)
    }

    /// Load a `LikelihoodExpression` so it can be evaluated repeatedly.
    pub fn load(&self) -> LikelihoodEvaluator {
        let parameter_manager = self.parameter_manager.clone();
        let free_parameter_indices = parameter_manager.free_parameter_indices();
        LikelihoodEvaluator {
            likelihood_registry: self.registry.clone(),
            likelihood_expression: self.tree.clone(),
            free_parameter_indices,
            parameter_manager,
        }
    }
}

impl Debug for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(f, "", "", "")
    }
}

impl Display for LikelihoodExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_tree(f, "", "", "")
    }
}

impl_op_ex!(+ |a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression {
    LikelihoodExpression::binary_op(a, b, LikelihoodNode::Add)
});
impl_op_ex!(
    *|a: &LikelihoodExpression, b: &LikelihoodExpression| -> LikelihoodExpression {
        LikelihoodExpression::binary_op(a, b, LikelihoodNode::Mul)
    }
);

#[derive(Clone, Default)]
struct LikelihoodRegistry {
    terms: Vec<Box<dyn LikelihoodTerm>>,
    parameter_manager: ParameterManager,
    param_layouts: Vec<Vec<usize>>,
    param_counts: Vec<usize>,
}

impl LikelihoodRegistry {
    fn singleton(term: Box<dyn LikelihoodTerm>) -> Self {
        let mut registry = Self::default();
        registry.push_term(term);
        registry
    }

    fn push_term(&mut self, term: Box<dyn LikelihoodTerm>) -> usize {
        let term_idx = self.terms.len();
        let term_manager = term.parameter_manager().clone();
        let (merged_manager, _left_map, layout) = self.parameter_manager.merge(&term_manager);
        self.parameter_manager = merged_manager;
        let param_layout = layout;
        self.param_layouts.push(param_layout);
        self.param_counts.push(term_manager.n_parameters());
        self.terms.push(term);
        term_idx
    }

    fn merge(&self, other: &Self) -> (Self, Vec<usize>, Vec<usize>) {
        let mut registry = Self::default();
        let mut left_map = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            let idx = registry.push_term(dyn_clone::clone_box(&**term));
            left_map.push(idx);
        }
        let mut right_map = Vec::with_capacity(other.terms.len());
        for term in &other.terms {
            let idx = registry.push_term(dyn_clone::clone_box(&**term));
            right_map.push(idx);
        }
        (registry, left_map, right_map)
    }
}

///
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodExpression", module = "laddu")]
#[derive(Clone)]
pub struct PyLikelihoodExpression(pub LikelihoodExpression);

/// A convenience method to sum sequences of [`LikelihoodExpression`]s or identifiers.
///
/// Parameters
/// ----------
/// terms : sequence of LikelihoodExpression
///     A non-empty sequence whose elements are summed. Single-element sequences are returned
///     unchanged while empty sequences evaluate to [`LikelihoodZero`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     A new expression representing the sum of all inputs.
///
/// See Also
/// --------
/// likelihood_product
/// LikelihoodZero
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodScalar, likelihood_sum
/// >>> expression = likelihood_sum([LikelihoodScalar('alpha')])
/// >>> expression.load().evaluate([0.5])
/// 0.5
/// >>> likelihood_sum([]).load().evaluate([])
/// 0.0
///
/// Notes
/// -----
/// When multiple inputs share the same parameter name, the value and fixed/free status from the
/// earliest term in the sequence take precedence.
#[cfg(feature = "python")]
#[pyfunction(name = "likelihood_sum")]
pub fn py_likelihood_sum(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyLikelihoodExpression> {
    if terms.is_empty() {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::zero()));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyLikelihoodExpression>() {
            return Ok(expression);
        }
        return Err(PyTypeError::new_err("Item is not a PyLikelihoodExpression"));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::zero()));
    };
    let PyLikelihoodExpression(mut summation) = first_term
        .extract::<PyLikelihoodExpression>()
        .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
    for term in iter {
        let PyLikelihoodExpression(expr) = term
            .extract::<PyLikelihoodExpression>()
            .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
        summation = summation + expr;
    }
    Ok(PyLikelihoodExpression(summation))
}

/// A convenience method to multiply sequences of [`LikelihoodExpression`]s.
///
/// Parameters
/// ----------
/// terms : sequence of LikelihoodExpression
///     A non-empty sequence whose elements are multiplied. Single-element sequences are returned
///     unchanged while empty sequences evaluate to [`LikelihoodOne`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     A new expression representing the product of all inputs.
///
/// See Also
/// --------
/// likelihood_sum
/// LikelihoodOne
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodScalar, likelihood_product
/// >>> evaluator = likelihood_product([LikelihoodScalar('alpha'), LikelihoodScalar('beta')]).load()
/// >>> evaluator.parameters
/// ['alpha', 'beta']
/// >>> evaluator.evaluate([2.0, 3.0])
/// 6.0
///
/// Notes
/// -----
/// When parameters overlap between inputs, the parameter definition from the earliest term is used.
#[cfg(feature = "python")]
#[pyfunction(name = "likelihood_product")]
pub fn py_likelihood_product(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyLikelihoodExpression> {
    if terms.is_empty() {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::one()));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyLikelihoodExpression>() {
            return Ok(expression);
        }
        return Err(PyTypeError::new_err("Item is not a PyLikelihoodExpression"));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::one()));
    };
    let PyLikelihoodExpression(mut product) = first_term
        .extract::<PyLikelihoodExpression>()
        .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
    for term in iter {
        let PyLikelihoodExpression(expr) = term
            .extract::<PyLikelihoodExpression>()
            .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
        product = product * expr;
    }
    Ok(PyLikelihoodExpression(product))
}

/// A convenience constructor for a zero-valued [`LikelihoodExpression`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     An expression that evaluates to ``0`` for any parameter values.
///
/// See Also
/// --------
/// LikelihoodOne
/// likelihood_sum
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodZero
/// >>> evaluator = LikelihoodZero().load()
/// >>> evaluator.parameters
/// []
/// >>> evaluator.evaluate([])
/// 0.0
#[cfg(feature = "python")]
#[pyfunction(name = "LikelihoodZero")]
pub fn py_likelihood_zero() -> PyLikelihoodExpression {
    PyLikelihoodExpression(LikelihoodExpression::zero())
}

/// A convenience constructor for a unit-valued [`LikelihoodExpression`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     An expression that evaluates to ``1`` for any parameter values.
///
/// See Also
/// --------
/// LikelihoodZero
/// likelihood_product
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodOne
/// >>> LikelihoodOne().load().evaluate([])
/// 1.0
#[cfg(feature = "python")]
#[pyfunction(name = "LikelihoodOne")]
pub fn py_likelihood_one() -> PyLikelihoodExpression {
    PyLikelihoodExpression(LikelihoodExpression::one())
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLikelihoodExpression {
    /// All parameter names referenced by the expression.
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }

    /// The free parameter names (those requiring optimization inputs).
    #[getter]
    fn free_parameters(&self) -> Vec<String> {
        self.0.free_parameters()
    }

    /// The names of parameters fixed to constant values.
    #[getter]
    fn fixed_parameters(&self) -> Vec<String> {
        self.0.fixed_parameters()
    }

    /// Load the expression into a reusable evaluator.
    fn load(&self) -> PyLikelihoodEvaluator {
        PyLikelihoodEvaluator(self.0.load())
    }

    /// Fix a parameter to a constant value.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the parameter.
    /// value : float
    ///     Value used during evaluation.
    ///
    /// Returns
    /// -------
    /// LikelihoodExpression
    ///     A new expression with the parameter fixed.
    ///
    fn fix(&self, name: &str, value: f64) -> PyResult<PyLikelihoodExpression> {
        Ok(PyLikelihoodExpression(self.0.fix(name, value)?))
    }

    /// Free a parameter that was previously fixed.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the parameter.
    ///
    /// Returns
    /// -------
    /// LikelihoodExpression
    ///     A new expression with the parameter restored as free.
    ///
    fn free(&self, name: &str) -> PyResult<PyLikelihoodExpression> {
        Ok(PyLikelihoodExpression(self.0.free(name)?))
    }

    /// Rename a parameter.
    ///
    /// Parameters
    /// ----------
    /// old : str
    ///     Current parameter name.
    /// new : str
    ///     Desired parameter name.
    ///
    /// Returns
    /// -------
    /// LikelihoodExpression
    ///     A new expression with the parameter renamed.
    ///
    fn rename_parameter(&self, old: &str, new: &str) -> PyResult<PyLikelihoodExpression> {
        Ok(PyLikelihoodExpression(self.0.rename_parameter(old, new)?))
    }

    /// Rename multiple parameters at once.
    ///
    /// Parameters
    /// ----------
    /// mapping : dict[str, str]
    ///     Mapping from old names to new names.
    ///
    /// Returns
    /// -------
    /// LikelihoodExpression
    ///     A new expression with all provided renames applied.
    ///
    fn rename_parameters(
        &self,
        mapping: HashMap<String, String>,
    ) -> PyResult<PyLikelihoodExpression> {
        Ok(PyLikelihoodExpression(self.0.rename_parameters(&mapping)?))
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
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
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
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
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() * other_expr.0.clone(),
            ))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
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

/// A structure to evaluate and minimize combinations of [`LikelihoodTerm`]s.
#[derive(Clone)]
pub struct LikelihoodEvaluator {
    likelihood_registry: LikelihoodRegistry,
    likelihood_expression: LikelihoodNode,
    free_parameter_indices: Vec<usize>,
    parameter_manager: ParameterManager,
}

impl LikelihoodEvaluator {
    /// The ordered list of parameter names (free and fixed) referenced by this expression.
    pub fn parameters(&self) -> Vec<String> {
        self.parameter_manager.parameters()
    }

    /// The ordered list of free parameter names required to evaluate this expression.
    pub fn free_parameters(&self) -> Vec<String> {
        self.parameter_manager.free_parameters()
    }

    /// The ordered list of parameters that currently have fixed values.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.parameter_manager.fixed_parameters()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.free_parameter_indices.len()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.parameter_manager.n_fixed_parameters()
    }

    /// Total number of parameters (free + fixed).
    pub fn n_parameters(&self) -> usize {
        self.parameter_manager.n_parameters()
    }

    fn assemble_parameters(&self, parameters: &[f64]) -> LadduResult<Vec<f64>> {
        validate_free_parameter_len(parameters.len(), self.free_parameter_indices.len())?;
        self.parameter_manager.assemble_full(parameters)
    }

    /// Evaluate the sum/product of all terms.
    pub fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        let parameters = self.assemble_parameters(parameters)?;
        let mut param_buffers: Vec<Vec<f64>> = self
            .likelihood_registry
            .param_counts
            .iter()
            .map(|&count| vec![0.0; count])
            .collect();
        for (layout, buffer) in self
            .likelihood_registry
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.likelihood_registry
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect::<LadduResult<Vec<_>>>()?,
        );
        Ok(self.likelihood_expression.evaluate(&likelihood_values))
    }

    /// Evaluate the gradient.
    pub fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        let parameters = self.assemble_parameters(parameters)?;
        let mut param_buffers: Vec<Vec<f64>> = self
            .likelihood_registry
            .param_counts
            .iter()
            .map(|&count| vec![0.0; count])
            .collect();
        for (layout, buffer) in self
            .likelihood_registry
            .param_layouts
            .iter()
            .zip(param_buffers.iter_mut())
        {
            for (buffer_idx, &param_idx) in layout.iter().enumerate() {
                buffer[buffer_idx] = parameters[param_idx];
            }
        }
        let likelihood_values = LikelihoodValues(
            self.likelihood_registry
                .terms
                .iter()
                .zip(param_buffers.iter())
                .map(|(term, buffer)| term.evaluate(buffer))
                .collect::<LadduResult<Vec<_>>>()?,
        );
        let mut gradient_buffers: Vec<DVector<f64>> = (0..self.likelihood_registry.terms.len())
            .map(|_| DVector::zeros(self.parameter_manager.n_parameters()))
            .collect();
        for (((term, param_buffer), gradient_buffer), layout) in self
            .likelihood_registry
            .terms
            .iter()
            .zip(param_buffers.iter())
            .zip(gradient_buffers.iter_mut())
            .zip(self.likelihood_registry.param_layouts.iter())
        {
            let term_gradient = term.evaluate_gradient(param_buffer)?; // This has a local layout
            for (term_idx, &buffer_idx) in layout.iter().enumerate() {
                gradient_buffer[buffer_idx] = term_gradient[term_idx] // This has a global layout
            }
        }
        let likelihood_gradients = LikelihoodGradients(gradient_buffers);
        let full_gradient = self
            .likelihood_expression
            .evaluate_gradient(&likelihood_values, &likelihood_gradients);
        let mut reduced = DVector::zeros(self.free_parameter_indices.len());
        for (out_idx, &global_idx) in self.free_parameter_indices.iter().enumerate() {
            reduced[out_idx] = full_gradient[global_idx];
        }
        Ok(reduced)
    }
}

impl LikelihoodTerm for LikelihoodEvaluator {
    fn update(&self) {
        self.likelihood_registry
            .terms
            .iter()
            .for_each(|term| term.update())
    }
    /// The parameter names associated with this evaluator (free and fixed).
    fn parameters(&self) -> Vec<String> {
        self.parameters()
    }
    fn parameter_manager(&self) -> &ParameterManager {
        &self.parameter_manager
    }
    /// A function that can be called to evaluate the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        LikelihoodEvaluator::evaluate(self, parameters)
    }

    /// Evaluate the gradient of the stored [`LikelihoodExpression`] over the events in the [`Dataset`]
    /// stored by the [`LikelihoodEvaluator`] with the given values for free parameters.
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        LikelihoodEvaluator::evaluate_gradient(self, parameters)
    }
}

/// A class which can be used to evaluate a collection of likelihood terms described by a
/// [`LikelihoodExpression`]
///
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodEvaluator", module = "laddu")]
pub struct PyLikelihoodEvaluator(LikelihoodEvaluator);

#[cfg(feature = "python")]
#[pymethods]
impl PyLikelihoodEvaluator {
    /// A list of the names of all parameters across all terms in all models.
    ///
    /// Returns
    /// -------
    /// parameters : list of str
    ///
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }

    /// A list of names of the free parameters.
    #[getter]
    fn free_parameters(&self) -> Vec<String> {
        self.0.free_parameters()
    }

    /// A list of names for parameters fixed to constant values.
    #[getter]
    fn fixed_parameters(&self) -> Vec<String> {
        self.0.fixed_parameters()
    }

    /// Number of free parameters in the evaluator.
    #[getter]
    fn n_free(&self) -> usize {
        self.0.n_free()
    }

    /// Number of fixed parameters in the evaluator.
    #[getter]
    fn n_fixed(&self) -> usize {
        self.0.n_fixed()
    }

    /// Total number of parameters (free + fixed).
    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }

    /// Evaluate the sum of all terms in the evaluator
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     Parameter values for the free parameters (length ``n_free``).
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
    #[cfg_attr(not(feature = "rayon"), allow(unused_variables))]
    fn evaluate(&self, parameters: Vec<f64>, threads: Option<usize>) -> PyResult<f64> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        #[cfg(feature = "rayon")]
        {
            ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| self.0.evaluate(&parameters))
                .map_err(PyErr::from)
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.0.evaluate(&parameters).map_err(PyErr::from)
        }
    }
    /// Evaluate the gradient of the sum of all terms in the evaluator
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     Parameter values for the free parameters (length ``n_free``).
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` array representing the gradient of the sum of all terms in the
    ///     evaluator with length ``n_free``.
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool or problem creating the resulting
    ///     ``numpy`` array
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    #[cfg_attr(not(feature = "rayon"), allow(unused_variables))]
    fn evaluate_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        #[cfg(feature = "rayon")]
        {
            let gradient = ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or(0))
                .build()
                .map_err(LadduError::from)?
                .install(|| self.0.evaluate_gradient(&parameters))
                .map_err(PyErr::from)?;
            Ok(PyArray1::from_slice(py, gradient.as_slice()))
        }
        #[cfg(not(feature = "rayon"))]
        {
            let gradient = self.0.evaluate_gradient(&parameters)?;
            Ok(PyArray1::from_slice(py, gradient.as_slice()))
        }
    }
    #[cfg_attr(doctest, doc = "```ignore")]
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
    /// Examples
    /// --------
    /// >>> s_nll.minimize([1.0, 0.1], method='adam', max_steps=500)  # doctest: +SKIP
    ///
    /// Notes
    /// -----
    /// The `settings` dict is forwarded to the solver as keyword arguments. Each algorithm
    /// recognises the following keys:
    ///
    /// Available algorithms
    /// --------------------
    /// **`lbfgsb` (Limited-memory BFGS-B)**
    ///     - `m` (int, default `10`): Number of correction vectors retained for the Hessian
    ///       approximation.
    ///     - `skip_hessian` (bool, default `False`): Skip the exact Hessian recomputation during
    ///       convergence checks.
    ///     - `line_search` (dict): fine-tunes the line search with entries such as::
    ///
    ///         {"method": "hagerzhang", "max_iterations": 100, "c1": 1e-4, "c2": 0.9}
    ///
    ///       Supported keys are `method` (`"morethuente"` or `"hagerzhang"`), `max_iterations`,
    ///       `c1`, `c2`, `max_zoom`, `delta`, `sigma`, `epsilon`, `theta`, `gamma`, and
    ///       `max_bisects`.
    ///
    /// **`adam` (Adaptive Moment Estimation)**
    ///     - `beta_c` (float, default `0.9`): Slope of the exponential moving average used to
    ///       judge convergence.
    ///     - `eps_loss` (float, default `sqrt(MACH_EPS)`): Minimum change in the averaged loss
    ///       required to reset the patience counter.
    ///     - `patience` (int, default `1`): Number of tolerated non-improving iterations.
    ///
    /// **`nelder-mead` (Simplex search with adaptive parameters)**
    ///     - `alpha`/`beta`/`gamma`/`delta` control the reflection, expansion, contraction, and
    ///       shrink steps (defaults: `1.0`, `2.0`, `0.5`, `0.5`).
    ///     - `adaptive` (bool): Enable the Gao & Han (2010) adaptive schedule.
    ///     - `expansion_method`: Either `"greedyminimization"` or `"greedyexpansion"` (see
    ///       Lagarias et al., 1998).
    ///     - `simplex_construction_method`: `"scaledorthogonal"`, `"orthogonal"`, or `"custom"`.
    ///       When `custom` is selected, provide `simplex` with explicit vertices.
    ///     - `f_terminator` / `x_terminator` and their tolerances (`eps_f` / `eps_x`) choose the
    ///       function- and position-based stopping criteria described by Singer & Singer (2004).
    ///
    /// **`pso` (Particle Swarm Optimisation)**
    ///     - `swarm_position_initializer`: `"randominlimits"`, `"latinhypercube"`, or `"custom"`.
    ///       Random initialisers require `swarm_position_bounds` and `swarm_size`; the custom mode
    ///       consumes the `swarm` argument directly.
    ///     - `swarm_topology` (`"global"` or `"ring"`) and `swarm_update_method`
    ///       (`"sync"`/`"async"`) select the neighbourhood structure.
    ///     - `swarm_boundary_method` (`"inf"` or `"shr"`) and `use_transform` control how bounds
    ///       are enforced.
    ///     - `swarm_velocity_bounds` (optional), `omega` (default `0.8`), `c1` and `c2` (defaults
    ///       `0.1`) tune the particle dynamics.
    ///
    /// References
    /// ----------
    /// Gao, F. & Han, L. (2010). *Implementing the Nelder-Mead simplex algorithm with adaptive
    /// parameters*. Comput. Optim. Appl. 51(1), 259–277. <https://doi.org/10.1007/s10589-010-9329-3>
    ///
    /// Lagarias, J. C., Reeds, J. A., Wright, M. H., & Wright, P. E. (1998). *Convergence Properties
    /// of the Nelder–Mead Simplex Method in Low Dimensions*. SIAM J. Optim. 9(1), 112–147.
    /// <https://doi.org/10.1137/S1052623496303470>
    ///
    /// Singer, S. & Singer, S. (2004). *Efficient Implementation of the Nelder–Mead Search Algorithm*.
    /// Appl. Numer. Anal. & Comput. 1(2), 524–534. <https://doi.org/10.1002/anac.200410015>
    ///
    #[cfg_attr(doctest, doc = "```")]
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
        validate_free_parameter_len(p0.len(), self.0.n_free())?;
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
    /// See Also
    /// --------
    /// NLL.mcmc
    /// StochasticNLL.mcmc
    ///
    /// Examples
    /// --------
    /// >>> expr = likelihood_sum([LikelihoodScalar('scale')])  # doctest: +SKIP
    /// >>> evaluator = expr.load()  # doctest: +SKIP
    /// >>> evaluator.minimize([1.0], method='pso', max_steps=150)  # doctest: +SKIP
    ///
    /// Examples
    /// --------
    /// >>> from laddu import LikelihoodScalar, likelihood_sum
    /// >>> evaluator = likelihood_sum([LikelihoodScalar('alpha')]).load()
    /// >>> summary = evaluator.mcmc([[0.0], [0.4]], max_steps=4, method='aies')
    /// >>> summary.dimension[2]
    /// 1
    /// >>> summary.get_flat_chain().shape[1]
    /// 1
    ///
    /// Notes
    /// -----
    /// The `settings` dict is passed to the MCMC algorithm as keyword arguments. Each
    /// algorithm exposes different keys:
    ///
    /// Sampling algorithms
    /// -------------------
    /// **`aies` (Affine-Invariant Ensemble Sampler; Goodman & Weare, 2010)**
    ///     - `moves`: Sequence of tuples `("stretch", freq)` or `("walk", freq)`. The optional
    ///       dictionary overrides parameters such as `{"a": 2.0}` for the stretch move.
    ///     - `n_adaptive`: Number of adaptive warm-up steps (default `0`).
    ///
    /// **`ess` (Ensemble Slice Sampler; Karamanis & Beutler, 2021)**
    ///     - `moves`: Sequence of tuples selecting `"differential"`, `"gaussian"`, or `"global"`
    ///       moves. The optional dictionary configures `scale` (default `1.0`), `rescale_cov`
    ///       (default `0.001`), and `n_components` (default `5`) for the global move.
    ///     - `n_adaptive`: Number of adaptive moves to perform at the start of sampling (default `0`).
    ///     - `max_steps`: Maximum number of expansions/contractions per slice-evaluation step (default `10000`).
    ///     - `mu`: Adaptive scaling parameter used when `n_adaptive > 0` (default `1.0`).
    ///
    /// References
    /// ----------
    /// Goodman, J. & Weare, J. (2010). *Ensemble samplers with affine invariance*. CAMCoS 5(1), 65–80. <https://doi.org/10.2140/camcos.2010.5.65>
    ///
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat Comput 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
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
        validate_mcmc_parameter_len(&p0, self.0.n_free())?;
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
pub struct LikelihoodScalar {
    name: String,
    parameter_manager: ParameterManager,
}

impl LikelihoodScalar {
    /// Create a new [`LikelihoodScalar`] with a parameter with the given name and wrap it as a
    /// [`LikelihoodExpression`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new<T: AsRef<str>>(name: T) -> LikelihoodExpression {
        Self::new_term(name).into_expression()
    }

    /// Construct the underlying [`LikelihoodTerm`] for advanced use cases.
    pub fn new_term<T: AsRef<str>>(name: T) -> Box<Self> {
        let name_str: String = name.as_ref().into();
        let manager = ParameterManager::new_from_names(std::slice::from_ref(&name_str));
        Self {
            name: name_str,
            parameter_manager: manager,
        }
        .into()
    }
}

impl LikelihoodTerm for LikelihoodScalar {
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        validate_free_parameter_len(parameters.len(), 1)?;
        Ok(parameters[0])
    }

    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        validate_free_parameter_len(parameters.len(), 1)?;
        Ok(DVector::from_vec(vec![1.0]))
    }

    fn parameters(&self) -> Vec<String> {
        vec![self.name.clone()]
    }

    fn parameter_manager(&self) -> &ParameterManager {
        &self.parameter_manager
    }
}

/// A parameterized scalar term which can be converted into a [`LikelihoodExpression`].
///
/// Parameters
/// ----------
/// name : str
///     The name of the new scalar parameter.
///
/// Returns
/// -------
/// LikelihoodExpression
///     A [`LikelihoodExpression`] representing a single free scaling parameter.
///
/// See Also
/// --------
/// likelihood_sum
/// likelihood_product
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodScalar, likelihood_sum
/// >>> expr = likelihood_sum([LikelihoodScalar('alpha')])
/// >>> expr.load().evaluate([1.25])
/// 1.25
#[cfg(feature = "python")]
#[pyfunction(name = "LikelihoodScalar")]
pub fn py_likelihood_scalar(name: String) -> PyLikelihoodExpression {
    PyLikelihoodExpression(LikelihoodScalar::new(name))
}

#[cfg(test)]
mod tests {
    use super::{LikelihoodScalar, LikelihoodTerm, NLL};
    #[cfg(feature = "mpi")]
    use accurate::{sum::Klein, traits::SumWithAccumulator};
    use approx::assert_relative_eq;
    #[cfg(feature = "mpi")]
    use laddu_core::mpi::{finalize_mpi, get_world, use_mpi, LadduMPI};
    use laddu_core::{
        amplitudes::{parameter, Amplitude, AmplitudeID, ParameterLike},
        data::{Dataset, DatasetMetadata, EventData},
        resources::{Cache, ParameterID, Parameters, Resources},
        utils::vectors::Vec4,
        Expression, LadduError, LadduResult,
    };
    #[cfg(feature = "mpi")]
    use mpi::topology::SimpleCommunicator;
    use nalgebra::DVector;
    use num::complex::Complex64;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    const LENGTH_MISMATCH_MESSAGE_FRAGMENT: &str = "length mismatch";
    const AMPLITUDE_NOT_FOUND_MESSAGE_FRAGMENT: &str = "No registered amplitude";

    #[derive(Clone, Serialize, Deserialize)]
    struct ConstantAmplitude {
        name: String,
        parameter: ParameterLike,
        pid: ParameterID,
    }

    impl ConstantAmplitude {
        #[allow(clippy::new_ret_no_self)]
        fn new(name: &str, parameter: ParameterLike) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                parameter,
                pid: ParameterID::default(),
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for ConstantAmplitude {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.pid = resources.register_parameter(&self.parameter)?;
            resources.register_amplitude(&self.name)
        }

        fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
            Complex64::new(parameters.get(self.pid), 0.0)
        }

        fn compute_gradient(
            &self,
            _parameters: &Parameters,
            _cache: &Cache,
            gradient: &mut DVector<Complex64>,
        ) {
            if let ParameterID::Parameter(index) = self.pid {
                gradient[index] = Complex64::ONE;
            }
        }
    }

    fn dataset_with_weights(weights: &[f64]) -> Arc<Dataset> {
        let metadata = Arc::new(DatasetMetadata::default());
        let events = weights
            .iter()
            .map(|&weight| {
                Arc::new(EventData {
                    p4s: vec![Vec4::new(0.0, 0.0, 0.0, 1.0)],
                    aux: vec![],
                    weight,
                })
            })
            .collect();
        Arc::new(Dataset::new_with_metadata(events, metadata))
    }

    fn make_constant_nll() -> (Box<NLL>, Vec<f64>) {
        let amp = ConstantAmplitude::new("amp", parameter("scale")).unwrap();
        let expr = amp.norm_sqr();
        let data = dataset_with_weights(&[1.0, 2.0]);
        let mc = dataset_with_weights(&[0.5, 1.5]);
        let nll = NLL::new(&expr, &data, &mc).unwrap();
        (nll, vec![2.0])
    }

    fn case_nll_evaluate_short(nll: &NLL) -> LadduResult<()> {
        nll.evaluate(&[]).map(|_| ())
    }

    fn case_nll_evaluate_gradient_long(nll: &NLL) -> LadduResult<()> {
        nll.evaluate_gradient(&[1.0, 2.0]).map(|_| ())
    }

    fn case_nll_project_short(nll: &NLL) -> LadduResult<()> {
        nll.project(&[], None).map(|_| ())
    }

    fn case_nll_project_gradient_long(nll: &NLL) -> LadduResult<()> {
        nll.project_gradient(&[1.0, 2.0], None).map(|_| ())
    }

    fn case_nll_project_with_short(nll: &NLL) -> LadduResult<()> {
        nll.project_with_local::<&str>(&[], &["missing_amplitude"], None)
            .map(|_| ())
    }

    fn case_nll_project_gradient_with_long(nll: &NLL) -> LadduResult<()> {
        nll.project_gradient_with_local::<&str>(&[1.0, 2.0], &["missing_amplitude"], None)
            .map(|_| ())
    }

    #[cfg(feature = "mpi")]
    fn sum_weights(dataset: &Dataset, indices: &[usize]) -> f64 {
        indices
            .iter()
            .map(|&idx| dataset.events_local()[idx].weight)
            .sum_with_accumulator::<Klein<f64>>()
    }

    #[cfg(feature = "mpi")]
    fn evaluate_partial_value(
        nll: &NLL,
        parameters: &[f64],
        indices: &[usize],
        n_data_batch: f64,
    ) -> f64 {
        let data_result = nll.data_evaluator.evaluate_batch_local(parameters, indices);
        let mc_result = nll.accmc_evaluator.evaluate_local(parameters);
        let n_mc = nll.accmc_evaluator.dataset.n_events_weighted();
        let n_data_total = nll.data_evaluator.dataset.n_events_weighted();
        let data_term = indices
            .iter()
            .zip(data_result.iter())
            .map(|(&i, &l)| {
                let e = &nll.data_evaluator.dataset.events_local()[i];
                e.weight * l.re.ln()
            })
            .sum_with_accumulator::<Klein<f64>>();
        let mc_term = mc_result
            .iter()
            .zip(nll.accmc_evaluator.dataset.events_local().iter())
            .map(|(l, e)| e.weight * l.re)
            .sum_with_accumulator::<Klein<f64>>();
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn partition_indices(world: &SimpleCommunicator, total: usize) -> Vec<Vec<usize>> {
        let (counts, displs): (Vec<i32>, Vec<i32>) = world.get_counts_displs(total);
        counts
            .into_iter()
            .zip(displs.into_iter())
            .map(|(count, displ)| {
                let start = displ as usize;
                let end = start + count as usize;
                (start..end).collect()
            })
            .collect()
    }

    fn case_likelihood_evaluate_short() -> LadduResult<()> {
        let alpha = LikelihoodScalar::new("alpha");
        let evaluator = alpha.load();
        evaluator.evaluate(&[]).map(|_| ())
    }

    fn case_likelihood_gradient_long() -> LadduResult<()> {
        let alpha = LikelihoodScalar::new("alpha");
        let evaluator = alpha.load();
        evaluator.evaluate_gradient(&[1.0, 2.0]).map(|_| ())
    }

    #[test]
    fn table_driven_length_mismatch_errors() {
        let (nll, _) = make_constant_nll();
        let cases: [(&str, LadduResult<()>); 8] = [
            ("nll.evaluate short", case_nll_evaluate_short(nll.as_ref())),
            (
                "nll.evaluate_gradient long",
                case_nll_evaluate_gradient_long(nll.as_ref()),
            ),
            ("nll.project short", case_nll_project_short(nll.as_ref())),
            (
                "nll.project_gradient long",
                case_nll_project_gradient_long(nll.as_ref()),
            ),
            (
                "nll.project_with short",
                case_nll_project_with_short(nll.as_ref()),
            ),
            (
                "nll.project_gradient_with long",
                case_nll_project_gradient_with_long(nll.as_ref()),
            ),
            (
                "likelihood.evaluate short",
                case_likelihood_evaluate_short(),
            ),
            (
                "likelihood.evaluate_gradient long",
                case_likelihood_gradient_long(),
            ),
        ];
        for (label, result) in cases {
            let err = result.unwrap_err();
            assert!(
                matches!(err, LadduError::LengthMismatch { .. }),
                "expected LengthMismatch for {label}, got {err:?}"
            );
            assert!(
                err.to_string().contains(LENGTH_MISMATCH_MESSAGE_FRAGMENT),
                "expected message containing \"{LENGTH_MISMATCH_MESSAGE_FRAGMENT}\" for {label}, got {}",
                err
            );
        }
    }

    #[test]
    fn table_driven_unknown_amplitude_errors() {
        let (nll, params) = make_constant_nll();
        let cases: [(&str, LadduResult<()>); 4] = [
            (
                "activate_strict unknown",
                nll.activate_strict("missing_amplitude"),
            ),
            (
                "isolate_strict unknown",
                nll.isolate_strict("missing_amplitude"),
            ),
            (
                "project_with unknown",
                nll.project_with_local::<&str>(&params, &["missing_amplitude"], None)
                    .map(|_| ()),
            ),
            (
                "project_gradient_with unknown",
                nll.project_gradient_with_local::<&str>(&params, &["missing_amplitude"], None)
                    .map(|_| ()),
            ),
        ];
        for (label, result) in cases {
            let err = result.unwrap_err();
            assert!(
                matches!(err, LadduError::AmplitudeNotFoundError { .. }),
                "expected AmplitudeNotFoundError for {label}, got {err:?}"
            );
            assert!(
                err.to_string()
                    .contains(AMPLITUDE_NOT_FOUND_MESSAGE_FRAGMENT),
                "expected message containing \"{AMPLITUDE_NOT_FOUND_MESSAGE_FRAGMENT}\" for {label}, got {}",
                err
            );
        }
    }

    #[test]
    fn likelihood_expression_evaluates_scalar_sum() {
        let alpha = LikelihoodScalar::new("alpha");
        let beta = LikelihoodScalar::new("beta");
        let expr = &alpha + &beta;
        assert_eq!(expr.parameters(), vec!["alpha", "beta"]);
        let evaluator = expr.load();
        let params = vec![2.0, 3.0];
        assert_relative_eq!(evaluator.evaluate(&params).unwrap(), 5.0);
        let grad = evaluator.evaluate_gradient(&params).unwrap();
        assert_relative_eq!(grad[0], 1.0);
        assert_relative_eq!(grad[1], 1.0);
    }

    #[test]
    fn likelihood_expression_evaluates_scalar_product() {
        let alpha = LikelihoodScalar::new("alpha");
        let beta = LikelihoodScalar::new("beta");
        let expr = &alpha * &beta;
        let evaluator = expr.load();
        let params = vec![2.0, 3.0];
        assert_relative_eq!(evaluator.evaluate(&params).unwrap(), 6.0);
        let grad = evaluator.evaluate_gradient(&params).unwrap();
        assert_relative_eq!(grad[0], 3.0);
        assert_relative_eq!(grad[1], 2.0);
    }

    #[test]
    fn likelihood_expression_tracks_fixed_parameters() {
        let alpha = LikelihoodScalar::new("alpha");
        let beta = LikelihoodScalar::new("beta");
        let expr = (&alpha + &beta).fix("alpha", 1.5).unwrap();
        assert_eq!(expr.parameters(), vec!["alpha", "beta"]);
        assert_eq!(expr.free_parameters(), vec!["beta"]);
        assert_eq!(expr.fixed_parameters(), vec!["alpha"]);
        let evaluator = expr.load();
        assert_eq!(evaluator.parameters(), vec!["alpha", "beta"]);
        assert_eq!(evaluator.free_parameters(), vec!["beta"]);
        assert_eq!(evaluator.fixed_parameters(), vec!["alpha"]);
        let params_free = vec![2.0];
        assert_relative_eq!(evaluator.evaluate(&params_free).unwrap(), 3.5);
        let grad_free = evaluator.evaluate_gradient(&params_free).unwrap();
        assert_eq!(grad_free.len(), 1);
        assert_relative_eq!(grad_free[0], 1.0);
    }

    #[test]
    fn nll_evaluate_and_gradient_match_closed_form() {
        let (nll, params) = make_constant_nll();
        let intensity = params[0] * params[0];
        let weight_sum = 3.0;
        let expected = -2.0 * (weight_sum * intensity.ln() - intensity);
        assert_relative_eq!(nll.evaluate(&params).unwrap(), expected, epsilon = 1e-12);
        let grad = nll.evaluate_gradient(&params).unwrap();
        let expected_grad = -4.0 * (weight_sum / params[0] - params[0]);
        assert_relative_eq!(grad[0], expected_grad, epsilon = 1e-12);
    }

    #[test]
    fn nll_project_returns_weighted_intensity() {
        let (nll, params) = make_constant_nll();
        let projection = nll.project_local(&params, None).unwrap();
        assert_relative_eq!(projection[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(projection[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn nll_project_reports_structured_length_error() {
        let (nll, _) = make_constant_nll();
        let err = nll.project(&[], None).unwrap_err();
        assert!(matches!(
            err,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn nll_project_with_reports_structured_missing_amplitude_error() {
        let (nll, params) = make_constant_nll();
        let err = nll
            .project_with_local::<&str>(&params, &["missing_amplitude"], None)
            .unwrap_err();
        assert!(matches!(err, LadduError::AmplitudeNotFoundError { .. }));
    }

    #[test]
    fn nll_project_with_validates_length_before_isolation() {
        let (nll, _) = make_constant_nll();
        let err = nll
            .project_with_local::<&str>(&[], &["missing_amplitude"], None)
            .unwrap_err();
        assert!(matches!(
            err,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn nll_project_gradient_with_validates_length_before_isolation() {
        let (nll, _) = make_constant_nll();
        let err = nll
            .project_gradient_with_local::<&str>(&[1.0, 2.0], &["missing_amplitude"], None)
            .unwrap_err();
        assert!(matches!(
            err,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn stochastic_nll_validates_batch_size() {
        let (nll, _params) = make_constant_nll();
        let err_zero = match nll.to_stochastic(0, Some(0)) {
            Ok(_) => panic!("expected batch_size=0 to return an error"),
            Err(err) => err,
        };
        assert!(matches!(
            err_zero,
            LadduError::LengthMismatch {
                expected: 2,
                actual: 0,
                ..
            }
        ));

        let err_large = match nll.to_stochastic(3, Some(0)) {
            Ok(_) => panic!("expected oversized batch to return an error"),
            Err(err) => err,
        };
        assert!(matches!(
            err_large,
            LadduError::LengthMismatch {
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }

    #[test]
    fn stochastic_nll_accepts_full_dataset_batch() {
        let (nll, params) = make_constant_nll();
        let stochastic = nll.to_stochastic(2, Some(0)).unwrap();
        let value = stochastic.evaluate(&params).unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn likelihood_evaluator_reports_length_mismatch() {
        let alpha = LikelihoodScalar::new("alpha");
        let evaluator = alpha.load();

        let err_short = evaluator.evaluate(&[]).unwrap_err();
        assert!(matches!(
            err_short,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));

        let err_long = evaluator.evaluate_gradient(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err_long,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn mpi_negative_paths_report_structured_errors() {
        use_mpi(true);
        let Some(world) = get_world() else {
            finalize_mpi();
            return;
        };
        let (nll, params) = make_constant_nll();

        let err_len = nll.project_mpi(&[], None, &world).unwrap_err();
        assert!(matches!(
            err_len,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));

        let err_amp = nll
            .project_with_mpi::<&str>(&params, &["missing_amplitude"], None, &world)
            .unwrap_err();
        assert!(matches!(err_amp, LadduError::AmplitudeNotFoundError { .. }));
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn mpi_value_and_gradient_match_total_non_mpi() {
        use_mpi(true);
        let Some(world) = get_world() else {
            finalize_mpi();
            return;
        };
        let (nll, params) = make_constant_nll();
        let total_events = nll.data_evaluator.dataset.n_events();
        let partitions = partition_indices(&world, total_events);

        let expected_value: f64 = partitions
            .iter()
            .map(|partition| {
                let n_data_batch = sum_weights(&nll.data_evaluator.dataset, partition);
                evaluate_partial_value(&nll, &params, partition, n_data_batch)
            })
            .sum_with_accumulator::<Klein<f64>>();

        let mpi_value = nll.evaluate_mpi(&params, &world);
        assert_relative_eq!(mpi_value, expected_value);

        let local_gradient = nll.evaluate_gradient(&params).unwrap();
        let mpi_gradient = nll.evaluate_gradient_mpi(&params, &world);
        assert_relative_eq!(mpi_gradient, local_gradient);

        finalize_mpi();
    }
}
