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
    amplitudes::{central_difference, Evaluator, Expression},
    data::Dataset,
    resources::Parameters,
    LadduResult,
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
use laddu_core::LadduError;
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
}

impl NLL {
    /// Construct an [`NLL`] from an [`Expression`] and two [`Dataset`]s (data and Monte Carlo). This mirrors loading a model but starts from
    /// the expression directly.
    pub fn new(
        expression: &Expression,
        ds_data: &Arc<Dataset>,
        ds_accmc: &Arc<Dataset>,
    ) -> LadduResult<Box<Self>> {
        Ok(Self {
            data_evaluator: expression.load(ds_data)?,
            accmc_evaluator: expression.load(ds_accmc)?,
        }
        .into())
    }

    /// The parameter names for this NLL.
    pub fn parameters(&self) -> Vec<String> {
        self.data_evaluator.parameters()
    }

    /// The free parameter names for this NLL.
    pub fn free_parameters(&self) -> Vec<String> {
        self.data_evaluator.free_parameters()
    }

    /// The fixed parameter names for this NLL.
    pub fn fixed_parameters(&self) -> Vec<String> {
        self.data_evaluator.fixed_parameters()
    }

    /// Number of free parameters.
    pub fn n_free(&self) -> usize {
        self.data_evaluator.n_free()
    }

    /// Number of fixed parameters.
    pub fn n_fixed(&self) -> usize {
        self.data_evaluator.n_fixed()
    }

    /// Total number of parameters.
    pub fn n_parameters(&self) -> usize {
        self.data_evaluator.n_parameters()
    }

    /// Return a new [`NLL`] with the given parameter fixed to a value.
    pub fn fix(&self, name: &str, value: f64) -> LadduResult<Box<Self>> {
        Ok(Self {
            data_evaluator: self.data_evaluator.fix(name, value)?,
            accmc_evaluator: self.accmc_evaluator.fix(name, value)?,
        }
        .into())
    }

    /// Return a new [`NLL`] with the given parameter freed.
    pub fn free(&self, name: &str) -> LadduResult<Box<Self>> {
        Ok(Self {
            data_evaluator: self.data_evaluator.free(name)?,
            accmc_evaluator: self.accmc_evaluator.free(name)?,
        }
        .into())
    }

    /// Return a new [`NLL`] with a single parameter renamed.
    pub fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<Box<Self>> {
        Ok(Self {
            data_evaluator: self.data_evaluator.rename_parameter(old, new)?,
            accmc_evaluator: self.accmc_evaluator.rename_parameter(old, new)?,
        }
        .into())
    }

    /// Return a new [`NLL`] with several parameters renamed.
    pub fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<Box<Self>> {
        Ok(Self {
            data_evaluator: self.data_evaluator.rename_parameters(mapping)?,
            accmc_evaluator: self.accmc_evaluator.rename_parameters(mapping)?,
        }
        .into())
    }
    /// Create a new [`StochasticNLL`] from this [`NLL`].
    pub fn to_stochastic(&self, batch_size: usize, seed: Option<usize>) -> StochasticNLL {
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
        self.accmc_evaluator.activate_strict(name)
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name, skipping missing entries.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.activate_many(names);
        self.accmc_evaluator.activate_many(names);
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name and return an error if any are missing.
    pub fn activate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.data_evaluator.activate_many_strict(names)?;
        self.accmc_evaluator.activate_many_strict(names)
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
        self.accmc_evaluator.deactivate_strict(name)
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name, skipping missing entries.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.deactivate_many(names);
        self.accmc_evaluator.deactivate_many(names);
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name and return an error if any are missing.
    pub fn deactivate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.data_evaluator.deactivate_many_strict(names)?;
        self.accmc_evaluator.deactivate_many_strict(names)
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
        self.accmc_evaluator.isolate_strict(name)
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name (deactivate the rest), skipping missing entries.
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.data_evaluator.isolate_many(names);
        self.accmc_evaluator.isolate_many(names);
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name (deactivate the rest) and return an error if any are missing.
    pub fn isolate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.data_evaluator.isolate_many_strict(names)?;
        self.accmc_evaluator.isolate_many_strict(names)
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
    ) -> LadduResult<Vec<f64>> {
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.resources.read().active.clone();
            mc_evaluator.isolate_many_strict(names)?;
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
            self.isolate_many_strict(names)?;
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
                        .collect::<Vec<_>>(),
                    gradient_values,
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
                        .collect::<Vec<_>>(),
                    gradient_values,
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
                        .collect::<Vec<_>>(),
                    gradient_values,
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
                        .collect::<Vec<_>>(),
                    gradient_values,
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
        self.data_evaluator.parameters()
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
        let indices = self.batch_indices.lock();
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_mpi(parameters, &indices, &world);
            }
        }
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
        self.evaluate_local(parameters, &indices, n_data_batch_local)
    }

    fn evaluate_gradient(&self, parameters: &[f64]) -> DVector<f64> {
        let indices = self.batch_indices.lock();
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_gradient_mpi(parameters, &indices, &world);
            }
        }
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
        self.evaluate_gradient_local(parameters, &indices, n_data_batch_local)
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
                .zip(data_result.par_iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.events[i];
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
                .zip(data_result.iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.events[i];
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
        let total = self.nll.data_evaluator.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
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
        let data_term: DVector<f64> = indices
            .par_iter()
            .map(|&idx| {
                let event = &self.nll.data_evaluator.dataset.events[idx];
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
                            amp.compute_gradient(&data_parameters, event, cache, grad)
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
                            amp.compute(&data_parameters, event, cache)
                        } else {
                            Complex64::ZERO
                        }
                    })
                    .collect();
                (
                    event.weight,
                    self.nll.data_evaluator.expression.evaluate(&amp_vals),
                    self.nll
                        .data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &gradient_values),
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
                        .collect::<Vec<_>>(),
                    gradient_values,
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
        let data_term: DVector<f64> = indices
            .iter()
            .map(|&idx| {
                let event = &self.nll.data_evaluator.dataset.events[idx];
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
                            amp.compute_gradient(&data_parameters, event, cache, grad)
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
                            amp.compute(&data_parameters, event, cache)
                        } else {
                            Complex64::ZERO
                        }
                    })
                    .collect();
                (
                    event.weight,
                    self.nll.data_evaluator.expression.evaluate(&amp_vals),
                    self.nll
                        .data_evaluator
                        .expression
                        .evaluate_gradient(&amp_vals, &gradient_values),
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
                        .collect::<Vec<_>>(),
                    gradient_values,
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
        let total = self.nll.data_evaluator.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
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
    fn to_stochastic(&self, batch_size: usize, seed: Option<usize>) -> PyStochasticNLL {
        PyStochasticNLL(self.0.to_stochastic(batch_size, seed))
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
    /// Examples
    /// --------
    /// >>> nll.minimize([1.0, 0.1], method='lbfgsb', max_steps=200)  # doctest: +SKIP
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
#[derive(Clone, Default)]
pub struct LikelihoodExpression {
    registry: LikelihoodRegistry,
    tree: LikelihoodNode,
}

impl LikelihoodExpression {
    /// Build a [`LikelihoodExpression`] from a single [`LikelihoodTerm`].
    pub fn from_term(term: Box<dyn LikelihoodTerm>) -> Self {
        Self {
            registry: LikelihoodRegistry::singleton(term),
            tree: LikelihoodNode::Term(0),
        }
    }

    /// Create an expression representing zero, the additive identity.
    pub fn zero() -> Self {
        Self {
            registry: LikelihoodRegistry::default(),
            tree: LikelihoodNode::Zero,
        }
    }

    /// Create an expression representing one, the multiplicative identity.
    pub fn one() -> Self {
        Self {
            registry: LikelihoodRegistry::default(),
            tree: LikelihoodNode::One,
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
        LikelihoodExpression {
            registry,
            tree: build(Box::new(left_tree), Box::new(right_tree)),
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

    /// The free parameters used across all terms in this expression.
    pub fn parameters(&self) -> Vec<String> {
        self.registry.parameters()
    }

    /// Load a `LikelihoodExpression` so it can be evaluated repeatedly.
    pub fn load(&self) -> LikelihoodEvaluator {
        LikelihoodEvaluator {
            likelihood_registry: self.registry.clone(),
            likelihood_expression: self.tree.clone(),
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
    param_name_to_index: HashMap<String, usize>,
    param_names: Vec<String>,
    param_layouts: Vec<Vec<usize>>,
    param_counts: Vec<usize>,
}

impl LikelihoodRegistry {
    fn singleton(term: Box<dyn LikelihoodTerm>) -> Self {
        let mut registry = Self::default();
        registry.push_term(term);
        registry
    }

    fn parameters(&self) -> Vec<String> {
        self.param_names.clone()
    }

    fn push_term(&mut self, term: Box<dyn LikelihoodTerm>) -> usize {
        let term_idx = self.terms.len();
        let parameters = term.parameters();
        for param_name in &parameters {
            if !self.param_name_to_index.contains_key(param_name) {
                let index = self.param_name_to_index.len();
                self.param_name_to_index.insert(param_name.clone(), index);
                self.param_names.push(param_name.clone());
            }
        }
        let param_layout: Vec<usize> = parameters
            .iter()
            .map(|name| self.param_name_to_index[name])
            .collect();
        self.param_layouts.push(param_layout);
        self.param_counts.push(parameters.len());
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
    /// The free parameters used by the expression.
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }

    /// Load the expression into a reusable evaluator.
    fn load(&self) -> PyLikelihoodEvaluator {
        PyLikelihoodEvaluator(self.0.load())
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
}

impl LikelihoodTerm for LikelihoodEvaluator {
    fn update(&self) {
        self.likelihood_registry
            .terms
            .iter()
            .for_each(|term| term.update())
    }
    /// The parameter names used in [`LikelihoodEvaluator::evaluate`]'s input in order.
    fn parameters(&self) -> Vec<String> {
        self.likelihood_registry.parameters()
    }
    /// A function that can be called to evaluate the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    fn evaluate(&self, parameters: &[f64]) -> f64 {
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
                .collect(),
        );
        self.likelihood_expression.evaluate(&likelihood_values)
    }

    /// Evaluate the gradient of the stored [`LikelihoodExpression`] over the events in the [`Dataset`]
    /// stored by the [`LikelihoodEvaluator`] with the given values for free parameters.
    fn evaluate_gradient(&self, parameters: &[f64]) -> DVector<f64> {
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
                .collect(),
        );
        let mut gradient_buffers: Vec<DVector<f64>> = (0..self.likelihood_registry.terms.len())
            .map(|_| DVector::zeros(self.likelihood_registry.param_names.len()))
            .collect();
        for (((term, param_buffer), gradient_buffer), layout) in self
            .likelihood_registry
            .terms
            .iter()
            .zip(param_buffers.iter())
            .zip(gradient_buffers.iter_mut())
            .zip(self.likelihood_registry.param_layouts.iter())
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

/// A class which can be used to evaluate a collection of likelihood terms described by a
/// [`LikelihoodExpression`]
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
    /// Create a new [`LikelihoodScalar`] with a parameter with the given name and wrap it as a
    /// [`LikelihoodExpression`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new<T: AsRef<str>>(name: T) -> LikelihoodExpression {
        Self::new_term(name).into_expression()
    }

    /// Construct the underlying [`LikelihoodTerm`] for advanced use cases.
    pub fn new_term<T: AsRef<str>>(name: T) -> Box<Self> {
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
    use approx::assert_relative_eq;
    use laddu_core::{
        amplitudes::{parameter, Amplitude, AmplitudeID, ParameterLike},
        data::{Dataset, DatasetMetadata, EventData},
        resources::{Cache, ParameterID, Parameters, Resources},
        utils::vectors::Vec4,
        Expression, LadduResult,
    };
    use nalgebra::DVector;
    use num::complex::Complex64;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

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

        fn compute(
            &self,
            parameters: &Parameters,
            _event: &EventData,
            _cache: &Cache,
        ) -> Complex64 {
            Complex64::new(parameters.get(self.pid), 0.0)
        }

        fn compute_gradient(
            &self,
            _parameters: &Parameters,
            _event: &EventData,
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

    #[test]
    fn likelihood_expression_evaluates_scalar_sum() {
        let alpha = LikelihoodScalar::new("alpha");
        let beta = LikelihoodScalar::new("beta");
        let expr = &alpha + &beta;
        assert_eq!(expr.parameters(), vec!["alpha", "beta"]);
        let evaluator = expr.load();
        let params = vec![2.0, 3.0];
        assert_relative_eq!(evaluator.evaluate(&params), 5.0);
        let grad = evaluator.evaluate_gradient(&params);
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
        assert_relative_eq!(evaluator.evaluate(&params), 6.0);
        let grad = evaluator.evaluate_gradient(&params);
        assert_relative_eq!(grad[0], 3.0);
        assert_relative_eq!(grad[1], 2.0);
    }

    #[test]
    fn nll_evaluate_and_gradient_match_closed_form() {
        let (nll, params) = make_constant_nll();
        let intensity = params[0] * params[0];
        let weight_sum = 3.0;
        let expected = -2.0 * (weight_sum * intensity.ln() - intensity);
        assert_relative_eq!(nll.evaluate(&params), expected, epsilon = 1e-12);
        let grad = nll.evaluate_gradient(&params);
        let expected_grad = -4.0 * (weight_sum / params[0] - params[0]);
        assert_relative_eq!(grad[0], expected_grad, epsilon = 1e-12);
    }

    #[test]
    fn nll_project_returns_weighted_intensity() {
        let (nll, params) = make_constant_nll();
        let projection = nll.project_local(&params, None);
        assert_relative_eq!(projection[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(projection[1], 3.0, epsilon = 1e-12);
    }
}
