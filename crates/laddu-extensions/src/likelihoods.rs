use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    sync::Arc,
};

use accurate::{sum::Klein, traits::*};
use auto_ops::*;
use dyn_clone::DynClone;
use fastrand::Rng;
use ganesh::{
    mcmc::{ESSMove, Ensemble},
    Function, Minimizer, Sampler, Status,
};
use laddu_core::{
    amplitudes::{central_difference, AmplitudeValues, Evaluator, GradientValues, Model},
    data::Dataset,
    resources::Parameters,
    Complex, DVector, Float, LadduError,
};

#[cfg(feature = "mpi")]
use laddu_core::mpi::LadduMPI;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

#[cfg(feature = "python")]
use crate::ganesh_ext::py_ganesh::{
    py_parse_mcmc_options, py_parse_minimizer_options, PyEnsemble, PyStatus,
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
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};

use crate::ganesh_ext::{MCMCOptions, MinimizerOptions};

/// A trait which describes a term that can be used like a likelihood (more correctly, a negative
/// log-likelihood) in a minimization.
pub trait LikelihoodTerm: DynClone + Send + Sync {
    /// Evaluate the term given some input parameters.
    fn evaluate(&self, parameters: &[Float]) -> Float;
    /// Evaluate the gradient of the term given some input parameters.
    fn evaluate_gradient(&self, parameters: &[Float]) -> DVector<Float> {
        central_difference(parameters, |parameters: &[Float]| self.evaluate(parameters))
    }
    /// The list of names of the input parameters for [`LikelihoodTerm::evaluate`].
    fn parameters(&self) -> Vec<String>;
}

dyn_clone::clone_trait_object!(LikelihoodTerm);

/// A term in an expression with multiple likelihood components
///
/// See Also
/// --------
/// NLL.as_term
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
    pub fn project_local(
        &self,
        parameters: &[Float],
        mc_evaluator: Option<Evaluator>,
    ) -> Vec<Float> {
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
        let n_mc = self.accmc_evaluator.dataset.n_events();
        #[cfg(feature = "rayon")]
        let output: Vec<Float> = result
            .par_iter()
            .zip(mc_dataset.events.par_iter())
            .map(|(l, e)| e.weight * l.re / n_mc as Float)
            .collect();

        #[cfg(not(feature = "rayon"))]
        let output: Vec<Float> = result
            .iter()
            .zip(mc_dataset.events.iter())
            .map(|(l, e)| e.weight * l.re / n_mc as Float)
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
        parameters: &[Float],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> Vec<Float> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projection = self.project_local(parameters, mc_evaluator);
        let mut buffer: Vec<Float> = vec![0.0; n_events];
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
    pub fn project(&self, parameters: &[Float], mc_evaluator: Option<Evaluator>) -> Vec<Float> {
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
        parameters: &[Float],
        mc_evaluator: Option<Evaluator>,
    ) -> (Vec<Float>, Vec<DVector<Float>>) {
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
        let n_mc = self.accmc_evaluator.dataset.n_events() as Float;
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
        parameters: &[Float],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> (Vec<Float>, Vec<DVector<Float>>) {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let (local_projection, local_gradient_projection) =
            self.project_gradient_local(parameters, mc_evaluator);
        let mut projection_result: Vec<Float> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut projection_result, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }

        let flattened_local_gradient_projection = local_gradient_projection
            .iter()
            .flat_map(|g| g.data.as_vec().to_vec())
            .collect::<Vec<Float>>();
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
        parameters: &[Float],
        mc_evaluator: Option<Evaluator>,
    ) -> (Vec<Float>, Vec<DVector<Float>>) {
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
        parameters: &[Float],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<Vec<Float>, LadduError> {
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.resources.read().active.clone();
            mc_evaluator.isolate_many(names)?;
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_local(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events();
            #[cfg(feature = "rayon")]
            let output: Vec<Float> = result
                .par_iter()
                .zip(mc_dataset.events.par_iter())
                .map(|(l, e)| e.weight * l.re / n_mc as Float)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<Float> = result
                .iter()
                .zip(mc_dataset.events.iter())
                .map(|(l, e)| e.weight * l.re / n_mc as Float)
                .collect();
            mc_evaluator.resources.write().active = current_active_mc;
            Ok(output)
        } else {
            let current_active_data = self.data_evaluator.resources.read().active.clone();
            let current_active_accmc = self.accmc_evaluator.resources.read().active.clone();
            self.isolate_many(names)?;
            let mc_dataset = &self.accmc_evaluator.dataset;
            let result = self.accmc_evaluator.evaluate_local(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events();
            #[cfg(feature = "rayon")]
            let output: Vec<Float> = result
                .par_iter()
                .zip(mc_dataset.events.par_iter())
                .map(|(l, e)| e.weight * l.re / n_mc as Float)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<Float> = result
                .iter()
                .zip(mc_dataset.events.iter())
                .map(|(l, e)| e.weight * l.re / n_mc as Float)
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
        parameters: &[Float],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> Result<Vec<Float>, LadduError> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projection = self.project_with_local(parameters, names, mc_evaluator)?;
        let mut buffer: Vec<Float> = vec![0.0; n_events];
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
        parameters: &[Float],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<Vec<Float>, LadduError> {
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
        parameters: &[Float],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<(Vec<Float>, Vec<DVector<Float>>), LadduError> {
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.resources.read().active.clone();
            mc_evaluator.isolate_many(names)?;
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_local(parameters);
            let result_gradient = mc_evaluator.evaluate_gradient(parameters);
            let n_mc = self.accmc_evaluator.dataset.n_events() as Float;
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
            let n_mc = self.accmc_evaluator.dataset.n_events() as Float;
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
        parameters: &[Float],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> Result<(Vec<Float>, Vec<DVector<Float>>), LadduError> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let (local_projection, local_gradient_projection) =
            self.project_gradient_with_local(parameters, names, mc_evaluator)?;
        let mut projection_result: Vec<Float> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut projection_result, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }

        let flattened_local_gradient_projection = local_gradient_projection
            .iter()
            .flat_map(|g| g.data.as_vec().to_vec())
            .collect::<Vec<Float>>();
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
        parameters: &[Float],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> Result<(Vec<Float>, Vec<DVector<Float>>), LadduError> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_gradient_with_mpi(parameters, names, mc_evaluator, &world);
            }
        }
        self.project_gradient_with_local(parameters, names, mc_evaluator)
    }

    fn evaluate_local(&self, parameters: &[Float]) -> Float {
        let data_result = self.data_evaluator.evaluate_local(parameters);
        let mc_result = self.accmc_evaluator.evaluate_local(parameters);
        let n_mc = self.accmc_evaluator.dataset.n_events() as Float;
        #[cfg(feature = "rayon")]
        let data_term: Float = data_result
            .par_iter()
            .zip(self.data_evaluator.dataset.events.par_iter())
            .map(|(l, e)| e.weight * Float::ln(l.re))
            .parallel_sum_with_accumulator::<Klein<Float>>();
        #[cfg(feature = "rayon")]
        let mc_term: Float = mc_result
            .par_iter()
            .zip(self.accmc_evaluator.dataset.events.par_iter())
            .map(|(l, e)| e.weight * l.re)
            .parallel_sum_with_accumulator::<Klein<Float>>();
        #[cfg(not(feature = "rayon"))]
        let data_term: Float = data_result
            .iter()
            .zip(self.data_evaluator.dataset.events.iter())
            .map(|(l, e)| e.weight * Float::ln(l.re))
            .sum_with_accumulator::<Klein<Float>>();
        #[cfg(not(feature = "rayon"))]
        let mc_term: Float = mc_result
            .iter()
            .zip(self.accmc_evaluator.dataset.events.iter())
            .map(|(l, e)| e.weight * l.re)
            .sum_with_accumulator::<Klein<Float>>();
        -2.0 * (data_term - mc_term / n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_mpi(&self, parameters: &[Float], world: &SimpleCommunicator) -> Float {
        let local_evaluation = self.evaluate_local(parameters);
        let mut buffer: Vec<Float> = vec![0.0; world.size() as usize];
        world.all_gather_into(&local_evaluation, &mut buffer);
        buffer.iter().sum()
    }

    fn evaluate_gradient_local(&self, parameters: &[Float]) -> DVector<Float> {
        let data_resources = self.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let mc_resources = self.accmc_evaluator.resources.read();
        let mc_parameters = Parameters::new(parameters, &mc_resources.constants);
        let n_mc = self.accmc_evaluator.dataset.n_events() as Float;
        #[cfg(feature = "rayon")]
        let data_term: DVector<Float> = self
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
                                    Complex::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
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
            .collect::<Vec<DVector<Float>>>()
            .iter()
            .sum(); // TODO: replace with custom implementation of accurate crate's trait
        #[cfg(feature = "rayon")]
        let mc_term: DVector<Float> = self
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
                                    Complex::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
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
            .collect::<Vec<DVector<Float>>>()
            .iter()
            .sum();
        #[cfg(not(feature = "rayon"))]
        let data_term: DVector<Float> = self
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
                                    Complex::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
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
        let mc_term: DVector<Float> = self
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
                                    Complex::ZERO
                                }
                            })
                            .collect(),
                    ),
                    GradientValues(gradient_values),
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
        parameters: &[Float],
        world: &SimpleCommunicator,
    ) -> DVector<Float> {
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
            .sum::<DVector<Float>>()
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
    fn evaluate(&self, parameters: &[Float]) -> Float {
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
    fn evaluate_gradient(&self, parameters: &[Float]) -> DVector<Float> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_gradient_mpi(parameters, &world);
            }
        }
        self.evaluate_gradient_local(parameters)
    }
}

#[cfg(feature = "rayon")]
impl Function<ThreadPool, LadduError> for NLL {
    fn evaluate(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<Float, LadduError> {
        Ok(thread_pool.install(|| LikelihoodTerm::evaluate(self, parameters)))
    }
    fn gradient(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<DVector<Float>, LadduError> {
        Ok(thread_pool.install(|| LikelihoodTerm::evaluate_gradient(self, parameters)))
    }
}

#[cfg(not(feature = "rayon"))]
impl Function<(), LadduError> for NLL {
    fn evaluate(&self, parameters: &[Float], _user_data: &mut ()) -> Result<Float, LadduError> {
        Ok(LikelihoodTerm::evaluate(self, parameters))
    }
    fn gradient(
        &self,
        parameters: &[Float],
        _user_data: &mut (),
    ) -> Result<DVector<Float>, LadduError> {
        Ok(LikelihoodTerm::evaluate_gradient(self, parameters))
    }
}

pub(crate) struct LogLikelihood<'a>(&'a NLL);

#[cfg(feature = "rayon")]
impl Function<ThreadPool, LadduError> for LogLikelihood<'_> {
    fn evaluate(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<Float, LadduError> {
        Function::evaluate(self.0, parameters, thread_pool).map(|res| -res)
    }
    fn gradient(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<DVector<Float>, LadduError> {
        Function::gradient(self.0, parameters, thread_pool).map(|res| -res)
    }
}

#[cfg(not(feature = "rayon"))]
impl Function<(), LadduError> for LogLikelihood<'_> {
    fn evaluate(&self, parameters: &[Float], user_data: &mut ()) -> Result<Float, LadduError> {
        Function::evaluate(self.0, parameters, user_data).map(|res| -res)
    }
    fn gradient(
        &self,
        parameters: &[Float],
        user_data: &mut (),
    ) -> Result<DVector<Float>, LadduError> {
        Function::gradient(self.0, parameters, user_data).map(|res| -res)
    }
}

impl NLL {
    /// Minimizes the negative log-likelihood using the L-BFGS-B algorithm (by default), a limited-memory
    /// quasi-Newton minimizer which supports bounded optimization.
    pub fn minimize(
        &self,
        p0: &[Float],
        bounds: Option<Vec<(Float, Float)>>,
        options: Option<MinimizerOptions>,
    ) -> Result<Status, LadduError> {
        let options = options.unwrap_or_default();
        let mut m = Minimizer::new(options.algorithm, self.parameters().len())
            .with_bounds(bounds)
            .with_max_steps(options.max_steps);
        for observer in options.observers {
            m = m.with_observer(observer);
        }
        #[cfg(feature = "rayon")]
        {
            m.minimize(
                self,
                p0,
                &mut ThreadPoolBuilder::new()
                    .num_threads(options.threads)
                    .build()
                    .map_err(LadduError::from)?,
            )?;
        }
        #[cfg(not(feature = "rayon"))]
        {
            m.minimize(self, p0, &mut ())?;
        }
        Ok(m.status)
    }
    /// Perform Markov Chain Monte Carlo sampling on this [`NLL`]. By default, this uses the [`ESS`](`ganesh::algorithms::mcmc::ESS`) sampler.
    pub fn mcmc<T: AsRef<[DVector<Float>]>>(
        &self,
        p0: T,
        n_steps: usize,
        options: Option<MCMCOptions>,
        rng: Rng,
    ) -> Result<Ensemble, LadduError> {
        let options = options.unwrap_or(MCMCOptions::new_ess(
            [ESSMove::differential(0.9), ESSMove::gaussian(0.1)],
            rng,
        ));
        let mut m = Sampler::new(options.algorithm, p0.as_ref().to_vec());
        for observer in options.observers {
            m = m.with_observer(observer);
        }
        let func = LogLikelihood(self);
        #[cfg(feature = "rayon")]
        {
            m.sample(
                &func,
                &mut ThreadPoolBuilder::new()
                    .num_threads(options.threads)
                    .build()
                    .map_err(LadduError::from)?,
                n_steps,
            )?;
        }
        #[cfg(not(feature = "rayon"))]
        {
            m.sample(&func, &mut (), n_steps)?;
        }
        Ok(m.ensemble)
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
    /// Turn an ``NLL`` into a term that can be used by a ``LikelihoodManager``
    ///
    /// Returns
    /// -------
    /// term : LikelihoodTerm
    ///     The isolated NLL which can be used to build more complex models
    ///
    fn as_term(&self) -> PyLikelihoodTerm {
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
    /// Activates all Amplitudes in the JNLL
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
    fn evaluate(&self, parameters: Vec<Float>, threads: Option<usize>) -> PyResult<Float> {
        #[cfg(feature = "rayon")]
        {
            Ok(ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or_else(num_cpus::get))
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
        parameters: Vec<Float>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<Float>>> {
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
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
        parameters: Vec<Float>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<Float>>> {
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
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
        parameters: Vec<Float>,
        arg: &Bound<'_, PyAny>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<Float>>> {
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
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
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
    /// This method "runs the fit". Given an initial position `p0` and optional `bounds`, this
    /// method performs a minimization over the negative log-likelihood, optimizing the model
    /// over the stored signal data and Monte Carlo.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial parameters at the start of optimization
    /// bounds : list of tuple of float, optional
    ///     A list of lower and upper bound pairs for each parameter (use ``None`` for no bound)
    /// method : {'lbfgsb', 'nelder-mead', 'nelder_mead'}
    ///     The minimization algorithm to use (see additional parameters for fine-tuning)
    /// max_steps : int, default=4000
    ///     The maximum number of algorithm steps to perform
    /// debug : bool, default=False
    ///     Set to ``True`` to print out debugging information at each step
    /// verbose : bool, default=False
    ///     Set to ``True`` to print verbose information at each step
    /// show_step : bool, default=True
    ///     Include step number in verbose output
    /// show_x : bool, default=True
    ///     Include current best position in verbose output
    /// show_fx : bool, default=True
    ///     Include current best NLL in verbose output
    /// observers : Observer or list of Observers
    ///     Callback functions which are applied after every algorithm step
    /// tol_x_rel : float
    ///     The relative position tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_x_abs : float
    ///     The absolute position tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_f_rel : float
    ///     The relative function tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_f_abs : float
    ///     The absolute function tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_g_abs : float
    ///     The absolute gradient tolerance used by termination methods (default is the cube
    ///     root of machine epsilon)
    /// g_tolerance : float, default=1e-5
    ///     Another gradient tolerance used by termination methods (particularly L-BFGS-B)
    /// adaptive : bool, default=False
    ///     Use adaptive values for Nelder-Mead parameters
    /// alpha : float, optional
    ///     Overwrite the default :math:`\alpha` parameter in the Nelder-Mead algorithm
    /// beta : float, optional
    ///     Overwrite the default :math:`\beta` parameter in the Nelder-Mead algorithm
    /// gamma : float, optional
    ///     Overwrite the default :math:`\gamma` parameter in the Nelder-Mead algorithm
    /// delta : float, optional
    ///     Overwrite the default :math:`\delta` parameter in the Nelder-Mead algorithm
    /// simplex_expansion_method : {'greedy_minimization', 'greedy_expansion'}
    ///     The expansion method used by the Nelder-Mead algorithm
    /// nelder_mead_f_terminator : {'stddev', 'absolute', 'stddev', 'none'}
    ///     The function terminator used by the Nelder-Mead algorithm
    /// nelder_mead_x_terminator : {'singer', 'diameter', 'rowan', 'higham', 'none'}
    ///     The positional terminator used by the Nelder-Mead algorithm
    /// skip_hessian : bool, default = False
    ///     Skip calculation of the Hessian matrix (and parameter errors) at the termination of the
    ///     algorithm (use this when uncertainties are not needed/important)
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// Status
    ///     The status of the minimization algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    /// ValueError
    ///     If any kwargs are invalid
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="lbfgsb", max_steps=4000, debug=false, verbose=false, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn minimize(
        &self,
        p0: Vec<Float>,
        bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
        method: &str,
        max_steps: usize,
        debug: bool,
        verbose: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyStatus> {
        let bounds = bounds.map(|bounds_vec| {
            bounds_vec
                .iter()
                .map(|(opt_lb, opt_ub)| {
                    (
                        opt_lb.unwrap_or(Float::NEG_INFINITY),
                        opt_ub.unwrap_or(Float::INFINITY),
                    )
                })
                .collect()
        });
        let n_parameters = p0.len();
        let options =
            py_parse_minimizer_options(n_parameters, method, max_steps, debug, verbose, kwargs)?;
        let status = self.0.minimize(&p0, bounds, Some(options))?;
        Ok(PyStatus(status))
    }
    /// Run an MCMC algorithm on the free parameters of the NLL's model
    ///
    /// This method can be used to sample the underlying log-likelihood given an initial
    /// position for each walker `p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial parameters at the start of optimization
    /// n_steps : int,
    ///     The number of MCMC steps each walker should take
    /// method : {'ESS', 'AIES'}
    ///     The MCMC algorithm to use (see additional parameters for fine-tuning)
    /// debug : bool, default=False
    ///     Set to ``True`` to print out debugging information at each step
    /// verbose : bool, default=False
    ///     Set to ``True`` to print verbose information at each step
    /// seed : int,
    ///     The seed for the random number generator
    /// ess_moves : list of tuple
    ///     A list of moves for the ESS algorithm (see notes)
    /// aies_moves : list of tuple
    ///     A list of moves for the AIES algorithm (see notes)
    /// n_adaptive : int, default=100
    ///     Number of adaptive ESS steps to perform at the start of sampling
    /// mu : float, default=1.0
    ///     ESS adaptive parameter
    /// max_ess_steps : int, default=10000
    ///     The maximum number of slice expansions/contractions performed in the ESS algorithm
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// Ensemble
    ///     The resulting ensemble of walkers
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    /// ValueError
    ///     If any kwargs are invalid
    ///
    /// Notes
    /// -----
    /// Moves may be specified as tuples of ``(move name, usage weight)`` where the move name
    /// depends on the algorithm and the usage weight gives the proportion of time that move is
    /// used relative to the others in the list.
    ///
    /// For the Ensemble Slice Sampler (ESS) algorithm, valid move types are "differential" and
    /// "gaussian", and the default move set is ``[("differential", 0.9), ("gaussian", 0.1)]``.
    ///
    /// For the Affine Invariant Ensemble Sampler (AIES) algorithm, valid move types are
    /// "stretch" and "walk", and the default move set is ``[("stretch", 0.9), ("walk", 0.1)]``.
    ///
    /// For AIES, the "stretch" move can also be given with an adaptive parameter ``a``
    /// (default=``2``). To add a stretch move with a different value of ``a``, the "move name"
    /// can be instead given as a tuple ``(move name, a)``. For example, ``(("stretch", 2.2), 0.3)``
    /// creates a stretch move with ``a=2.2`` and usage weight of ``0.3``.
    ///
    /// Since MCMC methods are inclined to sample maxima rather than minima, the underlying
    /// function sign is automatically flipped when calling this method.
    ///
    #[pyo3(signature = (p0, n_steps, *, method="ESS", debug=false, verbose=false, seed=0, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc(
        &self,
        p0: Vec<Vec<Float>>,
        n_steps: usize,
        method: &str,
        debug: bool,
        verbose: bool,
        seed: u64,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyEnsemble> {
        let p0 = p0.into_iter().map(DVector::from_vec).collect::<Vec<_>>();
        let mut rng = Rng::new();
        rng.seed(seed);
        let options = py_parse_mcmc_options(method, debug, verbose, kwargs, rng.clone())?;
        let ensemble = self.0.mcmc(&p0, n_steps, Some(options), rng)?;
        Ok(PyEnsemble(ensemble))
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
struct LikelihoodValues(Vec<Float>);

#[derive(Debug)]
struct LikelihoodGradients(Vec<DVector<Float>>);

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
    fn evaluate(&self, likelihood_values: &LikelihoodValues) -> Float {
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
    ) -> DVector<Float> {
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
pub struct LikelihoodEvaluator {
    likelihood_manager: LikelihoodManager,
    likelihood_expression: LikelihoodExpression,
}

#[cfg(feature = "rayon")]
impl Function<ThreadPool, LadduError> for LikelihoodEvaluator {
    fn evaluate(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<Float, LadduError> {
        thread_pool.install(|| self.evaluate(parameters))
    }
    fn gradient(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<DVector<Float>, LadduError> {
        thread_pool.install(|| self.evaluate_gradient(parameters))
    }
}

#[cfg(not(feature = "rayon"))]
impl Function<(), LadduError> for LikelihoodEvaluator {
    fn evaluate(&self, parameters: &[Float], _user_data: &mut ()) -> Result<Float, LadduError> {
        self.evaluate(parameters)
    }
    fn gradient(
        &self,
        parameters: &[Float],
        _user_data: &mut (),
    ) -> Result<DVector<Float>, LadduError> {
        self.evaluate_gradient(parameters)
    }
}

pub(crate) struct NegativeLikelihoodEvaluator<'a>(&'a LikelihoodEvaluator);
#[cfg(feature = "rayon")]
impl Function<ThreadPool, LadduError> for NegativeLikelihoodEvaluator<'_> {
    fn evaluate(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<Float, LadduError> {
        Function::evaluate(self.0, parameters, thread_pool).map(|res| -res)
    }
    fn gradient(
        &self,
        parameters: &[Float],
        thread_pool: &mut ThreadPool,
    ) -> Result<DVector<Float>, LadduError> {
        Function::gradient(self.0, parameters, thread_pool).map(|res| -res)
    }
}

#[cfg(not(feature = "rayon"))]
impl Function<(), LadduError> for NegativeLikelihoodEvaluator<'_> {
    fn evaluate(&self, parameters: &[Float], user_data: &mut ()) -> Result<Float, LadduError> {
        Function::evaluate(self.0, parameters, user_data).map(|res| -res)
    }
    fn gradient(
        &self,
        parameters: &[Float],
        user_data: &mut (),
    ) -> Result<DVector<Float>, LadduError> {
        Function::gradient(self.0, parameters, user_data).map(|res| -res)
    }
}

impl LikelihoodEvaluator {
    /// The parameter names used in [`LikelihoodEvaluator::evaluate`]'s input in order.
    pub fn parameters(&self) -> Vec<String> {
        self.likelihood_manager.parameters()
    }
    /// A function that can be called to evaluate the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    pub fn evaluate(&self, parameters: &[Float]) -> Result<Float, LadduError> {
        let mut param_buffers: Vec<Vec<Float>> = self
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
        Ok(self.likelihood_expression.evaluate(&likelihood_values))
    }

    /// Evaluate the gradient of the stored [`LikelihoodExpression`] over the events in the [`Dataset`]
    /// stored by the [`LikelihoodEvaluator`] with the given values for free parameters.
    pub fn evaluate_gradient(&self, parameters: &[Float]) -> Result<DVector<Float>, LadduError> {
        let mut param_buffers: Vec<Vec<Float>> = self
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
        let mut gradient_buffers: Vec<DVector<Float>> = (0..self.likelihood_manager.terms.len())
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
        Ok(self
            .likelihood_expression
            .evaluate_gradient(&likelihood_values, &likelihood_gradients))
    }

    /// A function that can be called to minimize the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    ///
    /// See [`NLL::minimize`] for more details.
    pub fn minimize(
        &self,
        p0: &[Float],
        bounds: Option<Vec<(Float, Float)>>,
        options: Option<MinimizerOptions>,
    ) -> Result<Status, LadduError> {
        let options = options.unwrap_or_default();
        let mut m = Minimizer::new(options.algorithm, self.parameters().len())
            .with_bounds(bounds)
            .with_max_steps(options.max_steps);
        for observer in options.observers {
            m = m.with_observer(observer)
        }
        #[cfg(feature = "rayon")]
        {
            m.minimize(
                self,
                p0,
                &mut ThreadPoolBuilder::new()
                    .num_threads(options.threads)
                    .build()
                    .map_err(LadduError::from)?,
            )?;
        }
        #[cfg(not(feature = "rayon"))]
        {
            m.minimize(self, p0, &mut ())?;
        }
        Ok(m.status)
    }

    /// A function that can be called to perform Markov Chain Monte Carlo sampling
    /// of the sum/product of the [`LikelihoodTerm`]s
    /// contained by this [`LikelihoodEvaluator`].
    ///
    /// See [`NLL::mcmc`] for more details.
    pub fn mcmc<T: AsRef<[DVector<Float>]>>(
        &self,
        p0: T,
        n_steps: usize,
        options: Option<MCMCOptions>,
        rng: Rng,
    ) -> Result<Ensemble, LadduError> {
        let options = options.unwrap_or(MCMCOptions::new_ess(
            [ESSMove::differential(0.9), ESSMove::gaussian(0.1)],
            rng,
        ));
        let mut m = Sampler::new(options.algorithm, p0.as_ref().to_vec());
        for observer in options.observers {
            m = m.with_observer(observer);
        }
        let func = NegativeLikelihoodEvaluator(self);
        #[cfg(feature = "rayon")]
        {
            m.sample(
                &func,
                &mut ThreadPoolBuilder::new()
                    .num_threads(options.threads)
                    .build()
                    .map_err(LadduError::from)?,
                n_steps,
            )?;
        }
        #[cfg(not(feature = "rayon"))]
        {
            m.sample(&func, &mut (), n_steps)?;
        }
        Ok(m.ensemble)
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
    fn evaluate(&self, parameters: Vec<Float>, threads: Option<usize>) -> PyResult<Float> {
        #[cfg(feature = "rayon")]
        {
            Ok(ThreadPoolBuilder::new()
                .num_threads(threads.unwrap_or_else(num_cpus::get))
                .build()
                .map_err(LadduError::from)?
                .install(|| self.0.evaluate(&parameters))?)
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(self.0.evaluate(&parameters)?)
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
        parameters: Vec<Float>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<Float>>> {
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| self.0.evaluate_gradient(&parameters))?
                    .as_slice(),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray1::from_slice(
                py,
                self.0.evaluate_gradient(&parameters)?.as_slice(),
            ))
        }
    }

    /// Minimize all LikelihoodTerms with respect to the free parameters in the model
    ///
    /// This method "runs the fit". Given an initial position `p0` and optional `bounds`, this
    /// method performs a minimization over the tatal negative log-likelihood, optimizing the model
    /// over the stored signal data and Monte Carlo.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial parameters at the start of optimization
    /// bounds : list of tuple of float, optional
    ///     A list of lower and upper bound pairs for each parameter (use ``None`` for no bound)
    /// method : {'lbfgsb', 'nelder-mead', 'nelder_mead'}
    ///     The minimization algorithm to use (see additional parameters for fine-tuning)
    /// max_steps : int, default=4000
    ///     The maximum number of algorithm steps to perform
    /// debug : bool, default=False
    ///     Set to ``True`` to print out debugging information at each step
    /// verbose : bool, default=False
    ///     Set to ``True`` to print verbose information at each step
    /// show_step : bool, default=True
    ///     Include step number in verbose output
    /// show_x : bool, default=True
    ///     Include current best position in verbose output
    /// show_fx : bool, default=True
    ///     Include current best NLL in verbose output
    /// observers : Observer or list of Observers
    ///     Callback functions which are applied after every algorithm step
    /// tol_x_rel : float
    ///     The relative position tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_x_abs : float
    ///     The absolute position tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_f_rel : float
    ///     The relative function tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_f_abs : float
    ///     The absolute function tolerance used by termination methods (default is machine
    ///     epsilon)
    /// tol_g_abs : float
    ///     The absolute gradient tolerance used by termination methods (default is the cube
    ///     root of machine epsilon)
    /// g_tolerance : float, default=1e-5
    ///     Another gradient tolerance used by termination methods (particularly L-BFGS-B)
    /// adaptive : bool, default=False
    ///     Use adaptive values for Nelder-Mead parameters
    /// alpha : float, optional
    ///     Overwrite the default :math:`\alpha` parameter in the Nelder-Mead algorithm
    /// beta : float, optional
    ///     Overwrite the default :math:`\beta` parameter in the Nelder-Mead algorithm
    /// gamma : float, optional
    ///     Overwrite the default :math:`\gamma` parameter in the Nelder-Mead algorithm
    /// delta : float, optional
    ///     Overwrite the default :math:`\delta` parameter in the Nelder-Mead algorithm
    /// simplex_expansion_method : {'greedy_minimization', 'greedy_expansion'}
    ///     The expansion method used by the Nelder-Mead algorithm
    /// nelder_mead_f_terminator : {'stddev', 'absolute', 'stddev', 'none'}
    ///     The function terminator used by the Nelder-Mead algorithm
    /// nelder_mead_x_terminator : {'singer', 'diameter', 'rowan', 'higham', 'none'}
    ///     The positional terminator used by the Nelder-Mead algorithm
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// Status
    ///     The status of the minimization algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    /// ValueError
    ///     If any kwargs are invalid
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="lbfgsb", max_steps=4000, debug=false, verbose=false, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn minimize(
        &self,
        p0: Vec<Float>,
        bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
        method: &str,
        max_steps: usize,
        debug: bool,
        verbose: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyStatus> {
        let bounds = bounds.map(|bounds_vec| {
            bounds_vec
                .iter()
                .map(|(opt_lb, opt_ub)| {
                    (
                        opt_lb.unwrap_or(Float::NEG_INFINITY),
                        opt_ub.unwrap_or(Float::INFINITY),
                    )
                })
                .collect()
        });
        let n_parameters = p0.len();
        let options =
            py_parse_minimizer_options(n_parameters, method, max_steps, debug, verbose, kwargs)?;
        let status = self.0.minimize(&p0, bounds, Some(options))?;
        Ok(PyStatus(status))
    }

    /// Run an MCMC algorithm on the free parameters of the LikelihoodTerm's model
    ///
    /// This method can be used to sample the underlying log-likelihood given an initial
    /// position for each walker `p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like
    ///     The initial parameters at the start of optimization
    /// n_steps : int,
    ///     The number of MCMC steps each walker should take
    /// method : {'ESS', 'AIES'}
    ///     The MCMC algorithm to use (see additional parameters for fine-tuning)
    /// debug : bool, default=False
    ///     Set to ``True`` to print out debugging information at each step
    /// verbose : bool, default=False
    ///     Set to ``True`` to print verbose information at each step
    /// seed : int,
    ///     The seed for the random number generator
    /// ess_moves : list of tuple
    ///     A list of moves for the ESS algorithm (see notes)
    /// aies_moves : list of tuple
    ///     A list of moves for the AIES algorithm (see notes)
    /// n_adaptive : int, default=100
    ///     Number of adaptive ESS steps to perform at the start of sampling
    /// mu : float, default=1.0
    ///     ESS adaptive parameter
    /// max_ess_steps : int, default=10000
    ///     The maximum number of slice expansions/contractions performed in the ESS algorithm
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
    ///
    /// Returns
    /// -------
    /// Ensemble
    ///     The resulting ensemble of walkers
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    /// ValueError
    ///     If any kwargs are invalid
    ///
    /// Notes
    /// -----
    /// Moves may be specified as tuples of ``(move name, usage weight)`` where the move name
    /// depends on the algorithm and the usage weight gives the proportion of time that move is
    /// used relative to the others in the list.
    ///
    /// For the Ensemble Slice Sampler (ESS) algorithm, valid move types are "differential" and
    /// "gaussian", and the default move set is ``[("differential", 0.9), ("gaussian", 0.1)]``.
    ///
    /// For the Affine Invariant Ensemble Sampler (AIES) algorithm, valid move types are
    /// "stretch" and "walk", and the default move set is ``[("stretch", 0.9), ("walk", 0.1)]``.
    ///
    /// For AIES, the "stretch" move can also be given with an adaptive parameter ``a``
    /// (default=``2``). To add a stretch move with a different value of ``a``, the "move name"
    /// can be instead given as a tuple ``(move name, a)``. For example, ``(("stretch", 2.2), 0.3)``
    /// creates a stretch move with ``a=2.2`` and usage weight of ``0.3``.
    ///
    /// Since MCMC methods are inclined to sample maxima rather than minima, the underlying
    /// function sign is automatically flipped when calling this method.
    ///
    #[pyo3(signature = (p0, n_steps, *, method="ESS", debug=false, verbose=false, seed=0, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc(
        &self,
        p0: Vec<Vec<Float>>,
        n_steps: usize,
        method: &str,
        debug: bool,
        verbose: bool,
        seed: u64,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyEnsemble> {
        let p0 = p0.into_iter().map(DVector::from_vec).collect::<Vec<_>>();
        let mut rng = Rng::new();
        rng.seed(seed);
        let options = py_parse_mcmc_options(method, debug, verbose, kwargs, rng.clone())?;
        let ensemble = self.0.mcmc(&p0, n_steps, Some(options), rng)?;
        Ok(PyEnsemble(ensemble))
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
    fn evaluate(&self, parameters: &[Float]) -> Float {
        parameters[0]
    }

    fn evaluate_gradient(&self, _parameters: &[Float]) -> DVector<Float> {
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
