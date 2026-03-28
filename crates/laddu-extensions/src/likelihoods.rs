use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::RngSubsetExtension;
use accurate::{sum::Klein, traits::*};
use auto_ops::*;
use dyn_clone::DynClone;
use fastrand::Rng;
#[cfg(feature = "python")]
use laddu_core::ThreadPoolManager;
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
use crate::ganesh_ext::py_ganesh::{
    mcmc_settings_from_python, minimization_settings_from_python, PyMCMCSummary,
    PyMinimizationSummary,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyEvaluator, PyExpression},
    data::PyDataset,
};
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2};
#[cfg(feature = "python")]
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyList},
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

#[cfg(feature = "python")]
fn install_laddu_with_threads<R: Send>(
    threads: Option<usize>,
    op: impl FnOnce() -> LadduResult<R> + Send,
) -> LadduResult<R> {
    ThreadPoolManager::shared().install(threads, op)?
}

#[cfg(feature = "mpi")]
fn reduce_scalar(world: &SimpleCommunicator, value: f64) -> f64 {
    let mut reduced = 0.0;
    world.all_reduce_into(&value, &mut reduced, SystemOperation::sum());
    reduced
}

#[cfg(feature = "mpi")]
fn reduce_gradient(world: &SimpleCommunicator, gradient: &DVector<f64>) -> DVector<f64> {
    let mut reduced = vec![0.0; gradient.len()];
    world.all_reduce_into(gradient.as_slice(), &mut reduced, SystemOperation::sum());
    DVector::from_vec(reduced)
}

fn evaluate_weighted_expression_sum_local<F>(
    evaluator: &Evaluator,
    parameters: &[f64],
    value_map: F,
) -> f64
where
    F: Fn(Complex64) -> f64 + Copy + Send + Sync,
{
    let resources = evaluator.resources.read();
    let parameters = Parameters::new(parameters, &resources.constants);
    let amplitude_len = evaluator.amplitudes.len();
    let active_indices = resources.active_indices().to_vec();
    let slot_count = evaluator.expression_slot_count();
    let program_snapshot = evaluator.expression_value_program_snapshot();
    #[cfg(feature = "rayon")]
    {
        resources
            .caches
            .par_iter()
            .zip(evaluator.dataset.events_local().par_iter())
            .map_init(
                || {
                    (
                        vec![Complex64::ZERO; amplitude_len],
                        vec![Complex64::ZERO; slot_count],
                    )
                },
                |(amplitude_values, expr_slots), (cache, event)| {
                    for &amp_idx in &active_indices {
                        amplitude_values[amp_idx] =
                            evaluator.amplitudes[amp_idx].compute(&parameters, cache);
                    }
                    let l = if evaluator.uses_ir_interpreter_backend() {
                        evaluator
                            .evaluate_expression_value_with_scratch(amplitude_values, expr_slots)
                    } else {
                        evaluator.evaluate_expression_value_with_program_snapshot(
                            &program_snapshot,
                            amplitude_values,
                            expr_slots,
                        )
                    };
                    event.weight * value_map(l)
                },
            )
            .parallel_sum_with_accumulator::<Klein<f64>>()
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
        let mut expr_slots = vec![Complex64::ZERO; slot_count];
        resources
            .caches
            .iter()
            .zip(evaluator.dataset.events_local().iter())
            .map(|(cache, event)| {
                for &amp_idx in &active_indices {
                    amplitude_values[amp_idx] =
                        evaluator.amplitudes[amp_idx].compute(&parameters, cache);
                }
                let l = if evaluator.uses_ir_interpreter_backend() {
                    evaluator
                        .evaluate_expression_value_with_scratch(&amplitude_values, &mut expr_slots)
                } else {
                    evaluator.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        &amplitude_values,
                        &mut expr_slots,
                    )
                };
                event.weight * value_map(l)
            })
            .sum_with_accumulator::<Klein<f64>>()
    }
}

fn project_weights_and_gradients_local_from_evaluator(
    evaluator: &Evaluator,
    parameters: &[f64],
    n_mc: f64,
) -> (Vec<f64>, Vec<DVector<f64>>) {
    let resources = evaluator.resources.read();
    let parameters = Parameters::new(parameters, &resources.constants);
    let amplitude_len = evaluator.amplitudes.len();
    let grad_dim = parameters.len();
    let active_indices = resources.active_indices().to_vec();
    let active_mask = resources.active.clone();
    let slot_count = evaluator.expression_value_gradient_slot_count_public();

    #[cfg(feature = "rayon")]
    {
        let weighted = resources
            .caches
            .par_iter()
            .zip(evaluator.dataset.events_local().par_iter())
            .map_init(
                || {
                    (
                        vec![Complex64::ZERO; amplitude_len],
                        vec![DVector::zeros(grad_dim); amplitude_len],
                        vec![Complex64::ZERO; slot_count],
                        vec![DVector::zeros(grad_dim); slot_count],
                    )
                },
                |(amplitude_values, gradient_values, value_slots, gradient_slots),
                 (cache, event)| {
                    evaluator.fill_amplitude_values_and_gradients_public(
                        amplitude_values,
                        gradient_values,
                        &active_indices,
                        &active_mask,
                        &parameters,
                        cache,
                    );
                    let (value, gradient) = evaluator
                        .evaluate_expression_value_gradient_with_scratch(
                            amplitude_values,
                            gradient_values,
                            value_slots,
                            gradient_slots,
                        );
                    (
                        event.weight * value.re / n_mc,
                        gradient.map(|g| g.re).scale(event.weight / n_mc),
                    )
                },
            )
            .collect::<Vec<_>>();
        weighted.into_iter().unzip()
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
        let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
        let mut value_slots = vec![Complex64::ZERO; slot_count];
        let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
        resources
            .caches
            .iter()
            .zip(evaluator.dataset.events_local().iter())
            .map(|(cache, event)| {
                evaluator.fill_amplitude_values_and_gradients_public(
                    &mut amplitude_values,
                    &mut gradient_values,
                    &active_indices,
                    &active_mask,
                    &parameters,
                    cache,
                );
                let (value, gradient) = evaluator.evaluate_expression_value_gradient_with_scratch(
                    &amplitude_values,
                    &gradient_values,
                    &mut value_slots,
                    &mut gradient_slots,
                );
                (
                    event.weight * value.re / n_mc,
                    gradient.map(|g| g.re).scale(event.weight / n_mc),
                )
            })
            .unzip()
    }
}

#[cfg(feature = "rayon")]
fn sum_dvectors_parallel(
    iter: impl rayon::iter::ParallelIterator<Item = DVector<f64>>,
    len: usize,
) -> DVector<f64> {
    iter.reduce(
        || DVector::zeros(len),
        |mut accum, value| {
            accum += value;
            accum
        },
    )
}

#[cfg(feature = "rayon")]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GradientScratchKey {
    n_parameters: usize,
    n_amplitudes: usize,
    n_expression_slots: usize,
}

#[cfg(feature = "rayon")]
struct GradientScratchWorkspace {
    amplitude_values: Vec<Complex64>,
    gradient_values: Vec<DVector<Complex64>>,
    value_slots: Vec<Complex64>,
    gradient_slots: Vec<DVector<Complex64>>,
}

#[cfg(feature = "rayon")]
impl GradientScratchWorkspace {
    fn new(key: GradientScratchKey) -> Self {
        Self {
            amplitude_values: vec![Complex64::ZERO; key.n_amplitudes],
            gradient_values: vec![DVector::zeros(key.n_parameters); key.n_amplitudes],
            value_slots: vec![Complex64::ZERO; key.n_expression_slots],
            gradient_slots: vec![DVector::zeros(key.n_parameters); key.n_expression_slots],
        }
    }

    fn matches_key(&self, key: GradientScratchKey) -> bool {
        self.amplitude_values.len() == key.n_amplitudes
            && self.gradient_values.len() == key.n_amplitudes
            && self.value_slots.len() == key.n_expression_slots
            && self.gradient_slots.len() == key.n_expression_slots
            && self
                .gradient_values
                .iter()
                .all(|gradient| gradient.len() == key.n_parameters)
            && self
                .gradient_slots
                .iter()
                .all(|slot| slot.len() == key.n_parameters)
    }
}

#[cfg(feature = "rayon")]
struct GradientScratchLease {
    key: GradientScratchKey,
    workspace: Option<GradientScratchWorkspace>,
}

#[cfg(feature = "rayon")]
impl GradientScratchLease {
    fn workspace_mut(&mut self) -> &mut GradientScratchWorkspace {
        self.workspace
            .as_mut()
            .expect("gradient scratch workspace must be available while leased")
    }
}

#[cfg(feature = "rayon")]
impl Drop for GradientScratchLease {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            TLS_GRADIENT_SCRATCH_POOL.with(|pool| {
                pool.borrow_mut().insert(self.key, workspace);
            });
        }
    }
}

#[cfg(feature = "rayon")]
fn acquire_gradient_scratch(key: GradientScratchKey) -> GradientScratchLease {
    let mut workspace = TLS_GRADIENT_SCRATCH_POOL.with(|pool| {
        pool.borrow_mut()
            .remove(&key)
            .unwrap_or_else(|| GradientScratchWorkspace::new(key))
    });
    if !workspace.matches_key(key) {
        workspace = GradientScratchWorkspace::new(key);
    }
    GradientScratchLease {
        key,
        workspace: Some(workspace),
    }
}

#[cfg(feature = "rayon")]
thread_local! {
    static TLS_GRADIENT_SCRATCH_POOL: RefCell<HashMap<GradientScratchKey, GradientScratchWorkspace>> =
        RefCell::new(HashMap::new());
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
    n_mc: f64,
    parameter_manager: ParameterManager,
    projection_active_mask_cache: Arc<Mutex<HashMap<Vec<String>, Vec<bool>>>>,
}

impl NLL {
    /// Construct an [`NLL`] from an [`Expression`] and two [`Dataset`]s (data and Monte Carlo). This mirrors loading a model but starts from
    /// the expression directly. The number of Monte Carlo events used in the denominator of the
    /// normalization integral may also be specified (uses the weighted number of accepted Monte
    /// Carlo events if None is given).
    pub fn new(
        expression: &Expression,
        ds_data: &Arc<Dataset>,
        ds_accmc: &Arc<Dataset>,
        n_mc: Option<f64>,
    ) -> LadduResult<Box<Self>> {
        let data_evaluator = expression.load(ds_data)?;
        let accmc_evaluator = expression.load(ds_accmc)?;
        Ok(Self {
            parameter_manager: data_evaluator.parameter_manager().clone(),
            data_evaluator,
            n_mc: n_mc.unwrap_or(accmc_evaluator.dataset.n_events_weighted()),
            accmc_evaluator,
            projection_active_mask_cache: Arc::new(Mutex::new(HashMap::new())),
        }
        .into())
    }

    fn normalized_projection_key<T: AsRef<str>>(names: &[T]) -> Vec<String> {
        let mut key = names
            .iter()
            .map(|name| name.as_ref().to_string())
            .collect::<Vec<_>>();
        key.sort_unstable();
        key.dedup();
        key
    }

    fn get_or_build_projection_active_mask<T: AsRef<str>>(
        &self,
        names: &[T],
    ) -> LadduResult<Vec<bool>> {
        let key = Self::normalized_projection_key(names);
        if let Some(mask) = self.projection_active_mask_cache.lock().get(&key).cloned() {
            return Ok(mask);
        }

        let current_active_accmc = self.accmc_evaluator.active_mask();
        let isolate_result = self.accmc_evaluator.isolate_many_strict(names);
        let resolved_mask = if isolate_result.is_ok() {
            self.accmc_evaluator.active_mask()
        } else {
            Vec::new()
        };
        self.accmc_evaluator
            .set_active_mask(&current_active_accmc)?;
        isolate_result?;
        self.projection_active_mask_cache
            .lock()
            .insert(key, resolved_mask.clone());
        Ok(resolved_mask)
    }

    fn invalidate_projection_mask_cache(&self) {
        self.projection_active_mask_cache.lock().clear();
    }

    fn project_weights_subsets_from_masks_local(
        &self,
        evaluator: &Evaluator,
        resolved_masks: &[Vec<bool>],
        parameters: &[f64],
    ) -> Vec<Vec<f64>> {
        if resolved_masks.is_empty() {
            return Vec::new();
        }
        let resources = evaluator.resources.read();
        let parameters = Parameters::new(parameters, &resources.constants);
        let amplitude_len = evaluator.amplitudes.len();
        let slot_count = evaluator.expression_reference_value_slot_count();
        let subset_active_indices = resolved_masks
            .iter()
            .map(|mask| {
                mask.iter()
                    .enumerate()
                    .filter_map(|(index, &active)| if active { Some(index) } else { None })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let union_active_indices = {
            let mut union_mask = vec![false; amplitude_len];
            for mask in resolved_masks {
                for (index, &active) in mask.iter().enumerate() {
                    if active {
                        union_mask[index] = true;
                    }
                }
            }
            union_mask
                .iter()
                .enumerate()
                .filter_map(|(index, &active)| if active { Some(index) } else { None })
                .collect::<Vec<_>>()
        };

        let n_subsets = resolved_masks.len();
        let mut output = vec![Vec::with_capacity(resources.caches.len()); n_subsets];
        let mut union_amplitudes = vec![Complex64::ZERO; amplitude_len];
        let mut subset_amplitudes = vec![vec![Complex64::ZERO; amplitude_len]; n_subsets];
        let mut subset_expr_slots = vec![vec![Complex64::ZERO; slot_count]; n_subsets];
        let program_snapshot = evaluator.expression_reference_value_program_snapshot();
        for (cache, event) in resources
            .caches
            .iter()
            .zip(evaluator.dataset.events_local().iter())
        {
            for &amp_idx in &union_active_indices {
                union_amplitudes[amp_idx] =
                    evaluator.amplitudes[amp_idx].compute(&parameters, cache);
            }
            for (subset_index, active_indices) in subset_active_indices.iter().enumerate() {
                let amplitude_values = &mut subset_amplitudes[subset_index];
                for &amp_idx in active_indices {
                    amplitude_values[amp_idx] = union_amplitudes[amp_idx];
                }
                let value = if evaluator.uses_ir_interpreter_backend() {
                    evaluator.evaluate_expression_value_with_scratch(
                        amplitude_values,
                        &mut subset_expr_slots[subset_index],
                    )
                } else {
                    evaluator.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        amplitude_values,
                        &mut subset_expr_slots[subset_index],
                    )
                };
                output[subset_index].push(event.weight * value.re / self.n_mc);
            }
        }
        output
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
            n_mc: self.n_mc,
            projection_active_mask_cache: Arc::new(Mutex::new(HashMap::new())),
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
            n_mc: self.n_mc,
            projection_active_mask_cache: Arc::new(Mutex::new(HashMap::new())),
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
            n_mc: self.n_mc,
            projection_active_mask_cache: Arc::new(Mutex::new(HashMap::new())),
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
            n_mc: self.n_mc,
            projection_active_mask_cache: Arc::new(Mutex::new(HashMap::new())),
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
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate(&name);
        self.accmc_evaluator.activate(name);
    }
    /// Activate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name and return an error if it is missing.
    pub fn activate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_strict(&name)?;
        self.accmc_evaluator.activate_strict(name)?;
        Ok(())
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name, skipping missing entries.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_many(names);
        self.accmc_evaluator.activate_many(names);
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name and return an error if any are missing.
    pub fn activate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_many_strict(names)?;
        self.accmc_evaluator.activate_many_strict(names)?;
        Ok(())
    }
    /// Activate all registered [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s.
    pub fn activate_all(&self) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_all();
        self.accmc_evaluator.activate_all();
    }
    /// Dectivate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name, skipping missing entries.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate(&name);
        self.accmc_evaluator.deactivate(name);
    }
    /// Dectivate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name and return an error if it is missing.
    pub fn deactivate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_strict(&name)?;
        self.accmc_evaluator.deactivate_strict(name)?;
        Ok(())
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name, skipping missing entries.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_many(names);
        self.accmc_evaluator.deactivate_many(names);
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name and return an error if any are missing.
    pub fn deactivate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_many_strict(names)?;
        self.accmc_evaluator.deactivate_many_strict(names)?;
        Ok(())
    }
    /// Deactivate all registered [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s.
    pub fn deactivate_all(&self) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_all();
        self.accmc_evaluator.deactivate_all();
    }
    /// Isolate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name (deactivate the rest), skipping missing entries.
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.isolate(&name);
        self.accmc_evaluator.isolate(name);
    }
    /// Isolate an [`Amplitude`](`laddu_core::amplitudes::Amplitude`) by name (deactivate the rest) and return an error if it is missing.
    pub fn isolate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.isolate_strict(&name)?;
        self.accmc_evaluator.isolate_strict(name)?;
        Ok(())
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name (deactivate the rest), skipping missing entries.
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.isolate_many(names);
        self.accmc_evaluator.isolate_many(names);
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s by name (deactivate the rest) and return an error if any are missing.
    pub fn isolate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
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
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights`] instead.
    pub fn project_weights_local(
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
        #[cfg(feature = "rayon")]
        let output: Vec<f64> = result
            .par_iter()
            .zip(mc_dataset.events_local().par_iter())
            .map(|(l, e)| e.weight * l.re / self.n_mc)
            .collect();

        #[cfg(not(feature = "rayon"))]
        let output: Vec<f64> = result
            .iter()
            .zip(mc_dataset.events_local().iter())
            .map(|(l, e)| e.weight * l.re / self.n_mc)
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
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_weights_mpi(
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
        let local_projection = self.project_weights_local(parameters, mc_evaluator)?;
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required because projection returns per-event global outputs.
            // Use all-reduce only for aggregate scalar/vector reductions.
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
    pub fn project_weights(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_mpi(parameters, mc_evaluator, &world);
            }
        }
        self.project_weights_local(parameters, mc_evaluator)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_and_gradients`] instead.
    pub fn project_weights_and_gradients_local(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if let Some(mc_evaluator) = mc_evaluator {
            Ok(project_weights_and_gradients_local_from_evaluator(
                &mc_evaluator,
                parameters,
                self.n_mc,
            ))
        } else {
            Ok(project_weights_and_gradients_local_from_evaluator(
                &self.accmc_evaluator,
                parameters,
                self.n_mc,
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
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_and_gradients`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_weights_and_gradients_mpi(
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
            self.project_weights_and_gradients_local(parameters, mc_evaluator)?;
        let mut projection_result: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required because projection-gradient returns per-event global outputs.
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
        // NOTE: gather is required because projection-gradient returns full per-event gradients.
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
    pub fn project_weights_and_gradients(
        &self,
        parameters: &[f64],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_and_gradients_mpi(parameters, mc_evaluator, &world);
            }
        }
        self.project_weights_and_gradients_local(parameters, mc_evaluator)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    pub fn project_weights_subset_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.active_mask();
            let isolate_result = mc_evaluator.isolate_many_strict(names);
            if let Err(err) = isolate_result {
                mc_evaluator.set_active_mask(&current_active_mc)?;
                return Err(err);
            }
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_local(parameters);
            #[cfg(feature = "rayon")]
            let output: Vec<f64> = result
                .par_iter()
                .zip(mc_dataset.events_local().par_iter())
                .map(|(l, e)| e.weight * l.re / self.n_mc)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<f64> = result
                .iter()
                .zip(mc_dataset.events_local().iter())
                .map(|(l, e)| e.weight * l.re / self.n_mc)
                .collect();
            mc_evaluator.set_active_mask(&current_active_mc)?;
            Ok(output)
        } else {
            let resolved_mask = self.get_or_build_projection_active_mask(names)?;
            let mc_dataset = &self.accmc_evaluator.dataset;
            let result = self
                .accmc_evaluator
                .evaluate_local_with_active_mask(parameters, &resolved_mask)?;
            #[cfg(feature = "rayon")]
            let output: Vec<f64> = result
                .par_iter()
                .zip(mc_dataset.events_local().par_iter())
                .map(|(l, e)| e.weight * l.re / self.n_mc)
                .collect();
            #[cfg(not(feature = "rayon"))]
            let output: Vec<f64> = result
                .iter()
                .zip(mc_dataset.events_local().iter())
                .map(|(l, e)| e.weight * l.re / self.n_mc)
                .collect();
            Ok(output)
        }
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_weights_subset_mpi<T: AsRef<str>>(
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
        let local_projection =
            self.project_weights_subset_local(parameters, names, mc_evaluator)?;
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required because projection returns per-event global outputs.
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }
        Ok(buffer)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
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
    pub fn project_weights_subset<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_subset_mpi(parameters, names, mc_evaluator, &world);
            }
        }
        self.project_weights_subset_local(parameters, names, mc_evaluator)
    }

    /// Project the stored model over multiple isolated amplitude subsets (non-MPI version).
    pub fn project_weights_subsets_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<Vec<f64>>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if subsets.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.active_mask();
            let mut resolved_masks = Vec::with_capacity(subsets.len());
            for names in subsets {
                let isolate_result = mc_evaluator.isolate_many_strict(names);
                if let Err(err) = isolate_result {
                    mc_evaluator.set_active_mask(&current_active_mc)?;
                    return Err(err);
                }
                resolved_masks.push(mc_evaluator.active_mask());
            }
            mc_evaluator.set_active_mask(&current_active_mc)?;
            Ok(self.project_weights_subsets_from_masks_local(
                mc_evaluator,
                &resolved_masks,
                parameters,
            ))
        } else {
            let mut resolved_masks = Vec::with_capacity(subsets.len());
            for names in subsets {
                resolved_masks.push(self.get_or_build_projection_active_mask(names)?);
            }
            Ok(self.project_weights_subsets_from_masks_local(
                &self.accmc_evaluator,
                &resolved_masks,
                parameters,
            ))
        }
    }

    /// Project the stored model over multiple isolated amplitude subsets (MPI-compatible version).
    #[cfg(feature = "mpi")]
    pub fn project_weights_subsets_mpi<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> LadduResult<Vec<Vec<f64>>> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projections =
            self.project_weights_subsets_local(parameters, subsets, mc_evaluator)?;
        let (counts, displs) = world.get_counts_displs(n_events);
        let mut gathered = Vec::with_capacity(local_projections.len());
        for local_projection in local_projections {
            let mut buffer = vec![0.0; n_events];
            {
                let mut partitioned_buffer =
                    PartitionMut::new(&mut buffer, counts.clone(), displs.clone());
                world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
            }
            gathered.push(buffer);
        }
        Ok(gathered)
    }

    /// Project the stored model over multiple isolated amplitude subsets.
    pub fn project_weights_subsets<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<Vec<f64>>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_subsets_mpi(parameters, subsets, mc_evaluator, &world);
            }
        }
        self.project_weights_subsets_local(parameters, subsets, mc_evaluator)
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights_and_gradients`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    pub fn project_weights_and_gradients_subset_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if let Some(mc_evaluator) = &mc_evaluator {
            let current_active_mc = mc_evaluator.active_mask();
            let isolate_result = mc_evaluator.isolate_many_strict(names);
            if let Err(err) = isolate_result {
                mc_evaluator.set_active_mask(&current_active_mc)?;
                return Err(err);
            }
            let mc_dataset = mc_evaluator.dataset.clone();
            let result = mc_evaluator.evaluate_with_gradient_local(parameters);
            #[cfg(feature = "rayon")]
            let (res, res_gradient) = {
                (
                    result
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|((l, _), e)| e.weight * l.re / self.n_mc)
                        .collect(),
                    result
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|((_, grad_l), e)| grad_l.map(|g| g.re).scale(e.weight / self.n_mc))
                        .collect(),
                )
            };
            #[cfg(not(feature = "rayon"))]
            let (res, res_gradient) = {
                (
                    result
                        .iter()
                        .zip(mc_dataset.events_local().iter())
                        .map(|((l, _), e)| e.weight * l.re / self.n_mc)
                        .collect(),
                    result
                        .iter()
                        .zip(mc_dataset.events_local().iter())
                        .map(|((_, grad_l), e)| grad_l.map(|g| g.re).scale(e.weight / self.n_mc))
                        .collect(),
                )
            };
            mc_evaluator.set_active_mask(&current_active_mc)?;
            Ok((res, res_gradient))
        } else {
            let resolved_mask = self.get_or_build_projection_active_mask(names)?;
            let mc_dataset = &self.accmc_evaluator.dataset;
            let result = self
                .accmc_evaluator
                .evaluate_with_gradient_local_with_active_mask(parameters, &resolved_mask)?;
            #[cfg(feature = "rayon")]
            let (res, res_gradient) = {
                (
                    result
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|((l, _), e)| e.weight * l.re / self.n_mc)
                        .collect(),
                    result
                        .par_iter()
                        .zip(mc_dataset.events_local().par_iter())
                        .map(|((_, grad_l), e)| grad_l.map(|g| g.re).scale(e.weight / self.n_mc))
                        .collect(),
                )
            };
            #[cfg(not(feature = "rayon"))]
            let (res, res_gradient) = {
                (
                    result
                        .iter()
                        .zip(mc_dataset.events_local().iter())
                        .map(|((l, _), e)| e.weight * l.re / self.n_mc)
                        .collect(),
                    result
                        .iter()
                        .zip(mc_dataset.events_local().iter())
                        .map(|((_, grad_l), e)| grad_l.map(|g| g.re).scale(e.weight / self.n_mc))
                        .collect(),
                )
            };
            Ok((res, res_gradient))
        }
    }

    /// Project the stored [`Model`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights_and_gradients`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitudes::Amplitude`)s
    /// by name, but returns the [`NLL`] to its prior state after calculation (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    #[cfg(feature = "mpi")]
    pub fn project_weights_and_gradients_subset_mpi<T: AsRef<str>>(
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
            self.project_weights_and_gradients_subset_local(parameters, names, mc_evaluator)?;
        let mut projection_result: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required because projection-gradient returns per-event global outputs.
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
        // NOTE: gather is required because projection-gradient returns full per-event gradients.
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
    /// Monte-Carlo event. This method differs from the standard [`NLL::project_weights_and_gradients`] in that it first
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
    pub fn project_weights_and_gradients_subset<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_and_gradients_subset_mpi(
                    parameters,
                    names,
                    mc_evaluator,
                    &world,
                );
            }
        }
        self.project_weights_and_gradients_subset_local(parameters, names, mc_evaluator)
    }

    fn evaluate_local(&self, parameters: &[f64]) -> f64 {
        let data_term =
            evaluate_weighted_expression_sum_local(&self.data_evaluator, parameters, |l| {
                f64::ln(l.re)
            });
        let mc_term = self
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(parameters);
        -2.0 * (data_term - mc_term / self.n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_mpi(&self, parameters: &[f64], world: &SimpleCommunicator) -> f64 {
        let data_term_local =
            evaluate_weighted_expression_sum_local(&self.data_evaluator, parameters, |l| {
                f64::ln(l.re)
            });
        let data_term = reduce_scalar(world, data_term_local);
        let mc_term = self
            .accmc_evaluator
            .evaluate_weighted_value_sum_mpi(parameters, world);
        -2.0 * (data_term - mc_term / self.n_mc)
    }

    #[cfg(feature = "mpi")]
    /// Evaluate the NLL value across all ranks.
    pub fn evaluate_mpi_value(&self, parameters: &[f64], world: &SimpleCommunicator) -> f64 {
        self.evaluate_mpi(parameters, world)
    }

    fn evaluate_data_gradient_term_local(&self, parameters: &[f64]) -> DVector<f64> {
        let data_resources = self.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let n_parameters = parameters.len();
        #[cfg(feature = "rayon")]
        let data_scratch_key = GradientScratchKey {
            n_parameters,
            n_amplitudes: self.data_evaluator.amplitudes.len(),
            n_expression_slots: self.data_evaluator.expression_slot_count(),
        };
        #[cfg(feature = "rayon")]
        let data_term: DVector<f64> = sum_dvectors_parallel(
            self.data_evaluator
                .dataset
                .events_local()
                .par_iter()
                .zip(data_resources.caches.par_iter())
                .map_init(
                    || acquire_gradient_scratch(data_scratch_key),
                    |scratch, (event, cache)| {
                        let workspace = scratch.workspace_mut();
                        let amp_vals = &mut workspace.amplitude_values;
                        let grad_vals = &mut workspace.gradient_values;
                        for (idx, amp) in self.data_evaluator.amplitudes.iter().enumerate() {
                            if data_resources.active[idx] {
                                grad_vals[idx].fill(Complex64::ZERO);
                                amp.compute_gradient(&data_parameters, cache, &mut grad_vals[idx]);
                                amp_vals[idx] = amp.compute(&data_parameters, cache);
                            } else {
                                grad_vals[idx].fill(Complex64::ZERO);
                                amp_vals[idx] = Complex64::ZERO;
                            }
                        }
                        let (value, gradient) = self
                            .data_evaluator
                            .evaluate_expression_value_gradient_with_scratch(
                                amp_vals,
                                grad_vals,
                                &mut workspace.value_slots,
                                &mut workspace.gradient_slots,
                            );
                        (event.weight, value, gradient)
                    },
                )
                .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re)),
            n_parameters,
        );
        #[cfg(not(feature = "rayon"))]
        let data_term: DVector<f64> = {
            let mut amp_vals = vec![Complex64::ZERO; self.data_evaluator.amplitudes.len()];
            let mut grad_vals =
                vec![DVector::zeros(parameters.len()); self.data_evaluator.amplitudes.len()];
            let mut value_slots =
                vec![Complex64::ZERO; self.data_evaluator.expression_slot_count()];
            let mut gradient_slots =
                vec![DVector::zeros(parameters.len()); self.data_evaluator.expression_slot_count()];
            self.data_evaluator
                .dataset
                .events_local()
                .iter()
                .zip(data_resources.caches.iter())
                .map(|(event, cache)| {
                    for (idx, amp) in self.data_evaluator.amplitudes.iter().enumerate() {
                        if data_resources.active[idx] {
                            grad_vals[idx].fill(Complex64::ZERO);
                            amp.compute_gradient(&data_parameters, cache, &mut grad_vals[idx]);
                            amp_vals[idx] = amp.compute(&data_parameters, cache);
                        } else {
                            grad_vals[idx].fill(Complex64::ZERO);
                            amp_vals[idx] = Complex64::ZERO;
                        }
                    }
                    let (value, gradient) = self
                        .data_evaluator
                        .evaluate_expression_value_gradient_with_scratch(
                            &amp_vals,
                            &grad_vals,
                            &mut value_slots,
                            &mut gradient_slots,
                        );
                    (event.weight, value, gradient)
                })
                .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
                .sum()
        };
        data_term
    }

    fn evaluate_gradient_local(&self, parameters: &[f64]) -> DVector<f64> {
        let data_term = self.evaluate_data_gradient_term_local(parameters);
        let mc_term = self
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(parameters);
        -2.0 * (data_term - mc_term / self.n_mc)
    }

    #[cfg(feature = "mpi")]
    fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> DVector<f64> {
        let data_term_local = self.evaluate_data_gradient_term_local(parameters);
        let data_term = reduce_gradient(world, &data_term_local);
        let mc_term = self
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_mpi(parameters, world);
        -2.0 * (data_term - mc_term / self.n_mc)
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

    fn evaluate_data_term_local(&self, parameters: &[f64], indices: &[usize]) -> f64 {
        let data_result = self
            .nll
            .data_evaluator
            .evaluate_batch_local(parameters, indices);
        #[cfg(feature = "rayon")]
        {
            indices
                .par_iter()
                .zip(data_result.par_iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.events_local()[i];
                    e.weight * l.re.ln()
                })
                .parallel_sum_with_accumulator::<Klein<f64>>()
        }
        #[cfg(not(feature = "rayon"))]
        {
            indices
                .iter()
                .zip(data_result.iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.events_local()[i];
                    e.weight * l.re.ln()
                })
                .sum_with_accumulator::<Klein<f64>>()
        }
    }

    fn evaluate_local(&self, parameters: &[f64], indices: &[usize], n_data_batch: f64) -> f64 {
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term = self.evaluate_data_term_local(parameters, indices);
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(parameters);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc)
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
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term_local = self.evaluate_data_term_local(parameters, &locals);
        let n_data_batch = reduce_scalar(world, n_data_batch_local);
        let data_term = reduce_scalar(world, data_term_local);
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_mpi(parameters, world);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc)
    }

    fn evaluate_data_gradient_term_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> DVector<f64> {
        let data_resources = self.nll.data_evaluator.resources.read();
        let data_parameters = Parameters::new(parameters, &data_resources.constants);
        let n_parameters = parameters.len();
        #[cfg(feature = "rayon")]
        let data_scratch_key = GradientScratchKey {
            n_parameters,
            n_amplitudes: self.nll.data_evaluator.amplitudes.len(),
            n_expression_slots: self.nll.data_evaluator.expression_slot_count(),
        };
        #[cfg(feature = "rayon")]
        let data_term: DVector<f64> = sum_dvectors_parallel(
            indices
                .par_iter()
                .map_init(
                    || acquire_gradient_scratch(data_scratch_key),
                    |scratch, &idx| {
                        let workspace = scratch.workspace_mut();
                        let amp_vals = &mut workspace.amplitude_values;
                        let grad_vals = &mut workspace.gradient_values;
                        let event = &self.nll.data_evaluator.dataset.events_local()[idx];
                        let cache = &data_resources.caches[idx];
                        for (amp_idx, amp) in self.nll.data_evaluator.amplitudes.iter().enumerate()
                        {
                            if data_resources.active[amp_idx] {
                                grad_vals[amp_idx].fill(Complex64::ZERO);
                                amp.compute_gradient(
                                    &data_parameters,
                                    cache,
                                    &mut grad_vals[amp_idx],
                                );
                                amp_vals[amp_idx] = amp.compute(&data_parameters, cache);
                            } else {
                                grad_vals[amp_idx].fill(Complex64::ZERO);
                                amp_vals[amp_idx] = Complex64::ZERO;
                            }
                        }
                        let (value, gradient) = self
                            .nll
                            .data_evaluator
                            .evaluate_expression_value_gradient_with_scratch(
                                amp_vals,
                                grad_vals,
                                &mut workspace.value_slots,
                                &mut workspace.gradient_slots,
                            );
                        (event.weight, value, gradient)
                    },
                )
                .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re)),
            n_parameters,
        );
        #[cfg(not(feature = "rayon"))]
        let data_term: DVector<f64> = {
            let mut amp_vals = vec![Complex64::ZERO; self.nll.data_evaluator.amplitudes.len()];
            let mut grad_vals =
                vec![DVector::zeros(parameters.len()); self.nll.data_evaluator.amplitudes.len()];
            let mut value_slots =
                vec![Complex64::ZERO; self.nll.data_evaluator.expression_slot_count()];
            let mut gradient_slots = vec![
                DVector::zeros(parameters.len());
                self.nll.data_evaluator.expression_slot_count()
            ];
            indices
                .iter()
                .map(|&idx| {
                    let event = &self.nll.data_evaluator.dataset.events_local()[idx];
                    let cache = &data_resources.caches[idx];
                    for (amp_idx, amp) in self.nll.data_evaluator.amplitudes.iter().enumerate() {
                        if data_resources.active[amp_idx] {
                            grad_vals[amp_idx].fill(Complex64::ZERO);
                            amp.compute_gradient(&data_parameters, cache, &mut grad_vals[amp_idx]);
                            amp_vals[amp_idx] = amp.compute(&data_parameters, cache);
                        } else {
                            grad_vals[amp_idx].fill(Complex64::ZERO);
                            amp_vals[amp_idx] = Complex64::ZERO;
                        }
                    }
                    let (value, gradient) = self
                        .nll
                        .data_evaluator
                        .evaluate_expression_value_gradient_with_scratch(
                            &amp_vals,
                            &grad_vals,
                            &mut value_slots,
                            &mut gradient_slots,
                        );
                    (event.weight, value, gradient)
                })
                .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
                .sum()
        };
        data_term
    }

    fn evaluate_gradient_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
        n_data_batch: f64,
    ) -> DVector<f64> {
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term = self.evaluate_data_gradient_term_local(parameters, indices);
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(parameters);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc)
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
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term_local = self.evaluate_data_gradient_term_local(parameters, &locals);
        let n_data_batch = reduce_scalar(world, n_data_batch_local);
        let data_term = reduce_gradient(world, &data_term_local);
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_mpi(parameters, world);
        -2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc)
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
/// n_mc : float, optional
///     The number of Monte Carlo events used in the denominator of the normalization integral
///     (uses the weighted number of accepted Monte Carlo events if None is given)
///
#[cfg(feature = "python")]
#[pyclass(name = "NLL", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyNLL(pub Box<NLL>);

#[cfg(feature = "python")]
#[pymethods]
impl PyNLL {
    #[new]
    #[pyo3(signature = (expression, ds_data, ds_accmc, *, n_mc=None))]
    fn new(
        expression: &PyExpression,
        ds_data: &PyDataset,
        ds_accmc: &PyDataset,
        n_mc: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(NLL::new(
            &expression.0,
            &ds_data.0,
            &ds_accmc.0,
            n_mc,
        )?))
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
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
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
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        install_laddu_with_threads(threads, || {
            LikelihoodTerm::evaluate(self.0.as_ref(), &parameters)
        })
        .map_err(PyErr::from)
    }
    /// Evaluate the gradient of the negative log-likelihood over the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
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
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        let gradient = install_laddu_with_threads(threads, || {
            LikelihoodTerm::evaluate_gradient(self.0.as_ref(), &parameters)
        })?;
        Ok(PyArray1::from_slice(py, gradient.as_slice()))
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
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
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
    fn project_weights<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        let mc_evaluator = mc_evaluator.map(|pyeval| pyeval.0.clone());
        let projection = install_laddu_with_threads(threads, || {
            self.0.project_weights(&parameters, mc_evaluator.clone())
        })?;
        Ok(PyArray1::from_slice(py, projection.as_slice()))
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
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
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
    fn project_weights_subset<'py>(
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
        let mc_evaluator = mc_evaluator.map(|pyeval| pyeval.0.clone());
        let projection = install_laddu_with_threads(threads, || {
            self.0
                .project_weights_subset(&parameters, &names, mc_evaluator.clone())
        })?;
        Ok(PyArray1::from_slice(py, projection.as_slice()))
    }

    /// Project the model over the Monte Carlo dataset for multiple isolated amplitude subsets.
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// subsets : list of list of str
    ///     Each inner list is an isolated amplitude subset
    /// mc_evaluator: Evaluator, optional
    ///     Project using the given Evaluator or use the stored ``accmc`` if None
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     2D array of shape ``(len(subsets), n_events)``
    #[pyo3(signature = (parameters, subsets, *, mc_evaluator = None, threads=None))]
    fn project_weights_subsets<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        subsets: Vec<Vec<String>>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        let mc_evaluator = mc_evaluator.map(|pyeval| pyeval.0.clone());
        let projection = install_laddu_with_threads(threads, || {
            self.0
                .project_weights_subsets(&parameters, &subsets, mc_evaluator.clone())
        })?;
        Ok(PyArray2::from_vec2(py, &projection).map_err(LadduError::NumpyError)?)
    }

    /// Project the model and gradients over the Monte Carlo dataset while isolating selected terms.
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
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// tuple
    ///     ``(weights, gradients)`` where ``weights`` has shape ``(n_events,)`` and
    ///     ``gradients`` has shape ``(n_events, n_parameters)``
    #[pyo3(signature = (parameters, arg, *, mc_evaluator = None, threads=None))]
    fn project_weights_and_gradients_subset<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        arg: &Bound<'_, PyAny>,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
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
        let mc_evaluator = mc_evaluator.map(|pyeval| pyeval.0.clone());
        let (weights, gradients) = install_laddu_with_threads(threads, || {
            self.0
                .project_weights_and_gradients_subset(&parameters, &names, mc_evaluator.clone())
        })?;
        let gradients = gradients
            .iter()
            .map(|gradient| gradient.as_slice().to_vec())
            .collect::<Vec<_>>();
        Ok((
            PyArray1::from_slice(py, weights.as_slice()),
            PyArray2::from_vec2(py, &gradients).map_err(LadduError::NumpyError)?,
        ))
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
    /// settings : LBFGSBSettings or AdamSettings or NelderMeadSettings or PSOSettings, optional
    ///     Typed settings for the selected minimization algorithm. The settings type must
    ///     match ``method``. For ``pso``, explicit settings are required.
    /// observers : MinimizerObserver or list of MinimizerObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MinimizerTerminator or list of MinimizerTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
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
    /// ``settings`` is a typed configuration object rather than a dictionary. Use
    /// ``LBFGSBSettings`` with ``method="lbfgsb"``, ``AdamSettings`` with ``method="adam"``,
    /// ``NelderMeadSettings`` with ``method="nelder-mead"``, and ``PSOSettings`` with
    /// ``method="pso"``.
    ///
    /// Nested typed helpers are also explicit:
    ///
    /// - ``LBFGSBSettings(line_search=LineSearchConfig.morethuente(...))``
    /// - ``NelderMeadSettings(simplex=SimplexConfig.orthogonal(...))``
    /// - ``PSOSettings(SwarmInitializerConfig.random_in_limits(...), ...)``
    ///
    /// ``PSOSettings`` is required for ``method="pso"`` because the swarm initializer must be
    /// provided explicitly.
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
    fn minimize(
        &self,
        p0: Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMinimizationSummary> {
        validate_free_parameter_len(p0.len(), self.0.n_free())?;
        let result = self.0.minimize(minimization_settings_from_python(
            &p0,
            bounds,
            method,
            settings.as_ref(),
            observers,
            terminators,
            max_steps,
            debug,
            threads,
        )?)?;
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
    /// settings : AIESSettings or ESSSettings, optional
    ///     Typed settings for the selected sampler. The settings type must match ``method``.
    /// observers : MCMCObserver or list of MCMCObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MCMCTerminator or list of MCMCTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
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
    /// ``settings`` is a typed configuration object rather than a dictionary. Use
    /// ``AIESSettings`` with ``method="aies"`` and ``ESSSettings`` with ``method="ess"``.
    ///
    /// Sampler moves are also declared explicitly:
    ///
    /// - ``AIESSettings(moves=[AIESMoveConfig.stretch(...), AIESMoveConfig.walk(...)])``
    /// - ``ESSSettings(moves=[ESSMoveConfig.differential(...), ESSMoveConfig.global_(...)])``
    ///
    /// References
    /// ----------
    /// Goodman, J. & Weare, J. (2010). *Ensemble samplers with affine invariance*. CAMCoS 5(1), 65–80. <https://doi.org/10.2140/camcos.2010.5.65>
    ///
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat Comput 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="aies".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc(
        &self,
        p0: Vec<Vec<f64>>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMCMCSummary> {
        validate_mcmc_parameter_len(&p0, self.0.n_free())?;
        let result = self.0.mcmc(mcmc_settings_from_python(
            &p0.into_iter().map(DVector::from_vec).collect(),
            bounds,
            method,
            settings.as_ref(),
            observers,
            terminators,
            max_steps,
            debug,
            threads,
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
#[pyclass(name = "StochasticNLL", module = "laddu", skip_from_py_object)]
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
    /// settings : LBFGSBSettings or AdamSettings or NelderMeadSettings or PSOSettings, optional
    ///     Typed settings for the selected minimization algorithm. The settings type must
    ///     match ``method``. For ``pso``, explicit settings are required.
    /// observers : MinimizerObserver or list of MinimizerObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MinimizerTerminator or list of MinimizerTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
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
    /// ``settings`` is a typed configuration object rather than a dictionary. Use
    /// ``LBFGSBSettings`` with ``method="lbfgsb"``, ``AdamSettings`` with ``method="adam"``,
    /// ``NelderMeadSettings`` with ``method="nelder-mead"``, and ``PSOSettings`` with
    /// ``method="pso"``.
    ///
    /// Nested typed helpers are also explicit:
    ///
    /// - ``LBFGSBSettings(line_search=LineSearchConfig.morethuente(...))``
    /// - ``NelderMeadSettings(simplex=SimplexConfig.orthogonal(...))``
    /// - ``PSOSettings(SwarmInitializerConfig.random_in_limits(...), ...)``
    ///
    /// ``PSOSettings`` is required for ``method="pso"`` because the swarm initializer must be
    /// provided explicitly.
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
    fn minimize(
        &self,
        p0: Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMinimizationSummary> {
        validate_free_parameter_len(p0.len(), self.0.n_free())?;
        let result = self.0.minimize(minimization_settings_from_python(
            &p0,
            bounds,
            method,
            settings.as_ref(),
            observers,
            terminators,
            max_steps,
            debug,
            threads,
        )?)?;
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
    /// settings : AIESSettings or ESSSettings, optional
    ///     Typed settings for the selected sampler. The settings type must match ``method``.
    /// observers : MCMCObserver or list of MCMCObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MCMCTerminator or list of MCMCTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
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
    /// ``settings`` is a typed configuration object rather than a dictionary. Use
    /// ``AIESSettings`` with ``method="aies"`` and ``ESSSettings`` with ``method="ess"``.
    ///
    /// Sampler moves are also declared explicitly:
    ///
    /// - ``AIESSettings(moves=[AIESMoveConfig.stretch(...), AIESMoveConfig.walk(...)])``
    /// - ``ESSSettings(moves=[ESSMoveConfig.differential(...), ESSMoveConfig.global_(...)])``
    ///
    /// References
    /// ----------
    /// Goodman, J. & Weare, J. (2010). *Ensemble samplers with affine invariance*. CAMCoS 5(1), 65–80. <https://doi.org/10.2140/camcos.2010.5.65>
    ///
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat Comput 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="aies".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc(
        &self,
        p0: Vec<Vec<f64>>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMCMCSummary> {
        validate_mcmc_parameter_len(&p0, self.0.n_free())?;
        let result = self.0.mcmc(mcmc_settings_from_python(
            &p0.into_iter().map(DVector::from_vec).collect(),
            bounds,
            method,
            settings.as_ref(),
            observers,
            terminators,
            max_steps,
            debug,
            threads,
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

/// Python wrapper for [`LikelihoodExpression`].
#[cfg(feature = "python")]
#[pyclass(name = "LikelihoodExpression", module = "laddu", from_py_object)]
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
#[pyclass(name = "LikelihoodEvaluator", module = "laddu", skip_from_py_object)]
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
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
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
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        install_laddu_with_threads(threads, || self.0.evaluate(&parameters)).map_err(PyErr::from)
    }
    /// Evaluate the gradient of the sum of all terms in the evaluator
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     Parameter values for the free parameters (length ``n_free``).
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
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
    fn evaluate_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        let gradient =
            install_laddu_with_threads(threads, || self.0.evaluate_gradient(&parameters))?;
        Ok(PyArray1::from_slice(py, gradient.as_slice()))
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
    /// settings : LBFGSBSettings or AdamSettings or NelderMeadSettings or PSOSettings, optional
    ///     Typed settings for the selected minimization algorithm. The settings type must
    ///     match ``method``. For ``pso``, explicit settings are required.
    /// observers : MinimizerObserver or list of MinimizerObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MinimizerTerminator or list of MinimizerTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
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
    /// ``settings`` is a typed configuration object rather than a dictionary. Use
    /// ``LBFGSBSettings`` with ``method="lbfgsb"``, ``AdamSettings`` with ``method="adam"``,
    /// ``NelderMeadSettings`` with ``method="nelder-mead"``, and ``PSOSettings`` with
    /// ``method="pso"``.
    ///
    /// Nested typed helpers are also explicit:
    ///
    /// - ``LBFGSBSettings(line_search=LineSearchConfig.morethuente(...))``
    /// - ``NelderMeadSettings(simplex=SimplexConfig.orthogonal(...))``
    /// - ``PSOSettings(SwarmInitializerConfig.random_in_limits(...), ...)``
    ///
    /// ``PSOSettings`` is required for ``method="pso"`` because the swarm initializer must be
    /// provided explicitly.
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
    fn minimize(
        &self,
        p0: Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMinimizationSummary> {
        validate_free_parameter_len(p0.len(), self.0.n_free())?;
        let result = self.0.minimize(minimization_settings_from_python(
            &p0,
            bounds,
            method,
            settings.as_ref(),
            observers,
            terminators,
            max_steps,
            debug,
            threads,
        )?)?;
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
    /// settings : AIESSettings or ESSSettings, optional
    ///     Typed settings for the selected sampler. The settings type must match ``method``.
    /// observers : MCMCObserver or list of MCMCObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MCMCTerminator or list of MCMCTerminator, optional
    ///     User-defined terminators which are called at each step
    /// max_steps : int, optional
    ///     Set the maximum number of steps
    /// debug : bool, default=False
    ///     Use a debug observer to print out debugging information at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
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
    /// ``settings`` is a typed configuration object rather than a dictionary. Use
    /// ``AIESSettings`` with ``method="aies"`` and ``ESSSettings`` with ``method="ess"``.
    ///
    /// Sampler moves are also declared explicitly:
    ///
    /// - ``AIESSettings(moves=[AIESMoveConfig.stretch(...), AIESMoveConfig.walk(...)])``
    /// - ``ESSSettings(moves=[ESSMoveConfig.differential(...), ESSMoveConfig.global_(...)])``
    ///
    /// References
    /// ----------
    /// Goodman, J. & Weare, J. (2010). *Ensemble samplers with affine invariance*. CAMCoS 5(1), 65–80. <https://doi.org/10.2140/camcos.2010.5.65>
    ///
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat Comput 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
    ///
    #[pyo3(signature = (p0, *, bounds=None, method="aies".to_string(), settings=None, observers=None, terminators=None, max_steps=None, debug=false, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc(
        &self,
        p0: Vec<Vec<f64>>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
        method: String,
        settings: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        max_steps: Option<usize>,
        debug: bool,
        threads: usize,
    ) -> PyResult<PyMCMCSummary> {
        validate_mcmc_parameter_len(&p0, self.0.n_free())?;
        let result = self.0.mcmc(mcmc_settings_from_python(
            &p0.into_iter().map(DVector::from_vec).collect(),
            bounds,
            method,
            settings.as_ref(),
            observers,
            terminators,
            max_steps,
            debug,
            threads,
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
    #[cfg(feature = "python")]
    use super::install_laddu_with_threads;
    use super::{LikelihoodScalar, LikelihoodTerm, NLL};
    use approx::assert_relative_eq;
    #[cfg(feature = "mpi")]
    use laddu_core::mpi::{finalize_mpi, get_world, use_mpi, LadduMPI};
    use laddu_core::{
        amplitudes::{parameter, Amplitude, AmplitudeID, ExpressionDependence, ParameterLike},
        data::{Dataset, DatasetMetadata, EventData},
        resources::{Cache, ParameterID, Parameters, Resources, ScalarID},
        utils::vectors::Vec4,
        Expression, LadduError, LadduResult,
    };
    #[cfg(feature = "mpi")]
    use mpi::topology::{Communicator, SimpleCommunicator};
    #[cfg(feature = "mpi")]
    use mpi_test::mpi_test;
    use nalgebra::DVector;
    use num::complex::Complex64;
    use serde::{Deserialize, Serialize};
    #[cfg(feature = "mpi")]
    use std::fs;
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

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::ParameterOnly
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

    #[derive(Clone, Serialize, Deserialize)]
    struct CachedBeamScaleAmplitude {
        name: String,
        parameter: ParameterLike,
        pid: ParameterID,
        sid: ScalarID,
        p4_index: usize,
    }

    impl CachedBeamScaleAmplitude {
        #[allow(clippy::new_ret_no_self)]
        fn new(name: &str, parameter: ParameterLike, p4_index: usize) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                parameter,
                pid: ParameterID::default(),
                sid: ScalarID::default(),
                p4_index,
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for CachedBeamScaleAmplitude {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.pid = resources.register_parameter(&self.parameter)?;
            self.sid = resources.register_scalar(Some(&format!("{}.beam_energy", self.name)));
            resources.register_amplitude(&self.name)
        }

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::Mixed
        }

        fn precompute(&self, event: &laddu_core::data::NamedEventView<'_>, cache: &mut Cache) {
            cache.store_scalar(self.sid, event.p4_at(self.p4_index).e());
        }

        fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
            Complex64::new(parameters.get(self.pid), 0.0) * cache.get_scalar(self.sid)
        }

        fn compute_gradient(
            &self,
            _parameters: &Parameters,
            cache: &Cache,
            gradient: &mut DVector<Complex64>,
        ) {
            if let ParameterID::Parameter(index) = self.pid {
                gradient[index] = Complex64::new(cache.get_scalar(self.sid), 0.0);
            }
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct CacheOnlyBeamAmplitude {
        name: String,
        sid: ScalarID,
        p4_index: usize,
    }

    impl CacheOnlyBeamAmplitude {
        #[allow(clippy::new_ret_no_self)]
        fn new(name: &str, p4_index: usize) -> LadduResult<Expression> {
            Self {
                name: name.to_string(),
                sid: ScalarID::default(),
                p4_index,
            }
            .into_expression()
        }
    }

    #[typetag::serde]
    impl Amplitude for CacheOnlyBeamAmplitude {
        fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
            self.sid = resources.register_scalar(Some(&format!("{}.beam_energy", self.name)));
            resources.register_amplitude(&self.name)
        }

        fn dependence_hint(&self) -> ExpressionDependence {
            ExpressionDependence::CacheOnly
        }

        fn precompute(&self, event: &laddu_core::data::NamedEventView<'_>, cache: &mut Cache) {
            cache.store_scalar(self.sid, event.p4_at(self.p4_index).e());
        }

        fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
            Complex64::new(cache.get_scalar(self.sid), 0.0)
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

    fn dataset_with_two_p4_and_weights(
        beam_energies: &[(f64, f64)],
        weights: &[f64],
    ) -> Arc<Dataset> {
        assert_eq!(beam_energies.len(), weights.len());
        let metadata = Arc::new(DatasetMetadata::default());
        let events = beam_energies
            .iter()
            .zip(weights.iter())
            .map(|(&(e0, e1), &weight)| {
                Arc::new(EventData {
                    p4s: vec![Vec4::new(0.0, 0.0, 0.0, e0), Vec4::new(0.0, 0.0, 0.0, e1)],
                    aux: vec![],
                    weight,
                })
            })
            .collect();
        Arc::new(Dataset::new_with_metadata(events, metadata))
    }

    #[cfg(feature = "mpi")]
    fn read_resident_rss_kb() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            let status = fs::read_to_string("/proc/self/status").ok()?;
            let vm_rss = status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))?
                .split_whitespace()
                .nth(1)?;
            vm_rss.parse::<u64>().ok()
        }

        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    #[cfg(feature = "mpi")]
    fn generated_two_p4_dataset(
        n_events: usize,
        base_energy: f64,
        weight_scale: f64,
    ) -> Arc<Dataset> {
        let metadata = Arc::new(DatasetMetadata::default());
        let events = (0..n_events)
            .map(|index| {
                let idx = index as f64;
                let beam_e0 = base_energy + (idx % 17.0) * 0.35 + idx * 0.0025;
                let beam_e1 = 0.5 * base_energy + (idx % 11.0) * 0.2 + idx * 0.0015;
                let weight = 0.75 + weight_scale * (1.0 + (index % 9) as f64);
                Arc::new(EventData {
                    p4s: vec![
                        Vec4::new(0.0, 0.0, 0.0, beam_e0),
                        Vec4::new(0.0, 0.0, 0.0, beam_e1),
                    ],
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
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        (nll, vec![2.0])
    }

    fn make_two_parameter_nll() -> (Box<NLL>, Vec<f64>) {
        let amp_a = ConstantAmplitude::new("amp_a", parameter("alpha")).unwrap();
        let amp_b = ConstantAmplitude::new("amp_b", parameter("beta")).unwrap();
        let expr = (amp_a + amp_b).norm_sqr();
        let data = dataset_with_weights(&[1.0, 2.0, 3.0, 1.0]);
        let mc = dataset_with_weights(&[0.5, 1.5, 2.5, 0.5]);
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        (nll, vec![0.75, -1.25])
    }

    #[cfg(feature = "python")]
    #[test]
    fn install_laddu_with_threads_handles_repeated_short_calls() {
        let (nll, params) = make_two_parameter_nll();

        let expected_value = nll
            .evaluate(&params)
            .expect("reference evaluation should succeed");
        let expected_gradient = nll
            .evaluate_gradient(&params)
            .expect("reference gradient should succeed");
        let expected_projection = nll
            .project_weights(&params, None)
            .expect("reference projection should succeed");

        for _ in 0..64 {
            let value = install_laddu_with_threads(Some(2), || nll.evaluate(&params))
                .expect("threaded evaluation should succeed");
            let gradient = install_laddu_with_threads(Some(2), || nll.evaluate_gradient(&params))
                .expect("threaded gradient should succeed");
            let projection =
                install_laddu_with_threads(Some(2), || nll.project_weights(&params, None))
                    .expect("threaded projection should succeed");

            assert_relative_eq!(value, expected_value, epsilon = 1e-12);
            assert_eq!(gradient.len(), expected_gradient.len());
            for (lhs, rhs) in gradient.iter().zip(expected_gradient.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
            assert_eq!(projection.len(), expected_projection.len());
            for (lhs, rhs) in projection.iter().zip(expected_projection.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
        }
    }

    #[derive(Clone, Copy)]
    enum DeterministicModelKind {
        Separable,
        Partial,
        NonSeparable,
    }

    struct DeterministicNllFixture {
        nll: Box<NLL>,
        parameters: Vec<f64>,
    }

    const DETERMINISTIC_STRICT_ABS_TOL: f64 = 1e-12;
    const DETERMINISTIC_STRICT_REL_TOL: f64 = 1e-10;

    fn assert_nll_fixture_matches_weighted_baseline(fixture: &DeterministicNllFixture) {
        let expected_value = super::evaluate_weighted_expression_sum_local(
            &fixture.nll.data_evaluator,
            &fixture.parameters,
            |l| f64::ln(l.re),
        );
        let expected_mc_term = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(&fixture.parameters);
        let expected_value = -2.0 * (expected_value - expected_mc_term / fixture.nll.n_mc);

        let expected_data_gradient = fixture
            .nll
            .evaluate_data_gradient_term_local(&fixture.parameters);
        let expected_mc_gradient = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(&fixture.parameters);
        let expected_gradient =
            -2.0 * (expected_data_gradient - expected_mc_gradient / fixture.nll.n_mc);

        let actual_value = fixture.nll.evaluate_local(&fixture.parameters);
        assert_relative_eq!(
            actual_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );

        let actual_gradient = fixture.nll.evaluate_gradient_local(&fixture.parameters);
        assert_eq!(
            actual_gradient.len(),
            expected_gradient.len(),
            "fixture NLL gradient length mismatch (actual={}, expected={})",
            actual_gradient.len(),
            expected_gradient.len()
        );
        for (actual_item, expected_item) in actual_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }
    }

    #[cfg(feature = "mpi")]
    fn assert_nll_fixture_matches_mpi_reduced_baseline(
        fixture: &DeterministicNllFixture,
        world: &SimpleCommunicator,
    ) {
        let data_term_local = super::evaluate_weighted_expression_sum_local(
            &fixture.nll.data_evaluator,
            &fixture.parameters,
            |l| f64::ln(l.re),
        );
        let mc_term_local = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(&fixture.parameters);
        let data_term = super::reduce_scalar(world, data_term_local);
        let mc_term = super::reduce_scalar(world, mc_term_local);
        let expected_value = -2.0 * (data_term - mc_term / fixture.nll.n_mc);
        let mpi_value = fixture.nll.evaluate_mpi_value(&fixture.parameters, world);
        assert_relative_eq!(
            mpi_value,
            expected_value,
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );

        let data_gradient_local = fixture
            .nll
            .evaluate_data_gradient_term_local(&fixture.parameters);
        let mc_gradient_local = fixture
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(&fixture.parameters);
        let data_gradient = super::reduce_gradient(world, &data_gradient_local);
        let mc_gradient = super::reduce_gradient(world, &mc_gradient_local);
        let expected_gradient = -2.0 * (data_gradient - mc_gradient / fixture.nll.n_mc);
        let mpi_gradient = fixture
            .nll
            .evaluate_mpi_gradient(&fixture.parameters, world);
        assert_eq!(
            mpi_gradient.len(),
            expected_gradient.len(),
            "fixture MPI gradient length mismatch (actual={}, expected={})",
            mpi_gradient.len(),
            expected_gradient.len()
        );
        for (actual_item, expected_item) in mpi_gradient.iter().zip(expected_gradient.iter()) {
            assert_relative_eq!(
                *actual_item,
                *expected_item,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );
        }
    }

    fn make_deterministic_nll_fixture(kind: DeterministicModelKind) -> DeterministicNllFixture {
        let data = dataset_with_two_p4_and_weights(
            &[
                (1.0, 0.8),
                (2.5, 1.7),
                (4.0, 2.4),
                (3.3, 1.1),
                (5.2, 2.8),
                (1.7, 0.9),
            ],
            &[0.7, 1.2, 0.9, 1.5, 0.8, 1.1],
        );
        let mc = dataset_with_two_p4_and_weights(
            &[
                (1.5, 1.0),
                (3.0, 2.1),
                (5.5, 2.9),
                (2.0, 1.2),
                (4.2, 1.8),
                (2.8, 1.4),
            ],
            &[0.8, 1.4, 0.6, 1.1, 0.75, 1.25],
        );

        match kind {
            DeterministicModelKind::Separable => {
                let p1 = ConstantAmplitude::new("p1", parameter("p1"))
                    .expect("separable p1 should build");
                let p2 = ConstantAmplitude::new("p2", parameter("p2"))
                    .expect("separable p2 should build");
                let c1 = CacheOnlyBeamAmplitude::new("c1", 0).expect("separable c1 should build");
                let c2 = CacheOnlyBeamAmplitude::new("c2", 1).expect("separable c2 should build");
                let expression = (&p1 * &c1) + &(&p2 * &c2);
                DeterministicNllFixture {
                    nll: NLL::new(&expression, &data, &mc, None)
                        .expect("separable NLL should build"),
                    parameters: vec![0.4, 0.2],
                }
            }
            DeterministicModelKind::Partial => {
                let p =
                    ConstantAmplitude::new("p", parameter("p")).expect("partial p should build");
                let c = CacheOnlyBeamAmplitude::new("c", 0).expect("partial c should build");
                let m = CachedBeamScaleAmplitude::new("m", parameter("m"), 1)
                    .expect("partial m should build");
                let expression = (&p * &c) + &m;
                DeterministicNllFixture {
                    nll: NLL::new(&expression, &data, &mc, None).expect("partial NLL should build"),
                    parameters: vec![0.35, 0.25],
                }
            }
            DeterministicModelKind::NonSeparable => {
                let m1 = CachedBeamScaleAmplitude::new("m1", parameter("m1"), 0)
                    .expect("non-separable m1 should build");
                let m2 = CachedBeamScaleAmplitude::new("m2", parameter("m2"), 1)
                    .expect("non-separable m2 should build");
                let expression = &m1 * &m2;
                DeterministicNllFixture {
                    nll: NLL::new(&expression, &data, &mc, None)
                        .expect("non-separable NLL should build"),
                    parameters: vec![0.2, 0.15],
                }
            }
        }
    }

    #[cfg(feature = "mpi")]
    fn make_mixed_workload_nll_fixture(n_events: usize) -> DeterministicNllFixture {
        let data = generated_two_p4_dataset(n_events, 1.4, 0.08);
        let mc = generated_two_p4_dataset(n_events, 1.9, 0.11);
        let p = ConstantAmplitude::new("p", parameter("p")).expect("mixed-workload p should build");
        let c = CacheOnlyBeamAmplitude::new("c", 0)
            .expect("mixed-workload cache amplitude should build");
        let m = CachedBeamScaleAmplitude::new("m", parameter("m"), 1)
            .expect("mixed-workload beam amplitude should build");
        let expression = (&p * &c) + &m;
        DeterministicNllFixture {
            nll: NLL::new(&expression, &data, &mc, None).expect("mixed-workload NLL should build"),
            parameters: vec![0.35, 0.25],
        }
    }

    fn case_nll_evaluate_short(nll: &NLL) -> LadduResult<()> {
        nll.evaluate(&[]).map(|_| ())
    }

    fn case_nll_evaluate_gradient_long(nll: &NLL) -> LadduResult<()> {
        nll.evaluate_gradient(&[1.0, 2.0]).map(|_| ())
    }

    fn case_nll_project_short(nll: &NLL) -> LadduResult<()> {
        nll.project_weights(&[], None).map(|_| ())
    }

    fn case_nll_project_weights_and_gradients_long(nll: &NLL) -> LadduResult<()> {
        nll.project_weights_and_gradients(&[1.0, 2.0], None)
            .map(|_| ())
    }

    fn case_nll_project_weights_subset_short(nll: &NLL) -> LadduResult<()> {
        nll.project_weights_subset_local::<&str>(&[], &["missing_amplitude"], None)
            .map(|_| ())
    }

    fn case_nll_project_weights_and_gradients_subset_long(nll: &NLL) -> LadduResult<()> {
        nll.project_weights_and_gradients_subset_local::<&str>(
            &[1.0, 2.0],
            &["missing_amplitude"],
            None,
        )
        .map(|_| ())
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
            (
                "nll.project_weights short",
                case_nll_project_short(nll.as_ref()),
            ),
            (
                "nll.project_weights_and_gradients long",
                case_nll_project_weights_and_gradients_long(nll.as_ref()),
            ),
            (
                "nll.project_weights_subset short",
                case_nll_project_weights_subset_short(nll.as_ref()),
            ),
            (
                "nll.project_weights_and_gradients_subset long",
                case_nll_project_weights_and_gradients_subset_long(nll.as_ref()),
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
                "project_weights_subset unknown",
                nll.project_weights_subset_local::<&str>(&params, &["missing_amplitude"], None)
                    .map(|_| ()),
            ),
            (
                "project_weights_and_gradients_subset unknown",
                nll.project_weights_and_gradients_subset_local::<&str>(
                    &params,
                    &["missing_amplitude"],
                    None,
                )
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

    #[cfg(feature = "rayon")]
    #[test]
    fn gradient_scratch_reuse_is_thread_safe_across_parallel_calls() {
        let (nll_single, params_single) = make_constant_nll();
        let (nll_multi, params_multi) = make_two_parameter_nll();
        let nll_single = Arc::new(*nll_single);
        let nll_multi = Arc::new(*nll_multi);
        let expected_single = nll_single
            .evaluate_gradient(&params_single)
            .expect("single-parameter gradient should evaluate");
        let expected_multi = nll_multi
            .evaluate_gradient(&params_multi)
            .expect("two-parameter gradient should evaluate");
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let nll_single = Arc::clone(&nll_single);
                let nll_multi = Arc::clone(&nll_multi);
                let params_single = params_single.clone();
                let params_multi = params_multi.clone();
                let expected_single = expected_single.clone();
                let expected_multi = expected_multi.clone();
                scope.spawn(move || {
                    for _ in 0..100 {
                        let single_gradient = nll_single
                            .evaluate_gradient(&params_single)
                            .expect("single-parameter gradient should evaluate");
                        assert_relative_eq!(
                            single_gradient[0],
                            expected_single[0],
                            epsilon = 1e-12
                        );
                        let multi_gradient = nll_multi
                            .evaluate_gradient(&params_multi)
                            .expect("two-parameter gradient should evaluate");
                        assert_eq!(multi_gradient.len(), expected_multi.len());
                        for index in 0..expected_multi.len() {
                            assert_relative_eq!(
                                multi_gradient[index],
                                expected_multi[index],
                                epsilon = 1e-12
                            );
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn nll_value_matches_mixed_scale_weighted_closed_form() {
        let amp = ConstantAmplitude::new("amp", parameter("scale")).unwrap();
        let expr = amp.norm_sqr();
        let data = dataset_with_weights(&[1.0e12, 1.0e-12, 3.5, 7.25e4, 2.0e-3]);
        let mc = dataset_with_weights(&[4.0e9, 9.0e-6, 1.25, 2.5e2, 8.0e-4]);
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        let params = vec![1.125];

        let intensity: f64 = params[0] * params[0];
        let data_weight_sum = data
            .events_local()
            .iter()
            .map(|event| event.weight)
            .sum::<f64>();
        let mc_weight_sum = mc
            .events_local()
            .iter()
            .map(|event| event.weight)
            .sum::<f64>();
        let n_mc = mc.n_events_weighted();
        let expected = -2.0 * (data_weight_sum * intensity.ln() - mc_weight_sum * intensity / n_mc);

        let value = nll.evaluate(&params).unwrap();
        assert_relative_eq!(value, expected, epsilon = 1e-9, max_relative = 1e-12);
    }

    #[test]
    fn nll_evaluate_and_gradient_match_hardcoded_weighted_reference() {
        let amp_a = CachedBeamScaleAmplitude::new("amp_a", parameter("alpha"), 0).unwrap();
        let amp_b = CachedBeamScaleAmplitude::new("amp_b", parameter("beta"), 1).unwrap();
        let expr = (&amp_a + &amp_b).norm_sqr();
        let data = dataset_with_two_p4_and_weights(
            &[(1.0, 0.8), (2.5, 1.7), (4.0, 2.4), (3.3, 1.1)],
            &[0.7, 1.2, 0.9, 1.5],
        );
        let mc = dataset_with_two_p4_and_weights(
            &[(1.5, 1.0), (3.0, 2.1), (5.5, 2.9), (2.0, 1.2), (4.2, 1.8)],
            &[0.8, 1.4, 0.6, 1.1, 0.75],
        );
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        let params = vec![0.6, 1.1];
        assert_eq!(nll.free_parameters(), vec!["alpha", "beta"]);

        let value = nll.evaluate(&params).unwrap();
        assert_relative_eq!(value, 12.242296380697244, epsilon = 1e-12);

        let gradient = nll.evaluate_gradient(&params).unwrap();
        assert_eq!(gradient.len(), 2);
        assert_relative_eq!(gradient[0], 37.78259267741666, epsilon = 1e-12);
        assert_relative_eq!(gradient[1], 21.8538272590435, epsilon = 1e-12);
    }

    #[test]
    fn nll_deterministic_fixtures_cover_separable_partial_and_non_separable_models() {
        let separable = make_deterministic_nll_fixture(DeterministicModelKind::Separable);
        let partial = make_deterministic_nll_fixture(DeterministicModelKind::Partial);
        let non_separable = make_deterministic_nll_fixture(DeterministicModelKind::NonSeparable);

        for fixture in [separable, partial, non_separable] {
            assert_nll_fixture_matches_weighted_baseline(&fixture);
        }
    }

    #[test]
    fn nll_deterministic_fixture_matches_baseline_across_activation_toggles() {
        let fixture = make_deterministic_nll_fixture(DeterministicModelKind::Partial);
        assert_nll_fixture_matches_weighted_baseline(&fixture);

        fixture.nll.isolate_many(&["p", "c"]);
        assert_nll_fixture_matches_weighted_baseline(&fixture);

        fixture.nll.activate_all();
        assert_nll_fixture_matches_weighted_baseline(&fixture);
    }

    #[test]
    fn nll_project_returns_weighted_intensity() {
        let (nll, params) = make_constant_nll();
        let projection = nll.project_weights_local(&params, None).unwrap();
        assert_relative_eq!(projection[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(projection[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn nll_project_reports_structured_length_error() {
        let (nll, _) = make_constant_nll();
        let err = nll.project_weights(&[], None).unwrap_err();
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
    fn nll_project_weights_subset_reports_structured_missing_amplitude_error() {
        let (nll, params) = make_constant_nll();
        let err = nll
            .project_weights_subset_local::<&str>(&params, &["missing_amplitude"], None)
            .unwrap_err();
        assert!(matches!(err, LadduError::AmplitudeNotFoundError { .. }));
    }

    #[test]
    fn nll_project_weights_subsets_matches_repeated_project_weights_subset_calls() {
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_a".to_string()],
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
        ];
        let batched = nll
            .project_weights_subsets_local(&params, &subsets, None)
            .expect("batched projection should evaluate");
        let repeated = subsets
            .iter()
            .map(|subset| {
                nll.project_weights_subset_local(&params, subset, None)
                    .expect("single subset projection should evaluate")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched.len(), repeated.len());
        for (lhs, rhs) in batched.iter().zip(repeated.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
                assert_relative_eq!(lhs_value, rhs_value, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn nll_project_weights_subsets_handles_empty_and_duplicate_subsets() {
        let (nll, params) = make_two_parameter_nll();
        let empty: Vec<Vec<String>> = Vec::new();
        let empty_projection = nll
            .project_weights_subsets_local(&params, &empty, None)
            .expect("empty subset list should evaluate");
        assert!(empty_projection.is_empty());

        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_b".to_string()],
        ];
        let batched = nll
            .project_weights_subsets_local(&params, &subsets, None)
            .expect("batched projection should evaluate");
        let repeated = subsets
            .iter()
            .map(|subset| {
                nll.project_weights_subset_local(&params, subset, None)
                    .expect("single subset projection should evaluate")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched.len(), repeated.len());
        for (lhs, rhs) in batched.iter().zip(repeated.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
                assert_relative_eq!(lhs_value, rhs_value, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn nll_project_weights_subsets_reports_missing_amplitude_error() {
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![vec!["amp_a".to_string()], vec!["missing".to_string()]];
        let err = nll
            .project_weights_subsets_local(&params, &subsets, None)
            .expect_err("missing amplitude should fail");
        assert!(matches!(err, LadduError::AmplitudeNotFoundError { .. }));
    }

    #[test]
    fn nll_project_weights_and_gradients_subset_matches_repeated_calls() {
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
            vec!["amp_a".to_string()],
        ];
        for subset in subsets {
            let (weights_local, gradients_local) = nll
                .project_weights_and_gradients_subset_local(&params, &subset, None)
                .expect("local gradient projection should evaluate");
            let (weights_auto, gradients_auto) = nll
                .project_weights_and_gradients_subset(&params, &subset, None)
                .expect("auto gradient projection should evaluate");
            assert_eq!(weights_local.len(), weights_auto.len());
            assert_eq!(gradients_local.len(), gradients_auto.len());
            for (lhs, rhs) in weights_local.iter().zip(weights_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
            for (lhs, rhs) in gradients_local.iter().zip(gradients_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn nll_activation_changes_invalidate_projection_mask_cache() {
        let (nll, params) = make_constant_nll();
        assert!(nll.projection_active_mask_cache.lock().is_empty());

        let _ = nll
            .project_weights_subset_local::<&str>(&params, &["amp"], None)
            .unwrap();
        assert!(!nll.projection_active_mask_cache.lock().is_empty());

        nll.deactivate("amp");
        assert!(nll.projection_active_mask_cache.lock().is_empty());

        let projection = nll
            .project_weights_subset_local::<&str>(&params, &["amp"], None)
            .unwrap();
        assert_relative_eq!(projection[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(projection[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn nll_project_weights_subset_validates_length_before_isolation() {
        let (nll, _) = make_constant_nll();
        let err = nll
            .project_weights_subset_local::<&str>(&[], &["missing_amplitude"], None)
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
    fn nll_project_weights_and_gradients_subset_validates_length_before_isolation() {
        let (nll, _) = make_constant_nll();
        let err = nll
            .project_weights_and_gradients_subset_local::<&str>(
                &[1.0, 2.0],
                &["missing_amplitude"],
                None,
            )
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
    fn stochastic_nll_matches_closed_form_on_full_batch() {
        let (nll, params) = make_constant_nll();
        let stochastic = nll
            .to_stochastic(nll.data_evaluator.dataset.n_events(), Some(0))
            .unwrap();
        let stochastic_value = stochastic.evaluate(&params).unwrap();
        let deterministic_value = nll.evaluate(&params).unwrap();
        assert_relative_eq!(stochastic_value, deterministic_value, epsilon = 1e-12);
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
    #[mpi_test(np = [2])]
    fn mpi_negative_paths_report_structured_errors() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_constant_nll();

        let err_len = nll.project_weights_mpi(&[], None, &world).unwrap_err();
        assert!(matches!(
            err_len,
            LadduError::LengthMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));

        let err_amp = nll
            .project_weights_subset_mpi::<&str>(&params, &["missing_amplitude"], None, &world)
            .unwrap_err();
        assert!(matches!(err_amp, LadduError::AmplitudeNotFoundError { .. }));
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_value_and_gradient_match_total_non_mpi() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_constant_nll();
        let data_term_local =
            super::evaluate_weighted_expression_sum_local(&nll.data_evaluator, &params, |l| {
                f64::ln(l.re)
            });
        let mc_term_local = nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(&params);
        let data_term = super::reduce_scalar(&world, data_term_local);
        let mc_term = super::reduce_scalar(&world, mc_term_local);
        let expected_value = -2.0 * (data_term - mc_term / nll.n_mc);

        let mpi_value = nll.evaluate_mpi(&params, &world);
        assert_relative_eq!(mpi_value, expected_value);

        let data_gradient_local = nll.evaluate_data_gradient_term_local(&params);
        let mc_gradient_local = nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(&params);
        let data_gradient = super::reduce_gradient(&world, &data_gradient_local);
        let mc_gradient = super::reduce_gradient(&world, &mc_gradient_local);
        let expected_gradient = -2.0 * (data_gradient - mc_gradient / nll.n_mc);
        let mpi_gradient = nll.evaluate_gradient_mpi(&params, &world);
        assert_relative_eq!(mpi_gradient, expected_gradient);

        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_deterministic_fixture_matches_local_and_reduced_baselines_across_activation_toggles() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");

        let fixture = make_deterministic_nll_fixture(DeterministicModelKind::Partial);
        assert_nll_fixture_matches_weighted_baseline(&fixture);
        assert_nll_fixture_matches_mpi_reduced_baseline(&fixture, &world);

        fixture.nll.isolate_many(&["p", "c"]);
        assert_nll_fixture_matches_weighted_baseline(&fixture);
        assert_nll_fixture_matches_mpi_reduced_baseline(&fixture, &world);

        fixture.nll.activate_all();
        assert_nll_fixture_matches_weighted_baseline(&fixture);
        assert_nll_fixture_matches_mpi_reduced_baseline(&fixture, &world);

        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_mixed_scale_value_matches_local_evaluate() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let amp_a = CachedBeamScaleAmplitude::new("amp_a", parameter("scale_a"), 0).unwrap();
        let amp_b = CachedBeamScaleAmplitude::new("amp_b", parameter("scale_b"), 1).unwrap();
        let expr = (amp_a + amp_b).norm_sqr();
        let data = dataset_with_two_p4_and_weights(
            &[(1.0, 0.5), (10.0, 1.0), (3.0, 5.0), (1.0e2, 2.0e-1)],
            &[1.0e12, 1.0e-12, 3.5, 7.25e4],
        );
        let mc = dataset_with_two_p4_and_weights(
            &[(4.0, 0.1), (6.0, 2.0), (8.0, 1.5), (1.0e1, 3.0)],
            &[4.0e9, 9.0e-6, 1.25, 2.5e2],
        );
        let nll = NLL::new(&expr, &data, &mc, None).unwrap();
        let params = vec![1.125, -0.375];

        let data_local = nll.data_evaluator.evaluate_local(&params);
        let mc_local = nll.accmc_evaluator.evaluate_local(&params);
        let data_term_local: f64 = data_local
            .iter()
            .zip(nll.data_evaluator.dataset.events_local().iter())
            .map(|(value, event)| event.weight * value.re.ln())
            .sum();
        let mc_term_local: f64 = mc_local
            .iter()
            .zip(nll.accmc_evaluator.dataset.events_local().iter())
            .map(|(value, event)| event.weight * value.re)
            .sum();
        let data_term = super::reduce_scalar(&world, data_term_local);
        let mc_term = super::reduce_scalar(&world, mc_term_local);
        let expected = -2.0 * (data_term - mc_term / nll.n_mc);
        let mpi_value = nll.evaluate_mpi_value(&params, &world);
        assert_relative_eq!(mpi_value, expected, epsilon = 1e-9, max_relative = 1e-12);
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_projection_paths_are_explicit_global_gathers() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_constant_nll();

        let local_projection = nll
            .project_weights_local(&params, None)
            .expect("local projection should evaluate");
        let gathered_projection = nll
            .project_weights_mpi(&params, None, &world)
            .expect("mpi projection should gather global projection");
        let local_len = nll.accmc_evaluator.dataset.n_events_local();
        let total_len = nll.accmc_evaluator.dataset.n_events();
        assert_eq!(local_projection.len(), local_len);
        assert_eq!(gathered_projection.len(), total_len);

        let (counts, displs) = world.get_counts_displs(total_len);
        let rank = world.rank() as usize;
        let start = displs[rank] as usize;
        let end = start + counts[rank] as usize;
        assert_eq!(
            &gathered_projection[start..end],
            local_projection.as_slice()
        );

        let (local_weights, local_gradients) = nll
            .project_weights_and_gradients_local(&params, None)
            .expect("local projection gradient should evaluate");
        let (gathered_weights, gathered_gradients) = nll
            .project_weights_and_gradients_mpi(&params, None, &world)
            .expect("mpi projection gradient should gather global projection");
        assert_eq!(local_weights.len(), local_len);
        assert_eq!(local_gradients.len(), local_len);
        assert_eq!(gathered_weights.len(), total_len);
        assert_eq!(gathered_gradients.len(), total_len);
        assert_eq!(&gathered_weights[start..end], local_weights.as_slice());

        let local_grad_slice = &gathered_gradients[start..end];
        for (lhs, rhs) in local_grad_slice.iter().zip(local_gradients.iter()) {
            assert_relative_eq!(lhs, rhs);
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_project_weights_subsets_matches_repeated_project_weights_subset_mpi() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
            vec!["amp_a".to_string()],
        ];
        let batched = nll
            .project_weights_subsets_mpi(&params, &subsets, None, &world)
            .expect("batched mpi projection should evaluate");
        let repeated = subsets
            .iter()
            .map(|subset| {
                nll.project_weights_subset_mpi(&params, subset, None, &world)
                    .expect("single mpi subset projection should evaluate")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched.len(), repeated.len());
        for (lhs, rhs) in batched.iter().zip(repeated.iter()) {
            assert_eq!(lhs.len(), rhs.len());
            for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
                assert_relative_eq!(lhs_value, rhs_value, epsilon = 1e-12);
            }
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_project_weights_and_gradients_subset_matches_repeated_project_weights_and_gradients_subset_mpi(
    ) {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let (nll, params) = make_two_parameter_nll();
        let subsets = vec![
            vec!["amp_b".to_string()],
            vec!["amp_a".to_string()],
            vec!["amp_a".to_string(), "amp_b".to_string()],
        ];
        for subset in subsets {
            let (weights_mpi, gradients_mpi) = nll
                .project_weights_and_gradients_subset_mpi(&params, &subset, None, &world)
                .expect("mpi gradient projection should evaluate");
            let (weights_auto, gradients_auto) = nll
                .project_weights_and_gradients_subset(&params, &subset, None)
                .expect("auto gradient projection should evaluate");
            assert_eq!(weights_mpi.len(), weights_auto.len());
            assert_eq!(gradients_mpi.len(), gradients_auto.len());
            for (lhs, rhs) in weights_mpi.iter().zip(weights_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
            for (lhs, rhs) in gradients_mpi.iter().zip(gradients_auto.iter()) {
                assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
            }
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2])]
    fn mpi_mixed_workload_rss_stays_bounded() {
        use_mpi(true);
        let world = get_world().expect("MPI world should be initialized");
        let fixture = make_mixed_workload_nll_fixture(2_048);

        let baseline_value = fixture.nll.evaluate_mpi(&fixture.parameters, &world);
        let baseline_gradient = fixture
            .nll
            .evaluate_gradient_mpi(&fixture.parameters, &world);
        let baseline_weights = fixture
            .nll
            .project_weights_mpi(&fixture.parameters, None, &world)
            .expect("baseline MPI projection should evaluate");
        let (baseline_projection_weights, baseline_projection_gradients) = fixture
            .nll
            .project_weights_and_gradients_mpi(&fixture.parameters, None, &world)
            .expect("baseline MPI projection gradient should evaluate");
        let mut post_warmup_rss_kb = Vec::new();

        assert_relative_eq!(
            baseline_weights.as_slice(),
            baseline_projection_weights.as_slice(),
            epsilon = DETERMINISTIC_STRICT_ABS_TOL,
            max_relative = DETERMINISTIC_STRICT_REL_TOL
        );

        for pass_index in 0..24 {
            let value = fixture.nll.evaluate_mpi(&fixture.parameters, &world);
            assert_relative_eq!(
                value,
                baseline_value,
                epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                max_relative = DETERMINISTIC_STRICT_REL_TOL
            );

            let gradient = fixture
                .nll
                .evaluate_gradient_mpi(&fixture.parameters, &world);
            assert_eq!(
                gradient.len(),
                baseline_gradient.len(),
                "mixed-workload MPI gradient length should remain stable"
            );
            for (actual_item, expected_item) in gradient.iter().zip(baseline_gradient.iter()) {
                assert_relative_eq!(
                    *actual_item,
                    *expected_item,
                    epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                    max_relative = DETERMINISTIC_STRICT_REL_TOL
                );
            }

            let weights = fixture
                .nll
                .project_weights_mpi(&fixture.parameters, None, &world)
                .expect("MPI projection should remain evaluable");
            assert_eq!(
                weights.len(),
                baseline_weights.len(),
                "mixed-workload MPI projection length should remain stable"
            );
            for (actual_item, expected_item) in weights.iter().zip(baseline_weights.iter()) {
                assert_relative_eq!(
                    *actual_item,
                    *expected_item,
                    epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                    max_relative = DETERMINISTIC_STRICT_REL_TOL
                );
            }

            let (projection_weights, projection_gradients) = fixture
                .nll
                .project_weights_and_gradients_mpi(&fixture.parameters, None, &world)
                .expect("MPI projection gradients should remain evaluable");
            assert_eq!(
                projection_weights.len(),
                baseline_projection_weights.len(),
                "mixed-workload MPI projection-gradient weight length should remain stable"
            );
            assert_eq!(
                projection_gradients.len(),
                baseline_projection_gradients.len(),
                "mixed-workload MPI projection-gradient length should remain stable"
            );
            for (actual_item, expected_item) in projection_weights
                .iter()
                .zip(baseline_projection_weights.iter())
            {
                assert_relative_eq!(
                    *actual_item,
                    *expected_item,
                    epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                    max_relative = DETERMINISTIC_STRICT_REL_TOL
                );
            }
            for (actual_gradient, expected_gradient) in projection_gradients
                .iter()
                .zip(baseline_projection_gradients.iter())
            {
                assert_eq!(
                    actual_gradient.len(),
                    expected_gradient.len(),
                    "mixed-workload MPI projection-gradient vector length should remain stable"
                );
                for (actual_item, expected_item) in
                    actual_gradient.iter().zip(expected_gradient.iter())
                {
                    assert_relative_eq!(
                        *actual_item,
                        *expected_item,
                        epsilon = DETERMINISTIC_STRICT_ABS_TOL,
                        max_relative = DETERMINISTIC_STRICT_REL_TOL
                    );
                }
            }

            if pass_index >= 3 {
                if let Some(rss_kb) = read_resident_rss_kb() {
                    post_warmup_rss_kb.push(rss_kb);
                }
            }
        }

        if let Some((&first_rss_kb, rest_rss_kb)) = post_warmup_rss_kb.split_first() {
            let last_rss_kb = *rest_rss_kb.last().unwrap_or(&first_rss_kb);
            let min_rss_kb = post_warmup_rss_kb
                .iter()
                .copied()
                .min()
                .expect("post-warmup RSS sample should exist");
            let max_rss_kb = post_warmup_rss_kb
                .iter()
                .copied()
                .max()
                .expect("post-warmup RSS sample should exist");
            const MAX_POST_WARMUP_RSS_GROWTH_KB: u64 = 64 * 1024;
            const MAX_POST_WARMUP_RSS_SPREAD_KB: u64 = 64 * 1024;
            assert!(
                last_rss_kb.saturating_sub(first_rss_kb) <= MAX_POST_WARMUP_RSS_GROWTH_KB,
                "mixed-workload post-warmup RSS grew by {} KiB (first={} KiB, last={} KiB)",
                last_rss_kb.saturating_sub(first_rss_kb),
                first_rss_kb,
                last_rss_kb
            );
            assert!(
                max_rss_kb.saturating_sub(min_rss_kb) <= MAX_POST_WARMUP_RSS_SPREAD_KB,
                "mixed-workload post-warmup RSS spread was {} KiB (min={} KiB, max={} KiB)",
                max_rss_kb.saturating_sub(min_rss_kb),
                min_rss_kb,
                max_rss_kb
            );
        }

        finalize_mpi();
    }
}
