#[cfg(feature = "rayon")]
use std::cell::RefCell;
use std::{collections::HashMap, fmt::Debug, sync::Arc};

use accurate::{sum::Klein, traits::*};
use fastrand::Rng;
#[cfg(feature = "mpi")]
use laddu_core::mpi::LadduMPI;
use laddu_core::{
    amplitude::{CompiledExpression, Evaluator, Expression, ParameterMap},
    data::Dataset,
    validate_free_parameter_len, LadduError, LadduResult,
};
#[cfg(feature = "mpi")]
use mpi::{
    collective::SystemOperation, datatype::PartitionMut, topology::SimpleCommunicator, traits::*,
};
use nalgebra::DVector;
use num::complex::Complex64;
use parking_lot::Mutex;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::term::LikelihoodTerm;
use crate::random::RngSubsetExtension;

pub(crate) type ProjectionMaskCacheKey = (bool, Vec<String>);

pub(crate) fn validate_stochastic_batch_size(
    batch_size: usize,
    n_events: usize,
) -> LadduResult<()> {
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

#[cfg(feature = "mpi")]
pub(crate) fn reduce_scalar(world: &SimpleCommunicator, value: f64) -> f64 {
    let mut reduced = 0.0;
    world.all_reduce_into(&value, &mut reduced, SystemOperation::sum());
    reduced
}

#[cfg(feature = "mpi")]
pub(crate) fn reduce_gradient(world: &SimpleCommunicator, gradient: &DVector<f64>) -> DVector<f64> {
    let mut reduced = vec![0.0; gradient.len()];
    world.all_reduce_into(gradient.as_slice(), &mut reduced, SystemOperation::sum());
    DVector::from_vec(reduced)
}

pub(crate) fn evaluate_weighted_expression_sum_local<F>(
    evaluator: &Evaluator,
    parameters: &[f64],
    value_map: F,
) -> LadduResult<f64>
where
    F: Fn(Complex64) -> f64 + Copy + Send + Sync,
{
    let resources = evaluator.resources.read();
    let parameters = resources.parameter_map.assemble(parameters)?;
    let amplitude_len = evaluator.amplitudes.len();
    let active_indices = resources.active_indices().to_vec();
    let program_snapshot = evaluator.expression_value_program_snapshot();
    let slot_count = evaluator.expression_value_program_snapshot_slot_count(&program_snapshot);
    #[cfg(feature = "rayon")]
    {
        Ok(resources
            .caches
            .par_iter()
            .zip(evaluator.dataset.weights_local().par_iter())
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
                    let l = evaluator.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        amplitude_values,
                        expr_slots,
                    );
                    *event * value_map(l)
                },
            )
            .parallel_sum_with_accumulator::<Klein<f64>>())
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
        let mut expr_slots = vec![Complex64::ZERO; slot_count];
        Ok(resources
            .caches
            .iter()
            .zip(evaluator.dataset.weights_local().iter())
            .map(|(cache, event)| {
                for &amp_idx in &active_indices {
                    amplitude_values[amp_idx] =
                        evaluator.amplitudes[amp_idx].compute(&parameters, cache);
                }
                let l = evaluator.evaluate_expression_value_with_program_snapshot(
                    &program_snapshot,
                    &amplitude_values,
                    &mut expr_slots,
                );
                *event * value_map(l)
            })
            .sum_with_accumulator::<Klein<f64>>())
    }
}

pub(crate) fn project_weights_local_from_evaluator(
    evaluator: &Evaluator,
    parameters: &[f64],
    n_mc: f64,
) -> LadduResult<Vec<f64>> {
    let resources = evaluator.resources.read();
    let parameters = resources.parameter_map.assemble(parameters)?;
    let amplitude_len = evaluator.amplitudes.len();
    let active_indices = resources.active_indices().to_vec();
    let program_snapshot = evaluator.expression_value_program_snapshot();
    let slot_count = evaluator.expression_value_program_snapshot_slot_count(&program_snapshot);
    #[cfg(feature = "rayon")]
    {
        Ok(resources
            .caches
            .par_iter()
            .zip(evaluator.dataset.weights_local().par_iter())
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
                    let value = evaluator.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        amplitude_values,
                        expr_slots,
                    );
                    *event * value.re / n_mc
                },
            )
            .collect())
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
        let mut expr_slots = vec![Complex64::ZERO; slot_count];
        Ok(resources
            .caches
            .iter()
            .zip(evaluator.dataset.weights_local().iter())
            .map(|(cache, event)| {
                for &amp_idx in &active_indices {
                    amplitude_values[amp_idx] =
                        evaluator.amplitudes[amp_idx].compute(&parameters, cache);
                }
                let value = evaluator.evaluate_expression_value_with_program_snapshot(
                    &program_snapshot,
                    &amplitude_values,
                    &mut expr_slots,
                );
                *event * value.re / n_mc
            })
            .collect())
    }
}

pub(crate) fn project_weights_local_from_resolved_mask(
    evaluator: &Evaluator,
    parameters: &[f64],
    n_mc: f64,
    resolved_mask: &[bool],
) -> LadduResult<Vec<f64>> {
    let resources = evaluator.resources.read();
    let parameters = resources.parameter_map.assemble(parameters)?;
    let amplitude_len = evaluator.amplitudes.len();
    let active_indices = resolved_mask
        .iter()
        .enumerate()
        .filter_map(|(index, &active)| if active { Some(index) } else { None })
        .collect::<Vec<_>>();
    let program_snapshot =
        evaluator.expression_value_program_snapshot_for_active_mask(resolved_mask)?;
    let slot_count = evaluator.expression_value_program_snapshot_slot_count(&program_snapshot);
    #[cfg(feature = "rayon")]
    {
        Ok(resources
            .caches
            .par_iter()
            .zip(evaluator.dataset.weights_local().par_iter())
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
                    let value = evaluator.evaluate_expression_value_with_program_snapshot(
                        &program_snapshot,
                        amplitude_values,
                        expr_slots,
                    );
                    *event * value.re / n_mc
                },
            )
            .collect())
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
        let mut expr_slots = vec![Complex64::ZERO; slot_count];
        Ok(resources
            .caches
            .iter()
            .zip(evaluator.dataset.weights_local().iter())
            .map(|(cache, event)| {
                for &amp_idx in &active_indices {
                    amplitude_values[amp_idx] =
                        evaluator.amplitudes[amp_idx].compute(&parameters, cache);
                }
                let value = evaluator.evaluate_expression_value_with_program_snapshot(
                    &program_snapshot,
                    &amplitude_values,
                    &mut expr_slots,
                );
                *event * value.re / n_mc
            })
            .collect())
    }
}

pub(crate) fn project_weights_and_gradients_local_from_evaluator(
    evaluator: &Evaluator,
    parameters: &[f64],
    n_mc: f64,
) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
    let resources = evaluator.resources.read();
    let parameters = resources.parameter_map.assemble(parameters)?;
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
            .zip(evaluator.dataset.weights_local().par_iter())
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
                        *event * value.re / n_mc,
                        gradient.map(|g| g.re).scale(*event / n_mc),
                    )
                },
            )
            .collect::<Vec<_>>();
        Ok(weighted.into_iter().unzip())
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut amplitude_values = vec![Complex64::ZERO; amplitude_len];
        let mut gradient_values = vec![DVector::zeros(grad_dim); amplitude_len];
        let mut value_slots = vec![Complex64::ZERO; slot_count];
        let mut gradient_slots = vec![DVector::zeros(grad_dim); slot_count];
        Ok(resources
            .caches
            .iter()
            .zip(evaluator.dataset.weights_local().iter())
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
                    *event * value.re / n_mc,
                    gradient.map(|g| g.re).scale(*event / n_mc),
                )
            })
            .unzip())
    }
}

#[cfg(feature = "rayon")]
pub(crate) fn sum_dvectors_parallel(
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
pub(crate) struct GradientScratchKey {
    n_parameters: usize,
    n_amplitudes: usize,
    n_expression_slots: usize,
}

#[cfg(feature = "rayon")]
pub(crate) struct GradientScratchWorkspace {
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
pub(crate) struct GradientScratchLease {
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
pub(crate) fn acquire_gradient_scratch(key: GradientScratchKey) -> GradientScratchLease {
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

/// An extended, unbinned negative log-likelihood evaluator.
#[derive(Clone)]
pub struct NLL {
    /// The internal [`Evaluator`] for data
    pub data_evaluator: Evaluator,
    /// The internal [`Evaluator`] for accepted Monte Carlo
    pub accmc_evaluator: Evaluator,
    pub(crate) n_mc: f64,
    pub(crate) projection_active_mask_cache: Arc<Mutex<HashMap<ProjectionMaskCacheKey, Vec<bool>>>>,
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

    fn projection_cache_key<T: AsRef<str>>(names: &[T], strict: bool) -> ProjectionMaskCacheKey {
        (strict, Self::normalized_projection_key(names))
    }

    fn resolve_projection_active_mask_for_evaluator<T: AsRef<str>>(
        evaluator: &Evaluator,
        names: &[T],
        strict: bool,
    ) -> LadduResult<Vec<bool>> {
        let current_active_mask = evaluator.active_mask();
        let isolate_result = if strict {
            evaluator.isolate_many_strict(names)
        } else {
            evaluator.isolate_many(names);
            Ok(())
        };
        if let Err(err) = isolate_result {
            evaluator.set_active_mask(&current_active_mask)?;
            return Err(err);
        }
        let resolved_mask = evaluator.active_mask();
        evaluator.set_active_mask(&current_active_mask)?;
        Ok(resolved_mask)
    }

    fn get_or_build_projection_active_mask<T: AsRef<str>>(
        &self,
        names: &[T],
        strict: bool,
    ) -> LadduResult<Vec<bool>> {
        let key = Self::projection_cache_key(names, strict);
        if let Some(mask) = self.projection_active_mask_cache.lock().get(&key).cloned() {
            return Ok(mask);
        }

        let resolved_mask = Self::resolve_projection_active_mask_for_evaluator(
            &self.accmc_evaluator,
            names,
            strict,
        )?;
        self.projection_active_mask_cache
            .lock()
            .insert(key, resolved_mask.clone());
        Ok(resolved_mask)
    }

    fn invalidate_projection_mask_cache(&self) {
        self.projection_active_mask_cache.lock().clear();
    }

    /// The parameters for this NLL.
    pub fn parameters(&self) -> ParameterMap {
        self.data_evaluator.parameters()
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

    /// Returns the expression represented by this NLL.
    pub fn expression(&self) -> Expression {
        self.data_evaluator.expression()
    }

    /// Returns a tree-like diagnostic snapshot of the compiled expression for this NLL's current
    /// active-amplitude mask.
    pub fn compiled_expression(&self) -> CompiledExpression {
        self.data_evaluator.compiled_expression()
    }

    /// Create a new [`StochasticNLL`] from this [`NLL`].
    pub fn to_stochastic(
        &self,
        batch_size: usize,
        seed: Option<usize>,
    ) -> LadduResult<StochasticNLL> {
        StochasticNLL::new(self.clone(), batch_size, seed)
    }
    /// Activate an [`Amplitude`](`laddu_core::amplitude::Amplitude`) by tag, skipping missing entries.
    pub fn activate<T: AsRef<str>>(&self, name: T) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate(&name);
        self.accmc_evaluator.activate(name);
    }
    /// Activate an [`Amplitude`](`laddu_core::amplitude::Amplitude`) by tag and return an error if it is missing.
    pub fn activate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_strict(&name)?;
        self.accmc_evaluator.activate_strict(name)?;
        Ok(())
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitude::Amplitude`)s by tag, skipping missing entries.
    pub fn activate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_many(names);
        self.accmc_evaluator.activate_many(names);
    }
    /// Activate several [`Amplitude`](`laddu_core::amplitude::Amplitude`)s by tag and return an error if any are missing.
    pub fn activate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_many_strict(names)?;
        self.accmc_evaluator.activate_many_strict(names)?;
        Ok(())
    }
    /// Activate all registered [`Amplitude`](`laddu_core::amplitude::Amplitude`)s.
    pub fn activate_all(&self) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.activate_all();
        self.accmc_evaluator.activate_all();
    }
    /// Deactivate an [`Amplitude`](`laddu_core::amplitude::Amplitude`) by tag, skipping missing entries.
    pub fn deactivate<T: AsRef<str>>(&self, name: T) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate(&name);
        self.accmc_evaluator.deactivate(name);
    }
    /// Deactivate an [`Amplitude`](`laddu_core::amplitude::Amplitude`) by tag and return an error if it is missing.
    pub fn deactivate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_strict(&name)?;
        self.accmc_evaluator.deactivate_strict(name)?;
        Ok(())
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitude::Amplitude`)s by tag, skipping missing entries.
    pub fn deactivate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_many(names);
        self.accmc_evaluator.deactivate_many(names);
    }
    /// Deactivate several [`Amplitude`](`laddu_core::amplitude::Amplitude`)s by tag and return an error if any are missing.
    pub fn deactivate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_many_strict(names)?;
        self.accmc_evaluator.deactivate_many_strict(names)?;
        Ok(())
    }
    /// Deactivate all registered [`Amplitude`](`laddu_core::amplitude::Amplitude`)s.
    pub fn deactivate_all(&self) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.deactivate_all();
        self.accmc_evaluator.deactivate_all();
    }
    /// Isolate an [`Amplitude`](`laddu_core::amplitude::Amplitude`) by tag (deactivate the rest), skipping missing entries.
    pub fn isolate<T: AsRef<str>>(&self, name: T) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.isolate(&name);
        self.accmc_evaluator.isolate(name);
    }
    /// Isolate an [`Amplitude`](`laddu_core::amplitude::Amplitude`) by tag (deactivate the rest) and return an error if it is missing.
    pub fn isolate_strict<T: AsRef<str>>(&self, name: T) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.isolate_strict(&name)?;
        self.accmc_evaluator.isolate_strict(name)?;
        Ok(())
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitude::Amplitude`)s by tag (deactivate the rest), skipping missing entries.
    pub fn isolate_many<T: AsRef<str>>(&self, names: &[T]) {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.isolate_many(names);
        self.accmc_evaluator.isolate_many(names);
    }
    /// Isolate several [`Amplitude`](`laddu_core::amplitude::Amplitude`)s by tag (deactivate the rest) and return an error if any are missing.
    pub fn isolate_many_strict<T: AsRef<str>>(&self, names: &[T]) -> LadduResult<()> {
        self.invalidate_projection_mask_cache();
        self.data_evaluator.isolate_many_strict(names)?;
        self.accmc_evaluator.isolate_many_strict(names)?;
        Ok(())
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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
        if let Some(mc_evaluator) = mc_evaluator {
            project_weights_local_from_evaluator(&mc_evaluator, parameters, self.n_mc)
        } else {
            project_weights_local_from_evaluator(&self.accmc_evaluator, parameters, self.n_mc)
        }
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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
            project_weights_and_gradients_local_from_evaluator(&mc_evaluator, parameters, self.n_mc)
        } else {
            project_weights_and_gradients_local_from_evaluator(
                &self.accmc_evaluator,
                parameters,
                self.n_mc,
            )
        }
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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
    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
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

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitude::Amplitude`)s
    /// by tag, but returns the [`NLL`] to its prior state after calculation (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    fn project_weights_subset_local_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        strict: bool,
    ) -> LadduResult<Vec<f64>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if let Some(mc_evaluator) = mc_evaluator.as_ref() {
            let resolved_mask =
                Self::resolve_projection_active_mask_for_evaluator(mc_evaluator, names, strict)?;
            project_weights_local_from_resolved_mask(
                mc_evaluator,
                parameters,
                self.n_mc,
                &resolved_mask,
            )
        } else {
            let resolved_mask = self.get_or_build_projection_active_mask(names, strict)?;
            project_weights_local_from_resolved_mask(
                &self.accmc_evaluator,
                parameters,
                self.n_mc,
                &resolved_mask,
            )
        }
    }

    /// Project the model over one isolated amplitude subset in local execution, skipping
    /// missing amplitude tags.
    pub fn project_weights_subset_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        self.project_weights_subset_local_with_strict(parameters, names, mc_evaluator, false)
    }

    /// Project the model over one isolated amplitude subset in local execution and return
    /// an error if any requested amplitude tag is missing.
    pub fn project_weights_subset_local_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        self.project_weights_subset_local_with_strict(parameters, names, mc_evaluator, true)
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitude::Amplitude`)s
    /// by tag, but returns the [`NLL`] to its prior state after calculation (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    #[cfg(feature = "mpi")]
    fn project_weights_subset_mpi_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
        strict: bool,
    ) -> LadduResult<Vec<f64>> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projection =
            self.project_weights_subset_local_with_strict(parameters, names, mc_evaluator, strict)?;
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required because projection returns per-event global outputs.
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_projection, &mut partitioned_buffer);
        }
        Ok(buffer)
    }

    #[cfg(feature = "mpi")]
    /// Project the model over one isolated amplitude subset in MPI execution, skipping
    /// missing amplitude tags.
    pub fn project_weights_subset_mpi<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> LadduResult<Vec<f64>> {
        self.project_weights_subset_mpi_with_strict(parameters, names, mc_evaluator, world, false)
    }

    #[cfg(feature = "mpi")]
    /// Project the model over one isolated amplitude subset in MPI execution and return
    /// an error if any requested amplitude tag is missing.
    pub fn project_weights_subset_mpi_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> LadduResult<Vec<f64>> {
        self.project_weights_subset_mpi_with_strict(parameters, names, mc_evaluator, world, true)
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitude::Amplitude`)s
    /// by tag, but returns the [`NLL`] to its prior state after calculation.
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
    fn project_weights_subset_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        strict: bool,
    ) -> LadduResult<Vec<f64>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_subset_mpi_with_strict(
                    parameters,
                    names,
                    mc_evaluator,
                    &world,
                    strict,
                );
            }
        }
        self.project_weights_subset_local_with_strict(parameters, names, mc_evaluator, strict)
    }

    /// Project the model over one isolated amplitude subset, skipping missing amplitude
    /// names.
    pub fn project_weights_subset<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        self.project_weights_subset_with_strict(parameters, names, mc_evaluator, false)
    }

    /// Project the model over one isolated amplitude subset and return an error if any
    /// requested amplitude tag is missing.
    pub fn project_weights_subset_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<f64>> {
        self.project_weights_subset_with_strict(parameters, names, mc_evaluator, true)
    }

    /// Project the stored model over multiple isolated amplitude subsets (non-MPI version).
    fn project_weights_subsets_local_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
        strict: bool,
    ) -> LadduResult<Vec<Vec<f64>>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        if subsets.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(mc_evaluator) = mc_evaluator.as_ref() {
            let resolved_masks = subsets
                .iter()
                .map(|names| {
                    Self::resolve_projection_active_mask_for_evaluator(mc_evaluator, names, strict)
                })
                .collect::<LadduResult<Vec<_>>>()?;
            resolved_masks
                .iter()
                .map(|mask| {
                    project_weights_local_from_resolved_mask(
                        mc_evaluator,
                        parameters,
                        self.n_mc,
                        mask,
                    )
                })
                .collect()
        } else {
            let resolved_masks = subsets
                .iter()
                .map(|names| self.get_or_build_projection_active_mask(names, strict))
                .collect::<LadduResult<Vec<_>>>()?;
            resolved_masks
                .iter()
                .map(|mask| {
                    project_weights_local_from_resolved_mask(
                        &self.accmc_evaluator,
                        parameters,
                        self.n_mc,
                        mask,
                    )
                })
                .collect()
        }
    }

    /// Project the model over multiple isolated amplitude subsets in local execution,
    /// skipping missing amplitude tags within each subset.
    pub fn project_weights_subsets_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<Vec<f64>>> {
        self.project_weights_subsets_local_with_strict(parameters, subsets, mc_evaluator, false)
    }

    /// Project the model over multiple isolated amplitude subsets in local execution and
    /// return an error if any requested amplitude tag is missing.
    pub fn project_weights_subsets_local_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<Vec<f64>>> {
        self.project_weights_subsets_local_with_strict(parameters, subsets, mc_evaluator, true)
    }

    /// Project the stored model over multiple isolated amplitude subsets (MPI-compatible version).
    #[cfg(feature = "mpi")]
    fn project_weights_subsets_mpi_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
        strict: bool,
    ) -> LadduResult<Vec<Vec<f64>>> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let local_projections = self.project_weights_subsets_local_with_strict(
            parameters,
            subsets,
            mc_evaluator,
            strict,
        )?;
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

    #[cfg(feature = "mpi")]
    /// Project the model over multiple isolated amplitude subsets in MPI execution,
    /// skipping missing amplitude tags within each subset.
    pub fn project_weights_subsets_mpi<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> LadduResult<Vec<Vec<f64>>> {
        self.project_weights_subsets_mpi_with_strict(
            parameters,
            subsets,
            mc_evaluator,
            world,
            false,
        )
    }

    #[cfg(feature = "mpi")]
    /// Project the model over multiple isolated amplitude subsets in MPI execution and
    /// return an error if any requested amplitude tag is missing.
    pub fn project_weights_subsets_mpi_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> LadduResult<Vec<Vec<f64>>> {
        self.project_weights_subsets_mpi_with_strict(parameters, subsets, mc_evaluator, world, true)
    }

    /// Project the stored model over multiple isolated amplitude subsets.
    fn project_weights_subsets_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
        strict: bool,
    ) -> LadduResult<Vec<Vec<f64>>> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_subsets_mpi_with_strict(
                    parameters,
                    subsets,
                    mc_evaluator,
                    &world,
                    strict,
                );
            }
        }
        self.project_weights_subsets_local_with_strict(parameters, subsets, mc_evaluator, strict)
    }

    /// Project the model over multiple isolated amplitude subsets, skipping missing
    /// amplitude tags within each subset.
    pub fn project_weights_subsets<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<Vec<f64>>> {
        self.project_weights_subsets_with_strict(parameters, subsets, mc_evaluator, false)
    }

    /// Project the model over multiple isolated amplitude subsets and return an error if
    /// any requested amplitude tag is missing.
    pub fn project_weights_subsets_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        subsets: &[Vec<T>],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<Vec<Vec<f64>>> {
        self.project_weights_subsets_with_strict(parameters, subsets, mc_evaluator, true)
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights_and_gradients`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitude::Amplitude`)s
    /// by tag, but returns the [`NLL`] to its prior state after calculation (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    fn project_weights_and_gradients_subset_local_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        strict: bool,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        let evaluator = mc_evaluator.as_ref().unwrap_or(&self.accmc_evaluator);
        let resolved_mask = if let Some(mc_evaluator) = mc_evaluator.as_ref() {
            Self::resolve_projection_active_mask_for_evaluator(mc_evaluator, names, strict)?
        } else {
            self.get_or_build_projection_active_mask(names, strict)?
        };
        let mc_dataset = &evaluator.dataset;
        let result =
            evaluator.evaluate_with_gradient_local_with_active_mask(parameters, &resolved_mask)?;
        #[cfg(feature = "rayon")]
        let (res, res_gradient) = {
            (
                result
                    .par_iter()
                    .zip(mc_dataset.weights_local().par_iter())
                    .map(|((l, _), e)| e * l.re / self.n_mc)
                    .collect(),
                result
                    .par_iter()
                    .zip(mc_dataset.weights_local().par_iter())
                    .map(|((_, grad_l), e)| grad_l.map(|g| g.re).scale(e / self.n_mc))
                    .collect(),
            )
        };
        #[cfg(not(feature = "rayon"))]
        let (res, res_gradient) = {
            (
                result
                    .iter()
                    .zip(mc_dataset.weights_local().iter())
                    .map(|((l, _), e)| e * l.re / self.n_mc)
                    .collect(),
                result
                    .iter()
                    .zip(mc_dataset.weights_local().iter())
                    .map(|((_, grad_l), e)| grad_l.map(|g| g.re).scale(e / self.n_mc))
                    .collect(),
            )
        };
        Ok((res, res_gradient))
    }

    /// Project the model and parameter gradients over one isolated amplitude subset in
    /// local execution, skipping missing amplitude tags.
    pub fn project_weights_and_gradients_subset_local<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        self.project_weights_and_gradients_subset_local_with_strict(
            parameters,
            names,
            mc_evaluator,
            false,
        )
    }

    /// Project the model and parameter gradients over one isolated amplitude subset in
    /// local execution and return an error if any requested amplitude tag is missing.
    pub fn project_weights_and_gradients_subset_local_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        self.project_weights_and_gradients_subset_local_with_strict(
            parameters,
            names,
            mc_evaluator,
            true,
        )
    }

    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each Monte-Carlo event. This method differs from the standard
    /// [`NLL::project_weights_and_gradients`] in that it first isolates the selected [`Amplitude`](`laddu_core::amplitude::Amplitude`)s
    /// by tag, but returns the [`NLL`] to its prior state after calculation (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users will want to call [`NLL::project_weights_subset`] instead.
    #[cfg(feature = "mpi")]
    fn project_weights_and_gradients_subset_mpi_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
        strict: bool,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        let n_events = mc_evaluator
            .as_ref()
            .unwrap_or(&self.accmc_evaluator)
            .dataset
            .n_events();
        let (local_projection, local_gradient_projection) = self
            .project_weights_and_gradients_subset_local_with_strict(
                parameters,
                names,
                mc_evaluator,
                strict,
            )?;
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

    #[cfg(feature = "mpi")]
    /// Project the model and parameter gradients over one isolated amplitude subset in
    /// MPI execution, skipping missing amplitude tags.
    pub fn project_weights_and_gradients_subset_mpi<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        self.project_weights_and_gradients_subset_mpi_with_strict(
            parameters,
            names,
            mc_evaluator,
            world,
            false,
        )
    }

    #[cfg(feature = "mpi")]
    /// Project the model and parameter gradients over one isolated amplitude subset in
    /// MPI execution and return an error if any requested amplitude tag is missing.
    pub fn project_weights_and_gradients_subset_mpi_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        world: &SimpleCommunicator,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        self.project_weights_and_gradients_subset_mpi_with_strict(
            parameters,
            names,
            mc_evaluator,
            world,
            true,
        )
    }
    /// Project the stored [`Expression`] over the events in the [`Dataset`] stored by the
    /// [`Evaluator`] with the given values for free parameters to obtain weights and gradients of
    /// those weights for each
    /// Monte-Carlo event. This method differs from the standard [`NLL::project_weights_and_gradients`] in that it first
    /// isolates the selected [`Amplitude`](`laddu_core::amplitude::Amplitude`)s by tag, but returns
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
    fn project_weights_and_gradients_subset_with_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
        strict: bool,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.project_weights_and_gradients_subset_mpi_with_strict(
                    parameters,
                    names,
                    mc_evaluator,
                    &world,
                    strict,
                );
            }
        }
        self.project_weights_and_gradients_subset_local_with_strict(
            parameters,
            names,
            mc_evaluator,
            strict,
        )
    }

    /// Project the model and parameter gradients over one isolated amplitude subset,
    /// skipping missing amplitude tags.
    pub fn project_weights_and_gradients_subset<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        self.project_weights_and_gradients_subset_with_strict(
            parameters,
            names,
            mc_evaluator,
            false,
        )
    }

    /// Project the model and parameter gradients over one isolated amplitude subset and
    /// return an error if any requested amplitude tag is missing.
    pub fn project_weights_and_gradients_subset_strict<T: AsRef<str>>(
        &self,
        parameters: &[f64],
        names: &[T],
        mc_evaluator: Option<Evaluator>,
    ) -> LadduResult<(Vec<f64>, Vec<DVector<f64>>)> {
        self.project_weights_and_gradients_subset_with_strict(parameters, names, mc_evaluator, true)
    }

    fn evaluate_data_term_local(&self, parameters: &[f64]) -> LadduResult<f64> {
        evaluate_weighted_expression_sum_local(&self.data_evaluator, parameters, |l| f64::ln(l.re))
    }

    fn evaluate_mc_term_local(&self, parameters: &[f64]) -> LadduResult<f64> {
        self.accmc_evaluator
            .evaluate_weighted_value_sum_local(parameters)
    }

    #[doc(hidden)]
    pub fn profile_data_term_local_value(&self, parameters: &[f64]) -> LadduResult<f64> {
        self.evaluate_data_term_local(parameters)
    }

    #[doc(hidden)]
    pub fn profile_mc_term_local_value(&self, parameters: &[f64]) -> LadduResult<f64> {
        self.evaluate_mc_term_local(parameters)
    }

    pub(crate) fn evaluate_local(&self, parameters: &[f64]) -> LadduResult<f64> {
        let data_term = self.evaluate_data_term_local(parameters)?;
        let mc_term = self.evaluate_mc_term_local(parameters)?;
        Ok(-2.0 * (data_term - mc_term / self.n_mc))
    }

    #[cfg(feature = "mpi")]
    #[doc(hidden)]
    pub fn evaluate_mpi(&self, parameters: &[f64], world: &SimpleCommunicator) -> LadduResult<f64> {
        let data_term_local = self.evaluate_data_term_local(parameters)?;
        let data_term = reduce_scalar(world, data_term_local);
        let mc_term = self
            .accmc_evaluator
            .evaluate_weighted_value_sum_mpi(parameters, world)?;
        Ok(-2.0 * (data_term - mc_term / self.n_mc))
    }

    pub(crate) fn evaluate_data_gradient_term_local(
        &self,
        parameters: &[f64],
    ) -> LadduResult<DVector<f64>> {
        let data_resources = self.data_evaluator.resources.read();
        let data_parameters = data_resources.parameter_map.assemble(parameters)?;
        #[cfg(feature = "rayon")]
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
                .weights_local()
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
                        (*event, value, gradient)
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
                .weights_local()
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
                    (*event, value, gradient)
                })
                .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
                .sum()
        };
        Ok(data_term)
    }

    #[doc(hidden)]
    pub fn evaluate_gradient_local(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        let data_term = self.evaluate_data_gradient_term_local(parameters)?;
        let mc_term = self
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(parameters)?;
        Ok(-2.0 * (data_term - mc_term / self.n_mc))
    }

    #[cfg(feature = "mpi")]
    #[doc(hidden)]
    pub fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        world: &SimpleCommunicator,
    ) -> LadduResult<DVector<f64>> {
        let data_term_local = self.evaluate_data_gradient_term_local(parameters)?;
        let data_term = reduce_gradient(world, &data_term_local);
        let mc_term = self
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_mpi(parameters, world)?;
        Ok(-2.0 * (data_term - mc_term / self.n_mc))
    }
}

impl LikelihoodTerm for NLL {
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_mpi(parameters, &world);
            }
        }
        self.evaluate_local(parameters)
    }
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        validate_free_parameter_len(parameters.len(), self.n_free())?;
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = laddu_core::mpi::get_world() {
                return self.evaluate_gradient_mpi(parameters, &world);
            }
        }
        self.evaluate_gradient_local(parameters)
    }
    fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.data_evaluator.fix_parameter(name, value)?;
        self.accmc_evaluator.fix_parameter(name, value)?;
        Ok(())
    }
    fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.data_evaluator.free_parameter(name)?;
        self.accmc_evaluator.free_parameter(name)?;
        Ok(())
    }
    fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<()> {
        self.data_evaluator.rename_parameter(old, new)?;
        self.accmc_evaluator.rename_parameter(old, new)?;
        Ok(())
    }
    fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.data_evaluator.rename_parameters(mapping)?;
        self.accmc_evaluator.rename_parameters(mapping)?;
        Ok(())
    }
    fn parameter_map(&self) -> ParameterMap {
        self.data_evaluator.resources.read().parameter_map.clone()
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
    fn evaluate(&self, parameters: &[f64]) -> LadduResult<f64> {
        validate_free_parameter_len(parameters.len(), self.nll.n_free())?;
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
            .map(|&i| self.nll.data_evaluator.dataset.weights_local()[i])
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.weights_local()[i])
            .sum_with_accumulator::<Klein<f64>>();
        self.evaluate_local(parameters, &indices, n_data_batch_local)
    }
    fn evaluate_gradient(&self, parameters: &[f64]) -> LadduResult<DVector<f64>> {
        validate_free_parameter_len(parameters.len(), self.nll.n_free())?;
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
            .map(|&i| self.nll.data_evaluator.dataset.weights_local()[i])
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let n_data_batch_local = indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.weights_local()[i])
            .sum_with_accumulator::<Klein<f64>>();
        self.evaluate_gradient_local(parameters, &indices, n_data_batch_local)
    }
    fn fix_parameter(&self, name: &str, value: f64) -> LadduResult<()> {
        self.nll.fix_parameter(name, value)
    }
    fn free_parameter(&self, name: &str) -> LadduResult<()> {
        self.nll.free_parameter(name)
    }
    fn rename_parameter(&self, old: &str, new: &str) -> LadduResult<()> {
        self.nll.rename_parameter(old, new)
    }
    fn rename_parameters(&self, mapping: &HashMap<String, String>) -> LadduResult<()> {
        self.nll.rename_parameters(mapping)
    }
    fn update(&self) {
        self.resample();
    }
    fn parameter_map(&self) -> ParameterMap {
        self.nll.parameter_map()
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

    /// The parameters for this stochastic NLL.
    pub fn parameters(&self) -> ParameterMap {
        self.nll.parameters()
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

    /// Returns the expression represented by this stochastic NLL.
    pub fn expression(&self) -> Expression {
        self.nll.expression()
    }

    /// Returns a tree-like diagnostic snapshot of the compiled expression for this stochastic
    /// NLL's current active-amplitude mask.
    pub fn compiled_expression(&self) -> CompiledExpression {
        self.nll.compiled_expression()
    }

    #[cfg(feature = "mpi")]
    fn data_batch_weight_local(&self, indices: &[usize]) -> f64 {
        #[cfg(feature = "rayon")]
        return indices
            .par_iter()
            .map(|&i| self.nll.data_evaluator.dataset.weights_local()[i])
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        return indices
            .iter()
            .map(|&i| self.nll.data_evaluator.dataset.weights_local()[i])
            .sum_with_accumulator::<Klein<f64>>();
    }

    fn evaluate_data_term_local(&self, parameters: &[f64], indices: &[usize]) -> LadduResult<f64> {
        let data_result = self
            .nll
            .data_evaluator
            .evaluate_batch_local(parameters, indices)?;
        #[cfg(feature = "rayon")]
        {
            Ok(indices
                .par_iter()
                .zip(data_result.par_iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.weights_local()[i];
                    e * l.re.ln()
                })
                .parallel_sum_with_accumulator::<Klein<f64>>())
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(indices
                .iter()
                .zip(data_result.iter())
                .map(|(&i, &l)| {
                    let e = &self.nll.data_evaluator.dataset.weights_local()[i];
                    e * l.re.ln()
                })
                .sum_with_accumulator::<Klein<f64>>())
        }
    }

    fn evaluate_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
        n_data_batch: f64,
    ) -> LadduResult<f64> {
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term = self.evaluate_data_term_local(parameters, indices)?;
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_local(parameters)?;
        Ok(-2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc))
    }

    #[cfg(feature = "mpi")]
    fn evaluate_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> LadduResult<f64> {
        let total = self.nll.data_evaluator.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let n_data_batch_local = self.data_batch_weight_local(&locals);
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term_local = self.evaluate_data_term_local(parameters, &locals)?;
        let n_data_batch = reduce_scalar(world, n_data_batch_local);
        let data_term = reduce_scalar(world, data_term_local);
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_value_sum_mpi(parameters, world)?;
        Ok(-2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc))
    }

    fn evaluate_data_gradient_term_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
    ) -> LadduResult<DVector<f64>> {
        let data_resources = self.nll.data_evaluator.resources.read();
        let data_parameters = data_resources.parameter_map.assemble(parameters)?;
        #[cfg(feature = "rayon")]
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
                        let event = &self.nll.data_evaluator.dataset.weights_local()[idx];
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
                        (*event, value, gradient)
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
                    let event = &self.nll.data_evaluator.dataset.weights_local()[idx];
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
                    (*event, value, gradient)
                })
                .map(|(w, l, g)| g.map(|gi| gi.re * w / l.re))
                .sum()
        };
        Ok(data_term)
    }

    fn evaluate_gradient_local(
        &self,
        parameters: &[f64],
        indices: &[usize],
        n_data_batch: f64,
    ) -> LadduResult<DVector<f64>> {
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term = self.evaluate_data_gradient_term_local(parameters, indices)?;
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_local(parameters)?;
        Ok(-2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc))
    }

    #[cfg(feature = "mpi")]
    fn evaluate_gradient_mpi(
        &self,
        parameters: &[f64],
        indices: &[usize],
        world: &SimpleCommunicator,
    ) -> LadduResult<DVector<f64>> {
        let total = self.nll.data_evaluator.dataset.n_events();
        let locals = world.locals_from_globals(indices, total);
        let n_data_batch_local = self.data_batch_weight_local(&locals);
        let n_data_total = self.nll.data_evaluator.dataset.n_events_weighted();
        let data_term_local = self.evaluate_data_gradient_term_local(parameters, &locals)?;
        let n_data_batch = reduce_scalar(world, n_data_batch_local);
        let data_term = reduce_gradient(world, &data_term_local);
        let mc_term = self
            .nll
            .accmc_evaluator
            .evaluate_weighted_gradient_sum_mpi(parameters, world)?;
        Ok(-2.0 * (data_term * n_data_total / n_data_batch - mc_term / self.nll.n_mc))
    }
}
