use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use laddu_autodiff::{AutodiffMode, AutodiffPlan, AutodiffResult};
use laddu_compile::{
    CachePlan, CompiledModel, ReductionPlan, SolveComponentPlan, SolveRowMatrixPlan,
};
use laddu_data::data::{CacheStorage, Dataset, EventBatch};
use laddu_expr::{
    ExprGraph, ExprId, P4Component,
    parameters::{ParamId, ParamLayout, ParamValues},
};
#[cfg(test)]
use laddu_kernel::ir::KernelValueClass;
use laddu_kernel::ir::{GradientKernelIr, ScalarKernelIr};
use nalgebra::{Dyn, LU};
use num::complex::Complex64;

use crate::{JitPolicy, Precision, RuntimeError, RuntimeResult, execution::Execution};

mod gradient_interpreter;
use gradient_interpreter::GradientInterpreter;
mod autodiff;
use autodiff::{DerivativeWorkspace, ReverseDerivativeWorkspace};
mod cache;
pub use cache::{CpuBatchCache, CpuCachedBatch, CpuCachedDataset, CpuPreparedDataset};
mod layout;
use layout::{Value, matrix_at, matrix_values_row_major, scalar_at, vector_at};

#[cfg(feature = "jit")]
use crate::jit::{JitGradientKernel, JitScalarKernel};

mod scalar;
use scalar::{SCALAR_BLOCK_SIZE, ScalarEvaluationPlan, ScalarEventWorkspace, ScalarExecutor};
mod planning;
use planning::GradientExecutor;
mod evaluation;
use evaluation::F32KernelInput;
mod prepared;
mod reduction;

/// Supplies event-dependent scalar values for direct CPU evaluation.
pub trait EventLookup {
    /// Returns the scalar named `name`, or `None` when it is unavailable.
    fn scalar(&self, name: &str) -> Option<f64>;

    /// Returns one component of a named four-momentum.
    fn p4_component(&self, name: &str, component: P4Component) -> Option<f64> {
        let key = format!("{}.{}", name, component.label());
        self.scalar(&key)
    }
}

impl<F> EventLookup for F
where
    F: for<'a> Fn(&'a str) -> Option<f64>,
{
    fn scalar(&self, name: &str) -> Option<f64> {
        self(name)
    }
}

impl EventLookup for HashMap<String, f64> {
    fn scalar(&self, name: &str) -> Option<f64> {
        self.get(name).copied()
    }
}

/// Prepares compiled models for CPU execution.
#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

/// CPU scalar-kernel execution strategy.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CpuExecutionMode {
    /// Prefer JIT execution when available and fall back to interpretation.
    #[default]
    Auto,
    /// Always interpret the scalar kernel.
    Interpreter,
}

/// A compiled model prepared for CPU evaluation.
#[derive(Clone, Debug)]
pub struct CpuPlan {
    pub(super) precision: Precision,
    pub(super) graph: ExprGraph,
    pub(super) params: ParamLayout,
    pub(in crate::cpu) parameter_slots: Vec<Option<ParamId>>,
    pub(super) autodiff: AutodiffPlan,
    pub(super) cache_plan: CachePlan,
    pub(super) cache_slots: Vec<Option<usize>>,
    pub(super) cached_evaluation_nodes: Vec<ExprId>,
    pub(super) cached_value_slots: Vec<Option<usize>>,
    pub(super) scalar_kernel: Option<ScalarKernelIr>,
    pub(in crate::cpu) scalar_executor: Option<ScalarExecutor>,
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(in crate::cpu) gradient_executor: GradientExecutor,
    // Direct EventLookup evaluation cannot use block JIT kernels, so f32 plans retain
    // an interpreter fallback even when cached and invariant gradients use the JIT.
    pub(in crate::cpu) f32_gradient_fallback_real: Option<GradientKernelIr>,
    pub(super) f32_gradient_fallback_imag: Option<GradientKernelIr>,
    pub(super) cache_materialization_nodes: Vec<ExprId>,
    pub(super) solve_components: Vec<Option<SolveComponentPlan>>,
    pub(super) solve_rhs_elements: Vec<Option<Vec<ExprId>>>,
    pub(in crate::cpu) solve_row_matrices: Vec<SolveRowMatrixPlan>,
    pub(super) solve_row_keys: Vec<(ExprId, usize, usize)>,
    pub(super) factor_matrix_slots: Vec<Option<usize>>,
    pub(super) factor_matrices: Vec<(ExprId, usize)>,
    pub(super) constant_factor_slots: Vec<Option<usize>>,
    pub(super) constant_factors: Vec<Arc<OnceLock<DynamicLu>>>,
}

impl CpuBackend {
    /// Prepares a model using the policies resolved by an execution context.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model lowering or differentiation fails,
    /// or the requested precision is unsupported for the model.
    pub fn prepare_for_execution(
        &self,
        model: &CompiledModel,
        execution: &Execution,
    ) -> RuntimeResult<CpuPlan> {
        let mode = match execution.jit_policy() {
            JitPolicy::Auto | JitPolicy::Enabled => CpuExecutionMode::Auto,
            JitPolicy::Disabled => CpuExecutionMode::Interpreter,
        };
        let plan = self
            .prepare_with_modes_precision(
                model,
                execution.autodiff_mode(),
                mode,
                execution.precision(),
            )
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        if execution.precision() == Precision::F32 && !plan.supports_f32_scalar_execution() {
            return Err(crate::ExecutionError::UnsupportedCpuF32Model.into());
        }
        Ok(plan)
    }

    /// Prepares a model with forward autodiff and automatic execution-mode selection.
    ///
    /// # Panics
    ///
    /// Panics if forward differentiation or executable-plan construction fails
    /// for the compiled model.
    pub fn prepare(&self, model: &CompiledModel) -> CpuPlan {
        self.prepare_with_modes(model, AutodiffMode::Forward, CpuExecutionMode::Auto)
            .expect("forward autodiff supports every compiled expression node")
    }

    /// Prepares a model with an explicit scalar-kernel execution mode.
    ///
    /// # Panics
    ///
    /// Panics if forward differentiation or executable-plan construction fails
    /// for the compiled model.
    pub fn prepare_with_execution_mode(
        &self,
        model: &CompiledModel,
        execution_mode: CpuExecutionMode,
    ) -> CpuPlan {
        self.prepare_with_modes(model, AutodiffMode::Forward, execution_mode)
            .expect("forward autodiff supports every compiled expression node")
    }

    /// Prepares a model with an explicit automatic-differentiation mode.
    ///
    /// # Errors
    ///
    /// Returns [`laddu_autodiff::AutodiffError`] when model lowering or
    /// differentiation fails.
    pub fn prepare_with_autodiff_mode(
        &self,
        model: &CompiledModel,
        mode: AutodiffMode,
    ) -> AutodiffResult<CpuPlan> {
        self.prepare_with_modes(model, mode, CpuExecutionMode::Auto)
    }

    /// Prepares a model with explicit autodiff and scalar execution modes.
    ///
    /// # Errors
    ///
    /// Returns [`laddu_autodiff::AutodiffError`] when model lowering or
    /// differentiation fails.
    pub fn prepare_with_modes(
        &self,
        model: &CompiledModel,
        autodiff_mode: AutodiffMode,
        execution_mode: CpuExecutionMode,
    ) -> AutodiffResult<CpuPlan> {
        self.prepare_with_modes_precision(model, autodiff_mode, execution_mode, Precision::F64)
    }
}

/// A complex model value and its derivatives with respect to free parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueGradient {
    value: Complex64,
    gradient: Vec<Complex64>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
/// Fixed statistics collected when a dataset is prepared for repeated evaluation.
pub struct PreparedDatasetStats {
    pub(super) local_events: usize,
    pub(super) global_events: usize,
    pub(super) local_batches: usize,
    pub(super) sum_weights: f64,
    pub(super) resident_bytes: usize,
    pub(super) storage: CacheStorage,
}

/// The scalar value and free-parameter gradient produced by a reduction.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionEvaluation {
    value: f64,
    gradient: Vec<f64>,
}

impl ReductionEvaluation {
    #[cfg(feature = "wgpu")]
    pub(crate) fn new(value: f64, gradient: Vec<f64>) -> Self {
        Self { value, gradient }
    }

    /// Returns the reduced scalar value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Returns derivatives in free-parameter order.
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    /// Consumes the evaluation and returns its value and gradient.
    pub fn into_parts(self) -> (f64, Vec<f64>) {
        (self.value, self.gradient)
    }
}

impl ValueGradient {
    /// Returns the complex model value.
    pub fn value(&self) -> Complex64 {
        self.value
    }

    /// Returns complex derivatives in free-parameter order.
    pub fn gradient(&self) -> &[Complex64] {
        &self.gradient
    }

    /// Consumes the evaluation and returns its value and gradient.
    pub fn into_parts(self) -> (Complex64, Vec<Complex64>) {
        (self.value, self.gradient)
    }
}

impl CpuPlan {
    pub(in crate::cpu) fn scalar_interpreter_plan(&self) -> Option<&ScalarEvaluationPlan> {
        match (&self.scalar_kernel, &self.scalar_executor) {
            (Some(_), Some(ScalarExecutor::Interpreter(plan))) => Some(plan),
            #[cfg(feature = "jit")]
            (Some(_), Some(ScalarExecutor::Jit(_))) => None,
            (Some(_), None) | (None, None) => None,
            (None, Some(_)) => unreachable!("executor requires kernel IR"),
        }
    }

    #[cfg(feature = "jit")]
    pub(super) fn scalar_jit_kernel(&self) -> Option<&JitScalarKernel> {
        match (&self.scalar_kernel, &self.scalar_executor) {
            (Some(_), Some(ScalarExecutor::Jit(kernel))) => Some(kernel),
            (Some(_), Some(ScalarExecutor::Interpreter(_))) | (Some(_), None) | (None, None) => {
                None
            }
            (None, Some(_)) => unreachable!("executor requires kernel IR"),
        }
    }

    #[cfg(feature = "jit")]
    pub(in crate::cpu) fn gradient_jit_kernel(&self) -> Option<&JitGradientKernel> {
        match &self.gradient_executor {
            GradientExecutor::Jit(kernel) => Some(kernel),
            GradientExecutor::Interpreter(_) => None,
        }
    }

    pub(in crate::cpu) fn gradient_interpreter(&self) -> Option<&GradientInterpreter> {
        match &self.gradient_executor {
            GradientExecutor::Interpreter(interpreter) => interpreter.as_ref(),
            #[cfg(feature = "jit")]
            GradientExecutor::Jit(_) => None,
        }
    }

    /// Returns the number of parameters, including fixed parameters.
    pub fn parameter_count(&self) -> usize {
        self.params.len()
    }

    /// Returns the number of free parameters.
    pub fn free_parameter_count(&self) -> usize {
        self.params.n_free()
    }

    /// Returns the event-cache layout required by this plan.
    pub fn cache_plan(&self) -> &CachePlan {
        &self.cache_plan
    }

    /// Evaluates a model that has no event-dependent inputs.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters are incompatible, the model
    /// requires event data, evaluation fails, or a matrix is singular.
    pub fn evaluate(&self, params: &ParamValues) -> RuntimeResult<Complex64> {
        self.evaluate_inner(params, None)
    }

    /// Evaluates an event-independent model and its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters are incompatible, the model
    /// requires event data, differentiation or evaluation fails, or a solve is
    /// singular.
    pub fn evaluate_with_gradient(&self, params: &ParamValues) -> RuntimeResult<ValueGradient> {
        self.evaluate_with_gradient_inner(params)
    }

    /// Evaluates the model using values supplied by an event lookup.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a required event value is missing,
    /// parameters are incompatible, evaluation fails, or a solve is singular.
    pub fn evaluate_with_event(
        &self,
        params: &ParamValues,
        event: &impl EventLookup,
    ) -> RuntimeResult<Complex64> {
        self.evaluate_inner(params, Some(event))
    }

    /// Evaluates the model and gradient using values supplied by an event lookup.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when a required event value is missing,
    /// parameters are incompatible, or differentiation or evaluation fails.
    pub fn evaluate_with_event_and_gradient(
        &self,
        params: &ParamValues,
        event: &impl EventLookup,
    ) -> RuntimeResult<ValueGradient> {
        if self.precision == Precision::F32 {
            return self.evaluate_f32_gradient(params, F32KernelInput::Event(event));
        }
        self.require_f64_gradient()?;
        let values = self.evaluate_values(params, Some(event))?;
        self.value_gradient(values, None)
    }

    /// Materializes the event-dependent cache for a batch.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when required columns are missing, expression
    /// shapes are invalid, cache construction fails, or a matrix is singular.
    ///
    /// # Panics
    ///
    /// Panics if a node selected by the validated cache plan was not evaluated.
    pub fn cache_event_batch(&self, batch: &EventBatch) -> RuntimeResult<CpuBatchCache> {
        self.materialize_cache_event_batch(batch)
    }

    /// Evaluates every row in a materialized batch cache.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or cache layout are
    /// incompatible, evaluation fails, or a matrix is singular.
    pub fn evaluate_cache(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
    ) -> RuntimeResult<Vec<Complex64>> {
        self.evaluate_cache_inner(params, cache)
    }

    /// Evaluates one row in a materialized batch cache.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `row` is out of range, parameters or cache
    /// layout are incompatible, evaluation fails, or a matrix is singular.
    pub fn evaluate_cache_row(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
    ) -> RuntimeResult<Complex64> {
        self.check_batch_cache(cache)?;
        self.evaluate_cache_row_unchecked(params, cache, row)
    }
}

impl CpuPlan {
    /// Evaluates one cached row and its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `row` is out of range, parameters or cache
    /// layout are incompatible, or differentiation or evaluation fails.
    pub fn evaluate_cache_row_with_gradient(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
        row: usize,
    ) -> RuntimeResult<ValueGradient> {
        self.check_batch_cache(cache)?;
        self.evaluate_cache_row_with_gradient_unchecked(params, cache, row)
    }

    /// Evaluates every cached row and its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or cache layout are
    /// incompatible, or differentiation or evaluation fails.
    pub fn evaluate_cache_with_gradient(
        &self,
        params: &ParamValues,
        cache: &CpuBatchCache,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        self.evaluate_cache_with_gradient_impl(params, cache)
    }

    /// Evaluates the model for every event in a batch.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when cache materialization or evaluation fails.
    pub fn evaluate_batch(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> RuntimeResult<Vec<Complex64>> {
        let cache = self.cache_event_batch(batch)?;
        self.evaluate_cache(params, &cache)
    }

    /// Evaluates the model and gradient for every event in a batch.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when cache materialization, differentiation, or
    /// evaluation fails.
    pub fn evaluate_batch_with_gradient(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        let cache = self.cache_event_batch(batch)?;
        self.evaluate_cache_with_gradient(params, &cache)
    }

    /// Materializes all event-dependent caches for a dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the dataset cannot be read, its schema is
    /// incompatible, cache construction fails, or a matrix is singular.
    pub fn cache_dataset(&self, dataset: &Dataset) -> RuntimeResult<CpuCachedDataset> {
        self.cache_dataset_impl(dataset)
    }

    /// Estimates retained compiled-cache bytes for `events`.
    pub fn cache_memory_estimate(&self, events: usize) -> usize {
        self.cache_memory_estimate_impl(events)
    }

    /// Prepares a dataset according to its cache-storage policy.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when dataset reading or cache construction
    /// fails, or another distributed worker reports failure.
    pub fn prepare_dataset(
        &self,
        execution: &Execution,
        dataset: &Dataset,
    ) -> RuntimeResult<CpuPreparedDataset> {
        self.prepare_dataset_impl(execution, dataset)
    }

    /// Execute a weighted reduction over a prepared dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when streaming, cache validation, evaluation,
    /// or reduction fails, or another distributed worker reports failure.
    pub fn reduce(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<f64> {
        self.reduce_impl(execution, params, dataset, reduction)
    }

    /// Execute a weighted reduction and its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when streaming, cache validation,
    /// differentiation, evaluation, or reduction fails, or another
    /// distributed worker reports failure.
    pub fn reduce_with_gradient(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<ReductionEvaluation> {
        self.reduce_with_gradient_impl(execution, params, dataset, reduction)
    }

    /// Evaluates every event in a fully cached dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or cache layout are
    /// incompatible, evaluation fails, or a matrix is singular.
    pub fn evaluate_cached_dataset(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
    ) -> RuntimeResult<Vec<Complex64>> {
        self.evaluate_cached_dataset_impl(params, dataset)
    }

    /// Evaluates every event and gradient in a fully cached dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or cache layout are
    /// incompatible, differentiation or evaluation fails, or a matrix is
    /// singular.
    pub fn evaluate_cached_dataset_with_gradient(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        self.evaluate_cached_dataset_with_gradient_impl(params, dataset)
    }

    pub(in crate::cpu) fn value_gradient(
        &self,
        values: Vec<Value>,
        cached_factors: Option<(&CpuBatchCache, usize)>,
    ) -> RuntimeResult<ValueGradient> {
        let value = if cached_factors.is_some() {
            self.cached_scalar_at(&values, self.graph.root())?
        } else {
            scalar_at(&values, self.graph.root().index())?
        };
        let gradient = match self.autodiff.mode() {
            AutodiffMode::Auto => unreachable!("autodiff mode is resolved during preparation"),
            AutodiffMode::Forward => {
                DerivativeWorkspace::new(self, &values, cached_factors).gradient()?
            }
            AutodiffMode::Reverse => {
                ReverseDerivativeWorkspace::new(self, &values, cached_factors).gradient()?
            }
        };
        Ok(ValueGradient { value, gradient })
    }
}

pub(super) type DynamicLu = LU<Complex64, Dyn, Dyn>;

#[cfg(test)]
#[path = "cpu/tests/mod.rs"]
mod tests;
