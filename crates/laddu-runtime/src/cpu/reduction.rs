use laddu_compile::{ReductionPlan, ReductionTransform};
#[cfg(test)]
use laddu_data::data::accurate::AccurateComplex64;
use laddu_data::data::accurate::AccurateF64;
use laddu_expr::parameters::ParamValues;
use num::complex::Complex64;
use rayon::prelude::*;

use crate::execution::Execution;
#[cfg(feature = "jit")]
use crate::jit::{JitGradientKernel, JitScalarKernel};

use super::{
    CpuCachedBatch, CpuCachedDataset, CpuPlan, CpuPreparedDataset, F32KernelInput, Precision,
    ReductionEvaluation, RuntimeError, RuntimeResult, SCALAR_BLOCK_SIZE, ScalarEventWorkspace,
    ValueGradient,
};

struct RealGradientAccumulator {
    value: AccurateF64,
    gradient: Vec<AccurateF64>,
}

impl RealGradientAccumulator {
    fn zero(parameter_count: usize) -> Self {
        Self {
            value: AccurateF64::zero(),
            gradient: (0..parameter_count).map(|_| AccurateF64::zero()).collect(),
        }
    }

    fn push(&mut self, weight: f64, value: f64, derivative: f64, model_gradient: &[Complex64]) {
        self.value.push(weight * value);
        for (sum, model_derivative) in self.gradient.iter_mut().zip(model_gradient) {
            sum.push(weight * derivative * model_derivative.re);
        }
    }

    fn push_f32(&mut self, weight: f64, value: f64, derivative: f64, model_gradient: &[f32]) {
        self.value.push(weight * value);
        for (sum, model_derivative) in self.gradient.iter_mut().zip(model_gradient) {
            sum.push(weight * derivative * f64::from(*model_derivative));
        }
    }

    fn merge(&mut self, other: Self) {
        self.value.merge(other.value);
        for (target, source) in self.gradient.iter_mut().zip(other.gradient) {
            target.merge(source);
        }
    }

    fn finish(self) -> (f64, Vec<f64>) {
        (
            self.value.finish(),
            self.gradient.into_iter().map(AccurateF64::finish).collect(),
        )
    }
}

impl CpuPlan {
    /// Execute a weighted reduction over a prepared dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when streaming, cache validation, evaluation,
    /// or reduction fails, or another distributed worker reports failure.
    pub(in crate::cpu) fn reduce_impl(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<f64> {
        let local = match dataset {
            CpuPreparedDataset::Resident { dataset, .. } => {
                self.reduce_cached(execution, params, dataset, reduction)
            }
            CpuPreparedDataset::Streaming {
                dataset,
                read_plan,
                transient_bytes,
                ..
            } => (|| {
                let _memory = execution
                    .host_memory()
                    .reserve(*transient_bytes)
                    .map_err(RuntimeError::from)?;
                let mut total = AccurateF64::zero();
                for batch in dataset
                    .stream_with_plan(*read_plan)
                    .map_err(|error| RuntimeError::Data(error.to_string()))?
                {
                    let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                    let cached = CpuCachedDataset {
                        sum_weights: (0..batch.len()).map(|row| batch.weights_at(row)).sum(),
                        batches: vec![CpuCachedBatch {
                            cache: self.cache_event_batch(&batch)?,
                        }],
                    };
                    total.push(self.reduce_cached(execution, params, &cached, reduction)?);
                }
                Ok(total.finish())
            })(),
        };
        if !execution.all_succeeded(local.is_ok()) {
            return local.and(Err(RuntimeError::DistributedPeerFailure));
        }
        Ok(execution.sum_f64(local?))
    }

    /// Execute a weighted reduction and its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when streaming, cache validation,
    /// differentiation, evaluation, or reduction fails, or another distributed
    /// worker reports failure.
    pub(in crate::cpu) fn reduce_with_gradient_impl(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<ReductionEvaluation> {
        let (value, gradient) =
            self.try_reduce_weighted_with_gradient(execution, params, dataset, |value| {
                reduction
                    .apply(value)
                    .map(|output| output.into_parts())
                    .map_err(RuntimeError::from)
            })?;
        Ok(ReductionEvaluation { value, gradient })
    }

    fn try_reduce_weighted_with_gradient<E, F>(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        let local = match dataset {
            CpuPreparedDataset::Resident { dataset, .. } => {
                self.try_reduce_weighted_with_gradient_cached(execution, params, dataset, transform)
            }
            CpuPreparedDataset::Streaming {
                dataset,
                read_plan,
                transient_bytes,
                ..
            } => (|| {
                let _memory = execution
                    .host_memory()
                    .reserve(*transient_bytes)
                    .map_err(RuntimeError::from)
                    .map_err(E::from)?;
                let mut value = AccurateF64::zero();
                let mut gradient = (0..self.free_parameter_count())
                    .map(|_| AccurateF64::zero())
                    .collect::<Vec<_>>();
                for batch in dataset
                    .stream_with_plan(*read_plan)
                    .map_err(|error| E::from(RuntimeError::Data(error.to_string())))?
                {
                    let batch =
                        batch.map_err(|error| E::from(RuntimeError::Data(error.to_string())))?;
                    let cached = CpuCachedDataset {
                        sum_weights: (0..batch.len()).map(|row| batch.weights_at(row)).sum(),
                        batches: vec![CpuCachedBatch {
                            cache: self.cache_event_batch(&batch)?,
                        }],
                    };
                    let (partial_value, partial_gradient) = self
                        .try_reduce_weighted_with_gradient_cached(
                            execution, params, &cached, &transform,
                        )?;
                    value.push(partial_value);
                    for (sum, partial) in gradient.iter_mut().zip(partial_gradient) {
                        sum.push(partial);
                    }
                }
                Ok::<_, E>((
                    value.finish(),
                    gradient.into_iter().map(AccurateF64::finish).collect(),
                ))
            })(),
        };
        if !execution.all_succeeded(local.is_ok()) {
            return local.and(Err(E::from(RuntimeError::DistributedPeerFailure)));
        }
        let (local_value, local_gradient) = local?;
        Ok((
            execution.sum_f64(local_value),
            execution.sum_slice(&local_gradient),
        ))
    }

    /// Evaluates every event in a fully cached dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or a cache layout are
    /// incompatible, evaluation fails, or a matrix is singular.
    pub(in crate::cpu) fn evaluate_cached_dataset_impl(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
    ) -> RuntimeResult<Vec<Complex64>> {
        let total_len = dataset.batches.iter().map(CpuCachedBatch::len).sum();
        let mut out = Vec::with_capacity(total_len);
        let invariant = self.scalar_invariant_values(params)?;
        let mut workspace = ScalarEventWorkspace::default();
        for batch in &dataset.batches {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                out.push(self.evaluate_cache_row_prepared(
                    params,
                    batch.cache(),
                    row,
                    invariant.as_ref(),
                    &mut workspace,
                )?);
            }
        }
        Ok(out)
    }

    /// Evaluates every event and gradient in a fully cached dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or a cache layout are
    /// incompatible, or differentiation or evaluation fails.
    pub(in crate::cpu) fn evaluate_cached_dataset_with_gradient_impl(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
    ) -> RuntimeResult<Vec<ValueGradient>> {
        let total_len = dataset.batches.iter().map(CpuCachedBatch::len).sum();
        let mut out = Vec::with_capacity(total_len);
        for batch in &dataset.batches {
            out.extend(self.evaluate_cache_with_gradient(params, batch.cache())?);
        }
        Ok(out)
    }

    fn try_weighted_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> Result<f64, E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<f64, E>,
    {
        let mut sum = 0.0;
        let invariant = self.scalar_invariant_values(params)?;
        let mut workspace = ScalarEventWorkspace::default();
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let value = self.evaluate_cache_row_prepared(
                    params,
                    batch.cache(),
                    row,
                    invariant.as_ref(),
                    &mut workspace,
                )?;
                sum += batch.weights()[row] * f(value)?;
            }
        }
        Ok(sum)
    }

    #[cfg(test)]
    pub(in crate::cpu) fn weighted_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> RuntimeResult<f64>
    where
        F: FnMut(Complex64) -> f64,
    {
        self.try_weighted_sum_cached(params, dataset, |value| Ok(f(value)))
    }

    pub(in crate::cpu) fn try_weighted_real_sum_with_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<(f64, f64), E>,
    {
        #[cfg(feature = "jit")]
        if let (Some(value_kernel), Some(gradient_kernel)) =
            (self.scalar_jit_kernel(), self.gradient_jit_kernel())
        {
            return self.try_weighted_real_sum_with_jit_gradient_cached(
                params,
                dataset,
                transform,
                value_kernel,
                gradient_kernel,
            );
        }
        if self.precision != Precision::F32
            && let Some(interpreter) = self.gradient_interpreter()
            && let Some(mut state) = interpreter.prepare_real_blocks(params)?
        {
            let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
            let output_count = state.output_count();
            for batch in dataset.batches() {
                self.check_batch_cache(batch.cache())?;
                for block in 0..batch.len().div_ceil(SCALAR_BLOCK_SIZE) {
                    let start = block * SCALAR_BLOCK_SIZE;
                    let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                    let outputs = state.evaluate(batch.cache(), start, end)?;
                    for (lane, row) in outputs.chunks_exact(output_count).enumerate() {
                        let (value, derivative) = transform(row[0])?;
                        total.push(batch.weights()[start + lane], value, derivative, &row[1..]);
                    }
                }
            }
            return Ok(total.finish());
        }
        if self.precision == Precision::F32
            && let Some(ir) = self.f32_gradient_fallback_real.as_ref()
        {
            let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
            let mut gradient = Vec::new();
            for batch in dataset.batches() {
                self.check_batch_cache(batch.cache())?;
                for row in 0..batch.len() {
                    let (value, model_gradient) = self.evaluate_f32_gradient_component_prepared(
                        ir,
                        params,
                        F32KernelInput::Cache(Some((batch.cache(), row))),
                        &mut gradient,
                    )?;
                    let (value, derivative) = transform(value)?;
                    total.push_f32(batch.weights()[row], value, derivative, model_gradient);
                }
            }
            return Ok(total.finish());
        }
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let evaluation =
                    self.evaluate_cache_row_with_gradient_unchecked(params, batch.cache(), row)?;
                let (value, derivative) = transform(evaluation.value())?;
                total.push(
                    batch.weights()[row],
                    value,
                    derivative,
                    evaluation.gradient(),
                );
            }
        }
        Ok(total.finish())
    }

    #[cfg(feature = "jit")]
    fn try_weighted_real_sum_with_jit_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut transform: F,
        value_kernel: &JitScalarKernel,
        gradient_kernel: &JitGradientKernel,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<(f64, f64), E>,
    {
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        let mut values = Vec::new();
        let mut tangents = Vec::new();
        let mut derivatives = Vec::new();
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let cache = JitScalarKernel::prepare_cache(batch.cache());
            for block in 0..batch.len().div_ceil(SCALAR_BLOCK_SIZE) {
                let start = block * SCALAR_BLOCK_SIZE;
                let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                value_kernel.evaluate_prepared(params, &cache, start, end, &mut values)?;
                derivatives.clear();
                derivatives.reserve(values.len());
                for (lane, value) in values.iter().copied().enumerate() {
                    let (value, derivative) = transform(value)?;
                    let weight = batch.weights()[start + lane];
                    total.value.push(weight * value);
                    derivatives.push(weight * derivative);
                }
                gradient_kernel.evaluate_prepared(params, &cache, start, end, 0, &mut tangents)?;
                for (lane, factor) in derivatives.iter().enumerate() {
                    for free_index in 0..self.free_parameter_count() {
                        total.gradient[free_index].push(
                            factor * tangents[lane * self.free_parameter_count() + free_index],
                        );
                    }
                }
            }
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    fn try_weighted_complex_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> Result<Complex64, E>
    where
        E: From<RuntimeError>,
        F: FnMut(Complex64) -> Result<Complex64, E>,
    {
        let mut sum = Complex64::default();
        let invariant = self.scalar_invariant_values(params)?;
        let mut workspace = ScalarEventWorkspace::default();
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            for row in 0..batch.len() {
                let value = self.evaluate_cache_row_prepared(
                    params,
                    batch.cache(),
                    row,
                    invariant.as_ref(),
                    &mut workspace,
                )?;
                sum += f(value)? * batch.weights()[row];
            }
        }
        Ok(sum)
    }

    #[cfg(test)]
    pub(in crate::cpu) fn weighted_complex_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        mut f: F,
    ) -> RuntimeResult<Complex64>
    where
        F: FnMut(Complex64) -> Complex64,
    {
        self.try_weighted_complex_sum_cached(params, dataset, |value| Ok(f(value)))
    }

    fn reduce_cached(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<f64> {
        if execution.is_parallel() && dataset.len().div_ceil(SCALAR_BLOCK_SIZE) >= 2 {
            execution.install(|| {
                self.par_try_weighted_sum_cached(params, dataset, |value| {
                    self.apply_reduction(reduction, value)
                })
            })
        } else {
            self.try_weighted_sum_cached(params, dataset, |value| {
                self.apply_reduction(reduction, value)
            })
        }
    }

    fn apply_reduction(&self, reduction: ReductionPlan, value: Complex64) -> RuntimeResult<f64> {
        if self.precision != Precision::F32 {
            return reduction
                .apply(value)
                .map(|output| output.value())
                .map_err(RuntimeError::from);
        }
        let real = value.re as f32;
        match reduction.transform() {
            ReductionTransform::Real => Ok(real as f64),
            ReductionTransform::PositiveReal if real > 0.0 => Ok(real as f64),
            ReductionTransform::LogPositiveReal if real > 0.0 => Ok(real.ln() as f64),
            ReductionTransform::PositiveReal | ReductionTransform::LogPositiveReal => reduction
                .apply(Complex64::from(real as f64))
                .map(|output| output.value())
                .map_err(RuntimeError::from),
        }
    }

    fn try_reduce_weighted_with_gradient_cached<E, F>(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        if execution.is_parallel() && dataset.len().div_ceil(SCALAR_BLOCK_SIZE) >= 2 {
            execution.install(|| {
                self.par_try_weighted_real_sum_with_gradient_cached(params, dataset, transform)
            })
        } else {
            self.try_weighted_real_sum_with_gradient_cached(params, dataset, transform)
        }
    }

    pub(crate) fn par_try_weighted_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> Result<f64, E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<f64, E> + Send + Sync,
    {
        let mut total = AccurateF64::zero();
        let invariant = self.scalar_invariant_values(params)?;
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            #[cfg(feature = "jit")]
            let jit_cache = self
                .scalar_jit_kernel()
                .map(|_| JitScalarKernel::prepare_cache(batch.cache()));
            let n_blocks = batch.len().div_ceil(SCALAR_BLOCK_SIZE);
            let partial = (0..n_blocks)
                .into_par_iter()
                .try_fold(
                    || {
                        (
                            AccurateF64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut acc, mut workspace, mut output), block| {
                        let start = block * SCALAR_BLOCK_SIZE;
                        let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                        self.evaluate_cache_block_prepared(
                            params,
                            batch.cache(),
                            start,
                            end,
                            invariant.as_ref(),
                            &mut workspace,
                            &mut output,
                            #[cfg(feature = "jit")]
                            jit_cache.as_ref(),
                        )?;
                        for (lane, value) in output.iter().copied().enumerate() {
                            acc.push(batch.weights()[start + lane] * f(value)?);
                        }
                        Ok::<_, E>((acc, workspace, output))
                    },
                )
                .try_reduce(
                    || {
                        (
                            AccurateF64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut lhs, workspace, output), (rhs, _, _)| {
                        lhs.merge(rhs);
                        Ok::<_, E>((lhs, workspace, output))
                    },
                )?;
            total.merge(partial.0);
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    pub(crate) fn par_weighted_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> RuntimeResult<f64>
    where
        F: Fn(Complex64) -> f64 + Send + Sync,
    {
        self.par_try_weighted_sum_cached(params, dataset, |value| Ok(f(value)))
    }

    pub(in crate::cpu) fn par_try_weighted_real_sum_with_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        transform: F,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        #[cfg(feature = "jit")]
        if let (Some(value_kernel), Some(gradient_kernel)) =
            (self.scalar_jit_kernel(), self.gradient_jit_kernel())
        {
            return self.par_try_weighted_real_sum_with_jit_gradient_cached(
                params,
                dataset,
                transform,
                value_kernel,
                gradient_kernel,
            );
        }
        if self.precision != Precision::F32
            && let Some(interpreter) = self.gradient_interpreter()
            && let Some(state) = interpreter.prepare_real_blocks(params)?
        {
            let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
            let output_count = state.output_count();
            for batch in dataset.batches() {
                self.check_batch_cache(batch.cache())?;
                let partial = (0..batch.len().div_ceil(SCALAR_BLOCK_SIZE))
                    .into_par_iter()
                    .try_fold(
                        || {
                            (
                                RealGradientAccumulator::zero(self.free_parameter_count()),
                                state.clone(),
                            )
                        },
                        |(mut accumulator, mut state), block| {
                            let start = block * SCALAR_BLOCK_SIZE;
                            let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                            let outputs = state.evaluate(batch.cache(), start, end)?;
                            for (lane, row) in outputs.chunks_exact(output_count).enumerate() {
                                let (value, derivative) = transform(row[0])?;
                                accumulator.push(
                                    batch.weights()[start + lane],
                                    value,
                                    derivative,
                                    &row[1..],
                                );
                            }
                            Ok::<_, E>((accumulator, state))
                        },
                    )
                    .try_reduce(
                        || {
                            (
                                RealGradientAccumulator::zero(self.free_parameter_count()),
                                state.clone(),
                            )
                        },
                        |(mut lhs, state), (rhs, _)| {
                            lhs.merge(rhs);
                            Ok::<_, E>((lhs, state))
                        },
                    )?;
                total.merge(partial.0);
            }
            return Ok(total.finish());
        }
        if self.precision == Precision::F32
            && let Some(ir) = self.f32_gradient_fallback_real.as_ref()
        {
            let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
            for batch in dataset.batches() {
                self.check_batch_cache(batch.cache())?;
                let partial = (0..batch.len())
                    .into_par_iter()
                    .try_fold(
                        || {
                            (
                                RealGradientAccumulator::zero(self.free_parameter_count()),
                                Vec::new(),
                            )
                        },
                        |(mut accumulator, mut gradient), row| {
                            let (value, model_gradient) = self
                                .evaluate_f32_gradient_component_prepared(
                                    ir,
                                    params,
                                    F32KernelInput::Cache(Some((batch.cache(), row))),
                                    &mut gradient,
                                )?;
                            let (value, derivative) = transform(value)?;
                            accumulator.push_f32(
                                batch.weights()[row],
                                value,
                                derivative,
                                model_gradient,
                            );
                            Ok::<_, E>((accumulator, gradient))
                        },
                    )
                    .try_reduce(
                        || {
                            (
                                RealGradientAccumulator::zero(self.free_parameter_count()),
                                Vec::new(),
                            )
                        },
                        |(mut lhs, gradient), (rhs, _)| {
                            lhs.merge(rhs);
                            Ok::<_, E>((lhs, gradient))
                        },
                    )?;
                total.merge(partial.0);
            }
            return Ok(total.finish());
        }
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let partial = (0..batch.len())
                .into_par_iter()
                .try_fold(
                    || RealGradientAccumulator::zero(self.free_parameter_count()),
                    |mut accumulator, row| {
                        let evaluation = self.evaluate_cache_row_with_gradient_unchecked(
                            params,
                            batch.cache(),
                            row,
                        )?;
                        let (value, derivative) = transform(evaluation.value())?;
                        accumulator.push(
                            batch.weights()[row],
                            value,
                            derivative,
                            evaluation.gradient(),
                        );
                        Ok::<_, E>(accumulator)
                    },
                )
                .try_reduce(
                    || RealGradientAccumulator::zero(self.free_parameter_count()),
                    |mut lhs, rhs| {
                        lhs.merge(rhs);
                        Ok::<_, E>(lhs)
                    },
                )?;
            total.merge(partial);
        }
        Ok(total.finish())
    }

    #[cfg(feature = "jit")]
    fn par_try_weighted_real_sum_with_jit_gradient_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        transform: F,
        value_kernel: &JitScalarKernel,
        gradient_kernel: &JitGradientKernel,
    ) -> Result<(f64, Vec<f64>), E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<(f64, f64), E> + Send + Sync,
    {
        let mut total = RealGradientAccumulator::zero(self.free_parameter_count());
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            let cache = JitScalarKernel::prepare_cache(batch.cache());
            let n_blocks = batch.len().div_ceil(SCALAR_BLOCK_SIZE);
            let partial = (0..n_blocks)
                .into_par_iter()
                .try_fold(
                    || {
                        (
                            RealGradientAccumulator::zero(self.free_parameter_count()),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        )
                    },
                    |(mut accumulator, mut values, mut tangents, mut derivatives), block| {
                        let start = block * SCALAR_BLOCK_SIZE;
                        let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                        value_kernel.evaluate_prepared(params, &cache, start, end, &mut values)?;
                        derivatives.clear();
                        derivatives.reserve(values.len());
                        for (lane, value) in values.iter().copied().enumerate() {
                            let (value, derivative) = transform(value)?;
                            let weight = batch.weights()[start + lane];
                            accumulator.value.push(weight * value);
                            derivatives.push(weight * derivative);
                        }
                        gradient_kernel.evaluate_prepared(
                            params,
                            &cache,
                            start,
                            end,
                            0,
                            &mut tangents,
                        )?;
                        for (lane, factor) in derivatives.iter().enumerate() {
                            for free_index in 0..self.free_parameter_count() {
                                accumulator.gradient[free_index].push(
                                    factor
                                        * tangents[lane * self.free_parameter_count() + free_index],
                                );
                            }
                        }
                        Ok::<_, E>((accumulator, values, tangents, derivatives))
                    },
                )
                .try_reduce(
                    || {
                        (
                            RealGradientAccumulator::zero(self.free_parameter_count()),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        )
                    },
                    |(mut lhs, values, tangents, derivatives), (rhs, _, _, _)| {
                        lhs.merge(rhs);
                        Ok::<_, E>((lhs, values, tangents, derivatives))
                    },
                )?;
            total.merge(partial.0);
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    pub(crate) fn par_try_weighted_complex_sum_cached<E, F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> Result<Complex64, E>
    where
        E: From<RuntimeError> + Send,
        F: Fn(Complex64) -> Result<Complex64, E> + Send + Sync,
    {
        let mut total = AccurateComplex64::zero();
        let invariant = self.scalar_invariant_values(params)?;
        for batch in dataset.batches() {
            self.check_batch_cache(batch.cache())?;
            #[cfg(feature = "jit")]
            let jit_cache = self
                .scalar_jit_kernel()
                .map(|_| JitScalarKernel::prepare_cache(batch.cache()));
            let n_blocks = batch.len().div_ceil(SCALAR_BLOCK_SIZE);
            let partial = (0..n_blocks)
                .into_par_iter()
                .try_fold(
                    || {
                        (
                            AccurateComplex64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut acc, mut workspace, mut output), block| {
                        let start = block * SCALAR_BLOCK_SIZE;
                        let end = (start + SCALAR_BLOCK_SIZE).min(batch.len());
                        self.evaluate_cache_block_prepared(
                            params,
                            batch.cache(),
                            start,
                            end,
                            invariant.as_ref(),
                            &mut workspace,
                            &mut output,
                            #[cfg(feature = "jit")]
                            jit_cache.as_ref(),
                        )?;
                        for (lane, value) in output.iter().copied().enumerate() {
                            acc.push(f(value)? * batch.weights()[start + lane]);
                        }
                        Ok::<_, E>((acc, workspace, output))
                    },
                )
                .try_reduce(
                    || {
                        (
                            AccurateComplex64::zero(),
                            ScalarEventWorkspace::default(),
                            Vec::new(),
                        )
                    },
                    |(mut lhs, workspace, output), (rhs, _, _)| {
                        lhs.merge(rhs);
                        Ok::<_, E>((lhs, workspace, output))
                    },
                )?;
            total.merge(partial.0);
        }
        Ok(total.finish())
    }

    #[cfg(test)]
    pub(crate) fn par_weighted_complex_sum_cached<F>(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        f: F,
    ) -> RuntimeResult<Complex64>
    where
        F: Fn(Complex64) -> Complex64 + Send + Sync,
    {
        self.par_try_weighted_complex_sum_cached(params, dataset, |value| Ok(f(value)))
    }
}
