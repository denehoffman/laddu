use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::data::Dataset;
use laddu_data::data::EventBatch;
#[cfg(feature = "wgpu")]
use laddu_data::data::{CacheStorage, MemoryPolicy, accurate::AccurateF64};
use laddu_expr::parameters::ParamValues;
use num::complex::Complex64;

use crate::{
    CpuBackend, CpuPlan, CpuPreparedDataset, Execution, PreparedDatasetStats, ReductionEvaluation,
    RuntimeError, RuntimeResult,
};

/// A compiled model prepared for a concrete execution backend.
#[derive(Clone, Debug)]
pub enum PreparedModel {
    /// A model prepared for CPU execution.
    Cpu(Box<CpuPlan>),
    #[cfg(feature = "wgpu")]
    /// A model prepared for WebGPU execution.
    Wgpu(WgpuPlan),
}

/// A dataset prepared for a concrete execution backend.
#[derive(Clone, Debug)]
pub enum PreparedDataset {
    /// A dataset prepared for CPU execution.
    Cpu(CpuPreparedDataset),
    #[cfg(feature = "wgpu")]
    /// A dataset prepared for WebGPU execution.
    Wgpu(WgpuPreparedDataset),
}

impl PreparedDataset {
    /// Returns statistics collected while preparing the dataset.
    pub fn stats(&self) -> &PreparedDatasetStats {
        match self {
            Self::Cpu(dataset) => dataset.stats(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(dataset) => dataset.stats(),
        }
    }
}

impl PreparedModel {
    /// Evaluates the model for every event in a batch.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when parameters or event columns are
    /// incompatible, evaluation fails, or a matrix solve is singular.
    pub fn evaluate_batch(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> RuntimeResult<Vec<Complex64>> {
        match self {
            Self::Cpu(plan) => plan.evaluate_batch(params, batch),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(plan) => plan
                .kernel
                .evaluate_batch(&plan.context, params, batch)
                .map(|values| {
                    values
                        .into_iter()
                        .map(|(re, im)| Complex64::new(re, im))
                        .collect()
                })
                .map_err(wgpu_error),
        }
    }

    /// Evaluates the model and its free-parameter gradient for every event in a batch.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when inputs are incompatible, differentiation
    /// or evaluation fails, or the selected backend lacks event-wise gradients.
    pub fn evaluate_batch_with_gradient(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> RuntimeResult<Vec<crate::ValueGradient>> {
        match self {
            Self::Cpu(plan) => plan.evaluate_batch_with_gradient(params, batch),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => Err(RuntimeError::Wgpu(
                "event-wise model gradients are not implemented by the WGPU backend".into(),
            )),
        }
    }

    /// Prepares a compiled model for the supplied execution context.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model lowering, differentiation, backend
    /// initialization, or precision selection fails.
    pub fn prepare(model: &CompiledModel, execution: &Execution) -> RuntimeResult<Self> {
        #[cfg(feature = "wgpu")]
        if let Some(context) = execution.wgpu_context() {
            return Ok(Self::Wgpu(WgpuPlan {
                context: context.clone(),
                preparation_params: model.params().default_values(),
                kernel: std::sync::Arc::new(
                    laddu_wgpu::WgpuScalarKernel::compile(context, model).map_err(wgpu_error)?,
                ),
            }));
        }
        Ok(Self::Cpu(Box::new(
            CpuBackend.prepare_for_execution(model, execution)?,
        )))
    }

    /// Prepares a dataset for repeated evaluation with this model.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the dataset cannot be read or cached, its
    /// schema is incompatible, or backend preparation fails.
    pub fn prepare_dataset(
        &self,
        execution: &Execution,
        dataset: &Dataset,
    ) -> RuntimeResult<PreparedDataset> {
        match self {
            Self::Cpu(plan) => Ok(PreparedDataset::Cpu(
                plan.prepare_dataset(execution, dataset)?,
            )),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(plan) => plan
                .prepare_dataset(execution, dataset)
                .map(PreparedDataset::Wgpu),
        }
    }

    /// Executes a weighted scalar reduction over a prepared dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model and dataset backends differ, inputs
    /// are incompatible, evaluation fails, or the reduction domain is invalid.
    pub fn reduce(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &PreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<f64> {
        #[allow(unreachable_patterns)]
        match (self, dataset) {
            (Self::Cpu(plan), PreparedDataset::Cpu(dataset)) => {
                plan.reduce(execution, params, dataset, reduction)
            }
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(plan), PreparedDataset::Wgpu(dataset)) => {
                plan.reduce(execution, params, dataset, reduction)
            }
            _ => Err(RuntimeError::InvalidShape {
                index: 0,
                message: "prepared model and dataset use different backends".into(),
            }),
        }
    }

    /// Executes a weighted reduction and computes its free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model and dataset backends differ,
    /// differentiation or evaluation fails, or the reduction domain is invalid.
    pub fn reduce_with_gradient(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &PreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<ReductionEvaluation> {
        #[allow(unreachable_patterns)]
        match (self, dataset) {
            (Self::Cpu(plan), PreparedDataset::Cpu(dataset)) => {
                plan.reduce_with_gradient(execution, params, dataset, reduction)
            }
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(plan), PreparedDataset::Wgpu(dataset)) => {
                plan.reduce_with_gradient(execution, params, dataset, reduction)
            }
            _ => Err(RuntimeError::InvalidShape {
                index: 0,
                message: "prepared model and dataset use different backends".into(),
            }),
        }
    }
}

#[cfg(feature = "wgpu")]
/// A compiled model prepared for WebGPU execution.
#[derive(Clone)]
pub struct WgpuPlan {
    context: std::sync::Arc<laddu_wgpu::WgpuContext>,
    preparation_params: ParamValues,
    kernel: std::sync::Arc<laddu_wgpu::WgpuScalarKernel>,
}

#[cfg(feature = "wgpu")]
impl std::fmt::Debug for WgpuPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuPlan")
            .field("adapter", &self.context.info().name)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "wgpu")]
/// Dataset storage prepared for WebGPU evaluation.
#[derive(Clone)]
pub enum WgpuPreparedDataset {
    /// GPU-resident prepared batches.
    Resident {
        /// Prepared GPU batches.
        batches: Vec<laddu_wgpu::WgpuPreparedBatch>,
        /// Preparation statistics.
        stats: PreparedDatasetStats,
        /// Persistent device-memory reservation.
        memory_lease: crate::MemoryLease,
    },
    /// Source data streamed and prepared one batch at a time.
    Streaming {
        /// Source dataset.
        dataset: Dataset,
        /// Read plan used on each pass.
        read_plan: laddu_data::io::ReadPlan,
        /// Reusable prepared-batch workspace.
        workspace: std::sync::Arc<std::sync::Mutex<Option<laddu_wgpu::WgpuPreparedBatch>>>,
        /// Preparation statistics.
        stats: PreparedDatasetStats,
        /// Peak transient device bytes reserved during reductions.
        transient_bytes: u64,
    },
}

#[cfg(feature = "wgpu")]
impl std::fmt::Debug for WgpuPreparedDataset {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuPreparedDataset")
            .field("stats", self.stats())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "wgpu")]
impl WgpuPreparedDataset {
    /// Returns statistics collected while preparing the dataset.
    pub fn stats(&self) -> &PreparedDatasetStats {
        match self {
            Self::Resident { stats, .. } | Self::Streaming { stats, .. } => stats,
        }
    }
}

#[cfg(feature = "wgpu")]
impl WgpuPlan {
    fn prepare_dataset(
        &self,
        execution: &Execution,
        dataset: &Dataset,
    ) -> RuntimeResult<WgpuPreparedDataset> {
        let read_plan = execution.read_plan(dataset.read_plan());
        let mut read_plan = read_plan;
        let local_event_limit = dataset
            .num_events()
            .map_err(|error| RuntimeError::Data(error.to_string()))?
            .and_then(|events| usize::try_from(events).ok())
            .unwrap_or(usize::MAX);
        let fixed = self
            .kernel
            .prepared_memory_estimate(&self.preparation_params, 0);
        let one = self
            .kernel
            .prepared_memory_estimate(&self.preparation_params, 1);
        let per_event = one.saturating_sub(fixed);
        let schema = dataset
            .schema()
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        let host_bytes_per_event =
            (4 * schema.n_p4s() + schema.n_scalars() + usize::from(schema.has_weight()))
                .saturating_mul(size_of::<f64>())
                .saturating_mul(2);
        let host_decision = crate::MemoryDecision::fit(
            "WGPU host staging",
            0,
            u64::try_from(host_bytes_per_event).unwrap_or(u64::MAX),
            execution.host_memory().remaining(),
            local_event_limit,
            "bounded host staging",
        )?;
        let host_chunks = local_event_limit
            .saturating_add(host_decision.chunk_events.saturating_sub(1))
            / host_decision.chunk_events.max(1);
        let full = per_event
            .saturating_mul(local_event_limit)
            .saturating_add(fixed.saturating_mul(host_chunks));
        let device_pool = execution
            .device_memory()
            .ok_or_else(|| RuntimeError::Wgpu("GPU execution has no device memory pool".into()))?;
        let storage = match dataset.memory_policy() {
            MemoryPolicy::Streaming => CacheStorage::Streaming,
            MemoryPolicy::Resident => {
                if u64::try_from(full).unwrap_or(u64::MAX) > device_pool.remaining() {
                    return Err(laddu_memory::MemoryError::BudgetExceeded {
                        resource: "device".into(),
                        requested: u64::try_from(full).unwrap_or(u64::MAX),
                        remaining: device_pool.remaining(),
                    }
                    .into());
                }
                CacheStorage::Resident
            }
            MemoryPolicy::Fastest
                if u64::try_from(full).unwrap_or(u64::MAX) <= device_pool.remaining() =>
            {
                CacheStorage::Resident
            }
            MemoryPolicy::Fastest => CacheStorage::Streaming,
        };
        let memory_lease = if storage == CacheStorage::Resident {
            Some(device_pool.reserve(u64::try_from(full).unwrap_or(u64::MAX))?)
        } else {
            None
        };
        let device_decision = if storage == CacheStorage::Resident {
            crate::MemoryDecision {
                label: "WGPU prepared dataset".into(),
                fixed_bytes: u64::try_from(fixed.saturating_mul(host_chunks)).unwrap_or(u64::MAX),
                bytes_per_event: u64::try_from(per_event).unwrap_or(u64::MAX),
                chunk_events: host_decision.chunk_events,
                estimated_peak_bytes: u64::try_from(full).unwrap_or(u64::MAX),
                actual_high_water_bytes: None,
                strategy: "resident".into(),
            }
        } else {
            crate::MemoryDecision::fit(
                "WGPU prepared dataset",
                u64::try_from(fixed).unwrap_or(u64::MAX),
                u64::try_from(per_event).unwrap_or(u64::MAX),
                device_pool.remaining(),
                local_event_limit,
                "streaming",
            )?
        };
        let chunk_events = device_decision
            .chunk_events
            .min(host_decision.chunk_events)
            .max(1);
        read_plan.chunk_size = Some(
            read_plan
                .chunk_size
                .map_or(chunk_events, |manual| manual.min(chunk_events))
                .max(1),
        );
        execution.record_memory_decision(device_decision.clone());
        execution.record_memory_decision(host_decision);
        let mut batches = Vec::new();
        let mut events = 0;
        let mut batch_count = 0;
        let mut sum_weights = AccurateF64::zero();
        for batch in dataset
            .batches_with_plan(read_plan)
            .map_err(|error| RuntimeError::Data(error.to_string()))?
        {
            let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
            events += batch.len();
            batch_count += 1;
            for row in 0..batch.len() {
                sum_weights.push(batch.weights_at(row));
            }
            if storage == CacheStorage::Resident {
                batches.push(
                    self.kernel
                        .prepare_batch(&self.context, &self.preparation_params, &batch)
                        .map_err(wgpu_error)?,
                );
            }
        }
        let resident_bytes = batches
            .iter()
            .map(laddu_wgpu::WgpuPreparedBatch::resident_bytes)
            .sum();
        let stats = PreparedDatasetStats::new(
            events,
            execution.sum_usize(events),
            batch_count,
            execution.sum_f64(sum_weights.finish()),
            resident_bytes,
            storage,
        );
        Ok(match storage {
            CacheStorage::Resident => WgpuPreparedDataset::Resident {
                batches,
                stats,
                memory_lease: memory_lease.ok_or_else(|| {
                    RuntimeError::Wgpu("resident GPU dataset did not reserve device memory".into())
                })?,
            },
            CacheStorage::Streaming => WgpuPreparedDataset::Streaming {
                dataset: dataset.clone(),
                read_plan,
                workspace: Default::default(),
                stats,
                transient_bytes: device_decision.estimated_peak_bytes,
            },
        })
    }

    fn reduce(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &WgpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<f64> {
        let mut total = AccurateF64::zero();
        match dataset {
            WgpuPreparedDataset::Resident { batches, .. } => {
                for batch in batches {
                    total.push(
                        self.kernel
                            .reduce_prepared_batch(&self.context, params, batch, reduction)
                            .map_err(wgpu_error)?,
                    );
                }
            }
            WgpuPreparedDataset::Streaming {
                dataset,
                read_plan,
                workspace,
                transient_bytes,
                ..
            } => {
                let _memory = execution
                    .device_memory()
                    .ok_or_else(|| {
                        RuntimeError::Wgpu("GPU execution has no device memory pool".into())
                    })?
                    .reserve(*transient_bytes)?;
                let mut workspace = workspace.lock().map_err(|_| {
                    RuntimeError::Wgpu("streaming workspace lock is poisoned".into())
                })?;
                for batch in dataset
                    .batches_with_plan(*read_plan)
                    .map_err(|error| RuntimeError::Data(error.to_string()))?
                {
                    let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                    if let Some(prepared) = workspace.as_mut() {
                        if !self
                            .kernel
                            .refresh_batch(
                                &self.context,
                                &self.preparation_params,
                                &batch,
                                prepared,
                            )
                            .map_err(wgpu_error)?
                        {
                            *prepared = self
                                .kernel
                                .prepare_batch(&self.context, &self.preparation_params, &batch)
                                .map_err(wgpu_error)?;
                        }
                    } else {
                        *workspace = Some(
                            self.kernel
                                .prepare_batch(&self.context, &self.preparation_params, &batch)
                                .map_err(wgpu_error)?,
                        );
                    }
                    total.push(
                        self.kernel
                            .reduce_prepared_batch(
                                &self.context,
                                params,
                                workspace
                                    .as_ref()
                                    .expect("streaming workspace was initialized"),
                                reduction,
                            )
                            .map_err(wgpu_error)?,
                    );
                }
            }
        }
        Ok(execution.sum_f64(total.finish()))
    }

    fn reduce_with_gradient(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &WgpuPreparedDataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<ReductionEvaluation> {
        let mut total = AccurateF64::zero();
        let mut gradient = (0..params.layout().n_free())
            .map(|_| AccurateF64::zero())
            .collect::<Vec<_>>();
        let mut consume = |batch: &laddu_wgpu::WgpuPreparedBatch| -> RuntimeResult<()> {
            let (value, values) = self
                .kernel
                .reduce_prepared_batch_with_gradient(&self.context, params, batch, reduction)
                .map_err(wgpu_error)?;
            total.push(value);
            for (sum, value) in gradient.iter_mut().zip(values) {
                sum.push(value);
            }
            Ok(())
        };
        match dataset {
            WgpuPreparedDataset::Resident { batches, .. } => {
                for batch in batches {
                    consume(batch)?;
                }
            }
            WgpuPreparedDataset::Streaming {
                dataset,
                read_plan,
                workspace,
                transient_bytes,
                ..
            } => {
                let _memory = execution
                    .device_memory()
                    .ok_or_else(|| {
                        RuntimeError::Wgpu("GPU execution has no device memory pool".into())
                    })?
                    .reserve(*transient_bytes)?;
                let mut workspace = workspace.lock().map_err(|_| {
                    RuntimeError::Wgpu("streaming workspace lock is poisoned".into())
                })?;
                for batch in dataset
                    .batches_with_plan(*read_plan)
                    .map_err(|error| RuntimeError::Data(error.to_string()))?
                {
                    let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                    if let Some(prepared) = workspace.as_mut() {
                        if !self
                            .kernel
                            .refresh_batch(
                                &self.context,
                                &self.preparation_params,
                                &batch,
                                prepared,
                            )
                            .map_err(wgpu_error)?
                        {
                            *prepared = self
                                .kernel
                                .prepare_batch(&self.context, &self.preparation_params, &batch)
                                .map_err(wgpu_error)?;
                        }
                    } else {
                        *workspace = Some(
                            self.kernel
                                .prepare_batch(&self.context, &self.preparation_params, &batch)
                                .map_err(wgpu_error)?,
                        );
                    }
                    consume(
                        workspace
                            .as_ref()
                            .expect("streaming workspace was initialized"),
                    )?;
                }
            }
        }
        let gradient = gradient
            .into_iter()
            .map(|sum| execution.sum_f64(sum.finish()))
            .collect();
        Ok(ReductionEvaluation::new(
            execution.sum_f64(total.finish()),
            gradient,
        ))
    }
}

#[cfg(feature = "wgpu")]
fn wgpu_error(error: laddu_wgpu::WgpuError) -> RuntimeError {
    RuntimeError::Wgpu(error.to_string())
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use std::sync::Arc;

    use laddu_compile::{CompiledModel, ReductionPlan};
    use laddu_data::{
        data::{Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{complex, event_scalar, parameter};

    use super::*;
    use crate::{CpuOptions, Device, ExecutionOptions, GpuBackend, GpuOptions, Precision};

    #[test]
    #[ignore = "requires a WGPU-compatible hardware adapter"]
    fn wgpu_resident_and_streaming_reductions_match_f32_cpu() {
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.25));
        let offset = laddu_expr::Expr::from(parameter!("offset", initial: 0.5));
        let x = event_scalar("x");
        let expression = (x.clone() * scale.clone() + offset.clone()).sin()
            + complex(scale, offset).norm_sqr()
            + 2.0;
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = model.params().default_values();
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let dataset = Dataset::from_batches(vec![
            EventBatch::from_events(
                schema.clone(),
                [
                    OwnedEvent::weighted(vec![], vec![0.25], 0.5),
                    OwnedEvent::weighted(vec![], vec![0.75], 1.5),
                ],
            )
            .unwrap(),
            EventBatch::from_events(schema, [OwnedEvent::weighted(vec![], vec![1.25], 2.0)])
                .unwrap(),
        ])
        .unwrap();
        let wgpu_execution = Execution::local(ExecutionOptions {
            device: Device::Gpu(GpuOptions {
                backend: GpuBackend::Wgpu,
                ..GpuOptions::default()
            }),
            memory: crate::MemoryPlan::host_device(
                crate::MemoryBudget::Auto,
                crate::MemoryBudget::Bytes(256),
            ),
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let cpu_execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let wgpu = PreparedModel::prepare(&model, &wgpu_execution).unwrap();
        let cpu = PreparedModel::prepare(&model, &cpu_execution).unwrap();
        let resident = wgpu
            .prepare_dataset(&wgpu_execution, &dataset.clone().resident())
            .unwrap();
        let streaming = wgpu
            .prepare_dataset(&wgpu_execution, &dataset.clone().streaming())
            .unwrap();
        let cpu_data = cpu.prepare_dataset(&cpu_execution, &dataset).unwrap();

        assert_eq!(resident.stats().storage(), CacheStorage::Resident);
        assert_eq!(streaming.stats().storage(), CacheStorage::Streaming);
        assert!(resident.stats().resident_bytes() > 0);
        assert_eq!(streaming.stats().resident_bytes(), 0);

        let cpu_reduction = cpu
            .reduce_with_gradient(
                &cpu_execution,
                &params,
                &cpu_data,
                ReductionPlan::weighted_real(),
            )
            .unwrap();
        let resident_reduction = wgpu
            .reduce_with_gradient(
                &wgpu_execution,
                &params,
                &resident,
                ReductionPlan::weighted_real(),
            )
            .unwrap();
        let streaming_reduction = wgpu
            .reduce_with_gradient(
                &wgpu_execution,
                &params,
                &streaming,
                ReductionPlan::weighted_real(),
            )
            .unwrap();

        for actual in [&resident_reduction, &streaming_reduction] {
            assert!((actual.value() - cpu_reduction.value()).abs() <= 1.0e-4);
            assert_eq!(actual.gradient().len(), cpu_reduction.gradient().len());
            for (actual, expected) in actual.gradient().iter().zip(cpu_reduction.gradient()) {
                assert!((actual - expected).abs() <= 1.0e-4);
            }
        }
        assert!((resident_reduction.value() - streaming_reduction.value()).abs() <= 1.0e-6);
        for (resident, streaming) in resident_reduction
            .gradient()
            .iter()
            .zip(streaming_reduction.gradient())
        {
            assert!((resident - streaming).abs() <= 1.0e-6);
        }
    }
}
