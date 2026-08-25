use laddu_compile::{CompiledModel, ReductionPlan};
#[cfg(feature = "wgpu")]
use laddu_data::BatchLayout;
use laddu_data::data::Dataset;
use laddu_data::data::EventBatch;
#[cfg(feature = "wgpu")]
use laddu_data::data::{CacheStorage, MemoryPolicy, accurate::AccurateF64};
#[cfg(feature = "wgpu")]
use laddu_data::schema::Precision as DataPrecision;
use laddu_expr::parameters::ParamValues;
#[cfg(feature = "wgpu")]
use laddu_memory::{MemoryDecision, MemoryFootprint};
use num::complex::Complex64;

#[cfg(feature = "wgpu")]
use crate::preparation::{DatasetPreparation, DatasetStatsAccumulator, RuntimePreparationPlan};
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
    /// Returns the event-scalar columns required by this prepared model.
    ///
    /// Requirements are deduplicated while retaining compiled graph order.
    pub fn required_event_scalars(&self) -> &[String] {
        match self {
            Self::Cpu(plan) => plan.required_event_scalars(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(plan) => &plan.required_event_scalars,
        }
    }

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

    /// Evaluates a vector-root model and returns one output column per root
    /// element. CPU supports this query-oriented ABI; scalar WGPU kernels
    /// retain their existing single-output contract.
    pub(crate) fn evaluate_batch_outputs(
        &self,
        params: &ParamValues,
        batch: &EventBatch,
        outputs: &[laddu_expr::ExprId],
    ) -> RuntimeResult<Vec<Vec<Complex64>>> {
        match self {
            Self::Cpu(plan) => plan.evaluate_batch_outputs(params, batch, outputs),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => Err(RuntimeError::Wgpu(
                "multi-output query evaluation is not implemented by the WGPU backend".into(),
            )),
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
                required_event_scalars: crate::required_event_scalars(model),
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

    /// Evaluates every event in a prepared dataset while preserving source order.
    ///
    /// Backends with a prepared event adapter reuse their retained or streaming
    /// prepared blocks. Other backends evaluate the supplied source through the
    /// already-selected backend; this operation never substitutes a backend.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model and dataset backends differ, source
    /// streaming fails, or event evaluation fails.
    pub fn evaluate_prepared(
        &self,
        execution: &Execution,
        params: &ParamValues,
        dataset: &PreparedDataset,
        source: &Dataset,
    ) -> RuntimeResult<Vec<Complex64>> {
        let mut output =
            self.evaluate_prepared_many(execution, std::slice::from_ref(params), dataset, source)?;
        Ok(output.pop().unwrap_or_default())
    }

    /// Evaluates multiple parameter sets while each prepared event block is active.
    ///
    /// Output rows retain parameter-set order and each row retains source-event order.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model and dataset backends differ, source
    /// streaming fails, or event evaluation fails.
    pub fn evaluate_prepared_many(
        &self,
        execution: &Execution,
        params: &[ParamValues],
        dataset: &PreparedDataset,
        source: &Dataset,
    ) -> RuntimeResult<Vec<Vec<Complex64>>> {
        #[cfg(not(feature = "wgpu"))]
        let _ = source;
        #[allow(unreachable_patterns)]
        match (self, dataset) {
            (Self::Cpu(plan), PreparedDataset::Cpu(dataset)) => {
                plan.evaluate_prepared_dataset_many(execution, params, dataset)
            }
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(_), PreparedDataset::Wgpu(_)) => {
                evaluate_source_many(self, execution, params, source, None)
                    .map(|(values, _)| values)
            }
            _ => Err(RuntimeError::InvalidShape {
                index: 0,
                message: "prepared model and dataset use different backends".into(),
            }),
        }
    }

    /// Visits one bounded block of prepared values at a time for several
    /// parameter sets without retaining full-dataset value rows.
    #[doc(hidden)]
    pub fn visit_prepared_many<F>(
        &self,
        execution: &Execution,
        parameter_sets: &[(&ParamValues, &str)],
        dataset: &PreparedDataset,
        source: &Dataset,
        reduction: Option<ReductionPlan>,
        consume: F,
    ) -> RuntimeResult<Vec<f64>>
    where
        F: FnMut(usize, usize, &[Complex64]) -> RuntimeResult<()>,
    {
        #[cfg(not(feature = "wgpu"))]
        let _ = source;
        #[allow(unreachable_patterns)]
        match (self, dataset) {
            (Self::Cpu(plan), PreparedDataset::Cpu(dataset)) => plan.visit_prepared_dataset_many(
                execution,
                parameter_sets,
                dataset,
                reduction,
                consume,
            ),
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(_), PreparedDataset::Wgpu(_)) => {
                visit_source_many(self, execution, parameter_sets, source, reduction, consume)
            }
            _ => Err(RuntimeError::InvalidShape {
                index: 0,
                message: "prepared model and dataset use different backends".into(),
            }),
        }
    }

    /// Evaluates and reduces multiple parameter sets while each prepared block is active.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when model and dataset backends differ, source
    /// streaming fails, event evaluation fails, or the reduction rejects a value.
    pub fn evaluate_prepared_many_with_reduction(
        &self,
        execution: &Execution,
        params: &[ParamValues],
        dataset: &PreparedDataset,
        source: &Dataset,
        reduction: ReductionPlan,
    ) -> RuntimeResult<(Vec<Vec<Complex64>>, Vec<f64>)> {
        #[cfg(not(feature = "wgpu"))]
        let _ = source;
        #[allow(unreachable_patterns)]
        match (self, dataset) {
            (Self::Cpu(plan), PreparedDataset::Cpu(dataset)) => plan
                .evaluate_prepared_dataset_many_with_reduction(
                    execution, params, dataset, reduction,
                ),
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(_), PreparedDataset::Wgpu(_)) => {
                let (values, sums) =
                    evaluate_source_many(self, execution, params, source, Some(reduction))?;
                Ok((values, sums.unwrap_or_default()))
            }
            _ => Err(RuntimeError::InvalidShape {
                index: 0,
                message: "prepared model and dataset use different backends".into(),
            }),
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
type PreparedValuesWithSums = (Vec<Vec<Complex64>>, Option<Vec<f64>>);

#[cfg(feature = "wgpu")]
fn evaluate_source_many(
    model: &PreparedModel,
    execution: &Execution,
    params: &[ParamValues],
    source: &Dataset,
    reduction: Option<ReductionPlan>,
) -> RuntimeResult<PreparedValuesWithSums> {
    let local = (|| {
        let mut output = params.iter().map(|_| Vec::new()).collect::<Vec<_>>();
        let mut sums = reduction.map(|_| {
            params
                .iter()
                .map(|_| AccurateF64::zero())
                .collect::<Vec<_>>()
        });
        for batch in source
            .batches()
            .map_err(|error| RuntimeError::Data(error.to_string()))?
        {
            let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
            for (index, (parameters, values)) in params.iter().zip(&mut output).enumerate() {
                let batch_values = model.evaluate_batch(parameters, &batch)?;
                if let (Some(reduction), Some(sums)) = (reduction, sums.as_mut()) {
                    for (row, value) in batch_values.iter().enumerate() {
                        sums[index].push(batch.weights_at(row) * reduction.apply(*value)?.value());
                    }
                }
                values.extend(batch_values);
            }
        }
        Ok((
            output,
            sums.map(|sums| sums.into_iter().map(AccurateF64::finish).collect()),
        ))
    })();
    if !execution.all_succeeded(local.is_ok()) {
        return local.and(Err(RuntimeError::DistributedPeerFailure));
    }
    let (output, sums) = local?;
    Ok((
        output,
        sums.map(|sums: Vec<f64>| sums.into_iter().map(|sum| execution.sum_f64(sum)).collect()),
    ))
}

#[cfg(feature = "wgpu")]
fn visit_source_many<F>(
    model: &PreparedModel,
    execution: &Execution,
    parameter_sets: &[(&ParamValues, &str)],
    source: &Dataset,
    reduction: Option<ReductionPlan>,
    mut consume: F,
) -> RuntimeResult<Vec<f64>>
where
    F: FnMut(usize, usize, &[Complex64]) -> RuntimeResult<()>,
{
    let local = (|| {
        let mut offset = 0;
        let mut sums = reduction.map(|_| {
            parameter_sets
                .iter()
                .map(|_| AccurateF64::zero())
                .collect::<Vec<_>>()
        });
        for batch in source
            .batches()
            .map_err(|error| RuntimeError::Data(error.to_string()))?
        {
            let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
            for (index, &(parameters, context)) in parameter_sets.iter().enumerate() {
                let values = model.evaluate_batch(parameters, &batch).map_err(|error| {
                    RuntimeError::Parameter(format!("{context} evaluation failed: {error}"))
                })?;
                if let (Some(reduction), Some(sums)) = (reduction, sums.as_mut()) {
                    for (row, value) in values.iter().enumerate() {
                        let reduced = reduction.apply(*value).map_err(|error| {
                            RuntimeError::Parameter(format!("{context} reduction failed: {error}"))
                        })?;
                        sums[index].push(batch.weights_at(row) * reduced.value());
                    }
                }
                consume(offset, index, &values)?;
            }
            offset += batch.len();
        }
        Ok(sums
            .unwrap_or_default()
            .into_iter()
            .map(AccurateF64::finish)
            .collect::<Vec<_>>())
    })();
    if !execution.all_succeeded(local.is_ok()) {
        return local.and(Err(RuntimeError::DistributedPeerFailure));
    }
    Ok(local?
        .into_iter()
        .map(|sum| execution.sum_f64(sum))
        .collect())
}

#[cfg(feature = "wgpu")]
/// A compiled model prepared for WebGPU execution.
#[derive(Clone)]
pub struct WgpuPlan {
    context: std::sync::Arc<laddu_wgpu::WgpuContext>,
    preparation_params: ParamValues,
    kernel: std::sync::Arc<laddu_wgpu::WgpuScalarKernel>,
    required_event_scalars: Vec<String>,
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Debug)]
struct WgpuDatasetPlan {
    read_plan: laddu_data::io::ReadPlan,
    preparation_plan: RuntimePreparationPlan,
    device_decision: MemoryDecision,
}

#[cfg(feature = "wgpu")]
impl WgpuDatasetPlan {
    fn resolve(
        read_plan: laddu_data::io::ReadPlan,
        memory_policy: MemoryPolicy,
        local_event_limit: usize,
        host_footprint: MemoryFootprint,
        prepared_footprint: MemoryFootprint,
        host_available: u64,
        device_available: Option<u64>,
    ) -> RuntimeResult<Self> {
        let mut preparation_plan = RuntimePreparationPlan::new(read_plan, local_event_limit);
        let host_decision = preparation_plan.fit_staging(
            "WGPU host staging",
            host_footprint,
            host_available,
            "bounded host staging",
        )?;
        let host_chunks = local_event_limit
            .saturating_add(host_decision.chunk_events.saturating_sub(1))
            / host_decision.chunk_events.max(1);
        let resident_footprint = MemoryFootprint::fixed(prepared_footprint.fixed_bytes)
            .checked_scale_usize(host_chunks)
            .and_then(|fixed| {
                fixed.checked_add(MemoryFootprint::per_event(
                    prepared_footprint.bytes_per_event,
                ))
            })
            .map_err(|error| RuntimeError::Data(format!("GPU working-set overflow: {error}")))?;
        let resident_peak = resident_footprint.peak_bytes(local_event_limit);
        let device_available = device_available
            .ok_or_else(|| RuntimeError::Wgpu("GPU execution has no device memory pool".into()))?;
        preparation_plan.select_storage(
            memory_policy,
            "device",
            resident_peak <= device_available,
            resident_peak,
            device_available,
        )?;
        let device_decision = if preparation_plan.storage() == CacheStorage::Resident {
            preparation_plan.fit_resident(
                "WGPU prepared dataset",
                resident_footprint,
                device_available,
                "resident",
                host_decision.chunk_events,
            )?
        } else {
            preparation_plan.fit_staging(
                "WGPU prepared dataset",
                prepared_footprint,
                device_available,
                "streaming",
            )?
        };
        let chunk_events = device_decision
            .chunk_events
            .min(host_decision.chunk_events)
            .max(1);
        preparation_plan.clamp_read_plan(read_plan.chunk_size, chunk_events);
        Ok(Self {
            read_plan: preparation_plan.read_plan(),
            preparation_plan,
            device_decision,
        })
    }

    fn storage(&self) -> CacheStorage {
        self.preparation_plan.storage()
    }

    fn reserve_storage(
        &mut self,
        pool: Option<&crate::MemoryPool>,
    ) -> RuntimeResult<Option<crate::MemoryLease>> {
        self.preparation_plan.reserve_storage(pool, || {
            RuntimeError::Wgpu("GPU execution has no device memory pool".into())
        })?;
        Ok(self.preparation_plan.take_memory_lease())
    }
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
        batches: std::sync::Arc<[laddu_wgpu::WgpuPreparedBatch]>,
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

    fn try_for_each_prepared_batch<F>(
        &self,
        execution: &Execution,
        context: &laddu_wgpu::WgpuContext,
        kernel: &laddu_wgpu::WgpuScalarKernel,
        preparation_params: &ParamValues,
        mut consume: F,
    ) -> RuntimeResult<()>
    where
        F: FnMut(&laddu_wgpu::WgpuPreparedBatch) -> RuntimeResult<()>,
    {
        match self {
            Self::Resident { batches, .. } => {
                for batch in batches.iter() {
                    consume(batch)?;
                }
            }
            Self::Streaming {
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
                    .stream_with_plan(*read_plan)
                    .map_err(|error| RuntimeError::Data(error.to_string()))?
                {
                    let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                    if let Some(prepared) = workspace.as_mut() {
                        if !kernel
                            .refresh_batch(context, preparation_params, &batch, prepared)
                            .map_err(wgpu_error)?
                        {
                            *prepared = kernel
                                .prepare_batch(context, preparation_params, &batch)
                                .map_err(wgpu_error)?;
                        }
                    } else {
                        *workspace = Some(
                            kernel
                                .prepare_batch(context, preparation_params, &batch)
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
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
impl WgpuPlan {
    fn prepare_dataset(
        &self,
        execution: &Execution,
        dataset: &Dataset,
    ) -> RuntimeResult<WgpuPreparedDataset> {
        let preparation = DatasetPreparation::new(execution, dataset);
        let preparation_plan = preparation.runtime_plan()?;
        let read_plan = preparation_plan.read_plan();
        let local_event_limit = preparation_plan.event_limit();
        let prepared_footprint = self
            .kernel
            .prepared_memory_footprint(&self.preparation_params)
            .map_err(|error| RuntimeError::Data(format!("GPU working-set overflow: {error}")))?;
        let schema = dataset
            .schema()
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        let host_footprint = BatchLayout::from_schema(&schema)
            .schema_working_set(DataPrecision::F64, 2)
            .map_err(|error| RuntimeError::Data(format!("host working-set overflow: {error}")))?;
        let mut plan = WgpuDatasetPlan::resolve(
            read_plan,
            dataset.memory_policy(),
            local_event_limit,
            host_footprint,
            prepared_footprint,
            execution.host_memory().remaining(),
            execution.device_memory().map(|pool| pool.remaining()),
        )?;
        let memory_lease = plan.reserve_storage(execution.device_memory())?;
        for decision in plan.preparation_plan.take_decisions().into_iter().rev() {
            execution.record_memory_decision(decision);
        }
        let local = (|| {
            let mut batches = Vec::new();
            let mut stats = DatasetStatsAccumulator::new();
            for batch in dataset
                .stream_with_plan(plan.read_plan)
                .map_err(|error| RuntimeError::Data(error.to_string()))?
            {
                let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                stats.observe(&batch);
                if plan.storage() == CacheStorage::Resident {
                    batches.push(
                        self.kernel
                            .prepare_batch(&self.context, &self.preparation_params, &batch)
                            .map_err(wgpu_error)?,
                    );
                }
            }
            Ok::<_, RuntimeError>((batches, stats.finish()))
        })();
        let (batches, local_stats) = preparation.coordinate(local)?;
        let resident_bytes = batches
            .iter()
            .map(laddu_wgpu::WgpuPreparedBatch::resident_bytes)
            .sum();
        let stats = preparation.finish_stats(local_stats, resident_bytes, plan.storage());
        Ok(match plan.storage() {
            CacheStorage::Resident => WgpuPreparedDataset::Resident {
                batches: batches.into(),
                stats,
                memory_lease: memory_lease.ok_or_else(|| {
                    RuntimeError::Wgpu("resident GPU dataset did not reserve device memory".into())
                })?,
            },
            CacheStorage::Streaming => WgpuPreparedDataset::Streaming {
                dataset: dataset.clone(),
                read_plan: plan.read_plan,
                workspace: Default::default(),
                stats,
                transient_bytes: plan.device_decision.estimated_peak_bytes,
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
        dataset.try_for_each_prepared_batch(
            execution,
            &self.context,
            &self.kernel,
            &self.preparation_params,
            |batch| {
                total.push(
                    self.kernel
                        .reduce_prepared_batch(&self.context, params, batch, reduction)
                        .map_err(wgpu_error)?,
                );
                Ok(())
            },
        )?;
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
        dataset.try_for_each_prepared_batch(
            execution,
            &self.context,
            &self.kernel,
            &self.preparation_params,
            &mut consume,
        )?;
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

#[cfg(test)]
mod prepared_evaluation_tests {
    use std::sync::Arc;

    use laddu_compile::{CompiledModel, ReductionPlan};
    use laddu_data::{
        data::{Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{event_scalar, parameter};
    use num::complex::Complex64;

    use super::PreparedModel;
    use crate::Execution;

    fn dataset(streaming: bool) -> Dataset {
        let schema = Arc::new(
            Schema::new(std::iter::empty::<&str>(), ["x"], false)
                .expect("test schema should be valid"),
        );
        let first_batch = EventBatch::from_events(
            schema.clone(),
            [
                OwnedEvent::new(vec![], vec![1.0]),
                OwnedEvent::new(vec![], vec![2.0]),
            ],
        )
        .expect("first test batch should be valid");
        let second_batch = EventBatch::from_events(schema, [OwnedEvent::new(vec![], vec![3.0])])
            .expect("second test batch should be valid");
        let dataset = Dataset::from_batches(vec![first_batch, second_batch])
            .expect("test dataset should be valid");
        if streaming {
            dataset.streaming()
        } else {
            dataset.fastest()
        }
    }

    #[test]
    fn prepared_evaluation_preserves_event_order_for_resident_and_streaming_data() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") + parameter!("offset", initial: 0.5)))
                .expect("test model should compile");
        let execution = Execution::default();
        let prepared_model =
            PreparedModel::prepare(&model, &execution).expect("model should prepare");
        let parameters = [
            model.params().values(&[0.5]).expect("first parameters"),
            model.params().values(&[1.5]).expect("second parameters"),
        ];
        let expected = [
            vec![
                Complex64::new(1.5, 0.0),
                Complex64::new(2.5, 0.0),
                Complex64::new(3.5, 0.0),
            ],
            vec![
                Complex64::new(2.5, 0.0),
                Complex64::new(3.5, 0.0),
                Complex64::new(4.5, 0.0),
            ],
        ];

        for streaming in [false, true] {
            let source = dataset(streaming);
            let prepared = prepared_model
                .prepare_dataset(&execution, &source)
                .expect("dataset should prepare");
            let actual = prepared_model
                .evaluate_prepared_many(&execution, &parameters, &prepared, &source)
                .expect("prepared dataset should evaluate");
            assert_eq!(actual, expected);
            let (actual, sums) = prepared_model
                .evaluate_prepared_many_with_reduction(
                    &execution,
                    &parameters,
                    &prepared,
                    &source,
                    ReductionPlan::weighted_positive_real(),
                )
                .expect("prepared dataset should evaluate and reduce");
            assert_eq!(actual, expected);
            assert_eq!(sums, [7.5, 10.5]);
            let mut visited = vec![Vec::new(), Vec::new()];
            let parameter_sets = [
                (&parameters[0], "central"),
                (&parameters[1], "ensemble draw 0"),
            ];
            let sums = prepared_model
                .visit_prepared_many(
                    &execution,
                    &parameter_sets,
                    &prepared,
                    &source,
                    Some(ReductionPlan::weighted_positive_real()),
                    |_, parameter_index, values| {
                        visited[parameter_index].extend_from_slice(values);
                        Ok(())
                    },
                )
                .expect("prepared blocks should be visited and reduced");
            assert_eq!(visited, expected);
            assert_eq!(sums, [7.5, 10.5]);
        }
    }
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
    fn wgpu_dataset_plan_resolves_storage_and_chunk_limits_without_hardware() {
        let mut read_plan = laddu_data::io::ReadPlan::serial();
        read_plan.chunk_size = Some(3);
        let plan = WgpuDatasetPlan::resolve(
            read_plan,
            MemoryPolicy::Fastest,
            100,
            MemoryFootprint::new(100, 8),
            MemoryFootprint::new(200, 4),
            1_000,
            Some(10_000),
        )
        .unwrap();

        assert_eq!(plan.storage(), CacheStorage::Resident);
        assert_eq!(plan.read_plan.chunk_size, Some(3));
        assert_eq!(plan.device_decision.chunk_events, 100);
        assert_eq!(plan.device_decision.estimated_peak_bytes, 600);
    }

    #[test]
    fn wgpu_dataset_plan_falls_back_to_streaming_when_resident_does_not_fit() {
        let plan = WgpuDatasetPlan::resolve(
            laddu_data::io::ReadPlan::serial(),
            MemoryPolicy::Fastest,
            100,
            MemoryFootprint::new(100, 8),
            MemoryFootprint::new(500, 10),
            1_000,
            Some(1_000),
        )
        .unwrap();

        assert_eq!(plan.storage(), CacheStorage::Streaming);
        assert_eq!(plan.read_plan.chunk_size, Some(50));
        assert_eq!(plan.device_decision.chunk_events, 50);
        assert_eq!(plan.device_decision.estimated_peak_bytes, 1_000);
    }

    #[test]
    fn wgpu_dataset_plan_reports_host_failure_before_missing_device_pool() {
        let error = WgpuDatasetPlan::resolve(
            laddu_data::io::ReadPlan::serial(),
            MemoryPolicy::Fastest,
            1,
            MemoryFootprint::new(100, 8),
            MemoryFootprint::new(200, 4),
            100,
            None,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RuntimeError::Memory(laddu_memory::MemoryError::BudgetExceeded { resource, .. })
                if resource == "WGPU host staging"
        ));
    }

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
