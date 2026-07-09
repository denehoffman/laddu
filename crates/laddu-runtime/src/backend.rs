use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::data::Dataset;
#[cfg(feature = "wgpu")]
use laddu_data::data::{CacheStorage, accurate::AccurateF64};
use laddu_expr::parameters::ParamValues;

use crate::{
    CpuBackend, CpuPlan, CpuPreparedDataset, Execution, PreparedDatasetStats, ReductionEvaluation,
    RuntimeError, RuntimeResult,
};

#[derive(Clone, Debug)]
pub enum PreparedModel {
    Cpu(CpuPlan),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuPlan),
}

#[derive(Clone, Debug)]
pub enum PreparedDataset {
    Cpu(CpuPreparedDataset),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuPreparedDataset),
}

impl PreparedDataset {
    pub fn stats(&self) -> &PreparedDatasetStats {
        match self {
            Self::Cpu(dataset) => dataset.stats(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(dataset) => dataset.stats(),
        }
    }
}

impl PreparedModel {
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
        Ok(Self::Cpu(
            CpuBackend.prepare_for_execution(model, execution)?,
        ))
    }

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
#[derive(Clone)]
pub enum WgpuPreparedDataset {
    Resident {
        batches: Vec<laddu_wgpu::WgpuPreparedBatch>,
        stats: PreparedDatasetStats,
    },
    Streaming {
        dataset: Dataset,
        read_plan: laddu_data::io::ReadPlan,
        workspace: std::sync::Arc<std::sync::Mutex<Vec<laddu_wgpu::WgpuPreparedBatch>>>,
        stats: PreparedDatasetStats,
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
            if dataset.cache_storage() == CacheStorage::Resident {
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
            dataset.cache_storage(),
        );
        Ok(match dataset.cache_storage() {
            CacheStorage::Resident => WgpuPreparedDataset::Resident { batches, stats },
            CacheStorage::Streaming => WgpuPreparedDataset::Streaming {
                dataset: dataset.clone(),
                read_plan,
                workspace: Default::default(),
                stats,
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
                ..
            } => {
                let mut workspace = workspace.lock().map_err(|_| {
                    RuntimeError::Wgpu("streaming workspace lock is poisoned".into())
                })?;
                let mut batch_index = 0;
                for batch in dataset
                    .batches_with_plan(*read_plan)
                    .map_err(|error| RuntimeError::Data(error.to_string()))?
                {
                    let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                    if let Some(prepared) = workspace.get_mut(batch_index) {
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
                        workspace.push(
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
                                &workspace[batch_index],
                                reduction,
                            )
                            .map_err(wgpu_error)?,
                    );
                    batch_index += 1;
                }
                workspace.truncate(batch_index);
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
                ..
            } => {
                let mut workspace = workspace.lock().map_err(|_| {
                    RuntimeError::Wgpu("streaming workspace lock is poisoned".into())
                })?;
                let mut batch_index = 0;
                for batch in dataset
                    .batches_with_plan(*read_plan)
                    .map_err(|error| RuntimeError::Data(error.to_string()))?
                {
                    let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                    if let Some(prepared) = workspace.get_mut(batch_index) {
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
                        workspace.push(
                            self.kernel
                                .prepare_batch(&self.context, &self.preparation_params, &batch)
                                .map_err(wgpu_error)?,
                        );
                    }
                    consume(&workspace[batch_index])?;
                    batch_index += 1;
                }
                workspace.truncate(batch_index);
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
