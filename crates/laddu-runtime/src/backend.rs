use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::data::Dataset;
use laddu_data::data::EventBatch;
#[cfg(feature = "wgpu")]
use laddu_data::data::{CacheStorage, accurate::AccurateF64};
use laddu_expr::parameters::ParamValues;
use num::complex::Complex64;

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
                memory_budget: Some(256),
                ..GpuOptions::default()
            }),
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
