use laddu_compile::{
    CompiledModel, NormalizationDiagnostics, NormalizationStrategy, ReductionPlan,
};
use laddu_data::{
    data::{CacheStorage, Dataset},
    io::ReadPlan,
};
use laddu_expr::parameters::{ParamLayout, ParamValues};
use laddu_memory::{MemoryFitRequest, MemoryFootprint};
use num::complex::{Complex32, Complex64};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::{
    CpuBackend, CpuPlan, Execution, MemoryLease, NormalizationMode, PreparedDataset,
    PreparedDatasetStats, PreparedModel, RuntimeError, RuntimeResult,
};

const AUTO_BREAK_EVEN_EVALUATIONS: usize = 16;

/// Runtime diagnostics for one compiler-native accepted normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedNormalizationDiagnostics {
    strategy: NormalizationStrategy,
    compiler: NormalizationDiagnostics,
    retained_bytes: usize,
    preparation_passes: usize,
    cache_hit: bool,
    tag_projection_reused_parent: bool,
}

impl PreparedNormalizationDiagnostics {
    /// Returns the runtime-selected normalization strategy.
    pub fn strategy(&self) -> NormalizationStrategy {
        self.strategy
    }

    /// Returns compiler analysis diagnostics.
    pub fn compiler(&self) -> &NormalizationDiagnostics {
        &self.compiler
    }

    /// Returns retained sufficient-statistic bytes on this rank.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Returns the number of accepted-source passes used during preparation.
    pub fn preparation_passes(&self) -> usize {
        self.preparation_passes
    }

    /// Returns whether an execution-scoped prepared artifact was reused.
    pub fn cache_hit(&self) -> bool {
        self.cache_hit
    }

    /// Returns whether a tag projection reused parent statistics.
    pub fn tag_projection_reused_parent(&self) -> bool {
        self.tag_projection_reused_parent
    }

    /// Constructs diagnostics for an ordinary prepared event reduction.
    #[doc(hidden)]
    pub fn general(compiler: NormalizationDiagnostics) -> Self {
        Self {
            strategy: NormalizationStrategy::General,
            compiler,
            retained_bytes: 0,
            preparation_passes: 1,
            cache_hit: false,
            tag_projection_reused_parent: false,
        }
    }
}

#[derive(Clone, Debug)]
struct GeneralResidual {
    plan: PreparedModel,
    dataset: PreparedDataset,
    parameters: ParameterMapping,
}

#[derive(Debug)]
enum StoredStatistics {
    F32(Vec<Complex32>),
    F64(Vec<Complex64>),
}

impl StoredStatistics {
    fn from_f64(values: Vec<Complex64>, precision: crate::Precision) -> Self {
        if precision == crate::Precision::F32 {
            Self::F32(
                values
                    .into_iter()
                    .map(|value| Complex32::new(value.re as f32, value.im as f32))
                    .collect(),
            )
        } else {
            Self::F64(values)
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            Self::F32(values) => values.capacity() * std::mem::size_of::<Complex32>(),
            Self::F64(values) => values.capacity() * std::mem::size_of::<Complex64>(),
        }
    }

    fn evaluator_values(&self) -> Vec<Complex64> {
        match self {
            Self::F32(values) => values
                .iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
            Self::F64(values) => values.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct ParameterMapping {
    layout: ParamLayout,
    child_to_parent_free: Vec<usize>,
    parent_free: usize,
}

impl ParameterMapping {
    fn new(child: &ParamLayout, parent: &ParamLayout) -> RuntimeResult<Self> {
        let child_to_parent_free = child
            .free_params()
            .iter()
            .map(|child_id| {
                let name = child
                    .name(*child_id)
                    .map_err(|error| RuntimeError::Parameter(error.to_string()))?;
                let parent_id = parent.id(name).ok_or_else(|| {
                    RuntimeError::Data(format!(
                        "normalization parameter `{name}` is absent from the source model"
                    ))
                })?;
                parent
                    .free_id(parent_id)
                    .map_err(|error| RuntimeError::Parameter(error.to_string()))?
                    .map(|id| id.index())
                    .ok_or_else(|| {
                        RuntimeError::Data(format!(
                            "normalization parameter `{name}` is unexpectedly fixed in the source model"
                        ))
                    })
            })
            .collect::<RuntimeResult<Vec<_>>>()?;
        Ok(Self {
            layout: child.clone(),
            child_to_parent_free,
            parent_free: parent.n_free(),
        })
    }

    fn project(&self, parent: &ParamValues) -> RuntimeResult<ParamValues> {
        let free = self
            .layout
            .free_params()
            .iter()
            .map(|child_id| {
                let name = self
                    .layout
                    .name(*child_id)
                    .map_err(|error| RuntimeError::Parameter(error.to_string()))?;
                let parent_id = parent.layout().id(name).ok_or_else(|| {
                    RuntimeError::Data(format!(
                        "normalization parameter `{name}` is absent from supplied values"
                    ))
                })?;
                parent
                    .get(parent_id)
                    .map_err(|error| RuntimeError::Parameter(error.to_string()))
            })
            .collect::<RuntimeResult<Vec<_>>>()?;
        self.layout
            .values(&free)
            .map_err(|error| RuntimeError::Parameter(error.to_string()))
    }

    fn scatter_add(&self, child: &[f64], parent: &mut [f64]) -> RuntimeResult<()> {
        if child.len() != self.child_to_parent_free.len() || parent.len() != self.parent_free {
            return Err(RuntimeError::Data(
                "normalization gradient has an incompatible parameter layout".into(),
            ));
        }
        for (value, parent_index) in child.iter().zip(&self.child_to_parent_free) {
            parent[*parent_index] += value;
        }
        Ok(())
    }
}

/// Prepared sufficient statistics and their parameter-only contraction.
#[derive(Debug)]
pub struct PreparedNormalization {
    evaluator: CpuPlan,
    evaluator_parameters: ParameterMapping,
    statistics: StoredStatistics,
    residual: Option<GeneralResidual>,
    verification: Option<GeneralResidual>,
    stats: PreparedDatasetStats,
    diagnostics: PreparedNormalizationDiagnostics,
    cache_reused: AtomicBool,
    _memory_lease: MemoryLease,
}

impl PreparedNormalization {
    /// Prepares compiler-native normalization when selected by execution policy.
    ///
    /// # Errors
    ///
    /// Returns a runtime error when basis compilation, dataset traversal,
    /// memory reservation, or backend preparation fails.
    pub fn prepare(
        model: &CompiledModel,
        general_plan: &PreparedModel,
        dataset: &Dataset,
        execution: &Execution,
    ) -> RuntimeResult<Option<Arc<Self>>> {
        if execution.normalization_mode() == NormalizationMode::General
            || model.normalization_diagnostics().strategy() == NormalizationStrategy::General
            || (execution.normalization_mode() == NormalizationMode::Auto
                && !model.normalization_plan().proven_nonnegative())
        {
            return Ok(None);
        }

        let key = (
            model.optimized_digest(),
            dataset.identity(),
            execution.normalization_mode(),
        );
        let mut cache = execution
            .normalization_cache()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cache.retain(|_, prepared| prepared.strong_count() > 0);
        if let Some(prepared) = cache.get(&key).and_then(std::sync::Weak::upgrade) {
            prepared.cache_reused.store(true, Ordering::Relaxed);
            return Ok(Some(prepared));
        }
        let Some(prepared) = Self::prepare_uncached(model, general_plan, dataset, execution)?
        else {
            return Ok(None);
        };
        let prepared = Arc::new(prepared);
        cache.insert(key, Arc::downgrade(&prepared));
        Ok(Some(prepared))
    }

    fn prepare_uncached(
        model: &CompiledModel,
        general_plan: &PreparedModel,
        dataset: &Dataset,
        execution: &Execution,
    ) -> RuntimeResult<Option<Self>> {
        let basis_models = model
            .normalization_plan()
            .basis_models()
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        let basis_work = basis_models
            .iter()
            .map(|basis| basis.graph().nodes().len())
            .sum::<usize>();
        let general_work = model.graph().nodes().len().max(1);
        if execution.normalization_mode() == NormalizationMode::Auto
            && basis_work > general_work.saturating_mul(AUTO_BREAK_EVEN_EVALUATIONS)
        {
            return Ok(None);
        }

        let statistic_bytes = if execution.precision() == crate::Precision::F32 {
            std::mem::size_of::<Complex32>()
        } else {
            std::mem::size_of::<Complex64>()
        };
        let retained_bytes = basis_models.len().saturating_mul(statistic_bytes);
        let memory_lease = match execution
            .host_memory()
            .reserve(u64::try_from(retained_bytes).unwrap_or(u64::MAX))
        {
            Ok(lease) => lease,
            Err(_) if execution.normalization_mode() == NormalizationMode::Auto => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let basis_plans = basis_models
            .iter()
            .map(|basis| {
                CpuBackend
                    .prepare_with_autodiff_mode(basis, execution.autodiff_mode())
                    .map_err(|error| RuntimeError::Data(error.to_string()))
            })
            .collect::<RuntimeResult<Vec<_>>>()?;
        let basis_params = basis_models
            .iter()
            .map(|basis| basis.params().default_values())
            .collect::<Vec<_>>();
        let (statistics, stats) =
            accumulate_statistics(&basis_plans, &basis_params, dataset, execution)?;
        let statistics = StoredStatistics::from_f64(statistics, execution.precision());
        let evaluator_statistics = statistics.evaluator_values();
        let evaluator_model = model
            .normalization_plan()
            .evaluator_model(&evaluator_statistics)
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        // Sufficient statistics may be prepared for an f32 accelerator, but
        // their tiny parameter-only contraction stays on the CPU in f64 so
        // value/gradient evaluation remains available and numerically stable.
        let evaluator = CpuBackend
            .prepare_with_autodiff_mode(&evaluator_model, execution.autodiff_mode())
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        let evaluator_parameters = ParameterMapping::new(evaluator_model.params(), model.params())?;

        let residual_model = model
            .normalization_plan()
            .residual_model()
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        let residual = if let Some(residual_model) = residual_model {
            let parameters = ParameterMapping::new(residual_model.params(), model.params())?;
            let plan = PreparedModel::prepare(&residual_model, execution)?;
            let dataset = plan.prepare_dataset(execution, dataset)?;
            Some(GeneralResidual {
                plan,
                dataset,
                parameters,
            })
        } else {
            None
        };
        let verification = if execution.normalization_mode() == NormalizationMode::Verify {
            Some(GeneralResidual {
                plan: general_plan.clone(),
                dataset: general_plan.prepare_dataset(execution, dataset)?,
                parameters: ParameterMapping::new(model.params(), model.params())?,
            })
        } else {
            None
        };
        let preparation_passes =
            1 + usize::from(residual.is_some()) + usize::from(verification.is_some());
        Ok(Some(Self {
            evaluator,
            evaluator_parameters,
            statistics,
            residual,
            verification,
            stats,
            diagnostics: PreparedNormalizationDiagnostics {
                strategy: model.normalization_diagnostics().strategy(),
                compiler: model.normalization_diagnostics().clone(),
                retained_bytes,
                preparation_passes,
                cache_hit: false,
                tag_projection_reused_parent: false,
            },
            cache_reused: AtomicBool::new(false),
            _memory_lease: memory_lease,
        }))
    }

    /// Returns accepted-dataset statistics collected during preparation.
    pub fn stats(&self) -> &PreparedDatasetStats {
        &self.stats
    }

    /// Returns normalization preparation diagnostics.
    pub fn diagnostics(&self) -> PreparedNormalizationDiagnostics {
        let mut diagnostics = self.diagnostics.clone();
        diagnostics.cache_hit = self.cache_reused.load(Ordering::Relaxed);
        diagnostics
    }

    /// Returns retained sufficient-statistic storage in bytes.
    pub fn resident_bytes(&self) -> usize {
        self.statistics.resident_bytes()
    }

    /// Evaluates the accepted normalization without constructing a gradient.
    ///
    /// # Errors
    ///
    /// Returns a runtime error for incompatible parameters, residual backend
    /// failures, or a verification mismatch.
    pub fn value(&self, params: &ParamValues, execution: &Execution) -> RuntimeResult<f64> {
        let evaluator_params = self.evaluator_parameters.project(params)?;
        let mut value = self.evaluator.evaluate(&evaluator_params)?.re;
        if let Some(residual) = &self.residual {
            let residual_params = residual.parameters.project(params)?;
            value += residual.plan.reduce(
                execution,
                &residual_params,
                &residual.dataset,
                ReductionPlan::weighted_real(),
            )?;
        }
        if let Some(general) = &self.verification {
            let general_params = general.parameters.project(params)?;
            let expected = general.plan.reduce(
                execution,
                &general_params,
                &general.dataset,
                ReductionPlan::weighted_real(),
            )?;
            verify_close("normalization value", value, expected, execution)?;
        }
        Ok(value)
    }

    /// Evaluates the accepted normalization and its local free-parameter gradient.
    ///
    /// # Errors
    ///
    /// Returns a runtime error for incompatible parameters, autodiff/backend
    /// failures, or a verification mismatch.
    pub fn value_gradient(
        &self,
        params: &ParamValues,
        execution: &Execution,
    ) -> RuntimeResult<(f64, Vec<f64>)> {
        let evaluator_params = self.evaluator_parameters.project(params)?;
        let evaluation = self.evaluator.evaluate_with_gradient(&evaluator_params)?;
        let mut value = evaluation.value().re;
        let evaluator_gradient = evaluation
            .gradient()
            .iter()
            .map(|value| value.re)
            .collect::<Vec<_>>();
        let mut gradient = vec![0.0; self.evaluator_parameters.parent_free];
        self.evaluator_parameters
            .scatter_add(&evaluator_gradient, &mut gradient)?;
        if let Some(residual) = &self.residual {
            let residual_params = residual.parameters.project(params)?;
            let residual_evaluation = residual.plan.reduce_with_gradient(
                execution,
                &residual_params,
                &residual.dataset,
                ReductionPlan::weighted_real(),
            )?;
            value += residual_evaluation.value();
            residual
                .parameters
                .scatter_add(residual_evaluation.gradient(), &mut gradient)?;
        }
        if let Some(general) = &self.verification {
            let general_params = general.parameters.project(params)?;
            let expected = general.plan.reduce_with_gradient(
                execution,
                &general_params,
                &general.dataset,
                ReductionPlan::weighted_real(),
            )?;
            verify_close("normalization value", value, expected.value(), execution)?;
            for (index, (actual, expected)) in gradient.iter().zip(expected.gradient()).enumerate()
            {
                verify_close(
                    &format!("normalization gradient[{index}]"),
                    *actual,
                    *expected,
                    execution,
                )?;
            }
        }
        Ok((value, gradient))
    }
}

fn accumulate_statistics(
    plans: &[CpuPlan],
    params: &[ParamValues],
    dataset: &Dataset,
    execution: &Execution,
) -> RuntimeResult<(Vec<Complex64>, PreparedDatasetStats)> {
    let mut sums = vec![Complex64::ZERO; plans.len()];
    let mut corrections = vec![Complex64::ZERO; plans.len()];
    let mut read_plan: ReadPlan = execution.read_plan(dataset.read_plan());
    let local_limit = dataset
        .num_events()
        .map_err(|error| RuntimeError::Data(error.to_string()))?
        .and_then(|events| usize::try_from(events).ok())
        .unwrap_or(usize::MAX);
    let statistic_bytes = plans.len().saturating_mul(std::mem::size_of::<Complex64>());
    let decision = MemoryFitRequest {
        label: "normalization statistics".into(),
        footprint: MemoryFootprint::from_usize(statistic_bytes, statistic_bytes),
        available_bytes: execution.host_memory().remaining(),
        event_limit: local_limit,
        strategy: "single-pass sufficient statistics".into(),
    }
    .evaluate()?;
    read_plan.chunk_size = Some(
        read_plan
            .chunk_size
            .map_or(decision.chunk_events, |manual| {
                manual.min(decision.chunk_events)
            })
            .max(1),
    );
    execution.record_memory_decision(decision);
    let local = (|| {
        let mut events = 0usize;
        let mut batches = 0usize;
        let mut weight_sum = 0.0;
        let mut weight_correction = 0.0;
        for batch in dataset
            .batches_with_plan(read_plan)
            .map_err(|error| RuntimeError::Data(error.to_string()))?
        {
            let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
            events += batch.len();
            batches += 1;
            for row in 0..batch.len() {
                let weight = batch.weights_at(row);
                let corrected = weight - weight_correction;
                let next = weight_sum + corrected;
                weight_correction = (next - weight_sum) - corrected;
                weight_sum = next;
            }
            for (index, (plan, params)) in plans.iter().zip(params).enumerate() {
                for (row, value) in plan.evaluate_batch(params, &batch)?.into_iter().enumerate() {
                    let value = value * batch.weights_at(row);
                    let corrected = value - corrections[index];
                    let next = sums[index] + corrected;
                    corrections[index] = (next - sums[index]) - corrected;
                    sums[index] = next;
                }
            }
        }
        Ok::<_, RuntimeError>((events, batches, weight_sum))
    })();
    if !execution.all_succeeded(local.is_ok()) {
        return local.and(Err(RuntimeError::DistributedPeerFailure));
    }
    let (events, batches, weight_sum) = local?;
    for sum in &mut sums {
        sum.re = execution.sum_f64(sum.re);
        sum.im = execution.sum_f64(sum.im);
    }
    let stats = PreparedDatasetStats::new(
        events,
        execution.sum_usize(events),
        batches,
        execution.sum_f64(weight_sum),
        sums.len() * std::mem::size_of::<Complex64>(),
        CacheStorage::Resident,
    );
    Ok((sums, stats))
}

fn verify_close(
    label: &str,
    actual: f64,
    expected: f64,
    execution: &Execution,
) -> RuntimeResult<()> {
    let tolerance = match execution.precision() {
        crate::Precision::F32 => 5.0e-4,
        crate::Precision::Auto | crate::Precision::F64 => 1.0e-10,
    } * expected.abs().max(1.0);
    if (actual - expected).abs() <= tolerance {
        Ok(())
    } else {
        Err(RuntimeError::Data(format!(
            "{label} verification failed: compiler-native={actual}, general={expected}, tolerance={tolerance}"
        )))
    }
}
