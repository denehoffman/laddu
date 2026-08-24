use laddu_data::{
    data::{CacheStorage, Dataset, EventBatch, MemoryPolicy, accurate::AccurateF64},
    io::ReadPlan,
};
use laddu_memory::{MemoryDecision, MemoryError, MemoryFootprint, MemoryLease, MemoryPool};

use crate::{Execution, PreparedDatasetStats, RuntimeError, RuntimeResult};

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct LocalDatasetStats {
    pub(crate) events: usize,
    pub(crate) batches: usize,
    pub(crate) sum_weights: f64,
}

#[derive(Clone)]
pub(crate) struct DatasetStatsAccumulator {
    events: usize,
    batches: usize,
    sum_weights: AccurateF64,
}

pub(crate) struct DatasetPreparation<'a> {
    execution: &'a Execution,
    dataset: &'a Dataset,
    read_plan: ReadPlan,
}

/// Backend-neutral preparation decisions shared by CPU and WGPU datasets.
///
/// Backends still own the footprints that describe their resources. This
/// module owns the policy around those inputs: fitting staging work, selecting
/// resident versus streaming storage, retaining a persistent lease, and
/// clamping the source read plan to the selected event chunk. Keeping that
/// policy here makes the preparation interface small while preserving the
/// backend-specific memory formulas at their call sites.
#[derive(Clone, Debug)]
pub(crate) struct RuntimePreparationPlan {
    read_plan: ReadPlan,
    event_limit: usize,
    storage: Option<PreparedStoragePlan>,
    memory_lease: Option<MemoryLease>,
    decisions: Vec<MemoryDecision>,
}

impl RuntimePreparationPlan {
    pub(crate) fn new(read_plan: ReadPlan, event_limit: usize) -> Self {
        Self {
            read_plan,
            event_limit,
            storage: None,
            memory_lease: None,
            decisions: Vec::new(),
        }
    }

    pub(crate) fn read_plan(&self) -> ReadPlan {
        self.read_plan
    }

    pub(crate) fn event_limit(&self) -> usize {
        self.event_limit
    }

    /// Fits a backend-provided staging footprint and records its decision in
    /// the order in which the backend requested it.
    pub(crate) fn fit_staging(
        &mut self,
        label: impl Into<String>,
        footprint: MemoryFootprint,
        available_bytes: u64,
        strategy: impl Into<String>,
    ) -> RuntimeResult<MemoryDecision> {
        let decision = laddu_memory::MemoryFitRequest {
            label: label.into(),
            footprint,
            available_bytes,
            event_limit: self.event_limit,
            strategy: strategy.into(),
        }
        .evaluate()?;
        self.decisions.push(decision.clone());
        Ok(decision)
    }

    /// Records a resident decision when a backend's resident footprint is
    /// measured over the full dataset but its batches use a staging chunk.
    #[cfg(feature = "wgpu")]
    pub(crate) fn fit_resident(
        &mut self,
        label: impl Into<String>,
        footprint: MemoryFootprint,
        available_bytes: u64,
        strategy: impl Into<String>,
        chunk_events: usize,
    ) -> RuntimeResult<MemoryDecision> {
        let decision = laddu_memory::MemoryFitRequest {
            label: label.into(),
            footprint,
            available_bytes,
            event_limit: self.event_limit,
            strategy: strategy.into(),
        }
        .evaluate_resident(chunk_events)?;
        self.decisions.push(decision.clone());
        Ok(decision)
    }

    pub(crate) fn select_storage(
        &mut self,
        policy: MemoryPolicy,
        resource: &str,
        resident_feasible: bool,
        resident_bytes: u64,
        available: u64,
    ) -> RuntimeResult<CacheStorage> {
        let storage = PreparedStoragePlan::resolve(
            policy,
            resource,
            resident_feasible,
            resident_bytes,
            available,
        )?;
        let selected = storage.storage();
        self.storage = Some(storage);
        Ok(selected)
    }

    pub(crate) fn reserve_storage(
        &mut self,
        pool: Option<&MemoryPool>,
        missing_pool: impl FnOnce() -> RuntimeError,
    ) -> RuntimeResult<()> {
        let storage = self.storage.ok_or_else(|| {
            RuntimeError::Data("storage was not selected before reservation".into())
        })?;
        let lease = storage.reserve(pool, missing_pool)?;
        self.memory_lease = lease;
        Ok(())
    }

    pub(crate) fn take_memory_lease(&mut self) -> Option<MemoryLease> {
        self.memory_lease.take()
    }

    pub(crate) fn take_decisions(&mut self) -> Vec<MemoryDecision> {
        std::mem::take(&mut self.decisions)
    }

    pub(crate) fn clamp_read_plan(&mut self, manual_chunk: Option<usize>, chunk_events: usize) {
        self.read_plan.chunk_size = Some(
            manual_chunk
                .map_or(chunk_events, |manual| manual.min(chunk_events))
                .max(1),
        );
    }

    pub(crate) fn storage(&self) -> CacheStorage {
        self.storage
            .expect("storage must be selected before it is read")
            .storage()
    }
}

impl<'a> DatasetPreparation<'a> {
    pub(crate) fn new(execution: &'a Execution, dataset: &'a Dataset) -> Self {
        Self {
            execution,
            dataset,
            read_plan: execution.read_plan(dataset.read_plan()),
        }
    }

    pub(crate) fn runtime_plan(&self) -> RuntimeResult<RuntimePreparationPlan> {
        Ok(RuntimePreparationPlan::new(
            self.read_plan,
            self.event_limit()?,
        ))
    }

    pub(crate) fn event_limit(&self) -> RuntimeResult<usize> {
        if let Some(events) = self
            .dataset
            .num_events()
            .map_err(|error| RuntimeError::Data(error.to_string()))?
            .and_then(|events| usize::try_from(events).ok())
        {
            return Ok(events);
        }
        if self.dataset.memory_policy() == MemoryPolicy::Streaming {
            return Ok(usize::MAX);
        }
        Ok(self.scan()?.events)
    }

    pub(crate) fn scan(&self) -> RuntimeResult<LocalDatasetStats> {
        let local = (|| {
            let mut stats = DatasetStatsAccumulator::new();
            for batch in self
                .dataset
                .stream_with_plan(self.read_plan)
                .map_err(|error| RuntimeError::Data(error.to_string()))?
            {
                let batch = batch.map_err(|error| RuntimeError::Data(error.to_string()))?;
                stats.observe(&batch);
            }
            Ok(stats.finish())
        })();
        self.coordinate(local)
    }

    pub(crate) fn coordinate<T>(&self, local: RuntimeResult<T>) -> RuntimeResult<T> {
        if self.execution.all_succeeded(local.is_ok()) {
            local
        } else {
            local.and(Err(RuntimeError::DistributedPeerFailure))
        }
    }

    pub(crate) fn finish_stats(
        &self,
        local: LocalDatasetStats,
        resident_bytes: usize,
        storage: CacheStorage,
    ) -> PreparedDatasetStats {
        PreparedDatasetStats::new(
            local.events,
            self.execution.sum_usize(local.events),
            local.batches,
            self.execution.sum_f64(local.sum_weights),
            resident_bytes,
            storage,
        )
    }
}

impl LocalDatasetStats {
    pub(crate) fn new(events: usize, batches: usize, sum_weights: f64) -> Self {
        Self {
            events,
            batches,
            sum_weights,
        }
    }
}

impl DatasetStatsAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            events: 0,
            batches: 0,
            sum_weights: AccurateF64::zero(),
        }
    }

    pub(crate) fn observe(&mut self, batch: &EventBatch) {
        self.events += batch.len();
        self.batches += 1;
        for row in 0..batch.len() {
            self.sum_weights.push(batch.weights_at(row));
        }
    }

    pub(crate) fn finish(self) -> LocalDatasetStats {
        LocalDatasetStats::new(self.events, self.batches, self.sum_weights.finish())
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PreparedStoragePlan {
    storage: CacheStorage,
    resident_bytes: u64,
}

impl PreparedStoragePlan {
    pub(crate) fn resolve(
        policy: MemoryPolicy,
        resource: &str,
        resident_feasible: bool,
        resident_bytes: u64,
        available: u64,
    ) -> RuntimeResult<Self> {
        let storage = match policy {
            MemoryPolicy::Streaming => CacheStorage::Streaming,
            MemoryPolicy::Resident if !resident_feasible => {
                return Err(MemoryError::BudgetExceeded {
                    resource: resource.into(),
                    requested: resident_bytes,
                    remaining: available,
                }
                .into());
            }
            MemoryPolicy::Resident => CacheStorage::Resident,
            MemoryPolicy::Fastest if resident_feasible => CacheStorage::Resident,
            MemoryPolicy::Fastest => CacheStorage::Streaming,
        };
        Ok(Self {
            storage,
            resident_bytes,
        })
    }

    pub(crate) fn storage(self) -> CacheStorage {
        self.storage
    }

    pub(crate) fn reserve(
        self,
        pool: Option<&MemoryPool>,
        missing_pool: impl FnOnce() -> RuntimeError,
    ) -> RuntimeResult<Option<MemoryLease>> {
        if self.storage == CacheStorage::Streaming {
            return Ok(None);
        }
        Ok(Some(
            pool.ok_or_else(missing_pool)?
                .reserve(self.resident_bytes)?,
        ))
    }
}
