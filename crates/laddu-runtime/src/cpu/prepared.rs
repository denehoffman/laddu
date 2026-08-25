use std::{mem::size_of, sync::Arc};

use laddu_data::{
    BatchLayout,
    data::{CacheStorage, Dataset},
    schema::Precision as DataPrecision,
};
use laddu_expr::{ExprId, ValueKind};
use laddu_memory::{FootprintOverflow, MemoryFootprint};
use num::complex::Complex64;

use super::cache::{CachedFactorSlot, CachedSlot, CachedSolveRowSlot};
use super::{
    CpuCachedBatch, CpuCachedDataset, CpuPlan, CpuPreparedDataset, DynamicLu, RuntimeError,
    RuntimeResult,
};
use crate::execution::Execution;
use crate::preparation::{DatasetPreparation, LocalDatasetStats};

impl CpuPlan {
    /// Materializes all event-dependent caches for a dataset.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the dataset cannot be read, a batch schema
    /// is incompatible, cache construction fails, or a matrix is singular.
    pub fn cache_dataset(&self, dataset: &Dataset) -> RuntimeResult<CpuCachedDataset> {
        self.cache_dataset_with_plan(dataset, dataset.read_plan())
    }

    /// Estimates retained compiled-cache bytes for `events`.
    pub fn cache_memory_estimate(&self, events: usize) -> usize {
        self.cache_memory_footprint()
            .map(|footprint| usize::try_from(footprint.peak_bytes(events)).unwrap_or(usize::MAX))
            .unwrap_or(usize::MAX)
    }

    fn cache_memory_footprint(&self) -> Result<MemoryFootprint, FootprintOverflow> {
        let mut fixed = MemoryFootprint::fixed(0);
        for (count, bytes) in [
            (self.cache_plan.entries().len(), size_of::<CachedSlot>()),
            (self.factor_matrices.len(), size_of::<CachedFactorSlot>()),
            (self.solve_row_keys.len(), size_of::<CachedSolveRowSlot>()),
            (self.cache_plan.entries().len(), size_of::<ExprId>()),
            (self.factor_matrices.len(), size_of::<ExprId>()),
            (
                self.solve_row_keys.len(),
                size_of::<(ExprId, usize, usize)>(),
            ),
        ] {
            fixed = fixed.checked_add(
                MemoryFootprint::from_usize_checked(bytes, 0)?.checked_scale_usize(count)?,
            )?;
        }

        let mut per_event = MemoryFootprint::per_event(size_of::<f64>() as u64);
        for entry in self.cache_plan.entries() {
            let bytes = match entry.value_kind() {
                ValueKind::Real => MemoryFootprint::per_event(size_of::<f64>() as u64),
                ValueKind::Complex => MemoryFootprint::per_event(size_of::<Complex64>() as u64),
                ValueKind::Vector { len } => {
                    MemoryFootprint::per_event(size_of::<Complex64>() as u64)
                        .checked_scale_usize(len)?
                }
                ValueKind::Matrix { rows, cols } => {
                    MemoryFootprint::per_event(size_of::<Complex64>() as u64)
                        .checked_scale_usize(rows)?
                        .checked_scale_usize(cols)?
                }
            };
            per_event = per_event.checked_add(bytes)?;
        }
        for (_, dimension) in &self.factor_matrices {
            let bytes = MemoryFootprint::per_event(size_of::<DynamicLu>() as u64)
                .checked_add(
                    MemoryFootprint::per_event(size_of::<Complex64>() as u64)
                        .checked_scale_usize(*dimension)?
                        .checked_scale_usize(*dimension)?,
                )?
                .checked_add(
                    MemoryFootprint::per_event(size_of::<usize>() as u64)
                        .checked_scale_usize(*dimension)?,
                )?;
            per_event = per_event.checked_add(bytes)?;
        }
        for (_, _, dimension) in &self.solve_row_keys {
            per_event = per_event.checked_add(
                MemoryFootprint::per_event(size_of::<Complex64>() as u64)
                    .checked_scale_usize(*dimension)?,
            )?;
        }
        fixed.checked_add(per_event)
    }

    fn cache_dataset_with_plan(
        &self,
        dataset: &Dataset,
        read_plan: laddu_data::io::ReadPlan,
    ) -> RuntimeResult<CpuCachedDataset> {
        let mut batches = Vec::new();
        let mut sum_weights = 0.0;
        for batch in dataset
            .stream_with_plan(read_plan)
            .map_err(|err| RuntimeError::Data(err.to_string()))?
        {
            let batch = batch.map_err(|err| RuntimeError::Data(err.to_string()))?;
            let cached = CpuCachedBatch::from_cache(self.cache_event_batch(&batch)?);
            sum_weights += cached.sum_weights();
            batches.push(cached);
        }
        Ok(CpuCachedDataset::from_parts(batches, sum_weights))
    }

    /// Prepares a dataset according to its cache-storage policy.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when dataset reading or cache construction
    /// fails, or another distributed worker reports failure.
    ///
    /// # Panics
    ///
    /// Panics if successful preparation planning fails to record its staging
    /// memory decision, which would violate the planner invariant.
    pub fn prepare_dataset(
        &self,
        execution: &Execution,
        dataset: &Dataset,
    ) -> RuntimeResult<CpuPreparedDataset> {
        let preparation = DatasetPreparation::new(execution, dataset);
        let mut planning = preparation.runtime_plan()?;
        let initial_read_plan = planning.read_plan();
        let schema = dataset
            .schema()
            .map_err(|error| RuntimeError::Data(error.to_string()))?;
        let source_footprint = BatchLayout::from_schema(&schema)
            .schema_footprint(DataPrecision::F64)
            .map_err(|error| RuntimeError::Data(format!("source working-set overflow: {error}")))?;
        let cache_footprint = self
            .cache_memory_footprint()
            .map_err(|error| RuntimeError::Data(format!("cache working-set overflow: {error}")))?;
        let source_bytes_per_event =
            usize::try_from(source_footprint.bytes_per_event).unwrap_or(usize::MAX);
        let cache_zero = usize::try_from(cache_footprint.fixed_bytes).unwrap_or(usize::MAX);
        let cache_bytes_per_event =
            usize::try_from(cache_footprint.bytes_per_event).unwrap_or(usize::MAX);
        let local_event_limit = planning.event_limit();
        let host_remaining = execution.host_memory().remaining();
        let resident_plan = resident_cache_plan(
            cache_zero,
            cache_bytes_per_event,
            source_bytes_per_event.saturating_mul(2),
            local_event_limit,
            usize::try_from(host_remaining).unwrap_or(usize::MAX),
        );
        let source_staging = source_footprint
            .checked_scale(2)
            .and_then(|footprint| {
                MemoryFootprint::fixed(footprint.bytes_per_event)
                    .checked_add(MemoryFootprint::per_event(cache_footprint.bytes_per_event))
            })
            .map_err(|error| RuntimeError::Data(format!("source working-set overflow: {error}")))?;
        let minimum = MemoryFootprint::fixed(cache_footprint.fixed_bytes)
            .checked_add(source_staging)
            .map_err(|error| RuntimeError::Data(format!("cache working-set overflow: {error}")))?
            .peak_bytes(local_event_limit);
        let resident_bytes = resident_plan
            .map(|(bytes, _)| u64::try_from(bytes).unwrap_or(u64::MAX))
            .unwrap_or(minimum);
        planning.select_storage(
            dataset.memory_policy(),
            "host",
            resident_plan.is_some(),
            resident_bytes,
            host_remaining,
        )?;
        let requested_storage = planning.storage();
        planning.reserve_storage(Some(execution.host_memory()), || {
            RuntimeError::Data("CPU execution has no host memory pool".into())
        })?;
        let persistent_lease = planning.take_memory_lease();
        let available_for_batch = execution.host_memory().remaining();
        // Sources may hold the current decoded batch plus one bounded
        // prefetched batch. A resident cache is already covered by its
        // persistent lease; streaming additionally needs one transient cache.
        let transient_footprint = if requested_storage == CacheStorage::Streaming {
            cache_footprint
                .checked_add(source_footprint.checked_scale(2).map_err(|error| {
                    RuntimeError::Data(format!("source working-set overflow: {error}"))
                })?)
                .map_err(|error| {
                    RuntimeError::Data(format!("cache working-set overflow: {error}"))
                })?
        } else {
            source_footprint.checked_scale(2).map_err(|error| {
                RuntimeError::Data(format!("source working-set overflow: {error}"))
            })?
        };
        let decision = planning.fit_staging(
            "CPU prepared dataset",
            transient_footprint,
            available_for_batch,
            if requested_storage == CacheStorage::Resident {
                "resident"
            } else {
                "streaming"
            },
        )?;
        planning.clamp_read_plan(initial_read_plan.chunk_size, decision.chunk_events);
        let decision = planning
            .take_decisions()
            .into_iter()
            .next()
            .expect("CPU preparation records one staging decision");
        execution.record_memory_decision(decision.clone());
        let read_plan = planning.read_plan();
        match requested_storage {
            CacheStorage::Resident => {
                let dataset =
                    preparation.coordinate(self.cache_dataset_with_plan(dataset, read_plan))?;
                let stats = preparation.finish_stats(
                    LocalDatasetStats::new(
                        dataset.len(),
                        dataset.batches().len(),
                        dataset.sum_weights(),
                    ),
                    dataset.resident_bytes(),
                    requested_storage,
                );
                let memory_lease = persistent_lease.ok_or_else(|| {
                    RuntimeError::Data(
                        "resident dataset preparation did not reserve host memory".into(),
                    )
                })?;
                Ok(CpuPreparedDataset::Resident {
                    dataset: Arc::new(dataset),
                    stats,
                    memory_lease,
                })
            }
            CacheStorage::Streaming => {
                let local = preparation.scan()?;
                Ok(CpuPreparedDataset::Streaming {
                    dataset: dataset.clone(),
                    stats: preparation.finish_stats(local, 0, requested_storage),
                    read_plan,
                    transient_bytes: decision.estimated_peak_bytes,
                })
            }
        }
    }
}

pub(super) fn resident_cache_plan(
    fixed_per_batch: usize,
    cache_bytes_per_event: usize,
    source_bytes_per_event: usize,
    events: usize,
    available: usize,
) -> Option<(usize, usize)> {
    if events == 0 {
        return Some((fixed_per_batch, 1));
    }
    let event_cache = cache_bytes_per_event.checked_mul(events)?;
    let minimum = event_cache
        .checked_add(fixed_per_batch)?
        .checked_add(source_bytes_per_event)?;
    if minimum > available {
        return None;
    }
    let mut chunk = events;
    for _ in 0..16 {
        let batches = events.saturating_add(chunk - 1) / chunk;
        let resident = event_cache.checked_add(fixed_per_batch.checked_mul(batches)?)?;
        let next = available
            .saturating_sub(resident)
            .checked_div(source_bytes_per_event.max(1))?
            .min(events);
        if next == 0 {
            return None;
        }
        if next == chunk {
            return Some((resident, chunk));
        }
        chunk = next;
    }
    let batches = events.saturating_add(chunk - 1) / chunk;
    let resident = event_cache.checked_add(fixed_per_batch.checked_mul(batches)?)?;
    (resident.checked_add(source_bytes_per_event.checked_mul(chunk)?)? <= available)
        .then_some((resident, chunk))
}
