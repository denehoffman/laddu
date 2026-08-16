use std::{
    mem::size_of,
    sync::{Arc, Mutex},
};

use crate::{
    LadduDataError, LadduDataResult,
    data::event::{Event, EventBatch},
    io::{EventBatchIter, ReadPlan},
};
use laddu_memory::{MemoryFitRequest, MemoryFootprint, MemoryState};

use super::ops::{eval_batch, materialize_batch};
use super::{Dataset, DatasetStats, DatasetStatsCache};

/// Resolved execution choices for one dataset traversal.
pub(super) struct DatasetExecutionPlan {
    read_plan: ReadPlan,
    target_batch_size: usize,
    commit_stats: bool,
}

impl DatasetExecutionPlan {
    pub(super) fn resolve(dataset: &Dataset, mut read_plan: ReadPlan) -> LadduDataResult<Self> {
        if read_plan.chunk_size.is_none() {
            let schema = dataset.schema()?;
            let bytes_per_event = 4_usize
                .saturating_mul(schema.n_p4s())
                .saturating_add(schema.n_scalars())
                .saturating_add(usize::from(schema.has_weight()))
                .saturating_mul(size_of::<f64>());
            let copies = if dataset.ops.is_empty() { 1 } else { 2 };
            let peak_per_event = bytes_per_event.saturating_mul(copies);
            let state = MemoryState::current();
            state.refresh();
            let available = dataset
                .memory_budget
                .resolve(&state.host())
                .map_err(|error| LadduDataError::Source(error.to_string()))?;
            let event_limit = dataset
                .source
                .num_events()?
                .and_then(|events| usize::try_from(events).ok())
                .unwrap_or(usize::MAX);
            let decision = MemoryFitRequest {
                label: "dataset read".into(),
                footprint: MemoryFootprint::from_usize(0, peak_per_event),
                available_bytes: available,
                event_limit,
                strategy: "memory-derived streaming".into(),
            }
            .evaluate()
            .map_err(|error| LadduDataError::Source(error.to_string()))?;
            read_plan.chunk_size = Some(decision.chunk_events.max(1));
            *dataset
                .last_memory_decision
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(decision);
        }

        let target_batch_size = read_plan.chunk_size.unwrap_or(usize::MAX).max(1);
        Ok(Self {
            commit_stats: !read_plan.is_distributed(),
            read_plan,
            target_batch_size,
        })
    }
}

pub(super) fn visit_events<F>(
    dataset: &Dataset,
    plan: DatasetExecutionPlan,
    mut f: F,
) -> LadduDataResult<()>
where
    F: FnMut(Event<'_>) -> LadduDataResult<()>,
{
    dataset
        .source_traversals
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let input = dataset.source.batches(plan.read_plan)?;
    let ops = Arc::clone(&dataset.ops);
    let mut offset = 0_u64;
    let mut stats = StatsAccumulator::default();

    for batch in input {
        let batch = batch?;
        let base = offset;
        offset += batch.len() as u64;
        eval_batch(&batch, &ops, base, |event| {
            f(event)?;
            stats.observe_weight(event.weight());
            Ok(())
        })?;
    }

    if plan.commit_stats {
        stats.commit(Arc::clone(&dataset.stats));
    }
    Ok(())
}

/// Compensated statistics for one fully consumed dataset traversal.
#[derive(Default)]
pub(super) struct StatsAccumulator {
    events: u64,
    sum_weights: f64,
    correction: f64,
}

impl StatsAccumulator {
    fn observe_weight(&mut self, weight: f64) {
        self.events = self.events.saturating_add(1);
        let corrected = weight - self.correction;
        let next = self.sum_weights + corrected;
        self.correction = (next - self.sum_weights) - corrected;
        self.sum_weights = next;
    }

    fn observe_batch(&mut self, batch: &EventBatch) {
        for row in 0..batch.len() {
            self.observe_weight(batch.weights_at(row));
        }
    }

    fn finish(&self) -> DatasetStats {
        DatasetStats {
            events: self.events,
            sum_weights: self.sum_weights,
        }
    }

    fn commit(&self, cache: Arc<Mutex<DatasetStatsCache>>) {
        let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
        cache.events = Some(self.events);
        cache.sum_weights = Some(self.sum_weights);
    }
}

/// Transforms source batches, coalesces them to the resolved target, and owns
/// traversal statistics/cache commit policy.
pub(super) struct DatasetExecutor {
    batches: CoalescedBatches,
    stats: StatsAccumulator,
    stats_cache: Option<Arc<Mutex<DatasetStatsCache>>>,
}

impl DatasetExecutor {
    pub(super) fn new(dataset: &Dataset, plan: DatasetExecutionPlan) -> LadduDataResult<Self> {
        dataset
            .source_traversals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let input = dataset.source.batches(plan.read_plan)?;
        let ops = Arc::clone(&dataset.ops);
        let transformed = Box::new(input.scan(0_u64, move |offset, batch| {
            let batch = match batch {
                Ok(batch) => batch,
                Err(error) => return Some(Err(error)),
            };

            let base = *offset;
            *offset += batch.len() as u64;
            Some(materialize_batch(&batch, &ops, base))
        }));

        Ok(Self {
            batches: CoalescedBatches::new(transformed, plan.target_batch_size),
            stats: StatsAccumulator::default(),
            stats_cache: plan.commit_stats.then(|| Arc::clone(&dataset.stats)),
        })
    }

    pub(super) fn stats(&self) -> DatasetStats {
        self.stats.finish()
    }
}

impl Iterator for DatasetExecutor {
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.batches.next() {
            Some(Ok(batch)) => {
                self.stats.observe_batch(&batch);
                Some(Ok(batch))
            }
            Some(Err(error)) => {
                self.stats_cache = None;
                Some(Err(error))
            }
            None => {
                if let Some(cache) = self.stats_cache.take() {
                    self.stats.commit(cache);
                }
                None
            }
        }
    }
}

struct CoalescedBatches {
    input: EventBatchIter,
    target: usize,
    pending: Vec<EventBatch>,
    pending_len: usize,
    deferred: Option<LadduDataResult<EventBatch>>,
    finished: bool,
}

impl CoalescedBatches {
    fn new(input: EventBatchIter, target: usize) -> Self {
        Self {
            input,
            target,
            pending: Vec::new(),
            pending_len: 0,
            deferred: None,
            finished: false,
        }
    }

    fn emit_pending(&mut self) -> Option<LadduDataResult<EventBatch>> {
        if self.pending.is_empty() {
            return None;
        }
        self.pending_len = 0;
        let batches = std::mem::take(&mut self.pending);
        Some(if batches.len() == 1 {
            Ok(batches.into_iter().next().expect("one pending batch"))
        } else {
            EventBatch::concat(&batches)
        })
    }
}

impl Iterator for CoalescedBatches {
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.pending_len == self.target {
                return self.emit_pending();
            }

            let item = if let Some(item) = self.deferred.take() {
                Some(item)
            } else if self.finished {
                None
            } else {
                self.input.next()
            };

            let Some(item) = item else {
                self.finished = true;
                if let Some(batch) = self.emit_pending() {
                    return Some(batch);
                }
                return None;
            };
            let batch = match item {
                Ok(batch) => batch,
                Err(error) => {
                    if self.pending.is_empty() {
                        return Some(Err(error));
                    }
                    self.deferred = Some(Err(error));
                    return self.emit_pending();
                }
            };
            if batch.is_empty() {
                continue;
            }

            let available = self.target - self.pending_len;
            if batch.len() <= available {
                self.pending_len += batch.len();
                self.pending.push(batch);
                continue;
            }

            self.pending.push(batch.slice(0, available));
            self.pending_len += available;
            self.deferred = Some(Ok(batch.slice(available, batch.len())));
        }
    }
}
