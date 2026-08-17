use std::sync::{Arc, Mutex};

use crate::{
    BatchLayout, LadduDataResult,
    data::event::{Event, EventBatch},
    io::{EventBatchIter, ReadPlan, source_error},
    schema::Precision,
};
use laddu_memory::{MemoryFitRequest, MemoryState};

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
            let copies = if dataset.ops.is_empty() { 1 } else { 2 };
            let footprint = BatchLayout::from_schema(&schema)
                .schema_working_set(Precision::F64, copies)
                .map_err(|error| {
                    source_error(
                        "plan dataset working set",
                        "dataset",
                        format!("dataset working-set overflow: {error}"),
                    )
                })?;
            let state = MemoryState::current();
            state.refresh();
            let available = dataset
                .memory_budget
                .resolve(&state.host())
                .map_err(|error| {
                    source_error("resolve dataset memory budget", "host memory", error)
                })?;
            let event_limit = dataset
                .source
                .num_events()?
                .and_then(|events| usize::try_from(events).ok())
                .unwrap_or(usize::MAX);
            let decision = MemoryFitRequest {
                label: "dataset read".into(),
                footprint,
                available_bytes: available,
                event_limit,
                strategy: "memory-derived streaming".into(),
            }
            .evaluate()
            .map_err(|error| source_error("plan dataset read", "dataset", error))?;
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
    pending: PendingBatches,
    state: CoalescedState,
}

#[derive(Default)]
struct PendingBatches {
    batches: Vec<EventBatch>,
    len: usize,
}

enum CoalescedState {
    Filling,
    Deferred(LadduDataResult<EventBatch>),
    Done,
}

impl CoalescedBatches {
    fn new(input: EventBatchIter, target: usize) -> Self {
        debug_assert!(target > 0, "coalescing target must be nonzero");
        Self {
            input,
            target: target.max(1),
            pending: PendingBatches::default(),
            state: CoalescedState::Filling,
        }
    }

    fn emit_pending(&mut self) -> Option<LadduDataResult<EventBatch>> {
        self.pending.emit()
    }

    fn take_input(&mut self) -> Option<LadduDataResult<EventBatch>> {
        let state = std::mem::replace(&mut self.state, CoalescedState::Filling);
        match state {
            CoalescedState::Filling => self.input.next(),
            CoalescedState::Deferred(item) => Some(item),
            CoalescedState::Done => {
                self.state = CoalescedState::Done;
                None
            }
        }
    }
}

impl PendingBatches {
    fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, batch: EventBatch) {
        debug_assert!(!batch.is_empty(), "empty batches are not pending");
        self.len = self
            .len
            .checked_add(batch.len())
            .expect("pending batch length overflow");
        self.batches.push(batch);
        self.debug_assert_invariant();
    }

    fn emit(&mut self) -> Option<LadduDataResult<EventBatch>> {
        self.debug_assert_invariant();
        if self.batches.is_empty() {
            return None;
        }

        let batches = std::mem::take(&mut self.batches);
        self.len = 0;
        self.debug_assert_invariant();

        Some(if batches.len() == 1 {
            Ok(batches.into_iter().next().expect("one pending batch"))
        } else {
            EventBatch::concat(&batches)
        })
    }

    fn debug_assert_invariant(&self) {
        debug_assert_eq!(
            self.len,
            self.batches.iter().map(EventBatch::len).sum::<usize>(),
            "pending batch length must match its contents",
        );
        debug_assert!(
            self.batches.iter().all(|batch| !batch.is_empty()),
            "pending batches must be nonempty",
        );
    }
}

impl Iterator for CoalescedBatches {
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.pending.debug_assert_invariant();
            if self.pending.len() == self.target {
                return self.emit_pending();
            }

            let item = self.take_input();

            let Some(item) = item else {
                self.state = CoalescedState::Done;
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
                    self.state = CoalescedState::Deferred(Err(error));
                    return self.emit_pending();
                }
            };
            if batch.is_empty() {
                continue;
            }

            let available = self.target - self.pending.len();
            if batch.len() <= available {
                self.pending.push(batch);
                continue;
            }

            self.pending.push(batch.slice(0, available));
            self.state = CoalescedState::Deferred(Ok(batch.slice(available, batch.len())));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LadduDataError, data::EventBatchBuilder, schema::Schema};
    use laddu_physics::vectors::RealVec4;

    fn batch(start: usize, len: usize) -> EventBatch {
        let schema = Arc::new(Schema::new(["p"], ["id"], true).unwrap());
        let mut builder = EventBatchBuilder::with_capacity(schema, len);
        for index in start..start + len {
            let value = index as f64;
            builder
                .push_weighted(
                    [RealVec4 {
                        e: value,
                        px: value,
                        py: value,
                        pz: value,
                    }],
                    [value],
                    1.0,
                )
                .unwrap();
        }
        builder.finish().unwrap()
    }

    #[test]
    fn coalesced_batches_splits_oversized_input_and_skips_empty_batches() {
        let input: EventBatchIter = Box::new(vec![Ok(batch(0, 0)), Ok(batch(0, 5))].into_iter());
        let mut batches = CoalescedBatches::new(input, 2);

        assert_eq!(batches.next().unwrap().unwrap().len(), 2);
        assert_eq!(batches.next().unwrap().unwrap().len(), 2);
        assert_eq!(batches.next().unwrap().unwrap().len(), 1);
        assert!(batches.next().is_none());
        assert!(batches.next().is_none());
    }

    #[test]
    fn coalesced_batches_defers_errors_after_pending_data_and_continues() {
        let input: EventBatchIter = Box::new(
            vec![
                Ok(batch(0, 2)),
                Err(LadduDataError::Source("deferred".into())),
                Ok(batch(2, 2)),
            ]
            .into_iter(),
        );
        let mut batches = CoalescedBatches::new(input, 3);

        assert_eq!(batches.next().unwrap().unwrap().len(), 2);
        assert!(matches!(
            batches.next(),
            Some(Err(LadduDataError::Source(message))) if message == "deferred"
        ));
        assert_eq!(batches.next().unwrap().unwrap().len(), 2);
        assert!(batches.next().is_none());
        assert!(batches.next().is_none());
    }
}
