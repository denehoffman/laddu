use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::{
    LadduDataError, LadduDataResult,
    data::event::{Event, EventBatch, OwnedEvent},
    io::{EventSink, EventSource, ReadPlan, SourceCapabilities, WritePlan, memory::MemorySource},
    schema::Schema,
};
use laddu_memory::{MemoryBudget, MemoryDecision};
use num::complex::Complex64;

#[cfg(feature = "parallel")]
pub mod accurate;
mod execution;
mod ops;

use execution::{DatasetExecutionPlan, DatasetExecutor, visit_events};
use ops::DatasetOp;
#[cfg(test)]
use ops::{poisson1_from_hash, uniform_hash_01};

static NEXT_DATASET_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_dataset_identity() -> u64 {
    NEXT_DATASET_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

/// Cached statistics for an immutable dataset view.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DatasetStats {
    events: u64,
    sum_weights: f64,
}

impl DatasetStats {
    /// Returns the number of transformed events.
    pub fn events(&self) -> u64 {
        self.events
    }

    /// Returns the accurately accumulated event-weight sum.
    pub fn sum_weights(&self) -> f64 {
        self.sum_weights
    }
}

#[derive(Default)]
struct DatasetStatsCache {
    events: Option<u64>,
    sum_weights: Option<f64>,
}

/// Lazy event dataset combining a source, read plan, and row transformations.
#[derive(Clone)]
pub struct Dataset {
    identity: u64,
    source: Arc<dyn EventSource>,
    plan: ReadPlan,
    ops: Arc<[DatasetOp]>,
    cache_storage: CacheStorage,
    memory_policy: MemoryPolicy,
    memory_budget: MemoryBudget,
    last_memory_decision: Arc<Mutex<Option<MemoryDecision>>>,
    stats: Arc<Mutex<DatasetStatsCache>>,
    source_traversals: Arc<AtomicU64>,
}

/// Memory policy for compiled event-dependent model caches.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CacheStorage {
    /// Materialize all event-dependent cache values once and retain them for repeated evaluations.
    #[default]
    Resident,
    /// Retain only dataset statistics and rebuild each batch cache during every evaluation.
    Streaming,
}

/// Strategy used to trade retained memory for execution speed.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum MemoryPolicy {
    /// Select the fastest supported resident or streaming strategy that fits.
    #[default]
    Fastest,
    /// Require the complete compiled event cache to remain resident.
    Resident,
    /// Retain no compiled event cache between traversals.
    Streaming,
}

impl Dataset {
    /// Creates a dataset from an event source.
    pub fn new<S>(source: S) -> Self
    where
        S: EventSource + 'static,
    {
        Self {
            identity: next_dataset_identity(),
            source: Arc::new(source),
            plan: ReadPlan::default(),
            ops: Arc::from([]),
            cache_storage: CacheStorage::Resident,
            memory_policy: MemoryPolicy::Fastest,
            memory_budget: MemoryBudget::Auto,
            last_memory_decision: Default::default(),
            stats: Default::default(),
            source_traversals: Default::default(),
        }
    }

    /// Creates a dataset from a shared dynamically dispatched source.
    pub fn from_arc(source: Arc<dyn EventSource>) -> Self {
        Self {
            identity: next_dataset_identity(),
            source,
            plan: ReadPlan::default(),
            ops: Arc::from([]),
            cache_storage: CacheStorage::Resident,
            memory_policy: MemoryPolicy::Fastest,
            memory_budget: MemoryBudget::Auto,
            last_memory_decision: Default::default(),
            stats: Default::default(),
            source_traversals: Default::default(),
        }
    }

    /// Build a derived dataset while preserving this dataset's read and cache policy.
    #[doc(hidden)]
    pub fn with_derived_source<S>(&self, source: S) -> Self
    where
        S: EventSource + 'static,
    {
        Self {
            identity: next_dataset_identity(),
            source: Arc::new(source),
            plan: self.plan,
            ops: Arc::from([]),
            cache_storage: self.cache_storage,
            memory_policy: self.memory_policy,
            memory_budget: self.memory_budget,
            last_memory_decision: Default::default(),
            stats: Default::default(),
            source_traversals: Default::default(),
        }
    }

    /// Creates an in-memory dataset from one batch.
    pub fn from_batch(batch: EventBatch) -> Self {
        Self::new(MemorySource::new(batch))
    }

    /// Creates an in-memory dataset from schema-compatible batches.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when batch schemas are incompatible.
    pub fn from_batches(batches: Vec<EventBatch>) -> LadduDataResult<Self> {
        Ok(Self::new(MemorySource::from_batches(batches)?))
    }

    /// Collects owned events into an in-memory dataset.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when an event does not match `schema`.
    pub fn from_events<I>(schema: Arc<Schema>, events: I) -> LadduDataResult<Self>
    where
        I: IntoIterator<Item = OwnedEvent>,
    {
        Ok(Self::new(MemorySource::from_events(schema, events)?))
    }

    /// Returns the source schema.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the underlying source cannot determine
    /// or load its schema.
    pub fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        self.source.schema()
    }

    /// Returns source planning capabilities.
    pub fn capabilities(&self) -> SourceCapabilities {
        self.source.capabilities()
    }

    /// Returns the source event count when cheaply available.
    ///
    /// # Errors
    ///
    /// Returns an error when source metadata cannot be read.
    pub fn num_events(&self) -> LadduDataResult<Option<u64>> {
        {
            let stats = self.stats.lock().unwrap_or_else(|error| error.into_inner());
            if let Some(events) = stats.events {
                return Ok(Some(events));
            }
        }

        if self
            .ops
            .iter()
            .any(|op| !matches!(op, DatasetOp::Bootstrap { .. }))
        {
            return Ok(None);
        }

        let events = self.source.num_events()?;
        if let Some(events) = events {
            self.stats
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .events = Some(events);
        }
        Ok(events)
    }

    /// Returns cached event-count and weight-sum statistics, computing them once if needed.
    ///
    /// Clones of a dataset view share this cache. Failed traversals are not cached and may be
    /// retried.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the dataset fails.
    pub fn stats(&self) -> LadduDataResult<DatasetStats> {
        {
            let cache = self.stats.lock().unwrap_or_else(|error| error.into_inner());
            if let (Some(events), Some(sum_weights)) = (cache.events, cache.sum_weights) {
                return Ok(DatasetStats {
                    events,
                    sum_weights,
                });
            }
        }

        if self.ops.is_empty()
            && let (Some(events), Some(sum_weights)) =
                (self.source.num_events()?, self.source.weighted_total()?)
        {
            let stats = DatasetStats {
                events,
                sum_weights,
            };
            let mut cache = self.stats.lock().unwrap_or_else(|error| error.into_inner());
            cache.events = Some(events);
            cache.sum_weights = Some(sum_weights);
            return Ok(stats);
        }

        let mut executor = self.executor_with_plan(self.plan)?;
        for batch in &mut executor {
            batch?;
        }

        Ok(executor.stats())
    }

    /// Returns the current read plan.
    pub fn read_plan(&self) -> ReadPlan {
        self.plan
    }

    /// Returns the compiled-cache memory policy.
    pub fn cache_storage(&self) -> CacheStorage {
        self.cache_storage
    }

    /// Returns the memory-first cache selection policy.
    pub fn memory_policy(&self) -> MemoryPolicy {
        self.memory_policy
    }

    /// Returns the dataset's host-memory budget.
    pub fn memory_budget(&self) -> MemoryBudget {
        self.memory_budget
    }

    /// Returns the most recent memory-derived read decision.
    pub fn last_memory_decision(&self) -> Option<MemoryDecision> {
        self.last_memory_decision
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Returns the number of transformed source iterators opened by this dataset view.
    pub fn source_traversals(&self) -> u64 {
        self.source_traversals.load(Ordering::Relaxed)
    }

    /// Returns the immutable identity of this dataset view.
    ///
    /// Clones retain identity, while row, weight, and source transformations
    /// create a new identity. This is intended for execution-scoped caches.
    #[doc(hidden)]
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Returns this dataset with a host-memory budget.
    pub fn with_memory_budget(mut self, budget: MemoryBudget) -> Self {
        self.memory_budget = budget;
        self
    }

    /// Select the fastest strategy allowed by the active memory budget.
    pub fn fastest(mut self) -> Self {
        self.memory_policy = MemoryPolicy::Fastest;
        self.cache_storage = CacheStorage::Resident;
        self
    }

    /// Retain compiled event-dependent values for every local event.
    ///
    /// This strict policy is intended for repeatedly evaluating a likelihood
    /// and returns an error if the resident cache cannot fit. [`Dataset::fastest`]
    /// is the default.
    pub fn resident(mut self) -> Self {
        self.memory_policy = MemoryPolicy::Resident;
        self.cache_storage = CacheStorage::Resident;
        self
    }

    /// Re-read the source and rebuild one batch cache during each parameter evaluation.
    ///
    /// Only fixed dataset statistics are retained. This minimizes memory use but requires a
    /// repeatable source and is expected to be slower than [`Dataset::resident`].
    pub fn streaming(mut self) -> Self {
        self.memory_policy = MemoryPolicy::Streaming;
        self.cache_storage = CacheStorage::Streaming;
        self
    }

    /// Returns this dataset with a low-level nonzero maximum event count.
    ///
    /// Prefer [`Dataset::with_memory_budget`] for portable application code.
    /// This explicit read-plan override remains available for source debugging
    /// and reproducibility and is always capped by execution memory planning.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError::InvalidArgument`] when `chunk_size` is zero.
    pub fn chunked(mut self, chunk_size: usize) -> LadduDataResult<Self> {
        if chunk_size == 0 {
            return Err(LadduDataError::InvalidArgument(
                "chunk_size must be nonzero",
            ));
        }
        self.plan.chunk_size = Some(chunk_size);
        Ok(self)
    }

    /// Returns this dataset with source-native batch sizes.
    pub fn unchunked(mut self) -> Self {
        self.plan.chunk_size = None;
        self
    }

    /// Lazily retains events satisfying `f`.
    pub fn filter<F>(self, f: F) -> Self
    where
        F: Fn(Event<'_>) -> bool + Send + Sync + 'static,
    {
        self.push_op(DatasetOp::Filter(Arc::new(f)))
    }

    /// Lazily retains a deterministic fraction of events.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError::InvalidArgument`] when `fraction` is outside
    /// `[0, 1]` or is NaN.
    pub fn subsample(self, fraction: f64, seed: u64) -> LadduDataResult<Self> {
        if !(0.0..=1.0).contains(&fraction) {
            return Err(LadduDataError::InvalidArgument(
                "fraction must be in [0, 1]",
            ));
        }

        Ok(self.push_op(DatasetOp::Subsample { fraction, seed }))
    }

    /// Applies deterministic Poisson bootstrap multiplicities to event weights.
    pub fn bootstrap(self, seed: u64) -> Self {
        self.push_op(DatasetOp::Bootstrap { seed })
    }

    /// Visits each transformed event.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the source
    /// fails.
    pub fn for_each_event<F>(&self, mut f: F) -> LadduDataResult<()>
    where
        F: FnMut(Event<'_>),
    {
        self.try_for_each_event(|ev| {
            f(ev);
            Ok(())
        })
    }

    /// Visits each transformed event and stops at the first error.
    ///
    /// # Errors
    ///
    /// Returns the first [`LadduDataError`] produced by the source,
    /// transformations, or callback.
    pub fn try_for_each_event<F>(&self, mut f: F) -> LadduDataResult<()>
    where
        F: FnMut(Event<'_>) -> LadduDataResult<()>,
    {
        visit_events(
            self,
            DatasetExecutionPlan::resolve(self, self.plan)?,
            &mut f,
        )
    }

    /// Fallibly maps transformed events into a vector.
    ///
    /// # Errors
    ///
    /// Returns the first [`LadduDataError`] produced while reading,
    /// transforming, or mapping an event.
    pub fn try_map_events<T, F>(&self, mut f: F) -> LadduDataResult<Vec<T>>
    where
        F: FnMut(Event<'_>) -> LadduDataResult<T>,
    {
        let mut out = Vec::new();
        self.try_for_each_event(|ev| {
            out.push(f(ev)?);
            Ok(())
        })?;

        Ok(out)
    }

    /// Maps transformed events into a vector.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the source
    /// fails.
    pub fn map_events<T, F>(&self, mut f: F) -> LadduDataResult<Vec<T>>
    where
        F: FnMut(Event<'_>) -> T,
    {
        let mut out = Vec::new();
        self.try_for_each_event(|ev| {
            out.push(f(ev));
            Ok(())
        })?;

        Ok(out)
    }

    /// Fallibly folds transformed events into an owned accumulator.
    ///
    /// # Errors
    ///
    /// Returns the first [`LadduDataError`] produced while reading,
    /// transforming, or folding an event.
    pub fn try_fold_events<T, F>(&self, init: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(T, Event<'_>) -> LadduDataResult<T>,
    {
        let mut acc = Some(init);

        self.try_for_each_event(|ev| {
            let current = acc.take().ok_or_else(|| {
                LadduDataError::Source("dataset fold accumulator was consumed".into())
            })?;
            acc = Some(f(current, ev)?);
            Ok(())
        })?;

        acc.ok_or_else(|| LadduDataError::Source("dataset fold produced no accumulator".into()))
    }

    /// Folds transformed events into an owned accumulator.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the source
    /// fails.
    pub fn fold_events<T, F>(&self, init: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(T, Event<'_>) -> T,
    {
        self.try_fold_events(init, |acc, ev| Ok(f(acc, ev)))
    }

    /// Fallibly updates a mutable accumulator for every transformed event.
    ///
    /// # Errors
    ///
    /// Returns the first [`LadduDataError`] produced while reading,
    /// transforming, or accumulating an event.
    pub fn try_accumulate_events<T, F>(&self, mut acc: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(&mut T, Event<'_>) -> LadduDataResult<()>,
    {
        self.try_for_each_event(|ev| f(&mut acc, ev))?;
        Ok(acc)
    }

    /// Updates a mutable accumulator for every transformed event.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the source
    /// fails.
    pub fn accumulate_events<T, F>(&self, acc: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(&mut T, Event<'_>),
    {
        self.try_accumulate_events(acc, |acc, ev| {
            f(acc, ev);
            Ok(())
        })
    }

    /// Sums effective event weights.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the source
    /// fails.
    pub fn sum_weights(&self) -> LadduDataResult<f64> {
        Ok(self.stats()?.sum_weights())
    }

    /// Sums `weight * f(event)` over transformed events.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the source
    /// fails.
    pub fn weighted_sum<F>(&self, mut f: F) -> LadduDataResult<f64>
    where
        F: FnMut(Event<'_>) -> f64,
    {
        self.fold_events(0.0, |sum, ev| sum + ev.weight() * f(ev))
    }

    /// Sums complex `weight * f(event)` contributions.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming the source
    /// fails.
    pub fn weighted_complex_sum<F>(&self, mut f: F) -> LadduDataResult<Complex64>
    where
        F: FnMut(Event<'_>) -> Complex64,
    {
        self.fold_events(0.0.into(), |sum, ev| sum + ev.weight() * f(ev))
    }

    /// Opens an iterator of fully transformed event batches.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the underlying source cannot initialize
    /// a batch stream for the current plan.
    pub fn batches(
        &self,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        self.stream_with_plan(self.plan)
    }

    #[doc(hidden)]
    /// Opens the shared transformed batch stream using an explicit read plan.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the underlying source cannot initialize
    /// the requested batch stream.
    pub fn stream_with_plan(
        &self,
        plan: ReadPlan,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        Ok(Box::new(self.executor_with_plan(plan)?))
    }

    #[doc(hidden)]
    /// Compatibility alias for the shared transformed batch stream.
    pub fn batches_with_plan(
        &self,
        plan: ReadPlan,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        self.stream_with_plan(plan)
    }

    fn executor_with_plan(&self, plan: ReadPlan) -> LadduDataResult<DatasetExecutor> {
        DatasetExecutor::new(self, DatasetExecutionPlan::resolve(self, plan)?)
    }

    /// Visits each transformed batch and stops at the first error.
    ///
    /// # Errors
    ///
    /// Returns the first [`LadduDataError`] produced by the source,
    /// transformations, or callback.
    pub fn try_for_each_batch<F>(&self, mut f: F) -> LadduDataResult<()>
    where
        F: FnMut(EventBatch) -> LadduDataResult<()>,
    {
        for batch in self.batches()? {
            f(batch?)?;
        }

        Ok(())
    }

    /// Maps transformed batches into a vector.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading or transforming a batch fails.
    pub fn map_batches<T, F>(&self, mut f: F) -> LadduDataResult<Vec<T>>
    where
        F: FnMut(EventBatch) -> T,
    {
        let mut out = Vec::new();

        self.try_for_each_batch(|batch| {
            out.push(f(batch));
            Ok(())
        })?;

        Ok(out)
    }

    /// Streams the transformed dataset into an event sink.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when reading, transforming, or writing a
    /// batch fails, or the sink cannot begin or finish the stream.
    pub fn write_to<S: EventSink>(&self, sink: &mut S) -> LadduDataResult<()> {
        sink.begin(self.schema()?, WritePlan::from(self.plan))?;

        for batch in self.batches()? {
            sink.write_batch(&batch?)?;
        }

        sink.finish()
    }

    fn push_op(self, op: DatasetOp) -> Self {
        let preserved_events = if matches!(&op, DatasetOp::Bootstrap { .. }) {
            self.num_events().ok().flatten()
        } else {
            None
        };
        let mut ops = self.ops.to_vec();
        ops.push(op);

        Self {
            identity: next_dataset_identity(),
            source: self.source,
            plan: self.plan,
            ops: ops.into(),
            cache_storage: self.cache_storage,
            memory_policy: self.memory_policy,
            memory_budget: self.memory_budget,
            last_memory_decision: Default::default(),
            stats: Arc::new(Mutex::new(DatasetStatsCache {
                events: preserved_events,
                sum_weights: None,
            })),
            source_traversals: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ops::materialize_batch;
    use super::*;
    use crate::io::{EventBatchIter, EventSource, ReadPlan, memory::MemorySink};
    use laddu_physics::vectors::RealVec4;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct CountingSource {
        batch: EventBatch,
        reads: Arc<AtomicUsize>,
    }

    impl EventSource for CountingSource {
        fn schema(&self) -> LadduDataResult<Arc<Schema>> {
            Ok(Arc::clone(self.batch.schema()))
        }

        fn batches(
            &self,
            _plan: ReadPlan,
        ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(std::iter::once(Ok(self.batch.clone()))))
        }
    }

    #[derive(Clone)]
    struct ErrorSource {
        schema: Arc<Schema>,
        items: Arc<[LadduDataResult<EventBatch>]>,
    }

    impl EventSource for ErrorSource {
        fn schema(&self) -> LadduDataResult<Arc<Schema>> {
            Ok(Arc::clone(&self.schema))
        }

        fn batches(&self, _plan: ReadPlan) -> LadduDataResult<EventBatchIter> {
            let items = Arc::clone(&self.items);
            Ok(Box::new(
                (0..items.len()).map(move |index| items[index].clone()),
            ))
        }
    }

    fn v(x: f64) -> RealVec4 {
        RealVec4 {
            e: x + 0.3,
            px: x,
            py: x + 0.1,
            pz: x + 0.2,
        }
    }

    fn schema_with_weight() -> Arc<Schema> {
        Arc::new(Schema::new(["p"], ["x"], true).unwrap())
    }

    fn schema_without_weight() -> Arc<Schema> {
        Arc::new(Schema::new(["p"], ["x"], false).unwrap())
    }

    fn weighted_batch(start: usize, len: usize) -> EventBatch {
        let schema = schema_with_weight();

        let events = (start..start + len)
            .map(|i| OwnedEvent::weighted(vec![v(i as f64)], vec![i as f64], 10.0 + i as f64));

        EventBatch::from_events(schema, events).unwrap()
    }

    fn unweighted_batch(start: usize, len: usize) -> EventBatch {
        let schema = schema_without_weight();

        let events =
            (start..start + len).map(|i| OwnedEvent::new(vec![v(i as f64)], vec![i as f64]));

        EventBatch::from_events(schema, events).unwrap()
    }

    fn error_source(error_after: Option<EventBatch>) -> ErrorSource {
        let schema = schema_with_weight();
        let mut items = Vec::new();
        if let Some(batch) = error_after {
            items.push(Ok(batch));
        }
        items.push(Err(LadduDataError::Unsupported("source")));
        ErrorSource {
            schema,
            items: items.into(),
        }
    }

    fn scalar_values(batch: &EventBatch) -> Vec<f64> {
        batch.scalar_column(0).to_vec()
    }

    #[test]
    fn dataset_statistics_are_shared_per_view_and_invalidated_by_selection() {
        let reads = Arc::new(AtomicUsize::new(0));
        let dataset = Dataset::new(CountingSource {
            batch: weighted_batch(0, 5),
            reads: Arc::clone(&reads),
        });
        let clone = dataset.clone();

        assert_eq!(dataset.num_events().unwrap(), None);
        assert_eq!(dataset.stats().unwrap().events(), 5);
        assert_eq!(clone.sum_weights().unwrap(), 60.0);
        assert_eq!(clone.num_events().unwrap(), Some(5));
        assert_eq!(reads.load(Ordering::Relaxed), 1);

        let bootstrapped = dataset.clone().bootstrap(7);
        assert_eq!(bootstrapped.num_events().unwrap(), Some(5));
        assert_eq!(reads.load(Ordering::Relaxed), 1);

        let filtered = dataset.filter(|event| event.scalar(0) >= 2.0);
        assert_eq!(filtered.num_events().unwrap(), None);
        assert_eq!(
            filtered.map_events(|event| event.scalar(0)).unwrap(),
            [2.0, 3.0, 4.0]
        );
        assert_eq!(reads.load(Ordering::Relaxed), 2);
        assert_eq!(filtered.stats().unwrap().events(), 3);
        assert_eq!(reads.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn transformed_fragments_are_coalesced_to_the_read_chunk_size() {
        let fragments = (0..10)
            .map(|index| weighted_batch(index, 1))
            .collect::<Vec<_>>();
        let dataset = Dataset::from_batches(fragments)
            .unwrap()
            .filter(|_| true)
            .chunked(4)
            .unwrap();

        let batches = dataset
            .batches()
            .unwrap()
            .collect::<LadduDataResult<Vec<_>>>()
            .unwrap();
        assert_eq!(
            batches.iter().map(EventBatch::len).collect::<Vec<_>>(),
            [4, 4, 2]
        );
        assert_eq!(
            batches
                .iter()
                .flat_map(|batch| batch.scalar_column(0).iter().copied())
                .collect::<Vec<_>>(),
            (0..10).map(|value| value as f64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn shared_stream_preserves_pending_batches_before_source_errors() {
        let dataset = Dataset::new(error_source(Some(weighted_batch(0, 2))))
            .chunked(4)
            .unwrap();
        let mut batches = dataset.batches().unwrap();

        assert_eq!(batches.next().unwrap().unwrap().len(), 2);
        assert!(matches!(
            batches.next().unwrap(),
            Err(LadduDataError::Unsupported("source"))
        ));
        assert!(batches.next().is_none());

        assert!(matches!(
            dataset.stats(),
            Err(LadduDataError::Unsupported("source"))
        ));
        assert_eq!(dataset.source_traversals(), 2);
    }

    #[test]
    fn event_visitors_share_the_execution_plan_without_changing_source_rows() {
        let dataset = Dataset::new(error_source(Some(weighted_batch(0, 2))))
            .chunked(4)
            .unwrap()
            .filter(|event| event.scalar(0) >= 0.0);
        let mut rows = Vec::new();

        let error = dataset
            .try_for_each_event(|event| {
                rows.push(event.row());
                Ok(())
            })
            .unwrap_err();

        assert!(matches!(error, LadduDataError::Unsupported("source")));
        assert_eq!(rows, [0, 1]);
    }

    #[test]
    fn dataset_map_fold_accumulate_complex_sum_and_error_paths_use_transformed_events() {
        let dataset =
            Dataset::from_batch(weighted_batch(0, 5)).filter(|ev| ev.scalar(0) % 2.0 == 0.0);

        let rows = dataset
            .map_events(|ev| (ev.row(), ev.scalar(0), ev.weight()))
            .unwrap();

        assert_eq!(rows, vec![(0, 0.0, 10.0), (2, 2.0, 12.0), (4, 4.0, 14.0)]);

        let folded = dataset
            .fold_events(String::new(), |mut out, ev| {
                out.push_str(&format!("{};", ev.scalar(0)));
                out
            })
            .unwrap();

        assert_eq!(folded, "0;2;4;");

        let accumulated = dataset
            .accumulate_events(Vec::<f64>::new(), |values, ev| values.push(ev.weight()))
            .unwrap();

        assert_eq!(accumulated, vec![10.0, 12.0, 14.0]);

        let weighted_sum = dataset.weighted_sum(|ev| ev.scalar(0)).unwrap();
        assert_eq!(weighted_sum, 0.0 * 10.0 + 2.0 * 12.0 + 4.0 * 14.0);

        let complex_sum = dataset
            .weighted_complex_sum(|ev| Complex64::new(ev.scalar(0), 1.0))
            .unwrap();

        assert_eq!(complex_sum.re, weighted_sum);
        assert_eq!(complex_sum.im, 10.0 + 12.0 + 14.0);

        let err = dataset
            .try_map_events(|ev| {
                if ev.scalar(0) == 2.0 {
                    Err(LadduDataError::Unsupported("stop"))
                } else {
                    Ok(ev.scalar(0))
                }
            })
            .unwrap_err();

        assert!(matches!(err, LadduDataError::Unsupported("stop")));
    }

    #[test]
    fn deterministic_subsample_and_bootstrap_use_global_event_ids_across_batches() {
        let seed = 0x0BAD_5EED;
        let bootstrap_seed = 0xB007_57A9;

        let dataset = Dataset::from_batches(vec![weighted_batch(0, 3), weighted_batch(3, 3)])
            .unwrap()
            .subsample(0.5, seed)
            .unwrap()
            .bootstrap(bootstrap_seed);

        let observed = dataset
            .map_events(|ev| (ev.scalar(0) as u64, ev.weight()))
            .unwrap();

        let expected: Vec<(u64, f64)> = (0_u64..6)
            .filter(|&event_id| uniform_hash_01(seed, event_id) < 0.5)
            .map(|event_id| {
                let original_weight = 10.0 + event_id as f64;
                let bootstrap_weight =
                    poisson1_from_hash(bootstrap_seed, event_id) as f64 * original_weight;
                (event_id, bootstrap_weight)
            })
            .collect();

        assert_eq!(observed, expected);
    }

    #[test]
    fn materialized_batches_store_weights_only_when_needed() {
        let unweighted = unweighted_batch(0, 4);

        let filtered = Dataset::from_batch(unweighted.clone())
            .filter(|ev| ev.scalar(0) >= 1.0)
            .subsample(1.0, 123)
            .unwrap();

        let filtered_batch = filtered.batches().unwrap().next().unwrap().unwrap();

        assert_eq!(scalar_values(&filtered_batch), vec![1.0, 2.0, 3.0]);
        assert!(filtered_batch.weights_column().is_none());

        let bootstrapped = Dataset::from_batch(unweighted).bootstrap(999);
        let bootstrapped_batch = bootstrapped.batches().unwrap().next().unwrap().unwrap();

        assert!(bootstrapped_batch.weights_column().is_some());

        let source = weighted_batch(0, 2);
        let empty_weighted =
            materialize_batch(&source, &[DatasetOp::Filter(Arc::new(|_| false))], 0).unwrap();
        assert!(empty_weighted.is_empty());
        assert_eq!(empty_weighted.weights_column(), Some([].as_slice()));
    }

    #[test]
    fn write_to_memory_sink_captures_transformed_dataset() {
        let dataset = Dataset::from_batch(weighted_batch(0, 5)).filter(|ev| ev.scalar(0) >= 2.0);

        let mut sink = MemorySink::new();
        dataset.write_to(&mut sink).unwrap();

        let captured = sink.into_batch().unwrap();

        assert_eq!(scalar_values(&captured), vec![2.0, 3.0, 4.0]);
        assert_eq!(captured.weights_column().unwrap(), &[12.0, 13.0, 14.0]);
    }

    #[test]
    fn immutable_dataset_identity_tracks_semantic_views() {
        let dataset = Dataset::from_batch(weighted_batch(0, 3));
        assert_eq!(dataset.identity(), dataset.clone().identity());
        assert_eq!(dataset.identity(), dataset.clone().streaming().identity());
        assert_ne!(
            dataset.identity(),
            dataset.clone().subsample(1.0, 7).unwrap().identity()
        );
        assert_ne!(dataset.identity(), dataset.clone().bootstrap(7).identity());
        assert_ne!(
            dataset.identity(),
            dataset.clone().filter(|_| true).identity()
        );
    }
}
