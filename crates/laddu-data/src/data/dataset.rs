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
use laddu_memory::{MemoryBudget, MemoryDecision, MemoryState};
use laddu_physics::vectors::RealVec4;
use num::complex::Complex64;

#[derive(Clone)]
enum DatasetOp {
    Filter(Arc<dyn Fn(Event<'_>) -> bool + Send + Sync>),
    Subsample { fraction: f64, seed: u64 },
    Bootstrap { seed: u64 },
}

struct CoalescedBatches {
    input: Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>,
    target: usize,
    pending: Vec<EventBatch>,
    pending_len: usize,
    deferred: Option<LadduDataResult<EventBatch>>,
    finished: bool,
    stats: Option<Arc<Mutex<DatasetStatsCache>>>,
    observed_events: u64,
    observed_sum: f64,
    observed_correction: f64,
}

impl CoalescedBatches {
    fn emit_pending(&mut self) -> Option<LadduDataResult<EventBatch>> {
        if self.pending.is_empty() {
            return None;
        }
        self.pending_len = 0;
        let batches = std::mem::take(&mut self.pending);
        let result = if batches.len() == 1 {
            Ok(batches.into_iter().next().expect("one pending batch"))
        } else {
            EventBatch::concat(&batches)
        };
        match &result {
            Ok(batch) => {
                self.observed_events = self.observed_events.saturating_add(batch.len() as u64);
                for row in 0..batch.len() {
                    let corrected = batch.weights_at(row) - self.observed_correction;
                    let next = self.observed_sum + corrected;
                    self.observed_correction = (next - self.observed_sum) - corrected;
                    self.observed_sum = next;
                }
            }
            Err(_) => self.stats = None,
        }
        Some(result)
    }

    fn commit_stats(&mut self) {
        let Some(stats) = self.stats.take() else {
            return;
        };
        let mut stats = stats.lock().unwrap_or_else(|error| error.into_inner());
        stats.events = Some(self.observed_events);
        stats.sum_weights = Some(self.observed_sum);
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
                self.commit_stats();
                return None;
            };
            let batch = match item {
                Ok(batch) => batch,
                Err(error) => {
                    self.stats = None;
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

        let mut events = 0_u64;
        let mut sum = 0.0;
        let mut correction = 0.0;
        for batch in self.batches()? {
            let batch = batch?;
            events = events.saturating_add(batch.len() as u64);
            for row in 0..batch.len() {
                let weight = batch.weights_at(row);
                let corrected = weight - correction;
                let next = sum + corrected;
                correction = (next - sum) - corrected;
                sum = next;
            }
        }

        let stats = DatasetStats {
            events,
            sum_weights: sum,
        };
        let mut cache = self.stats.lock().unwrap_or_else(|error| error.into_inner());
        cache.events = Some(events);
        cache.sum_weights = Some(sum);
        Ok(stats)
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
        let mut offset = 0_u64;
        let mut events = 0_u64;
        let mut sum = 0.0;
        let mut correction = 0.0;

        self.source_traversals.fetch_add(1, Ordering::Relaxed);
        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;
            eval_batch(&batch, &self.ops, base, |event| {
                f(event)?;
                events = events.saturating_add(1);
                let corrected = event.weight() - correction;
                let next = sum + corrected;
                correction = (next - sum) - corrected;
                sum = next;
                Ok(())
            })?;
        }

        let mut stats = self.stats.lock().unwrap_or_else(|error| error.into_inner());
        stats.events = Some(events);
        stats.sum_weights = Some(sum);

        Ok(())
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
        self.batches_with_plan(self.plan)
    }

    #[doc(hidden)]
    /// Opens transformed batches using an explicit read plan.
    ///
    /// # Errors
    ///
    /// Returns [`LadduDataError`] when the underlying source cannot initialize
    /// the requested batch stream.
    pub fn batches_with_plan(
        &self,
        mut plan: ReadPlan,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        if plan.chunk_size.is_none() {
            let schema = self.schema()?;
            let bytes_per_event = 4_u64
                .saturating_mul(schema.n_p4s() as u64)
                .saturating_add(schema.n_scalars() as u64)
                .saturating_add(u64::from(schema.has_weight()))
                .saturating_mul(size_of::<f64>() as u64);
            let copies = if self.ops.is_empty() { 1 } else { 2 };
            let peak_per_event = bytes_per_event.saturating_mul(copies);
            let state = MemoryState::current();
            state.refresh();
            let available = self
                .memory_budget
                .resolve(&state.host())
                .map_err(|error| LadduDataError::Source(error.to_string()))?;
            let event_limit = self
                .source
                .num_events()?
                .and_then(|events| usize::try_from(events).ok())
                .unwrap_or(usize::MAX);
            let decision = MemoryDecision::fit(
                "dataset read",
                0,
                peak_per_event,
                available,
                event_limit,
                "memory-derived streaming",
            )
            .map_err(|error| LadduDataError::Source(error.to_string()))?;
            plan.chunk_size = Some(decision.chunk_events.max(1));
            *self
                .last_memory_decision
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(decision);
        }
        self.source_traversals.fetch_add(1, Ordering::Relaxed);
        let iter = self.source.batches(plan)?;
        let ops = Arc::clone(&self.ops);

        let transformed = Box::new(iter.scan(0_u64, move |offset, batch| {
            let batch = match batch {
                Ok(batch) => batch,
                Err(err) => return Some(Err(err)),
            };

            let base = *offset;
            *offset += batch.len() as u64;

            Some(materialize_batch(&batch, &ops, base))
        }));
        Ok(Box::new(CoalescedBatches {
            input: transformed,
            target: plan.chunk_size.unwrap_or(usize::MAX).max(1),
            pending: Vec::new(),
            pending_len: 0,
            deferred: None,
            finished: false,
            stats: (!plan.is_distributed()).then(|| Arc::clone(&self.stats)),
            observed_events: 0,
            observed_sum: 0.0,
            observed_correction: 0.0,
        }))
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

fn eval_batch<F>(batch: &EventBatch, ops: &[DatasetOp], base: u64, mut f: F) -> LadduDataResult<()>
where
    F: FnMut(Event<'_>) -> LadduDataResult<()>,
{
    'rows: for row in 0..batch.len() {
        let event_id = base + row as u64;
        let mut weight = batch.weights_at(row);

        for op in ops {
            match op {
                DatasetOp::Filter(pred) => {
                    let ev = Event { batch, row, weight };
                    if !pred(ev) {
                        continue 'rows;
                    }
                }
                DatasetOp::Subsample { fraction, seed } => {
                    if uniform_hash_01(*seed, event_id) >= *fraction {
                        continue 'rows;
                    }
                }
                DatasetOp::Bootstrap { seed } => {
                    let k = poisson1_from_hash(*seed, event_id);
                    weight *= k as f64;
                }
            }
        }

        f(Event { batch, row, weight })?;
    }
    Ok(())
}

fn materialize_batch(
    batch: &EventBatch,
    ops: &[DatasetOp],
    base: u64,
) -> LadduDataResult<EventBatch> {
    if ops.is_empty() {
        return Ok(batch.clone());
    }

    let schema = Arc::clone(batch.schema());

    let store_weights = batch.weights_column().is_some()
        || ops
            .iter()
            .any(|op| matches!(op, DatasetOp::Bootstrap { .. }));

    let mut p4s: Vec<Vec<RealVec4>> = (0..schema.n_p4s())
        .map(|_| Vec::with_capacity(batch.len()))
        .collect();
    let mut scalars: Vec<Vec<f64>> = (0..schema.n_scalars())
        .map(|_| Vec::with_capacity(batch.len()))
        .collect();
    let mut weights = if store_weights {
        Some(Vec::with_capacity(batch.len()))
    } else {
        None
    };

    eval_batch(batch, ops, base, |ev| {
        for (col, p4) in p4s.iter_mut().enumerate() {
            p4.push(ev.p4(col));
        }

        for (col, scalar) in scalars.iter_mut().enumerate() {
            scalar.push(ev.scalar(col));
        }

        if let Some(weights) = weights.as_mut() {
            weights.push(ev.weight());
        }

        Ok(())
    })?;

    let p4s = p4s.into_iter().map(Arc::from).collect();
    let scalars = scalars.into_iter().map(Arc::from).collect();
    let weights = weights.map(Arc::from);

    EventBatch::new(schema, p4s, scalars, weights)
}

fn uniform_hash_01(seed: u64, index: u64) -> f64 {
    let x = splitmix64(seed ^ index);
    ((x >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);

    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn poisson1_from_hash(seed: u64, index: u64) -> u32 {
    let u = uniform_hash_01(seed, index);

    let mut k = 0;
    let mut p = (-1.0_f64).exp();
    let mut cdf = p;

    while u > cdf {
        k += 1;
        p /= k as f64;
        cdf += p;
    }

    k
}

#[cfg(feature = "parallel")]
/// Numerically accurate accumulators used by parallel reductions.
pub mod accurate {
    use accurate::{sum::Sum2, traits::*};
    use num::complex::Complex64;

    /// Compensated accumulator for real values.
    #[derive(Clone)]
    pub struct AccurateF64 {
        sum: Sum2<f64>,
    }

    impl AccurateF64 {
        /// Creates a zero accumulator.
        pub fn zero() -> Self {
            Self { sum: Sum2::zero() }
        }

        /// Adds one value.
        pub fn push(&mut self, value: f64) {
            let sum = std::mem::replace(&mut self.sum, Sum2::zero());
            self.sum = sum + value;
        }

        /// Merges another accumulator.
        pub fn merge(&mut self, other: Self) {
            self.push(other.finish());
        }

        /// Returns the accumulated sum.
        pub fn finish(self) -> f64 {
            self.sum.sum()
        }
    }

    /// Pair of compensated accumulators for complex values.
    #[derive(Clone)]
    pub struct AccurateComplex64 {
        re: AccurateF64,
        im: AccurateF64,
    }

    impl AccurateComplex64 {
        /// Creates a zero accumulator.
        pub fn zero() -> Self {
            Self {
                re: AccurateF64::zero(),
                im: AccurateF64::zero(),
            }
        }

        /// Adds one complex value.
        pub fn push(&mut self, value: Complex64) {
            self.re.push(value.re);
            self.im.push(value.im);
        }

        /// Merges another accumulator.
        pub fn merge(&mut self, other: Self) {
            self.re.merge(other.re);
            self.im.merge(other.im);
        }

        /// Returns the accumulated complex sum.
        pub fn finish(self) -> Complex64 {
            Complex64::new(self.re.finish(), self.im.finish())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{EventSource, ReadPlan, memory::MemorySink};
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
}
