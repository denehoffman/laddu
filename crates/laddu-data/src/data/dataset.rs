use std::sync::Arc;

use laddu_physics::vectors::RealVec4;
use num::complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::{ThreadPool, prelude::*};

#[cfg(feature = "parallel")]
use crate::data::dataset::accurate::{AccurateComplex64, AccurateF64};
use crate::{
    LadduDataError, LadduDataResult,
    data::event::{Event, EventBatch, OwnedEvent},
    io::{EventSink, EventSource, ReadPlan, SourceCapabilities, WritePlan, memory::MemorySource},
    schema::Schema,
};

#[derive(Clone)]
enum DatasetOp {
    Filter(Arc<dyn Fn(Event<'_>) -> bool + Send + Sync>),
    Subsample { fraction: f64, seed: u64 },
    Bootstrap { seed: u64 },
}

#[derive(Clone)]
pub struct Dataset {
    source: Arc<dyn EventSource>,
    plan: ReadPlan,
    ops: Arc<[DatasetOp]>,
}

impl Dataset {
    pub fn new<S>(source: S) -> Self
    where
        S: EventSource + 'static,
    {
        Self {
            source: Arc::new(source),
            plan: ReadPlan::default(),
            ops: Arc::from([]),
        }
    }

    pub fn from_arc(source: Arc<dyn EventSource>) -> Self {
        Self {
            source,
            plan: ReadPlan::default(),
            ops: Arc::from([]),
        }
    }

    pub fn from_batch(batch: EventBatch) -> Self {
        Self::new(MemorySource::new(batch))
    }

    pub fn from_batches(batches: Vec<EventBatch>) -> LadduDataResult<Self> {
        Ok(Self::new(MemorySource::from_batches(batches)?))
    }

    pub fn from_events<I>(schema: Arc<Schema>, events: I) -> LadduDataResult<Self>
    where
        I: IntoIterator<Item = OwnedEvent>,
    {
        Ok(Self::new(MemorySource::from_events(schema, events)?))
    }

    pub fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        self.source.schema()
    }

    pub fn capabilities(&self) -> SourceCapabilities {
        self.source.capabilities()
    }

    pub fn read_plan(&self) -> ReadPlan {
        self.plan
    }

    pub fn chunked(mut self, chunk_size: usize) -> LadduDataResult<Self> {
        if chunk_size == 0 {
            return Err(LadduDataError::InvalidArgument(
                "chunk_size must be nonzero",
            ));
        }
        self.plan.chunk_size = Some(chunk_size);
        Ok(self)
    }

    pub fn unchunked(mut self) -> Self {
        self.plan.chunk_size = None;
        self
    }

    pub fn filter<F>(self, f: F) -> Self
    where
        F: Fn(Event<'_>) -> bool + Send + Sync + 'static,
    {
        self.push_op(DatasetOp::Filter(Arc::new(f)))
    }

    pub fn subsample(self, fraction: f64, seed: u64) -> LadduDataResult<Self> {
        if !(0.0..=1.0).contains(&fraction) {
            return Err(LadduDataError::InvalidArgument(
                "fraction must be in [0, 1]",
            ));
        }

        Ok(self.push_op(DatasetOp::Subsample { fraction, seed }))
    }

    pub fn bootstrap(self, seed: u64) -> Self {
        self.push_op(DatasetOp::Bootstrap { seed })
    }

    pub fn for_each_event<F>(&self, mut f: F) -> LadduDataResult<()>
    where
        F: FnMut(Event<'_>),
    {
        self.try_for_each_event(|ev| {
            f(ev);
            Ok(())
        })
    }

    pub fn try_for_each_event<F>(&self, mut f: F) -> LadduDataResult<()>
    where
        F: FnMut(Event<'_>) -> LadduDataResult<()>,
    {
        let mut offset = 0_u64;

        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;
            eval_batch(&batch, &self.ops, base, |ev| f(ev))?;
        }

        Ok(())
    }

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

    pub fn try_fold_events<T, F>(&self, init: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(T, Event<'_>) -> LadduDataResult<T>,
    {
        let mut acc = Some(init);

        self.try_for_each_event(|ev| {
            let current = acc
                .take()
                .expect("fold accumulator should always be present");
            acc = Some(f(current, ev)?);
            Ok(())
        })?;

        Ok(acc.expect("fold accumulator should always be present"))
    }

    pub fn fold_events<T, F>(&self, init: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(T, Event<'_>) -> T,
    {
        self.try_fold_events(init, |acc, ev| Ok(f(acc, ev)))
    }

    pub fn try_accumulate_events<T, F>(&self, mut acc: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(&mut T, Event<'_>) -> LadduDataResult<()>,
    {
        self.try_for_each_event(|ev| f(&mut acc, ev))?;
        Ok(acc)
    }

    pub fn accumulate_events<T, F>(&self, acc: T, mut f: F) -> LadduDataResult<T>
    where
        F: FnMut(&mut T, Event<'_>),
    {
        self.try_accumulate_events(acc, |acc, ev| {
            f(acc, ev);
            Ok(())
        })
    }

    pub fn sum_weights(&self) -> LadduDataResult<f64> {
        self.fold_events(0.0, |sum, ev| sum + ev.weight())
    }

    pub fn weighted_sum<F>(&self, mut f: F) -> LadduDataResult<f64>
    where
        F: FnMut(Event<'_>) -> f64,
    {
        self.fold_events(0.0, |sum, ev| sum + ev.weight() * f(ev))
    }

    pub fn weighted_complex_sum<F>(&self, mut f: F) -> LadduDataResult<Complex64>
    where
        F: FnMut(Event<'_>) -> Complex64,
    {
        self.fold_events(0.0.into(), |sum, ev| sum + ev.weight() * f(ev))
    }

    pub fn batches(
        &self,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        let iter = self.source.batches(self.plan)?;
        let ops = Arc::clone(&self.ops);

        Ok(Box::new(iter.scan(0_u64, move |offset, batch| {
            let batch = match batch {
                Ok(batch) => batch,
                Err(err) => return Some(Err(err)),
            };

            let base = *offset;
            *offset += batch.len() as u64;

            Some(materialize_batch(&batch, &ops, base))
        })))
    }

    pub fn try_for_each_batch<F>(&self, mut f: F) -> LadduDataResult<()>
    where
        F: FnMut(EventBatch) -> LadduDataResult<()>,
    {
        for batch in self.batches()? {
            f(batch?)?;
        }

        Ok(())
    }

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

    pub fn write_to<S: EventSink>(&self, sink: &mut S) -> LadduDataResult<()> {
        sink.begin(self.schema()?, WritePlan::from(self.plan))?;

        for batch in self.batches()? {
            sink.write_batch(&batch?)?;
        }

        sink.finish()
    }

    fn push_op(self, op: DatasetOp) -> Self {
        let mut ops = self.ops.to_vec();
        ops.push(op);

        Self {
            source: self.source,
            plan: self.plan,
            ops: ops.into(),
        }
    }
}

#[cfg(feature = "parallel")]
impl Dataset {
    pub fn par_for_each_event<F>(&self, f: F) -> LadduDataResult<()>
    where
        F: Fn(Event<'_>) + Send + Sync,
    {
        self.par_try_for_each_event(|ev| {
            f(ev);
            Ok(())
        })
    }

    pub fn par_for_each_event_in<F>(&self, pool: &ThreadPool, f: F) -> LadduDataResult<()>
    where
        F: Fn(Event<'_>) + Send + Sync,
    {
        pool.install(move || self.par_for_each_event(f))
    }

    pub fn par_try_for_each_event<F>(&self, f: F) -> LadduDataResult<()>
    where
        F: Fn(Event<'_>) -> LadduDataResult<()> + Send + Sync,
    {
        let mut offset = 0_u64;

        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;

            (0..batch.len())
                .into_par_iter()
                .filter_map(|row| eval_event(&batch, &self.ops, base, row))
                .try_for_each(|ev| f(ev))?;
        }

        Ok(())
    }

    pub fn par_try_for_each_event_in<F>(&self, pool: &ThreadPool, f: F) -> LadduDataResult<()>
    where
        F: Fn(Event<'_>) -> LadduDataResult<()> + Send + Sync,
    {
        pool.install(move || self.par_try_for_each_event(f))
    }

    pub fn par_try_map_events<T, F>(&self, f: F) -> LadduDataResult<Vec<T>>
    where
        T: Send,
        F: Fn(Event<'_>) -> LadduDataResult<T> + Send + Sync,
    {
        let mut out = Vec::new();
        let mut offset = 0_u64;

        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;

            let mut batch_out = (0..batch.len())
                .into_par_iter()
                .filter_map(|row| eval_event(&batch, &self.ops, base, row))
                .map(|ev| f(ev))
                .collect::<LadduDataResult<Vec<_>>>()?;

            out.append(&mut batch_out);
        }

        Ok(out)
    }

    pub fn par_try_map_events_in<T, F>(&self, pool: &ThreadPool, f: F) -> LadduDataResult<Vec<T>>
    where
        T: Send,
        F: Fn(Event<'_>) -> LadduDataResult<T> + Send + Sync,
    {
        pool.install(move || self.par_try_map_events(f))
    }

    pub fn par_map_events<T, F>(&self, f: F) -> LadduDataResult<Vec<T>>
    where
        T: Send,
        F: Fn(Event<'_>) -> T + Send + Sync,
    {
        self.par_try_map_events(|ev| Ok(f(ev)))
    }

    pub fn par_map_events_in<T, F>(&self, pool: &ThreadPool, f: F) -> LadduDataResult<Vec<T>>
    where
        T: Send,
        F: Fn(Event<'_>) -> T + Send + Sync,
    {
        pool.install(move || self.par_map_events(f))
    }

    pub fn par_try_fold_events<T, Init, Fold, Reduce>(
        &self,
        init: Init,
        fold: Fold,
        reduce: Reduce,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Fold: Fn(T, Event<'_>) -> LadduDataResult<T> + Send + Sync,
        Reduce: Fn(T, T) -> LadduDataResult<T> + Send + Sync,
    {
        let mut total = init();
        let mut offset = 0_u64;

        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;

            let partial = (0..batch.len())
                .into_par_iter()
                .filter_map(|row| eval_event(&batch, &self.ops, base, row))
                .try_fold(&init, |acc, ev| fold(acc, ev))
                .try_reduce(&init, |a, b| reduce(a, b))?;

            total = reduce(total, partial)?;
        }

        Ok(total)
    }

    pub fn par_try_fold_events_in<T, Init, Fold, Reduce>(
        &self,
        pool: &ThreadPool,
        init: Init,
        fold: Fold,
        reduce: Reduce,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Fold: Fn(T, Event<'_>) -> LadduDataResult<T> + Send + Sync,
        Reduce: Fn(T, T) -> LadduDataResult<T> + Send + Sync,
    {
        pool.install(move || self.par_try_fold_events(init, fold, reduce))
    }

    pub fn par_fold_events<T, Init, Fold, Reduce>(
        &self,
        init: Init,
        fold: Fold,
        reduce: Reduce,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Fold: Fn(T, Event<'_>) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        self.par_try_fold_events(init, |acc, ev| Ok(fold(acc, ev)), |a, b| Ok(reduce(a, b)))
    }

    pub fn par_fold_events_in<T, Init, Fold, Reduce>(
        &self,
        pool: &ThreadPool,
        init: Init,
        fold: Fold,
        reduce: Reduce,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Fold: Fn(T, Event<'_>) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        pool.install(move || self.par_fold_events(init, fold, reduce))
    }

    pub fn par_try_accumulate_events<T, Init, Accumulate, Merge>(
        &self,
        init: Init,
        accumulate: Accumulate,
        merge: Merge,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Accumulate: Fn(&mut T, Event<'_>) -> LadduDataResult<()> + Send + Sync,
        Merge: Fn(&mut T, T) -> LadduDataResult<()> + Send + Sync,
    {
        let mut total = init();
        let mut offset = 0_u64;

        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;

            let partial = (0..batch.len())
                .into_par_iter()
                .filter_map(|row| eval_event(&batch, &self.ops, base, row))
                .try_fold(&init, |mut acc, ev| {
                    accumulate(&mut acc, ev)?;
                    Ok(acc)
                })
                .try_reduce(&init, |mut a, b| {
                    merge(&mut a, b)?;
                    Ok(a)
                })?;

            merge(&mut total, partial)?;
        }

        Ok(total)
    }

    pub fn par_try_accumulate_events_in<T, Init, Accumulate, Merge>(
        &self,
        pool: &ThreadPool,
        init: Init,
        accumulate: Accumulate,
        merge: Merge,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Accumulate: Fn(&mut T, Event<'_>) -> LadduDataResult<()> + Send + Sync,
        Merge: Fn(&mut T, T) -> LadduDataResult<()> + Send + Sync,
    {
        pool.install(move || self.par_try_accumulate_events(init, accumulate, merge))
    }

    pub fn par_accumulate_events<T, Init, Accumulate, Merge>(
        &self,
        init: Init,
        accumulate: Accumulate,
        merge: Merge,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Accumulate: Fn(&mut T, Event<'_>) + Send + Sync,
        Merge: Fn(&mut T, T) + Send + Sync,
    {
        self.par_try_accumulate_events(
            init,
            |acc, ev| {
                accumulate(acc, ev);
                Ok(())
            },
            |acc, other| {
                merge(acc, other);
                Ok(())
            },
        )
    }

    pub fn par_accumulate_events_in<T, Init, Accumulate, Merge>(
        &self,
        pool: &ThreadPool,
        init: Init,
        accumulate: Accumulate,
        merge: Merge,
    ) -> LadduDataResult<T>
    where
        T: Send,
        Init: Fn() -> T + Send + Sync,
        Accumulate: Fn(&mut T, Event<'_>) + Send + Sync,
        Merge: Fn(&mut T, T) + Send + Sync,
    {
        pool.install(move || self.par_accumulate_events(init, accumulate, merge))
    }

    pub fn par_sum_weights(&self) -> LadduDataResult<f64> {
        self.par_sum_real(|ev| ev.weight())
    }

    pub fn par_sum_weights_in(&self, pool: &ThreadPool) -> LadduDataResult<f64> {
        pool.install(|| self.par_sum_weights())
    }

    pub fn par_weighted_sum<F>(&self, f: F) -> LadduDataResult<f64>
    where
        F: Fn(Event<'_>) -> f64 + Send + Sync,
    {
        self.par_sum_real(|ev| ev.weight() * f(ev))
    }

    pub fn par_weighted_sum_in<F>(&self, pool: &ThreadPool, f: F) -> LadduDataResult<f64>
    where
        F: Fn(Event<'_>) -> f64 + Send + Sync,
    {
        pool.install(move || self.par_weighted_sum(f))
    }

    pub fn par_weighted_complex_sum<F>(&self, f: F) -> LadduDataResult<Complex64>
    where
        F: Fn(Event<'_>) -> Complex64 + Send + Sync,
    {
        self.par_sum_complex(|ev| f(ev) * ev.weight())
    }

    pub fn par_weighted_complex_sum_in<F>(
        &self,
        pool: &ThreadPool,
        f: F,
    ) -> LadduDataResult<Complex64>
    where
        F: Fn(Event<'_>) -> Complex64 + Send + Sync,
    {
        pool.install(move || self.par_weighted_complex_sum(f))
    }

    fn par_sum_real<F>(&self, f: F) -> LadduDataResult<f64>
    where
        F: Fn(Event<'_>) -> f64 + Send + Sync,
    {
        let mut total = AccurateF64::zero();
        let mut offset = 0_u64;

        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;

            let partial = (0..batch.len())
                .into_par_iter()
                .filter_map(|row| eval_event(&batch, &self.ops, base, row))
                .map(|ev| f(ev))
                .fold(AccurateF64::zero, |mut acc, value| {
                    acc.push(value);
                    acc
                })
                .reduce(AccurateF64::zero, |mut a, b| {
                    a.merge(b);
                    a
                });

            total.merge(partial);
        }

        Ok(total.finish())
    }

    fn par_sum_complex<F>(&self, f: F) -> LadduDataResult<Complex64>
    where
        F: Fn(Event<'_>) -> Complex64 + Send + Sync,
    {
        let mut total = AccurateComplex64::zero();
        let mut offset = 0_u64;

        for batch in self.source.batches(self.plan)? {
            let batch = batch?;
            let base = offset;
            offset += batch.len() as u64;

            let partial = (0..batch.len())
                .into_par_iter()
                .filter_map(|row| eval_event(&batch, &self.ops, base, row))
                .map(|ev| f(ev))
                .fold(AccurateComplex64::zero, |mut acc, value| {
                    acc.push(value);
                    acc
                })
                .reduce(AccurateComplex64::zero, |mut a, b| {
                    a.merge(b);
                    a
                });

            total.merge(partial);
        }

        Ok(total.finish())
    }
}

#[cfg(feature = "mpi")]
impl Dataset {
    pub fn distributed<C>(mut self, world: &C) -> Self
    where
        C: mpi::topology::Communicator,
    {
        self.plan.distribution = Distribution::from_world(world);
        self
    }

    pub fn partitioning(mut self, partitioning: Partitioning) -> Self {
        self.plan.distribution = match self.plan.distribution {
            Distribution::Serial => Distribution::Serial,
            Distribution::Mpi {
                rank,
                nranks,
                partitioning: _,
            } => Distribution::Mpi {
                rank,
                nranks,
                partitioning,
            },
        };

        self
    }
}

fn eval_event<'a>(
    batch: &'a EventBatch,
    ops: &[DatasetOp],
    base: u64,
    row: usize,
) -> Option<Event<'a>> {
    let event_id = base + row as u64;
    let mut weight = batch.weights_at(row);

    for op in ops {
        match op {
            DatasetOp::Filter(pred) => {
                let ev = Event { batch, row, weight };
                if !pred(ev) {
                    return None;
                }
            }
            DatasetOp::Subsample { fraction, seed } => {
                if uniform_hash_01(*seed, event_id) >= *fraction {
                    return None;
                }
            }
            DatasetOp::Bootstrap { seed } => {
                weight *= poisson1_from_hash(*seed, event_id) as f64;
            }
        }
    }

    Some(Event { batch, row, weight })
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
        for col in 0..schema.n_p4s() {
            p4s[col].push(ev.p4(col));
        }

        for col in 0..schema.n_scalars() {
            scalars[col].push(ev.scalar(col));
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
pub mod accurate {
    use accurate::{sum::Sum2, traits::*};
    use num::complex::Complex64;

    #[derive(Clone)]
    pub struct AccurateF64 {
        sum: Sum2<f64>,
    }

    impl AccurateF64 {
        pub fn zero() -> Self {
            Self { sum: Sum2::zero() }
        }

        pub fn push(&mut self, value: f64) {
            let sum = std::mem::replace(&mut self.sum, Sum2::zero());
            self.sum = sum + value;
        }

        pub fn merge(&mut self, other: Self) {
            self.push(other.finish());
        }

        pub fn finish(self) -> f64 {
            self.sum.sum()
        }
    }

    #[derive(Clone)]
    pub struct AccurateComplex64 {
        re: AccurateF64,
        im: AccurateF64,
    }

    impl AccurateComplex64 {
        pub fn zero() -> Self {
            Self {
                re: AccurateF64::zero(),
                im: AccurateF64::zero(),
            }
        }

        pub fn push(&mut self, value: Complex64) {
            self.re.push(value.re);
            self.im.push(value.im);
        }

        pub fn merge(&mut self, other: Self) {
            self.re.merge(other.re);
            self.im.merge(other.im);
        }

        pub fn finish(self) -> Complex64 {
            Complex64::new(self.re.finish(), self.im.finish())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::memory::MemorySink;

    fn v(x: f64) -> RealVec4 {
        RealVec4 {
            x: x,
            y: x + 0.1,
            z: x + 0.2,
            t: x + 0.3,
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
        let seed = 0xBAD5_EED;
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

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_dataset_methods_match_serial_methods_and_custom_pool() {
        let dataset = Dataset::from_batches(vec![weighted_batch(0, 64), weighted_batch(64, 64)])
            .unwrap()
            .filter(|ev| ev.scalar(0) % 3.0 != 1.0)
            .bootstrap(12345);

        let serial_values = dataset.map_events(|ev| ev.scalar(0)).unwrap();
        let parallel_values = dataset.par_map_events(|ev| ev.scalar(0)).unwrap();
        assert_eq!(parallel_values, serial_values);

        let serial_weighted = dataset.weighted_sum(|ev| ev.scalar(0).sin()).unwrap();
        let parallel_weighted = dataset.par_weighted_sum(|ev| ev.scalar(0).sin()).unwrap();
        assert!((parallel_weighted - serial_weighted).abs() < 1.0e-10);

        let serial_complex = dataset
            .weighted_complex_sum(|ev| Complex64::new(ev.scalar(0), ev.scalar(0).cos()))
            .unwrap();

        let parallel_complex = dataset
            .par_weighted_complex_sum(|ev| Complex64::new(ev.scalar(0), ev.scalar(0).cos()))
            .unwrap();

        assert!((parallel_complex.re - serial_complex.re).abs() < 1.0e-10);
        assert!((parallel_complex.im - serial_complex.im).abs() < 1.0e-10);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let pooled = dataset
            .par_weighted_sum_in(&pool, |ev| ev.scalar(0).sqrt())
            .unwrap();

        let serial = dataset.weighted_sum(|ev| ev.scalar(0).sqrt()).unwrap();

        assert!((pooled - serial).abs() < 1.0e-10);
    }

    #[cfg(feature = "mpi")]
    use mpi::traits::*;
    #[cfg(feature = "mpi")]
    use mpi_test::mpi_test;

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2, 3, 4])]
    fn mpi_dataset_distributed_rows_partitioning_matches_global_row_modulo_rule() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let rank = world.rank() as usize;
        let nranks = world.size() as usize;

        let dataset = Dataset::from_batches(vec![weighted_batch(0, 5), weighted_batch(5, 7)])
            .unwrap()
            .distributed(&world)
            .partitioning(Partitioning::Rows);

        assert_eq!(dataset.read_plan().rank(), rank);
        assert_eq!(dataset.read_plan().nranks(), nranks);
        assert!(dataset.read_plan().is_distributed());

        let observed = dataset.map_events(|ev| ev.scalar(0) as usize).unwrap();

        let expected = (0..12)
            .filter(|row| row % nranks == rank)
            .collect::<Vec<_>>();

        assert_eq!(observed, expected);
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2, 3, 4])]
    fn mpi_dataset_distributed_contiguous_partitioning_covers_rank_range() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let rank = world.rank() as usize;
        let nranks = world.size() as usize;

        let n = 17usize;

        let dataset = Dataset::from_batch(weighted_batch(0, n))
            .distributed(&world)
            .partitioning(Partitioning::Contiguous);

        let observed = dataset.map_events(|ev| ev.scalar(0) as usize).unwrap();

        let start = n * rank / nranks;
        let end = n * (rank + 1) / nranks;
        let expected = (start..end).collect::<Vec<_>>();

        assert_eq!(observed, expected);
    }

    #[cfg(feature = "mpi")]
    #[mpi_test(np = [2, 3])]
    fn mpi_dataset_distributed_file_group_partitioning_reads_whole_memory_fragments() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let rank = world.rank() as usize;
        let nranks = world.size() as usize;

        let dataset = Dataset::from_batches(vec![
            weighted_batch(0, 2),
            weighted_batch(2, 3),
            weighted_batch(5, 4),
            weighted_batch(9, 1),
        ])
        .unwrap()
        .distributed(&world)
        .partitioning(Partitioning::FileGroups);

        let observed = dataset.map_events(|ev| ev.scalar(0) as usize).unwrap();

        let fragment_values = [vec![0, 1], vec![2, 3, 4], vec![5, 6, 7, 8], vec![9]];

        let expected = fragment_values
            .into_iter()
            .enumerate()
            .filter(|(fragment_index, _)| fragment_index % nranks == rank)
            .flat_map(|(_, values)| values)
            .collect::<Vec<_>>();

        assert_eq!(observed, expected);
    }
}
