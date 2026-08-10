//! Synthetic preparation and steady-state likelihood pipeline benchmarks.
#![allow(missing_docs, reason = "criterion benchmark support types are local")]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use laddu::prelude::*;

#[derive(Clone)]
struct CountingSource {
    schema: Arc<Schema>,
    batches: Arc<[EventBatch]>,
    known_events: bool,
    traversals: Arc<AtomicUsize>,
    emitted_rows: Arc<AtomicUsize>,
}

impl EventSource for CountingSource {
    fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        Ok(Arc::clone(&self.schema))
    }

    fn num_events(&self) -> LadduDataResult<Option<u64>> {
        Ok(self
            .known_events
            .then(|| self.batches.iter().map(|batch| batch.len() as u64).sum()))
    }

    fn batches(
        &self,
        _plan: ReadPlan,
    ) -> LadduDataResult<Box<dyn Iterator<Item = LadduDataResult<EventBatch>> + Send>> {
        self.traversals.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(CountingIter {
            batches: Arc::clone(&self.batches),
            emitted_rows: Arc::clone(&self.emitted_rows),
            index: 0,
        }))
    }
}

struct CountingIter {
    batches: Arc<[EventBatch]>,
    emitted_rows: Arc<AtomicUsize>,
    index: usize,
}

impl Iterator for CountingIter {
    type Item = LadduDataResult<EventBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.batches.get(self.index)?.clone();
        self.index += 1;
        self.emitted_rows.fetch_add(batch.len(), Ordering::Relaxed);
        Some(Ok(batch))
    }
}

fn source(events: usize, fragments: usize, known_events: bool) -> CountingSource {
    let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
    let rows = (0..events)
        .map(|index| {
            let x = index as f64 / events as f64;
            OwnedEvent::weighted(vec![], vec![x], 0.5 + x)
        })
        .collect::<Vec<_>>();
    let width = events.div_ceil(fragments).max(1);
    let batches = rows
        .chunks(width)
        .map(|chunk| EventBatch::from_events(Arc::clone(&schema), chunk.iter().cloned()).unwrap())
        .collect::<Vec<_>>();
    CountingSource {
        schema,
        batches: batches.into(),
        known_events,
        traversals: Default::default(),
        emitted_rows: Default::default(),
    }
}

fn model() -> CompiledModel {
    let coefficient = complex(
        parameter!("coefficient_re", initial: 0.8),
        parameter!("coefficient_im", initial: -0.1),
    );
    let amplitude = complex(event_scalar("x") + 0.2, event_scalar("x").cos());
    CompiledModel::from_expr(&(coefficient * amplitude).norm_sqr()).unwrap()
}

fn likelihood(source: CountingSource) -> Likelihood {
    let dataset = Dataset::new(source).fastest();
    let model = model();
    Likelihood::new([NllTerm::new("synthetic", &model, &dataset, &dataset).unwrap()]).unwrap()
}

fn pipeline_benchmark(criterion: &mut Criterion) {
    let mut preparation = criterion.benchmark_group("likelihood preparation");
    for known in [true, false] {
        for fragments in [1, 1_000] {
            preparation.bench_with_input(
                BenchmarkId::new(if known { "known" } else { "unknown" }, fragments),
                &(known, fragments),
                |bencher, &(known, fragments)| {
                    bencher.iter(|| likelihood(black_box(source(10_000, fragments, known))))
                },
            );
        }
    }
    preparation.finish();

    let source = source(10_000, 1_000, false);
    let traversals = Arc::clone(&source.traversals);
    let likelihood = likelihood(source);
    let reads_after_preparation = traversals.load(Ordering::Relaxed);
    let parameters = likelihood.default_params();
    let mut evaluation = criterion.benchmark_group("resident likelihood evaluation");
    evaluation.bench_function("value", |bencher| {
        bencher.iter(|| likelihood.nll(black_box(&parameters)).unwrap())
    });
    evaluation.bench_function("value and gradient", |bencher| {
        bencher.iter(|| {
            likelihood
                .nll_with_gradient(black_box(&parameters))
                .unwrap()
        })
    });
    evaluation.finish();
    assert_eq!(traversals.load(Ordering::Relaxed), reads_after_preparation);
}

criterion_group!(benches, pipeline_benchmark);
criterion_main!(benches);
