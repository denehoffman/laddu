//! Synthetic preparation and steady-state likelihood pipeline benchmarks.
#![allow(missing_docs, reason = "criterion benchmark support types are local")]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use laddu::prelude::*;
use laddu_runtime::NormalizationMode;

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

fn model(coefficients: usize) -> CompiledModel {
    let x = event_scalar("x");
    let amplitude = (0..coefficients)
        .map(|index| {
            let re =
                Expr::from(Parameter::free(format!("coefficient_{index}_re")).with_initial(0.8));
            let im =
                Expr::from(Parameter::free(format!("coefficient_{index}_im")).with_initial(-0.1));
            let basis = complex(
                x.clone().powi((index % 3 + 1) as i32) + 0.2,
                (x.clone() * (index + 1) as f64).cos(),
            );
            complex(re, im) * basis
        })
        .reduce(|sum, term| sum + term)
        .expect("benchmark models contain at least one coefficient");
    CompiledModel::from_expr(&amplitude.norm_sqr()).unwrap()
}

fn likelihood(
    source: CountingSource,
    coefficients: usize,
    normalization: NormalizationMode,
) -> Likelihood {
    let dataset = Dataset::new(source).fastest();
    let model = model(coefficients);
    let execution = Execution::local(ExecutionOptions {
        normalization,
        ..ExecutionOptions::default()
    })
    .unwrap();
    Likelihood::with_execution(
        [NllTerm::new("synthetic", &model, &dataset, &dataset).unwrap()],
        &execution,
    )
    .unwrap()
}

fn pipeline_benchmark(criterion: &mut Criterion) {
    let mut preparation = criterion.benchmark_group("likelihood preparation");
    for known in [true, false] {
        for fragments in [1, 1_000] {
            preparation.bench_with_input(
                BenchmarkId::new(if known { "known" } else { "unknown" }, fragments),
                &(known, fragments),
                |bencher, &(known, fragments)| {
                    bencher.iter(|| {
                        likelihood(
                            black_box(source(10_000, fragments, known)),
                            1,
                            NormalizationMode::Auto,
                        )
                    })
                },
            );
        }
    }
    preparation.finish();

    let dataset_source = source(10_000, 1_000, false);
    let traversals = Arc::clone(&dataset_source.traversals);
    let resident_likelihood = likelihood(dataset_source, 1, NormalizationMode::Auto);
    let reads_after_preparation = traversals.load(Ordering::Relaxed);
    let parameters = resident_likelihood.default_params();
    let mut evaluation = criterion.benchmark_group("resident likelihood evaluation");
    evaluation.bench_function("value", |bencher| {
        bencher.iter(|| resident_likelihood.nll(black_box(&parameters)).unwrap())
    });
    evaluation.bench_function("value and gradient", |bencher| {
        bencher.iter(|| {
            resident_likelihood
                .nll_with_gradient(black_box(&parameters))
                .unwrap()
        })
    });
    evaluation.finish();
    assert_eq!(traversals.load(Ordering::Relaxed), reads_after_preparation);

    let mut scaling = criterion.benchmark_group("normalization strategy scaling");
    for coefficients in [1, 4, 8] {
        for normalization in [NormalizationMode::Auto, NormalizationMode::General] {
            let likelihood = likelihood(source(10_000, 1_000, false), coefficients, normalization);
            let parameters = likelihood.default_params();
            let strategy = match normalization {
                NormalizationMode::Auto => "statistics",
                NormalizationMode::General => "general",
                NormalizationMode::Verify => unreachable!(),
            };
            scaling.bench_with_input(
                BenchmarkId::new(strategy, coefficients),
                &coefficients,
                |bencher, _| bencher.iter(|| likelihood.nll(black_box(&parameters)).unwrap()),
            );
        }
    }
    scaling.finish();
}

criterion_group!(benches, pipeline_benchmark);
criterion_main!(benches);
