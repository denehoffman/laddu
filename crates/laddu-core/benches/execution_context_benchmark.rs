use std::{sync::Arc, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use laddu_core::{
    amplitudes::TestAmplitude,
    data::{read_parquet, DatasetReadOptions},
    parameter, Dataset, Evaluator,
};
#[cfg(feature = "execution-context-prototype")]
use laddu_core::{ExecutionContext, ThreadPolicy};

const BENCH_DATASET_PATH: &str = "benches/bench.parquet";
const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];
const PARAMS: [f64; 2] = [1.25, -0.75];
const TIERS: [(&str, usize); 2] = [("tiny", 256), ("full", usize::MAX)];

fn load_bench_dataset() -> Arc<Dataset> {
    read_parquet(
        BENCH_DATASET_PATH,
        &DatasetReadOptions::default()
            .p4_names(P4_NAMES)
            .aux_names(AUX_NAMES),
    )
    .expect("benchmark dataset should open")
}

fn dataset_tier(dataset: &Arc<Dataset>, max_events: usize) -> Arc<Dataset> {
    if max_events >= dataset.events_local().len() {
        return dataset.clone();
    }
    let events = dataset
        .events_local()
        .iter()
        .take(max_events)
        .map(|event| event.data_arc())
        .collect();
    Arc::new(Dataset::new_with_metadata(events, dataset.metadata_arc()))
}

fn build_test_evaluator(dataset: &Arc<Dataset>) -> Evaluator {
    let expression = TestAmplitude::new("ctx_probe", parameter("ctx_re"), parameter("ctx_im"))
        .expect("test amplitude should construct")
        .norm_sqr();
    expression
        .load(dataset)
        .expect("evaluator should load benchmark dataset")
}

#[cfg(feature = "execution-context-prototype")]
fn context_creation_overhead_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_context_creation_overhead");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(800));
    group.measurement_time(Duration::from_millis(1200));

    group.bench_function("new_single", |b| {
        b.iter(|| black_box(ExecutionContext::new(ThreadPolicy::Single).expect("single context")))
    });

    #[cfg(feature = "rayon")]
    group.bench_function("new_global_pool", |b| {
        b.iter(|| {
            black_box(ExecutionContext::new(ThreadPolicy::GlobalPool).expect("global-pool context"))
        })
    });

    #[cfg(feature = "rayon")]
    group.bench_function("new_dedicated_4", |b| {
        b.iter(|| black_box(ExecutionContext::new(ThreadPolicy::Dedicated(4)).expect("pool")))
    });
    group.finish();
}

#[cfg(not(feature = "execution-context-prototype"))]
fn context_creation_overhead_benchmarks(_c: &mut Criterion) {}

#[cfg(feature = "execution-context-prototype")]
fn evaluate_path_benchmarks(c: &mut Criterion) {
    let dataset = load_bench_dataset();
    let mut group = c.benchmark_group("execution_context_evaluate_paths");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for (label, max_events) in TIERS {
        let tier_dataset = dataset_tier(&dataset, max_events);
        let evaluator = build_test_evaluator(&tier_dataset);

        group.bench_with_input(
            BenchmarkId::new("baseline_evaluate", label),
            &label,
            |b, &_label| b.iter(|| black_box(evaluator.evaluate(black_box(&PARAMS)))),
        );

        group.bench_with_input(
            BenchmarkId::new("ctx_single_reused_evaluate", label),
            &label,
            |b, &_label| {
                let ctx = ExecutionContext::new(ThreadPolicy::Single)
                    .expect("single policy should build");
                b.iter(|| black_box(evaluator.evaluate_with_ctx(black_box(&PARAMS), &ctx)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ctx_single_new_per_call_evaluate", label),
            &label,
            |b, &_label| {
                b.iter(|| {
                    let ctx = ExecutionContext::new(ThreadPolicy::Single)
                        .expect("single policy should build");
                    black_box(evaluator.evaluate_with_ctx(black_box(&PARAMS), &ctx))
                })
            },
        );

        #[cfg(feature = "rayon")]
        group.bench_with_input(
            BenchmarkId::new("ctx_global_reused_evaluate", label),
            &label,
            |b, &_label| {
                let ctx = ExecutionContext::new(ThreadPolicy::GlobalPool)
                    .expect("global policy should build");
                b.iter(|| black_box(evaluator.evaluate_with_ctx(black_box(&PARAMS), &ctx)))
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "execution-context-prototype"))]
fn evaluate_path_benchmarks(_c: &mut Criterion) {}

#[cfg(feature = "execution-context-prototype")]
fn gradient_path_benchmarks(c: &mut Criterion) {
    let dataset = load_bench_dataset();
    let mut group = c.benchmark_group("execution_context_gradient_paths");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for (label, max_events) in TIERS {
        let tier_dataset = dataset_tier(&dataset, max_events);
        let evaluator = build_test_evaluator(&tier_dataset);

        group.bench_with_input(
            BenchmarkId::new("baseline_evaluate_gradient", label),
            &label,
            |b, &_label| b.iter(|| black_box(evaluator.evaluate_gradient(black_box(&PARAMS)))),
        );

        group.bench_with_input(
            BenchmarkId::new("ctx_single_reused_evaluate_gradient", label),
            &label,
            |b, &_label| {
                let ctx = ExecutionContext::new(ThreadPolicy::Single)
                    .expect("single policy should build");
                b.iter(|| black_box(evaluator.evaluate_gradient_with_ctx(black_box(&PARAMS), &ctx)))
            },
        );

        #[cfg(feature = "rayon")]
        group.bench_with_input(
            BenchmarkId::new("ctx_global_reused_evaluate_gradient", label),
            &label,
            |b, &_label| {
                let ctx = ExecutionContext::new(ThreadPolicy::GlobalPool)
                    .expect("global policy should build");
                b.iter(|| black_box(evaluator.evaluate_gradient_with_ctx(black_box(&PARAMS), &ctx)))
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "execution-context-prototype"))]
fn gradient_path_benchmarks(_c: &mut Criterion) {}

#[cfg(feature = "execution-context-prototype")]
fn repeated_value_plus_gradient_batch_benchmarks(c: &mut Criterion) {
    let dataset = load_bench_dataset();
    let mut group = c.benchmark_group("execution_context_repeated_calls");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for (label, max_events) in TIERS {
        let tier_dataset = dataset_tier(&dataset, max_events);
        let evaluator = build_test_evaluator(&tier_dataset);

        group.bench_with_input(
            BenchmarkId::new("baseline_value_then_gradient", label),
            &label,
            |b, &_label| {
                b.iter_batched(
                    || PARAMS.to_vec(),
                    |parameters| {
                        let values = evaluator.evaluate(&parameters);
                        let gradients = evaluator.evaluate_gradient(&parameters);
                        black_box((values, gradients))
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ctx_single_reused_value_then_gradient", label),
            &label,
            |b, &_label| {
                let ctx = ExecutionContext::new(ThreadPolicy::Single)
                    .expect("single policy should build");
                b.iter_batched(
                    || PARAMS.to_vec(),
                    |parameters| {
                        let values = evaluator.evaluate_with_ctx(&parameters, &ctx);
                        let gradients = evaluator.evaluate_gradient_with_ctx(&parameters, &ctx);
                        black_box((values, gradients))
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        #[cfg(feature = "rayon")]
        group.bench_with_input(
            BenchmarkId::new("ctx_global_reused_value_then_gradient", label),
            &label,
            |b, &_label| {
                let ctx = ExecutionContext::new(ThreadPolicy::GlobalPool)
                    .expect("global policy should build");
                b.iter_batched(
                    || PARAMS.to_vec(),
                    |parameters| {
                        let values = evaluator.evaluate_with_ctx(&parameters, &ctx);
                        let gradients = evaluator.evaluate_gradient_with_ctx(&parameters, &ctx);
                        black_box((values, gradients))
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "execution-context-prototype"))]
fn repeated_value_plus_gradient_batch_benchmarks(_c: &mut Criterion) {}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = context_creation_overhead_benchmarks,
        evaluate_path_benchmarks,
        gradient_path_benchmarks,
        repeated_value_plus_gradient_batch_benchmarks
);
criterion_main!(benches);
