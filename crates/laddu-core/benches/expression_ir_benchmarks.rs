use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use laddu_core::{
    amplitudes::TestAmplitude,
    data::{read_parquet, DatasetReadOptions},
    parameter, Dataset, Evaluator,
};

const BENCH_DATASET_PATH: &str = "benches/bench.parquet";
const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];
const PARAMS: [f64; 2] = [1.25, -0.75];
const SAMPLE_EVENTS: usize = 512;

fn load_bench_dataset() -> Arc<Dataset> {
    read_parquet(
        BENCH_DATASET_PATH,
        &DatasetReadOptions::default()
            .p4_names(P4_NAMES)
            .aux_names(AUX_NAMES),
    )
    .expect("benchmark dataset should open")
}

fn sampled_dataset(dataset: &Arc<Dataset>, n_take: usize) -> Arc<Dataset> {
    let n_take = n_take.min(dataset.events_local().len());
    let events = dataset
        .events_local()
        .iter()
        .take(n_take)
        .map(|event| event.data_arc())
        .collect();
    Arc::new(Dataset::new_with_metadata(events, dataset.metadata_arc()))
}

fn build_test_evaluator(dataset: &Arc<Dataset>) -> Evaluator {
    let expression = TestAmplitude::new("ir_probe", parameter("ir_re"), parameter("ir_im"))
        .expect("test amplitude should construct")
        .norm_sqr();
    expression
        .load(dataset)
        .expect("evaluator should load benchmark dataset")
}

fn expression_backend_benchmarks(c: &mut Criterion) {
    let dataset = load_bench_dataset();
    let sampled = sampled_dataset(&dataset, SAMPLE_EVENTS);
    let evaluator = build_test_evaluator(&sampled);
    let n_events = sampled.n_events() as u64;

    let mut group = c.benchmark_group("expression_backend_paths");
    group.sample_size(30);
    group.throughput(Throughput::Elements(n_events));

    group.bench_function("evaluate_value_small_sample", |b| {
        b.iter(|| black_box(evaluator.evaluate(black_box(&PARAMS))))
    });

    group.bench_function("evaluate_gradient_small_sample", |b| {
        b.iter(|| black_box(evaluator.evaluate_gradient(black_box(&PARAMS))))
    });

    group.bench_function("evaluate_value_and_gradient_small_sample", |b| {
        b.iter(|| {
            let values = evaluator.evaluate(black_box(&PARAMS));
            let gradients = evaluator.evaluate_gradient(black_box(&PARAMS));
            black_box((values, gradients))
        })
    });
    group.finish();
}

criterion_group!(benches, expression_backend_benchmarks);
criterion_main!(benches);
