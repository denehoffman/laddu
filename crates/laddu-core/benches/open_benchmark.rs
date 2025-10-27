use criterion::{black_box, criterion_group, criterion_main, Criterion};
use laddu_core::data::Dataset;

fn open_data_benchmark(c: &mut Criterion) {
    c.bench_function("open benchmark", |b| {
        b.iter(|| {
            black_box(Dataset::open("benches/bench.parquet").unwrap());
        });
    });
}

criterion_group!(benches, open_data_benchmark);
criterion_main!(benches);
