use criterion::{black_box, criterion_group, criterion_main, Criterion};
use laddu::data::open;

fn open_data_benchmark(c: &mut Criterion) {
    c.bench_function("open benchmark", |b| {
        b.iter(|| {
            black_box(open("benches/bench.parquet").unwrap());
        });
    });
}

criterion_group!(benches, open_data_benchmark);
criterion_main!(benches);
