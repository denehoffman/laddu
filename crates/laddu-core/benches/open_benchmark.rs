use criterion::{black_box, criterion_group, criterion_main, Criterion};
use laddu_core::data::{Dataset, DatasetReadOptions};

fn open_data_benchmark(c: &mut Criterion) {
    c.bench_function("open benchmark", |b| {
        b.iter(|| {
            let p4_names = ["beam", "proton", "kshort1", "kshort2"];
            let aux_names = ["pol_magnitude", "pol_angle"];
            black_box(
                Dataset::from_parquet(
                    "benches/bench.parquet",
                    &DatasetReadOptions::default()
                        .p4_names(p4_names)
                        .aux_names(aux_names),
                )
                .unwrap(),
            );
        });
    });
}

criterion_group!(benches, open_data_benchmark);
criterion_main!(benches);
