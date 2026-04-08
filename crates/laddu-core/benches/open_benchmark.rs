use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use laddu_core::data::{
    read_parquet, read_parquet_chunks_with_options, read_root, DatasetReadOptions,
};
use std::path::PathBuf;

const DATASETS: [(&str, &str, &str); 4] = [
    ("parquet", "f32", "data_f32.parquet"),
    ("parquet", "f64", "data_f64.parquet"),
    ("root", "f32", "data_f32.root"),
    ("root", "f64", "data_f64.root"),
];

fn dataset_path(file_name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(file_name)
        .to_string_lossy()
        .into_owned()
}

fn io_read_matrix_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_read_matrix_small");
    let base_options = DatasetReadOptions::new()
        .p4_names(["beam", "proton", "kshort1", "kshort2"])
        .aux_names(["pol_magnitude", "pol_angle"]);

    for (format, precision, file_name) in DATASETS {
        let path = dataset_path(file_name);

        match format {
            "parquet" => {
                group.bench_with_input(
                    BenchmarkId::new("read_full", format!("{format}_{precision}_small")),
                    &path,
                    |b, input_path| {
                        b.iter(|| {
                            let dataset = read_parquet(input_path, &base_options)
                                .expect("benchmark parquet read_full should succeed");
                            black_box(dataset.n_events_local());
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new(
                        "read_chunked_chunk_10",
                        format!("{format}_{precision}_small"),
                    ),
                    &path,
                    |b, input_path| {
                        let options = base_options.clone().chunk_size(10);
                        b.iter(|| {
                            let chunks = read_parquet_chunks_with_options(input_path, &options)
                                .expect("benchmark parquet chunked read should open");
                            let total_events = chunks
                                .map(|chunk| {
                                    chunk
                                        .expect("benchmark parquet chunk should read")
                                        .n_events_local()
                                })
                                .sum::<usize>();
                            black_box(total_events);
                        });
                    },
                );
            }
            "root" => {
                group.bench_with_input(
                    BenchmarkId::new("read_full", format!("{format}_{precision}_small")),
                    &path,
                    |b, input_path| {
                        b.iter(|| {
                            let dataset = read_root(input_path, &base_options)
                                .expect("benchmark root read_full should succeed");
                            black_box(dataset.n_events_local());
                        });
                    },
                );
            }
            _ => unreachable!("unsupported format in benchmark matrix"),
        }
    }

    group.finish();
}

fn open_data_benchmark(c: &mut Criterion) {
    c.bench_function("open benchmark", |b| {
        b.iter(|| {
            let p4_names = ["beam", "proton", "kshort1", "kshort2"];
            let aux_names = ["pol_magnitude", "pol_angle"];
            black_box(
                read_parquet(
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

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = io_read_matrix_benchmark, open_data_benchmark
);
criterion_main!(benches);
