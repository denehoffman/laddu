//! Fast, deterministic baselines for repeated differential cross-section projections.
#![allow(missing_docs, reason = "criterion benchmark items are not public API")]

mod support {
    pub mod projection;
}

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use laddu::prelude::ThreadPolicy;
use support::projection::{ProjectionFixture, Storage};

const FAST_EVENTS: usize = 256;

fn projection_benchmark(criterion: &mut Criterion) {
    let resident_20 =
        ProjectionFixture::new(FAST_EVENTS, 20, Storage::Resident, ThreadPolicy::Serial).unwrap();
    let streaming_20 =
        ProjectionFixture::new(FAST_EVENTS, 20, Storage::Streaming, ThreadPolicy::Serial).unwrap();
    let resident_200 =
        ProjectionFixture::new(FAST_EVENTS, 200, Storage::Resident, ThreadPolicy::Serial).unwrap();

    let mut group = criterion.benchmark_group("projection baseline");
    group.sample_size(10);
    for projections in [1, 4] {
        group.bench_with_input(
            BenchmarkId::new("single/resident/20-draws/aliases", projections),
            &projections,
            |bencher, &projections| {
                bencher.iter(|| {
                    black_box(resident_20.evaluate_single(black_box(projections)).unwrap())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("single-set/resident/20-draws/aliases", projections),
            &projections,
            |bencher, &projections| {
                bencher.iter(|| {
                    black_box(
                        resident_20
                            .evaluate_single_set(black_box(projections))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("combined/resident/20-draws/aliases", projections),
            &projections,
            |bencher, &projections| {
                bencher.iter(|| {
                    black_box(
                        resident_20
                            .evaluate_combined(black_box(projections))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("combined-set/resident/20-draws/aliases", projections),
            &projections,
            |bencher, &projections| {
                bencher.iter(|| {
                    black_box(
                        resident_20
                            .evaluate_combined_set(black_box(projections))
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.bench_function("combined/resident/20-draws/unique/4", |bencher| {
        bencher.iter(|| black_box(resident_20.evaluate_combined_unique(4).unwrap()));
    });
    group.bench_function("combined/streaming/20-draws/aliases/4", |bencher| {
        bencher.iter(|| black_box(streaming_20.evaluate_combined(4).unwrap()));
    });
    group.bench_function("combined-set/streaming/20-draws/aliases/4", |bencher| {
        bencher.iter(|| black_box(streaming_20.evaluate_combined_set(4).unwrap()));
    });
    group.bench_function("combined/resident/200-draws/aliases/4", |bencher| {
        bencher.iter(|| black_box(resident_200.evaluate_combined(4).unwrap()));
    });
    group.bench_function("combined-set/resident/200-draws/aliases/4", |bencher| {
        bencher.iter(|| black_box(resident_200.evaluate_combined_set(4).unwrap()));
    });
    group.finish();
}

criterion_group!(benches, projection_benchmark);
criterion_main!(benches);
