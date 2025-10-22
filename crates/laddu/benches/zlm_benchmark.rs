use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use fastrand_contrib::RngExt;
use laddu::{
    amplitudes::{parameter, zlm::Zlm, Manager},
    data::open,
    utils::{
        enums::{Frame, Sign},
        variables::{Angles, Polarization},
    },
    ComplexScalar, Float, Scalar,
};

use rayon::ThreadPoolBuilder;

fn zlm_benchmark(c: &mut Criterion) {
    let ds_data = open("benches/bench.parquet").unwrap();

    let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
    let polarization = Polarization::new(0, [1], 0);
    let mut manager = Manager::default();
    let z00p = manager
        .register(Zlm::new(
            "Z00+",
            0,
            0,
            Sign::Positive,
            &angles,
            &polarization,
        ))
        .unwrap();
    let z00n = manager
        .register(Zlm::new(
            "Z00-",
            0,
            0,
            Sign::Negative,
            &angles,
            &polarization,
        ))
        .unwrap();
    let z22p = manager
        .register(Zlm::new(
            "Z22+",
            2,
            2,
            Sign::Positive,
            &angles,
            &polarization,
        ))
        .unwrap();
    let s0p = manager
        .register(Scalar::new("c00+", parameter("c00+ re")))
        .unwrap();
    let s0n = manager
        .register(Scalar::new("c00-", parameter("c00- re")))
        .unwrap();
    let d2p = manager
        .register(ComplexScalar::new(
            "c22+",
            parameter("c22+ re"),
            parameter("c22+ im"),
        ))
        .unwrap();
    let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
    let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
    let neg_re = (&s0n * z00n.real()).norm_sqr();
    let neg_im = (&s0n * z00n.imag()).norm_sqr();
    let expr = pos_re + pos_im + neg_re + neg_im;
    let model = manager.model(&expr);
    let evaluator = model.load(&ds_data);
    let mut group = c.benchmark_group("Zlm Eval Performance");
    let n_threads: Vec<usize> = (0..)
        .map(|x| 1 << x)
        .take_while(|&p| p <= num_cpus::get())
        .collect();
    for threads in n_threads {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &_threads| {
                let mut rng = fastrand::Rng::new();
                b.iter_batched(
                    || {
                        #[cfg(feature = "f32")]
                        let p: Vec<Float> = (0..evaluator.parameters().len())
                            .map(|_| rng.f32_range(-100.0..100.0))
                            .collect();
                        #[cfg(not(feature = "f32"))]
                        let p: Vec<Float> = (0..evaluator.parameters().len())
                            .map(|_| rng.f64_range(-100.0..100.0))
                            .collect();
                        p
                    },
                    |p| pool.install(|| black_box(evaluator.evaluate(&p))),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30)).sample_size(5000);
    targets = zlm_benchmark
}
criterion_main!(benches);
