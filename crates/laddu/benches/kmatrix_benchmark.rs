use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use fastrand_contrib::RngExt;
use laddu::{
    amplitudes::{
        constant,
        kmatrix::{KopfKMatrixA0, KopfKMatrixA2, KopfKMatrixF0, KopfKMatrixF2},
        parameter,
        zlm::Zlm,
        Manager,
    },
    data::Dataset,
    extensions::NLL,
    traits::*,
    utils::{
        enums::{Frame, Sign},
        variables::{Angles, Mass, Polarization},
    },
};

use rayon::ThreadPoolBuilder;

fn kmatrix_nll_benchmark(c: &mut Criterion) {
    let p4_names = ["beam", "proton", "kshort1", "kshort2"];
    let aux_names = ["pol_magnitude", "pol_angle"];
    let ds_data = Dataset::open("benches/bench.parquet", &p4_names, &aux_names).unwrap();
    let ds_mc = Dataset::open("benches/bench.parquet", &p4_names, &aux_names).unwrap();

    let angles = Angles::new(
        "beam",
        ["proton"],
        ["kshort1"],
        ["kshort1", "kshort2"],
        Frame::Helicity,
    );
    let polarization = Polarization::new("beam", ["proton"], "pol_magnitude", "pol_angle");
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
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
    let f0p = manager
        .register(KopfKMatrixF0::new(
            "f0+",
            [
                [constant(0.0), constant(0.0)],
                [parameter("f0(980)+ re"), constant(0.0)],
                [parameter("f0(1370)+ re"), parameter("f0(1370)+ im")],
                [parameter("f0(1500)+ re"), parameter("f0(1500)+ im")],
                [parameter("f0(1710)+ re"), parameter("f0(1710)+ im")],
            ],
            0,
            &resonance_mass,
            None,
        ))
        .unwrap();
    let a0p = manager
        .register(KopfKMatrixA0::new(
            "a0+",
            [
                [parameter("a0(980)+ re"), parameter("a0(980)+ im")],
                [parameter("a0(1450)+ re"), parameter("a0(1450)+ im")],
            ],
            0,
            &resonance_mass,
            None,
        ))
        .unwrap();
    let f0n = manager
        .register(KopfKMatrixF0::new(
            "f0-",
            [
                [constant(0.0), constant(0.0)],
                [parameter("f0(980)- re"), constant(0.0)],
                [parameter("f0(1370)- re"), parameter("f0(1370)- im")],
                [parameter("f0(1500)- re"), parameter("f0(1500)- im")],
                [parameter("f0(1710)- re"), parameter("f0(1710)- im")],
            ],
            0,
            &resonance_mass,
            None,
        ))
        .unwrap();
    let a0n = manager
        .register(KopfKMatrixA0::new(
            "a0-",
            [
                [parameter("a0(980)- re"), parameter("a0(980)- im")],
                [parameter("a0(1450)- re"), parameter("a0(1450)- im")],
            ],
            0,
            &resonance_mass,
            None,
        ))
        .unwrap();
    let f2 = manager
        .register(KopfKMatrixF2::new(
            "f2",
            [
                [parameter("f2(1270) re"), parameter("f2(1270) im")],
                [parameter("f2(1525) re"), parameter("f2(1525) im")],
                [parameter("f2(1850) re"), parameter("f2(1850) im")],
                [parameter("f2(1910) re"), parameter("f2(1910) im")],
            ],
            2,
            &resonance_mass,
            None,
        ))
        .unwrap();
    let a2 = manager
        .register(KopfKMatrixA2::new(
            "a2",
            [
                [parameter("a2(1320) re"), parameter("a2(1320) im")],
                [parameter("a2(1700) re"), parameter("a2(1700) im")],
            ],
            2,
            &resonance_mass,
            None,
        ))
        .unwrap();
    let s0p = f0p + a0p;
    let s0n = f0n + a0n;
    let d2p = f2 + a2;
    let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
    let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
    let neg_re = (&s0n * z00n.real()).norm_sqr();
    let neg_im = (&s0n * z00n.imag()).norm_sqr();
    let expr = pos_re + pos_im + neg_re + neg_im;
    let model = manager.model(&expr);
    let nll = NLL::new(&model, &ds_data, &ds_mc);
    let mut group = c.benchmark_group("K-Matrix NLL Performance");
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
                        (0..nll.parameters().len())
                            .map(|_| rng.f64_range(-100.0..100.0))
                            .collect::<Vec<f64>>()
                    },
                    |p| pool.install(|| black_box(nll.evaluate(&p))),
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
    targets = kmatrix_nll_benchmark
}
criterion_main!(benches);
