use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use fastrand_contrib::RngExt;
use laddu::{
    amplitudes::{
        constant,
        kmatrix::{
            KopfKMatrixA0, KopfKMatrixA0Channel, KopfKMatrixA2, KopfKMatrixA2Channel,
            KopfKMatrixF0, KopfKMatrixF0Channel, KopfKMatrixF2, KopfKMatrixF2Channel,
        },
        parameter,
        zlm::Zlm,
    },
    data::DatasetReadOptions,
    extensions::NLL,
    io,
    traits::LikelihoodTerm,
    utils::{
        enums::{Frame, Sign},
        variables::Mass,
    },
};

use rayon::ThreadPoolBuilder;

fn reaction_variables() -> (laddu::Angles, laddu::Polarization, Mass) {
    let beam = laddu::Particle::measured("beam", "beam");
    let target = laddu::Particle::missing("target");
    let kshort1 = laddu::Particle::measured("K_S1", "kshort1");
    let kshort2 = laddu::Particle::measured("K_S2", "kshort2");
    let kk = laddu::Particle::composite("KK", [&kshort1, &kshort2]).unwrap();
    let proton = laddu::Particle::measured("proton", "proton");
    let reaction = laddu::Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap();
    let angles = reaction
        .decay(&kk)
        .unwrap()
        .angles(&kshort1, Frame::Helicity)
        .unwrap();
    let polarization = reaction.polarization("pol_magnitude", "pol_angle");
    let resonance_mass = reaction.mass(&kk);
    (angles, polarization, resonance_mass)
}

fn kmatrix_nll_benchmark(c: &mut Criterion) {
    let p4_names = ["beam", "proton", "kshort1", "kshort2"];
    let aux_names = ["pol_magnitude", "pol_angle"];
    let options = DatasetReadOptions::default()
        .p4_names(p4_names)
        .aux_names(aux_names);
    let ds_data = io::read_parquet("benches/bench.parquet", &options).unwrap();
    let ds_mc = io::read_parquet("benches/bench.parquet", &options).unwrap();

    let (angles, polarization, resonance_mass) = reaction_variables();
    let z00p = Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization).unwrap();
    let z00n = Zlm::new("Z00-", 0, 0, Sign::Negative, &angles, &polarization).unwrap();
    let z22p = Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization).unwrap();
    let f0p = KopfKMatrixF0::new(
        "f0+",
        [
            [constant("f0+ c00 re", 0.0), constant("f0+ c00 im", 0.0)],
            [parameter("f0(980)+ re"), constant("f0(980)+ im_fix", 0.0)],
            [parameter("f0(1370)+ re"), parameter("f0(1370)+ im")],
            [parameter("f0(1500)+ re"), parameter("f0(1500)+ im")],
            [parameter("f0(1710)+ re"), parameter("f0(1710)+ im")],
        ],
        KopfKMatrixF0Channel::PiPi,
        &resonance_mass,
        None,
    )
    .unwrap();
    let a0p = KopfKMatrixA0::new(
        "a0+",
        [
            [parameter("a0(980)+ re"), parameter("a0(980)+ im")],
            [parameter("a0(1450)+ re"), parameter("a0(1450)+ im")],
        ],
        KopfKMatrixA0Channel::PiEta,
        &resonance_mass,
        None,
    )
    .unwrap();
    let f0n = KopfKMatrixF0::new(
        "f0-",
        [
            [constant("f0- c00 re", 0.0), constant("f0- c00 im", 0.0)],
            [parameter("f0(980)- re"), constant("f0(980)- im_fix", 0.0)],
            [parameter("f0(1370)- re"), parameter("f0(1370)- im")],
            [parameter("f0(1500)- re"), parameter("f0(1500)- im")],
            [parameter("f0(1710)- re"), parameter("f0(1710)- im")],
        ],
        KopfKMatrixF0Channel::PiPi,
        &resonance_mass,
        None,
    )
    .unwrap();
    let a0n = KopfKMatrixA0::new(
        "a0-",
        [
            [parameter("a0(980)- re"), parameter("a0(980)- im")],
            [parameter("a0(1450)- re"), parameter("a0(1450)- im")],
        ],
        KopfKMatrixA0Channel::PiEta,
        &resonance_mass,
        None,
    )
    .unwrap();
    let f2 = KopfKMatrixF2::new(
        "f2",
        [
            [parameter("f2(1270) re"), parameter("f2(1270) im")],
            [parameter("f2(1525) re"), parameter("f2(1525) im")],
            [parameter("f2(1850) re"), parameter("f2(1850) im")],
            [parameter("f2(1910) re"), parameter("f2(1910) im")],
        ],
        KopfKMatrixF2Channel::KKbar,
        &resonance_mass,
        None,
    )
    .unwrap();
    let a2 = KopfKMatrixA2::new(
        "a2",
        [
            [parameter("a2(1320) re"), parameter("a2(1320) im")],
            [parameter("a2(1700) re"), parameter("a2(1700) im")],
        ],
        KopfKMatrixA2Channel::PiEtaPrime,
        &resonance_mass,
        None,
    )
    .unwrap();
    let s0p = f0p + a0p;
    let s0n = f0n + a0n;
    let d2p = f2 + a2;
    let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
    let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
    let neg_re = (&s0n * z00n.real()).norm_sqr();
    let neg_im = (&s0n * z00n.imag()).norm_sqr();
    let expr = pos_re + pos_im + neg_re + neg_im;
    let nll = NLL::new(&expr, &ds_data, &ds_mc, None).unwrap();
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
                        (0..nll.n_free())
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
    config = Criterion::default().sample_size(500);
    targets = kmatrix_nll_benchmark
}
criterion_main!(benches);
