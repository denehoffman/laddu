use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use laddu::{
    amplitudes::{
        KopfA0Channel, KopfA2Channel, KopfF0Channel, KopfF2Channel, kopf_a0, kopf_a2, kopf_f0,
        kopf_f2,
    },
    compile::CompiledModel,
    complex,
    data::{data::Dataset, io::parquet::ParquetSource},
    event_scalar,
    expr::{Expr, cis},
    likelihood::{CpuLikelihood, CpuLikelihoodTerm, CpuNllTerm},
    parameter,
    physics::{
        channel::Channel,
        math::spherical_harmonic,
        quantum::Reflectivity,
        vectors::{Vec3, Vec4},
    },
};
use rayon::ThreadPoolBuilder;

fn reaction_variables() -> (Expr, Expr, Expr, Expr) {
    let mut channel = Channel::new("gamma p -> Ks Ks p");
    channel.edge("beam").p4(Vec4::event("beam"));
    channel.edge("target");
    channel.edge("kk");
    channel.edge("proton").p4(Vec4::event("proton"));
    channel.edge("kshort1").p4(Vec4::event("kshort1"));
    channel.edge("kshort2").p4(Vec4::event("kshort2"));
    channel
        .vertex("production")
        .incoming(["beam", "target"])
        .outgoing(["kk", "proton"])
        .validate()
        .unwrap();
    channel
        .vertex("kk_decay")
        .incoming(["kk"])
        .outgoing(["kshort1", "kshort2"])
        .validate()
        .unwrap();

    let production = channel.get_vertex("production").unwrap();
    let y_hint = -production
        .vec3("beam")
        .unwrap()
        .cross(&production.vec3("proton").unwrap());
    let decay = channel.get_vertex("kk_decay").unwrap();
    let z_axis = -decay.vec3("proton").unwrap();
    let costheta = decay
        .costheta("kshort1", z_axis.clone(), y_hint.clone())
        .unwrap();
    let phi = decay.phi("kshort1", z_axis, y_hint).unwrap();
    let resonance_s = channel.s("kk").unwrap();

    let lab_angle = event_scalar("pol_angle");
    let lab_polarization = Vec3::new(lab_angle.cos(), lab_angle.sin(), 0.0);
    let reference = channel.vec3("beam").unwrap();
    let spectator = channel.vec3("proton").unwrap();
    let production_normal = reference.cross(&(-spectator)).unit();
    let polarization_angle = laddu::atan2(
        production_normal.dot(&lab_polarization),
        reference
            .unit()
            .dot(&lab_polarization.cross(&production_normal)),
    );
    (costheta, phi, resonance_s, polarization_angle)
}

fn zlm(
    l: usize,
    m: isize,
    reflectivity: Reflectivity,
    costheta: &Expr,
    phi: &Expr,
    polarization: &Expr,
    polarization_angle: &Expr,
) -> Expr {
    let ylm = spherical_harmonic(l, m, costheta, phi).unwrap();
    let rotated = ylm * cis(-polarization_angle);
    match reflectivity {
        Reflectivity::Positive => complex(
            (1.0 + polarization).sqrt() * rotated.real(),
            (1.0 - polarization).sqrt() * rotated.imag(),
        ),
        Reflectivity::Negative => complex(
            (1.0 - polarization).sqrt() * rotated.real(),
            (1.0 + polarization).sqrt() * rotated.imag(),
        ),
    }
}

fn kmatrix_likelihood() -> CpuLikelihood {
    const DATASET: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/bench.parquet");
    let dataset = Dataset::new(ParquetSource::open(DATASET).unwrap());
    let (costheta, phi, resonance_s, polarization_angle) = reaction_variables();
    let polarization = event_scalar("pol_magnitude");
    let z00p = zlm(
        0,
        0,
        Reflectivity::Positive,
        &costheta,
        &phi,
        &polarization,
        &polarization_angle,
    );
    let z00n = zlm(
        0,
        0,
        Reflectivity::Negative,
        &costheta,
        &phi,
        &polarization,
        &polarization_angle,
    );
    let z22p = zlm(
        2,
        2,
        Reflectivity::Positive,
        &costheta,
        &phi,
        &polarization,
        &polarization_angle,
    );

    let f0p = kopf_f0(
        &resonance_s,
        [
            complex(parameter!("f0+ c00 re", 0.0), parameter!("f0+ c00 im", 0.0)),
            complex(
                parameter!("f0(980)+ re"),
                parameter!("f0(980)+ im_fix", 0.0),
            ),
            complex(parameter!("f0(1370)+ re"), parameter!("f0(1370)+ im")),
            complex(parameter!("f0(1500)+ re"), parameter!("f0(1500)+ im")),
            complex(parameter!("f0(1710)+ re"), parameter!("f0(1710)+ im")),
        ],
    )
    .unwrap()
    .component(KopfF0Channel::KKbar);
    let a0p = kopf_a0(
        &resonance_s,
        [
            complex(parameter!("a0(980)+ re"), parameter!("a0(980)+ im")),
            complex(parameter!("a0(1450)+ re"), parameter!("a0(1450)+ im")),
        ],
    )
    .unwrap()
    .component(KopfA0Channel::KKbar);
    let f0n = kopf_f0(
        &resonance_s,
        [
            complex(parameter!("f0- c00 re", 0.0), parameter!("f0- c00 im", 0.0)),
            complex(
                parameter!("f0(980)- re"),
                parameter!("f0(980)- im_fix", 0.0),
            ),
            complex(parameter!("f0(1370)- re"), parameter!("f0(1370)- im")),
            complex(parameter!("f0(1500)- re"), parameter!("f0(1500)- im")),
            complex(parameter!("f0(1710)- re"), parameter!("f0(1710)- im")),
        ],
    )
    .unwrap()
    .component(KopfF0Channel::KKbar);
    let a0n = kopf_a0(
        &resonance_s,
        [
            complex(parameter!("a0(980)- re"), parameter!("a0(980)- im")),
            complex(parameter!("a0(1450)- re"), parameter!("a0(1450)- im")),
        ],
    )
    .unwrap()
    .component(KopfA0Channel::KKbar);
    let f2 = kopf_f2(
        &resonance_s,
        [
            complex(parameter!("f2(1270) re"), parameter!("f2(1270) im")),
            complex(parameter!("f2(1525) re"), parameter!("f2(1525) im")),
            complex(parameter!("f2(1850) re"), parameter!("f2(1850) im")),
            complex(parameter!("f2(1910) re"), parameter!("f2(1910) im")),
        ],
    )
    .unwrap()
    .component(KopfF2Channel::KKbar);
    let a2 = kopf_a2(
        &resonance_s,
        [
            complex(parameter!("a2(1320) re"), parameter!("a2(1320) im")),
            complex(parameter!("a2(1700) re"), parameter!("a2(1700) im")),
        ],
    )
    .unwrap()
    .component(KopfA2Channel::KKbar);

    let s0p = f0p + a0p;
    let s0n = f0n + a0n;
    let d2p = f2 + a2;
    let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
    let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
    let neg_re = (&s0n * z00n.real()).norm_sqr();
    let neg_im = (&s0n * z00n.imag()).norm_sqr();
    let model = CompiledModel::from_expr(&(pos_re + pos_im + neg_re + neg_im)).unwrap();
    let nll = CpuNllTerm::new("K-Matrix", &model, &dataset, &dataset).unwrap();
    CpuLikelihood::new([nll.boxed()]).unwrap()
}

fn kmatrix_nll_benchmark(c: &mut Criterion) {
    let likelihood = kmatrix_likelihood();
    let mut group = c.benchmark_group("K-Matrix NLL Performance");
    let n_threads = (0..)
        .map(|power| 1 << power)
        .take_while(|threads| *threads <= num_cpus::get());

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
                        let mut params = likelihood.default_params();
                        for id in likelihood.params().free_params() {
                            params.set_full(*id, rng.f64() * 200.0 - 100.0).unwrap();
                        }
                        params
                    },
                    |params| {
                        pool.install(|| black_box(likelihood.nll(black_box(&params)).unwrap()))
                    },
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
