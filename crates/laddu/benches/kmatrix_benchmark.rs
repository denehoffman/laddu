use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use laddu::{
    amplitudes::{
        KopfA0Channel, KopfA2Channel, KopfF0Channel, KopfF2Channel, kopf_a0, kopf_a2, kopf_f0,
        kopf_f2,
    },
    compile::CompiledModel,
    complex,
    data::{
        data::{Dataset, EventBatch},
        io::parquet::ParquetSource,
    },
    event_scalar,
    expr::{Expr, cis},
    likelihood::{Likelihood, LikelihoodTerm, NllTerm},
    parameter,
    physics::{
        channel::Channel,
        math::spherical_harmonic,
        quantum::Reflectivity,
        vectors::{Vec3, Vec4},
    },
    runtime::{
        CpuOptions, Device, Execution, ExecutionOptions, GpuBackend, GpuOptions, JitPolicy,
        Precision, ThreadPolicy,
    },
};

#[derive(Copy, Clone)]
enum BenchmarkBackend {
    CpuInterpreter(usize),
    CpuJit(usize),
    Wgpu,
}

impl BenchmarkBackend {
    fn name(self) -> String {
        match self {
            Self::CpuInterpreter(threads) => format!("cpu-interpreter/{threads}-threads"),
            Self::CpuJit(threads) => format!("cpu-jit/{threads}-threads"),
            Self::Wgpu => "wgpu-f32".to_string(),
        }
    }

    fn execution(self) -> Execution {
        let (device, precision) = match self {
            Self::CpuInterpreter(threads) => (
                Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(threads),
                    jit: JitPolicy::Disabled,
                }),
                Precision::F64,
            ),
            Self::CpuJit(threads) => (
                Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(threads),
                    jit: JitPolicy::Enabled,
                }),
                Precision::F64,
            ),
            Self::Wgpu => (
                Device::Gpu(GpuOptions {
                    backend: GpuBackend::Wgpu,
                    ..GpuOptions::default()
                }),
                Precision::F32,
            ),
        };
        Execution::local(ExecutionOptions {
            device,
            precision,
            ..ExecutionOptions::default()
        })
        .unwrap()
    }

    fn is_wgpu(self) -> bool {
        matches!(self, Self::Wgpu)
    }
}

fn benchmark_backends() -> Vec<BenchmarkBackend> {
    let maximum = num_cpus::get().max(1);
    let mut backends = vec![
        BenchmarkBackend::CpuInterpreter(1),
        BenchmarkBackend::CpuJit(1),
    ];
    if maximum != 1 {
        backends.extend([
            BenchmarkBackend::CpuInterpreter(maximum),
            BenchmarkBackend::CpuJit(maximum),
        ]);
    }
    backends.push(BenchmarkBackend::Wgpu);
    backends
}

fn likelihood(term: &NllTerm, backend: BenchmarkBackend) -> Likelihood {
    Likelihood::with_execution([term.clone().boxed()], backend.execution()).unwrap()
}

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

fn benchmark_dataset(batches: usize) -> Dataset {
    const DATASET: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/bench.parquet");
    let dataset = Dataset::new(ParquetSource::open(DATASET).unwrap());
    let materialized = dataset
        .batches()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let batch = EventBatch::concat(&materialized).unwrap();

    match batches {
        1 => Dataset::from_batch(batch),
        2 => {
            let midpoint = batch.len() / 2;
            let largest = midpoint.max(batch.len() - midpoint);
            Dataset::from_batches(vec![
                batch.slice(0, midpoint),
                batch.slice(midpoint, batch.len()),
            ])
            .unwrap()
            .chunked(largest)
            .unwrap()
        }
        _ => unreachable!("benchmark only defines one- and two-batch layouts"),
    }
}

fn kmatrix_term(batches: usize) -> NllTerm {
    let dataset = benchmark_dataset(batches);
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
    NllTerm::new("K-Matrix", &model, &dataset, &dataset).unwrap()
}

fn kmatrix_nll_benchmark(c: &mut Criterion) {
    for batches in [1, 2] {
        let term = kmatrix_term(batches);
        let backends = benchmark_backends();
        let likelihoods = backends
            .iter()
            .copied()
            .map(|backend| (backend, likelihood(&term, backend)))
            .collect::<Vec<_>>();
        validate_nll_parity(&likelihoods);
        let mut group = c.benchmark_group(format!("K-Matrix NLL/{batches}-batches"));

        for (backend, likelihood) in likelihoods {
            assert_eq!(
                likelihood.terms()[0]
                    .as_intensity()
                    .unwrap()
                    .data()
                    .unwrap()
                    .stats()
                    .local_batches(),
                batches
            );
            group.bench_with_input(
                BenchmarkId::from_parameter(backend.name()),
                &backend,
                |b, _backend| {
                    let mut rng = fastrand::Rng::with_seed(0x4b_4d_41_54_52_49_58);
                    b.iter_batched(
                        || likelihood.params_with(|_| rng.f64() * 200.0 - 100.0),
                        |params| black_box(likelihood.nll(black_box(&params)).unwrap()),
                        BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn kmatrix_nll_gradient_benchmark(c: &mut Criterion) {
    for batches in [1, 2] {
        let term = kmatrix_term(batches);
        let backends = benchmark_backends();
        let likelihoods = backends
            .iter()
            .copied()
            .map(|backend| (backend, likelihood(&term, backend)))
            .collect::<Vec<_>>();
        validate_gradient_parity(&likelihoods);
        let mut group = c.benchmark_group(format!("K-Matrix NLL Gradient/{batches}-batches"));
        group.sample_size(10);

        for (backend, likelihood) in likelihoods {
            group.bench_with_input(
                BenchmarkId::from_parameter(backend.name()),
                &backend,
                |b, _backend| {
                    let mut rng = fastrand::Rng::with_seed(0x4b_4d_41_54_52_49_58);
                    b.iter_batched(
                        || likelihood.params_with(|_| rng.f64() * 200.0 - 100.0),
                        |params| {
                            black_box(likelihood.nll_with_gradient(black_box(&params)).unwrap())
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn close_f32(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 2.0e-3 * expected.abs().max(1.0)
}

fn validate_nll_parity(likelihoods: &[(BenchmarkBackend, Likelihood)]) {
    let params = likelihoods[0].1.params_with(|_| 0.25);
    let expected = likelihoods[0].1.nll(&params).unwrap();
    for (backend, likelihood) in &likelihoods[1..] {
        let actual = likelihood.nll(&params).unwrap();
        assert!(
            close_f32(actual, expected),
            "{} NLL {actual} differs from CPU interpreter {expected}",
            backend.name()
        );
    }
}

fn validate_gradient_parity(likelihoods: &[(BenchmarkBackend, Likelihood)]) {
    let params = likelihoods[0].1.params_with(|_| 0.25);
    let expected = likelihoods[0].1.nll_with_gradient(&params).unwrap();
    for (backend, likelihood) in &likelihoods[1..] {
        let actual = likelihood.nll_with_gradient(&params).unwrap();
        assert!(
            close_f32(actual.value(), expected.value()),
            "{} NLL {} differs from CPU interpreter {}",
            backend.name(),
            actual.value(),
            expected.value()
        );
        assert_eq!(actual.gradient().len(), expected.gradient().len());
        let mut worst = (0_usize, 0.0_f64, 0.0_f64, 0.0_f64);
        for (index, (actual, expected)) in actual
            .gradient()
            .iter()
            .zip(expected.gradient())
            .enumerate()
        {
            let relative_error = (*actual - *expected).abs() / expected.abs().max(1.0);
            if relative_error > worst.3 {
                worst = (index, *actual, *expected, relative_error);
            }
        }
        if backend.is_wgpu() {
            eprintln!(
                "{} worst gradient discrepancy: gradient[{}] {} versus CPU interpreter {} ({:.3}%)",
                backend.name(),
                worst.0,
                worst.1,
                worst.2,
                worst.3 * 100.0
            );
            assert!(actual.gradient().iter().all(|value| value.is_finite()));
        } else {
            assert!(
                worst.3 <= 1.0e-10,
                "{} gradient[{}] {} differs from CPU interpreter {} by {:.3}%",
                backend.name(),
                worst.0,
                worst.1,
                worst.2,
                worst.3 * 100.0
            );
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = kmatrix_nll_benchmark, kmatrix_nll_gradient_benchmark
}
criterion_main!(benches);
