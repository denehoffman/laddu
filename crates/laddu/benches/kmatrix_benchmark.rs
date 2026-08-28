//! Benchmarks K-matrix likelihood values and gradients.
#![allow(
    missing_docs,
    reason = "criterion generates an undocumented public function"
)]

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use laddu::{
    autodiff::AutodiffMode,
    compile::CompiledModel,
    complex,
    data::{
        data::{Dataset, EventBatch},
        io::parquet::ParquetSource,
    },
    event_scalar,
    expr::{Expr, cis},
    likelihood::{Likelihood, LikelihoodEvaluation, NllTerm},
    physics::{
        channel::Channel,
        math::spherical_harmonic,
        quantum::Reflectivity,
        vectors::{Vec3, Vec4},
    },
    runtime::{
        CpuOptions, Device, Execution, ExecutionOptions, JitPolicy, Precision, ThreadPolicy,
    },
};

#[path = "support/kmatrix.rs"]
mod kmatrix_fixture;

use kmatrix_fixture::fictitious_kmatrix_components;

#[cfg(feature = "wgpu")]
use laddu::runtime::{GpuBackend, GpuOptions};

#[derive(Copy, Clone)]
#[allow(
    clippy::enum_variant_names,
    reason = "CPU-only builds compile out the WGPU variants"
)]
enum BenchmarkBackend {
    CpuInterpreter {
        threads: usize,
        precision: Precision,
    },
    CpuJit {
        threads: usize,
        precision: Precision,
    },
    CpuReverse {
        threads: usize,
        precision: Precision,
        jit: JitPolicy,
    },
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "wgpu")]
    WgpuReverse,
}

impl BenchmarkBackend {
    fn name(self) -> String {
        match self {
            Self::CpuInterpreter { threads, precision } => {
                format!(
                    "cpu-interpreter-{}/{threads}-threads",
                    precision_name(precision)
                )
            }
            Self::CpuJit { threads, precision } => {
                format!("cpu-jit-{}/{threads}-threads", precision_name(precision))
            }
            Self::CpuReverse {
                threads,
                precision,
                jit,
            } => format!(
                "cpu-reverse-{}-{}/{threads}-threads",
                precision_name(precision),
                match jit {
                    JitPolicy::Disabled => "interpreter",
                    JitPolicy::Enabled => "jit",
                    JitPolicy::Auto => "auto",
                }
            ),
            #[cfg(feature = "wgpu")]
            Self::Wgpu => "wgpu-f32".to_string(),
            #[cfg(feature = "wgpu")]
            Self::WgpuReverse => "wgpu-reverse-f32".to_string(),
        }
    }

    fn precision(self) -> Precision {
        match self {
            Self::CpuInterpreter { precision, .. } | Self::CpuJit { precision, .. } => precision,
            Self::CpuReverse { precision, .. } => precision,
            #[cfg(feature = "wgpu")]
            Self::Wgpu | Self::WgpuReverse => Precision::F32,
        }
    }

    fn execution(self) -> Execution {
        let (device, precision) = match self {
            Self::CpuInterpreter { threads, precision } => (
                Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(threads),
                    jit: JitPolicy::Disabled,
                }),
                precision,
            ),
            Self::CpuJit { threads, precision } => (
                Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(threads),
                    jit: JitPolicy::Enabled,
                }),
                precision,
            ),
            Self::CpuReverse {
                threads,
                precision,
                jit,
            } => (
                Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(threads),
                    jit,
                }),
                precision,
            ),
            #[cfg(feature = "wgpu")]
            Self::Wgpu | Self::WgpuReverse => (
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
            autodiff: match self {
                Self::CpuReverse { .. } => AutodiffMode::Reverse,
                #[cfg(feature = "wgpu")]
                Self::WgpuReverse => AutodiffMode::Reverse,
                _ => AutodiffMode::Forward,
            },
            ..ExecutionOptions::default()
        })
        .unwrap()
    }
}

fn benchmark_backends() -> Vec<BenchmarkBackend> {
    let maximum = num_cpus::get().max(1);
    let mut backends = cpu_backends(1).to_vec();
    if maximum != 1 {
        backends.extend(cpu_backends(maximum));
    }
    #[cfg(feature = "wgpu")]
    backends.push(BenchmarkBackend::Wgpu);
    backends
}

fn benchmark_gradient_backends() -> Vec<BenchmarkBackend> {
    let maximum = num_cpus::get().max(1);
    let mut backends = benchmark_backends();
    #[cfg(feature = "wgpu")]
    backends.push(BenchmarkBackend::WgpuReverse);
    backends.extend(reverse_backends(1));
    if maximum != 1 {
        backends.extend(reverse_backends(maximum));
    }
    backends
}

fn reverse_backends(threads: usize) -> [BenchmarkBackend; 4] {
    [
        BenchmarkBackend::CpuReverse {
            threads,
            precision: Precision::F64,
            jit: JitPolicy::Disabled,
        },
        BenchmarkBackend::CpuReverse {
            threads,
            precision: Precision::F64,
            jit: JitPolicy::Enabled,
        },
        BenchmarkBackend::CpuReverse {
            threads,
            precision: Precision::F32,
            jit: JitPolicy::Disabled,
        },
        BenchmarkBackend::CpuReverse {
            threads,
            precision: Precision::F32,
            jit: JitPolicy::Enabled,
        },
    ]
}

fn cpu_backends(threads: usize) -> [BenchmarkBackend; 4] {
    [
        BenchmarkBackend::CpuInterpreter {
            threads,
            precision: Precision::F64,
        },
        BenchmarkBackend::CpuJit {
            threads,
            precision: Precision::F64,
        },
        BenchmarkBackend::CpuInterpreter {
            threads,
            precision: Precision::F32,
        },
        BenchmarkBackend::CpuJit {
            threads,
            precision: Precision::F32,
        },
    ]
}

fn likelihood(term: &NllTerm, backend: BenchmarkBackend) -> Likelihood {
    Likelihood::with_execution([term.clone()], &backend.execution()).unwrap()
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

    // This is a fictitious workload baseline; only its matrix dimensions and
    // production parameter count are retained for benchmark continuity.
    let (s0p, s0n, d2p) = fictitious_kmatrix_components(&resonance_s);

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

/// Retains the benchmark IDs used through v0.19.6 so CodSpeed can compare the
/// kernel rewrite against the previous evaluator. The f64 CPU interpreter is
/// the closest current equivalent to that evaluator.
fn historical_kmatrix_nll_benchmark(c: &mut Criterion) {
    let term = kmatrix_term(1);
    let mut group = c.benchmark_group("K-Matrix NLL Performance");
    let thread_counts = (0..)
        .map(|power| 1 << power)
        .take_while(|threads| *threads <= num_cpus::get());

    for threads in thread_counts {
        let backend = BenchmarkBackend::CpuInterpreter {
            threads,
            precision: Precision::F64,
        };
        let likelihood = likelihood(&term, backend);
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, _threads| {
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

fn kmatrix_nll_gradient_benchmark(c: &mut Criterion) {
    for batches in [1, 2] {
        let term = kmatrix_term(batches);
        let backends = benchmark_gradient_backends();
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

fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::F64 => "f64",
        Precision::F32 => "f32",
        Precision::Auto => "auto",
    }
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

fn validate_nll_parity(likelihoods: &[(BenchmarkBackend, Likelihood)]) {
    let params = likelihoods[0].1.params_with(|_| 0.25);
    let values = likelihoods
        .iter()
        .map(|(backend, likelihood)| (*backend, likelihood.nll(&params)))
        .collect::<Vec<_>>();

    let f64_reference = values
        .iter()
        .find(|(backend, result)| backend.precision() == Precision::F64 && result.is_ok())
        .map(|(backend, result)| (*backend, *result.as_ref().unwrap()));
    let f32_reference = values
        .iter()
        .find(|(backend, result)| backend.precision() == Precision::F32 && result.is_ok())
        .map(|(backend, result)| (*backend, *result.as_ref().unwrap()));

    for (backend, result) in &values {
        let actual = match result {
            Ok(value) => *value,
            Err(error) => {
                eprintln!("{} NLL unavailable: {error}", backend.name());
                continue;
            }
        };
        match backend.precision() {
            Precision::F64 => {
                let (reference_backend, expected) = f64_reference.unwrap();
                let difference = relative_error(actual, expected);
                if difference > 1.0e-10 {
                    eprintln!(
                        "warning: {} NLL differs from {} by {:.3}% ({actual} versus {expected})",
                        backend.name(),
                        reference_backend.name(),
                        difference * 100.0
                    );
                }
            }
            Precision::F32 => {
                let (reference_backend, expected) = f32_reference.unwrap();
                if !close_f32(actual, expected) {
                    eprintln!(
                        "warning: {} NLL differs from {} by {:.3}% ({actual} versus {expected})",
                        backend.name(),
                        reference_backend.name(),
                        relative_error(actual, expected) * 100.0
                    );
                }
                if let Some((f64_backend, f64_expected)) = f64_reference {
                    eprintln!(
                        "{} NLL differs from {} by {:.3}% ({actual} versus {f64_expected})",
                        backend.name(),
                        f64_backend.name(),
                        relative_error(actual, f64_expected) * 100.0
                    );
                }
            }
            Precision::Auto => unreachable!("benchmark backends choose an explicit precision"),
        }
    }

    for (_, result) in values {
        result.unwrap();
    }
}

fn worst_gradient_difference(actual: &[f64], expected: &[f64]) -> (usize, f64, f64, f64) {
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| {
            (
                index,
                *actual,
                *expected,
                relative_error(*actual, *expected),
            )
        })
        .max_by(|left, right| left.3.total_cmp(&right.3))
        .unwrap_or((0, 0.0, 0.0, 0.0))
}

fn validate_gradient_against_reference(
    backend: BenchmarkBackend,
    actual: &LikelihoodEvaluation,
    reference_backend: BenchmarkBackend,
    expected: &LikelihoodEvaluation,
    tolerance: f64,
) {
    let value_difference = relative_error(actual.value(), expected.value());
    if value_difference > tolerance {
        eprintln!(
            "warning: {} NLL differs from {} by {:.3}% ({} versus {})",
            backend.name(),
            reference_backend.name(),
            value_difference * 100.0,
            actual.value(),
            expected.value()
        );
    }
    if actual.gradient().len() != expected.gradient().len() {
        eprintln!(
            "warning: {} has {} gradient components but {} has {}",
            backend.name(),
            actual.gradient().len(),
            reference_backend.name(),
            expected.gradient().len()
        );
        return;
    }
    let worst = worst_gradient_difference(actual.gradient(), expected.gradient());
    if worst.3 > tolerance {
        eprintln!(
            "warning: {} gradient[{}] differs from {} by {:.3}% ({} versus {})",
            backend.name(),
            worst.0,
            reference_backend.name(),
            worst.3 * 100.0,
            worst.1,
            worst.2
        );
    }
}

fn validate_gradient_parity(likelihoods: &[(BenchmarkBackend, Likelihood)]) {
    let params = likelihoods[0].1.params_with(|_| 0.25);
    let values = likelihoods
        .iter()
        .map(|(backend, likelihood)| (*backend, likelihood.nll_with_gradient(&params)))
        .collect::<Vec<_>>();

    let f64_reference = values
        .iter()
        .find(|(backend, result)| backend.precision() == Precision::F64 && result.is_ok())
        .map(|(backend, result)| (*backend, result.as_ref().unwrap()));
    let f32_reference = values
        .iter()
        .find(|(backend, result)| backend.precision() == Precision::F32 && result.is_ok())
        .map(|(backend, result)| (*backend, result.as_ref().unwrap()));

    for (backend, result) in &values {
        let actual = match result {
            Ok(value) => value,
            Err(error) => {
                eprintln!("{} gradient unavailable: {error}", backend.name());
                continue;
            }
        };
        match backend.precision() {
            Precision::F64 => {
                let (reference_backend, expected) = f64_reference.unwrap();
                validate_gradient_against_reference(
                    *backend,
                    actual,
                    reference_backend,
                    expected,
                    1.0e-10,
                );
            }
            Precision::F32 => {
                let (reference_backend, expected) = f32_reference.unwrap();
                validate_gradient_against_reference(
                    *backend,
                    actual,
                    reference_backend,
                    expected,
                    5.0e-3,
                );
                if let Some((f64_backend, f64_expected)) = f64_reference {
                    validate_gradient_against_reference(
                        *backend,
                        actual,
                        f64_backend,
                        f64_expected,
                        5.0e-3,
                    );
                }
            }
            Precision::Auto => unreachable!("benchmark backends choose an explicit precision"),
        }
    }

    for (_, result) in values {
        result.unwrap();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = historical_kmatrix_nll_benchmark, kmatrix_nll_benchmark, kmatrix_nll_gradient_benchmark
}
criterion_main!(benches);
