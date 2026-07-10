#![cfg(feature = "wgpu")]

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
    matrix, parameter,
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
    solve, vector,
};

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

fn benchmark_dataset() -> Dataset {
    const DATASET: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/bench.parquet");
    let dataset = Dataset::new(ParquetSource::open(DATASET).unwrap());
    let materialized = dataset
        .batches()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    Dataset::from_batch(EventBatch::concat(&materialized).unwrap())
}

fn kmatrix_term() -> NllTerm {
    let dataset = benchmark_dataset();
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

fn likelihood(term: &NllTerm, execution: Execution) -> Likelihood {
    Likelihood::with_execution([term.clone().boxed()], execution).unwrap()
}

fn cpu_f32_execution() -> Execution {
    Execution::local(ExecutionOptions {
        device: Device::Cpu(CpuOptions {
            threads: ThreadPolicy::Serial,
            jit: JitPolicy::Disabled,
        }),
        precision: Precision::F32,
        ..ExecutionOptions::default()
    })
    .unwrap()
}

fn wgpu_execution() -> Execution {
    Execution::local(ExecutionOptions {
        device: Device::Gpu(GpuOptions {
            backend: GpuBackend::Wgpu,
            ..GpuOptions::default()
        }),
        precision: Precision::F32,
        ..ExecutionOptions::default()
    })
    .unwrap()
}

#[test]
fn wgpu_f32_parameter_dependent_solve_gradient_matches_cpu_f32() {
    let x = event_scalar("x");
    let p = Expr::from(parameter!("p", initial: 0.4));
    let q = Expr::from(parameter!("q", initial: -0.2));
    let solution = solve(
        matrix([
            [x.clone() + p.clone() + 2.0, Expr::from(0.25)],
            [q.clone(), x + 3.0],
        ]),
        vector([Expr::from(1.0), Expr::from(0.5)]),
    );
    let model = CompiledModel::from_expr(&solution.component(0).norm_sqr()).unwrap();
    let data = Dataset::from_batch(
        EventBatch::from_events(
            std::sync::Arc::new(
                laddu::data::schema::Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap(),
            ),
            [
                laddu::data::data::OwnedEvent::weighted(vec![], vec![0.25], 0.5),
                laddu::data::data::OwnedEvent::weighted(vec![], vec![0.75], 1.5),
                laddu::data::data::OwnedEvent::weighted(vec![], vec![1.25], 2.0),
            ],
        )
        .unwrap(),
    );
    let term = NllTerm::new("solve", &model, &data, &data).unwrap();
    let cpu = likelihood(&term, cpu_f32_execution());
    let wgpu = likelihood(&term, wgpu_execution());
    let params = cpu.default_params();

    let expected = cpu.nll_with_gradient(&params).unwrap();
    let actual = wgpu.nll_with_gradient(&params).unwrap();
    let (index, actual_value, expected_value, relative) =
        worst_gradient_difference(actual.gradient(), expected.gradient());

    assert!(
        relative < 5.0e-3,
        "gradient[{index}] {actual_value} versus {expected_value} ({:.3}%)",
        relative * 100.0
    );
}

fn worst_gradient_difference(actual: &[f64], expected: &[f64]) -> (usize, f64, f64, f64) {
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| {
            let absolute = (actual - expected).abs();
            let scale = expected.abs().max(1.0);
            (index, *actual, *expected, absolute / scale)
        })
        .max_by(|lhs, rhs| lhs.3.total_cmp(&rhs.3))
        .unwrap()
}

#[test]
fn kmatrix_wgpu_f32_nll_gradient_matches_finite_difference() {
    let term = kmatrix_term();
    let cpu = likelihood(&term, cpu_f32_execution());
    let wgpu = likelihood(&term, wgpu_execution());
    let params = cpu.params_with(|_| 0.25);

    let expected = cpu.nll_with_gradient(&params).unwrap();
    let actual = wgpu.nll_with_gradient(&params).unwrap();
    let (index, actual_value, expected_value, relative) =
        worst_gradient_difference(actual.gradient(), expected.gradient());
    let parameter = cpu
        .params()
        .name(cpu.params().free_params()[index])
        .unwrap();
    let step = 1.0e-2;
    let mut plus = params.clone();
    let mut minus = params.clone();
    plus[index] += step;
    minus[index] -= step;
    let wgpu_finite_difference =
        (wgpu.nll(&plus).unwrap() - wgpu.nll(&minus).unwrap()) / (2.0 * step);
    let finite_difference_relative =
        (actual_value - wgpu_finite_difference).abs() / wgpu_finite_difference.abs().max(1.0);

    assert!(
        finite_difference_relative < 5.0e-3,
        "value {} versus {}; gradient[{index}] {parameter} {actual_value} versus CPU f32 {expected_value}; wgpu finite difference {wgpu_finite_difference}; CPU relative difference {:.3}%",
        actual.value(),
        expected.value(),
        relative * 100.0
    );
}
