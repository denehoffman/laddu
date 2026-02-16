use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use laddu::{
    amplitudes::{
        breit_wigner::BreitWigner, common::ComplexScalar, common::Scalar, parameter, ylm::Ylm,
        zlm::Zlm,
    },
    data::{Dataset, DatasetReadOptions},
    extensions::NLL,
    io,
    traits::{LikelihoodTerm, Variable},
    utils::{
        enums::{Frame, Sign},
        variables::{Angles, Mass, PolAngle, PolMagnitude, Topology},
    },
    RngSubsetExtension,
};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

const BENCH_DATASET_PATH: &str = "benches/bench.parquet";
const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];

fn read_benchmark_dataset() -> Arc<Dataset> {
    let options = DatasetReadOptions::default()
        .p4_names(P4_NAMES)
        .aux_names(AUX_NAMES);
    io::read_parquet(BENCH_DATASET_PATH, &options).expect("benchmark dataset should open")
}

fn sample_dataset(dataset: &Arc<Dataset>, seed: u64, max_events: usize) -> Arc<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let n_total = dataset.n_events();
    let n_take = max_events.min(n_total);
    let indices = rng.subset(n_total, n_take);
    let events = indices
        .into_iter()
        .map(|index| {
            dataset
                .event(index)
                .expect("subset index should be valid")
                .data_arc()
        })
        .collect();
    Arc::new(Dataset::new_with_metadata(events, dataset.metadata_arc()))
}

fn build_breit_wigner_partial_wave_model() -> laddu::Expression {
    let topology = Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton");
    let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
    let polarization = laddu::Polarization::new(topology, "pol_magnitude", "pol_angle");
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let daughter_1_mass = Mass::new(["kshort1"]);
    let daughter_2_mass = Mass::new(["kshort2"]);

    let z00p = Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization)
        .expect("z00 should construct");
    let z22p = Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization)
        .expect("z22 should construct");
    let bw_f01500 = BreitWigner::new(
        "f0(1500)",
        laddu::constant("f0_mass", 1.506),
        parameter("f0_width"),
        0,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
    )
    .expect("f0(1500) should construct");
    let bw_f21525 = BreitWigner::new(
        "f2(1525)",
        laddu::constant("f2_mass", 1.517),
        parameter("f2_width"),
        2,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
    )
    .expect("f2(1525) should construct");
    let s0p = Scalar::new("S0+", parameter("S0+ re")).expect("S0+ should construct");
    let d2p = ComplexScalar::new("D2+", parameter("D2+ re"), parameter("D2+ im"))
        .expect("D2+ should construct");

    let pos_re = (&s0p * &bw_f01500 * z00p.real() + &d2p * &bw_f21525 * z22p.real()).norm_sqr();
    let pos_im = (&s0p * &bw_f01500 * z00p.imag() + &d2p * &bw_f21525 * z22p.imag()).norm_sqr();
    pos_re + pos_im
}

fn breit_wigner_partial_wave_benchmarks(c: &mut Criterion) {
    let dataset = read_benchmark_dataset();
    let ds_data = sample_dataset(&dataset, 11, 512);
    let ds_accmc = sample_dataset(&dataset, 17, 512);
    let ds_genmc = sample_dataset(&dataset, 23, 512);
    let model = build_breit_wigner_partial_wave_model();
    let nll = NLL::new(&model, &ds_data, &ds_accmc).expect("unbinned NLL should build");
    let gen_evaluator = model
        .load(&ds_genmc)
        .expect("generated-mc evaluator should build");
    let params = vec![100.0, 0.112, 50.0, 50.0, 0.086];

    let mut group = c.benchmark_group("breit_wigner_partial_wave_unbinned");
    group.bench_function("nll_value_small_sample", |b| {
        b.iter(|| black_box(nll.evaluate(black_box(&params))))
    });
    group.bench_function("nll_gradient_small_sample", |b| {
        b.iter(|| black_box(nll.evaluate_gradient(black_box(&params))))
    });
    group.bench_function("projection_total_small_sample", |b| {
        b.iter(|| black_box(nll.project(black_box(&params), None)))
    });
    group.bench_function("projection_component_s_wave_small_sample", |b| {
        b.iter(|| {
            black_box(nll.project_with(black_box(&params), &["S0+", "Z00+", "f0(1500)"], None))
        })
    });
    group.bench_function("projection_total_generated_mc_small_sample", |b| {
        b.iter(|| black_box(nll.project(black_box(&params), Some(gen_evaluator.clone()))))
    });
    group.finish();

    let mass = Mass::new(["kshort1", "kshort2"]);
    let data_binned = ds_data
        .bin_by(mass.clone(), 8, (1.0, 2.0))
        .expect("binned data should build");
    let accmc_binned = ds_accmc
        .bin_by(mass, 8, (1.0, 2.0))
        .expect("binned accmc should build");
    let target_bin = (0..data_binned.n_bins())
        .find(|&index| {
            data_binned
                .get(index)
                .zip(accmc_binned.get(index))
                .map(|(data_bin, accmc_bin)| data_bin.n_events() >= 8 && accmc_bin.n_events() >= 8)
                .unwrap_or(false)
        })
        .expect("at least one dense bin should exist");
    let bin_nll = NLL::new(
        &model,
        data_binned.get(target_bin).expect("bin should exist"),
        accmc_binned.get(target_bin).expect("bin should exist"),
    )
    .expect("single-bin NLL should build");

    c.bench_function("single_bin_nll_value_small_sample", |b| {
        b.iter(|| black_box(bin_nll.evaluate(black_box(&params))))
    });
}

fn unpolarized_moment_ilms(l_max: usize) -> Vec<(usize, isize)> {
    let mut ilms = Vec::new();
    for l in 0..=l_max {
        for m in 0..=l {
            ilms.push((l, m as isize));
        }
    }
    ilms
}

fn moment_analysis_benchmarks(c: &mut Criterion) {
    let dataset = read_benchmark_dataset();
    let ds_data = sample_dataset(&dataset, 41, 256);
    let ds_accmc = sample_dataset(&dataset, 53, 256);
    let topology = Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton");
    let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
    let pol_angle = PolAngle::new(topology, "pol_angle");
    let pol_magnitude = PolMagnitude::new("pol_magnitude");
    let big_phi = pol_angle
        .value_on(&ds_data)
        .expect("pol angle values should evaluate");
    let p_gamma = pol_magnitude
        .value_on(&ds_data)
        .expect("pol magnitude values should evaluate");
    let weights = ds_data.weights();

    let measured_ylm = Ylm::new("moment_measured_ylm", 2, 1, &angles)
        .expect("ylm should construct")
        .conj()
        .load(&ds_data)
        .expect("measured-moment evaluator should build");
    c.bench_function(
        "polarization_weighted_ylm_measurement_l2_m1_small_sample",
        |b| {
            b.iter(|| {
                let values = measured_ylm.evaluate(&[]);
                let mut sum = Complex64::new(0.0, 0.0);
                for index in 0..values.len() {
                    let pol_term = (2.0 * big_phi[index]).cos() / p_gamma[index];
                    sum += values[index] * (weights[index] * pol_term);
                }
                black_box(sum)
            })
        },
    );

    let l_max = 2usize;
    let ilms = unpolarized_moment_ilms(l_max);
    let dim = ilms.len();
    let acc_weights = ds_accmc.weights();
    let mut norm_evaluators = Vec::with_capacity(dim * dim);
    for (l, m) in &ilms {
        for (lp, mp) in &ilms {
            let ylm = Ylm::new("norm_ylm_left", *l, *m, &angles).expect("ylm should construct");
            let ylpmp =
                Ylm::new("norm_ylm_right", *lp, *mp, &angles).expect("ylm should construct");
            let evaluator = (ylm.conj() * ylpmp.real())
                .load(&ds_accmc)
                .expect("normalization evaluator should build");
            norm_evaluators.push(evaluator);
        }
    }

    c.bench_function(
        "normalization_integral_matrix_assembly_unpolarized_lmax2_small_sample",
        |b| {
            b.iter_batched(
                || DMatrix::<Complex64>::zeros(dim, dim),
                |mut matrix| {
                    for (slot, evaluator) in norm_evaluators.iter().enumerate() {
                        let row = slot / dim;
                        let col = slot % dim;
                        let values = evaluator.evaluate(&[]);
                        let mut term = Complex64::new(0.0, 0.0);
                        for index in 0..values.len() {
                            term += values[index] * acc_weights[index];
                        }
                        matrix[(row, col)] = term;
                    }
                    black_box(matrix)
                },
                BatchSize::SmallInput,
            )
        },
    );

    let mut matrix = DMatrix::<Complex64>::zeros(dim, dim);
    let mut rhs = DVector::<Complex64>::zeros(dim);
    for row in 0..dim {
        rhs[row] = Complex64::new(row as f64 + 1.0, 0.0);
        for col in 0..dim {
            matrix[(row, col)] = Complex64::new((row + col + 1) as f64, 0.0);
        }
        matrix[(row, row)] += Complex64::new(dim as f64, 0.0);
    }
    c.bench_function("moment_linear_system_solve_unpolarized_lmax2", |b| {
        b.iter(|| {
            let lu = matrix.clone().lu();
            black_box(
                lu.solve(&rhs)
                    .expect("benchmark matrix should remain invertible"),
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(120);
    targets = breit_wigner_partial_wave_benchmarks, moment_analysis_benchmarks
}
criterion_main!(benches);
