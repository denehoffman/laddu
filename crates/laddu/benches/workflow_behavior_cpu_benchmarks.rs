use std::sync::Arc;

use criterion::{black_box, BatchSize, BenchmarkId, Criterion, Throughput};
use laddu::{
    amplitudes::{
        breit_wigner::BreitWigner,
        common::ComplexScalar,
        common::Scalar,
        kmatrix::{KopfKMatrixA0, KopfKMatrixA2, KopfKMatrixF0, KopfKMatrixF2},
        parameter,
        ylm::Ylm,
        zlm::Zlm,
    },
    data::{Dataset, DatasetReadOptions},
    extensions::NLL,
    io,
    resources::Parameters,
    traits::{LikelihoodTerm, Variable},
    utils::{
        enums::{Frame, Sign},
        variables::{Angles, Mass, PolAngle, PolMagnitude, Polarization, Topology},
    },
    RngSubsetExtension,
};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;
use rayon::ThreadPoolBuilder;

const BENCH_DATASET_RELATIVE_PATH: &str = "benches/bench.parquet";
const ROOT_BENCH_DATASET_RELATIVE_PATH: &str = "../laddu-core/test_data/data_f32.root";
const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];
const PARTIAL_WAVE_DATA_SAMPLE_EVENTS: usize = 512;
const PARTIAL_WAVE_ACCMC_SAMPLE_EVENTS: usize = 512;
const PARTIAL_WAVE_GENMC_SAMPLE_EVENTS: usize = 512;
const MOMENT_DATA_SAMPLE_EVENTS: usize = 256;
const MOMENT_ACCMC_SAMPLE_EVENTS: usize = 256;
const PARTIAL_WAVE_DATA_SEED: u64 = 11;
const PARTIAL_WAVE_ACCMC_SEED: u64 = 17;
const PARTIAL_WAVE_GENMC_SEED: u64 = 23;
const MOMENT_DATA_SEED: u64 = 41;
const MOMENT_ACCMC_SEED: u64 = 53;
const KMATRIX_DATASET_SEED: u64 = 71;
const PARTIAL_WAVE_BIN_COUNT: usize = 8;
const MOMENT_COMPACT_BASIS_MAX_L: usize = 2;

fn read_benchmark_dataset() -> Arc<Dataset> {
    let options = DatasetReadOptions::default()
        .p4_names(P4_NAMES)
        .aux_names(AUX_NAMES);
    let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(BENCH_DATASET_RELATIVE_PATH)
        .to_string_lossy()
        .into_owned();
    io::read_parquet(&dataset_path, &options).expect("benchmark dataset should open")
}

fn benchmark_parquet_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(BENCH_DATASET_RELATIVE_PATH)
        .to_string_lossy()
        .into_owned()
}

fn benchmark_root_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(ROOT_BENCH_DATASET_RELATIVE_PATH)
        .to_string_lossy()
        .into_owned()
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

fn deterministic_parameter_vector(n_free: usize, offset: f64) -> Vec<f64> {
    (0..n_free)
        .map(|index| offset + (index as f64 + 1.0) * 0.25)
        .collect()
}

fn kmatrix_max_events_from_env() -> Option<usize> {
    std::env::var("LADDU_BENCH_MAX_EVENTS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
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
    // Dataset/setup notes:
    // - Source parquet: benches/bench.parquet
    // - Subsampled events per dataset: 512 data, 512 accepted-MC, 512 generated-MC
    // - Seeds are fixed by named constants for reproducible sampling.
    let dataset = read_benchmark_dataset();
    let ds_data = sample_dataset(
        &dataset,
        PARTIAL_WAVE_DATA_SEED,
        PARTIAL_WAVE_DATA_SAMPLE_EVENTS,
    );
    let ds_accmc = sample_dataset(
        &dataset,
        PARTIAL_WAVE_ACCMC_SEED,
        PARTIAL_WAVE_ACCMC_SAMPLE_EVENTS,
    );
    let ds_genmc = sample_dataset(
        &dataset,
        PARTIAL_WAVE_GENMC_SEED,
        PARTIAL_WAVE_GENMC_SAMPLE_EVENTS,
    );
    let model = build_breit_wigner_partial_wave_model();
    let nll = NLL::new(&model, &ds_data, &ds_accmc).expect("unbinned NLL should build");
    let gen_evaluator = model
        .load(&ds_genmc)
        .expect("generated-mc evaluator should build");
    let params = vec![100.0, 0.112, 50.0, 50.0, 0.086];
    let n_data_events = ds_data.n_events() as u64;
    let n_gen_events = ds_genmc.n_events() as u64;

    let mut group = c.benchmark_group("breit_wigner_partial_wave_unbinned");
    group.throughput(Throughput::Elements(n_data_events));
    group.bench_function("nll_value_small_sample", |b| {
        b.iter(|| black_box(nll.evaluate(black_box(&params))))
    });
    group.throughput(Throughput::Elements(1));
    group.bench_function("nll_gradient_small_sample", |b| {
        b.iter(|| black_box(nll.evaluate_gradient(black_box(&params))))
    });
    group.throughput(Throughput::Elements(n_data_events));
    group.bench_function("nll_value_and_gradient_small_sample", |b| {
        b.iter(|| {
            let value = nll.evaluate(black_box(&params));
            let gradient = nll.evaluate_gradient(black_box(&params));
            black_box((value, gradient))
        })
    });
    group.throughput(Throughput::Elements(n_data_events));
    group.bench_function("projection_total_small_sample", |b| {
        b.iter(|| black_box(nll.project(black_box(&params), None)))
    });
    group.throughput(Throughput::Elements(1));
    group.bench_function("projection_total_with_gradient_small_sample", |b| {
        b.iter(|| black_box(nll.project_gradient(black_box(&params), None)))
    });
    group.throughput(Throughput::Elements(n_data_events));
    group.bench_function("projection_component_s_wave_small_sample", |b| {
        b.iter(|| {
            black_box(nll.project_with(black_box(&params), &["S0+", "Z00+", "f0(1500)"], None))
        })
    });
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        "projection_component_s_wave_with_gradient_small_sample",
        |b| {
            b.iter(|| {
                black_box(nll.project_gradient_with(
                    black_box(&params),
                    &["S0+", "Z00+", "f0(1500)"],
                    None,
                ))
            })
        },
    );
    group.throughput(Throughput::Elements(n_gen_events));
    group.bench_function("projection_total_generated_mc_small_sample", |b| {
        b.iter(|| black_box(nll.project(black_box(&params), Some(gen_evaluator.clone()))))
    });
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        "projection_total_generated_mc_with_gradient_small_sample",
        |b| {
            b.iter(|| {
                black_box(nll.project_gradient(black_box(&params), Some(gen_evaluator.clone())))
            })
        },
    );
    group.finish();

    let mass = Mass::new(["kshort1", "kshort2"]);
    let data_binned = ds_data
        .bin_by(mass.clone(), PARTIAL_WAVE_BIN_COUNT, (1.0, 2.0))
        .expect("binned data should build");
    let accmc_binned = ds_accmc
        .bin_by(mass, PARTIAL_WAVE_BIN_COUNT, (1.0, 2.0))
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

    let mut single_bin_group = c.benchmark_group("breit_wigner_partial_wave_single_bin");
    let n_bin_events = data_binned
        .get(target_bin)
        .expect("bin should exist")
        .n_events() as u64;
    single_bin_group.throughput(Throughput::Elements(n_bin_events));
    single_bin_group.bench_function("single_bin_nll_value_small_sample", |b| {
        b.iter(|| black_box(bin_nll.evaluate(black_box(&params))))
    });
    single_bin_group.throughput(Throughput::Elements(n_bin_events));
    single_bin_group.bench_function("single_bin_nll_value_and_gradient_small_sample", |b| {
        b.iter(|| {
            let value = bin_nll.evaluate(black_box(&params));
            let gradient = bin_nll.evaluate_gradient(black_box(&params));
            black_box((value, gradient))
        })
    });
    single_bin_group.finish();
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
    // Dataset/setup notes:
    // - Source parquet: benches/bench.parquet
    // - Subsampled events per dataset: 256 data, 256 accepted-MC
    // - Compact unpolarized basis: l <= 2
    // - Seeds are fixed by named constants for reproducible sampling.
    let dataset = read_benchmark_dataset();
    let ds_data = sample_dataset(&dataset, MOMENT_DATA_SEED, MOMENT_DATA_SAMPLE_EVENTS);
    let ds_accmc = sample_dataset(&dataset, MOMENT_ACCMC_SEED, MOMENT_ACCMC_SAMPLE_EVENTS);
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
    let mut group = c.benchmark_group("moment_analysis_unpolarized_compact_basis");
    let n_moment_events = ds_data.n_events() as u64;
    group.throughput(Throughput::Elements(n_moment_events));
    group.bench_function(
        "polarization_weighted_ylm_measurement_reference_mode_small_sample",
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
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        "polarization_weighted_ylm_measurement_value_and_gradient_small_sample",
        |b| {
            b.iter(|| {
                let values = measured_ylm.evaluate(&[]);
                let gradients = measured_ylm.evaluate_gradient(&[]);
                let mut value_sum = Complex64::new(0.0, 0.0);
                let mut gradient_sum = Complex64::new(0.0, 0.0);
                for index in 0..values.len() {
                    let pol_term = (2.0 * big_phi[index]).cos() / p_gamma[index];
                    let weighted = weights[index] * pol_term;
                    value_sum += values[index] * weighted;
                    if let Some(first) = gradients[index].get(0) {
                        gradient_sum += *first * weighted;
                    }
                }
                black_box((value_sum, gradient_sum))
            })
        },
    );

    let l_max = MOMENT_COMPACT_BASIS_MAX_L;
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

    let normalization_elements = (dim * dim) as u64;
    group.throughput(Throughput::Elements(normalization_elements));
    group.bench_function(
        "normalization_integral_matrix_assembly_unpolarized_compact_basis_small_sample",
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
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        "normalization_integral_matrix_value_and_gradient_assembly_unpolarized_compact_basis_small_sample",
        |b| {
            b.iter_batched(
                || DMatrix::<Complex64>::zeros(dim, dim),
                |mut matrix| {
                    let mut gradient_trace = Complex64::new(0.0, 0.0);
                    for (slot, evaluator) in norm_evaluators.iter().enumerate() {
                        let row = slot / dim;
                        let col = slot % dim;
                        let values = evaluator.evaluate(&[]);
                        let gradients = evaluator.evaluate_gradient(&[]);
                        let mut term = Complex64::new(0.0, 0.0);
                        for index in 0..values.len() {
                            term += values[index] * acc_weights[index];
                            if let Some(first) = gradients[index].get(0) {
                                gradient_trace += *first * acc_weights[index];
                            }
                        }
                        matrix[(row, col)] = term;
                    }
                    black_box((matrix, gradient_trace))
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
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        "moment_linear_system_solve_unpolarized_compact_basis",
        |b| {
            b.iter(|| {
                let lu = matrix.clone().lu();
                black_box(
                    lu.solve(&rhs)
                        .expect("benchmark matrix should remain invertible"),
                )
            })
        },
    );
    group.finish();
}

fn build_kmatrix_nll() -> Box<NLL> {
    let dataset = read_benchmark_dataset();
    let (ds_data, ds_mc) = if let Some(max_events) = kmatrix_max_events_from_env() {
        let sampled = sample_dataset(&dataset, KMATRIX_DATASET_SEED, max_events);
        (sampled.clone(), sampled)
    } else {
        (dataset.clone(), dataset)
    };
    let topology = Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton");
    let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
    let polarization = Polarization::new(topology.clone(), "pol_magnitude", "pol_angle");
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let z00p = Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization)
        .expect("z00+ should construct");
    let z00n = Zlm::new("Z00-", 0, 0, Sign::Negative, &angles, &polarization)
        .expect("z00- should construct");
    let z22p = Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization)
        .expect("z22+ should construct");
    let f0p = KopfKMatrixF0::new(
        "f0+",
        [
            [
                laddu::constant("f0+ c00 re", 0.0),
                laddu::constant("f0+ c00 im", 0.0),
            ],
            [
                parameter("f0(980)+ re"),
                laddu::constant("f0(980)+ im_fix", 0.0),
            ],
            [parameter("f0(1370)+ re"), parameter("f0(1370)+ im")],
            [parameter("f0(1500)+ re"), parameter("f0(1500)+ im")],
            [parameter("f0(1710)+ re"), parameter("f0(1710)+ im")],
        ],
        0,
        &resonance_mass,
        None,
    )
    .expect("f0+ should construct");
    let a0p = KopfKMatrixA0::new(
        "a0+",
        [
            [parameter("a0(980)+ re"), parameter("a0(980)+ im")],
            [parameter("a0(1450)+ re"), parameter("a0(1450)+ im")],
        ],
        0,
        &resonance_mass,
        None,
    )
    .expect("a0+ should construct");
    let f0n = KopfKMatrixF0::new(
        "f0-",
        [
            [
                laddu::constant("f0- c00 re", 0.0),
                laddu::constant("f0- c00 im", 0.0),
            ],
            [
                parameter("f0(980)- re"),
                laddu::constant("f0(980)- im_fix", 0.0),
            ],
            [parameter("f0(1370)- re"), parameter("f0(1370)- im")],
            [parameter("f0(1500)- re"), parameter("f0(1500)- im")],
            [parameter("f0(1710)- re"), parameter("f0(1710)- im")],
        ],
        0,
        &resonance_mass,
        None,
    )
    .expect("f0- should construct");
    let a0n = KopfKMatrixA0::new(
        "a0-",
        [
            [parameter("a0(980)- re"), parameter("a0(980)- im")],
            [parameter("a0(1450)- re"), parameter("a0(1450)- im")],
        ],
        0,
        &resonance_mass,
        None,
    )
    .expect("a0- should construct");
    let f2 = KopfKMatrixF2::new(
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
    )
    .expect("f2 should construct");
    let a2 = KopfKMatrixA2::new(
        "a2",
        [
            [parameter("a2(1320) re"), parameter("a2(1320) im")],
            [parameter("a2(1700) re"), parameter("a2(1700) im")],
        ],
        2,
        &resonance_mass,
        None,
    )
    .expect("a2 should construct");
    let s0p = f0p + a0p;
    let s0n = f0n + a0n;
    let d2p = f2 + a2;
    let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
    let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
    let neg_re = (&s0n * z00n.real()).norm_sqr();
    let neg_im = (&s0n * z00n.imag()).norm_sqr();
    let expr = pos_re + pos_im + neg_re + neg_im;
    NLL::new(&expr, &ds_data, &ds_mc).expect("k-matrix NLL should build")
}

fn kmatrix_nll_thread_scaling_benchmarks(c: &mut Criterion) {
    // Dataset/setup notes:
    // - Source parquet: benches/bench.parquet
    // - Full benchmark parquet is used for data and accepted-MC.
    // - Parameter vectors are deterministic.
    let nll = build_kmatrix_nll();
    let params = deterministic_parameter_vector(nll.n_free(), -100.0);
    let n_events = nll.data_evaluator.dataset.n_events() as u64;
    let mut group = c.benchmark_group("kmatrix_nll_thread_scaling");
    let thread_counts: Vec<usize> = (0..)
        .map(|i| 1usize << i)
        .take_while(|threads| *threads <= num_cpus::get())
        .collect();
    for threads in thread_counts {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool should build");
        group.throughput(Throughput::Elements(n_events));
        group.bench_with_input(
            BenchmarkId::new("value_only", threads),
            &threads,
            |b, &_threads| {
                b.iter_batched(
                    || params.clone(),
                    |parameters| pool.install(|| black_box(nll.evaluate(&parameters))),
                    BatchSize::SmallInput,
                )
            },
        );
        group.throughput(Throughput::Elements(n_events));
        group.bench_with_input(
            BenchmarkId::new("value_and_gradient", threads),
            &threads,
            |b, &_threads| {
                b.iter_batched(
                    || params.clone(),
                    |parameters| {
                        pool.install(|| {
                            let value = nll.evaluate(&parameters);
                            let gradient = nll.evaluate_gradient(&parameters);
                            black_box((value, gradient))
                        })
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn cached_evaluator_and_precompute_backend_benchmarks(c: &mut Criterion) {
    // Dataset/setup notes:
    // - Source parquet: benches/bench.parquet
    // - Subsampled events: 512
    let dataset = read_benchmark_dataset();
    let ds_data = sample_dataset(
        &dataset,
        PARTIAL_WAVE_DATA_SEED,
        PARTIAL_WAVE_DATA_SAMPLE_EVENTS,
    );
    let model = build_breit_wigner_partial_wave_model();
    let evaluator = model.load(&ds_data).expect("evaluator should build");
    let params = vec![100.0, 0.112, 50.0, 50.0, 0.086];
    let n_events = ds_data.n_events() as u64;

    let resources = evaluator.resources.read();
    let params_obj = Parameters::new(&params, &resources.constants);
    let active_indices = resources.active_indices().to_vec();
    let active_mask = resources.active.clone();
    let first_cache = resources.caches[0].clone();
    let amplitude_len = evaluator.amplitudes.len();
    let grad_dim = params_obj.len();
    let slot_count = evaluator.expression_slot_count();

    let mut value_stage_group = c.benchmark_group("stage_isolated_cached_value_and_expression");
    value_stage_group.throughput(Throughput::Elements(1));
    value_stage_group.bench_function("cached_value_fill_only", |b| {
        b.iter_batched(
            || vec![Complex64::ZERO; amplitude_len],
            |mut amplitude_values| {
                for &amp_idx in &active_indices {
                    amplitude_values[amp_idx] =
                        evaluator.amplitudes[amp_idx].compute(&params_obj, &first_cache);
                }
                black_box(amplitude_values)
            },
            BatchSize::SmallInput,
        )
    });

    let mut prefetched_values = vec![Complex64::ZERO; amplitude_len];
    for &amp_idx in &active_indices {
        prefetched_values[amp_idx] =
            evaluator.amplitudes[amp_idx].compute(&params_obj, &first_cache);
    }
    value_stage_group.bench_function("expression_value_eval_only", |b| {
        b.iter_batched(
            || vec![Complex64::ZERO; slot_count],
            |mut slots| {
                black_box(
                    evaluator
                        .evaluate_expression_value_with_scratch(&prefetched_values, &mut slots),
                )
            },
            BatchSize::SmallInput,
        )
    });
    value_stage_group.finish();

    let mut gradient_stage_group =
        c.benchmark_group("stage_isolated_cached_gradient_and_expression");
    gradient_stage_group.throughput(Throughput::Elements(1));
    gradient_stage_group.bench_function("cached_gradient_fill_only", |b| {
        b.iter_batched(
            || vec![DVector::zeros(grad_dim); amplitude_len],
            |mut gradient_values| {
                for ((amp, active), grad) in evaluator
                    .amplitudes
                    .iter()
                    .zip(active_mask.iter())
                    .zip(gradient_values.iter_mut())
                {
                    grad.iter_mut().for_each(|entry| *entry = Complex64::ZERO);
                    if *active {
                        amp.compute_gradient(&params_obj, &first_cache, grad);
                    }
                }
                black_box(gradient_values)
            },
            BatchSize::SmallInput,
        )
    });

    let mut prefetched_gradients = vec![DVector::zeros(grad_dim); amplitude_len];
    for ((amp, active), grad) in evaluator
        .amplitudes
        .iter()
        .zip(active_mask.iter())
        .zip(prefetched_gradients.iter_mut())
    {
        grad.iter_mut().for_each(|entry| *entry = Complex64::ZERO);
        if *active {
            amp.compute_gradient(&params_obj, &first_cache, grad);
        }
    }
    gradient_stage_group.bench_function("expression_gradient_eval_only", |b| {
        b.iter_batched(
            || {
                (
                    vec![Complex64::ZERO; slot_count],
                    vec![DVector::zeros(grad_dim); slot_count],
                )
            },
            |(mut value_slots, mut gradient_slots)| {
                black_box(evaluator.evaluate_expression_gradient_with_scratch(
                    &prefetched_values,
                    &prefetched_gradients,
                    &mut value_slots,
                    &mut gradient_slots,
                ))
            },
            BatchSize::SmallInput,
        )
    });
    gradient_stage_group.finish();

    let mut precompute_only_group = c.benchmark_group("precompute_stage_only");
    precompute_only_group.throughput(Throughput::Elements(n_events));
    precompute_only_group.bench_function("precompute_only", |b| {
        b.iter_batched(
            || resources.clone(),
            |mut stage_resources| {
                for amp in &evaluator.amplitudes {
                    amp.precompute_all(&ds_data, &mut stage_resources);
                }
                black_box(stage_resources.caches.len())
            },
            BatchSize::SmallInput,
        )
    });
    precompute_only_group.throughput(Throughput::Elements(n_events));
    precompute_only_group.bench_function("precompute_view_only", |b| {
        b.iter_batched(
            || resources.clone(),
            |mut stage_resources| {
                for amp in &evaluator.amplitudes {
                    amp.precompute_all_view(&ds_data, &mut stage_resources);
                }
                black_box(stage_resources.caches.len())
            },
            BatchSize::SmallInput,
        )
    });
    precompute_only_group.finish();
}

fn io_open_benchmarks(c: &mut Criterion) {
    let parquet_options = DatasetReadOptions::default()
        .p4_names(P4_NAMES)
        .aux_names(AUX_NAMES);
    let root_options = DatasetReadOptions::default();
    let parquet_path = benchmark_parquet_path();
    let root_path = benchmark_root_path();

    let parquet_events = io::read_parquet(&parquet_path, &parquet_options)
        .expect("parquet warmup should open")
        .n_events() as u64;
    let root_events = io::read_root(&root_path, &root_options)
        .expect("root warmup should open")
        .n_events() as u64;

    let mut group = c.benchmark_group("file_open");
    group.throughput(Throughput::Elements(parquet_events));
    group.bench_function("parquet_open", |b| {
        b.iter(|| {
            let dataset = io::read_parquet(black_box(&parquet_path), black_box(&parquet_options))
                .expect("parquet load should succeed");
            black_box(dataset.n_events())
        })
    });
    group.throughput(Throughput::Elements(root_events));
    group.bench_function("root_open", |b| {
        b.iter(|| {
            let dataset = io::read_root(black_box(&root_path), black_box(&root_options))
                .expect("root load should succeed");
            black_box(dataset.n_events())
        })
    });
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default().sample_size(120).configure_from_args();
    breit_wigner_partial_wave_benchmarks(&mut criterion);
    moment_analysis_benchmarks(&mut criterion);
    kmatrix_nll_thread_scaling_benchmarks(&mut criterion);
    cached_evaluator_and_precompute_backend_benchmarks(&mut criterion);
    io_open_benchmarks(&mut criterion);
    criterion.final_summary();
}
