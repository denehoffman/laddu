use std::{env, fs, sync::Arc, time::Instant};

use laddu::{
    amplitudes::{
        breit_wigner::BreitWigner,
        common::ComplexScalar,
        common::Scalar,
        kmatrix::{KopfKMatrixA0, KopfKMatrixA2, KopfKMatrixF0, KopfKMatrixF2},
        parameter,
        zlm::Zlm,
    },
    data::{Dataset, DatasetReadOptions},
    io,
    traits::LikelihoodTerm,
    utils::{
        enums::{Frame, Sign},
        variables::{Angles, Mass, Polarization, Topology},
    },
};

const BENCH_DATASET_RELATIVE_PATH: &str = "benches/bench.parquet";
const ROOT_BENCH_DATASET_RELATIVE_PATH: &str = "../laddu-core/test_data/data_f32.root";
const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];
const SAMPLE_EVENTS: usize = 512;
const SAMPLE_SEED: u64 = 11;
const WARMUP_ITERS: usize = 64;
const DEFAULT_ITERS: usize = 4096;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Evaluate,
    Gradient,
    Load,
    LoadOnce,
    Precompute,
    PrecomputeOnce,
    PrecomputeCompare,
    Nll,
    IoOnce,
    ParquetOpen,
    RootOpen,
}

impl Mode {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "evaluate" | "eval_aos" | "eval_cached" => Some(Self::Evaluate),
            "evaluate_gradient" | "grad_aos" | "grad_cached" => Some(Self::Gradient),
            "load" | "load_aos" | "load_columnar" | "load_storage" => Some(Self::Load),
            "load_once" | "load_aos_once" | "load_columnar_once" => Some(Self::LoadOnce),
            "precompute" | "precompute_aos" | "precompute_columnar" => Some(Self::Precompute),
            "precompute_once" | "precompute_aos_once" | "precompute_columnar_once" => {
                Some(Self::PrecomputeOnce)
            }
            "precompute_compare" => Some(Self::PrecomputeCompare),
            "nll" | "nll_aos" => Some(Self::Nll),
            "io_once" | "io_aos_once" | "io_columnar_once" => Some(Self::IoOnce),
            "parquet_open" | "parquet_open_aos" | "parquet_open_columnar" => {
                Some(Self::ParquetOpen)
            }
            "root_open" | "root_open_aos" | "root_open_columnar" => Some(Self::RootOpen),
            _ => None,
        }
    }
}

fn usage() {
    eprintln!(
        "Usage: cargo run --release --bin profile_cached_paths -- <mode> [iters]\n\
         modes: evaluate | evaluate_gradient | load | load_once | precompute | precompute_once | precompute_compare | nll | io_once | parquet_open | root_open"
    );
}

fn read_peak_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let vm_hwm = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?;
    vm_hwm.parse::<u64>().ok()
}

fn print_precompute_breakdown_header(mode: &str, iters: usize) {
    eprintln!("mode={mode} breakdown iters={iters}");
    eprintln!("rank amp_index total_ns avg_ns_per_iter");
}

fn print_precompute_breakdown_rows(totals_ns: &[u128], iters: usize) {
    let mut indexed: Vec<(usize, u128)> = totals_ns.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));
    for (rank, (amp_index, total_ns)) in indexed.into_iter().enumerate() {
        let avg_ns = total_ns as f64 / iters as f64;
        eprintln!("{} {} {} {:.3}", rank + 1, amp_index, total_ns, avg_ns);
    }
}

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
    let _ = seed;
    let n_total = dataset.n_events();
    let n_take = max_events.min(n_total);
    let events = (0..n_take)
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
    let polarization = Polarization::new(topology, "pol_magnitude", "pol_angle");
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let daughter_1_mass = Mass::new(["kshort1"]);
    let daughter_2_mass = Mass::new(["kshort2"]);

    let z00p =
        Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization).expect("z00 should build");
    let z22p =
        Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization).expect("z22 should build");
    let bw_f01500 = BreitWigner::new(
        "f0(1500)",
        laddu::constant("f0_mass", 1.506),
        parameter("f0_width"),
        0,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
    )
    .expect("f0(1500) should build");
    let bw_f21525 = BreitWigner::new(
        "f2(1525)",
        laddu::constant("f2_mass", 1.517),
        parameter("f2_width"),
        2,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
    )
    .expect("f2(1525) should build");
    let s0p = Scalar::new("S0+", parameter("S0+ re")).expect("S0+ should build");
    let d2p = ComplexScalar::new("D2+", parameter("D2+ re"), parameter("D2+ im"))
        .expect("D2+ should build");

    let pos_re = (&s0p * &bw_f01500 * z00p.real() + &d2p * &bw_f21525 * z22p.real()).norm_sqr();
    let pos_im = (&s0p * &bw_f01500 * z00p.imag() + &d2p * &bw_f21525 * z22p.imag()).norm_sqr();
    pos_re + pos_im
}

fn deterministic_parameter_vector(n_free: usize, offset: f64) -> Vec<f64> {
    (0..n_free)
        .map(|index| offset + (index as f64 + 1.0) * 0.25)
        .collect()
}

fn build_kmatrix_nll() -> (Box<laddu::extensions::NLL>, Vec<f64>) {
    let dataset = read_benchmark_dataset();
    let ds_data = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
    let ds_mc = ds_data.clone();
    let topology = Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton");
    let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
    let polarization = Polarization::new(topology.clone(), "pol_magnitude", "pol_angle");
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let z00p =
        Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization).expect("z00+ should build");
    let z00n =
        Zlm::new("Z00-", 0, 0, Sign::Negative, &angles, &polarization).expect("z00- should build");
    let z22p =
        Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization).expect("z22+ should build");

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
    .expect("f0+ should build");
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
    .expect("a0+ should build");
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
    .expect("f0- should build");
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
    .expect("a0- should build");
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
    .expect("f2 should build");
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
    .expect("a2 should build");
    let s0p = f0p + a0p;
    let s0n = f0n + a0n;
    let d2p = f2 + a2;
    let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
    let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
    let neg_re = (&s0n * z00n.real()).norm_sqr();
    let neg_im = (&s0n * z00n.imag()).norm_sqr();
    let expr = pos_re + pos_im + neg_re + neg_im;
    let nll = laddu::extensions::NLL::new(&expr, &ds_data, &ds_mc).expect("nll should build");
    let params = deterministic_parameter_vector(nll.n_free(), -100.0);
    (nll, params)
}

fn run_mode(mode: Mode, iters: usize) {
    match mode {
        Mode::Evaluate => {
            let dataset = read_benchmark_dataset();
            let ds = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
            let model = build_breit_wigner_partial_wave_model();
            let evaluator = model.load(&ds).expect("evaluator should build");
            let params = vec![100.0, 0.112, 50.0, 50.0, 0.086];
            let mut sink = 0.0;
            for _ in 0..WARMUP_ITERS {
                sink += evaluator.evaluate_local(&params)[0].re;
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                sink += evaluator.evaluate_local(&params)[0].re;
            }
            eprintln!(
                "mode=evaluate iters={iters} sink={sink} elapsed={:?}",
                t0.elapsed()
            );
        }
        Mode::Gradient => {
            let dataset = read_benchmark_dataset();
            let ds = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
            let model = build_breit_wigner_partial_wave_model();
            let evaluator = model.load(&ds).expect("evaluator should build");
            let params = vec![100.0, 0.112, 50.0, 50.0, 0.086];
            let mut sink = 0.0;
            for _ in 0..WARMUP_ITERS {
                sink += evaluator.evaluate_gradient_local(&params)[0][0].re;
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                sink += evaluator.evaluate_gradient_local(&params)[0][0].re;
            }
            eprintln!(
                "mode=evaluate_gradient iters={iters} sink={sink} elapsed={:?}",
                t0.elapsed()
            );
        }
        Mode::Load => {
            let dataset = read_benchmark_dataset();
            let ds = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
            let model = build_breit_wigner_partial_wave_model();
            let mut sink = 0.0;
            for _ in 0..WARMUP_ITERS {
                let ev = model.load(&ds).expect("load should succeed");
                sink += ev.n_parameters() as f64;
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                let ev = model.load(&ds).expect("load should succeed");
                sink += ev.n_parameters() as f64;
            }
            eprintln!(
                "mode=load iters={iters} sink={sink} elapsed={:?}",
                t0.elapsed()
            );
        }
        Mode::LoadOnce => {
            let dataset = read_benchmark_dataset();
            let ds = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
            let model = build_breit_wigner_partial_wave_model();
            let t0 = Instant::now();
            let ev = model.load(&ds).expect("load should succeed");
            let sink = ev.n_parameters() as f64;
            eprintln!("mode=load_once sink={sink} elapsed={:?}", t0.elapsed());
        }
        Mode::Precompute => {
            let dataset = read_benchmark_dataset();
            let ds = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
            let model = build_breit_wigner_partial_wave_model();
            let evaluator = model.load(&ds).expect("evaluator should build");
            let base_resources = evaluator.resources.read().clone();
            let mut sink = 0.0;
            for _ in 0..WARMUP_ITERS {
                let mut resources = base_resources.clone();
                for amplitude in &evaluator.amplitudes {
                    amplitude.precompute_all(&ds, &mut resources);
                }
                sink += resources.caches.len() as f64;
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut resources = base_resources.clone();
                for amplitude in &evaluator.amplitudes {
                    amplitude.precompute_all(&ds, &mut resources);
                }
                sink += resources.caches.len() as f64;
            }
            eprintln!(
                "mode=precompute iters={iters} sink={sink} elapsed={:?}",
                t0.elapsed()
            );
        }
        Mode::PrecomputeOnce => {
            let dataset = read_benchmark_dataset();
            let ds = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
            let model = build_breit_wigner_partial_wave_model();
            let evaluator = model.load(&ds).expect("evaluator should build");
            let mut resources = evaluator.resources.read().clone();
            let t0 = Instant::now();
            for amplitude in &evaluator.amplitudes {
                amplitude.precompute_all(&ds, &mut resources);
            }
            let sink = resources.caches.len() as f64;
            eprintln!(
                "mode=precompute_once sink={sink} elapsed={:?}",
                t0.elapsed()
            );
        }
        Mode::PrecomputeCompare => {
            let dataset = read_benchmark_dataset();
            let ds = sample_dataset(&dataset, SAMPLE_SEED, SAMPLE_EVENTS);
            let ds_canonical = ds.clone();
            let model = build_breit_wigner_partial_wave_model();
            let evaluator_baseline = model.load(&ds).expect("baseline evaluator should build");
            let evaluator_canonical = model
                .load(&ds_canonical)
                .expect("canonical evaluator should build");

            let amp_len = evaluator_baseline.amplitudes.len();
            let mut totals_baseline_ns = vec![0u128; amp_len];
            let mut totals_canonical_ns = vec![0u128; amp_len];
            let mut sink = 0.0;

            for _ in 0..iters {
                let mut resources_baseline = evaluator_baseline.resources.read().clone();
                for (amp_index, amplitude) in evaluator_baseline.amplitudes.iter().enumerate() {
                    let t0 = Instant::now();
                    amplitude.precompute_all(&ds, &mut resources_baseline);
                    totals_baseline_ns[amp_index] += t0.elapsed().as_nanos();
                }
                sink += resources_baseline.caches.len() as f64;

                let mut resources_canonical = evaluator_canonical.resources.read().clone();
                for (amp_index, amplitude) in evaluator_canonical.amplitudes.iter().enumerate() {
                    let t0 = Instant::now();
                    amplitude.precompute_all(&ds_canonical, &mut resources_canonical);
                    totals_canonical_ns[amp_index] += t0.elapsed().as_nanos();
                }
                sink += resources_canonical.caches.len() as f64;
            }

            print_precompute_breakdown_header("precompute_compare_baseline", iters);
            print_precompute_breakdown_rows(&totals_baseline_ns, iters);

            print_precompute_breakdown_header("precompute_compare_canonical", iters);
            print_precompute_breakdown_rows(&totals_canonical_ns, iters);

            let mut compare_rows: Vec<(usize, u128, u128, f64)> = (0..amp_len)
                .map(|amp_index| {
                    let baseline = totals_baseline_ns[amp_index];
                    let canonical = totals_canonical_ns[amp_index];
                    let ratio = if baseline == 0 {
                        f64::INFINITY
                    } else {
                        canonical as f64 / baseline as f64
                    };
                    (amp_index, baseline, canonical, ratio)
                })
                .collect();
            compare_rows.sort_by(|a, b| b.3.total_cmp(&a.3));
            eprintln!("mode=precompute_compare slowdown_rank iters={iters} sink={sink}");
            eprintln!(
                "rank amp_index baseline_total_ns canonical_total_ns canonical_over_baseline"
            );
            for (rank, (amp_index, baseline, canonical, ratio)) in compare_rows.iter().enumerate() {
                eprintln!(
                    "{} {} {} {} {:.6}",
                    rank + 1,
                    amp_index,
                    baseline,
                    canonical,
                    ratio
                );
            }
        }
        Mode::Nll => {
            let (nll, params) = build_kmatrix_nll();
            let mut sink = 0.0;
            for _ in 0..WARMUP_ITERS {
                sink += nll.evaluate(&params).expect("nll value should succeed");
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                sink += nll.evaluate(&params).expect("nll value should succeed");
            }
            eprintln!(
                "mode=nll iters={iters} sink={sink} elapsed={:?}",
                t0.elapsed()
            );
        }
        Mode::IoOnce => {
            let options = DatasetReadOptions::default()
                .p4_names(P4_NAMES)
                .aux_names(AUX_NAMES);
            let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(BENCH_DATASET_RELATIVE_PATH)
                .to_string_lossy()
                .into_owned();
            let t0 = Instant::now();
            let ds =
                io::read_parquet(&dataset_path, &options).expect("aos parquet load should work");
            let elapsed = t0.elapsed();
            let n_events = ds.n_events() as u64;
            let n_p4 = ds.p4_names().len() as u64;
            let n_aux = ds.aux_names().len() as u64;
            let scalar_copies = n_events * (4 * n_p4 + n_aux + 1);
            let row_materializations = 0u64;
            let event_arc_allocations = 0u64;
            let peak_rss_kb = read_peak_rss_kb().unwrap_or(0);
            eprintln!(
                "mode=io_once elapsed={elapsed:?} n_events={n_events} n_p4={n_p4} n_aux={n_aux} scalar_copies={scalar_copies} row_materializations={row_materializations} event_arc_allocations={event_arc_allocations} peak_rss_kb={peak_rss_kb}"
            );
        }
        Mode::ParquetOpen => {
            let options = DatasetReadOptions::default()
                .p4_names(P4_NAMES)
                .aux_names(AUX_NAMES);
            let dataset_path = benchmark_parquet_path();
            let t0 = Instant::now();
            let ds = io::read_parquet(&dataset_path, &options).expect("parquet load should work");
            let elapsed = t0.elapsed();
            let peak_rss_kb = read_peak_rss_kb().unwrap_or(0);
            eprintln!(
                "mode=parquet_open elapsed={elapsed:?} n_events={} n_p4={} n_aux={} peak_rss_kb={peak_rss_kb}",
                ds.n_events(),
                ds.p4_names().len(),
                ds.aux_names().len()
            );
        }
        Mode::RootOpen => {
            let options = DatasetReadOptions::default();
            let dataset_path = benchmark_root_path();
            let t0 = Instant::now();
            let ds = io::read_root(&dataset_path, &options).expect("root load should work");
            let elapsed = t0.elapsed();
            let peak_rss_kb = read_peak_rss_kb().unwrap_or(0);
            eprintln!(
                "mode=root_open elapsed={elapsed:?} n_events={} n_p4={} n_aux={} peak_rss_kb={peak_rss_kb}",
                ds.n_events(),
                ds.p4_names().len(),
                ds.aux_names().len()
            );
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
        std::process::exit(2);
    }
    let mode = match Mode::parse(&args[1]) {
        Some(mode) => mode,
        None => {
            usage();
            std::process::exit(2);
        }
    };
    let iters = args
        .get(2)
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ITERS);
    run_mode(mode, iters);
}
