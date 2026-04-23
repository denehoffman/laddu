use std::{env, hint::black_box, process, sync::Arc, time::Instant};

use laddu::{
    amplitudes::{
        breit_wigner::BreitWigner,
        common::{ComplexScalar, Scalar},
        kmatrix::{
            KopfKMatrixA0, KopfKMatrixA0Channel, KopfKMatrixA2, KopfKMatrixA2Channel,
            KopfKMatrixF0, KopfKMatrixF0Channel, KopfKMatrixF2, KopfKMatrixF2Channel,
        },
        zlm::Zlm,
    },
    data::{Dataset, DatasetReadOptions},
    extensions::NLL,
    io, parameter,
    quantum::{Frame, Sign},
    traits::LikelihoodTerm,
    variables::Mass,
    Evaluator, Expression, RngSubsetExtension,
};
use rayon::ThreadPoolBuilder;

const BENCH_DATASET_RELATIVE_PATH: &str = "benches/bench.parquet";
const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];
const PARTIAL_WAVE_DATA_SAMPLE_EVENTS: usize = 512;
const PARTIAL_WAVE_ACCMC_SAMPLE_EVENTS: usize = 512;
const PARTIAL_WAVE_GENMC_SAMPLE_EVENTS: usize = 512;
const PARTIAL_WAVE_DATA_SEED: u64 = 11;
const PARTIAL_WAVE_ACCMC_SEED: u64 = 17;
const PARTIAL_WAVE_GENMC_SEED: u64 = 23;
const KMATRIX_DATASET_SEED: u64 = 71;
const DEFAULT_WARMUP_ITERS: usize = 64;
const DEFAULT_ITERS: usize = 4096;

fn reaction_variables() -> (laddu::Angles, laddu::Polarization, Mass, Mass, Mass) {
    let beam = laddu::Particle::measured("beam", "beam");
    let target = laddu::Particle::missing("target");
    let kshort1 = laddu::Particle::measured("K_S1", "kshort1");
    let kshort2 = laddu::Particle::measured("K_S2", "kshort2");
    let kk = laddu::Particle::composite("KK", [&kshort1, &kshort2]).unwrap();
    let proton = laddu::Particle::measured("proton", "proton");
    let reaction = laddu::Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap();
    let decay = reaction.decay(&kk).unwrap();
    let angles = decay.angles(&kshort1, Frame::Helicity).unwrap();
    let polarization = reaction.polarization("pol_magnitude", "pol_angle");
    let resonance_mass = decay.parent_mass();
    let daughter_1_mass = decay.daughter_1_mass();
    let daughter_2_mass = decay.daughter_2_mass();
    (
        angles,
        polarization,
        resonance_mass,
        daughter_1_mass,
        daughter_2_mass,
    )
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    BreitWignerEvaluateLocal,
    BreitWignerNllValue,
    BreitWignerProjectionTotal,
    BreitWignerProjectionGeneratedMc,
    KmatrixValue,
    KmatrixDataTerm,
    KmatrixMcTerm,
}

impl Mode {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "bw_evaluate_local" => Some(Self::BreitWignerEvaluateLocal),
            "bw_nll_value" => Some(Self::BreitWignerNllValue),
            "bw_projection_total" => Some(Self::BreitWignerProjectionTotal),
            "bw_projection_generated_mc" => Some(Self::BreitWignerProjectionGeneratedMc),
            "kmatrix_value" => Some(Self::KmatrixValue),
            "kmatrix_data_term" => Some(Self::KmatrixDataTerm),
            "kmatrix_mc_term" => Some(Self::KmatrixMcTerm),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::BreitWignerEvaluateLocal => "bw_evaluate_local",
            Self::BreitWignerNllValue => "bw_nll_value",
            Self::BreitWignerProjectionTotal => "bw_projection_total",
            Self::BreitWignerProjectionGeneratedMc => "bw_projection_generated_mc",
            Self::KmatrixValue => "kmatrix_value",
            Self::KmatrixDataTerm => "kmatrix_data_term",
            Self::KmatrixMcTerm => "kmatrix_mc_term",
        }
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    dataset_path: String,
    iterations: usize,
    warmup_iters: usize,
    thread_count: Option<usize>,
    max_events: Option<usize>,
}

impl Args {
    fn parse() -> Self {
        let mut mode = None;
        let mut dataset_path = benchmark_parquet_path();
        let mut iterations = DEFAULT_ITERS;
        let mut warmup_iters = DEFAULT_WARMUP_ITERS;
        let mut thread_count = None;
        let mut max_events = None;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" => mode = Some(parse_mode(args.next())),
                _ if arg.starts_with("--mode=") => {
                    mode = Some(parse_mode(Some(arg["--mode=".len()..].to_string())));
                }
                "--dataset" => {
                    dataset_path = args
                        .next()
                        .unwrap_or_else(|| usage_and_exit("missing dataset path"));
                }
                _ if arg.starts_with("--dataset=") => {
                    dataset_path = arg["--dataset=".len()..].to_string();
                }
                "--iterations" => {
                    iterations = parse_positive(args.next(), "iterations");
                }
                _ if arg.starts_with("--iterations=") => {
                    iterations = parse_positive(
                        Some(arg["--iterations=".len()..].to_string()),
                        "iterations",
                    );
                }
                "--warmup" => {
                    warmup_iters = parse_nonnegative(args.next(), "warmup");
                }
                _ if arg.starts_with("--warmup=") => {
                    warmup_iters =
                        parse_nonnegative(Some(arg["--warmup=".len()..].to_string()), "warmup");
                }
                "--threads" => {
                    thread_count = Some(parse_positive(args.next(), "threads"));
                }
                _ if arg.starts_with("--threads=") => {
                    thread_count = Some(parse_positive(
                        Some(arg["--threads=".len()..].to_string()),
                        "threads",
                    ));
                }
                "--max-events" => {
                    max_events = Some(parse_positive(args.next(), "max-events"));
                }
                _ if arg.starts_with("--max-events=") => {
                    max_events = Some(parse_positive(
                        Some(arg["--max-events=".len()..].to_string()),
                        "max-events",
                    ));
                }
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                other => usage_and_exit(&format!("unknown argument: {other}")),
            }
        }

        Self {
            mode: mode.unwrap_or_else(|| usage_and_exit("missing required --mode")),
            dataset_path,
            iterations,
            warmup_iters,
            thread_count,
            max_events,
        }
    }
}

struct BreitWignerContext {
    nll: Box<NLL>,
    gen_evaluator: Evaluator,
    evaluator: Evaluator,
    params: Vec<f64>,
    n_data_events: usize,
    n_gen_events: usize,
}

fn main() {
    let args = Args::parse();
    match args.mode {
        Mode::BreitWignerEvaluateLocal
        | Mode::BreitWignerNllValue
        | Mode::BreitWignerProjectionTotal
        | Mode::BreitWignerProjectionGeneratedMc => {
            let ctx = build_breit_wigner_context(&args.dataset_path);
            run_breit_wigner_mode(&args, &ctx);
        }
        Mode::KmatrixValue | Mode::KmatrixDataTerm | Mode::KmatrixMcTerm => {
            let nll = build_kmatrix_nll(&args.dataset_path, args.max_events);
            let params = deterministic_parameter_vector(nll.n_free(), -100.0);
            run_kmatrix_mode(&args, &nll, &params);
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --release -p laddu --bin profile_value_hotspots -- \\\n+         --mode <mode> [--dataset PATH] [--iterations N] [--warmup N] [--threads N] [--max-events N]\n\
         modes:\n\
           bw_evaluate_local\n\
           bw_nll_value\n\
           bw_projection_total\n\
           bw_projection_generated_mc\n\
           kmatrix_value\n\
           kmatrix_data_term\n\
           kmatrix_mc_term"
    );
}

fn usage_and_exit(message: &str) -> ! {
    eprintln!("{message}");
    print_usage();
    process::exit(1);
}

fn parse_mode(value: Option<String>) -> Mode {
    let value = value.unwrap_or_else(|| usage_and_exit("missing value for --mode"));
    Mode::parse(&value).unwrap_or_else(|| usage_and_exit("invalid profiling mode"))
}

fn parse_positive(value: Option<String>, flag: &str) -> usize {
    let parsed = value
        .unwrap_or_else(|| usage_and_exit(&format!("missing value for --{flag}")))
        .parse::<usize>()
        .unwrap_or_else(|_| usage_and_exit(&format!("invalid integer for --{flag}")));
    if parsed == 0 {
        usage_and_exit(&format!("--{flag} must be greater than zero"));
    }
    parsed
}

fn parse_nonnegative(value: Option<String>, flag: &str) -> usize {
    value
        .unwrap_or_else(|| usage_and_exit(&format!("missing value for --{flag}")))
        .parse::<usize>()
        .unwrap_or_else(|_| usage_and_exit(&format!("invalid integer for --{flag}")))
}

fn benchmark_parquet_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(BENCH_DATASET_RELATIVE_PATH)
        .to_string_lossy()
        .into_owned()
}

fn read_benchmark_dataset(dataset_path: &str) -> Arc<Dataset> {
    let options = DatasetReadOptions::default()
        .p4_names(P4_NAMES)
        .aux_names(AUX_NAMES);
    io::read_parquet(dataset_path, &options).expect("benchmark dataset should open")
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

fn build_breit_wigner_partial_wave_model() -> Expression {
    let (angles, polarization, resonance_mass, daughter_1_mass, daughter_2_mass) =
        reaction_variables();

    let z00p = Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization)
        .expect("z00 should construct");
    let z22p = Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization)
        .expect("z22 should construct");
    let bw_f01500 = BreitWigner::new(
        "f0(1500)",
        parameter!("f0_mass", 1.506),
        parameter!("f0_width"),
        0,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
    )
    .expect("f0(1500) should construct");
    let bw_f21525 = BreitWigner::new(
        "f2(1525)",
        parameter!("f2_mass", 1.517),
        parameter!("f2_width"),
        2,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
    )
    .expect("f2(1525) should construct");
    let s0p = Scalar::new("S0+", parameter!("S0+ re")).expect("S0+ should construct");
    let d2p = ComplexScalar::new("D2+", parameter!("D2+ re"), parameter!("D2+ im"))
        .expect("D2+ should construct");

    let pos_re = (&s0p * &bw_f01500 * z00p.real() + &d2p * &bw_f21525 * z22p.real()).norm_sqr();
    let pos_im = (&s0p * &bw_f01500 * z00p.imag() + &d2p * &bw_f21525 * z22p.imag()).norm_sqr();
    pos_re + pos_im
}

fn build_breit_wigner_context(dataset_path: &str) -> BreitWignerContext {
    let dataset = read_benchmark_dataset(dataset_path);
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
    let nll = NLL::new(&model, &ds_data, &ds_accmc, None).expect("unbinned NLL should build");
    let gen_evaluator = model
        .load(&ds_genmc)
        .expect("generated-mc evaluator should build");
    let evaluator = model.load(&ds_data).expect("data evaluator should build");
    let params = vec![100.0, 0.112, 50.0, 50.0, 0.086];
    BreitWignerContext {
        nll,
        gen_evaluator,
        evaluator,
        params,
        n_data_events: ds_data.n_events(),
        n_gen_events: ds_genmc.n_events(),
    }
}

fn build_kmatrix_nll(dataset_path: &str, max_events: Option<usize>) -> Box<NLL> {
    let dataset = read_benchmark_dataset(dataset_path);
    let (ds_data, ds_mc) = if let Some(max_events) = max_events {
        let sampled = sample_dataset(&dataset, KMATRIX_DATASET_SEED, max_events);
        (sampled.clone(), sampled)
    } else {
        (dataset.clone(), dataset)
    };
    let (angles, polarization, resonance_mass, _, _) = reaction_variables();
    let z00p = Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization)
        .expect("z00+ should construct");
    let z00n = Zlm::new("Z00-", 0, 0, Sign::Negative, &angles, &polarization)
        .expect("z00- should construct");
    let z22p = Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization)
        .expect("z22+ should construct");
    let f0p = KopfKMatrixF0::new(
        "f0+",
        [
            [parameter!("f0+ c00 re", 0.0), parameter!("f0+ c00 im", 0.0)],
            [
                parameter!("f0(980)+ re"),
                parameter!("f0(980)+ im_fix", 0.0),
            ],
            [parameter!("f0(1370)+ re"), parameter!("f0(1370)+ im")],
            [parameter!("f0(1500)+ re"), parameter!("f0(1500)+ im")],
            [parameter!("f0(1710)+ re"), parameter!("f0(1710)+ im")],
        ],
        KopfKMatrixF0Channel::PiPi,
        &resonance_mass,
        None,
    )
    .expect("f0+ should construct");
    let a0p = KopfKMatrixA0::new(
        "a0+",
        [
            [parameter!("a0(980)+ re"), parameter!("a0(980)+ im")],
            [parameter!("a0(1450)+ re"), parameter!("a0(1450)+ im")],
        ],
        KopfKMatrixA0Channel::PiEta,
        &resonance_mass,
        None,
    )
    .expect("a0+ should construct");
    let f0n = KopfKMatrixF0::new(
        "f0-",
        [
            [parameter!("f0- c00 re", 0.0), parameter!("f0- c00 im", 0.0)],
            [
                parameter!("f0(980)- re"),
                parameter!("f0(980)- im_fix", 0.0),
            ],
            [parameter!("f0(1370)- re"), parameter!("f0(1370)- im")],
            [parameter!("f0(1500)- re"), parameter!("f0(1500)- im")],
            [parameter!("f0(1710)- re"), parameter!("f0(1710)- im")],
        ],
        KopfKMatrixF0Channel::PiPi,
        &resonance_mass,
        None,
    )
    .expect("f0- should construct");
    let a0n = KopfKMatrixA0::new(
        "a0-",
        [
            [parameter!("a0(980)- re"), parameter!("a0(980)- im")],
            [parameter!("a0(1450)- re"), parameter!("a0(1450)- im")],
        ],
        KopfKMatrixA0Channel::PiEta,
        &resonance_mass,
        None,
    )
    .expect("a0- should construct");
    let f2 = KopfKMatrixF2::new(
        "f2",
        [
            [parameter!("f2(1270) re"), parameter!("f2(1270) im")],
            [parameter!("f2(1525) re"), parameter!("f2(1525) im")],
            [parameter!("f2(1850) re"), parameter!("f2(1850) im")],
            [parameter!("f2(1910) re"), parameter!("f2(1910) im")],
        ],
        KopfKMatrixF2Channel::KKbar,
        &resonance_mass,
        None,
    )
    .expect("f2 should construct");
    let a2 = KopfKMatrixA2::new(
        "a2",
        [
            [parameter!("a2(1320) re"), parameter!("a2(1320) im")],
            [parameter!("a2(1700) re"), parameter!("a2(1700) im")],
        ],
        KopfKMatrixA2Channel::PiEtaPrime,
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
    NLL::new(&expr, &ds_data, &ds_mc, None).expect("k-matrix NLL should build")
}

fn run_breit_wigner_mode(args: &Args, ctx: &BreitWignerContext) {
    let warmup = args.warmup_iters;
    let iters = args.iterations;
    match args.mode {
        Mode::BreitWignerEvaluateLocal => {
            for _ in 0..warmup {
                let values = ctx
                    .evaluator
                    .evaluate(black_box(&ctx.params))
                    .expect("Breit-Wigner evaluation should succeed");
                black_box(values.iter().map(|value| value.re).sum::<f64>());
            }
            let mut sink = 0.0;
            let t0 = Instant::now();
            for _ in 0..iters {
                let values = ctx
                    .evaluator
                    .evaluate(black_box(&ctx.params))
                    .expect("Breit-Wigner evaluation should succeed");
                sink += black_box(values.iter().map(|value| value.re).sum::<f64>());
            }
            eprintln!(
                "mode={} iters={iters} warmup={warmup} n_events={} sink={sink} elapsed={:?}",
                args.mode.as_str(),
                ctx.n_data_events,
                t0.elapsed(),
            );
        }
        Mode::BreitWignerNllValue => {
            for _ in 0..warmup {
                black_box(
                    ctx.nll
                        .evaluate(black_box(&ctx.params))
                        .expect("breit-wigner nll should succeed"),
                );
            }
            let mut sink = 0.0;
            let t0 = Instant::now();
            for _ in 0..iters {
                sink += black_box(
                    ctx.nll
                        .evaluate(black_box(&ctx.params))
                        .expect("breit-wigner nll should succeed"),
                );
            }
            eprintln!(
                "mode={} iters={iters} warmup={warmup} n_events={} sink={sink} elapsed={:?}",
                args.mode.as_str(),
                ctx.n_data_events,
                t0.elapsed(),
            );
        }
        Mode::BreitWignerProjectionTotal => {
            for _ in 0..warmup {
                let weights = ctx
                    .nll
                    .project_weights(black_box(&ctx.params), None)
                    .expect("breit-wigner projection should succeed");
                black_box(weights.iter().sum::<f64>());
            }
            let mut sink = 0.0;
            let t0 = Instant::now();
            for _ in 0..iters {
                let weights = ctx
                    .nll
                    .project_weights(black_box(&ctx.params), None)
                    .expect("breit-wigner projection should succeed");
                sink += black_box(weights.iter().sum::<f64>());
            }
            eprintln!(
                "mode={} iters={iters} warmup={warmup} n_events={} sink={sink} elapsed={:?}",
                args.mode.as_str(),
                ctx.n_data_events,
                t0.elapsed(),
            );
        }
        Mode::BreitWignerProjectionGeneratedMc => {
            for _ in 0..warmup {
                let weights = ctx
                    .nll
                    .project_weights(black_box(&ctx.params), Some(ctx.gen_evaluator.clone()))
                    .expect("generated-mc projection should succeed");
                black_box(weights.iter().sum::<f64>());
            }
            let mut sink = 0.0;
            let t0 = Instant::now();
            for _ in 0..iters {
                let weights = ctx
                    .nll
                    .project_weights(black_box(&ctx.params), Some(ctx.gen_evaluator.clone()))
                    .expect("generated-mc projection should succeed");
                sink += black_box(weights.iter().sum::<f64>());
            }
            eprintln!(
                "mode={} iters={iters} warmup={warmup} n_events={} sink={sink} elapsed={:?}",
                args.mode.as_str(),
                ctx.n_gen_events,
                t0.elapsed(),
            );
        }
        Mode::KmatrixValue | Mode::KmatrixDataTerm | Mode::KmatrixMcTerm => {
            unreachable!("kmatrix mode handled separately")
        }
    }
}

fn run_kmatrix_mode(args: &Args, nll: &NLL, params: &[f64]) {
    let warmup = args.warmup_iters;
    let iters = args.iterations;
    let threads = args.thread_count.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    });
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool should build");

    let run_once = || match args.mode {
        Mode::KmatrixValue => nll
            .evaluate(black_box(params))
            .expect("k-matrix value should succeed"),
        Mode::KmatrixDataTerm => nll
            .profile_data_term_local_value(black_box(params))
            .expect("k-matrix data-term profiling should succeed"),
        Mode::KmatrixMcTerm => nll
            .profile_mc_term_local_value(black_box(params))
            .expect("k-matrix MC-term profiling should succeed"),
        Mode::BreitWignerEvaluateLocal
        | Mode::BreitWignerNllValue
        | Mode::BreitWignerProjectionTotal
        | Mode::BreitWignerProjectionGeneratedMc => unreachable!("non-kmatrix mode"),
    };

    for _ in 0..warmup {
        black_box(pool.install(run_once));
    }
    let mut sink = 0.0;
    let t0 = Instant::now();
    for _ in 0..iters {
        sink += black_box(pool.install(run_once));
    }
    eprintln!(
        "mode={} iters={iters} warmup={warmup} threads={threads} n_events={} sink={sink} elapsed={:?}",
        args.mode.as_str(),
        nll.data_evaluator.dataset.n_events(),
        t0.elapsed(),
    );
}
