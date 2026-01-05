//! Example harness for profiling Laddu with tools such as Samply.
//! Build with `cargo run -p laddu --example profile_kmatrix -- --help`.

use std::{env, hint::black_box, process, time::Instant};

use laddu::{
    amplitudes::{
        constant,
        kmatrix::{KopfKMatrixA0, KopfKMatrixA2, KopfKMatrixF0, KopfKMatrixF2},
        parameter,
        zlm::Zlm,
    },
    data::DatasetReadOptions,
    extensions::NLL,
    io,
    traits::LikelihoodTerm,
    utils::{
        enums::{Frame, Sign},
        variables::{Angles, Mass, Polarization, Topology},
    },
};
use rayon::{prelude::*, ThreadPoolBuilder};

fn main() {
    let args = Args::parse();
    let nll = build_kmatrix_nll(&args.dataset_path);
    let pool = ThreadPoolBuilder::new()
        .num_threads(args.thread_count.unwrap_or_else(num_cpus::get))
        .build()
        .expect("failed to build rayon pool");
    let nll_ref: &NLL = &nll;
    println!(
        "Profiling {} iterations ({:?}) on {} threads...",
        args.iterations,
        args.mode,
        pool.current_num_threads()
    );
    let start = Instant::now();
    pool.install(|| {
        let nll = nll_ref;
        (0..args.iterations)
            .into_par_iter()
            .for_each(|iteration| run_iteration(nll, iteration, args.mode));
    });
    let elapsed = start.elapsed();
    println!(
        "Finished {} iterations in {:.2?} ({:.4?}/iter)",
        args.iterations,
        elapsed,
        elapsed / args.iterations as u32
    );
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Value,
    Gradient,
    ValueAndGradient,
}

impl Mode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "value" => Some(Self::Value),
            "gradient" => Some(Self::Gradient),
            "both" => Some(Self::ValueAndGradient),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct Args {
    dataset_path: String,
    iterations: usize,
    thread_count: Option<usize>,
    mode: Mode,
}

impl Args {
    fn parse() -> Self {
        const DEFAULT_DATASET: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/bench.parquet");
        let mut dataset_path = DEFAULT_DATASET.to_string();
        let mut iterations = 512usize;
        let mut thread_count = None;
        let mut mode = Mode::ValueAndGradient;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--dataset" => {
                    dataset_path = args
                        .next()
                        .unwrap_or_else(|| usage_and_exit("missing dataset path"))
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
                "--threads" => {
                    thread_count = Some(parse_positive(args.next(), "threads"));
                }
                _ if arg.starts_with("--threads=") => {
                    thread_count = Some(parse_positive(
                        Some(arg["--threads=".len()..].to_string()),
                        "threads",
                    ));
                }
                "--mode" => {
                    mode = parse_mode(args.next());
                }
                _ if arg.starts_with("--mode=") => {
                    mode = parse_mode(Some(arg["--mode=".len()..].to_string()));
                }
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                other => {
                    usage_and_exit(&format!("unknown argument: {other}"));
                }
            }
        }
        Args {
            dataset_path,
            iterations,
            thread_count,
            mode,
        }
    }
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

fn parse_mode(value: Option<String>) -> Mode {
    let value = value.unwrap_or_else(|| usage_and_exit("missing value for --mode"));
    Mode::parse(&value)
        .unwrap_or_else(|| usage_and_exit("mode must be one of: value, gradient, both"))
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run -p laddu --example profile_kmatrix [--dataset PATH] [--iterations N] \
         [--threads N] [--mode value|gradient|both]"
    );
}

fn usage_and_exit(message: &str) -> ! {
    eprintln!("{message}");
    print_usage();
    process::exit(1);
}

fn build_kmatrix_nll(dataset_path: &str) -> Box<NLL> {
    let p4_names = ["beam", "proton", "kshort1", "kshort2"];
    let aux_names = ["pol_magnitude", "pol_angle"];
    let options = DatasetReadOptions::default()
        .p4_names(p4_names)
        .aux_names(aux_names);
    let ds_data = io::read_parquet(dataset_path, &options).expect("failed to load data sample");
    let ds_mc = io::read_parquet(dataset_path, &options).expect("failed to load MC sample");
    let topology = Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton");
    let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
    let polarization = Polarization::new(topology.clone(), "pol_magnitude", "pol_angle");
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
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
        0,
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
        0,
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
        0,
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
        0,
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
        2,
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
        2,
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
    NLL::new(&expr, &ds_data, &ds_mc).unwrap()
}

fn run_iteration(nll: &NLL, iteration: usize, mode: Mode) {
    let params = generate_parameters(iteration, nll.n_free());
    match mode {
        Mode::Value => {
            black_box(nll.evaluate(&params));
        }
        Mode::Gradient => {
            black_box(nll.evaluate_gradient(&params));
        }
        Mode::ValueAndGradient => {
            let value = nll.evaluate(&params);
            let gradient = nll.evaluate_gradient(&params);
            black_box((value, gradient));
        }
    }
}

fn generate_parameters(iteration: usize, len: usize) -> Vec<f64> {
    (0..len)
        .map(|idx| {
            let seed = iteration as f64 + idx as f64 * 0.5;
            (seed.sin() + seed.cos()) * 0.5
        })
        .collect()
}
