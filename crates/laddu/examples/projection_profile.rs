//! Runs the representative repeated-projection workload under selectable execution policies.

#[path = "../benches/support/projection.rs"]
mod projection;

use std::{env, fs, path::PathBuf, time::Instant};

use laddu::prelude::ThreadPolicy;
use projection::{ProjectionFixture, ProjectionTarget, Storage};

const BASELINE_DRAWS: usize = 20;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let options = Options::parse()?;
    let target = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    let output_dir = target.join("projection-profile");
    fs::create_dir_all(&output_dir)?;

    let mut summary = String::from("workflow,storage,threads,events,draws,elapsed_seconds\n");
    for storage in &options.storage {
        for thread_case in &options.threads {
            let projection_set_fixture = ProjectionFixture::new(
                options.events,
                options.draws,
                *storage,
                thread_case.policy,
            )?;
            let started = Instant::now();
            let projections =
                projection_set_fixture.evaluate_projection_set(ProjectionTarget::Combined, 4)?;
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(projections.len(), 4);
            summary.push_str(&format!(
                "projection-set,{},{},{},{},{elapsed:.9}\n",
                storage.label(),
                thread_case.label,
                options.events,
                options.draws
            ));

            let repeated_fixture = ProjectionFixture::new(
                options.events,
                BASELINE_DRAWS,
                *storage,
                thread_case.policy,
            )?;
            let started = Instant::now();
            let projections =
                repeated_fixture.evaluate_differentials(ProjectionTarget::Combined, 4)?;
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(projections.len(), 4);
            summary.push_str(&format!(
                "repeated-differentials,{},{},{},{BASELINE_DRAWS},{elapsed:.9}\n",
                storage.label(),
                thread_case.label,
                options.events
            ));
        }
    }
    let output = output_dir.join("summary.csv");
    fs::write(&output, summary)?;
    println!("wrote {}", output.display());
    Ok(())
}

struct Options {
    events: usize,
    draws: usize,
    storage: Vec<Storage>,
    threads: Vec<ThreadCase>,
}

struct ThreadCase {
    label: String,
    policy: ThreadPolicy,
}

impl ThreadCase {
    fn new(label: impl Into<String>, policy: ThreadPolicy) -> Self {
        Self {
            label: label.into(),
            policy,
        }
    }
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut events = 25_000;
        let mut draws = 200;
        let mut storage = "all".to_owned();
        let mut threads = "all".to_owned();
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value after {argument}"))?;
            match argument.as_str() {
                "--events" => events = parse_usize("events", &value)?,
                "--draws" => draws = parse_usize("draws", &value)?,
                "--storage" => storage = value,
                "--threads" => threads = value,
                _ => return Err(format!("unknown option {argument}")),
            }
        }
        let storage = match storage.as_str() {
            "resident" => vec![Storage::Resident],
            "streaming" => vec![Storage::Streaming],
            "all" => vec![Storage::Resident, Storage::Streaming],
            value => {
                return Err(format!(
                    "storage must be resident, streaming, or all; got {value}"
                ));
            }
        };
        let available = num_cpus::get().max(1);
        let threads = match threads.as_str() {
            "serial" => vec![ThreadCase::new("serial", ThreadPolicy::Serial)],
            "auto" => vec![ThreadCase::new("auto", ThreadPolicy::Auto)],
            "all" => vec![
                ThreadCase::new("serial", ThreadPolicy::Serial),
                ThreadCase::new(format!("fixed-{available}"), ThreadPolicy::Fixed(available)),
                ThreadCase::new("auto", ThreadPolicy::Auto),
            ],
            value if value.starts_with("fixed:") => {
                let count = parse_usize("fixed thread count", &value[6..])?;
                vec![ThreadCase::new(
                    format!("fixed-{count}"),
                    ThreadPolicy::Fixed(count),
                )]
            }
            value => {
                return Err(format!(
                    "threads must be serial, fixed:N, auto, or all; got {value}"
                ));
            }
        };
        Ok(Self {
            events,
            draws,
            storage,
            threads,
        })
    }
}

fn parse_usize(name: &str, value: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))?;
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}
