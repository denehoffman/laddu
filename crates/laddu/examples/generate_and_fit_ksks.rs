//! Generate a K_S K_S pseudo-dataset and recover its injected wave coupling.
//!
//! Run with
//! `cargo run --release -p laddu --example generate_and_fit_ksks --features generation,fit -- [data events] [normalization events] [projection JSON] [cpu|jit|gpu]`.

mod common;

use std::{error::Error, io, path::PathBuf};

use common::{
    closure::{ClosureConfig, run_closure, wrapped_phase_residual},
    ksks::{F2_MAGNITUDE_TRUTH, F2_PHASE_TRUTH, print_report},
};
use laddu::prelude::{
    CpuOptions, Device, Execution, ExecutionOptions, GpuBackend, GpuOptions, JitPolicy, Precision,
    ThreadPolicy,
};

#[derive(Clone, Copy, Debug)]
enum Backend {
    Cpu,
    Jit,
    Gpu,
}

impl Backend {
    fn parse(argument: Option<String>) -> Result<Self, io::Error> {
        match argument.as_deref().unwrap_or("cpu") {
            "cpu" => Ok(Self::Cpu),
            "jit" => Ok(Self::Jit),
            "gpu" => Ok(Self::Gpu),
            backend => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown backend `{backend}`; expected cpu, jit, or gpu"),
            )),
        }
    }

    fn execution(self) -> Result<Execution, Box<dyn Error>> {
        let options = match self {
            Self::Cpu => ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Auto,
                    jit: JitPolicy::Disabled,
                }),
                precision: Precision::F64,
                ..ExecutionOptions::default()
            },
            Self::Jit => ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Auto,
                    jit: JitPolicy::Enabled,
                }),
                precision: Precision::F64,
                ..ExecutionOptions::default()
            },
            Self::Gpu => ExecutionOptions {
                device: Device::Gpu(GpuOptions {
                    backend: GpuBackend::Wgpu,
                    ..GpuOptions::default()
                }),
                precision: Precision::F32,
                ..ExecutionOptions::default()
            },
        };
        Ok(Execution::local(options)?)
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU interpreter",
            Self::Jit => "CPU JIT",
            Self::Gpu => "WGPU",
        }
    }
}

fn event_count(
    argument: Option<String>,
    default: usize,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    let Some(argument) = argument else {
        return Ok(default);
    };
    let value = argument.parse::<usize>()?;
    if value == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} must be nonzero"),
        )
        .into());
    }
    Ok(value)
}

fn main() -> Result<(), Box<dyn Error>> {
    let defaults = ClosureConfig::default();
    let mut arguments = std::env::args().skip(1);
    let data_events = event_count(arguments.next(), defaults.data_events, "data events")?;
    let normalization_events = event_count(
        arguments.next(),
        defaults.normalization_events,
        "normalization events",
    )?;
    let projection_path = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/ksks-closure.json"));
    let backend = Backend::parse(arguments.next())?;
    if arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected at most four arguments: data events, normalization events, projection JSON, and backend",
        )
        .into());
    }

    println!("execution backend: {}", backend.name());
    let result = run_closure(
        ClosureConfig {
            data_events,
            normalization_events,
            ..defaults
        },
        backend.execution()?,
    )?;

    print_report("pseudo-data", &result.data_report);
    print_report("normalization MC", &result.normalization_report);
    println!(
        "timings: data {:?}, normalization MC {:?}, likelihood preparation {:?}, fit {:?}, projection {:?}",
        result.data_generation_time,
        result.normalization_generation_time,
        result.likelihood_preparation_time,
        result.fit_time,
        result.projection_time,
    );
    println!(
        "NLL: initial {:.8e}, final {:.8e}; initial gradient norm {:.8e}",
        result.initial_nll,
        result.fit.value(),
        result
            .initial_gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
    );

    let magnitude = result
        .fitted("f2_magnitude")
        .ok_or("fit did not return f2_magnitude")?;
    let phase = result
        .fitted("f2_phase")
        .ok_or("fit did not return f2_phase")?;
    println!("parameter       truth        fitted       residual");
    println!(
        "f2_magnitude  {F2_MAGNITUDE_TRUTH:>10.6}  {magnitude:>12.6}  {:>12.6}",
        magnitude - F2_MAGNITUDE_TRUTH,
    );
    println!(
        "f2_phase      {F2_PHASE_TRUTH:>10.6}  {phase:>12.6}  {:>12.6}",
        wrapped_phase_residual(phase, F2_PHASE_TRUTH),
    );
    println!();
    println!("{}", result.fit.raw);
    result.projection.write_json(&projection_path)?;
    println!("wrote projection data to {}", projection_path.display());
    println!(
        "plot with: uv run crates/laddu/examples/plot_ksks.py {}",
        projection_path.display()
    );
    Ok(())
}
