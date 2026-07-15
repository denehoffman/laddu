//! Generate gamma p -> X p, X -> K_S K_S samples in Parquet and ROOT formats.
//!
//! Run with
//! `cargo run -p laddu --example generate_ksks --features generation -- [output directory]`.

#[allow(dead_code)]
#[path = "common/ksks.rs"]
mod ksks;

use std::{error::Error, path::PathBuf};

use ksks::{ksks_channel, ksks_intensities, print_report, truth_parameters};
use laddu::prelude::*;

const EVENTS: usize = 1_000_000;
const SEED: u64 = 0x4b53_4b53;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/ksks-generation"));
    std::fs::create_dir_all(&output)?;

    let channel = ksks_channel()?;
    let compiled = CompiledModel::from_expr(&ksks_intensities(&channel)?.coherent)?;
    let evaluator = ModelEvaluator::prepare(
        &compiled,
        truth_parameters(&compiled)?,
        &Execution::default(),
    )?;
    let generator = ChannelGenerator::new(channel)?;

    let weighted = WeightedConfig {
        events: EVENTS,
        batch_size: 256,
        seed: SEED,
        diagnostics: true,
    };
    let unweighted = UnweightedConfig {
        events: EVENTS,
        max_proposals: None,
        batch_size: 2_048,
        seed: SEED.wrapping_add(1),
        diagnostics: true,
        envelope_overflow: EnvelopeOverflow::Grow { safety_factor: 1.5 },
    };
    let envelope = EnvelopeMode::Pilot {
        proposals: 50_000,
        safety_factor: 2.0,
    };

    let mut weighted_parquet = ParquetSink::create(output.join("ksks_weighted.parquet"));
    let report =
        generator.generate_weighted_to(weighted, Some(&evaluator), &mut weighted_parquet)?;
    print_report("weighted Parquet", &report);

    let mut weighted_root = RootSink::builder(output.join("ksks_weighted.root"))
        .tree("events")
        .build();
    let report = generator.generate_weighted_to(weighted, Some(&evaluator), &mut weighted_root)?;
    print_report("weighted ROOT", &report);

    let mut unweighted_parquet = ParquetSink::create(output.join("ksks_unweighted.parquet"));
    let report = generator.generate_unweighted_to(
        unweighted,
        &evaluator,
        envelope,
        &mut unweighted_parquet,
    )?;
    print_report("unweighted Parquet", &report);

    let mut unweighted_root = RootSink::builder(output.join("ksks_unweighted.root"))
        .tree("events")
        .build();
    let report =
        generator.generate_unweighted_to(unweighted, &evaluator, envelope, &mut unweighted_root)?;
    print_report("unweighted ROOT", &report);

    println!("wrote four {EVENTS}-event samples to {}", output.display());
    Ok(())
}
