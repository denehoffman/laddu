use std::{collections::HashMap, sync::Arc};

use fastrand::Rng;
use laddu::{parameter, BreitWigner, LadduResult};
use laddu_core::traits::Variable;
use laddu_generation::{
    distributions::{LadduGenRngExt, MandelstamTDistribution},
    topology::{
        CompositeGenerator, EventGenerator, GeneratedParticle, GeneratedReaction, InitialGenerator,
        Reconstruction, StableGenerator,
    },
};

fn demo_gen_reaction() -> LadduResult<GeneratedReaction> {
    let beam = GeneratedParticle::initial(
        "beam",
        InitialGenerator::beam_with_fixed_energy(0.0, 8.0),
        Reconstruction::Stored,
    );
    let target = GeneratedParticle::initial(
        "target",
        InitialGenerator::target(0.938272),
        Reconstruction::Missing,
    );
    let ks1 = GeneratedParticle::stable(
        "kshort1",
        StableGenerator::new(0.497611),
        Reconstruction::Stored,
    );
    let ks2 = GeneratedParticle::stable(
        "kshort2",
        StableGenerator::new(0.497611),
        Reconstruction::Stored,
    );
    let kk = GeneratedParticle::composite(
        "kk",
        CompositeGenerator::new(1.1, 1.6),
        (&ks1, &ks2),
        Reconstruction::Composite,
    );
    let recoil = GeneratedParticle::stable(
        "recoil",
        StableGenerator::new(0.938272),
        Reconstruction::Stored,
    );
    let tdist = MandelstamTDistribution::Exponential { slope: 0.1 };
    GeneratedReaction::two_to_two(beam, target, kk, recoil, tdist)
}

fn main() -> LadduResult<()> {
    let gen_reaction = demo_gen_reaction()?;
    let reaction = gen_reaction.reconstructed_reaction()?;
    let generator = EventGenerator::new(gen_reaction, HashMap::new(), Some(12345));
    let n_events = 1_000_000;
    let dataset = Arc::new(generator.generate_dataset(n_events)?);

    let bw = BreitWigner::new(
        "f0_1500",
        parameter!("mass", 1.5),
        parameter!("width", 0.15),
        0,
        &reaction.mass("kshort1"),
        &reaction.mass("kshort2"),
        &reaction.mass("kk"),
    )?;

    let model = bw.norm_sqr();
    let eval = model.load(&dataset)?;
    let weights: Vec<f64> = eval.evaluate(&[])?.iter().map(|v| v.re).collect();

    let mut rng = Rng::with_seed(0);
    let accept = weights.iter().map(|w| *w > rng.uniform(0.0, 25.0));

    println!(
        "{{\"values\": [{}], \"bins\": 40, \"min\": 0.9, \"max\": 2.0}}",
        reaction
            .mass("kk")
            .value_on(&dataset)
            .unwrap()
            .iter()
            .zip(accept)
            .filter_map(|(x, acc)| if acc { Some(x.to_string()) } else { None })
            // .map(|(x, _)| x.to_string())
            .collect::<Vec<String>>()
            .join(", ")
    );
    Ok(())
}
