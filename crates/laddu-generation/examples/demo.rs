use std::{collections::HashMap, sync::Arc};

use fastrand::Rng;
use laddu::{parameter, BreitWigner, LadduResult};
use laddu_core::{traits::Variable, Particle, Reaction};
use laddu_generation::{
    distributions::{LadduGenRngExt, MandelstamTDistribution},
    topology::{
        EventGenerator, FinalStateParticle, GenComposite, GenFinalState, GenInitialState,
        GenReaction, InitialStateParticle, Reconstruction,
    },
};

fn demo_gen_reaction() -> GenReaction {
    let beam = InitialStateParticle::new(
        "beam",
        GenInitialState::beam_with_fixed_energy(0.0, 8.0),
        Reconstruction::Reconstructed {
            p4_names: vec!["beam".to_string()],
        },
    );
    let target = InitialStateParticle::new(
        "target",
        GenInitialState::target(0.938272),
        Reconstruction::Missing,
    );
    let ks1 = FinalStateParticle::new(
        "kshort1",
        GenFinalState::new(0.497611),
        Reconstruction::Reconstructed {
            p4_names: vec!["kshort1".to_string()],
        },
    );
    let ks2 = FinalStateParticle::new(
        "kshort2",
        GenFinalState::new(0.497611),
        Reconstruction::Reconstructed {
            p4_names: vec!["kshort2".to_string()],
        },
    );
    let kk = FinalStateParticle::composite("kk", GenComposite::new(1.1, 1.6), (&ks1, &ks2));
    let recoil = FinalStateParticle::new(
        "recoil",
        GenFinalState::new(0.938272),
        Reconstruction::Reconstructed {
            p4_names: vec!["recoil".to_string()],
        },
    );
    let tdist = MandelstamTDistribution::Exponential { slope: 0.1 };
    GenReaction::two_to_two(beam, target, kk, recoil, tdist)
}

fn main() -> LadduResult<()> {
    let gen_reaction = demo_gen_reaction();
    let generator = EventGenerator::new(gen_reaction, HashMap::new(), Some(12345));
    let n_events = 1_000_000;
    let dataset = Arc::new(generator.generate_dataset(n_events)?);
    let beam = Particle::stored("beam");
    let target = Particle::missing("target");
    let ks1 = Particle::stored("kshort1");
    let ks2 = Particle::stored("kshort2");
    let kk = Particle::composite("kk", (&ks1, &ks2))?;
    let recoil = Particle::stored("recoil");
    let reaction = Reaction::two_to_two(&beam, &target, &kk, &recoil)?;

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
