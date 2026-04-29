use std::collections::HashMap;

use laddu::generation::{
    EventGenerator, FinalStateParticle, GenComposite, GenFinalState, GenInitialState, GenReaction,
    InitialStateParticle, MandelstamTDistribution, Reconstruction,
};

#[test]
fn umbrella_crate_exposes_generation_api() {
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
    let kshort1 = FinalStateParticle::new(
        "kshort1",
        GenFinalState::new(0.497611),
        Reconstruction::Reconstructed {
            p4_names: vec!["kshort1".to_string()],
        },
    );
    let kshort2 = FinalStateParticle::new(
        "kshort2",
        GenFinalState::new(0.497611),
        Reconstruction::Reconstructed {
            p4_names: vec!["kshort2".to_string()],
        },
    );
    let kk = FinalStateParticle::composite("kk", GenComposite::new(1.1, 1.6), (&kshort1, &kshort2));
    let recoil = FinalStateParticle::new(
        "recoil",
        GenFinalState::new(0.938272),
        Reconstruction::Reconstructed {
            p4_names: vec!["recoil".to_string()],
        },
    );
    let reaction = GenReaction::two_to_two(
        beam,
        target,
        kk,
        recoil,
        MandelstamTDistribution::Exponential { slope: 0.1 },
    );
    let generator = EventGenerator::new(reaction, HashMap::new(), Some(12345));
    let dataset = generator.generate_dataset(4).unwrap();

    assert_eq!(dataset.n_events(), 4);
    assert_eq!(
        dataset.metadata().p4_names(),
        &["beam", "target", "kk", "kshort1", "kshort2", "recoil"]
    );
}
