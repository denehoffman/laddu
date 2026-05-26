use std::collections::HashMap;

use laddu::{
    generation::{
        CompositeGenerator, EventGenerator, GeneratedParticle, GeneratedReaction, InitialGenerator,
        MandelstamTDistribution, Reconstruction, StableGenerator,
    },
    j, l, m, s, J, L, M, S,
};

#[test]
fn umbrella_crate_exposes_quantum_macros() {
    assert_eq!(j!(3 / 2), J::half(3));
    assert_eq!(s!(1), S::int(1));
    assert_eq!(m!(-1 / 2), M::half(-1));
    assert_eq!(l!(2), L::int(2));
    assert_eq!(laddu::quantum::j!(1), J::int(1));
}

#[test]
fn umbrella_crate_exposes_generation_api() {
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
    let kshort1 = GeneratedParticle::stable(
        "kshort1",
        StableGenerator::new(0.497611),
        Reconstruction::Stored,
    );
    let kshort2 = GeneratedParticle::stable(
        "kshort2",
        StableGenerator::new(0.497611),
        Reconstruction::Stored,
    );
    let kk = GeneratedParticle::composite(
        "kk",
        CompositeGenerator::new(1.1, 1.6),
        (&kshort1, &kshort2),
        Reconstruction::Composite,
    );
    let recoil = GeneratedParticle::stable(
        "recoil",
        StableGenerator::new(0.938272),
        Reconstruction::Stored,
    );
    let reaction = GeneratedReaction::two_to_two(
        beam,
        target,
        kk,
        recoil,
        MandelstamTDistribution::Exponential { slope: 0.1 },
    )
    .unwrap();
    let reconstructed = reaction.reconstructed_reaction().unwrap();
    let generator = EventGenerator::new(reaction, HashMap::new(), Some(12345));
    let dataset = generator.generate_dataset(4).unwrap();

    assert_eq!(dataset.n_events(), 4);
    assert_eq!(
        dataset.metadata().p4_names(),
        &["beam", "target", "kk", "kshort1", "kshort2", "recoil"]
    );
    assert_eq!(
        reconstructed.decay("kk").unwrap().daughters(),
        ["kshort1", "kshort2"]
    );
}
