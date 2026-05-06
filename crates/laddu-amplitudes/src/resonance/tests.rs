use std::sync::Arc;

use approx::assert_relative_eq;
use laddu_core::{
    amplitudes::IntoTags,
    data::test_dataset,
    parameter,
    reaction::{Particle, Reaction},
    variables::Mass,
    Channel,
};

use super::{BreitWigner, BreitWignerNonRelativistic, Flatte, PhaseSpaceFactor, Voigt};

fn test_phase_space_expression(tags: impl IntoTags, channel: Channel) -> laddu_core::Expression {
    let beam = Particle::stored("beam");
    let target = Particle::missing("target");
    let kshort1 = Particle::stored("kshort1");
    let kshort2 = Particle::stored("kshort2");
    let kk = Particle::composite("kk", (&kshort1, &kshort2)).unwrap();
    let proton = Particle::stored("proton");
    let reaction = Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap();
    let decay = reaction.decay("kk").unwrap();
    let recoil_mass = reaction.mass("proton");
    let daughter_1_mass = decay.daughter_1_mass();
    let daughter_2_mass = decay.daughter_2_mass();
    let resonance_mass = decay.parent_mass();
    let mandelstam_s = reaction.mandelstam(channel).unwrap();
    PhaseSpaceFactor::new(
        tags,
        &recoil_mass,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
        &mandelstam_s,
    )
    .unwrap()
}

#[test]
fn test_bw_evaluation() {
    let dataset = Arc::new(test_dataset());
    let daughter_1_mass = Mass::new(["kshort1"]);
    let daughter_2_mass = Mass::new(["kshort2"]);
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let amp = BreitWigner::new(
        "bw",
        parameter!("mass"),
        parameter!("width"),
        2,
        &daughter_1_mass,
        &daughter_2_mass,
        &resonance_mass,
    )
    .unwrap();
    let evaluator = amp.load(&dataset).unwrap();

    let result = evaluator.evaluate(&[1.5, 0.3]).unwrap();

    assert_relative_eq!(result[0].re, 1.4308791652435877);
    assert_relative_eq!(result[0].im, 1.3839522217669178);
}

#[test]
fn test_bw_nonrel_evaluation() {
    let dataset = Arc::new(test_dataset());
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let amp = BreitWignerNonRelativistic::new(
        "bw",
        parameter!("mass"),
        parameter!("width"),
        &resonance_mass,
    )
    .unwrap();
    let evaluator = amp.load(&dataset).unwrap();

    let result = evaluator.evaluate(&[1.5, 0.3]).unwrap();

    assert_relative_eq!(result[0].re, 1.084721431628924);
    assert_relative_eq!(result[0].im, 1.3518336007116172);
}

#[test]
fn test_flatte_evaluation() {
    let dataset = Arc::new(test_dataset());
    let daughter_1_mass = Mass::new(["kshort1"]);
    let daughter_2_mass = Mass::new(["kshort2"]);
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let amp = Flatte::new(
        "flatte",
        parameter!("mass"),
        parameter!("g_obs"),
        parameter!("g_alt"),
        (&daughter_1_mass, &daughter_2_mass),
        (0.1349768, 0.547862),
        &resonance_mass,
    )
    .unwrap();
    let evaluator = amp.load(&dataset).unwrap();

    let result = evaluator.evaluate(&[0.98, 0.7, 0.2]).unwrap();

    assert_relative_eq!(result[0].re, -0.7338320342780681);
    assert_relative_eq!(result[0].im, 0.5018145529787819);
}

#[test]
fn test_voigt_sqrt_profile_evaluation() {
    let dataset = Arc::new(test_dataset());
    let resonance_mass = Mass::new(["kshort1", "kshort2"]);
    let amp = Voigt::new(
        "voigt",
        parameter!("mass"),
        parameter!("width"),
        parameter!("sigma"),
        &resonance_mass,
    )
    .unwrap();
    let evaluator = amp.load(&dataset).unwrap();

    let result = evaluator.evaluate(&[0.98, 0.08, 0.02]).unwrap();

    assert_relative_eq!(result[0].re, 0.2857389147779551);
    assert_relative_eq!(result[0].im, 0.0);
}

#[test]
fn test_phase_space_factor_evaluation() {
    let dataset = Arc::new(test_dataset());
    let expr = test_phase_space_expression("kappa", Channel::S);
    let evaluator = expr.load(&dataset).unwrap();

    let result = evaluator.evaluate(&[]).unwrap();

    assert_relative_eq!(result[0].re, 7.028417575882146e-5);
    assert_relative_eq!(result[0].im, 0.0);
}
