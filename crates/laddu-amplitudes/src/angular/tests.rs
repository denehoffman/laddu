use std::sync::Arc;

use approx::assert_relative_eq;
use laddu_core::{
    data::test_dataset,
    math::{BarrierKind, WignerDMatrix, QR_DEFAULT},
    parameter,
    reaction::{Particle, Reaction},
    traits::Variable,
    Frame,
};

use super::{
    barrier::BlattWeisskopf,
    constants::ClebschGordan,
    sdme::{PhotonHelicity, PhotonPolarization, PhotonSDME},
    wigner::{DecayAmplitudeExt, WignerD},
};
use crate::scalar::Scalar;

fn reaction_context() -> (Reaction, Particle, Particle) {
    let beam = Particle::stored("beam");
    let target = Particle::missing("target");
    let kshort1 = Particle::stored("kshort1");
    let kshort2 = Particle::stored("kshort2");
    let kk = Particle::composite("kk", (&kshort1, &kshort2)).unwrap();
    let proton = Particle::stored("proton");
    (
        Reaction::two_to_two(&beam, &target, &kk, &proton).unwrap(),
        kk,
        kshort1,
    )
}

#[test]
fn wigner_d_matches_core_function() {
    let dataset = Arc::new(test_dataset());
    let (reaction, _, _) = reaction_context();
    let decay = reaction.decay("kk").unwrap();
    let angles = decay.angles("kshort1", Frame::Helicity).unwrap();
    let expr = WignerD::new(
        "d",
        laddu_core::AngularMomentum::from_twice(2),
        laddu_core::AngularMomentumProjection::from_twice(2),
        laddu_core::AngularMomentumProjection::from_twice(0),
        &angles,
    )
    .unwrap();
    let evaluator = expr.load(&dataset).unwrap();
    let event = dataset.event_local(0).unwrap();
    let mut costheta = angles.costheta.clone();
    let mut phi = angles.phi.clone();
    costheta.bind(dataset.metadata()).unwrap();
    phi.bind(dataset.metadata()).unwrap();
    let expected = WignerDMatrix::new(2, 2, 0).D(
        event.evaluate(&phi),
        event.evaluate(&costheta).clamp(-1.0, 1.0).acos(),
        0.0,
    );
    let value = evaluator.evaluate(&[]).unwrap()[0];

    assert_relative_eq!(value.re, expected.re);
    assert_relative_eq!(value.im, expected.im);
}

#[test]
fn clebsch_gordan_constant_matches_core_function() {
    let dataset = Arc::new(test_dataset());
    let expr = ClebschGordan::new(
        "cg",
        laddu_core::AngularMomentum::from_twice(1),
        laddu_core::AngularMomentumProjection::from_twice(1),
        laddu_core::AngularMomentum::from_twice(1),
        laddu_core::AngularMomentumProjection::from_twice(-1),
        laddu_core::AngularMomentum::from_twice(2),
        laddu_core::AngularMomentumProjection::from_twice(0),
    )
    .unwrap();
    let value = expr.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];

    assert_relative_eq!(value.re, 1.0 / 2.0_f64.sqrt());
    assert_relative_eq!(value.im, 0.0);
}

#[test]
fn photon_sdme_unpolarized_is_diagonal() {
    let dataset = Arc::new(test_dataset());
    let diagonal = PhotonSDME::new(
        "rho_diag",
        PhotonPolarization::Unpolarized,
        PhotonHelicity::new(1).unwrap(),
        PhotonHelicity::new(1).unwrap(),
    )
    .unwrap();
    let off_diagonal = PhotonSDME::new(
        "rho_off",
        PhotonPolarization::Unpolarized,
        PhotonHelicity::new(1).unwrap(),
        PhotonHelicity::new(-1).unwrap(),
    )
    .unwrap();

    assert_relative_eq!(
        diagonal.load(&dataset).unwrap().evaluate(&[]).unwrap()[0].re,
        0.5
    );
    assert_relative_eq!(
        off_diagonal.load(&dataset).unwrap().evaluate(&[]).unwrap()[0].norm(),
        0.0
    );
}

#[test]
fn blatt_weisskopf_accepts_reaction_decay_context() {
    let beam = laddu_core::Particle::stored("beam");
    let target = laddu_core::Particle::stored("target");
    let k1 = laddu_core::Particle::stored("kshort1");
    let k2 = laddu_core::Particle::stored("kshort2");
    let x = laddu_core::Particle::composite("x", (&k1, &k2)).unwrap();
    let recoil = laddu_core::Particle::stored("proton");
    let reaction = laddu_core::Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
    let decay = reaction.decay("x").unwrap();
    let expr = BlattWeisskopf::new(
        "b",
        &decay,
        laddu_core::OrbitalAngularMomentum::integer(2),
        1.5,
        QR_DEFAULT,
        laddu_core::math::Sheet::Physical,
        BarrierKind::Full,
    )
    .unwrap();
    let dataset = Arc::new(test_dataset());
    let event = dataset.event_local(0).unwrap();
    let value = expr.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];
    let expected = laddu_core::math::blatt_weisskopf_m(
        event.get_p4_sum(["kshort1", "kshort2"]).unwrap().m(),
        event.p4("kshort1").unwrap().m(),
        event.p4("kshort2").unwrap().m(),
        2,
        QR_DEFAULT,
        laddu_core::math::Sheet::Physical,
        BarrierKind::Full,
    ) / laddu_core::math::blatt_weisskopf_m(
        1.5,
        event.p4("kshort1").unwrap().m(),
        event.p4("kshort2").unwrap().m(),
        2,
        QR_DEFAULT,
        laddu_core::math::Sheet::Physical,
        BarrierKind::Full,
    );

    assert_relative_eq!(value.re, expected.re);
    assert_relative_eq!(value.im, expected.im);
}

#[test]
fn helicity_factor_matches_conjugated_wigner_d() {
    let dataset = Arc::new(test_dataset());
    let (reaction, _, _) = reaction_context();
    let decay = reaction.decay("kk").unwrap();
    let factor = DecayAmplitudeExt::helicity_factor(
        &decay,
        "h",
        laddu_core::AngularMomentum::integer(2),
        laddu_core::AngularMomentumProjection::integer(1),
        "kshort1",
        laddu_core::AngularMomentumProjection::integer(1),
        laddu_core::AngularMomentumProjection::integer(0),
        Frame::Helicity,
    )
    .unwrap();
    let angles = decay.angles("kshort1", Frame::Helicity).unwrap();
    let explicit = WignerD::new(
        "d",
        laddu_core::AngularMomentum::integer(2),
        laddu_core::AngularMomentumProjection::integer(1),
        laddu_core::AngularMomentumProjection::integer(1),
        &angles,
    )
    .unwrap()
    .conj();

    let factor_value = factor.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];
    let explicit_value = explicit.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];

    assert_relative_eq!(factor_value.re, explicit_value.re);
    assert_relative_eq!(factor_value.im, explicit_value.im);
}

#[test]
fn canonical_factor_matches_explicit_product() {
    let dataset = Arc::new(test_dataset());
    let (reaction, _, _) = reaction_context();
    let decay = reaction.decay("kk").unwrap();
    let factor = DecayAmplitudeExt::canonical_factor(
        &decay,
        "c",
        laddu_core::AngularMomentum::integer(2),
        laddu_core::AngularMomentumProjection::integer(0),
        laddu_core::OrbitalAngularMomentum::integer(2),
        laddu_core::AngularMomentum::integer(0),
        "kshort1",
        laddu_core::AngularMomentum::integer(0),
        laddu_core::AngularMomentum::integer(0),
        laddu_core::AngularMomentumProjection::integer(0),
        laddu_core::AngularMomentumProjection::integer(0),
        Frame::Helicity,
    )
    .unwrap();
    let explicit = Scalar::new("norm", parameter!("norm.value", 5.0_f64.sqrt())).unwrap()
        * ClebschGordan::new(
            "orbital_spin",
            laddu_core::AngularMomentum::integer(2),
            laddu_core::AngularMomentumProjection::integer(0),
            laddu_core::AngularMomentum::integer(0),
            laddu_core::AngularMomentumProjection::integer(0),
            laddu_core::AngularMomentum::integer(2),
            laddu_core::AngularMomentumProjection::integer(0),
        )
        .unwrap()
        * ClebschGordan::new(
            "daughter_spin",
            laddu_core::AngularMomentum::integer(0),
            laddu_core::AngularMomentumProjection::integer(0),
            laddu_core::AngularMomentum::integer(0),
            laddu_core::AngularMomentumProjection::integer(0),
            laddu_core::AngularMomentum::integer(0),
            laddu_core::AngularMomentumProjection::integer(0),
        )
        .unwrap()
        * DecayAmplitudeExt::helicity_factor(
            &decay,
            "d",
            laddu_core::AngularMomentum::integer(2),
            laddu_core::AngularMomentumProjection::integer(0),
            "kshort1",
            laddu_core::AngularMomentumProjection::integer(0),
            laddu_core::AngularMomentumProjection::integer(0),
            Frame::Helicity,
        )
        .unwrap();

    let factor_value = factor.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];
    let explicit_value = explicit.load(&dataset).unwrap().evaluate(&[]).unwrap()[0];

    assert_relative_eq!(factor_value.re, explicit_value.re);
    assert_relative_eq!(factor_value.im, explicit_value.im);
}

#[test]
fn decay_factors_with_matching_names_deduplicate() {
    let dataset = Arc::new(test_dataset());
    let (reaction, _, _) = reaction_context();
    let decay = reaction.decay("kk").unwrap();
    let factor_1 = DecayAmplitudeExt::canonical_factor(
        &decay,
        "rho.factor",
        laddu_core::AngularMomentum::integer(1),
        laddu_core::AngularMomentumProjection::integer(0),
        laddu_core::OrbitalAngularMomentum::integer(1),
        laddu_core::AngularMomentum::integer(0),
        "kshort1",
        laddu_core::AngularMomentum::integer(0),
        laddu_core::AngularMomentum::integer(0),
        laddu_core::AngularMomentumProjection::integer(0),
        laddu_core::AngularMomentumProjection::integer(0),
        Frame::Helicity,
    )
    .unwrap();
    let factor_2 = DecayAmplitudeExt::canonical_factor(
        &decay,
        "rho.factor",
        laddu_core::AngularMomentum::integer(1),
        laddu_core::AngularMomentumProjection::integer(0),
        laddu_core::OrbitalAngularMomentum::integer(1),
        laddu_core::AngularMomentum::integer(0),
        "kshort1",
        laddu_core::AngularMomentum::integer(0),
        laddu_core::AngularMomentum::integer(0),
        laddu_core::AngularMomentumProjection::integer(0),
        laddu_core::AngularMomentumProjection::integer(0),
        Frame::Helicity,
    )
    .unwrap();

    let evaluator = (&factor_1 + &factor_2).load(&dataset).unwrap();

    assert_eq!(
        evaluator.amplitudes.len(),
        factor_1.load(&dataset).unwrap().amplitudes.len()
    );
}
