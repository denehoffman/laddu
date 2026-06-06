use std::sync::Arc;

use approx::assert_relative_eq;
use laddu_core::{
    data::test_dataset,
    kinematics::{Axes, Axis, Frame},
    math::{BarrierKind, WignerDMatrix, QR_DEFAULT},
    reaction::Channel,
    traits::Variable,
    variables::{Angles, Mass},
};

use super::{
    barrier::BlattWeisskopf,
    sdme::{PhotonHelicity, PhotonPolarization, PhotonSDME},
    wigner::WignerD,
};

fn channel() -> Channel {
    let mut channel = Channel::new();
    channel
        .create_production("production", ["beam", "target"], ["kk", "proton"])
        .unwrap();
    channel
        .create_decay("kk_decay", "kk", ["kshort1", "kshort2"])
        .unwrap();
    channel.edit_particle("beam").unwrap().stored();
    channel.edit_particle("target").unwrap().missing().unwrap();
    channel.edit_particle("kshort1").unwrap().stored();
    channel.edit_particle("kshort2").unwrap().stored();
    channel.edit_particle("proton").unwrap().stored();
    channel
}

fn helicity_like_frame() -> Frame {
    Frame::new(
        "kk_decay",
        Axes::from_y_z(
            Axis::normal("beam", "proton").at("production").flipped(),
            Axis::opposite("proton").at("kk_decay"),
        ),
    )
    .unwrap()
}

fn angles(channel: &Channel) -> Angles {
    channel.angles("kshort1", helicity_like_frame()).unwrap()
}

fn masses(channel: &Channel) -> (Mass, Mass, Mass) {
    (
        channel.mass("kk").unwrap(),
        channel.mass("kshort1").unwrap(),
        channel.mass("kshort2").unwrap(),
    )
}

#[test]
fn wigner_d_matches_core_function() {
    let dataset = Arc::new(test_dataset());
    let channel = channel();
    let angles = angles(&channel);
    let expr = WignerD::new(
        "d",
        laddu_core::J::half(2),
        laddu_core::M::half(2),
        laddu_core::M::half(0),
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
    let channel = channel();
    let (parent_mass, daughter_1_mass, daughter_2_mass) = masses(&channel);
    let expr = BlattWeisskopf::new(
        "b",
        &parent_mass,
        &daughter_1_mass,
        &daughter_2_mass,
        laddu_core::L::int(2),
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
fn wigner_d_with_matching_names_deduplicates() {
    let dataset = Arc::new(test_dataset());
    let channel = channel();
    let angles = angles(&channel);
    let factor_1 = WignerD::new(
        "rho.d",
        laddu_core::J::int(1),
        laddu_core::M::int(0),
        laddu_core::M::int(0),
        &angles,
    )
    .unwrap();
    let factor_2 = WignerD::new(
        "rho.d",
        laddu_core::J::int(1),
        laddu_core::M::int(0),
        laddu_core::M::int(0),
        &angles,
    )
    .unwrap();

    let evaluator = (&factor_1 + &factor_2).load(&dataset).unwrap();

    assert_eq!(
        evaluator.amplitudes.len(),
        factor_1.load(&dataset).unwrap().amplitudes.len()
    );
}
