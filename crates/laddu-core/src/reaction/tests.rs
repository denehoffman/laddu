use std::sync::Arc;

use approx::assert_relative_eq;

use super::*;
use crate::{
    data::{Dataset, DatasetMetadata, EventData},
    kinematics::{Axes, Axis, Frame},
    vectors::Vec3,
    Charge, Parity, ParticleProperties, RuleSet, Statistics, J, L,
};

fn labels(channel: &Channel) -> Vec<&str> {
    channel.particles().iter().map(Particle::label).collect()
}

fn wave_quantum_numbers(couplings: &[TwoBodyCoupling]) -> Vec<(J, L, J)> {
    couplings.iter().map(|c| (c.j(), c.l(), c.s())).collect()
}

#[test]
fn vertex_declaration_creates_particle_edges() {
    let mut channel = Channel::new();

    channel
        .create_production("production", ["beam", "target"], ["rho", "recoil"])
        .unwrap();

    assert_eq!(labels(&channel), vec!["beam", "target", "rho", "recoil"]);
    assert!(matches!(
        channel.particle("beam").unwrap().from(),
        Endpoint::ExternalIn
    ));
    assert!(matches!(
        channel.particle("beam").unwrap().to(),
        Endpoint::Vertex(_)
    ));
    assert!(matches!(
        channel.particle("rho").unwrap().from(),
        Endpoint::Vertex(_)
    ));
    assert!(matches!(
        channel.particle("rho").unwrap().to(),
        Endpoint::ExternalOut
    ));
}

#[test]
fn vertex_created_particles_can_be_annotated() {
    let mut channel = Channel::new();
    channel
        .create_decay("rho_decay", "rho", ["pi_plus", "pi_minus"])
        .unwrap();
    channel.edit_particle("rho").unwrap().mass(0.775);

    assert_eq!(
        channel
            .particle("rho")
            .unwrap()
            .properties()
            .mass()
            .unwrap(),
        0.775
    );
}

#[test]
fn unannotated_two_body_couplings_apply_only_angular_rules() {
    let mut channel = Channel::new();
    channel.create_decay("decay", "x", ["a", "b"]).unwrap();
    channel.edit_particle("a").unwrap().spin(J::int(0));
    channel.edit_particle("b").unwrap().spin(J::int(0));

    let couplings = channel
        .two_body_couplings("decay", J::int(2), L::int(2))
        .unwrap();

    assert_eq!(
        wave_quantum_numbers(&couplings),
        vec![
            (J::int(0), L::int(0), J::int(0)),
            (J::int(1), L::int(1), J::int(0)),
            (J::int(2), L::int(2), J::int(0)),
        ]
    );
}

#[test]
fn strong_two_body_couplings_filter_identical_ksks_waves() {
    let mut channel = Channel::new();
    channel
        .create_decay("x_decay", "x", ["ks1", "ks2"])
        .unwrap()
        .rules(RuleSet::strong());
    let ks = ParticleProperties::jp(J::int(0), Parity::Negative)
        .with_species("K_S")
        .with_charge(Charge::int(0))
        .with_strangeness(0)
        .with_baryon_number(0)
        .with_statistics(Statistics::Boson)
        .unwrap();
    channel.edit_particle("ks1").unwrap().properties(ks.clone());
    channel.edit_particle("ks2").unwrap().properties(ks);

    let couplings = channel
        .two_body_couplings("x_decay", J::int(2), L::int(2))
        .unwrap();

    assert_eq!(
        wave_quantum_numbers(&couplings),
        vec![
            (J::int(0), L::int(0), J::int(0)),
            (J::int(2), L::int(2), J::int(0)),
        ]
    );
    assert_eq!(
        couplings
            .iter()
            .map(|c| c.parent_properties().parity)
            .collect::<Vec<_>>(),
        vec![Some(Parity::Positive), Some(Parity::Positive)]
    );
}

#[test]
fn two_body_couplings_enumerate_nonzero_daughter_spin_channel_spins() {
    let mut channel = Channel::new();
    channel.create_decay("decay", "x", ["v1", "v2"]).unwrap();
    channel.edit_particle("v1").unwrap().spin(J::int(1));
    channel.edit_particle("v2").unwrap().spin(J::int(1));

    let couplings = channel
        .two_body_couplings("decay", J::int(1), L::int(1))
        .unwrap();

    assert!(couplings
        .iter()
        .any(|c| c.j() == J::int(1) && c.l() == L::int(0) && c.s() == J::int(1)));
    assert!(couplings
        .iter()
        .any(|c| c.j() == J::int(1) && c.l() == L::int(1) && c.s() == J::int(0)));
    assert!(couplings
        .iter()
        .any(|c| c.j() == J::int(1) && c.l() == L::int(1) && c.s() == J::int(1)));
    assert!(couplings
        .iter()
        .any(|c| c.j() == J::int(1) && c.l() == L::int(1) && c.s() == J::int(2)));
}

#[test]
fn known_parent_spin_is_restricted_by_j_max() {
    let mut channel = Channel::new();
    channel.create_decay("decay", "x", ["a", "b"]).unwrap();
    channel.edit_particle("x").unwrap().spin(J::int(2));
    channel.edit_particle("a").unwrap().spin(J::int(0));
    channel.edit_particle("b").unwrap().spin(J::int(0));

    assert!(channel
        .two_body_couplings("decay", J::int(1), L::int(4))
        .unwrap()
        .is_empty());
    assert_eq!(
        wave_quantum_numbers(
            &channel
                .two_body_couplings("decay", J::int(2), L::int(4))
                .unwrap()
        ),
        vec![(J::int(2), L::int(2), J::int(0))]
    );
}

#[test]
fn missing_daughter_spin_is_rejected_for_two_body_couplings() {
    let mut channel = Channel::new();
    channel.create_decay("decay", "x", ["a", "b"]).unwrap();
    channel.edit_particle("a").unwrap().spin(J::int(0));

    let error = channel
        .two_body_couplings("decay", J::int(2), L::int(2))
        .unwrap_err()
        .to_string();

    assert!(error.contains("particle 'b' has no spin property"));
}

#[test]
fn editing_unknown_particles_is_rejected() {
    let mut channel = Channel::new();

    assert!(channel.edit_particle("rho").is_err());
    assert!(channel.particle("rho").is_err());
    assert!(channel.particles().is_empty());
}

#[test]
fn duplicate_vertex_labels_are_rejected_atomically() {
    let mut channel = Channel::new();
    channel
        .create_decay("decay", "rho", ["pi_plus", "pi_minus"])
        .unwrap();

    assert!(channel.create_decay("decay", "x", ["a", "b"]).is_err());
    assert_eq!(channel.vertices().len(), 1);
    assert!(channel.particle("x").is_err());
}

#[test]
fn repeated_particle_labels_in_vertex_are_rejected() {
    let mut channel = Channel::new();

    assert!(channel.create_decay("bad", "rho", ["pi", "pi"]).is_err());
    assert!(channel.vertex("bad").is_err());
    assert!(channel.particles().is_empty());
}

#[test]
fn reused_particle_producer_or_consumer_is_rejected() {
    let mut channel = Channel::new();
    channel
        .create_decay("rho_decay", "rho", ["pi_plus", "pi_minus"])
        .unwrap();

    assert!(channel
        .create_decay("other_decay", "rho", ["a", "b"])
        .is_err());
    channel
        .create_production("production", ["beam", "target"], ["rho", "recoil"])
        .unwrap();
    assert!(channel
        .create_production("bad_prod", ["other_beam", "other_target"], ["rho", "other"])
        .is_err());
    assert!(channel.vertex("other_decay").is_err());
    assert!(channel.vertex("bad_prod").is_err());
}

#[test]
fn cycle_insertion_is_rejected_atomically() {
    let mut channel = Channel::new();
    channel.create_vertex("v1", ["a"], ["b"]).unwrap();

    assert!(channel.create_vertex("v2", ["b"], ["a"]).is_err());
    assert!(channel.vertex("v2").is_err());
    assert!(matches!(
        channel.particle("a").unwrap().from(),
        Endpoint::ExternalIn
    ));
    assert!(matches!(
        channel.particle("b").unwrap().to(),
        Endpoint::ExternalOut
    ));
}

#[test]
fn internal_particles_cannot_be_marked_missing_directly() {
    let mut channel = Channel::new();
    channel
        .create_production("prod", ["beam", "target"], ["rho", "recoil"])
        .unwrap();
    channel
        .create_decay("rho_decay", "rho", ["pi_plus", "pi_minus"])
        .unwrap();

    assert!(channel.edit_particle("rho").unwrap().missing().is_err());
}

#[test]
fn missing_particles_cannot_later_become_internal() {
    let mut channel = Channel::new();
    channel
        .create_production("prod", ["beam", "target"], ["rho", "recoil"])
        .unwrap();
    channel.edit_particle("rho").unwrap().missing().unwrap();

    assert!(channel
        .create_decay("rho_decay", "rho", ["pi_plus", "pi_minus"])
        .is_err());
    assert!(channel.vertex("rho_decay").is_err());
    assert!(matches!(
        channel.particle("rho").unwrap().to(),
        Endpoint::ExternalOut
    ));
}

#[test]
fn inferred_internal_p4_sums_decay_products() {
    let mut channel = Channel::new();
    channel
        .create_decay("rho_decay", "rho", ["pi_plus", "pi_minus"])
        .unwrap();

    let pi_plus = Vec3::new(0.1, 0.2, 0.3).with_mass(0.139);
    let pi_minus = Vec3::new(-0.2, 0.1, 0.4).with_mass(0.139);
    let metadata =
        Arc::new(DatasetMetadata::new(vec!["pi_plus", "pi_minus"], Vec::<&str>::new()).unwrap());
    let dataset = Dataset::new_with_metadata(
        vec![Arc::new(EventData {
            p4s: vec![pi_plus, pi_minus],
            aux: vec![],
            weight: 1.0,
        })],
        metadata,
    );
    let event = dataset.event_local(0).unwrap();

    assert_relative_eq!(
        channel.p4("rho", &event).unwrap().e(),
        (pi_plus + pi_minus).e()
    );
    assert_relative_eq!(
        channel.p4("rho", &event).unwrap().pz(),
        (pi_plus + pi_minus).pz()
    );
}

#[test]
fn stored_source_overrides_inferred_decay_sum() {
    let mut channel = Channel::new();
    channel
        .create_decay("rho_decay", "rho", ["pi_plus", "pi_minus"])
        .unwrap();
    channel.edit_particle("rho").unwrap().stored();

    let stored_rho = Vec3::new(0.0, 0.0, 1.0).with_mass(1.0);
    let pi_plus = Vec3::new(0.1, 0.2, 0.3).with_mass(0.139);
    let pi_minus = Vec3::new(-0.2, 0.1, 0.4).with_mass(0.139);
    let metadata = Arc::new(
        DatasetMetadata::new(vec!["rho", "pi_plus", "pi_minus"], Vec::<&str>::new()).unwrap(),
    );
    let dataset = Dataset::new_with_metadata(
        vec![Arc::new(EventData {
            p4s: vec![stored_rho, pi_plus, pi_minus],
            aux: vec![],
            weight: 1.0,
        })],
        metadata,
    );
    let event = dataset.event_local(0).unwrap();

    assert_relative_eq!(channel.p4("rho", &event).unwrap().e(), stored_rho.e());
    assert_relative_eq!(channel.p4("rho", &event).unwrap().pz(), stored_rho.pz());
}

#[test]
fn missing_incoming_particle_is_outgoing_minus_known_incoming() {
    let mut channel = Channel::new();
    channel
        .create_production("production", ["beam", "target"], ["rho", "recoil"])
        .unwrap();
    channel.edit_particle("beam").unwrap().stored();
    channel.edit_particle("target").unwrap().missing().unwrap();
    channel.edit_particle("rho").unwrap().stored();
    channel.edit_particle("recoil").unwrap().stored();

    let beam = Vec3::new(0.0, 0.0, 8.0).with_mass(0.0);
    let rho = Vec3::new(0.2, 0.0, 5.0).with_mass(1.0);
    let recoil = Vec3::new(-0.2, 0.0, 3.1).with_mass(0.938);
    let metadata =
        Arc::new(DatasetMetadata::new(vec!["beam", "rho", "recoil"], Vec::<&str>::new()).unwrap());
    let dataset = Dataset::new_with_metadata(
        vec![Arc::new(EventData {
            p4s: vec![beam, rho, recoil],
            aux: vec![],
            weight: 1.0,
        })],
        metadata,
    );
    let event = dataset.event_local(0).unwrap();
    let expected = rho + recoil - beam;

    assert_relative_eq!(channel.p4("target", &event).unwrap().e(), expected.e());
    assert_relative_eq!(channel.p4("target", &event).unwrap().pz(), expected.pz());
}

#[test]
fn missing_outgoing_particle_is_incoming_minus_known_outgoing() {
    let mut channel = Channel::new();
    channel
        .create_production("production", ["beam", "target"], ["rho", "recoil"])
        .unwrap();
    channel.edit_particle("beam").unwrap().stored();
    channel.edit_particle("target").unwrap().stored();
    channel.edit_particle("rho").unwrap().stored();
    channel.edit_particle("recoil").unwrap().missing().unwrap();

    let beam = Vec3::new(0.0, 0.0, 8.0).with_mass(0.0);
    let target = Vec3::zero().with_mass(0.938);
    let rho = Vec3::new(0.2, 0.0, 5.0).with_mass(1.0);
    let metadata =
        Arc::new(DatasetMetadata::new(vec!["beam", "target", "rho"], Vec::<&str>::new()).unwrap());
    let dataset = Dataset::new_with_metadata(
        vec![Arc::new(EventData {
            p4s: vec![beam, target, rho],
            aux: vec![],
            weight: 1.0,
        })],
        metadata,
    );
    let event = dataset.event_local(0).unwrap();
    let expected = beam + target - rho;

    assert_relative_eq!(channel.p4("recoil", &event).unwrap().e(), expected.e());
    assert_relative_eq!(channel.p4("recoil", &event).unwrap().pz(), expected.pz());
}

#[test]
fn custom_frame_angles_project_measured_particle() {
    let mut channel = Channel::new();
    channel
        .create_production("production", ["beam", "target"], ["pi_plus", "spectator"])
        .unwrap();

    let beam = Vec3::new(0.0, 0.0, 1.0).with_mass(0.0);
    let target = Vec3::new(0.0, 0.0, -1.0).with_mass(0.0);
    let pi_plus = Vec3::new(1.0, 0.0, 0.0).with_mass(0.0);
    let spectator = Vec3::new(-1.0, 0.0, 0.0).with_mass(0.0);
    let metadata = Arc::new(
        DatasetMetadata::new(
            vec!["beam", "target", "pi_plus", "spectator"],
            Vec::<&str>::new(),
        )
        .unwrap(),
    );
    let dataset = Dataset::new_with_metadata(
        vec![Arc::new(EventData {
            p4s: vec![beam, target, pi_plus, spectator],
            aux: vec![],
            weight: 1.0,
        })],
        metadata,
    );
    let event = dataset.event_local(0).unwrap();
    let frame = Frame::new(
        "production",
        Axes::from_y_z(
            Axis::normal("beam", "pi_plus").at("production"),
            Axis::particle("pi_plus").at("production"),
        ),
    )
    .unwrap();

    let angles = channel.angles("beam", frame).unwrap();

    assert_relative_eq!(angles.costheta.evaluator.costheta(&event).unwrap(), 0.0);
    assert_relative_eq!(
        angles.phi.evaluator.phi(&event).unwrap(),
        std::f64::consts::PI
    );
}
