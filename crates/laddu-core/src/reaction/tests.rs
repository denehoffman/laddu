use std::sync::Arc;

use approx::assert_relative_eq;

use super::*;
use crate::{
    data::{Dataset, DatasetMetadata, EventData},
    kinematics::{Axes, Axis, Frame},
    vectors::Vec3,
};

fn labels(channel: &Channel) -> Vec<&str> {
    channel.particles().iter().map(Particle::label).collect()
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
