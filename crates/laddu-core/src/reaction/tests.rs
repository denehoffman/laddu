use std::sync::Arc;

use approx::assert_relative_eq;
use serde_pickle::{DeOptions, SerOptions};

use super::*;
use crate::{
    kinematics::FrameAxes, traits::Variable, Channel, Dataset, DatasetMetadata, EventData, Frame,
    Vec3, Vec4,
};

fn two_body_momentum(parent_mass: f64, daughter_1_mass: f64, daughter_2_mass: f64) -> f64 {
    let sum = daughter_1_mass + daughter_2_mass;
    let difference = daughter_1_mass - daughter_2_mass;
    ((parent_mass * parent_mass - sum * sum)
        * (parent_mass * parent_mass - difference * difference))
        .sqrt()
        / (2.0 * parent_mass)
}

fn pion_cascade_dataset() -> (Dataset, Reaction, Particle, Particle, Particle) {
    let pion_mass = 0.139570000000000;
    let rho_mass = 0.775260000000000;
    let rho_momentum_in_x_rest = 0.450000000000000;
    let expected_costheta = 0.500000000000000;
    let expected_phi = 0.700000000000000_f64;
    let rho_in_x_rest = Vec3::new(rho_momentum_in_x_rest, 0.0, 0.0).with_mass(rho_mass);
    let bachelor_in_x_rest = Vec3::new(-rho_momentum_in_x_rest, 0.0, 0.0).with_mass(pion_mass);
    let x_rest = rho_in_x_rest + bachelor_in_x_rest;
    let lab_boost = Vec3::new(0.0, 0.0, 0.350000000000000);
    let x_lab = x_rest.boost(&lab_boost);
    let recoil_lab =
        Vec3::new(-0.200000000000000, 0.0, 0.400000000000000).with_mass(0.938000000000000);
    let beam_lab = Vec4::new(0.0, 0.0, 6.000000000000000, 6.000000000000000);
    let target_lab = x_lab + recoil_lab - beam_lab;
    let x_rest_axes = FrameAxes::from_production_frame(
        Frame::Helicity,
        beam_lab,
        x_lab,
        recoil_lab,
        -(beam_lab + target_lab).beta(),
    )
    .unwrap();
    let pion_momentum_in_rho_rest = two_body_momentum(rho_mass, pion_mass, pion_mass);
    let rho_axes = x_rest_axes.for_daughter(rho_in_x_rest.vec3()).unwrap();
    let sintheta = f64::sqrt(1.0 - expected_costheta * expected_costheta);
    let pi_plus_direction_in_rho_rest = rho_axes.x() * (sintheta * expected_phi.cos())
        + rho_axes.y() * (sintheta * expected_phi.sin())
        + rho_axes.z() * expected_costheta;
    let pi_plus_in_rho_rest =
        (pi_plus_direction_in_rho_rest * pion_momentum_in_rho_rest).with_mass(pion_mass);
    let pi_minus_in_rho_rest =
        (-pi_plus_direction_in_rho_rest * pion_momentum_in_rho_rest).with_mass(pion_mass);
    let rho_beta_in_x_rest = rho_in_x_rest.beta();
    let pi_plus_in_x_rest = pi_plus_in_rho_rest.boost(&rho_beta_in_x_rest);
    let pi_minus_in_x_rest = pi_minus_in_rho_rest.boost(&rho_beta_in_x_rest);
    let bachelor_lab = bachelor_in_x_rest.boost(&lab_boost);
    let pi_plus_lab = pi_plus_in_x_rest.boost(&lab_boost);
    let pi_minus_lab = pi_minus_in_x_rest.boost(&lab_boost);
    let metadata = Arc::new(
        DatasetMetadata::new(
            vec!["beam", "target", "pi_plus", "pi_minus", "pi0", "recoil"],
            Vec::<String>::new(),
        )
        .unwrap(),
    );
    let dataset = Dataset::new_with_metadata(
        vec![Arc::new(EventData {
            p4s: vec![
                beam_lab,
                target_lab,
                pi_plus_lab,
                pi_minus_lab,
                bachelor_lab,
                recoil_lab,
            ],
            aux: vec![],
            weight: 1.0,
        })],
        metadata,
    );
    let pi_plus = Particle::stored("pi_plus");
    let pi_minus = Particle::stored("pi_minus");
    let pi0 = Particle::stored("pi0");
    let rho = Particle::composite("rho", (&pi_plus, &pi_minus)).unwrap();
    let x = Particle::composite("x", (&rho, &pi0)).unwrap();
    let beam = Particle::stored("beam");
    let target = Particle::stored("target");
    let recoil = Particle::stored("recoil");
    let reaction = Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
    (dataset, reaction, rho, pi_plus, x)
}

#[test]
fn reaction_reconstructs_composites_from_lab_final_state() {
    let (dataset, reaction, _, _, _) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let rho_p4 = reaction.p4(&event, "rho").unwrap();
    let x_p4 = reaction.p4(&event, "x").unwrap();

    assert_relative_eq!(rho_p4.m(), 0.775260000000000);
    assert!(x_p4.m() > rho_p4.m());
}

#[test]
fn reaction_angle_variables_use_particles_not_paths() {
    let (dataset, reaction, _, _, _) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let decay = reaction.decay("rho");
    assert!(decay.is_ok());
    let angles = decay.unwrap().angles("pi_plus", Frame::Helicity).unwrap();

    assert_relative_eq!(angles.costheta.value(&event), 0.500000000000000);
    assert_relative_eq!(angles.phi.value(&event), 0.700000000000000);
}

#[test]
fn reaction_particles_are_queryable_by_identifier() {
    let (_, reaction, _, _, _) = pion_cascade_dataset();
    let labels = reaction
        .particles()
        .into_iter()
        .map(|particle| particle.label().to_string())
        .collect::<Vec<_>>();

    assert!(reaction.contains("rho"));
    assert_eq!(reaction.particle("pi_plus").unwrap().label(), "pi_plus");
    assert_eq!(
        labels,
        vec!["beam", "target", "x", "rho", "pi_plus", "pi_minus", "pi0", "recoil"]
    );
}

#[test]
fn reaction_topology_roles_are_queryable() {
    let (_, reaction, _, _, _) = pion_cascade_dataset();
    let topology = reaction.two_to_two_topology().unwrap();

    assert_eq!(topology.p1(), "beam");
    assert_eq!(topology.p2(), "target");
    assert_eq!(topology.p3(), "x");
    assert_eq!(topology.p4(), "recoil");
    assert_eq!(topology.role("p1").unwrap(), "beam");
    assert_eq!(reaction.role("p3").unwrap().label(), "x");
    assert!(topology.role("beam").is_err());
    assert!(reaction.role("unknown").is_err());
}

#[test]
fn reaction_can_be_constructed_from_graph_and_topology() {
    let pi_plus = Particle::stored("pi_plus");
    let pi_minus = Particle::stored("pi_minus");
    let x = Particle::composite("x", (&pi_plus, &pi_minus)).unwrap();
    let beam = Particle::stored("beam");
    let target = Particle::missing("target");
    let recoil = Particle::stored("recoil");
    let graph =
        ParticleGraph::new([beam.clone(), target.clone(), x.clone(), recoil.clone()]).unwrap();
    let topology =
        ReactionTopology::TwoToTwo(TwoToTwoReaction::new(&beam, &target, &x, &recoil).unwrap());
    let reaction = Reaction::new(graph, topology).unwrap();

    assert_eq!(reaction.role("p2").unwrap().label(), "target");
    assert_eq!(reaction.particles().len(), 6);
}

#[test]
fn reaction_serialization_preserves_identifiers_and_topology() {
    let (_, reaction, _, _, _) = pion_cascade_dataset();
    let serialized = serde_pickle::to_vec(&reaction, SerOptions::new()).unwrap();
    let round_tripped: Reaction = serde_pickle::from_slice(&serialized, DeOptions::new()).unwrap();

    assert_eq!(round_tripped, reaction);
    assert!(round_tripped.contains("beam"));
    assert!(round_tripped.contains("rho"));
    assert_eq!(
        round_tripped.particle("pi_minus").unwrap().label(),
        "pi_minus"
    );
    assert_eq!(round_tripped.decay("rho").unwrap().daughter_1(), "pi_plus");
    assert_eq!(
        round_tripped
            .particles()
            .into_iter()
            .map(Particle::label)
            .collect::<Vec<_>>(),
        vec!["beam", "target", "x", "rho", "pi_plus", "pi_minus", "pi0", "recoil"]
    );
}

#[test]
fn reaction_rejects_duplicate_particle_identifiers() {
    let beam = Particle::stored("beam");
    let target = Particle::stored("target");
    let daughter_1 = Particle::stored("pi");
    let daughter_2 = Particle::stored("pi");
    let x = Particle::composite("x", (&daughter_1, &daughter_2)).unwrap();
    let recoil = Particle::stored("recoil");

    assert!(Reaction::two_to_two(&beam, &target, &x, &recoil).is_err());
}

#[test]
fn reaction_reports_unknown_identifiers() {
    let (dataset, reaction, _, _, _) = pion_cascade_dataset();
    let event = dataset.event_view(0);

    assert!(reaction.particle("unknown").is_err());
    assert!(reaction.p4(&event, "unknown").is_err());
    assert!(reaction.decay("unknown").is_err());
    assert!(reaction.decay("pi_plus").is_err());
    assert!(reaction
        .angles_value(&event, "rho", "pi0", Frame::Helicity)
        .is_err());
}

#[test]
fn composite_particles_reject_missing_daughters() {
    let missing = Particle::missing("missing");
    let stored = Particle::stored("stored");

    assert!(Particle::composite("bad", (&missing, &stored)).is_err());
    assert!(Particle::composite("bad", (&stored, &missing)).is_err());
}

#[test]
fn two_to_two_reaction_rejects_multiple_missing_particles() {
    let beam = Particle::stored("beam");
    let target = Particle::missing("target");
    let x = Particle::missing("x");
    let recoil = Particle::stored("recoil");

    assert!(Reaction::two_to_two(&beam, &target, &x, &recoil).is_err());
}

#[test]
fn two_to_two_reaction_solves_missing_particle() {
    let (dataset, reaction, _, _, x) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let full = reaction.resolve_two_to_two(&event).unwrap();
    let beam = Particle::stored("beam");
    let target = Particle::missing("target");
    let recoil = Particle::stored("recoil");
    let missing_reaction = Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
    let resolved = missing_reaction.resolve_two_to_two(&event).unwrap();

    assert_relative_eq!(resolved.p2().px(), full.p2().px());
    assert_relative_eq!(resolved.p2().py(), full.p2().py());
    assert_relative_eq!(resolved.p2().pz(), full.p2().pz());
    assert_relative_eq!(resolved.p2().e(), full.p2().e());
}

#[test]
fn two_to_two_reaction_solves_each_missing_slot() {
    let (dataset, reaction, _, _, x) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let full = reaction.resolve_two_to_two(&event).unwrap();
    let beam = Particle::stored("beam");
    let target = Particle::stored("target");
    let recoil = Particle::stored("recoil");
    let missing_beam = Particle::missing("beam");
    let missing_target = Particle::missing("target");
    let missing_x = Particle::missing("x");
    let missing_recoil = Particle::missing("recoil");
    let cases = [
        (
            Reaction::two_to_two(&missing_beam, &target, &x, &recoil).unwrap(),
            full.p1(),
            0,
        ),
        (
            Reaction::two_to_two(&beam, &missing_target, &x, &recoil).unwrap(),
            full.p2(),
            1,
        ),
        (
            Reaction::two_to_two(&beam, &target, &missing_x, &recoil).unwrap(),
            full.p3(),
            2,
        ),
        (
            Reaction::two_to_two(&beam, &target, &x, &missing_recoil).unwrap(),
            full.p4(),
            3,
        ),
    ];

    for (reaction, expected, slot) in cases {
        let resolved = reaction.resolve_two_to_two(&event).unwrap();
        let actual = match slot {
            0 => resolved.p1(),
            1 => resolved.p2(),
            2 => resolved.p3(),
            3 => resolved.p4(),
            _ => unreachable!("test slot"),
        };
        assert_relative_eq!(actual.px(), expected.px(), epsilon = 1.0e-12);
        assert_relative_eq!(actual.py(), expected.py(), epsilon = 1.0e-12);
        assert_relative_eq!(actual.pz(), expected.pz(), epsilon = 1.0e-12);
        assert_relative_eq!(actual.e(), expected.e(), epsilon = 1.0e-12);
    }
}

#[test]
fn fixed_particle_can_define_a_reaction_role() {
    let (dataset, _, _, _, x) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let beam = Particle::stored("beam");
    let fixed_target = Particle::fixed("target", event.p4("target").unwrap());
    let recoil = Particle::stored("recoil");
    let reaction = Reaction::two_to_two(&beam, &fixed_target, &x, &recoil).unwrap();
    let resolved = reaction.resolve_two_to_two(&event).unwrap();

    assert_relative_eq!(resolved.p2().e(), event.p4("target").unwrap().e());
}

#[test]
fn reaction_mandelstam_variables_match_resolved_values() {
    let (dataset, reaction, _, _, _) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let resolved = reaction.resolve_two_to_two(&event).unwrap();

    assert!(reaction.topology().supports_mandelstam());
    assert!(reaction.topology().require_mandelstam().is_ok());
    assert_relative_eq!(
        reaction.mandelstam(Channel::S).unwrap().value(&event),
        resolved.s()
    );
    assert_relative_eq!(
        reaction.mandelstam(Channel::T).unwrap().value(&event),
        resolved.t()
    );
    assert_relative_eq!(
        reaction.mandelstam(Channel::U).unwrap().value(&event),
        resolved.u()
    );
}

#[test]
fn production_frame_axes_support_known_frames() {
    let (dataset, reaction, _, _, _) = pion_cascade_dataset();
    let event = dataset.event_view(0);

    assert!(reaction.axes(&event, "x", Frame::Helicity).is_ok());
    assert!(reaction.axes(&event, "x", Frame::GottfriedJackson).is_ok());
    assert!(reaction.axes(&event, "x", Frame::Adair).is_err());
}
