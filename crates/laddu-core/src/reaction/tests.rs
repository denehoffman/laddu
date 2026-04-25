use std::sync::Arc;

use approx::assert_relative_eq;

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
    let pi_plus = Particle::measured("pi+", "pi_plus");
    let pi_minus = Particle::measured("pi-", "pi_minus");
    let pi0 = Particle::measured("pi0", "pi0");
    let rho = Particle::composite("rho", [&pi_plus, &pi_minus]).unwrap();
    let x = Particle::composite("x", [&rho, &pi0]).unwrap();
    let beam = Particle::measured("beam", "beam");
    let target = Particle::measured("target", "target");
    let recoil = Particle::measured("recoil", "recoil");
    let reaction = Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
    (dataset, reaction, rho, pi_plus, x)
}

#[test]
fn reaction_reconstructs_composites_from_lab_final_state() {
    let (dataset, reaction, rho, _, x) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let rho_p4 = reaction.p4(&event, &rho).unwrap();
    let x_p4 = reaction.p4(&event, &x).unwrap();

    assert_relative_eq!(rho_p4.m(), 0.775260000000000);
    assert!(x_p4.m() > rho_p4.m());
}

#[test]
fn reaction_angle_variables_use_particles_not_paths() {
    let (dataset, reaction, rho, pi_plus, _) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let decay = reaction.decay(&rho);
    assert!(decay.is_ok());
    let angles = decay.unwrap().angles(&pi_plus, Frame::Helicity).unwrap();

    assert_relative_eq!(angles.costheta.value(&event), 0.500000000000000);
    assert_relative_eq!(angles.phi.value(&event), 0.700000000000000);
}

#[test]
fn two_to_two_reaction_solves_missing_particle() {
    let (dataset, reaction, _, _, x) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let full = reaction.resolve_two_to_two(&event).unwrap();
    let beam = Particle::measured("beam", "beam");
    let target = Particle::missing("target");
    let recoil = Particle::measured("recoil", "recoil");
    let missing_reaction = Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
    let resolved = missing_reaction.resolve_two_to_two(&event).unwrap();

    assert_relative_eq!(resolved.p2().px(), full.p2().px());
    assert_relative_eq!(resolved.p2().py(), full.p2().py());
    assert_relative_eq!(resolved.p2().pz(), full.p2().pz());
    assert_relative_eq!(resolved.p2().e(), full.p2().e());
}

#[test]
fn fixed_particle_can_define_a_reaction_role() {
    let (dataset, _, _, _, x) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let beam = Particle::measured("beam", "beam");
    let fixed_target = Particle::fixed("target", event.p4("target").unwrap());
    let recoil = Particle::measured("recoil", "recoil");
    let reaction = Reaction::two_to_two(&beam, &fixed_target, &x, &recoil).unwrap();
    let resolved = reaction.resolve_two_to_two(&event).unwrap();

    assert_relative_eq!(resolved.p2().e(), event.p4("target").unwrap().e());
}

#[test]
fn reaction_mandelstam_variables_match_resolved_values() {
    let (dataset, reaction, _, _, _) = pion_cascade_dataset();
    let event = dataset.event_view(0);
    let resolved = reaction.resolve_two_to_two(&event).unwrap();

    assert_relative_eq!(reaction.mandelstam(Channel::S).value(&event), resolved.s());
    assert_relative_eq!(reaction.mandelstam(Channel::T).value(&event), resolved.t());
    assert_relative_eq!(reaction.mandelstam(Channel::U).value(&event), resolved.u());
}
