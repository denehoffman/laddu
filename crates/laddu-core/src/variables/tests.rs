use approx::assert_relative_eq;

use super::*;
use crate::{
    data::test_dataset,
    quantum::{Channel, Frame},
    reaction::{Particle, Reaction},
};

fn reaction() -> (Reaction, Particle, Particle, Particle) {
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
        kshort2,
    )
}

#[test]
fn test_mass_single_particle() {
    let dataset = test_dataset();
    let mut mass = Mass::new("proton");
    mass.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(mass.value(&event), 1.007);
}

#[test]
fn test_mass_multiple_particles() {
    let dataset = test_dataset();
    let mut mass = Mass::new(["kshort1", "kshort2"]);
    mass.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(mass.value(&event), 1.3743786309153077);
}

#[test]
fn test_mass_display() {
    let mass = Mass::new(["kshort1", "kshort2"]);
    assert_eq!(mass.to_string(), "Mass(constituents=[kshort1, kshort2])");
}

#[test]
fn test_costheta_helicity() {
    let dataset = test_dataset();
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let mut costheta = decay.costheta("kshort1", Frame::Helicity).unwrap();
    costheta.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(costheta.value(&event), -0.4611175068834202);
}

#[test]
fn test_costheta_display() {
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let costheta = decay.costheta("kshort1", Frame::Helicity).unwrap();
    assert_eq!(
        costheta.to_string(),
        "CosTheta(parent=kk, daughter=kshort1, frame=Helicity)"
    );
}

#[test]
fn test_phi_helicity() {
    let dataset = test_dataset();
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let mut phi = decay.phi("kshort1", Frame::Helicity).unwrap();
    phi.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(phi.value(&event), -2.657462587335066);
}

#[test]
fn test_phi_display() {
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let phi = decay.phi("kshort1", Frame::Helicity).unwrap();
    assert_eq!(
        phi.to_string(),
        "Phi(parent=kk, daughter=kshort1, frame=Helicity)"
    );
}

#[test]
fn test_costheta_gottfried_jackson() {
    let dataset = test_dataset();
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let mut costheta = decay.costheta("kshort1", Frame::GottfriedJackson).unwrap();
    costheta.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(costheta.value(&event), 0.09198832278032032);
}

#[test]
fn test_phi_gottfried_jackson() {
    let dataset = test_dataset();
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let mut phi = decay.phi("kshort1", Frame::GottfriedJackson).unwrap();
    phi.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(phi.value(&event), -2.7139131991339056);
}

#[test]
fn test_angles() {
    let dataset = test_dataset();
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let mut angles = decay.angles("kshort1", Frame::Helicity).unwrap();
    angles.costheta.bind(dataset.metadata()).unwrap();
    angles.phi.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(angles.costheta.value(&event), -0.4611175068834202);
    assert_relative_eq!(angles.phi.value(&event), -2.657462587335066);
}

#[test]
fn test_angles_display() {
    let (reaction, _, _, _) = reaction();
    let decay = reaction.decay("kk").unwrap();
    let angles = decay.angles("kshort1", Frame::Helicity).unwrap();
    assert_eq!(
        angles.to_string(),
        "Angles(parent=kk, daughter=kshort1, frame=Helicity)"
    );
}

#[test]
fn test_pol_angle() {
    let dataset = test_dataset();
    let (reaction, _, _, _) = reaction();
    let mut pol_angle = reaction.pol_angle("pol_angle");
    pol_angle.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(pol_angle.value(&event), 1.935929887818673);
}

#[test]
fn test_pol_magnitude() {
    let dataset = test_dataset();
    let mut pol_magnitude = PolMagnitude::new("pol_magnitude");
    pol_magnitude.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(pol_magnitude.value(&event), 0.38562805);
}

#[test]
fn test_pol_magnitude_display() {
    let pol_magnitude = PolMagnitude::new("pol_magnitude");
    assert_eq!(
        pol_magnitude.to_string(),
        "PolMagnitude(magnitude_aux=pol_magnitude)"
    );
}

#[test]
fn test_polarization() {
    let dataset = test_dataset();
    let (reaction, _, _, _) = reaction();
    let mut polarization = reaction.polarization("pol_magnitude", "pol_angle");
    polarization.pol_angle.bind(dataset.metadata()).unwrap();
    polarization.pol_magnitude.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(polarization.pol_angle.value(&event), 1.935929887818673);
    assert_relative_eq!(polarization.pol_magnitude.value(&event), 0.38562805);
}

#[test]
fn test_mandelstam() {
    let dataset = test_dataset();
    let metadata = dataset.metadata();
    let (reaction, _, _, _) = reaction();
    let mut s = reaction.mandelstam(Channel::S).unwrap();
    let mut t = reaction.mandelstam(Channel::T).unwrap();
    let mut u = reaction.mandelstam(Channel::U).unwrap();
    for variable in [&mut s, &mut t, &mut u] {
        variable.bind(metadata).unwrap();
    }
    let event = dataset.event_local(0).unwrap();
    let resolved = reaction.resolve_two_to_two(&event).unwrap();
    assert_relative_eq!(s.value(&event), resolved.s());
    assert_relative_eq!(t.value(&event), resolved.t());
    assert_relative_eq!(u.value(&event), resolved.u());
}

#[test]
fn test_mandelstam_display() {
    let (reaction, _, _, _) = reaction();
    let s = reaction.mandelstam(Channel::S).unwrap();
    assert_eq!(s.to_string(), "Mandelstam(channel=s)");
}

#[test]
fn test_variable_value_on() {
    let dataset = test_dataset();
    let mass = Mass::new(["kshort1", "kshort2"]);

    let values = mass.value_on(&dataset).unwrap();
    assert_eq!(values.len(), 1);
    assert_relative_eq!(values[0], 1.3743786309153077);
}
