use approx::assert_relative_eq;

use super::*;
use crate::{
    data::test_dataset,
    kinematics::{Axes, Axis, Frame},
    quantum::MandelstamChannel,
    reaction::Channel,
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

fn gj_like_frame() -> Frame {
    Frame::new(
        "kk_decay",
        Axes::from_y_z(
            Axis::normal("beam", "proton").at("production").flipped(),
            Axis::particle("beam").at("kk_decay"),
        ),
    )
    .unwrap()
}

#[test]
fn test_mass_single_particle() {
    let dataset = test_dataset();
    let channel = channel();
    let mass = channel.mass("proton").unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(mass.value(&event), 1.007);
}

#[test]
fn test_mass_multiple_particles() {
    let dataset = test_dataset();
    let channel = channel();
    let mass = channel.mass("kk").unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(mass.value(&event), 1.3743786309153077);
}

#[test]
fn test_mass_display() {
    let channel = channel();
    let mass = channel.mass("kk").unwrap();
    assert_eq!(mass.to_string(), "Mass(particle=kk)");
}

#[test]
fn test_costheta_helicity() {
    let dataset = test_dataset();
    let channel = channel();
    let costheta = channel
        .angles("kshort1", helicity_like_frame())
        .unwrap()
        .costheta;
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(costheta.value(&event), -0.4611175068834202);
}

#[test]
fn test_costheta_display() {
    let channel = channel();
    let costheta = channel
        .angles("kshort1", helicity_like_frame())
        .unwrap()
        .costheta;
    assert_eq!(
        costheta.to_string(),
        "CosTheta(particle=kshort1, frame_origin=kk_decay)"
    );
}

#[test]
fn test_phi_helicity() {
    let dataset = test_dataset();
    let channel = channel();
    let mut phi = channel
        .angles("kshort1", helicity_like_frame())
        .unwrap()
        .phi;
    phi.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(phi.value(&event), -2.657462587335066);
}

#[test]
fn test_phi_display() {
    let channel = channel();
    let phi = channel
        .angles("kshort1", helicity_like_frame())
        .unwrap()
        .phi;
    assert_eq!(
        phi.to_string(),
        "Phi(particle=kshort1, frame_origin=kk_decay)"
    );
}

#[test]
fn test_costheta_gottfried_jackson() {
    let dataset = test_dataset();
    let channel = channel();
    let mut costheta = channel.angles("kshort1", gj_like_frame()).unwrap().costheta;
    costheta.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(costheta.value(&event), 0.09198832278032032);
}

#[test]
fn test_phi_gottfried_jackson() {
    let dataset = test_dataset();
    let channel = channel();
    let mut phi = channel.angles("kshort1", gj_like_frame()).unwrap().phi;
    phi.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(phi.value(&event), -2.7139131991339056);
}

#[test]
fn test_angles() {
    let dataset = test_dataset();
    let channel = channel();
    let mut angles = channel.angles("kshort1", helicity_like_frame()).unwrap();
    angles.costheta.bind(dataset.metadata()).unwrap();
    angles.phi.bind(dataset.metadata()).unwrap();
    let event = dataset.event_local(0).unwrap();
    assert_relative_eq!(angles.costheta.value(&event), -0.4611175068834202);
    assert_relative_eq!(angles.phi.value(&event), -2.657462587335066);
}

#[test]
fn test_angles_display() {
    let channel = channel();
    let angles = channel.angles("kshort1", helicity_like_frame()).unwrap();
    assert_eq!(
        angles.to_string(),
        "Angles(particle=kshort1, frame_origin=kk_decay)"
    );
}

#[test]
fn test_production_angles_display() {
    let channel = channel();
    let frame = Frame::new(
        "production",
        Axes::from_y_z(
            Axis::normal("beam", "kk").at("production"),
            Axis::particle("kk").at("production"),
        ),
    )
    .unwrap();
    let angles = channel.angles("kk", frame).unwrap();

    assert_eq!(
        angles.costheta.to_string(),
        "CosTheta(particle=kk, frame_origin=production)"
    );
    assert_eq!(
        angles.phi.to_string(),
        "Phi(particle=kk, frame_origin=production)"
    );
    assert_eq!(
        angles.to_string(),
        "Angles(particle=kk, frame_origin=production)"
    );
}

#[test]
fn test_pol_angle() {
    let dataset = test_dataset();
    let channel = channel();
    let mut pol_angle = channel.pol_angle("production", "pol_angle").unwrap();
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
    let channel = channel();
    let mut polarization = channel
        .polarization("production", "pol_magnitude", "pol_angle")
        .unwrap();
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
    let channel = channel();
    let mut s = channel
        .mandelstam("production", MandelstamChannel::S)
        .unwrap();
    let mut t = channel
        .mandelstam("production", MandelstamChannel::T)
        .unwrap();
    let mut u = channel
        .mandelstam("production", MandelstamChannel::U)
        .unwrap();
    for variable in [&mut s, &mut t, &mut u] {
        variable.bind(metadata).unwrap();
    }
    let event = dataset.event_local(0).unwrap();
    let beam = event.p4("beam").unwrap();
    let kk = event.p4("kshort1").unwrap() + event.p4("kshort2").unwrap();
    let proton = event.p4("proton").unwrap();
    assert_relative_eq!(s.value(&event), (kk + proton).m2());
    assert_relative_eq!(t.value(&event), (beam - kk).m2());
    assert_relative_eq!(u.value(&event), (beam - proton).m2());
}

#[test]
fn test_mandelstam_display() {
    let channel = channel();
    let s = channel
        .mandelstam("production", MandelstamChannel::S)
        .unwrap();
    assert_eq!(s.to_string(), "Mandelstam(channel=s)");
}

#[test]
fn test_variable_value_on() {
    let dataset = test_dataset();
    let channel = channel();
    let mass = channel.mass("kk").unwrap();

    let values = mass.value_on(&dataset).unwrap();
    assert_eq!(values.len(), 1);
    assert_relative_eq!(values[0], 1.3743786309153077);
}
