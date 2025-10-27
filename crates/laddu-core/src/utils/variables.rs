use crate::utils::{
    enums::{Channel, Frame},
    vectors::Vec4,
};
use polars::prelude::*;

pub fn mass<I, S>(constituents: I) -> Expr
where
    I: IntoIterator<Item = S>,
    S: Into<PlSmallStr>,
{
    Vec4::sum(constituents).mag()
}

pub fn costheta<Ir, Id, Ires, Sb, Sr, Sd, Sres>(
    beam: Sb,
    recoil: Ir,
    daughter: Id,
    resonance: Ires,
    frame: Frame,
) -> Expr
where
    Sb: Into<PlSmallStr>,
    Ir: IntoIterator<Item = Sr>,
    Sr: Into<PlSmallStr>,
    Id: IntoIterator<Item = Sd>,
    Sd: Into<PlSmallStr>,
    Ires: IntoIterator<Item = Sres>,
    Sres: Into<PlSmallStr>,
{
    let beam = Vec4::new(beam);
    let recoil = Vec4::sum(recoil);
    let daughter = Vec4::sum(daughter);
    let resonance = Vec4::sum(resonance);
    let resonance_boost = resonance.beta().neg();
    let daughter_res = daughter.boost(&resonance_boost);
    match frame {
        Frame::Helicity => {
            let recoil_res = recoil.boost(&resonance_boost);
            let z = recoil_res.vec3().neg().unit();
            daughter_res.vec3().unit().dot(&z)
        }
        Frame::GottfriedJackson => {
            let beam_res = beam.boost(&resonance_boost);
            let z = beam_res.vec3().unit();
            daughter_res.vec3().unit().dot(&z)
        }
    }
}

pub fn phi<Ir, Id, Ires, Sb, Sr, Sd, Sres>(
    beam: Sb,
    recoil: Ir,
    daughter: Id,
    resonance: Ires,
    frame: Frame,
) -> Expr
where
    Sb: Into<PlSmallStr>,
    Ir: IntoIterator<Item = Sr>,
    Sr: Into<PlSmallStr>,
    Id: IntoIterator<Item = Sd>,
    Sd: Into<PlSmallStr>,
    Ires: IntoIterator<Item = Sres>,
    Sres: Into<PlSmallStr>,
{
    let beam = Vec4::new(beam);
    let recoil = Vec4::sum(recoil);
    let daughter = Vec4::sum(daughter);
    let resonance = Vec4::sum(resonance);
    let resonance_boost = resonance.beta().neg();
    let daughter_res = daughter.boost(&resonance_boost);
    let z = match frame {
        Frame::Helicity => {
            let recoil_res = recoil.boost(&resonance_boost);
            recoil_res.vec3().neg().unit()
        }
        Frame::GottfriedJackson => {
            let beam_res = beam.boost(&resonance_boost);
            beam_res.vec3().unit()
        }
    };
    let y = beam.vec3().cross(&recoil.vec3().neg()).unit();
    let x = y.cross(&z);
    let p = daughter_res.vec3();
    p.dot(&y).arctan2(p.dot(&x))
}

pub fn angles<Ir, Id, Ires, Sb, Sr, Sd, Sres>(
    beam: Sb,
    recoil: Ir,
    daughter: Id,
    resonance: Ires,
    frame: Frame,
) -> (Expr, Expr)
where
    Sb: Into<PlSmallStr>,
    Ir: IntoIterator<Item = Sr>,
    Sr: Into<PlSmallStr>,
    Id: IntoIterator<Item = Sd>,
    Sd: Into<PlSmallStr>,
    Ires: IntoIterator<Item = Sres>,
    Sres: Into<PlSmallStr>,
{
    let beam = Vec4::new(beam);
    let recoil = Vec4::sum(recoil);
    let daughter = Vec4::sum(daughter);
    let resonance = Vec4::sum(resonance);
    let resonance_boost = resonance.beta().neg();
    let daughter_res = daughter.boost(&resonance_boost);
    let z = match frame {
        Frame::Helicity => {
            let recoil_res = recoil.boost(&resonance_boost);
            recoil_res.vec3().neg().unit()
        }
        Frame::GottfriedJackson => {
            let beam_res = beam.boost(&resonance_boost);
            beam_res.vec3().unit()
        }
    };
    let y = beam.vec3().cross(&recoil.vec3().neg()).unit();
    let x = y.cross(&z);
    let p = daughter_res.vec3();
    (p.unit().dot(&z), p.dot(&y).arctan2(p.dot(&x)))
}

pub fn pol_angle<Ir, Sb, Sr, Sphi>(beam: Sb, recoil: Ir, phi: Sphi) -> Expr
where
    Sb: Into<PlSmallStr>,
    Ir: IntoIterator<Item = Sr>,
    Sr: Into<PlSmallStr>,
    Sphi: Into<PlSmallStr>,
{
    let beam = Vec4::new(beam);
    let recoil = Vec4::sum(recoil);
    let phi = col(phi);
    let u = beam.vec3().unit();
    let w = beam.vec3().cross(&recoil.vec3().unit());
    (w.x() * phi.clone().cos() + w.y() * phi.clone().sin()).arctan2(
        w.z() * (u.x() * phi.clone().sin() - u.y() * phi.clone().cos())
            + u.z() * (w.y() * phi.clone().cos() - w.x() * phi.sin()),
    )
}

pub fn pol_magnitude<Spgamma>(p_gamma: Spgamma) -> Expr
where
    Spgamma: Into<PlSmallStr>,
{
    col(p_gamma)
}

pub fn polarization<Ir, Sb, Sr, Spgamma, Sphi>(
    beam: Sb,
    recoil: Ir,
    p_gamma: Spgamma,
    phi: Sphi,
) -> (Expr, Expr)
where
    Sb: Into<PlSmallStr>,
    Ir: IntoIterator<Item = Sr>,
    Sr: Into<PlSmallStr>,
    Spgamma: Into<PlSmallStr>,
    Sphi: Into<PlSmallStr>,
{
    (col(p_gamma), pol_angle(beam, recoil, phi))
}

enum Missing {
    None,
    P1,
    P2,
    P3,
    P4,
}
impl Missing {
    fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

pub fn mandelstam<I1, S1, I2, S2, I3, S3, I4, S4>(
    p1: I1,
    p2: I2,
    p3: I3,
    p4: I4,
    channel: Channel,
) -> Expr
where
    I1: IntoIterator<Item = S1>,
    S1: Into<PlSmallStr>,
    I2: IntoIterator<Item = S2>,
    S2: Into<PlSmallStr>,
    I3: IntoIterator<Item = S3>,
    S3: Into<PlSmallStr>,
    I4: IntoIterator<Item = S4>,
    S4: Into<PlSmallStr>,
{
    let mut missing = Missing::None;
    let p1: Vec<PlSmallStr> = p1.into_iter().map(|s| s.into()).collect();
    let p2: Vec<PlSmallStr> = p2.into_iter().map(|s| s.into()).collect();
    let p3: Vec<PlSmallStr> = p3.into_iter().map(|s| s.into()).collect();
    let p4: Vec<PlSmallStr> = p4.into_iter().map(|s| s.into()).collect();
    if p1.is_empty() {
        missing = Missing::P1
    }
    if p2.is_empty() {
        if missing.is_none() {
            missing = Missing::P2
        } else {
            unimplemented!()
        }
    }
    if p3.is_empty() {
        if missing.is_none() {
            missing = Missing::P3
        } else {
            unimplemented!()
        }
    }
    if p4.is_empty() {
        if missing.is_none() {
            missing = Missing::P4
        } else {
            unimplemented!()
        }
    }
    match channel {
        Channel::S => match missing {
            Missing::None | Missing::P3 | Missing::P4 => (Vec4::sum(p1).add(&Vec4::sum(p2))).mag2(),
            Missing::P1 | Missing::P2 => (Vec4::sum(p3).add(&Vec4::sum(p4))).mag2(),
        },
        Channel::T => match missing {
            Missing::None | Missing::P2 | Missing::P4 => (Vec4::sum(p1).sub(&Vec4::sum(p3))).mag2(),
            Missing::P1 | Missing::P3 => (Vec4::sum(p4).sub(&Vec4::sum(p2))).mag2(),
        },
        Channel::U => match missing {
            Missing::None | Missing::P2 | Missing::P3 => (Vec4::sum(p1).sub(&Vec4::sum(p4))).mag2(),
            Missing::P1 | Missing::P4 => (Vec4::sum(p3).sub(&Vec4::sum(p2))).mag2(),
        },
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::data::{test_dataset, test_event};
    // use approx::assert_relative_eq;
    //
    // #[test]
    // fn test_mass_single_particle() {
    //     let event = test_event();
    //     let mass = Mass::new([1]);
    //     assert_relative_eq!(mass.value(&event), 1.007);
    // }
    //
    // #[test]
    // fn test_mass_multiple_particles() {
    //     let event = test_event();
    //     let mass = Mass::new([2, 3]);
    //     assert_relative_eq!(
    //         mass.value(&event),
    //         1.37437863,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_mass_display() {
    //     let mass = Mass::new([2, 3]);
    //     assert_eq!(mass.to_string(), "Mass(constituents=[2, 3])");
    // }
    //
    // #[test]
    // fn test_costheta_helicity() {
    //     let event = test_event();
    //     let costheta = CosTheta::new(0, [1], [2], [2, 3], Frame::Helicity);
    //     assert_relative_eq!(
    //         costheta.value(&event),
    //         -0.4611175,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_costheta_display() {
    //     let costheta = CosTheta::new(0, [1], [2], [2, 3], Frame::Helicity);
    //     assert_eq!(
    //         costheta.to_string(),
    //         "CosTheta(beam=0, recoil=[1], daughter=[2], resonance=[2, 3], frame=Helicity)"
    //     );
    // }
    //
    // #[test]
    // fn test_phi_helicity() {
    //     let event = test_event();
    //     let phi = Phi::new(0, [1], [2], [2, 3], Frame::Helicity);
    //     assert_relative_eq!(
    //         phi.value(&event),
    //         -2.65746258,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_phi_display() {
    //     let phi = Phi::new(0, [1], [2], [2, 3], Frame::Helicity);
    //     assert_eq!(
    //         phi.to_string(),
    //         "Phi(beam=0, recoil=[1], daughter=[2], resonance=[2, 3], frame=Helicity)"
    //     );
    // }
    //
    // #[test]
    // fn test_costheta_gottfried_jackson() {
    //     let event = test_event();
    //     let costheta = CosTheta::new(0, [1], [2], [2, 3], Frame::GottfriedJackson);
    //     assert_relative_eq!(
    //         costheta.value(&event),
    //         0.09198832,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_phi_gottfried_jackson() {
    //     let event = test_event();
    //     let phi = Phi::new(0, [1], [2], [2, 3], Frame::GottfriedJackson);
    //     assert_relative_eq!(
    //         phi.value(&event),
    //         -2.71391319,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_angles() {
    //     let event = test_event();
    //     let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
    //     assert_relative_eq!(
    //         angles.costheta.value(&event),
    //         -0.4611175,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    //     assert_relative_eq!(
    //         angles.phi.value(&event),
    //         -2.65746258,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_angles_display() {
    //     let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
    //     assert_eq!(
    //         angles.to_string(),
    //         "Angles(beam=0, recoil=[1], daughter=[2], resonance=[2, 3], frame=Helicity)"
    //     );
    // }
    //
    // #[test]
    // fn test_pol_angle() {
    //     let event = test_event();
    //     let pol_angle = PolAngle::new(0, vec![1], 0);
    //     assert_relative_eq!(
    //         pol_angle.value(&event),
    //         1.93592989,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_pol_angle_display() {
    //     let pol_angle = PolAngle::new(0, vec![1], 0);
    //     assert_eq!(
    //         pol_angle.to_string(),
    //         "PolAngle(beam=0, recoil=[1], beam_polarization=0)"
    //     );
    // }
    //
    // #[test]
    // fn test_pol_magnitude() {
    //     let event = test_event();
    //     let pol_magnitude = PolMagnitude::new(0);
    //     assert_relative_eq!(
    //         pol_magnitude.value(&event),
    //         0.38562805,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_pol_magnitude_display() {
    //     let pol_magnitude = PolMagnitude::new(0);
    //     assert_eq!(
    //         pol_magnitude.to_string(),
    //         "PolMagnitude(beam_polarization=0)"
    //     );
    // }
    //
    // #[test]
    // fn test_polarization() {
    //     let event = test_event();
    //     let polarization = Polarization::new(0, vec![1], 0);
    //     assert_relative_eq!(
    //         polarization.pol_angle.value(&event),
    //         1.93592989,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    //     assert_relative_eq!(
    //         polarization.pol_magnitude.value(&event),
    //         0.38562805,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
    //
    // #[test]
    // fn test_polarization_display() {
    //     let polarization = Polarization::new(0, vec![1], 0);
    //     assert_eq!(
    //         polarization.to_string(),
    //         "Polarization(beam=0, recoil=[1], beam_polarization=0)"
    //     );
    // }
    //
    // #[test]
    // fn test_mandelstam() {
    //     let event = test_event();
    //     let s = Mandelstam::new([0], [], [2, 3], [1], Channel::S).unwrap();
    //     let t = Mandelstam::new([0], [], [2, 3], [1], Channel::T).unwrap();
    //     let u = Mandelstam::new([0], [], [2, 3], [1], Channel::U).unwrap();
    //     let sp = Mandelstam::new([], [0], [1], [2, 3], Channel::S).unwrap();
    //     let tp = Mandelstam::new([], [0], [1], [2, 3], Channel::T).unwrap();
    //     let up = Mandelstam::new([], [0], [1], [2, 3], Channel::U).unwrap();
    //     assert_relative_eq!(s.value(&event), 18.50401105, epsilon = f64::EPSILON.sqrt());
    //     assert_relative_eq!(s.value(&event), sp.value(&event),);
    //     assert_relative_eq!(t.value(&event), -0.19222859, epsilon = f64::EPSILON.sqrt());
    //     assert_relative_eq!(t.value(&event), tp.value(&event),);
    //     assert_relative_eq!(u.value(&event), -14.40419893, epsilon = f64::EPSILON.sqrt());
    //     assert_relative_eq!(u.value(&event), up.value(&event),);
    //     let m2_beam = test_event().get_Vec4::sum([0]).m2();
    //     let m2_recoil = test_event().get_Vec4::sum([1]).m2();
    //     let m2_res = test_event().get_Vec4::sum([2, 3]).m2();
    //     assert_relative_eq!(
    //         s.value(&event) + t.value(&event) + u.value(&event) - m2_beam - m2_recoil - m2_res,
    //         1.00,
    //         epsilon = 1e-2
    //     );
    //     // Note: not very accurate, but considering the values in test_event only go to about 3
    //     // decimal places, this is probably okay
    // }
    //
    // #[test]
    // fn test_mandelstam_display() {
    //     let s = Mandelstam::new([0], [], [2, 3], [1], Channel::S).unwrap();
    //     assert_eq!(
    //         s.to_string(),
    //         "Mandelstam(p1=[0], p2=[], p3=[2, 3], p4=[1], channel=s)"
    //     );
    // }
    //
    // #[test]
    // fn test_variable_value_on() {
    //     let dataset = test_dataset();
    //     let mass = Mass::new(vec![2, 3]);
    //
    //     let values = mass.value_on(&dataset);
    //     assert_eq!(values.len(), 1);
    //     assert_relative_eq!(values[0], 1.37437863, epsilon = f64::EPSILON.sqrt());
    // }
}
