use crate::{
    utils::{
        enums::{Channel, Frame, Topology},
        list_to_name,
        vectors::Vec4,
    },
    Vec3,
};
use polars::prelude::*;

pub fn mass<I, S>(constituents: I) -> Expr
where
    I: IntoIterator<Item = S> + Clone,
    S: Into<PlSmallStr>,
{
    Vec4::sum(constituents.clone())
        .mag()
        .alias(format!("mass({})", list_to_name(&constituents)))
}

pub fn costheta<Ir, Id, Ires, Sb, Sr, Sd, Sres>(
    beam: Sb,
    recoil: Ir,
    daughter: Id,
    resonance: Ires,
    frame: Frame,
) -> Expr
where
    Sb: Into<PlSmallStr> + Clone,
    Ir: IntoIterator<Item = Sr> + Clone,
    Sr: Into<PlSmallStr>,
    Id: IntoIterator<Item = Sd> + Clone,
    Sd: Into<PlSmallStr>,
    Ires: IntoIterator<Item = Sres> + Clone,
    Sres: Into<PlSmallStr>,
{
    let name = format!(
        "costheta({}, [{}], [{}], [{}], {})",
        beam.clone().into(),
        list_to_name(&recoil),
        list_to_name(&daughter),
        list_to_name(&resonance),
        frame
    );
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
            daughter_res.vec3().unit().dot(&z).alias(name)
        }
        Frame::GottfriedJackson => {
            let beam_res = beam.boost(&resonance_boost);
            let z = beam_res.vec3().unit();
            daughter_res.vec3().unit().dot(&z).alias(name)
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
    Sb: Into<PlSmallStr> + Clone,
    Ir: IntoIterator<Item = Sr> + Clone,
    Sr: Into<PlSmallStr>,
    Id: IntoIterator<Item = Sd> + Clone,
    Sd: Into<PlSmallStr>,
    Ires: IntoIterator<Item = Sres> + Clone,
    Sres: Into<PlSmallStr>,
{
    let name = format!(
        "phi({}, [{}], [{}], [{}], {})",
        beam.clone().into(),
        list_to_name(&recoil),
        list_to_name(&daughter),
        list_to_name(&resonance),
        frame
    );
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
    p.dot(&y).arctan2(p.dot(&x)).alias(name)
}

pub fn angles<Ir, Id, Ires, Sb, Sr, Sd, Sres>(
    beam: Sb,
    recoil: Ir,
    daughter: Id,
    resonance: Ires,
    frame: Frame,
) -> [Expr; 2]
where
    Sb: Into<PlSmallStr> + Clone,
    Ir: IntoIterator<Item = Sr> + Clone,
    Sr: Into<PlSmallStr>,
    Id: IntoIterator<Item = Sd> + Clone,
    Sd: Into<PlSmallStr>,
    Ires: IntoIterator<Item = Sres> + Clone,
    Sres: Into<PlSmallStr>,
{
    let name_costheta = format!(
        "costheta({}, [{}], [{}], [{}], {})",
        beam.clone().into(),
        list_to_name(&recoil),
        list_to_name(&daughter),
        list_to_name(&resonance),
        frame
    );
    let name_phi = format!(
        "phi({}, [{}], [{}], [{}], {})",
        beam.clone().into(),
        list_to_name(&recoil),
        list_to_name(&daughter),
        list_to_name(&resonance),
        frame
    );
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
    [
        p.unit().dot(&z).alias(name_costheta),
        p.dot(&y).arctan2(p.dot(&x)).alias(name_phi),
    ]
}

pub fn pol_angle<Ir, Sb, Sr, Sphi>(beam: Sb, recoil: Ir, phi: Sphi) -> Expr
where
    Sb: Into<PlSmallStr> + Clone,
    Ir: IntoIterator<Item = Sr> + Clone,
    Sr: Into<PlSmallStr>,
    Sphi: Into<PlSmallStr> + Clone,
{
    let name = format!(
        "pol_angle({}, [{}], {})",
        beam.clone().into(),
        list_to_name(&recoil),
        phi.clone().into()
    );
    let beam_u3 = Vec4::new(beam).vec3().unit();
    let recoil_u3 = Vec4::sum(recoil).vec3().unit();
    let phi = col(phi).cast(DataType::Float64);
    let eps = Vec3::from([phi.clone().cos(), phi.sin(), lit(0.0)]);
    let y = beam_u3.cross(&-recoil_u3);
    y.dot(&eps).arctan2(beam_u3.dot(&eps.cross(&y))).alias(name)
}

pub fn pol_magnitude<Spgamma>(p_gamma: Spgamma) -> Expr
where
    Spgamma: Into<PlSmallStr> + Clone,
{
    let name = format!("pol_magnitude({})", p_gamma.clone().into(),);
    col(p_gamma).cast(DataType::Float64).alias(name)
}

pub fn polarization<Ir, Sb, Sr, Spgamma, Sphi>(
    beam: Sb,
    recoil: Ir,
    p_gamma: Spgamma,
    phi: Sphi,
) -> [Expr; 2]
where
    Sb: Into<PlSmallStr> + Clone,
    Ir: IntoIterator<Item = Sr> + Clone,
    Sr: Into<PlSmallStr> + Clone,
    Spgamma: Into<PlSmallStr> + Clone,
    Sphi: Into<PlSmallStr> + Clone,
{
    [pol_magnitude(p_gamma), pol_angle(beam, recoil, phi)]
}

pub fn mandelstam(topology: Topology, channel: Channel) -> Expr {
    let name = format!("mandelstam({}, {})", topology, channel);
    match channel {
        Channel::S => match topology {
            Topology::All { p1, p2, .. }
            | Topology::MissingP3 { p1, p2, .. }
            | Topology::MissingP4 { p1, p2, .. } => Vec4::sum(p1).add(&Vec4::sum(p2)).mag2(),
            Topology::MissingP1 { p3, p4, .. } | Topology::MissingP2 { p3, p4, .. } => {
                Vec4::sum(p3).add(&Vec4::sum(p4)).mag2()
            }
        },
        Channel::T => match topology {
            Topology::All { p1, p3, .. }
            | Topology::MissingP2 { p1, p3, .. }
            | Topology::MissingP4 { p1, p3, .. } => Vec4::sum(p1).sub(&Vec4::sum(p3)).mag2(),
            Topology::MissingP1 { p2, p4, .. } | Topology::MissingP3 { p2, p4, .. } => {
                Vec4::sum(p4).sub(&Vec4::sum(p2)).mag2()
            }
        },
        Channel::U => match topology {
            Topology::All { p1, p4, .. }
            | Topology::MissingP2 { p1, p4, .. }
            | Topology::MissingP3 { p1, p4, .. } => Vec4::sum(p1).sub(&Vec4::sum(p4)).mag2(),
            Topology::MissingP1 { p2, p3, .. } | Topology::MissingP4 { p2, p3, .. } => {
                Vec4::sum(p3).sub(&Vec4::sum(p2)).mag2()
            }
        },
    }
    .alias(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_dataset;
    use crate::tests::val1;
    use approx::assert_relative_eq;
    #[test]
    fn test_mass_single_particle() {
        let lf = test_dataset().lf.lazy();
        let res = lf.with_column(mass(["proton"])).collect().unwrap();
        assert_relative_eq!(val1(&res, "mass(proton)"), 1.0069946154771634);
    }
    #[test]
    fn test_mass_multiple_particles() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_column(mass(["kshort1", "kshort2"]))
            .collect()
            .unwrap();
        assert_relative_eq!(val1(&res, "mass(kshort1, kshort2)"), 1.3743574213427878);
    }
    #[test]
    fn test_costheta_helicity() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_column(costheta(
                "beam",
                ["proton"],
                ["kshort1"],
                ["kshort1", "kshort2"],
                Frame::Helicity,
            ))
            .collect()
            .unwrap();
        assert_relative_eq!(
            val1(
                &res,
                "costheta(beam, [proton], [kshort1], [kshort1, kshort2], Helicity)"
            ),
            -0.4611206717087849
        );
    }
    #[test]
    fn test_phi_helicity() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_column(phi(
                "beam",
                ["proton"],
                ["kshort1"],
                ["kshort1", "kshort2"],
                Frame::Helicity,
            ))
            .collect()
            .unwrap();
        assert_relative_eq!(
            val1(
                &res,
                "phi(beam, [proton], [kshort1], [kshort1, kshort2], Helicity)"
            ),
            -2.6574625881071583
        );
    }
    #[test]
    fn test_costheta_gottfried_jackson() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_column(costheta(
                "beam",
                ["proton"],
                ["kshort1"],
                ["kshort1", "kshort2"],
                Frame::GottfriedJackson,
            ))
            .collect()
            .unwrap();
        assert_relative_eq!(
            val1(
                &res,
                "costheta(beam, [proton], [kshort1], [kshort1, kshort2], Gottfried-Jackson)"
            ),
            0.0919915889924921,
        );
    }
    #[test]
    fn test_phi_gottfried_jackson() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_column(phi(
                "beam",
                ["proton"],
                ["kshort1"],
                ["kshort1", "kshort2"],
                Frame::GottfriedJackson,
            ))
            .collect()
            .unwrap();
        assert_relative_eq!(
            val1(
                &res,
                "phi(beam, [proton], [kshort1], [kshort1, kshort2], Gottfried-Jackson)"
            ),
            -2.7139139065178943
        );
    }
    #[test]
    fn test_angles() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_columns(angles(
                "beam",
                ["proton"],
                ["kshort1"],
                ["kshort1", "kshort2"],
                Frame::Helicity,
            ))
            .collect()
            .unwrap();
        assert_relative_eq!(
            val1(
                &res,
                "costheta(beam, [proton], [kshort1], [kshort1, kshort2], Helicity)"
            ),
            -0.4611206717087849
        );
        assert_relative_eq!(
            val1(
                &res,
                "phi(beam, [proton], [kshort1], [kshort1, kshort2], Helicity)"
            ),
            -2.6574625881071583
        );
    }
    #[test]
    fn test_pol_angle() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_column(pol_angle("beam", ["proton"], "pol_angle"))
            .collect()
            .unwrap();
        assert_relative_eq!(
            val1(&res, "pol_angle(beam, [proton], pol_angle)"),
            1.9359299078186731,
        );
    }
    #[test]
    fn test_pol_magnitude() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_column(pol_magnitude("pol_magnitude"))
            .collect()
            .unwrap();
        assert_relative_eq!(val1(&res, "pol_magnitude(pol_magnitude)"), 0.385628);
    }
    #[test]
    fn test_polarization() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_columns(polarization(
                "beam",
                ["proton"],
                "pol_magnitude",
                "pol_angle",
            ))
            .collect()
            .unwrap();
        assert_relative_eq!(
            val1(&res, "pol_angle(beam, [proton], pol_angle)"),
            1.9359299078186731,
        );
        assert_relative_eq!(val1(&res, "pol_magnitude(pol_magnitude)"), 0.385628);
    }
    #[test]
    fn test_mandelstam() {
        let lf = test_dataset().lf.lazy();
        let res = lf
            .with_columns([
                mandelstam(
                    Topology::missing_p2(["beam"], ["kshort1", "kshort2"], ["proton"]),
                    Channel::S,
                ),
                mandelstam(
                    Topology::missing_p2(["beam"], ["kshort1", "kshort2"], ["proton"]),
                    Channel::T,
                ),
                mandelstam(
                    Topology::missing_p2(["beam"], ["kshort1", "kshort2"], ["proton"]),
                    Channel::U,
                ),
                mandelstam(
                    Topology::missing_p1(["beam"], ["proton"], ["kshort1", "kshort2"]),
                    Channel::S,
                ),
                mandelstam(
                    Topology::missing_p1(["beam"], ["proton"], ["kshort1", "kshort2"]),
                    Channel::T,
                ),
                mandelstam(
                    Topology::missing_p1(["beam"], ["proton"], ["kshort1", "kshort2"]),
                    Channel::U,
                ),
                mass(["beam"]),
                mass(["kshort1", "kshort2"]),
                mass(["proton"]),
            ])
            .collect()
            .unwrap();
        let s = val1(
            &res,
            "mandelstam(Topology::MissingP2(p1: [beam], p3: [kshort1, kshort2], p4: [proton]), s)",
        );
        let t = val1(
            &res,
            "mandelstam(Topology::MissingP2(p1: [beam], p3: [kshort1, kshort2], p4: [proton]), t)",
        );
        let u = val1(
            &res,
            "mandelstam(Topology::MissingP2(p1: [beam], p3: [kshort1, kshort2], p4: [proton]), u)",
        );
        let sp = val1(
            &res,
            "mandelstam(Topology::MissingP1(p2: [beam], p3: [proton], p4: [kshort1, kshort2]), s)",
        );
        let tp = val1(
            &res,
            "mandelstam(Topology::MissingP1(p2: [beam], p3: [proton], p4: [kshort1, kshort2]), t)",
        );
        let up = val1(
            &res,
            "mandelstam(Topology::MissingP1(p2: [beam], p3: [proton], p4: [kshort1, kshort2]), u)",
        );
        let m2_beam = val1(&res, "mass(beam)").powi(2);
        let m2_proton = val1(&res, "mass(proton)").powi(2);
        let m2_res = val1(&res, "mass(kshort1, kshort2)").powi(2);
        assert_relative_eq!(s, 18.50384948999998);
        assert_relative_eq!(s, sp);
        assert_relative_eq!(t, -0.1922279184000001);
        assert_relative_eq!(t, tp);
        assert_relative_eq!(u, -14.404123804400015);
        assert_relative_eq!(u, up);
        assert_relative_eq!(
            s + t + u - m2_beam - m2_proton - m2_res,
            1.0,
            epsilon = 1e-2
        );
        // Note: not very accurate, but considering the values in test_event only go to about 3
        // decimal places, this is probably okay
    }
}
