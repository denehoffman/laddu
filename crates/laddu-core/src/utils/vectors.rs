use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use polars::prelude::*;

pub struct Vec3([Expr; 3]);
impl Vec3 {
    pub fn new<S: Into<PlSmallStr>>(name: S) -> Self {
        let name: PlSmallStr = name.into();
        Self([
            col(format!("{}_x", name)),
            col(format!("{}_y", name)),
            col(format!("{}_z", name)),
        ])
    }
    pub fn alias<S: AsRef<str>>(&self, base: S) -> [Expr; 3] {
        let b = base.as_ref();
        [
            self.0[0].clone().alias(format!("{b}_x")),
            self.0[1].clone().alias(format!("{b}_y")),
            self.0[2].clone().alias(format!("{b}_z")),
        ]
    }
    pub fn x(&self) -> Expr {
        self.0[0].clone()
    }
    pub fn y(&self) -> Expr {
        self.0[1].clone()
    }
    pub fn z(&self) -> Expr {
        self.0[2].clone()
    }

    pub fn with_mass(&self, mass: f64) -> Vec4 {
        let e = (lit(mass.powi(2)) + self.mag2()).sqrt();
        Vec4([self.x(), self.y(), self.z(), e])
    }

    pub fn with_energy(&self, energy: f64) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), lit(energy)])
    }

    pub fn dot(&self, other: &Self) -> Expr {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
    pub fn cross(&self, other: &Self) -> Self {
        Self([
            self.y() * other.z() - self.z() * other.y(),
            self.z() * other.x() - self.x() * other.z(),
            self.x() * other.y() - self.y() * other.x(),
        ])
    }
    pub fn mag2(&self) -> Expr {
        self.dot(self)
    }
    pub fn mag(&self) -> Expr {
        self.mag2().sqrt()
    }
    pub fn costheta(&self) -> Expr {
        self.z() / self.mag()
    }
    pub fn theta(&self) -> Expr {
        self.costheta().arccos()
    }
    pub fn phi(&self) -> Expr {
        self.y().arctan2(self.x())
    }
    pub fn unit(&self) -> Self {
        Self([
            self.x() / self.mag(),
            self.y() / self.mag(),
            self.z() / self.mag(),
        ])
    }
    pub fn add(&self, other: &Self) -> Self {
        Self([
            self.x() + other.x(),
            self.y() + other.y(),
            self.z() + other.z(),
        ])
    }
    pub fn scalar_add(&self, other: &Expr) -> Self {
        Self([
            self.x() + other.clone(),
            self.y() + other.clone(),
            self.z() + other.clone(),
        ])
    }
    pub fn sub(&self, other: &Self) -> Self {
        Self([
            self.x() - other.x(),
            self.y() - other.y(),
            self.z() - other.z(),
        ])
    }
    pub fn scalar_sub(&self, other: &Expr) -> Self {
        Self([
            self.x() - other.clone(),
            self.y() - other.clone(),
            self.z() - other.clone(),
        ])
    }
    pub fn scale(&self, other: &Expr) -> Self {
        Self([
            self.x() * other.clone(),
            self.y() * other.clone(),
            self.z() * other.clone(),
        ])
    }
    pub fn unscale(&self, other: &Expr) -> Self {
        Self([
            self.x() / other.clone(),
            self.y() / other.clone(),
            self.z() / other.clone(),
        ])
    }
    pub fn neg(&self) -> Self {
        Self([-self.x(), -self.y(), -self.z()])
    }
}

impl_op_ex!(+ |a: &Vec3, b: &Vec3| -> Vec3 { a.add(b) });
impl_op_ex!(-|a: &Vec3, b: &Vec3| -> Vec3 { a.sub(b) });
impl_op_ex!(-|a: &Vec3| -> Vec3 { a.neg() });
impl_op_ex_commutative!(+ |a: &Vec3, b: &Expr| -> Vec3 { a.scalar_add(b) });
impl_op_ex_commutative!(-|a: &Vec3, b: &Expr| -> Vec3 { a.scalar_sub(b) });
impl_op_ex_commutative!(*|a: &Vec3, b: &Expr| -> Vec3 { a.scale(b) });
impl_op_ex!(/ |a: &Vec3, b: &Expr| -> Vec3 { a.unscale(b) });

pub struct Vec4([Expr; 4]);
impl Vec4 {
    pub fn new<S: Into<PlSmallStr>>(name: S) -> Self {
        let name: PlSmallStr = name.into();
        Self([
            col(format!("{}_px", name)),
            col(format!("{}_py", name)),
            col(format!("{}_pz", name)),
            col(format!("{}_e", name)),
        ])
    }

    pub fn sum<I, S>(constituents: I) -> Vec4
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let mut it = constituents.into_iter();
        let mut total = if let Some(first) = it.next() {
            Vec4::new(first)
        } else {
            Vec4([lit(0.0), lit(0.0), lit(0.0), lit(0.0)])
        };
        for n in it {
            total = total.add(&Vec4::new(n));
        }
        total
    }

    pub fn alias<S: AsRef<str>>(&self, base: S) -> [Expr; 4] {
        let b = base.as_ref();
        [
            self.0[0].clone().alias(format!("{b}_px")),
            self.0[1].clone().alias(format!("{b}_py")),
            self.0[2].clone().alias(format!("{b}_pz")),
            self.0[3].clone().alias(format!("{b}_e")),
        ]
    }
    pub fn px(&self) -> Expr {
        self.0[0].clone()
    }
    pub fn py(&self) -> Expr {
        self.0[1].clone()
    }
    pub fn pz(&self) -> Expr {
        self.0[2].clone()
    }
    pub fn e(&self) -> Expr {
        self.0[3].clone()
    }
    // let's get rid of "momentum"
    pub fn vec3(&self) -> Vec3 {
        Vec3([self.px(), self.py(), self.pz()])
    }
    pub fn beta(&self) -> Vec3 {
        self.vec3().unscale(&self.e())
    }
    pub fn gamma(&self) -> Expr {
        let e = self.e();
        let e2 = e.clone() * e.clone();
        let p2 = self.vec3().mag2();
        e / (e2 - p2).sqrt()
    }
    // let's also get rid of m and m2, unix philosophy
    pub fn mag2(&self) -> Expr {
        self.e() * self.e() - self.vec3().mag2()
    }
    pub fn mag(&self) -> Expr {
        self.mag2().sqrt()
    }
    pub fn boost(&self, beta: &Vec3) -> Self {
        let b2 = beta.dot(beta);
        let gamma = lit(1.0) / (lit(1.0) - b2.clone()).sqrt();
        let p3 = self.vec3()
            + (beta
                * (&((gamma.clone() - lit(1.0)) * self.vec3().dot(beta) / b2
                    + gamma.clone() * self.e())));
        Self([
            p3.x(),
            p3.y(),
            p3.z(),
            gamma * (self.e() + beta.dot(&self.vec3())),
        ])
    }
    pub fn add(&self, other: &Self) -> Self {
        Self([
            self.px() + other.px(),
            self.py() + other.py(),
            self.pz() + other.pz(),
            self.e() + other.e(),
        ])
    }
    pub fn sub(&self, other: &Self) -> Self {
        Self([
            self.px() - other.px(),
            self.py() - other.py(),
            self.pz() - other.pz(),
            self.e() - other.e(),
        ])
    }
    pub fn neg(&self) -> Self {
        Self([-self.px(), -self.py(), -self.pz(), -self.e()])
    }
}

impl_op_ex!(+ |a: &Vec4, b: &Vec4| -> Vec4 { a.add(b) });
impl_op_ex!(-|a: &Vec4, b: &Vec4| -> Vec4 { a.sub(b) });
impl_op_ex!(-|a: &Vec4| -> Vec4 { a.neg() });

#[cfg(test)]
mod tests {
    // use approx::{assert_abs_diff_eq, assert_relative_eq};
    // use nalgebra::{Vector3, Vector4};
    //
    // use super::*;
    //
    // #[test]
    // fn test_vec_sums() {
    //     let df = df!(
    //         "a_x" => [1.0],
    //         "a_y" => [2.0],
    //         "a_z" => [3.0],
    //         "b_x" => [4.0],
    //         "b_y" => [5.0],
    //         "b_z" => [6.0],
    //     );
    // }
    //
    // #[test]
    // fn test_three_to_four_momentum_conversion() {
    //     let p3 = Vec3::new(1.0, 2.0, 3.0);
    //     let target_p4 = Vec4::new(1.0, 2.0, 3.0, 10.0);
    //     let p4_from_mass = p3.with_mass(target_p4.m());
    //     assert_eq!(target_p4.e(), p4_from_mass.e());
    //     assert_eq!(target_p4.px(), p4_from_mass.px());
    //     assert_eq!(target_p4.py(), p4_from_mass.py());
    //     assert_eq!(target_p4.pz(), p4_from_mass.pz());
    //     let p4_from_energy = p3.with_energy(target_p4.e());
    //     assert_eq!(target_p4.e(), p4_from_energy.e());
    //     assert_eq!(target_p4.px(), p4_from_energy.px());
    //     assert_eq!(target_p4.py(), p4_from_energy.py());
    //     assert_eq!(target_p4.pz(), p4_from_energy.pz());
    // }
    //
    // #[test]
    // fn test_four_momentum_basics() {
    //     let p = Vec4::new(3.0, 4.0, 5.0, 10.0);
    //     assert_eq!(p.e(), 10.0);
    //     assert_eq!(p.px(), 3.0);
    //     assert_eq!(p.py(), 4.0);
    //     assert_eq!(p.pz(), 5.0);
    //     assert_eq!(p.momentum().px(), 3.0);
    //     assert_eq!(p.momentum().py(), 4.0);
    //     assert_eq!(p.momentum().pz(), 5.0);
    //     assert_relative_eq!(p.beta().x, 0.3);
    //     assert_relative_eq!(p.beta().y, 0.4);
    //     assert_relative_eq!(p.beta().z, 0.5);
    //     assert_relative_eq!(p.m2(), 50.0);
    //     assert_relative_eq!(p.m(), f64::sqrt(50.0));
    //     assert_eq!(
    //         format!("{}", p.to_p4_string()),
    //         "[e = 10.00000; p = (3.00000, 4.00000, 5.00000); m = 7.07107]"
    //     );
    //     assert_relative_eq!(Vec3::x().x, 1.0);
    //     assert_relative_eq!(Vec3::x().y, 0.0);
    //     assert_relative_eq!(Vec3::x().z, 0.0);
    //     assert_relative_eq!(Vec3::y().x, 0.0);
    //     assert_relative_eq!(Vec3::y().y, 1.0);
    //     assert_relative_eq!(Vec3::y().z, 0.0);
    //     assert_relative_eq!(Vec3::z().x, 0.0);
    //     assert_relative_eq!(Vec3::z().y, 0.0);
    //     assert_relative_eq!(Vec3::z().z, 1.0);
    //     assert_relative_eq!(Vec3::default().x, 0.0);
    //     assert_relative_eq!(Vec3::default().y, 0.0);
    //     assert_relative_eq!(Vec3::default().z, 0.0);
    // }
    //
    // #[test]
    // fn test_three_momentum_basics() {
    //     let p = Vec4::new(3.0, 4.0, 5.0, 10.0);
    //     let q = Vec4::new(1.2, -3.4, 7.6, 0.0);
    //     let p3_view = p.momentum();
    //     let q3_view = q.momentum();
    //     assert_eq!(p3_view.px(), 3.0);
    //     assert_eq!(p3_view.py(), 4.0);
    //     assert_eq!(p3_view.pz(), 5.0);
    //     assert_relative_eq!(p3_view.mag2(), 50.0);
    //     assert_relative_eq!(p3_view.mag(), f64::sqrt(50.0));
    //     assert_relative_eq!(p3_view.costheta(), 5.0 / f64::sqrt(50.0));
    //     assert_relative_eq!(p3_view.theta(), f64::acos(5.0 / f64::sqrt(50.0)));
    //     assert_relative_eq!(p3_view.phi(), f64::atan2(4.0, 3.0));
    //     assert_relative_eq!(
    //         p3_view.unit(),
    //         Vec3::new(
    //             3.0 / f64::sqrt(50.0),
    //             4.0 / f64::sqrt(50.0),
    //             5.0 / f64::sqrt(50.0)
    //         )
    //     );
    //     assert_relative_eq!(p3_view.cross(&q3_view), Vec3::new(47.4, -16.8, -15.0));
    // }
    //
    // #[test]
    // fn test_vec_equality() {
    //     let p = Vec3::new(1.1, 2.2, 3.3);
    //     let p2 = Vec3::new(1.1 * 2.0, 2.2 * 2.0, 3.3 * 2.0);
    //     assert_abs_diff_eq!(p * 2.0, p2);
    //     assert_relative_eq!(p * 2.0, p2);
    //     let p = Vec4::new(1.1, 2.2, 3.3, 10.0);
    //     let p2 = Vec4::new(1.1 * 2.0, 2.2 * 2.0, 3.3 * 2.0, 10.0);
    //     assert_abs_diff_eq!(p * 2.0, p2);
    //     assert_relative_eq!(p * 2.0, p2);
    // }
    //
    // #[test]
    // fn test_boost_com() {
    //     let p = Vec4::new(3.0, 4.0, 5.0, 10.0);
    //     let zero = p.boost(&-p.beta()).momentum();
    //     assert_relative_eq!(zero, Vec3::zero());
    // }
    //
    // #[test]
    // fn test_boost() {
    //     let p0 = Vec4::new(0.0, 0.0, 0.0, 1.0);
    //     assert_relative_eq!(p0.gamma(), 1.0);
    //     let p0 = Vec4::new(f64::sqrt(3.0) / 2.0, 0.0, 0.0, 1.0);
    //     assert_relative_eq!(p0.gamma(), 2.0);
    //     let p1 = Vec4::new(3.0, 4.0, 5.0, 10.0);
    //     let p2 = Vec4::new(3.4, 2.3, 1.2, 9.0);
    //     let p1_boosted = p1.boost(&-p2.beta());
    //     assert_relative_eq!(
    //         p1_boosted.e(),
    //         8.157632144622882,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    //     assert_relative_eq!(
    //         p1_boosted.px(),
    //         -0.6489200627053444,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    //     assert_relative_eq!(
    //         p1_boosted.py(),
    //         1.5316128987581492,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    //     assert_relative_eq!(
    //         p1_boosted.pz(),
    //         3.712145860221643,
    //         epsilon = f64::EPSILON.sqrt()
    //     );
    // }
}
