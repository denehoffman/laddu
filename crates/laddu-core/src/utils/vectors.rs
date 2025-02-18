use std::{fmt::Display, ops::Index};

use approx::{AbsDiffEq, RelativeEq};
use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use nalgebra::{Vector3, Vector4};

use crate::Float;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:6.3}, {:6.3}, {:6.3}]", self.x, self.y, self.z)
    }
}

impl AbsDiffEq for Vec3 {
    type Epsilon = <Float as approx::AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        Float::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Float::abs_diff_eq(&self.x, &other.x, epsilon)
            && Float::abs_diff_eq(&self.y, &other.y, epsilon)
            && Float::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}
impl RelativeEq for Vec3 {
    fn default_max_relative() -> Self::Epsilon {
        Float::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        Float::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && Float::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && Float::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

impl From<Vec3> for Vector3<Float> {
    fn from(value: Vec3) -> Self {
        Vector3::new(value.x, value.y, value.z)
    }
}

impl From<Vector3<Float>> for Vec3 {
    fn from(value: Vector3<Float>) -> Self {
        Vec3::new(value.x, value.y, value.z)
    }
}

impl From<Vec<Float>> for Vec3 {
    fn from(value: Vec<Float>) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Vec3> for Vec<Float> {
    fn from(value: Vec3) -> Self {
        vec![value.x, value.y, value.z]
    }
}

impl From<[Float; 3]> for Vec3 {
    fn from(value: [Float; 3]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Vec3> for [Float; 3] {
    fn from(value: Vec3) -> Self {
        [value.x, value.y, value.z]
    }
}

impl Vec3 {
    pub fn new(x: Float, y: Float, z: Float) -> Self {
        Vec3 { x, y, z }
    }

    pub fn into_vec(&self) -> Vec<Float> {
        vec![self.x, self.y, self.z]
    }

    pub const fn zero() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub const fn x() -> Self {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub const fn y() -> Self {
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    pub const fn z() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    pub fn px(&self) -> Float {
        self.x
    }

    pub fn py(&self) -> Float {
        self.y
    }

    pub fn pz(&self) -> Float {
        self.z
    }

    pub fn with_mass(&self, mass: Float) -> Vec4 {
        let e = Float::sqrt(mass.powi(2) + self.mag2());
        Vec4::new(self.px(), self.py(), self.pz(), e)
    }

    pub fn with_energy(&self, energy: Float) -> Vec4 {
        Vec4::new(self.px(), self.py(), self.pz(), energy)
    }

    pub fn dot(&self, other: &Vec3) -> Float {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - other.y * self.z,
            self.z * other.x - other.z * self.x,
            self.x * other.y - other.x * self.y,
        )
    }

    pub fn mag(&self) -> Float {
        Float::sqrt(self.mag2())
    }

    pub fn mag2(&self) -> Float {
        self.dot(self)
    }

    pub fn costheta(&self) -> Float {
        self.z / self.mag()
    }

    pub fn theta(&self) -> Float {
        Float::acos(self.costheta())
    }

    pub fn phi(&self) -> Float {
        Float::atan2(self.y, self.x)
    }

    pub fn unit(&self) -> Vec3 {
        let mag = self.mag();
        Vec3::new(self.x / mag, self.y / mag, self.z / mag)
    }
}

impl<'a> std::iter::Sum<&'a Vec3> for Vec3 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}
impl std::iter::Sum<Vec3> for Vec3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl_op_ex!(+ |a: &Vec3, b: &Vec3| -> Vec3 { Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z) });
impl_op_ex!(-|a: &Vec3, b: &Vec3| -> Vec3 { Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z) });
impl_op_ex!(-|a: &Vec3| -> Vec3 { Vec3::new(-a.x, -a.y, -a.z) });
impl_op_ex_commutative!(+ |a: &Vec3, b: &Float| -> Vec3 { Vec3::new(a.x + b, a.y + b, a.z + b) });
impl_op_ex_commutative!(-|a: &Vec3, b: &Float| -> Vec3 { Vec3::new(a.x - b, a.y - b, a.z - b) });
impl_op_ex_commutative!(*|a: &Vec3, b: &Float| -> Vec3 { Vec3::new(a.x * b, a.y * b, a.z * b) });
impl_op_ex!(/ |a: &Vec3, b: &Float| -> Vec3 { Vec3::new(a.x / b, a.y / b, a.z / b) });

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vec4 {
    pub x: Float,
    pub y: Float,
    pub z: Float,
    pub t: Float,
}

impl Display for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:6.3}, {:6.3}, {:6.3}; {:6.3}]",
            self.x, self.y, self.z, self.t
        )
    }
}

impl AbsDiffEq for Vec4 {
    type Epsilon = <Float as approx::AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        Float::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Float::abs_diff_eq(&self.x, &other.x, epsilon)
            && Float::abs_diff_eq(&self.y, &other.y, epsilon)
            && Float::abs_diff_eq(&self.z, &other.z, epsilon)
            && Float::abs_diff_eq(&self.t, &other.t, epsilon)
    }
}
impl RelativeEq for Vec4 {
    fn default_max_relative() -> Self::Epsilon {
        Float::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        Float::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && Float::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && Float::relative_eq(&self.z, &other.z, epsilon, max_relative)
            && Float::relative_eq(&self.t, &other.t, epsilon, max_relative)
    }
}

impl From<Vec4> for Vector4<Float> {
    fn from(value: Vec4) -> Self {
        Vector4::new(value.x, value.y, value.z, value.t)
    }
}

impl From<Vector4<Float>> for Vec4 {
    fn from(value: Vector4<Float>) -> Self {
        Vec4::new(value.x, value.y, value.z, value.w)
    }
}

impl From<Vec<Float>> for Vec4 {
    fn from(value: Vec<Float>) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
            t: value[3],
        }
    }
}

impl From<Vec4> for Vec<Float> {
    fn from(value: Vec4) -> Self {
        vec![value.x, value.y, value.z, value.t]
    }
}

impl From<[Float; 4]> for Vec4 {
    fn from(value: [Float; 4]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
            t: value[3],
        }
    }
}

impl From<Vec4> for [Float; 4] {
    fn from(value: Vec4) -> Self {
        [value.x, value.y, value.z, value.t]
    }
}

impl Vec4 {
    pub fn new(x: Float, y: Float, z: Float, t: Float) -> Self {
        Vec4 { x, y, z, t }
    }

    pub fn px(&self) -> Float {
        self.x
    }

    pub fn py(&self) -> Float {
        self.y
    }

    pub fn pz(&self) -> Float {
        self.z
    }

    pub fn e(&self) -> Float {
        self.t
    }

    pub fn momentum(&self) -> Vec3 {
        self.vec3()
    }

    pub fn gamma(&self) -> Float {
        let beta = self.beta();
        let b2 = beta.dot(&beta);
        1.0 / Float::sqrt(1.0 - b2)
    }

    pub fn beta(&self) -> Vec3 {
        self.momentum() / self.e()
    }

    pub fn m(&self) -> Float {
        self.mag()
    }

    pub fn m2(&self) -> Float {
        self.mag2()
    }

    /// Pretty-prints the four-momentum.
    pub fn to_p4_string(&self) -> String {
        format!(
            "[e = {:.5}; p = ({:.5}, {:.5}, {:.5}); m = {:.5}]",
            self.e(),
            self.px(),
            self.py(),
            self.pz(),
            self.m()
        )
    }

    pub fn mag(&self) -> Float {
        Float::sqrt(self.mag2())
    }

    pub fn mag2(&self) -> Float {
        self.t * self.t - (self.x * self.x + self.y * self.y + self.z * self.z)
    }

    pub fn boost(&self, beta: &Vec3) -> Self {
        let b2 = beta.dot(beta);
        let gamma = 1.0 / Float::sqrt(1.0 - b2);
        let p3 = self.vec3() + beta * ((gamma - 1.0) * self.vec3().dot(beta) / b2 + gamma * self.t);
        Vec4::new(p3.x, p3.y, p3.z, gamma * (self.t + beta.dot(&self.vec3())))
    }

    pub fn vec3(&self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl_op_ex!(+ |a: &Vec4, b: &Vec4| -> Vec4 { Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.t + b.t) });
impl_op_ex!(-|a: &Vec4, b: &Vec4| -> Vec4 {
    Vec4::new(a.x - b.x, a.y - b.y, a.z - b.z, a.t - b.t)
});
impl_op_ex!(-|a: &Vec4| -> Vec4 { Vec4::new(-a.x, -a.y, -a.z, a.t) });
impl_op_ex_commutative!(+ |a: &Vec4, b: &Float| -> Vec4 { Vec4::new(a.x + b, a.y + b, a.z + b, a.t) });
impl_op_ex_commutative!(-|a: &Vec4, b: &Float| -> Vec4 {
    Vec4::new(a.x - b, a.y - b, a.z - b, a.t)
});
impl_op_ex_commutative!(*|a: &Vec4, b: &Float| -> Vec4 {
    Vec4::new(a.x * b, a.y * b, a.z * b, a.t)
});
impl_op_ex!(/ |a: &Vec4, b: &Float| -> Vec4 { Vec4::new(a.x / b, a.y / b, a.z / b, a.t) });

impl<'a> std::iter::Sum<&'a Vec4> for Vec4 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::new(0.0, 0.0, 0.0, 0.0), |a, b| a + b)
    }
}

impl std::iter::Sum<Vec4> for Vec4 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(0.0, 0.0, 0.0, 0.0), |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_three_to_four_momentum_conversion() {
        let p3 = Vec3::new(1.0, 2.0, 3.0);
        let target_p4 = Vec4::new(1.0, 2.0, 3.0, 10.0);
        let p4_from_mass = p3.with_mass(target_p4.m());
        assert_eq!(target_p4.e(), p4_from_mass.e());
        assert_eq!(target_p4.px(), p4_from_mass.px());
        assert_eq!(target_p4.py(), p4_from_mass.py());
        assert_eq!(target_p4.pz(), p4_from_mass.pz());
        let p4_from_energy = p3.with_energy(target_p4.e());
        assert_eq!(target_p4.e(), p4_from_energy.e());
        assert_eq!(target_p4.px(), p4_from_energy.px());
        assert_eq!(target_p4.py(), p4_from_energy.py());
        assert_eq!(target_p4.pz(), p4_from_energy.pz());
    }

    #[test]
    fn test_four_momentum_basics() {
        let p = Vec4::new(3.0, 4.0, 5.0, 10.0);
        assert_eq!(p.e(), 10.0);
        assert_eq!(p.px(), 3.0);
        assert_eq!(p.py(), 4.0);
        assert_eq!(p.pz(), 5.0);
        assert_eq!(p.momentum().px(), 3.0);
        assert_eq!(p.momentum().py(), 4.0);
        assert_eq!(p.momentum().pz(), 5.0);
        assert_relative_eq!(p.beta().x, 0.3);
        assert_relative_eq!(p.beta().y, 0.4);
        assert_relative_eq!(p.beta().z, 0.5);
        assert_relative_eq!(p.m2(), 50.0);
        assert_relative_eq!(p.m(), Float::sqrt(50.0));
        assert_eq!(
            format!("{}", p.to_p4_string()),
            "[e = 10.00000; p = (3.00000, 4.00000, 5.00000); m = 7.07107]"
        );
    }

    #[test]
    fn test_three_momentum_basics() {
        let p = Vec4::new(3.0, 4.0, 5.0, 10.0);
        let q = Vec4::new(1.2, -3.4, 7.6, 0.0);
        let p3_view = p.momentum();
        let q3_view = q.momentum();
        assert_eq!(p3_view.px(), 3.0);
        assert_eq!(p3_view.py(), 4.0);
        assert_eq!(p3_view.pz(), 5.0);
        assert_relative_eq!(p3_view.mag2(), 50.0);
        assert_relative_eq!(p3_view.mag(), Float::sqrt(50.0));
        assert_relative_eq!(p3_view.costheta(), 5.0 / Float::sqrt(50.0));
        assert_relative_eq!(p3_view.theta(), Float::acos(5.0 / Float::sqrt(50.0)));
        assert_relative_eq!(p3_view.phi(), Float::atan2(4.0, 3.0));
        assert_relative_eq!(
            p3_view.unit(),
            Vec3::new(
                3.0 / Float::sqrt(50.0),
                4.0 / Float::sqrt(50.0),
                5.0 / Float::sqrt(50.0)
            )
        );
        assert_relative_eq!(p3_view.cross(&q3_view), Vec3::new(47.4, -16.8, -15.0));
    }

    #[test]
    fn test_boost_com() {
        let p = Vec4::new(3.0, 4.0, 5.0, 10.0);
        let zero = p.boost(&-p.beta()).momentum();
        assert_relative_eq!(zero, Vec3::zero());
    }

    #[test]
    fn test_boost() {
        let p1 = Vec4::new(3.0, 4.0, 5.0, 10.0);
        let p2 = Vec4::new(3.4, 2.3, 1.2, 9.0);
        let p1_boosted = p1.boost(&-p2.beta());
        assert_relative_eq!(
            p1_boosted.e(),
            8.157632144622882,
            epsilon = Float::EPSILON.sqrt()
        );
        assert_relative_eq!(
            p1_boosted.px(),
            -0.6489200627053444,
            epsilon = Float::EPSILON.sqrt()
        );
        assert_relative_eq!(
            p1_boosted.py(),
            1.5316128987581492,
            epsilon = Float::EPSILON.sqrt()
        );
        assert_relative_eq!(
            p1_boosted.pz(),
            3.712145860221643,
            epsilon = Float::EPSILON.sqrt()
        );
    }
}
