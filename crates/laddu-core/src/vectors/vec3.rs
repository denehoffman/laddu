use std::fmt::Display;

use approx::{AbsDiffEq, RelativeEq};
use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::vec4::Vec4;

/// A vector with three components.
///
/// # Examples
/// ```rust
/// use laddu_core::vectors::Vec3;
///
/// let cross = Vec3::x().cross(&Vec3::y());
/// assert_eq!(cross, Vec3::z());
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    /// The x-component of the vector
    pub x: f64,
    /// The y-component of the vector
    pub y: f64,
    /// The z-component of the vector
    pub z: f64,
}

impl Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:6.3}, {:6.3}, {:6.3}]", self.x, self.y, self.z)
    }
}

impl AbsDiffEq for Vec3 {
    type Epsilon = <f64 as approx::AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.x, &other.x, epsilon)
            && f64::abs_diff_eq(&self.y, &other.y, epsilon)
            && f64::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}
impl RelativeEq for Vec3 {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && f64::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && f64::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

impl From<Vec3> for Vector3<f64> {
    fn from(value: Vec3) -> Self {
        Vector3::new(value.x, value.y, value.z)
    }
}

impl From<Vector3<f64>> for Vec3 {
    fn from(value: Vector3<f64>) -> Self {
        Vec3::new(value.x, value.y, value.z)
    }
}

impl From<Vec<f64>> for Vec3 {
    fn from(value: Vec<f64>) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Vec3> for Vec<f64> {
    fn from(value: Vec3) -> Self {
        vec![value.x, value.y, value.z]
    }
}

impl From<[f64; 3]> for Vec3 {
    fn from(value: [f64; 3]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Vec3> for [f64; 3] {
    fn from(value: Vec3) -> Self {
        [value.x, value.y, value.z]
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3::zero()
    }
}

impl Vec3 {
    /// Create a new 3-vector from its components
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    /// Create a zero vector
    pub const fn zero() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a unit vector pointing in the x-direction
    pub const fn x() -> Self {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a unit vector pointing in the y-direction
    pub const fn y() -> Self {
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    /// Create a unit vector pointing in the z-direction
    pub const fn z() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    /// Momentum in the x-direction
    pub fn px(&self) -> f64 {
        self.x
    }

    /// Momentum in the y-direction
    pub fn py(&self) -> f64 {
        self.y
    }

    /// Momentum in the z-direction
    pub fn pz(&self) -> f64 {
        self.z
    }

    /// Create a [`Vec4`] with this vector as the 3-momentum and the given mass
    pub fn with_mass(&self, mass: f64) -> Vec4 {
        let e = f64::sqrt(mass.powi(2) + self.mag2());
        Vec4::new(self.px(), self.py(), self.pz(), e)
    }

    /// Create a [`Vec4`] with this vector as the 3-momentum and the given energy
    pub fn with_energy(&self, energy: f64) -> Vec4 {
        Vec4::new(self.px(), self.py(), self.pz(), energy)
    }

    /// Compute the dot product of this [`Vec3`] and another
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Compute the cross product of this [`Vec3`] and another
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - other.y * self.z,
            self.z * other.x - other.z * self.x,
            self.x * other.y - other.x * self.y,
        )
    }

    /// The magnitude of the vector
    pub fn mag(&self) -> f64 {
        f64::sqrt(self.mag2())
    }

    /// The squared magnitude of the vector
    pub fn mag2(&self) -> f64 {
        self.dot(self)
    }

    /// The cosine of the polar angle $`\theta`$
    pub fn costheta(&self) -> f64 {
        self.z / self.mag()
    }

    /// The polar angle $`\theta`$
    pub fn theta(&self) -> f64 {
        f64::acos(self.costheta())
    }

    /// The azimuthal angle $`\phi`$
    pub fn phi(&self) -> f64 {
        f64::atan2(self.y, self.x)
    }

    /// Create a unit vector in the same direction as this [`Vec3`]
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
impl_op_ex_commutative!(+ |a: &Vec3, b: &f64| -> Vec3 { Vec3::new(a.x + b, a.y + b, a.z + b) });
impl_op_ex_commutative!(-|a: &Vec3, b: &f64| -> Vec3 { Vec3::new(a.x - b, a.y - b, a.z - b) });
impl_op_ex_commutative!(*|a: &Vec3, b: &f64| -> Vec3 { Vec3::new(a.x * b, a.y * b, a.z * b) });
impl_op_ex!(/ |a: &Vec3, b: &f64| -> Vec3 { Vec3::new(a.x / b, a.y / b, a.z / b) });
