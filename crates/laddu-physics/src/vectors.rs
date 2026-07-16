use std::fmt::Display;

use approx::{AbsDiffEq, RelativeEq};
use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use laddu_expr::{Expr, P4Component, atan2, event_p4_component, vector};
use nalgebra::{Vector3, Vector4};
use serde::{Deserialize, Serialize};

use crate::{LadduPhysicsError, LadduPhysicsResult};

/// A vector with three components.
///
/// # Examples
/// ```rust
/// use laddu_physics::vectors::RealVec3;
///
/// let cross = RealVec3::x().cross(&RealVec3::y());
/// assert_eq!(cross, RealVec3::z());
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealVec3 {
    /// The x-component of the vector
    pub x: f64,
    /// The y-component of the vector
    pub y: f64,
    /// The z-component of the vector
    pub z: f64,
}

impl Display for RealVec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:6.3}, {:6.3}, {:6.3}]", self.x, self.y, self.z)
    }
}

impl AbsDiffEq for RealVec3 {
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
impl RelativeEq for RealVec3 {
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

impl From<RealVec3> for Vector3<f64> {
    fn from(value: RealVec3) -> Self {
        Vector3::new(value.x, value.y, value.z)
    }
}

impl From<Vector3<f64>> for RealVec3 {
    fn from(value: Vector3<f64>) -> Self {
        RealVec3::new(value.x, value.y, value.z)
    }
}

impl TryFrom<Vec<f64>> for RealVec3 {
    type Error = LadduPhysicsError;

    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        if value.len() != 3 {
            return Err(LadduPhysicsError::custom(
                "Attempted to convert Vec<f64> to RealVec3 for Vec with len != 3",
            ));
        }
        Ok(Self {
            x: value[0],
            y: value[1],
            z: value[2],
        })
    }
}

impl From<RealVec3> for Vec<f64> {
    fn from(value: RealVec3) -> Self {
        vec![value.x, value.y, value.z]
    }
}

impl From<[f64; 3]> for RealVec3 {
    fn from(value: [f64; 3]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<RealVec3> for [f64; 3] {
    fn from(value: RealVec3) -> Self {
        [value.x, value.y, value.z]
    }
}

impl Default for RealVec3 {
    fn default() -> Self {
        RealVec3::zero()
    }
}

impl RealVec3 {
    /// Create a new 3-vector from its components
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        RealVec3 { x, y, z }
    }

    /// Create a zero vector
    pub const fn zero() -> Self {
        RealVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a unit vector pointing in the x-direction
    pub const fn x() -> Self {
        RealVec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a unit vector pointing in the y-direction
    pub const fn y() -> Self {
        RealVec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    /// Create a unit vector pointing in the z-direction
    pub const fn z() -> Self {
        RealVec3 {
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

    /// Create a [`RealVec4`] with this vector as the 3-momentum and the given mass
    pub fn with_mass(&self, mass: f64) -> RealVec4 {
        let e = f64::sqrt(mass.powi(2) + self.mag2());
        RealVec4::new(self.px(), self.py(), self.pz(), e)
    }

    /// Create a [`RealVec4`] with this vector as the 3-momentum and the given energy
    pub fn with_energy(&self, energy: f64) -> RealVec4 {
        RealVec4::new(self.px(), self.py(), self.pz(), energy)
    }

    /// Compute the dot product of this [`RealVec3`] and another
    pub fn dot(&self, other: &RealVec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Compute the cross product of this [`RealVec3`] and another
    pub fn cross(&self, other: &RealVec3) -> RealVec3 {
        RealVec3::new(
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
    pub fn costheta(&self) -> LadduPhysicsResult<f64> {
        let mag = self.mag();
        if mag <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "vector magnitude",
                "positive when calculating cos(theta)",
                mag,
            ));
        }
        Ok(self.z / self.mag())
    }

    /// The polar angle $`\theta`$
    pub fn theta(&self) -> LadduPhysicsResult<f64> {
        Ok(f64::acos(self.costheta()?))
    }

    /// The azimuthal angle $`\phi`$
    pub fn phi(&self) -> f64 {
        f64::atan2(self.y, self.x)
    }

    /// Create a unit vector in the same direction as this [`RealVec3`]
    pub fn unit(&self) -> LadduPhysicsResult<RealVec3> {
        let mag = self.mag();
        if mag <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "vector magnitude",
                "positive when constructing unit vector",
                mag,
            ));
        }
        Ok(RealVec3::new(self.x / mag, self.y / mag, self.z / mag))
    }
}

impl<'a> std::iter::Sum<&'a RealVec3> for RealVec3 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}
impl std::iter::Sum<RealVec3> for RealVec3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl_op_ex!(+ |a: &RealVec3, b: &RealVec3| -> RealVec3 { RealVec3::new(a.x + b.x, a.y + b.y, a.z + b.z) });
impl_op_ex!(-|a: &RealVec3, b: &RealVec3| -> RealVec3 {
    RealVec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
});
impl_op_ex!(-|a: &RealVec3| -> RealVec3 { RealVec3::new(-a.x, -a.y, -a.z) });
impl_op_ex_commutative!(+ |a: &RealVec3, b: &f64| -> RealVec3 { RealVec3::new(a.x + b, a.y + b, a.z + b) });
impl_op_ex_commutative!(-|a: &RealVec3, b: &f64| -> RealVec3 {
    RealVec3::new(a.x - b, a.y - b, a.z - b)
});
impl_op_ex_commutative!(*|a: &RealVec3, b: &f64| -> RealVec3 {
    RealVec3::new(a.x * b, a.y * b, a.z * b)
});
impl_op_ex!(/ |a: &RealVec3, b: &f64| -> RealVec3 { RealVec3::new(a.x / b, a.y / b, a.z / b) });

/// A four-vector (Lorentz vector) whose last component stores the energy.
///
/// # Examples
/// ```rust
/// use laddu_physics::vectors::{RealVec3, RealVec4};
///
/// let momentum = RealVec3::new(1.0, 0.0, 0.0);
/// let four_vector = momentum.with_mass(2.0);
/// assert!((four_vector.m2() - 4.0).abs() < 1e-12);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealVec4 {
    /// The x-component of the vector
    pub x: f64,
    /// The y-component of the vector
    pub y: f64,
    /// The z-component of the vector
    pub z: f64,
    /// The t-component of the vector
    pub t: f64,
}

impl Display for RealVec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:6.3}, {:6.3}, {:6.3}; {:6.3}]",
            self.x, self.y, self.z, self.t
        )
    }
}

impl AbsDiffEq for RealVec4 {
    type Epsilon = <f64 as approx::AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.x, &other.x, epsilon)
            && f64::abs_diff_eq(&self.y, &other.y, epsilon)
            && f64::abs_diff_eq(&self.z, &other.z, epsilon)
            && f64::abs_diff_eq(&self.t, &other.t, epsilon)
    }
}
impl RelativeEq for RealVec4 {
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
            && f64::relative_eq(&self.t, &other.t, epsilon, max_relative)
    }
}

impl From<RealVec4> for Vector4<f64> {
    fn from(value: RealVec4) -> Self {
        Vector4::new(value.x, value.y, value.z, value.t)
    }
}

impl From<Vector4<f64>> for RealVec4 {
    fn from(value: Vector4<f64>) -> Self {
        RealVec4::new(value.x, value.y, value.z, value.w)
    }
}

impl TryFrom<Vec<f64>> for RealVec4 {
    type Error = LadduPhysicsError;

    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        if value.len() != 4 {
            return Err(LadduPhysicsError::custom(
                "Attempted to convert Vec<f64> to RealVec4 for Vec with len != 4",
            ));
        }
        Ok(Self {
            x: value[0],
            y: value[1],
            z: value[2],
            t: value[3],
        })
    }
}

impl From<RealVec4> for Vec<f64> {
    fn from(value: RealVec4) -> Self {
        vec![value.x, value.y, value.z, value.t]
    }
}

impl From<[f64; 4]> for RealVec4 {
    fn from(value: [f64; 4]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
            t: value[3],
        }
    }
}

impl From<RealVec4> for [f64; 4] {
    fn from(value: RealVec4) -> Self {
        [value.x, value.y, value.z, value.t]
    }
}

impl RealVec4 {
    /// Create a new 4-vector from its components
    pub fn new(x: f64, y: f64, z: f64, t: f64) -> Self {
        RealVec4 { x, y, z, t }
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

    /// The energy of the 4-vector
    pub fn e(&self) -> f64 {
        self.t
    }

    /// The 3-momentum
    pub fn momentum(&self) -> RealVec3 {
        self.vec3()
    }

    /// The $`\gamma`$ factor $`\frac{1}{\sqrt{1 - \beta^2}}`$.
    pub fn gamma(&self) -> LadduPhysicsResult<f64> {
        let beta = self.beta()?;
        let b2 = beta.dot(&beta);
        if b2 >= 1.0 {
            return Err(LadduPhysicsError::invalid_value("|beta|^2", "< 1", b2));
        }
        Ok(1.0 / f64::sqrt(1.0 - b2))
    }

    /// The $`\vec{\beta}`$ vector $`\frac{\vec{p}}{E}`$.
    pub fn beta(&self) -> LadduPhysicsResult<RealVec3> {
        let e = self.e();
        if e <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "four-momentum energy",
                "positive",
                e,
            ));
        }
        Ok(self.momentum() / e)
    }

    /// The invariant mass corresponding to this 4-momentum
    pub fn m(&self) -> LadduPhysicsResult<f64> {
        self.mag()
    }

    #[inline(always)]
    pub fn m_unchecked(&self) -> f64 {
        self.m2().sqrt()
    }

    pub fn signed_m(&self) -> LadduPhysicsResult<f64> {
        self.signed_mag()
    }

    #[inline(always)]
    pub fn signed_m_unchecked(&self) -> f64 {
        self.signed_mag_unchecked()
    }

    /// The squared invariant mass corresponding to this 4-momentum
    pub fn m2(&self) -> f64 {
        self.mag2()
    }

    /// Pretty-prints the four-momentum.
    pub fn to_p4_string(&self) -> String {
        let mass = self
            .m()
            .map(|m| format!("{m:.5}"))
            .unwrap_or_else(|_| format!("{:.5}i", (-self.m2()).sqrt()));
        format!(
            "[e = {:.5}; p = ({:.5}, {:.5}, {:.5}); m = {}]",
            self.e(),
            self.px(),
            self.py(),
            self.pz(),
            mass
        )
    }

    /// The physical magnitude (with $`---+`$ signature).
    ///
    /// Returns an error for spacelike four-vectors, where `m2 < 0`.
    pub fn mag(&self) -> LadduPhysicsResult<f64> {
        let mag2 = self.mag2();

        if !mag2.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "magnitude squared",
                "finite",
                mag2,
            ));
        }

        if mag2 < 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "magnitude squared",
                "nonnegative",
                mag2,
            ));
        }

        Ok(mag2.sqrt())
    }

    /// Signed magnitude useful for diagnostics:
    ///
    /// - `sqrt(mag2)` for timelike/null vectors
    /// - `-sqrt(-mag2)` for spacelike vectors
    pub fn signed_mag(&self) -> LadduPhysicsResult<f64> {
        let mag2 = self.mag2();

        if !mag2.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "magnitude squared",
                "finite",
                mag2,
            ));
        }

        if mag2 >= 0.0 {
            Ok(mag2.sqrt())
        } else {
            Ok(-(-mag2).sqrt())
        }
    }

    #[inline(always)]
    pub fn signed_mag_unchecked(&self) -> f64 {
        let mag2 = self.mag2();
        if mag2 >= 0.0 {
            mag2.sqrt()
        } else {
            -(-mag2).sqrt()
        }
    }

    /// The squared magnitude of the vector (with $`---+`$ signature).
    pub fn mag2(&self) -> f64 {
        self.t * self.t - (self.x * self.x + self.y * self.y + self.z * self.z)
    }

    /// Gives the vector boosted along a $`\vec{\beta}`$ vector.
    pub fn boost(&self, beta: &RealVec3) -> Self {
        let b2 = beta.dot(beta);
        if b2 == 0.0 {
            return *self;
        }
        let gamma = 1.0 / f64::sqrt(1.0 - b2);
        let p3 = self.vec3() + beta * ((gamma - 1.0) * self.vec3().dot(beta) / b2 + gamma * self.t);
        RealVec4::new(p3.x, p3.y, p3.z, gamma * (self.t + beta.dot(&self.vec3())))
    }

    /// The 3-vector contained in this 4-vector
    pub fn vec3(&self) -> RealVec3 {
        RealVec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl_op_ex!(+ |a: &RealVec4, b: &RealVec4| -> RealVec4 { RealVec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.t + b.t) });
impl_op_ex!(-|a: &RealVec4, b: &RealVec4| -> RealVec4 {
    RealVec4::new(a.x - b.x, a.y - b.y, a.z - b.z, a.t - b.t)
});
impl_op_ex!(-|a: &RealVec4| -> RealVec4 { RealVec4::new(-a.x, -a.y, -a.z, a.t) });

impl<'a> std::iter::Sum<&'a RealVec4> for RealVec4 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::new(0.0, 0.0, 0.0, 0.0), |a, b| a + b)
    }
}

impl std::iter::Sum<RealVec4> for RealVec4 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(0.0, 0.0, 0.0, 0.0), |a, b| a + b)
    }
}

/// Expression-valued vector with three spatial components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: Expr,
    pub y: Expr,
    pub z: Expr,
}

impl Vec3 {
    pub fn new(x: impl Into<Expr>, y: impl Into<Expr>, z: impl Into<Expr>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    pub fn y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    pub fn z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    pub fn event(prefix: &str) -> Self {
        Self::new(
            event_p4_component(prefix, P4Component::Px),
            event_p4_component(prefix, P4Component::Py),
            event_p4_component(prefix, P4Component::Pz),
        )
    }

    pub fn px(&self) -> Expr {
        self.x.clone()
    }

    pub fn py(&self) -> Expr {
        self.y.clone()
    }

    pub fn pz(&self) -> Expr {
        self.z.clone()
    }

    pub fn dot(&self, other: &Self) -> Expr {
        &self.x * &other.x + &self.y * &other.y + &self.z * &other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            &self.y * &other.z - &other.y * &self.z,
            &self.z * &other.x - &other.z * &self.x,
            &self.x * &other.y - &other.x * &self.y,
        )
    }

    pub fn mag2(&self) -> Expr {
        self.dot(self)
    }

    pub fn mag(&self) -> Expr {
        self.mag2().sqrt()
    }

    pub fn costheta(&self) -> Expr {
        &self.z / self.mag()
    }

    pub fn unit(&self) -> Self {
        self / &self.mag()
    }

    pub fn phi(&self) -> Expr {
        atan2(self.py(), self.px())
    }

    pub fn with_mass(&self, mass: impl Into<Expr>) -> Vec4 {
        let mass = mass.into();
        Vec4::new(
            self.px(),
            self.py(),
            self.pz(),
            (mass.powi(2) + self.mag2()).sqrt(),
        )
    }

    pub fn with_energy(&self, energy: impl Into<Expr>) -> Vec4 {
        Vec4::new(self.px(), self.py(), self.pz(), energy)
    }

    pub fn as_expr(&self) -> Expr {
        vector([self.x.clone(), self.y.clone(), self.z.clone()])
    }

    fn scale(&self, scalar: impl Into<Expr>) -> Self {
        let scalar = scalar.into();
        Self::new(&self.x * &scalar, &self.y * &scalar, &self.z * scalar)
    }
}

impl From<RealVec3> for Vec3 {
    fn from(value: RealVec3) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::zero()
    }
}

impl_op_ex!(+ |a: &Vec3, b: &Vec3| -> Vec3 { Vec3::new(&a.x + &b.x, &a.y + &b.y, &a.z + &b.z) });
impl_op_ex!(-|a: &Vec3, b: &Vec3| -> Vec3 { Vec3::new(&a.x - &b.x, &a.y - &b.y, &a.z - &b.z) });
impl_op_ex!(-|a: &Vec3| -> Vec3 { Vec3::new(-&a.x, -&a.y, -&a.z) });
impl_op_ex!(*|a: &Vec3, b: &Expr| -> Vec3 { a.scale(b) });
impl_op_ex!(*|a: &Expr, b: &Vec3| -> Vec3 { b.scale(a) });
impl_op_ex!(*|a: &Vec3, b: &f64| -> Vec3 { a.scale(b) });
impl_op_ex!(*|a: &f64, b: &Vec3| -> Vec3 { b.scale(a) });
impl_op_ex!(/ |a: &Vec3, b: &Expr| -> Vec3 {
    Vec3::new(&a.x / b, &a.y / b, &a.z / b)
});
impl_op_ex!(/ |a: &Vec3, b: &f64| -> Vec3 {
    Vec3::new(&a.x / b, &a.y / b, &a.z / b)
});

/// Expression-valued four-vector with `---+` metric convention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vec4 {
    pub x: Expr,
    pub y: Expr,
    pub z: Expr,
    pub t: Expr,
}

impl Vec4 {
    pub fn new(
        px: impl Into<Expr>,
        py: impl Into<Expr>,
        pz: impl Into<Expr>,
        e: impl Into<Expr>,
    ) -> Self {
        Self {
            x: px.into(),
            y: py.into(),
            z: pz.into(),
            t: e.into(),
        }
    }

    pub fn event(prefix: &str) -> Self {
        Self::new(
            event_p4_component(prefix, P4Component::Px),
            event_p4_component(prefix, P4Component::Py),
            event_p4_component(prefix, P4Component::Pz),
            event_p4_component(prefix, P4Component::E),
        )
    }

    pub fn px(&self) -> Expr {
        self.x.clone()
    }

    pub fn py(&self) -> Expr {
        self.y.clone()
    }

    pub fn pz(&self) -> Expr {
        self.z.clone()
    }

    pub fn e(&self) -> Expr {
        self.t.clone()
    }

    pub fn momentum(&self) -> Vec3 {
        self.vec3()
    }

    pub fn vec3(&self) -> Vec3 {
        Vec3::new(self.px(), self.py(), self.pz())
    }

    pub fn beta(&self) -> Vec3 {
        self.momentum() / self.e()
    }

    pub fn gamma(&self) -> Expr {
        (1.0 - self.beta().mag2()).sqrt()
    }

    pub fn m2(&self) -> Expr {
        self.mag2()
    }

    pub fn m(&self) -> Expr {
        self.mag()
    }

    pub fn mag2(&self) -> Expr {
        self.t.powi(2) - (self.x.powi(2) + self.y.powi(2) + self.z.powi(2))
    }

    pub fn mag(&self) -> Expr {
        self.mag2().sqrt()
    }

    pub fn boost(&self, beta: &Vec3) -> Self {
        let b2 = beta.dot(beta);
        let gamma = (1.0 - &b2).sqrt();
        let gamma = 1.0 / gamma;
        let boost_factor = gamma.powi(2) / (&gamma + 1.0);
        let p3 = self.vec3() + beta * ((boost_factor * self.vec3().dot(beta)) + &gamma * &self.t);
        Self::new(p3.x, p3.y, p3.z, gamma * (&self.t + beta.dot(&self.vec3())))
    }

    pub fn as_expr(&self) -> Expr {
        vector([
            self.x.clone(),
            self.y.clone(),
            self.z.clone(),
            self.t.clone(),
        ])
    }
}

impl From<RealVec4> for Vec4 {
    fn from(value: RealVec4) -> Self {
        Self::new(value.x, value.y, value.z, value.t)
    }
}

impl_op_ex!(+ |a: &Vec4, b: &Vec4| -> Vec4 {
    Vec4::new(&a.x + &b.x, &a.y + &b.y, &a.z + &b.z, &a.t + &b.t)
});
impl_op_ex!(-|a: &Vec4, b: &Vec4| -> Vec4 {
    Vec4::new(&a.x - &b.x, &a.y - &b.y, &a.z - &b.z, &a.t - &b.t)
});
impl_op_ex!(-|a: &Vec4| -> Vec4 { Vec4::new(-&a.x, -&a.y, -&a.z, -&a.t) });

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use laddu_compile::CompiledModel;
    use laddu_runtime::CpuBackend;
    use nalgebra::{Vector3, Vector4};
    use num::complex::Complex64;

    use super::*;

    fn evaluate(expr: laddu_expr::Expr) -> Complex64 {
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = model.params().default_values();
        CpuBackend.prepare(&model).evaluate(&params).unwrap()
    }

    #[test]
    fn test_display() {
        let v3 = RealVec3::new(1.2341, -2.3452, 3.4563);
        assert_eq!(format!("{}", v3), "[ 1.234, -2.345,  3.456]");
        let v4 = RealVec4::new(1.2341, -2.3452, 3.4563, 4.5674);
        assert_eq!(format!("{}", v4), "[ 1.234, -2.345,  3.456;  4.567]");
    }

    #[test]
    fn test_vec_vector_conversion() {
        let v = RealVec3::new(1.0, 2.0, 3.0);
        let vector3: Vec<f64> = v.into();
        assert_eq!(vector3[0], 1.0);
        assert_eq!(vector3[1], 2.0);
        assert_eq!(vector3[2], 3.0);

        let v_from_vec: RealVec3 = vector3.try_into().unwrap();
        assert_eq!(v_from_vec, v);

        let v = RealVec4::new(1.0, 2.0, 3.0, 4.0);
        let vector4: Vec<f64> = v.into();
        assert_eq!(vector4[0], 1.0);
        assert_eq!(vector4[1], 2.0);
        assert_eq!(vector4[2], 3.0);
        assert_eq!(vector4[3], 4.0);

        let v_from_vec: RealVec4 = vector4.try_into().unwrap();
        assert_eq!(v_from_vec, v);
    }

    #[test]
    fn test_vec_array_conversion() {
        let arr = [1.0, 2.0, 3.0];
        let v: RealVec3 = arr.into();
        assert_eq!(v, RealVec3::new(1.0, 2.0, 3.0));

        let back_to_array: [f64; 3] = v.into();
        assert_eq!(back_to_array, arr);

        let arr = [1.0, 2.0, 3.0, 4.0];
        let v: RealVec4 = arr.into();
        assert_eq!(v, RealVec4::new(1.0, 2.0, 3.0, 4.0));

        let back_to_array: [f64; 4] = v.into();
        assert_eq!(back_to_array, arr);
    }

    #[test]
    fn test_vec_nalgebra_conversion() {
        let v = RealVec3::new(1.0, 2.0, 3.0);
        let vector3: Vector3<f64> = v.into();
        assert_eq!(vector3.x, 1.0);
        assert_eq!(vector3.y, 2.0);
        assert_eq!(vector3.z, 3.0);

        let v_from_vec: RealVec3 = vector3.into();
        assert_eq!(v_from_vec, v);

        let v = RealVec4::new(1.0, 2.0, 3.0, 4.0);
        let vector4: Vector4<f64> = v.into();
        assert_eq!(vector4.x, 1.0);
        assert_eq!(vector4.y, 2.0);
        assert_eq!(vector4.z, 3.0);
        assert_eq!(vector4.w, 4.0);

        let v_from_vec: RealVec4 = vector4.into();
        assert_eq!(v_from_vec, v);
    }

    #[test]
    fn test_vec_sums() {
        let vectors = [RealVec3::new(1.0, 2.0, 3.0), RealVec3::new(4.0, 5.0, 6.0)];
        let sum: RealVec3 = vectors.iter().sum();
        assert_eq!(sum, RealVec3::new(5.0, 7.0, 9.0));
        let sum: RealVec3 = vectors.into_iter().sum();
        assert_eq!(sum, RealVec3::new(5.0, 7.0, 9.0));

        let vectors = [
            RealVec4::new(1.0, 2.0, 3.0, 4.0),
            RealVec4::new(4.0, 5.0, 6.0, 7.0),
        ];
        let sum: RealVec4 = vectors.iter().sum();
        assert_eq!(sum, RealVec4::new(5.0, 7.0, 9.0, 11.0));
        let sum: RealVec4 = vectors.into_iter().sum();
        assert_eq!(sum, RealVec4::new(5.0, 7.0, 9.0, 11.0));
    }

    #[test]
    fn test_three_to_four_momentum_conversion() {
        let p3 = RealVec3::new(1.0, 2.0, 3.0);
        let target_p4 = RealVec4::new(1.0, 2.0, 3.0, 10.0);
        let p4_from_mass = p3.with_mass(target_p4.m().unwrap());
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
        let p = RealVec4::new(3.0, 4.0, 5.0, 10.0);
        assert_eq!(p.e(), 10.0);
        assert_eq!(p.px(), 3.0);
        assert_eq!(p.py(), 4.0);
        assert_eq!(p.pz(), 5.0);
        assert_eq!(p.momentum().px(), 3.0);
        assert_eq!(p.momentum().py(), 4.0);
        assert_eq!(p.momentum().pz(), 5.0);
        assert_relative_eq!(p.beta().unwrap().x, 0.3);
        assert_relative_eq!(p.beta().unwrap().y, 0.4);
        assert_relative_eq!(p.beta().unwrap().z, 0.5);
        assert_relative_eq!(p.m2(), 50.0);
        assert_relative_eq!(p.m().unwrap(), f64::sqrt(50.0));
        assert_eq!(
            format!("{}", p.to_p4_string()),
            "[e = 10.00000; p = (3.00000, 4.00000, 5.00000); m = 7.07107]"
        );
        assert_relative_eq!(RealVec3::x().x, 1.0);
        assert_relative_eq!(RealVec3::x().y, 0.0);
        assert_relative_eq!(RealVec3::x().z, 0.0);
        assert_relative_eq!(RealVec3::y().x, 0.0);
        assert_relative_eq!(RealVec3::y().y, 1.0);
        assert_relative_eq!(RealVec3::y().z, 0.0);
        assert_relative_eq!(RealVec3::z().x, 0.0);
        assert_relative_eq!(RealVec3::z().y, 0.0);
        assert_relative_eq!(RealVec3::z().z, 1.0);
        assert_relative_eq!(RealVec3::default().x, 0.0);
        assert_relative_eq!(RealVec3::default().y, 0.0);
        assert_relative_eq!(RealVec3::default().z, 0.0);
    }

    #[test]
    fn test_three_momentum_basics() {
        let p = RealVec4::new(3.0, 4.0, 5.0, 10.0);
        let q = RealVec4::new(1.2, -3.4, 7.6, 0.0);
        let p3_view = p.momentum();
        let q3_view = q.momentum();
        assert_eq!(p3_view.px(), 3.0);
        assert_eq!(p3_view.py(), 4.0);
        assert_eq!(p3_view.pz(), 5.0);
        assert_relative_eq!(p3_view.mag2(), 50.0);
        assert_relative_eq!(p3_view.mag(), f64::sqrt(50.0));
        assert_relative_eq!(p3_view.costheta().unwrap(), 5.0 / f64::sqrt(50.0));
        assert_relative_eq!(p3_view.theta().unwrap(), f64::acos(5.0 / f64::sqrt(50.0)));
        assert_relative_eq!(p3_view.phi(), f64::atan2(4.0, 3.0));
        assert_relative_eq!(
            p3_view.unit().unwrap(),
            RealVec3::new(
                3.0 / f64::sqrt(50.0),
                4.0 / f64::sqrt(50.0),
                5.0 / f64::sqrt(50.0)
            )
        );
        assert_relative_eq!(p3_view.cross(&q3_view), RealVec3::new(47.4, -16.8, -15.0));
    }

    #[test]
    fn test_vec_equality() {
        let p = RealVec3::new(1.1, 2.2, 3.3);
        let p2 = RealVec3::new(1.1 * 2.0, 2.2 * 2.0, 3.3 * 2.0);
        assert_abs_diff_eq!(p * 2.0, p2);
        assert_relative_eq!(p * 2.0, p2);
    }

    #[test]
    fn test_boost_com() {
        let p = RealVec4::new(3.0, 4.0, 5.0, 10.0);
        let zero = p.boost(&-p.beta().unwrap()).momentum();
        assert_relative_eq!(zero, RealVec3::zero());
    }

    #[test]
    fn test_boost() {
        let p0 = RealVec4::new(0.0, 0.0, 0.0, 1.0);
        assert_relative_eq!(p0.gamma().unwrap(), 1.0);
        let p0 = RealVec4::new(f64::sqrt(3.0) / 2.0, 0.0, 0.0, 1.0);
        assert_relative_eq!(p0.gamma().unwrap(), 2.0);
        let p1 = RealVec4::new(3.0, 4.0, 5.0, 10.0);
        let p2 = RealVec4::new(3.4, 2.3, 1.2, 9.0);
        let p1_boosted = p1.boost(&-p2.beta().unwrap());
        assert_relative_eq!(p1_boosted.e(), 8.157632144622882);
        assert_relative_eq!(p1_boosted.px(), -0.6489200627053444);
        assert_relative_eq!(p1_boosted.py(), 1.5316128987581492);
        assert_relative_eq!(p1_boosted.pz(), 3.712145860221643);
    }

    #[test]
    fn expression_vectors_build_and_evaluate_scalar_observables() {
        let p = Vec4::new(3.0, 4.0, 5.0, 10.0);
        assert_eq!(evaluate(p.m2()), Complex64::from(50.0));
        assert_eq!(evaluate(p.momentum().mag2()), Complex64::from(50.0));

        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(evaluate(a.dot(&b)), Complex64::from(32.0));
        assert_eq!(evaluate(a.cross(&b).z), Complex64::from(-3.0));
    }
}
