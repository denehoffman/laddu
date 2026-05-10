use std::fmt::Display;

use num::rational::Ratio;
use serde::{Deserialize, Serialize};

use crate::{LadduError, LadduResult};

const QUANTUM_NUMBER_FLOAT_TOLERANCE: f64 = 1.0e-12;

/// A non-negative angular momentum stored as twice its physical value.
///
/// This representation keeps integer and half-integer quantum numbers exact. For example,
/// `AngularMomentum::from_twice(1)` represents `$1/2$`, and
/// `AngularMomentum::from_twice(2)` represents `$1$`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AngularMomentum(u32);

impl AngularMomentum {
    /// Construct a non-negative integer angular momentum.
    pub const fn integer(value: u32) -> Self {
        Self(2 * value)
    }

    /// Construct a non-negative angular momentum from the given numerator over two.
    pub const fn half_integer(value: u32) -> Self {
        Self(value)
    }

    /// Return the doubled integer value.
    pub const fn value(self) -> u32 {
        self.0
    }

    /// Return the physical value as `f64`.
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 2.0
    }

    /// Return whether this quantum number represents an integer value.
    pub const fn is_integer(self) -> bool {
        self.0 & 1 == 0
    }

    /// Return whether this angular momentum has the same integer/half-integer parity as
    /// `projection`.
    pub const fn has_same_parity_as(self, projection: Projection) -> bool {
        (self.0 & 1) as i32 == projection.value() & 1
    }

    /// Returns true if the given angular momenta can couple to produce this one.
    pub fn can_couple_to(&self, j1: Self, j2: Self) -> bool {
        let min = j1.value().abs_diff(j2.value());
        let max = j1.value() + j2.value();
        self.value() >= min && self.value() <= max && (self.value() - min).is_multiple_of(2)
    }
}

impl TryFrom<Ratio<i32>> for AngularMomentum {
    type Error = LadduError;

    fn try_from(value: Ratio<i32>) -> Result<Self, Self::Error> {
        let twice = value * Ratio::from_integer(2);
        if !twice.is_integer() {
            return Err(LadduError::Custom(format!(
                "angular momentum must be integer or half-integer, got {value}"
            )));
        }
        Ok(Self(u32::try_from(*twice.numer()).map_err(|_| {
            LadduError::Custom("angular momentum cannot be negative".to_string())
        })?))
    }
}

impl TryFrom<f64> for AngularMomentum {
    type Error = LadduError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err(LadduError::Custom(
                "angular momentum must be finite".to_string(),
            ));
        }
        let twice = 2.0 * value;
        let rounded = twice.round();
        if (twice - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduError::Custom(format!(
                "angular momentum must be integer or half-integer, got {value}"
            )));
        }
        if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
            return Err(LadduError::Custom(
                "angular momentum is too large".to_string(),
            ));
        }
        Ok(Self(u32::try_from(rounded as i32).map_err(|_| {
            LadduError::Custom("angular momentum cannot be negative".to_string())
        })?))
    }
}
impl Display for AngularMomentum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.value() / 2)
        } else {
            write!(f, "{}/2", self.value())
        }
    }
}

/// A non-negative integer orbital angular momentum.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct OrbitalAngularMomentum(u32);

impl OrbitalAngularMomentum {
    /// Construct an orbital angular momentum.
    pub const fn integer(value: u32) -> Self {
        Self(value)
    }

    /// Construct an orbital angular momentum from a non-negative angular momentum.
    pub fn from_angular_momentum(value: AngularMomentum) -> LadduResult<Self> {
        if !value.is_integer() {
            return Err(LadduError::Custom(
                "orbital angular momentum must be an integer".to_string(),
            ));
        }
        Ok(Self(value.value() / 2))
    }

    /// Return the integer orbital angular momentum.
    pub const fn value(self) -> u32 {
        self.0
    }

    /// Return the orbital angular momentum as a doubled non-negative angular momentum.
    pub fn angular_momentum(self) -> AngularMomentum {
        AngularMomentum::integer(self.0)
    }
}

impl TryFrom<Ratio<i32>> for OrbitalAngularMomentum {
    type Error = LadduError;

    fn try_from(value: Ratio<i32>) -> Result<Self, Self::Error> {
        if !value.is_integer() {
            return Err(LadduError::Custom(format!(
                "orbital angular momentum must be integer, got {value}"
            )));
        }
        Ok(Self(u32::try_from(*value.numer()).map_err(|_| {
            LadduError::Custom("orbital angular momentum cannot be negative".to_string())
        })?))
    }
}

impl TryFrom<f64> for OrbitalAngularMomentum {
    type Error = LadduError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err(LadduError::Custom(
                "orbital angular momentum must be finite".to_string(),
            ));
        }
        let rounded = value.round();
        if (value - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduError::Custom(format!(
                "orbital angular momentum must be integer, got {value}"
            )));
        }
        if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
            return Err(LadduError::Custom(
                "angular momentum is too large".to_string(),
            ));
        }
        Ok(Self(u32::try_from(rounded as i32).map_err(|_| {
            LadduError::Custom("angular momentum cannot be negative".to_string())
        })?))
    }
}

impl Display for OrbitalAngularMomentum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self.value() {
                0 => "S".to_string(),
                1 => "P".to_string(),
                2 => "D".to_string(),
                3 => "F".to_string(),
                4 => "G".to_string(),
                5 => "H".to_string(),
                6 => "I".to_string(),
                n => format!("L{n}"),
            }
        )
    }
}

/// A signed integer or half-integer projection stored as twice its physical value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Projection(i32);

impl Projection {
    /// Construct a signed integer projection.
    pub const fn integer(value: i32) -> Self {
        Self(2 * value)
    }

    /// Construct a signed projection from the given numerator over two.
    pub const fn half_integer(value: i32) -> Self {
        Self(value)
    }

    /// Return the doubled integer value.
    pub const fn value(self) -> i32 {
        self.0
    }

    /// Return the physical value as `f64`.
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 2.0
    }

    /// Return whether this projection represents an integer value.
    pub const fn is_integer(self) -> bool {
        self.0 & 1 == 0
    }
}

impl TryFrom<Ratio<i32>> for Projection {
    type Error = LadduError;

    fn try_from(value: Ratio<i32>) -> Result<Self, Self::Error> {
        let twice = value * Ratio::from_integer(2);
        if !twice.is_integer() {
            return Err(LadduError::Custom(format!(
                "projection must be integer or half-integer, got {value}"
            )));
        }
        Ok(Self(*twice.numer()))
    }
}
impl TryFrom<f64> for Projection {
    type Error = LadduError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err(LadduError::Custom("projection must be finite".to_string()));
        }
        let twice = 2.0 * value;
        let rounded = twice.round();
        if (twice - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduError::Custom(format!(
                "projection must be integer or half-integer, got {value}"
            )));
        }
        if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
            return Err(LadduError::Custom("projection is too large".to_string()));
        }
        Ok(Self(rounded as i32))
    }
}
impl Display for Projection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.value() / 2)
        } else {
            write!(f, "{}/2", self.value())
        }
    }
}

/// Intrinsic parity assignment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum Parity {
    /// Positive intrinsic parity.
    Positive,
    /// Negative intrinsic parity.
    Negative,
}

impl Parity {
    /// The signed value of parity.
    pub fn value(self) -> i32 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
}
impl Display for Parity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Positive => "+",
                Self::Negative => "-",
            }
        )
    }
}

/// Electric charge store as three times its physical value.
///
/// Because quarks may have fractional charge, we represent a charge of `+1` using the value `3`, a
/// charge of `+2/3` by `2`, and so on.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Charge(i32);

impl Charge {
    /// Construct a signed charge.
    pub const fn integer(value: i32) -> Self {
        Self(3 * value)
    }

    /// Construct a signed charge from a numerator over three.
    pub const fn third_integer(value: i32) -> Self {
        Self(value)
    }

    /// Return the tripled integer value.
    pub const fn value(self) -> i32 {
        self.0
    }

    /// Return whether this quantum number represents an integer value.
    pub const fn is_integer(self) -> bool {
        self.0 % 3 == 0
    }
}
impl Display for Charge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.value() / 3)
        } else {
            write!(f, "{}/3", self.value())
        }
    }
}

/// An enum representing the statistics of a particle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Statistics {
    /// Variant for bosonic statistics
    Boson,
    /// Variant for fermionic statistics
    Fermion,
}

impl Statistics {
    /// Construct quantum statistics from a spin value.
    pub fn from_spin(spin: AngularMomentum) -> Self {
        if spin.is_integer() {
            Self::Boson
        } else {
            Self::Fermion
        }
    }
}

impl TryFrom<Ratio<i32>> for Charge {
    type Error = LadduError;

    fn try_from(value: Ratio<i32>) -> Result<Self, Self::Error> {
        let thrice = value * Ratio::from_integer(3);
        if !thrice.is_integer() {
            return Err(LadduError::Custom(format!(
                "electric charge must a multiple of 1/3, got {value}"
            )));
        }
        Ok(Self(*thrice.numer()))
    }
}

impl TryFrom<f64> for Charge {
    type Error = LadduError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err(LadduError::Custom(
                "electric charge must be finite".to_string(),
            ));
        }
        let thrice = 3.0 * value;
        let rounded = thrice.round();
        if (thrice - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduError::Custom(format!(
                "electric charge must a multiple of 1/3, got {value}"
            )));
        }
        if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
            return Err(LadduError::Custom(
                "electric charge is too large".to_string(),
            ));
        }
        Ok(Self(rounded as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbital_angular_momentum_rejects_half_integer_values() {
        assert_eq!(
            OrbitalAngularMomentum::from_angular_momentum(AngularMomentum::integer(2))
                .unwrap()
                .value(),
            2
        );
        assert!(
            OrbitalAngularMomentum::from_angular_momentum(AngularMomentum::half_integer(3))
                .is_err()
        );
    }

    #[test]
    fn angular_momentum_accepts_ratio_and_float_physical_values() {
        assert_eq!(
            AngularMomentum::try_from(Ratio::new(3, 2)).unwrap().value(),
            3
        );
        assert_eq!(AngularMomentum::try_from(1.5).unwrap().value(), 3);
        assert_eq!(Projection::try_from(Ratio::new(-1, 2)).unwrap().value(), -1);
        assert_eq!(Projection::try_from(-0.5).unwrap().value(), -1);
        assert!(AngularMomentum::try_from(Ratio::new(1, 3)).is_err());
        assert!(Projection::try_from(0.25).is_err());
    }

    #[test]
    fn orbital_angular_momentum_accepts_integer_ratio_and_float_values() {
        assert_eq!(
            OrbitalAngularMomentum::try_from(Ratio::new(2, 1))
                .unwrap()
                .value(),
            2
        );
        assert_eq!(OrbitalAngularMomentum::try_from(2.0).unwrap().value(), 2);
        assert!(OrbitalAngularMomentum::try_from(Ratio::new(3, 2)).is_err());
        assert!(OrbitalAngularMomentum::try_from(1.5).is_err());
    }

    #[test]
    fn parity_returns_signed_value() {
        assert_eq!(Parity::Positive.value(), 1);
        assert_eq!(Parity::Negative.value(), -1);
    }

    #[test]
    fn charge_accepts_integer_ratio_and_float_values() {
        assert_eq!(Charge::try_from(Ratio::new(3, 1)).unwrap().value(), 9);
        assert_eq!(Charge::try_from(2.0).unwrap().value(), 6);
        assert_eq!(Charge::try_from(Ratio::new(2, 3)).unwrap().value(), 2);
        assert!(Charge::try_from(Ratio::new(3, 2)).is_err());
        assert!(Charge::try_from(1.5).is_err());
    }
}
