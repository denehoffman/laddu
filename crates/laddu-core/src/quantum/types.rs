use std::{fmt::Display, str::FromStr};

use auto_ops::impl_op_ex;
use num::rational::Ratio;
use serde::{Deserialize, Serialize};

use crate::{quantum::parse_sign_value, LadduError};

const QUANTUM_NUMBER_FLOAT_TOLERANCE: f64 = 1.0e-12;

/// A helper enum denoting the sign of a state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub(crate) enum Sign {
    /// A positive sign.
    Positive,
    /// A negative sign.
    Negative,
}

/// A non-negative total angular momentum stored as twice its physical value.
///
/// This representation keeps integer and half-integer quantum numbers exact. For example,
/// `J::half(1)` represents `$1/2$`, and `J::int(1)` represents `$1$`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct J(u32);

impl J {
    /// Construct a non-negative integer angular momentum.
    pub const fn int(value: u32) -> Self {
        Self(2 * value)
    }

    /// Construct a non-negative angular momentum from the given numerator over two.
    pub const fn half(value: u32) -> Self {
        Self(value)
    }

    /// Enumerate the valid signed projections for this angular momentum.
    pub fn projections(self) -> Vec<M> {
        let twice = self.0 as i32;
        (-twice..=twice).step_by(2).map(M::half).collect()
    }

    /// Return the doubled integer value.
    pub const fn doubled(self) -> u32 {
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
    pub(crate) const fn has_same_parity_as(self, projection: M) -> bool {
        (self.0 & 1) as i32 == projection.doubled() & 1
    }

    /// Returns true if the given angular momenta can couple to produce this one.
    pub(crate) fn can_couple_to(&self, j1: Self, j2: Self) -> bool {
        let min = j1.doubled().abs_diff(j2.doubled());
        let max = j1.doubled() + j2.doubled();
        self.doubled() >= min && self.doubled() <= max && (self.doubled() - min).is_multiple_of(2)
    }
}

impl TryFrom<Ratio<i32>> for J {
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

impl TryFrom<f64> for J {
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
impl Display for J {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.doubled() / 2)
        } else {
            write!(f, "{}/2", self.doubled())
        }
    }
}

/// A non-negative integer orbital angular momentum.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct L(u32);

impl L {
    /// Construct an orbital angular momentum.
    pub const fn int(value: u32) -> Self {
        Self(value)
    }

    /// Return the integer orbital angular momentum.
    pub const fn value(self) -> u32 {
        self.0
    }

    /// Enumerate the valid signed projections for this orbital angular momentum.
    pub fn projections(self) -> Vec<M> {
        J::int(self.0).projections()
    }

    /// Return the orbital angular momentum as a total angular momentum.
    pub const fn as_j(self) -> J {
        J::int(self.0)
    }

    /// Returns the parity derived from the orbital angular momentum, $`(-1)^{\ell}`$
    pub const fn orbital_parity(self) -> Parity {
        if self.value().is_multiple_of(2) {
            Parity::Positive
        } else {
            Parity::Negative
        }
    }
}

impl TryFrom<Ratio<i32>> for L {
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

impl TryFrom<f64> for L {
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
                "orbital angular momentum is too large".to_string(),
            ));
        }
        Ok(Self(u32::try_from(rounded as i32).map_err(|_| {
            LadduError::Custom("orbital angular momentum cannot be negative".to_string())
        })?))
    }
}

impl Display for L {
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
pub struct M(i32);

impl M {
    /// Construct a signed integer projection.
    pub const fn int(value: i32) -> Self {
        Self(2 * value)
    }

    /// Construct a signed projection from the given numerator over two.
    pub const fn half(value: i32) -> Self {
        Self(value)
    }

    /// Return the doubled integer value.
    pub const fn doubled(self) -> i32 {
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

impl TryFrom<Ratio<i32>> for M {
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
impl TryFrom<f64> for M {
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
impl Display for M {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.doubled() / 2)
        } else {
            write!(f, "{}/2", self.doubled())
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

impl FromStr for Parity {
    type Err = LadduError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match parse_sign_value(s, "Parity")? {
            Sign::Positive => Ok(Self::Positive),
            Sign::Negative => Ok(Self::Negative),
        }
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
    pub const fn int(value: i32) -> Self {
        Self(3 * value)
    }

    /// Construct a signed charge from a numerator over three.
    pub const fn third(value: i32) -> Self {
        Self(value)
    }

    /// Return the tripled integer value.
    pub const fn tripled(self) -> i32 {
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
            write!(f, "{}", self.tripled() / 3)
        } else {
            write!(f, "{}/3", self.tripled())
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
    pub fn from_spin(spin: J) -> Self {
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

// Operations
#[rustfmt::skip]
impl_op_ex!(+ |m1: &M, m2: &M| -> M { M::half(m1.doubled() + m2.doubled()) });
#[rustfmt::skip]
impl_op_ex!(- |m1: &M, m2: &M| -> M { M::half(m1.doubled() - m2.doubled()) });
#[rustfmt::skip]
impl_op_ex!(- |m: &M| -> M { M::half(-m.doubled()) });
#[rustfmt::skip]
impl_op_ex!(+= |m1: &mut M, m2: &M| { *m1 = *m1 + m2 });
#[rustfmt::skip]
impl_op_ex!(-= |m1: &mut M, m2: &M| { *m1 = *m1 - m2 });

#[rustfmt::skip]
impl_op_ex!(+ |c1: &Charge, c2: &Charge| -> Charge { Charge::third(c1.tripled() + c2.tripled()) });
#[rustfmt::skip]
impl_op_ex!(- |c1: &Charge, c2: &Charge| -> Charge { Charge::third(c1.tripled() - c2.tripled()) });
#[rustfmt::skip]
impl_op_ex!(- |c: &Charge| -> Charge { Charge::third(-c.tripled()) });
#[rustfmt::skip]
impl_op_ex!(+= |c1: &mut Charge, c2: &Charge| { *c1 = *c1 + c2 });
#[rustfmt::skip]
impl_op_ex!(-= |c1: &mut Charge, c2: &Charge| { *c1 = *c1 - c2 });

#[rustfmt::skip]
impl_op_ex!(* |p1: &Parity, p2: &Parity| -> Parity {
    match (p1, p2) {
        (Parity::Positive, Parity::Positive) | (Parity::Negative, Parity::Negative) => {
            Parity::Positive
        }
        (Parity::Positive, Parity::Negative) | (Parity::Negative, Parity::Positive) => {
            Parity::Negative
        }
    }
});
#[rustfmt::skip]
impl_op_ex!(*= |p1: &mut Parity, p2: &Parity| { *p1 = *p1 * p2});
#[rustfmt::skip]
impl_op_ex!(- |p: &Parity| -> Parity {
    if matches!(p, Parity::Positive) {
        Parity::Negative
    } else {
        Parity::Positive
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbital_angular_momentum_rejects_half_integer_values() {
        assert_eq!(L::try_from(Ratio::new(2, 1)).unwrap().value(), 2);
        assert!(L::try_from(Ratio::new(3, 2)).is_err());
    }

    #[test]
    fn angular_momentum_accepts_ratio_and_float_physical_values() {
        assert_eq!(J::try_from(Ratio::new(3, 2)).unwrap().doubled(), 3);
        assert_eq!(J::try_from(1.5).unwrap().doubled(), 3);
        assert_eq!(M::try_from(Ratio::new(-1, 2)).unwrap().doubled(), -1);
        assert_eq!(M::try_from(-0.5).unwrap().doubled(), -1);
        assert!(J::try_from(Ratio::new(1, 3)).is_err());
        assert!(M::try_from(0.25).is_err());
    }

    #[test]
    fn orbital_angular_momentum_accepts_integer_ratio_and_float_values() {
        assert_eq!(L::try_from(Ratio::new(2, 1)).unwrap().value(), 2);
        assert_eq!(L::try_from(2.0).unwrap().value(), 2);
        assert!(L::try_from(Ratio::new(3, 2)).is_err());
        assert!(L::try_from(1.5).is_err());
    }

    #[test]
    fn parity_returns_signed_value() {
        assert_eq!(Parity::Positive.value(), 1);
        assert_eq!(Parity::Negative.value(), -1);
    }

    #[test]
    fn charge_accepts_integer_ratio_and_float_values() {
        assert_eq!(Charge::try_from(Ratio::new(3, 1)).unwrap().tripled(), 9);
        assert_eq!(Charge::try_from(2.0).unwrap().tripled(), 6);
        assert_eq!(Charge::try_from(Ratio::new(2, 3)).unwrap().tripled(), 2);
        assert!(Charge::try_from(Ratio::new(3, 2)).is_err());
        assert!(Charge::try_from(1.5).is_err());
    }
}
