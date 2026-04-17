use serde::{Deserialize, Serialize};

use crate::{LadduError, LadduResult};

/// A non-negative angular momentum stored as twice its physical value.
///
/// This representation keeps integer and half-integer quantum numbers exact. For example,
/// `AngularMomentum::new(1)` represents `$1/2$`, and `AngularMomentum::new(2)` represents `$1$`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AngularMomentum(u32);

impl AngularMomentum {
    /// Construct a doubled non-negative angular momentum.
    pub const fn new(value: u32) -> Self {
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
    pub const fn has_same_parity_as(self, projection: AngularMomentumProjection) -> bool {
        (self.0 & 1) as i32 == projection.value() & 1
    }
}

/// A signed angular-momentum projection stored as twice its physical value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AngularMomentumProjection(i32);

impl AngularMomentumProjection {
    /// Construct a doubled signed angular-momentum projection.
    pub const fn new(value: i32) -> Self {
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

/// A validated spin state with spin and projection stored as doubled quantum numbers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SpinState {
    spin: AngularMomentum,
    projection: AngularMomentumProjection,
}

impl SpinState {
    /// Construct a spin state after validating projection bounds and parity.
    pub fn new(spin: AngularMomentum, projection: AngularMomentumProjection) -> LadduResult<Self> {
        validate_projection(spin, projection)?;
        Ok(Self { spin, projection })
    }

    /// Return the spin quantum number.
    pub const fn spin(self) -> AngularMomentum {
        self.spin
    }

    /// Return the spin projection quantum number.
    pub const fn projection(self) -> AngularMomentumProjection {
        self.projection
    }

    /// Enumerate all allowed projections for `spin`.
    pub fn allowed_projections(spin: AngularMomentum) -> Vec<Self> {
        let spin_value = spin.value() as i32;
        (-spin_value..=spin_value)
            .step_by(2)
            .map(|projection| Self {
                spin,
                projection: AngularMomentumProjection::new(projection),
            })
            .collect()
    }
}

/// A non-negative integer orbital angular momentum.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct OrbitalAngularMomentum(u32);

impl OrbitalAngularMomentum {
    /// Construct an orbital angular momentum.
    pub const fn new(value: u32) -> Self {
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
        AngularMomentum::new(2 * self.0)
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
    /// Return the parity as `+1` or `-1`.
    pub const fn value(self) -> i8 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
}

fn validate_projection(
    spin: AngularMomentum,
    projection: AngularMomentumProjection,
) -> LadduResult<()> {
    if projection.value().unsigned_abs() > spin.value() {
        return Err(LadduError::Custom(
            "spin projection must satisfy -J <= m <= J".to_string(),
        ));
    }
    if !spin.has_same_parity_as(projection) {
        return Err(LadduError::Custom(
            "spin projection must have the same integer or half-integer parity as spin".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_state_accepts_integer_and_half_integer_values() {
        let spin_one = AngularMomentum::new(2);
        let spin_half = AngularMomentum::new(1);
        assert_eq!(
            SpinState::new(spin_one, AngularMomentumProjection::new(0))
                .unwrap()
                .projection()
                .value(),
            0
        );
        assert_eq!(
            SpinState::new(spin_half, AngularMomentumProjection::new(-1))
                .unwrap()
                .projection()
                .value(),
            -1
        );
    }

    #[test]
    fn spin_state_rejects_invalid_projection() {
        let spin_one = AngularMomentum::new(2);
        assert!(SpinState::new(spin_one, AngularMomentumProjection::new(4)).is_err());
        assert!(SpinState::new(spin_one, AngularMomentumProjection::new(1)).is_err());
    }

    #[test]
    fn spin_state_enumerates_allowed_projections() {
        let spin_zero = SpinState::allowed_projections(AngularMomentum::new(0));
        assert_eq!(
            spin_zero
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![0]
        );

        let spin_half = SpinState::allowed_projections(AngularMomentum::new(1));
        assert_eq!(
            spin_half
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-1, 1]
        );

        let spin_one = SpinState::allowed_projections(AngularMomentum::new(2));
        assert_eq!(
            spin_one
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-2, 0, 2]
        );

        let spin_three_halves = SpinState::allowed_projections(AngularMomentum::new(3));
        assert_eq!(
            spin_three_halves
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-3, -1, 1, 3]
        );
    }

    #[test]
    fn orbital_angular_momentum_rejects_half_integer_values() {
        assert_eq!(
            OrbitalAngularMomentum::from_angular_momentum(AngularMomentum::new(4))
                .unwrap()
                .value(),
            2
        );
        assert!(OrbitalAngularMomentum::from_angular_momentum(AngularMomentum::new(3)).is_err());
    }

    #[test]
    fn parity_returns_signed_value() {
        assert_eq!(Parity::Positive.value(), 1);
        assert_eq!(Parity::Negative.value(), -1);
    }
}
