//! Quantum-number helpers and discrete analysis enums.

use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::LadduError;

mod types;
pub use types::{AngularMomentum, Charge, OrbitalAngularMomentum, Parity, Projection, Statistics};

mod state;
pub use state::{AllowedPartialWave, Isospin, PartialWave, ParticleProperties, SpinState};

mod rules;
pub use rules::{RuleSet, SelectionRules};

/// A two-particle helicity combination.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct HelicityCombination {
    lambda_1: Projection,
    lambda_2: Projection,
    helicity: Projection,
}

impl HelicityCombination {
    /// Construct a helicity combination from two daughter spin projections.
    pub fn new(lambda_1: Projection, lambda_2: Projection) -> Self {
        Self {
            lambda_1,
            lambda_2,
            helicity: Projection::half_integer(lambda_1.value() - lambda_2.value()),
        }
    }

    /// Return the first daughter projection.
    pub const fn lambda_1(self) -> Projection {
        self.lambda_1
    }

    /// Return the second daughter projection.
    pub const fn lambda_2(self) -> Projection {
        self.lambda_2
    }

    /// Return `lambda_1 - lambda_2`.
    pub const fn helicity(self) -> Projection {
        self.helicity
    }
}

/// Enumerate allowed projections for a spin.
pub fn allowed_projections(spin: AngularMomentum) -> Vec<Projection> {
    SpinState::allowed_projections(spin)
        .into_iter()
        .map(SpinState::projection)
        .collect()
}

/// Enumerate all daughter helicity combinations for two spins.
pub fn helicity_combinations(
    spin_1: AngularMomentum,
    spin_2: AngularMomentum,
) -> Vec<HelicityCombination> {
    let projections_1 = allowed_projections(spin_1);
    let projections_2 = allowed_projections(spin_2);
    projections_1
        .into_iter()
        .flat_map(|lambda_1| {
            projections_2
                .iter()
                .copied()
                .map(move |lambda_2| HelicityCombination::new(lambda_1, lambda_2))
        })
        .collect()
}

/// Standard reference frames for angular analyses.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frame {
    /// The helicity frame, obtained by setting the $`z`$-axis equal to the boost direction from
    /// the center-of-momentum to the rest frame of the resonance in question and the $`y`$-axis
    /// perpendicular to the production plane.
    Helicity,
    /// The Gottfried-Jackson frame, obtained by setting the $`z`$-axis proportional to the beam's
    /// direction in the rest frame of the resonance in question and the $`y`$-axis perpendicular
    /// to the production plane.
    GottfriedJackson,
    /// The Adair frame.
    Adair,
}
impl Display for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Frame::Helicity => write!(f, "Helicity"),
            Frame::GottfriedJackson => write!(f, "Gottfried-Jackson"),
            Frame::Adair => write!(f, "Adair"),
        }
    }
}
impl FromStr for Frame {
    type Err = LadduError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "helicity" | "hx" | "hel" => Ok(Self::Helicity),
            "gottfriedjackson" | "gottfried jackson" | "gj" | "gottfried-jackson" => {
                Ok(Self::GottfriedJackson)
            }
            "adair" => Ok(Self::Adair),
            _ => Err(LadduError::ParseError {
                name: s.to_string(),
                object: "Frame".to_string(),
            }),
        }
    }
}

/// A simple enum describing a binary sign.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reflectivity {
    /// A positive indicator.
    Positive,
    /// A negative indicator.
    Negative,
}
impl Display for Reflectivity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reflectivity::Positive => write!(f, "+"),
            Reflectivity::Negative => write!(f, "-"),
        }
    }
}

impl FromStr for Reflectivity {
    type Err = LadduError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "+" | "plus" | "pos" | "positive" => Ok(Self::Positive),
            "-" | "minus" | "neg" | "negative" => Ok(Self::Negative),
            _ => Err(LadduError::ParseError {
                name: s.to_string(),
                object: "Reflectivity".to_string(),
            }),
        }
    }
}

/// An enum for Mandelstam variables
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    /// s-channel
    S,
    /// t-channel
    T,
    /// u-channel
    U,
}

impl Display for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Channel::S => write!(f, "s"),
            Channel::T => write!(f, "t"),
            Channel::U => write!(f, "u"),
        }
    }
}

impl FromStr for Channel {
    type Err = LadduError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "s" => Ok(Self::S),
            "t" => Ok(Self::T),
            "u" => Ok(Self::U),
            _ => Err(LadduError::ParseError {
                name: s.to_string(),
                object: "Channel".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_state_accepts_integer_and_half_integer_values() {
        let spin_one = AngularMomentum::integer(1);
        let spin_half = AngularMomentum::half_integer(1);
        assert_eq!(
            SpinState::new(spin_one, Projection::integer(0))
                .unwrap()
                .projection()
                .value(),
            0
        );
        assert_eq!(
            SpinState::new(spin_half, Projection::half_integer(-1))
                .unwrap()
                .projection()
                .value(),
            -1
        );
    }

    #[test]
    fn spin_state_rejects_invalid_projection() {
        let spin_one = AngularMomentum::integer(1);
        assert!(SpinState::new(spin_one, Projection::half_integer(4)).is_err());
        assert!(SpinState::new(spin_one, Projection::half_integer(1)).is_err());
    }

    #[test]
    fn spin_state_enumerates_allowed_projections() {
        let spin_zero = SpinState::allowed_projections(AngularMomentum::integer(0));
        assert_eq!(
            spin_zero
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![0]
        );

        let spin_half = SpinState::allowed_projections(AngularMomentum::half_integer(1));
        assert_eq!(
            spin_half
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-1, 1]
        );

        let spin_one = SpinState::allowed_projections(AngularMomentum::integer(1));
        assert_eq!(
            spin_one
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-2, 0, 2]
        );

        let spin_three_halves = SpinState::allowed_projections(AngularMomentum::half_integer(3));
        assert_eq!(
            spin_three_halves
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-3, -1, 1, 3]
        );
    }

    #[test]
    fn allowed_projection_helper_returns_projection_values() {
        assert_eq!(
            allowed_projections(AngularMomentum::integer(1)),
            vec![
                Projection::integer(-1),
                Projection::integer(0),
                Projection::integer(1),
            ]
        );
    }

    #[test]
    fn helicity_combinations_enumerate_daughter_projection_products() {
        let spin_half = AngularMomentum::half_integer(1);
        let combinations = helicity_combinations(spin_half, spin_half);

        assert_eq!(
            combinations,
            vec![
                HelicityCombination::new(
                    Projection::half_integer(-1),
                    Projection::half_integer(-1),
                ),
                HelicityCombination::new(Projection::half_integer(-1), Projection::half_integer(1),),
                HelicityCombination::new(Projection::half_integer(1), Projection::half_integer(-1),),
                HelicityCombination::new(Projection::half_integer(1), Projection::half_integer(1),),
            ]
        );
        assert_eq!(combinations[1].helicity(), Projection::integer(-1));
        assert_eq!(combinations[2].helicity(), Projection::integer(1));
    }

    #[test]
    fn enum_displays() {
        assert_eq!(format!("{}", Frame::Helicity), "Helicity");
        assert_eq!(format!("{}", Frame::GottfriedJackson), "Gottfried-Jackson");
        assert_eq!(format!("{}", Frame::Adair), "Adair");
        assert_eq!(format!("{}", Reflectivity::Positive), "+");
        assert_eq!(format!("{}", Reflectivity::Negative), "-");
        assert_eq!(format!("{}", Channel::S), "s");
        assert_eq!(format!("{}", Channel::T), "t");
        assert_eq!(format!("{}", Channel::U), "u");
    }

    #[test]
    fn enum_from_str() {
        assert_eq!(Frame::from_str("Helicity").unwrap(), Frame::Helicity);
        assert_eq!(Frame::from_str("HX").unwrap(), Frame::Helicity);
        assert_eq!(Frame::from_str("HEL").unwrap(), Frame::Helicity);
        assert_eq!(
            Frame::from_str("GottfriedJackson").unwrap(),
            Frame::GottfriedJackson
        );
        assert_eq!(Frame::from_str("GJ").unwrap(), Frame::GottfriedJackson);
        assert_eq!(
            Frame::from_str("Gottfried-Jackson").unwrap(),
            Frame::GottfriedJackson
        );
        assert_eq!(
            Frame::from_str("Gottfried Jackson").unwrap(),
            Frame::GottfriedJackson
        );
        assert_eq!(Frame::from_str("Adair").unwrap(), Frame::Adair);
        assert_eq!(Reflectivity::from_str("+").unwrap(), Reflectivity::Positive);
        assert_eq!(
            Reflectivity::from_str("pos").unwrap(),
            Reflectivity::Positive
        );
        assert_eq!(
            Reflectivity::from_str("plus").unwrap(),
            Reflectivity::Positive
        );
        assert_eq!(
            Reflectivity::from_str("Positive").unwrap(),
            Reflectivity::Positive
        );
        assert_eq!(Reflectivity::from_str("-").unwrap(), Reflectivity::Negative);
        assert_eq!(
            Reflectivity::from_str("minus").unwrap(),
            Reflectivity::Negative
        );
        assert_eq!(
            Reflectivity::from_str("neg").unwrap(),
            Reflectivity::Negative
        );
        assert_eq!(
            Reflectivity::from_str("Negative").unwrap(),
            Reflectivity::Negative
        );
        assert_eq!(Channel::from_str("S").unwrap(), Channel::S);
        assert_eq!(Channel::from_str("s").unwrap(), Channel::S);
        assert_eq!(Channel::from_str("T").unwrap(), Channel::T);
        assert_eq!(Channel::from_str("t").unwrap(), Channel::T);
        assert_eq!(Channel::from_str("U").unwrap(), Channel::U);
        assert_eq!(Channel::from_str("u").unwrap(), Channel::U);
    }
}
