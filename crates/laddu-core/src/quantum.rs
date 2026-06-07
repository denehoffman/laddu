//! Quantum-number helpers and discrete analysis enums.

use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{quantum::types::Sign, LadduError};

mod types;
pub use types::{Charge, Parity, Statistics, J, L, M};

/// Coupled intrinsic spin, represented by the same quantum-number type as total angular momentum.
pub type S = J;

mod state;
pub use state::{
    AllowedPartialWave, ExternalId, Isospin, PartialWave, ParticleProperties, SpinState,
};

mod rules;
pub use rules::{RuleSet, SelectionRules};

/// Shorthand macro for constructing a total angular momentum [`J`].
#[allow(unused_macros)]
#[macro_export]
macro_rules! j {
    ($n:literal / 2) => {
        $crate::quantum::J::half($n)
    };
    ($n:literal) => {
        $crate::quantum::J::int($n)
    };
    ($($bad:tt)*) => {
        compile_error!("expected j!(N) or j!(N/2) where N is an integer literal");
    };
}
/// Shorthand macro for constructing a spin [`S`](`crate::quantum::J`).
#[allow(unused_macros)]
#[macro_export]
macro_rules! s {
    ($n:literal / 2) => {
        $crate::quantum::S::half($n)
    };
    ($n:literal) => {
        $crate::quantum::S::int($n)
    };
    ($($bad:tt)*) => {
        compile_error!("expected s!(N) or s!(N/2) where N is an integer literal");
    };
}
/// Shorthand macro for constructing an angular momentum projection [`M`].
#[allow(unused_macros)]
#[macro_export]
macro_rules! m {
    ($n:literal / 2) => {
        $crate::quantum::M::half($n)
    };
    (- $n:literal / 2) => {
        $crate::quantum::M::half(-$n)
    };
    ($n:literal) => {
        $crate::quantum::M::int($n)
    };
    (- $n:literal) => {
        $crate::quantum::M::int(-$n)
    };
    ($($bad:tt)*) => {
        compile_error!(
            "expected m!(N), m!(-N), m!(N/2), or m!(-N/2) where N is an integer literal"
        );
    };
}
/// Shorthand macro for constructing an orbital angular momentum [`L`].
#[allow(unused_macros)]
#[macro_export]
macro_rules! l {
    ($n:literal) => {
        $crate::quantum::L::int($n)
    };
    ($($bad:tt)*) => {
        compile_error!("expected l!(N) where N is an integer literal");
    };
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
        match parse_sign_value(s, "Reflectivity")? {
            Sign::Positive => Ok(Self::Positive),
            Sign::Negative => Ok(Self::Negative),
        }
    }
}

pub(crate) fn parse_sign_value(s: &str, object: &str) -> Result<Sign, LadduError> {
    match s.to_lowercase().as_ref() {
        "+" | "plus" | "pos" | "positive" => Ok(Sign::Positive),
        "-" | "minus" | "neg" | "negative" => Ok(Sign::Negative),
        _ => Err(LadduError::ParseError {
            name: s.to_string(),
            object: object.to_string(),
        }),
    }
}

/// An enum for Mandelstam variables.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MandelstamChannel {
    /// s-channel
    S,
    /// t-channel
    T,
    /// u-channel
    U,
}

impl Display for MandelstamChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MandelstamChannel::S => write!(f, "s"),
            MandelstamChannel::T => write!(f, "t"),
            MandelstamChannel::U => write!(f, "u"),
        }
    }
}

impl FromStr for MandelstamChannel {
    type Err = LadduError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "s" => Ok(Self::S),
            "t" => Ok(Self::T),
            "u" => Ok(Self::U),
            _ => Err(LadduError::ParseError {
                name: s.to_string(),
                object: "MandelstamChannel".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_angular_momentum_macros() {
        assert_eq!(J::int(1), j!(1));
        assert_eq!(J::half(1), j!(1 / 2));
        assert_eq!(S::int(1), s!(1));
        assert_eq!(S::half(1), s!(1 / 2));
        assert_eq!(M::int(1), m!(1));
        assert_eq!(M::half(1), m!(1 / 2));
        assert_eq!(M::int(-1), m!(-1));
        assert_eq!(M::half(-1), m!(-1 / 2));
        assert_eq!(L::int(1), l!(1));
    }

    #[test]
    fn spin_state_accepts_integer_and_half_integer_values() {
        let spin_one = j!(1);
        let spin_half = j!(1 / 2);
        assert_eq!(
            SpinState::new(spin_one, m!(0))
                .unwrap()
                .projection()
                .doubled(),
            0
        );
        assert_eq!(
            SpinState::new(spin_half, m!(-1 / 2))
                .unwrap()
                .projection()
                .doubled(),
            -1
        );
    }

    #[test]
    fn spin_state_rejects_invalid_projection() {
        let spin_one = j!(1);
        assert!(SpinState::new(spin_one, m!(4 / 2)).is_err());
        assert!(SpinState::new(spin_one, m!(1 / 2)).is_err());
    }

    #[test]
    fn angular_momenta_return_projection_values() {
        assert_eq!(j!(1).projections(), vec![m!(-1), m!(0), m!(1)]);
        assert_eq!(
            j!(3 / 2).projections(),
            vec![m!(-3 / 2), m!(-1 / 2), m!(1 / 2), m!(3 / 2)]
        );
        assert_eq!(l!(1).projections(), j!(1).projections());
    }

    #[test]
    fn enum_displays() {
        assert_eq!(format!("{}", Reflectivity::Positive), "+");
        assert_eq!(format!("{}", Reflectivity::Negative), "-");
        assert_eq!(format!("{}", MandelstamChannel::S), "s");
        assert_eq!(format!("{}", MandelstamChannel::T), "t");
        assert_eq!(format!("{}", MandelstamChannel::U), "u");
    }

    #[test]
    fn enum_from_str() {
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
        assert_eq!(
            MandelstamChannel::from_str("S").unwrap(),
            MandelstamChannel::S
        );
        assert_eq!(
            MandelstamChannel::from_str("s").unwrap(),
            MandelstamChannel::S
        );
        assert_eq!(
            MandelstamChannel::from_str("T").unwrap(),
            MandelstamChannel::T
        );
        assert_eq!(
            MandelstamChannel::from_str("t").unwrap(),
            MandelstamChannel::T
        );
        assert_eq!(
            MandelstamChannel::from_str("U").unwrap(),
            MandelstamChannel::U
        );
        assert_eq!(
            MandelstamChannel::from_str("u").unwrap(),
            MandelstamChannel::U
        );
    }
}
