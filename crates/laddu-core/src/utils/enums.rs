use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::LadduError;

/// Standard reference frames for angular analyses.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Frame {
    /// The helicity frame, obtained by setting the $`z`$-axis equal to the boost direction from
    /// the center-of-momentum to the rest frame of the resonance in question and the $`y`$-axis
    /// perpendicular to the production plane.
    Helicity,
    /// The Gottfried-Jackson frame, obtained by setting the $`z`$-axis proportional to the beam's
    /// direction in the rest frame of the resonance in question and the $`y`$-axis perpendicular
    /// to the production plane.
    GottfriedJackson,
}
impl Display for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Frame::Helicity => write!(f, "Helicity"),
            Frame::GottfriedJackson => write!(f, "Gottfried-Jackson"),
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
            _ => Err(LadduError::ParseError {
                name: s.to_string(),
                object: "Frame".to_string(),
            }),
        }
    }
}

/// A simple enum describing a binary sign.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sign {
    /// A positive indicator.
    Positive,
    /// A negative indicator.
    Negative,
}
impl Display for Sign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sign::Positive => write!(f, "+"),
            Sign::Negative => write!(f, "-"),
        }
    }
}

impl FromStr for Sign {
    type Err = LadduError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "+" | "plus" | "pos" | "positive" => Ok(Self::Positive),
            "-" | "minus" | "neg" | "negative" => Ok(Self::Negative),
            _ => Err(LadduError::ParseError {
                name: s.to_string(),
                object: "Sign".to_string(),
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
    use std::str::FromStr;

    #[test]
    fn enum_displays() {
        assert_eq!(format!("{}", Frame::Helicity), "Helicity");
        assert_eq!(format!("{}", Frame::GottfriedJackson), "Gottfried-Jackson");
        assert_eq!(format!("{}", Sign::Positive), "+");
        assert_eq!(format!("{}", Sign::Negative), "-");
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
        assert_eq!(Sign::from_str("+").unwrap(), Sign::Positive);
        assert_eq!(Sign::from_str("pos").unwrap(), Sign::Positive);
        assert_eq!(Sign::from_str("plus").unwrap(), Sign::Positive);
        assert_eq!(Sign::from_str("Positive").unwrap(), Sign::Positive);
        assert_eq!(Sign::from_str("-").unwrap(), Sign::Negative);
        assert_eq!(Sign::from_str("minus").unwrap(), Sign::Negative);
        assert_eq!(Sign::from_str("neg").unwrap(), Sign::Negative);
        assert_eq!(Sign::from_str("Negative").unwrap(), Sign::Negative);
        assert_eq!(Channel::from_str("S").unwrap(), Channel::S);
        assert_eq!(Channel::from_str("s").unwrap(), Channel::S);
        assert_eq!(Channel::from_str("T").unwrap(), Channel::T);
        assert_eq!(Channel::from_str("t").unwrap(), Channel::T);
        assert_eq!(Channel::from_str("U").unwrap(), Channel::U);
        assert_eq!(Channel::from_str("u").unwrap(), Channel::U);
    }
}
