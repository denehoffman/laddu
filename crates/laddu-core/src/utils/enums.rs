use std::{fmt::Display, str::FromStr};

use polars::prelude::PlSmallStr;
use serde::{Deserialize, Serialize};

use crate::{utils::list_to_name, LadduError};

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

pub enum Topology {
    All {
        p1: Vec<PlSmallStr>,
        p2: Vec<PlSmallStr>,
        p3: Vec<PlSmallStr>,
        p4: Vec<PlSmallStr>,
    },
    MissingP1 {
        p2: Vec<PlSmallStr>,
        p3: Vec<PlSmallStr>,
        p4: Vec<PlSmallStr>,
    },
    MissingP2 {
        p1: Vec<PlSmallStr>,
        p3: Vec<PlSmallStr>,
        p4: Vec<PlSmallStr>,
    },
    MissingP3 {
        p1: Vec<PlSmallStr>,
        p2: Vec<PlSmallStr>,
        p4: Vec<PlSmallStr>,
    },
    MissingP4 {
        p1: Vec<PlSmallStr>,
        p2: Vec<PlSmallStr>,
        p3: Vec<PlSmallStr>,
    },
}

impl Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Topology::All { p1, p2, p3, p4 } => write!(
                f,
                "Topology::All(p1: [{}], p2: [{}], p3: [{}], p4: [{}])",
                list_to_name(p1),
                list_to_name(p2),
                list_to_name(p3),
                list_to_name(p4)
            ),
            Topology::MissingP1 { p2, p3, p4 } => write!(
                f,
                "Topology::MissingP1(p2: [{}], p3: [{}], p4: [{}])",
                list_to_name(p2),
                list_to_name(p3),
                list_to_name(p4)
            ),
            Topology::MissingP2 { p1, p3, p4 } => write!(
                f,
                "Topology::MissingP2(p1: [{}], p3: [{}], p4: [{}])",
                list_to_name(p1),
                list_to_name(p3),
                list_to_name(p4)
            ),
            Topology::MissingP3 { p1, p2, p4 } => write!(
                f,
                "Topology::MissingP3(p1: [{}], p2: [{}], p4: [{}])",
                list_to_name(p1),
                list_to_name(p2),
                list_to_name(p4)
            ),
            Topology::MissingP4 { p1, p2, p3 } => write!(
                f,
                "Topology::MissingP4(p1: [{}], p2: [{}], p3: [{}])",
                list_to_name(p1),
                list_to_name(p2),
                list_to_name(p3)
            ),
        }
    }
}

impl Topology {
    #[inline]
    fn to_vecs<P, S>(xs: P) -> Vec<PlSmallStr>
    where
        P: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        xs.into_iter().map(|s| s.into()).collect()
    }

    pub fn all<P1, S1, P2, S2, P3, S3, P4, S4>(p1: P1, p2: P2, p3: P3, p4: P4) -> Self
    where
        P1: IntoIterator<Item = S1>,
        S1: Into<PlSmallStr>,
        P2: IntoIterator<Item = S2>,
        S2: Into<PlSmallStr>,
        P3: IntoIterator<Item = S3>,
        S3: Into<PlSmallStr>,
        P4: IntoIterator<Item = S4>,
        S4: Into<PlSmallStr>,
    {
        Self::All {
            p1: Self::to_vecs(p1),
            p2: Self::to_vecs(p2),
            p3: Self::to_vecs(p3),
            p4: Self::to_vecs(p4),
        }
    }

    pub fn missing_p1<P2, S2, P3, S3, P4, S4>(p2: P2, p3: P3, p4: P4) -> Self
    where
        P2: IntoIterator<Item = S2>,
        S2: Into<PlSmallStr>,
        P3: IntoIterator<Item = S3>,
        S3: Into<PlSmallStr>,
        P4: IntoIterator<Item = S4>,
        S4: Into<PlSmallStr>,
    {
        Self::MissingP1 {
            p2: Self::to_vecs(p2),
            p3: Self::to_vecs(p3),
            p4: Self::to_vecs(p4),
        }
    }

    pub fn missing_p2<P1, S1, P3, S3, P4, S4>(p1: P1, p3: P3, p4: P4) -> Self
    where
        P1: IntoIterator<Item = S1>,
        S1: Into<PlSmallStr>,
        P3: IntoIterator<Item = S3>,
        S3: Into<PlSmallStr>,
        P4: IntoIterator<Item = S4>,
        S4: Into<PlSmallStr>,
    {
        Self::MissingP2 {
            p1: Self::to_vecs(p1),
            p3: Self::to_vecs(p3),
            p4: Self::to_vecs(p4),
        }
    }

    pub fn missing_p3<P1, S1, P2, S2, P4, S4>(p1: P1, p2: P2, p4: P4) -> Self
    where
        P1: IntoIterator<Item = S1>,
        S1: Into<PlSmallStr>,
        P2: IntoIterator<Item = S2>,
        S2: Into<PlSmallStr>,
        P4: IntoIterator<Item = S4>,
        S4: Into<PlSmallStr>,
    {
        Self::MissingP3 {
            p1: Self::to_vecs(p1),
            p2: Self::to_vecs(p2),
            p4: Self::to_vecs(p4),
        }
    }

    pub fn missing_p4<P1, S1, P2, S2, P3, S3>(p1: P1, p2: P2, p3: P3) -> Self
    where
        P1: IntoIterator<Item = S1>,
        S1: Into<PlSmallStr>,
        P2: IntoIterator<Item = S2>,
        S2: Into<PlSmallStr>,
        P3: IntoIterator<Item = S3>,
        S3: Into<PlSmallStr>,
    {
        Self::MissingP4 {
            p1: Self::to_vecs(p1),
            p2: Self::to_vecs(p2),
            p3: Self::to_vecs(p3),
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
