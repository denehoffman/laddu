use std::{fmt::Display, str::FromStr};

/// Standard reference frames for angular analyses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "helicity" | "hx" | "hel" => Ok(Self::Helicity),
            "gottfriedjackson" | "gottfried jackson" | "gj" | "gottfried-jackson" => {
                Ok(Self::Helicity)
            }
            _ => Err("Invalid frame".to_string()),
        }
    }
}

/// A simple enum describing a binary sign.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "+" | "plus" | "pos" | "positive" => Ok(Self::Positive),
            "-" | "minus" | "neg" | "negative" => Ok(Self::Negative),
            _ => Err("Invalid sign".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::enums::{Frame, Sign};

    #[test]
    fn enum_displays() {
        assert_eq!(format!("{}", Frame::Helicity), "Helicity");
        assert_eq!(format!("{}", Frame::GottfriedJackson), "Gottfried-Jackson");
        assert_eq!(format!("{}", Sign::Positive), "+");
        assert_eq!(format!("{}", Sign::Negative), "-");
    }
}
