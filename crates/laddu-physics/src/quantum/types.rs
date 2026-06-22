use std::{fmt::Display, str::FromStr};

use auto_ops::impl_op_ex;
use num::rational::Ratio;
use serde::{Deserialize, Serialize};

use crate::LadduPhysicsError;

const QUANTUM_NUMBER_FLOAT_TOLERANCE: f64 = 1.0e-12;

macro_rules! impl_try_from_str {
    ($ty:ty) => {
        impl ::std::convert::TryFrom<&str> for $ty {
            type Error = <$ty as ::std::str::FromStr>::Err;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                <Self as ::std::str::FromStr>::from_str(value)
            }
        }

        impl ::std::convert::TryFrom<String> for $ty {
            type Error = <$ty as ::std::str::FromStr>::Err;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                <Self as ::std::str::FromStr>::from_str(&value)
            }
        }
    };
}

macro_rules! impl_try_from_signed_ints {
    ($ty:ty; $($int:ty),+ $(,)?) => {
        $(
            impl TryFrom<$int> for $ty {
                type Error = LadduPhysicsError;

                fn try_from(value: $int) -> Result<Self, Self::Error> {
                    <$ty>::try_from_i128(value as i128)
                }
            }
        )+
    };
}

macro_rules! impl_try_from_unsigned_ints {
    ($ty:ty; $($int:ty),+ $(,)?) => {
        $(
            impl TryFrom<$int> for $ty {
                type Error = LadduPhysicsError;

                fn try_from(value: $int) -> Result<Self, Self::Error> {
                    <$ty>::try_from_u128(value as u128)
                }
            }
        )+
    };
}

macro_rules! impl_try_from_signed_ratios {
    ($ty:ty; $($int:ty),+ $(,)?) => {
        $(
            impl TryFrom<Ratio<$int>> for $ty {
                type Error = LadduPhysicsError;

                fn try_from(value: Ratio<$int>) -> Result<Self, Self::Error> {
                    <$ty>::try_from_ratio_i128(
                        *value.numer() as i128,
                        *value.denom() as i128,
                    )
                }
            }
        )+
    };
}

macro_rules! impl_try_from_unsigned_ratios {
    ($ty:ty; $($int:ty),+ $(,)?) => {
        $(
            impl TryFrom<Ratio<$int>> for $ty {
                type Error = LadduPhysicsError;

                fn try_from(value: Ratio<$int>) -> Result<Self, Self::Error> {
                    let numer = i128::try_from(*value.numer()).map_err(|_| {
                        LadduPhysicsError::invalid_value(
        "ratio numerator",
        "representable as i128",
        *value.numer(),
    )
                    })?;
                    let denom = i128::try_from(*value.denom()).map_err(|_| {
                        LadduPhysicsError::invalid_value(
        "ratio denominator",
        "representable as i128",
        *value.denom(),
    )
                    })?;
                    <$ty>::try_from_ratio_i128(numer, denom)
                }
            }
        )+
    };
}

macro_rules! impl_try_from_floats {
    ($ty:ty; $($float:ty),+ $(,)?) => {
        $(
            impl TryFrom<$float> for $ty {
                type Error = LadduPhysicsError;

                fn try_from(value: $float) -> Result<Self, Self::Error> {
                    <$ty>::try_from_f64(value as f64)
                }
            }
        )+
    };
}

macro_rules! impl_from_quantum_number_for_floats {
    ($ty:ty, $getter:ident, $scale:expr) => {
        impl From<$ty> for f32 {
            fn from(value: $ty) -> Self {
                value.$getter() as Self / $scale as Self
            }
        }

        impl From<$ty> for f64 {
            fn from(value: $ty) -> Self {
                value.$getter() as Self / $scale as Self
            }
        }
    };
}

macro_rules! impl_half_integer_display {
    ($ty:ty, $getter:ident) => {
        impl Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let value = self.$getter();
                if value % 2 == 0 {
                    write!(f, "{}", value / 2)
                } else {
                    write!(f, "{value}/2")
                }
            }
        }
    };
}

macro_rules! impl_j_conversions {
    () => {
        impl_try_from_signed_ints!(J; i8, i16, i32, i64, i128, isize);
        impl_try_from_unsigned_ints!(J; u8, u16, u32, u64, u128, usize);
        impl_try_from_signed_ratios!(J; i8, i16, i32, i64, i128, isize);
        impl_try_from_unsigned_ratios!(J; u8, u16, u32, u64, u128, usize);
        impl_try_from_floats!(J; f32, f64);
        impl_from_quantum_number_for_floats!(J, doubled, 2);
        impl_half_integer_display!(J, doubled);
    };
}

macro_rules! impl_l_conversions {
    () => {
        impl_try_from_signed_ints!(L; i8, i16, i32, i64, i128, isize);
        impl_try_from_unsigned_ints!(L; u8, u16, u32, u64, u128, usize);
        impl_try_from_signed_ratios!(L; i8, i16, i32, i64, i128, isize);
        impl_try_from_unsigned_ratios!(L; u8, u16, u32, u64, u128, usize);
        impl_try_from_floats!(L; f32, f64);
        impl_from_quantum_number_for_floats!(L, value, 1);
    };
}

macro_rules! impl_m_conversions {
    () => {
        impl_try_from_signed_ints!(M; i8, i16, i32, i64, i128, isize);
        impl_try_from_unsigned_ints!(M; u8, u16, u32, u64, u128, usize);
        impl_try_from_signed_ratios!(M; i8, i16, i32, i64, i128, isize);
        impl_try_from_unsigned_ratios!(M; u8, u16, u32, u64, u128, usize);
        impl_try_from_floats!(M; f32, f64);
        impl_from_quantum_number_for_floats!(M, doubled, 2);
        impl_half_integer_display!(M, doubled);
    };
}

pub mod signed {
    use super::*;

    macro_rules! signed_value_type {
        (
            $(#[$meta:meta])*
            $vis:vis enum $name:ident, $object:literal
        ) => {
            signed_value_type! {
                @impl
                $(#[$meta])*
                $vis enum $name, $object;
                i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64
            }
        };

        (
            @impl
            $(#[$meta:meta])*
            $vis:vis enum $name:ident, $object:literal;
            $($number:ty)*
        ) => {
            $(#[$meta])*
            #[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
            $vis enum $name {
                /// A positive value.
                Positive,
                /// A negative value.
                Negative,
            }

            impl $name {
                pub const fn value(self) -> i32 {
                    match self {
                        Self::Positive => 1,
                        Self::Negative => -1,
                    }
                }
            }

            impl ::std::fmt::Display for $name {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    match self {
                        Self::Positive => write!(f, "+"),
                        Self::Negative => write!(f, "-"),
                    }
                }
            }

            impl ::std::str::FromStr for $name {
                type Err = LadduPhysicsError;

                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    match parse_sign_value(s, $object)? {
                        Sign::Positive => Ok(Self::Positive),
                        Sign::Negative => Ok(Self::Negative),
                    }
                }
            }

            impl_try_from_str!($name);

            $(
                impl From<$name> for $number {
                    fn from(value: $name) -> Self {
                        value.value() as $number
                    }
                }
            )*

            impl_op_ex!(* |p1: &$name, p2: &$name| -> $name {
                match (p1, p2) {
                    ($name::Positive, $name::Positive)
                    | ($name::Negative, $name::Negative) => $name::Positive,

                    ($name::Positive, $name::Negative)
                    | ($name::Negative, $name::Positive) => $name::Negative,
                }
            });

            impl_op_ex!(*= |p1: &mut $name, p2: &$name| {
                *p1 = *p1 * p2
            });

            impl_op_ex!(- |p: &$name| -> $name {
                match p {
                    $name::Positive => $name::Negative,
                    $name::Negative => $name::Positive,
                }
            });
        };
    }

    signed_value_type! {
        /// A helper enum denoting the sign of a state.
        pub enum Sign, "Sign"
    }

    signed_value_type! {
        /// An enum describing the reflectivity of a state.
        pub enum Reflectivity, "Reflectivity"
    }

    signed_value_type! {
        /// An enum describing the parity (or G/C-parity) of a state.
        pub enum Parity, "Parity"
    }

    fn parse_sign_value(s: &str, object: &str) -> Result<Sign, LadduPhysicsError> {
        match s.to_lowercase().as_ref() {
            "+" | "plus" | "pos" | "positive" => Ok(Sign::Positive),
            "-" | "minus" | "neg" | "negative" => Ok(Sign::Negative),
            _ => Err(LadduPhysicsError::ParseError {
                name: s.to_string(),
                object: object.to_string(),
            }),
        }
    }

    impl From<Reflectivity> for Sign {
        fn from(value: Reflectivity) -> Self {
            match value {
                Reflectivity::Positive => Self::Positive,
                Reflectivity::Negative => Self::Negative,
            }
        }
    }

    impl From<Sign> for Reflectivity {
        fn from(value: Sign) -> Self {
            match value {
                Sign::Positive => Self::Positive,
                Sign::Negative => Self::Negative,
            }
        }
    }

    impl From<Parity> for Sign {
        fn from(value: Parity) -> Self {
            match value {
                Parity::Positive => Self::Positive,
                Parity::Negative => Self::Negative,
            }
        }
    }

    impl From<Sign> for Parity {
        fn from(value: Sign) -> Self {
            match value {
                Sign::Positive => Self::Positive,
                Sign::Negative => Self::Negative,
            }
        }
    }
}
pub use signed::*;

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

    fn try_from_i128(value: i128) -> Result<Self, LadduPhysicsError> {
        if value < 0 {
            return Err(LadduPhysicsError::invalid_value(
                "angular momentum",
                "nonnegative",
                value,
            ));
        }
        Self::try_from_scaled_i128(value.checked_mul(2).ok_or_else(|| {
            LadduPhysicsError::numeric_overflow(format!("2 * angular momentum for value {value}"))
        })?)
    }

    fn try_from_u128(value: u128) -> Result<Self, LadduPhysicsError> {
        let value = i128::try_from(value).map_err(|_| {
            LadduPhysicsError::invalid_value("angular momentum", "representable as i128", value)
        })?;
        Self::try_from_i128(value)
    }

    fn try_from_ratio_i128(numer: i128, denom: i128) -> Result<Self, LadduPhysicsError> {
        let scaled = numer.checked_mul(2).ok_or_else(|| {
            LadduPhysicsError::numeric_overflow(format!("2 * angular momentum numerator {numer}"))
        })?;
        if scaled % denom != 0 {
            return Err(LadduPhysicsError::invalid_value(
                "angular momentum",
                "integer or half-integer",
                format!("{numer}/{denom}"),
            ));
        }
        Self::try_from_scaled_i128(scaled / denom)
    }

    fn try_from_f64(value: f64) -> Result<Self, LadduPhysicsError> {
        if !value.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "angular momentum",
                "finite",
                value,
            ));
        }
        let scaled = 2.0 * value;
        let rounded = scaled.round();
        if (scaled - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduPhysicsError::invalid_value(
                "angular momentum",
                "integer or half-integer",
                value,
            ));
        }
        if rounded < i128::MIN as f64 || rounded > i128::MAX as f64 {
            return Err(LadduPhysicsError::invalid_value(
                "angular momentum",
                "representable as i128",
                rounded,
            ));
        }
        Self::try_from_scaled_i128(rounded as i128)
    }

    fn try_from_scaled_i128(value: i128) -> Result<Self, LadduPhysicsError> {
        Ok(Self(u32::try_from(value).map_err(|_| {
            LadduPhysicsError::invalid_value(
                "angular momentum",
                "nonnegative and representable as doubled u32",
                value,
            )
        })?))
    }
}

impl From<L> for J {
    fn from(value: L) -> Self {
        Self::int(value.0)
    }
}

impl_j_conversions!();

/// Coupled intrinsic spin, represented by the same quantum-number type as total angular momentum.
pub type S = J;

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

    /// Returns the parity derived from the orbital angular momentum, $`(-1)^{\ell}`$
    pub const fn orbital_parity(self) -> Parity {
        if self.value().is_multiple_of(2) {
            Parity::Positive
        } else {
            Parity::Negative
        }
    }

    fn try_from_i128(value: i128) -> Result<Self, LadduPhysicsError> {
        if value < 0 {
            return Err(LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "nonnegative",
                value,
            ));
        }
        Self::try_from_scaled_i128(value)
    }

    fn try_from_u128(value: u128) -> Result<Self, LadduPhysicsError> {
        let value = i128::try_from(value).map_err(|_| {
            LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "representable as i128",
                value,
            )
        })?;
        Self::try_from_i128(value)
    }

    fn try_from_ratio_i128(numer: i128, denom: i128) -> Result<Self, LadduPhysicsError> {
        if numer % denom != 0 {
            return Err(LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "integer",
                format!("{numer}/{denom}"),
            ));
        }
        Self::try_from_scaled_i128(numer / denom)
    }

    fn try_from_f64(value: f64) -> Result<Self, LadduPhysicsError> {
        if !value.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "finite",
                value,
            ));
        }
        let rounded = value.round();
        if (value - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "integer",
                value,
            ));
        }
        if rounded < i128::MIN as f64 || rounded > i128::MAX as f64 {
            return Err(LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "representable as i128",
                rounded,
            ));
        }
        Self::try_from_scaled_i128(rounded as i128)
    }

    fn try_from_scaled_i128(value: i128) -> Result<Self, LadduPhysicsError> {
        Ok(Self(u32::try_from(value).map_err(|_| {
            LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "nonnegative and representable as u32",
                value,
            )
        })?))
    }
}

impl TryFrom<J> for L {
    type Error = LadduPhysicsError;

    fn try_from(value: J) -> Result<Self, Self::Error> {
        if !value.is_integer() {
            return Err(LadduPhysicsError::invalid_value(
                "orbital angular momentum",
                "integer",
                value,
            ));
        }
        Ok(Self::int(value.doubled() / 2))
    }
}

impl_l_conversions!();

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

    /// Return whether this projection represents an integer value.
    pub const fn is_integer(self) -> bool {
        self.0 & 1 == 0
    }

    fn try_from_i128(value: i128) -> Result<Self, LadduPhysicsError> {
        Self::try_from_scaled_i128(value.checked_mul(2).ok_or_else(|| {
            LadduPhysicsError::numeric_overflow(format!("2 * projection for value {value}"))
        })?)
    }

    fn try_from_u128(value: u128) -> Result<Self, LadduPhysicsError> {
        let value = i128::try_from(value).map_err(|_| {
            LadduPhysicsError::invalid_value("projection", "representable as i128", value)
        })?;
        Self::try_from_i128(value)
    }

    fn try_from_ratio_i128(numer: i128, denom: i128) -> Result<Self, LadduPhysicsError> {
        let scaled = numer.checked_mul(2).ok_or_else(|| {
            LadduPhysicsError::numeric_overflow(format!("2 * projection numerator {numer}"))
        })?;
        if scaled % denom != 0 {
            return Err(LadduPhysicsError::invalid_value(
                "projection",
                "integer or half-integer",
                format!("{numer}/{denom}"),
            ));
        }
        Self::try_from_scaled_i128(scaled / denom)
    }

    fn try_from_f64(value: f64) -> Result<Self, LadduPhysicsError> {
        if !value.is_finite() {
            return Err(LadduPhysicsError::Custom(
                "projection must be finite".to_string(),
            ));
        }
        let scaled = 2.0 * value;
        let rounded = scaled.round();
        if (scaled - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduPhysicsError::invalid_value(
                "projection",
                "integer or half-integer",
                value,
            ));
        }
        if rounded < i128::MIN as f64 || rounded > i128::MAX as f64 {
            return Err(LadduPhysicsError::invalid_value(
                "projection",
                "representable as i128",
                rounded,
            ));
        }
        Self::try_from_scaled_i128(rounded as i128)
    }

    fn try_from_scaled_i128(value: i128) -> Result<Self, LadduPhysicsError> {
        Ok(Self(i32::try_from(value).map_err(|_| {
            LadduPhysicsError::invalid_value("projection", "representable as i32", value)
        })?))
    }
}

impl_m_conversions!();

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

/// An enum representing the statistics of a particle.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
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

impl FromStr for Statistics {
    type Err = LadduPhysicsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "fermion" => Ok(Self::Fermion),
            "boson" => Ok(Self::Boson),
            _ => Err(LadduPhysicsError::ParseError {
                name: s.to_string(),
                object: "Statistics".to_string(),
            }),
        }
    }
}

impl_try_from_str!(Statistics);

/// An enum for Mandelstam variables.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
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
    type Err = LadduPhysicsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "s" => Ok(Self::S),
            "t" => Ok(Self::T),
            "u" => Ok(Self::U),
            _ => Err(LadduPhysicsError::ParseError {
                name: s.to_string(),
                object: "MandelstamChannel".to_string(),
            }),
        }
    }
}

impl_try_from_str!(MandelstamChannel);

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
    fn quantum_numbers_convert_to_floats() {
        assert_eq!(f64::from(J::half(3)), 1.5);
        assert_eq!(f32::from(L::int(2)), 2.0);
        assert_eq!(f64::from(M::half(-1)), -0.5);
    }

    #[test]
    fn quantum_numbers_accept_more_numeric_inputs() {
        assert_eq!(J::try_from(2_u8).unwrap().doubled(), 4);
        assert_eq!(L::try_from(2_u16).unwrap().value(), 2);
        assert_eq!(M::try_from(-2_i8).unwrap().doubled(), -4);
        assert!(J::try_from(-1_i8).is_err());
        assert!(L::try_from(-1_i8).is_err());
    }
}
