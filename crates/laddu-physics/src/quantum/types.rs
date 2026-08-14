use std::{fmt::Display, str::FromStr};

use auto_ops::impl_op_ex;
use num::rational::Ratio;
use serde::{Deserialize, Serialize};

use crate::LadduPhysicsError;

const QUANTUM_NUMBER_FLOAT_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Copy)]
enum Signedness {
    Nonnegative,
    Signed,
}

#[derive(Clone, Copy)]
enum NonFiniteError {
    InvalidValue,
    Custom(&'static str),
}

#[derive(Clone, Copy)]
struct QuantumNumberConversion {
    scale: i128,
    signedness: Signedness,
    storage_min: i128,
    storage_max: i128,
    domain: &'static str,
    physical_values: &'static str,
    storage_values: &'static str,
    non_finite: NonFiniteError,
}

impl QuantumNumberConversion {
    fn signed(self, value: i128) -> Result<i128, LadduPhysicsError> {
        if matches!(self.signedness, Signedness::Nonnegative) && value < 0 {
            return Err(LadduPhysicsError::invalid_value(
                self.domain,
                "nonnegative",
                value,
            ));
        }
        let scaled = self.scale_integer(value, format!("{} for value {value}", self.domain))?;
        self.scaled(scaled)
    }

    fn unsigned(self, value: u128) -> Result<i128, LadduPhysicsError> {
        let value = i128::try_from(value).map_err(|_| {
            LadduPhysicsError::invalid_value(self.domain, "representable as i128", value)
        })?;
        self.signed(value)
    }

    fn ratio(self, numer: i128, denom: i128) -> Result<i128, LadduPhysicsError> {
        let scaled = self.scale_integer(numer, format!("{} numerator {numer}", self.domain))?;
        if scaled % denom != 0 {
            return Err(LadduPhysicsError::invalid_value(
                self.domain,
                self.physical_values,
                format!("{numer}/{denom}"),
            ));
        }
        self.scaled(scaled / denom)
    }

    fn float(self, value: f64) -> Result<i128, LadduPhysicsError> {
        if !value.is_finite() {
            return Err(match self.non_finite {
                NonFiniteError::InvalidValue => {
                    LadduPhysicsError::invalid_value(self.domain, "finite", value)
                }
                NonFiniteError::Custom(message) => LadduPhysicsError::Custom(message.to_string()),
            });
        }
        let scaled = self.scale as f64 * value;
        let rounded = scaled.round();
        if (scaled - rounded).abs() > QUANTUM_NUMBER_FLOAT_TOLERANCE {
            return Err(LadduPhysicsError::invalid_value(
                self.domain,
                self.physical_values,
                value,
            ));
        }
        if rounded < i128::MIN as f64 || rounded > i128::MAX as f64 {
            return Err(LadduPhysicsError::invalid_value(
                self.domain,
                "representable as i128",
                rounded,
            ));
        }
        self.scaled(rounded as i128)
    }

    fn scale_integer(self, value: i128, operation: String) -> Result<i128, LadduPhysicsError> {
        if self.scale == 1 {
            return Ok(value);
        }
        value.checked_mul(self.scale).ok_or_else(|| {
            LadduPhysicsError::numeric_overflow(format!("{} * {operation}", self.scale))
        })
    }

    fn scaled(self, value: i128) -> Result<i128, LadduPhysicsError> {
        if value < self.storage_min || value > self.storage_max {
            return Err(LadduPhysicsError::invalid_value(
                self.domain,
                self.storage_values,
                value,
            ));
        }
        Ok(value)
    }
}

trait ScaledQuantumNumber: Sized {
    const CONVERSION: QuantumNumberConversion;

    fn from_scaled(value: i128) -> Self;
}

fn quantum_from_signed<T: ScaledQuantumNumber>(value: i128) -> Result<T, LadduPhysicsError> {
    Ok(T::from_scaled(T::CONVERSION.signed(value)?))
}

fn quantum_from_unsigned<T: ScaledQuantumNumber>(value: u128) -> Result<T, LadduPhysicsError> {
    Ok(T::from_scaled(T::CONVERSION.unsigned(value)?))
}

fn quantum_from_ratio<T: ScaledQuantumNumber>(
    numer: i128,
    denom: i128,
) -> Result<T, LadduPhysicsError> {
    Ok(T::from_scaled(T::CONVERSION.ratio(numer, denom)?))
}

fn quantum_from_float<T: ScaledQuantumNumber>(value: f64) -> Result<T, LadduPhysicsError> {
    Ok(T::from_scaled(T::CONVERSION.float(value)?))
}

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
                    quantum_from_signed::<$ty>(value as i128)
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
                    quantum_from_unsigned::<$ty>(value as u128)
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
                    quantum_from_ratio::<$ty>(
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
                    quantum_from_ratio::<$ty>(numer, denom)
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
                    quantum_from_float::<$ty>(value as f64)
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

/// Signed discrete quantum numbers such as parity and reflectivity.
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
                /// Return `+1` or `-1` for this signed quantum number.
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

    /// Return the number of projections, $2J + 1$.
    pub const fn multiplicity(self) -> u32 {
        self.0 + 1
    }

    /// Return whether this quantum number represents an integer value.
    pub const fn is_integer(self) -> bool {
        self.0 & 1 == 0
    }

    /// Enumerate the total angular momenta obtainable by coupling this value to `other`.
    ///
    /// Results run from $\lvert J_1 - J_2\rvert$ through $J_1 + J_2$ in unit steps.
    pub fn coupled_with(self, other: Self) -> Vec<Self> {
        let min = self.doubled().abs_diff(other.doubled());
        let max = self.doubled() + other.doubled();
        (min..=max).step_by(2).map(Self::half).collect()
    }

    /// Return whether this angular momentum has the same integer/half-integer parity as
    /// `projection`.
    pub(crate) const fn has_same_parity_as(self, projection: M) -> bool {
        (self.0 & 1) as i32 == projection.doubled() & 1
    }

    /// Return whether `j1` and `j2` can couple to produce this angular momentum.
    pub fn can_couple_to(self, j1: Self, j2: Self) -> bool {
        let min = j1.doubled().abs_diff(j2.doubled());
        let max = j1.doubled() + j2.doubled();
        self.doubled() >= min && self.doubled() <= max && (self.doubled() - min).is_multiple_of(2)
    }
}

impl ScaledQuantumNumber for J {
    const CONVERSION: QuantumNumberConversion = QuantumNumberConversion {
        scale: 2,
        signedness: Signedness::Nonnegative,
        storage_min: 0,
        storage_max: u32::MAX as i128,
        domain: "angular momentum",
        physical_values: "integer or half-integer",
        storage_values: "nonnegative and representable as doubled u32",
        non_finite: NonFiniteError::InvalidValue,
    };

    fn from_scaled(value: i128) -> Self {
        Self(value as u32)
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

    /// Return the number of projections, $2L + 1$.
    pub const fn multiplicity(self) -> u32 {
        2 * self.0 + 1
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
}

impl ScaledQuantumNumber for L {
    const CONVERSION: QuantumNumberConversion = QuantumNumberConversion {
        scale: 1,
        signedness: Signedness::Nonnegative,
        storage_min: 0,
        storage_max: u32::MAX as i128,
        domain: "orbital angular momentum",
        physical_values: "integer",
        storage_values: "nonnegative and representable as u32",
        non_finite: NonFiniteError::InvalidValue,
    };

    fn from_scaled(value: i128) -> Self {
        Self(value as u32)
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
}

impl ScaledQuantumNumber for M {
    const CONVERSION: QuantumNumberConversion = QuantumNumberConversion {
        scale: 2,
        signedness: Signedness::Signed,
        storage_min: i32::MIN as i128,
        storage_max: i32::MAX as i128,
        domain: "projection",
        physical_values: "integer or half-integer",
        storage_values: "representable as i32",
        non_finite: NonFiniteError::Custom("projection must be finite"),
    };

    fn from_scaled(value: i128) -> Self {
        Self(value as i32)
    }
}

impl_m_conversions!();

// Operations
#[rustfmt::skip]
impl_op_ex!(+ |j1: &J, j2: &J| -> J { J::half(j1.doubled() + j2.doubled()) });
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
    fn angular_momenta_add_and_report_multiplicities() {
        assert_eq!(J::int(1) + J::half(1), J::half(3));
        assert_eq!(J::half(3).multiplicity(), 4);
        assert_eq!(L::int(2).multiplicity(), 5);
    }

    #[test]
    fn angular_momenta_enumerate_and_validate_couplings() {
        assert_eq!(
            J::half(1).coupled_with(J::int(1)),
            vec![J::half(1), J::half(3)]
        );
        assert!(J::half(3).can_couple_to(J::half(1), J::int(1)));
        assert!(!J::int(0).can_couple_to(J::half(1), J::int(1)));
    }

    #[test]
    fn quantum_numbers_accept_more_numeric_inputs() {
        assert_eq!(J::try_from(2_u8).unwrap().doubled(), 4);
        assert_eq!(L::try_from(2_u16).unwrap().value(), 2);
        assert_eq!(M::try_from(-2_i8).unwrap().doubled(), -4);
        assert!(J::try_from(-1_i8).is_err());
        assert!(L::try_from(-1_i8).is_err());
    }

    #[test]
    fn every_integer_conversion_uses_the_same_physical_scaling() {
        macro_rules! assert_signed {
            ($($ty:ty),+ $(,)?) => {$({
                assert_eq!(J::try_from(2 as $ty).unwrap(), J::int(2));
                assert_eq!(L::try_from(2 as $ty).unwrap(), L::int(2));
                assert_eq!(M::try_from(-2 as $ty).unwrap(), M::int(-2));
            })+};
        }
        macro_rules! assert_unsigned {
            ($($ty:ty),+ $(,)?) => {$({
                assert_eq!(J::try_from(2 as $ty).unwrap(), J::int(2));
                assert_eq!(L::try_from(2 as $ty).unwrap(), L::int(2));
                assert_eq!(M::try_from(2 as $ty).unwrap(), M::int(2));
            })+};
        }

        assert_signed!(i8, i16, i32, i64, i128, isize);
        assert_unsigned!(u8, u16, u32, u64, u128, usize);
    }

    #[test]
    fn every_ratio_conversion_uses_the_same_physical_scaling() {
        macro_rules! assert_signed {
            ($($ty:ty),+ $(,)?) => {$({
                assert_eq!(
                    J::try_from(Ratio::<$ty>::new(3, 2)).unwrap(),
                    J::half(3)
                );
                assert_eq!(
                    L::try_from(Ratio::<$ty>::new(2, 1)).unwrap(),
                    L::int(2)
                );
                assert_eq!(
                    M::try_from(Ratio::<$ty>::new(-3, 2)).unwrap(),
                    M::half(-3)
                );
            })+};
        }
        macro_rules! assert_unsigned {
            ($($ty:ty),+ $(,)?) => {$({
                assert_eq!(
                    J::try_from(Ratio::<$ty>::new(3, 2)).unwrap(),
                    J::half(3)
                );
                assert_eq!(
                    L::try_from(Ratio::<$ty>::new(2, 1)).unwrap(),
                    L::int(2)
                );
                assert_eq!(
                    M::try_from(Ratio::<$ty>::new(3, 2)).unwrap(),
                    M::half(3)
                );
            })+};
        }

        assert_signed!(i8, i16, i32, i64, i128, isize);
        assert_unsigned!(u8, u16, u32, u64, u128, usize);
    }

    #[test]
    fn float_conversion_obeys_tolerance_and_storage_boundaries() {
        assert_eq!(
            J::try_from(0.5 + QUANTUM_NUMBER_FLOAT_TOLERANCE / 8.0).unwrap(),
            J::half(1)
        );
        assert!(J::try_from(0.5 + QUANTUM_NUMBER_FLOAT_TOLERANCE).is_err());
        assert_eq!(
            L::try_from(1.0 + QUANTUM_NUMBER_FLOAT_TOLERANCE / 2.0).unwrap(),
            L::int(1)
        );
        assert!(L::try_from(1.0 + 2.0 * QUANTUM_NUMBER_FLOAT_TOLERANCE).is_err());
        assert_eq!(M::try_from(-0.5_f32).unwrap(), M::half(-1));
        assert!(M::try_from(f32::INFINITY).is_err());
        assert!(J::try_from((u32::MAX as f64 + 1.0) / 2.0).is_err());
        assert!(L::try_from(u32::MAX as f64 + 1.0).is_err());
        assert!(M::try_from((i32::MAX as f64 + 1.0) / 2.0).is_err());
    }

    #[test]
    fn conversion_error_variants_and_messages_remain_compatible() {
        let j_negative = J::try_from(-1_i8).unwrap_err();
        assert!(matches!(j_negative, LadduPhysicsError::InvalidValue { .. }));
        assert_eq!(
            j_negative.to_string(),
            "Invalid value for angular momentum: expected nonnegative, got -1"
        );

        let j_overflow = J::try_from(i128::MAX).unwrap_err();
        assert!(matches!(
            j_overflow,
            LadduPhysicsError::NumericOverflow { .. }
        ));
        assert_eq!(
            j_overflow.to_string(),
            format!(
                "Numeric overflow while computing 2 * angular momentum for value {}",
                i128::MAX
            )
        );

        let l_storage = L::try_from(u32::MAX as u64 + 1).unwrap_err();
        assert!(matches!(l_storage, LadduPhysicsError::InvalidValue { .. }));
        assert_eq!(
            l_storage.to_string(),
            format!(
                "Invalid value for orbital angular momentum: expected nonnegative and representable as u32, got {}",
                u32::MAX as u64 + 1
            )
        );

        let m_non_finite = M::try_from(f64::NAN).unwrap_err();
        assert!(matches!(m_non_finite, LadduPhysicsError::Custom(_)));
        assert_eq!(m_non_finite.to_string(), "projection must be finite");

        let unsigned_ratio = J::try_from(Ratio::new(u128::MAX, 1)).unwrap_err();
        assert!(matches!(
            unsigned_ratio,
            LadduPhysicsError::InvalidValue { .. }
        ));
        assert_eq!(
            unsigned_ratio.to_string(),
            format!(
                "Invalid value for ratio numerator: expected representable as i128, got {}",
                u128::MAX
            )
        );
    }

    #[test]
    fn ratio_storage_validation_happens_after_division() {
        assert_eq!(
            quantum_from_ratio::<L>(u32::MAX as i128 + 1, 2).unwrap(),
            L::int((u32::MAX / 2) + 1)
        );
    }

    #[test]
    #[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
    fn zero_ratio_denominator_retains_the_existing_failure_mode() {
        let _ = quantum_from_ratio::<J>(1, 0);
    }
}
