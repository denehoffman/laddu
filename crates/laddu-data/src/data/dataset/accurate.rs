//! Numerically accurate accumulators used by parallel reductions.

use accurate::{sum::Sum2, traits::*};
use num::complex::Complex64;

/// Compensated accumulator for real values.
#[derive(Clone)]
pub struct AccurateF64 {
    sum: Sum2<f64>,
}

impl AccurateF64 {
    /// Creates a zero accumulator.
    pub fn zero() -> Self {
        Self { sum: Sum2::zero() }
    }

    /// Adds one value.
    pub fn push(&mut self, value: f64) {
        let sum = std::mem::replace(&mut self.sum, Sum2::zero());
        self.sum = sum + value;
    }

    /// Merges another accumulator.
    pub fn merge(&mut self, other: Self) {
        self.push(other.finish());
    }

    /// Returns the accumulated sum.
    pub fn finish(self) -> f64 {
        self.sum.sum()
    }
}

/// Pair of compensated accumulators for complex values.
#[derive(Clone)]
pub struct AccurateComplex64 {
    re: AccurateF64,
    im: AccurateF64,
}

impl AccurateComplex64 {
    /// Creates a zero accumulator.
    pub fn zero() -> Self {
        Self {
            re: AccurateF64::zero(),
            im: AccurateF64::zero(),
        }
    }

    /// Adds one complex value.
    pub fn push(&mut self, value: Complex64) {
        self.re.push(value.re);
        self.im.push(value.im);
    }

    /// Merges another accumulator.
    pub fn merge(&mut self, other: Self) {
        self.re.merge(other.re);
        self.im.merge(other.im);
    }

    /// Returns the accumulated complex sum.
    pub fn finish(self) -> Complex64 {
        Complex64::new(self.re.finish(), self.im.finish())
    }
}
