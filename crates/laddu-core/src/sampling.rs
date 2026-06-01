//! Reusable random sampling primitives.

use fastrand::Rng;
use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::{
    math::Histogram,
    vectors::{Vec3, Vec4},
    LadduResult,
};

/// Sampler for drawing values from a weighted histogram.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistogramSampler {
    hist: Histogram,
    cdf: Vec<f64>,
    total: f64,
}

impl HistogramSampler {
    /// Construct a histogram sampler.
    pub fn new(hist: Histogram) -> LadduResult<Self> {
        hist.validate()?;
        hist.validate_positive_counts()?;
        let mut cdf = Vec::with_capacity(hist.counts().len());
        let mut total = 0.0;

        for &count in hist.counts() {
            total += count;
            cdf.push(total);
        }
        Ok(Self { hist, cdf, total })
    }

    /// Return the histogram backing this sampler.
    pub fn histogram(&self) -> &Histogram {
        &self.hist
    }

    /// Sample a value uniformly within a histogram bin selected by bin weight.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let r = rng.f64() * self.total;
        let bin = self.cdf.partition_point(|&x| x <= r);
        let lo = self.hist.bin_edges()[bin];
        let hi = self.hist.bin_edges()[bin + 1];
        lo + rng.f64() * (hi - lo)
    }
}

/// A scalar distribution that can be sampled by generators and user code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScalarDistribution {
    /// Always return the same value.
    Fixed(f64),
    /// Sample uniformly from the closed interval `[min, max]`.
    Uniform {
        /// Lower bound.
        min: f64,
        /// Upper bound.
        max: f64,
    },
    /// Sample from an approximate normal distribution.
    Normal {
        /// Mean value.
        mu: f64,
        /// Standard deviation.
        sigma: f64,
    },
    /// Sample from an exponential distribution with the given slope.
    Exponential {
        /// Exponential slope.
        slope: f64,
    },
    /// Sample from a weighted histogram.
    Histogram(HistogramSampler),
}

impl ScalarDistribution {
    /// Sample from this distribution.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            Self::Fixed(value) => *value,
            Self::Uniform { min, max } => rng.uniform(*min, *max),
            Self::Normal { mu, sigma } => rng.normal(*mu, *sigma),
            Self::Exponential { slope } => rng.exponential(*slope),
            Self::Histogram(sampler) => sampler.sample(rng),
        }
    }

    /// Sample from this distribution until the value falls inside the open interval.
    pub fn sample_open_interval(&self, rng: &mut Rng, range: (f64, f64)) -> f64 {
        let mut result = self.sample(rng);
        while result <= range.0 || result >= range.1 {
            result = self.sample(rng);
        }
        result
    }

    /// Sample an exponential distribution truncated to the open interval.
    pub fn sample_negative_exponential(
        rng: &mut Rng,
        slope: f64,
        range: Option<(f64, f64)>,
    ) -> f64 {
        if let Some(range) = range {
            let mut result = -rng.truncated_exponential(slope, range);
            while result <= range.0 || result >= range.1 {
                result = -rng.truncated_exponential(slope, range);
            }
            result
        } else {
            -rng.exponential(slope)
        }
    }
}

/// Extra helpers for random generation.
pub trait LadduRngExt {
    /// Sample uniformly from the closed interval `[min, max]`.
    fn uniform(&mut self, min: f64, max: f64) -> f64;
    /// Sample from an approximate normal distribution.
    fn normal(&mut self, mu: f64, sigma: f64) -> f64;
    /// Sample from an exponential distribution with the given slope.
    fn exponential(&mut self, slope: f64) -> f64;
    /// Sample from an exponential distribution truncated to an interval.
    fn truncated_exponential(&mut self, slope: f64, range: (f64, f64)) -> f64;
    /// Build a four-vector from mass, energy, and direction.
    fn p4(&mut self, mass: f64, energy: f64, direction: Vec3) -> Vec4;
}

impl LadduRngExt for Rng {
    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        self.f64_range(min..=max)
    }

    fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        self.f64_normal_approx(mu, sigma)
    }

    fn exponential(&mut self, slope: f64) -> f64 {
        -(-self.f64()).ln_1p() / slope
    }

    fn truncated_exponential(&mut self, slope: f64, range: (f64, f64)) -> f64 {
        -(1. / slope) * (1.0 - self.f64() * (1.0 - (-slope * (range.1 - range.0)).exp())).ln()
    }

    fn p4(&mut self, mass: f64, energy: f64, direction: Vec3) -> Vec4 {
        debug_assert!(
            energy >= mass,
            "Mass cannot be greater than energy!\nEnergy: {}\nMass: {}",
            energy,
            mass
        );
        let momentum = ((energy - mass) * (energy + mass)).max(0.0).sqrt();
        (momentum * direction).with_mass(mass)
    }
}
