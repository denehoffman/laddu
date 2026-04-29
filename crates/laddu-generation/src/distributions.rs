use std::fmt::Debug;

use fastrand::Rng;
use fastrand_contrib::RngExt;
use laddu_core::{math::Histogram, Vec3, Vec4};

// TODO: move this to laddu and make it part of Histogram
pub struct HistogramSampler {
    pub(crate) hist: Histogram,
    cdf: Vec<f64>,
    total: f64,
}

// TODO: impl Clone/Debug for Histogram!
impl Clone for HistogramSampler {
    fn clone(&self) -> Self {
        Self {
            hist: Histogram {
                counts: self.hist.counts.clone(),
                bin_edges: self.hist.bin_edges.clone(),
            },
            cdf: self.cdf.clone(),
            total: self.total,
        }
    }
}

impl Debug for HistogramSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistogramSampler")
            .field("hist counts", &self.hist.counts)
            .field("hist bin_edges", &self.hist.bin_edges)
            .field("cdf", &self.cdf)
            .field("total", &self.total)
            .finish()
    }
}

impl HistogramSampler {
    pub fn new(hist: Histogram) -> Self {
        debug_assert!(
            hist.bin_edges.len() == hist.counts.len() + 1,
            "# bin edges = {}, # counts = {}",
            hist.bin_edges.len(),
            hist.counts.len()
        );
        let mut cdf = Vec::with_capacity(hist.counts.len());
        let mut total = 0.0;

        for &count in &hist.counts {
            debug_assert!(count >= 0.0, "Count cannot be negative!\nCount: {}", count);
            debug_assert!(count.is_finite(), "Count must be finite!\nCount: {}", count);
            total += count;
            cdf.push(total);
        }
        debug_assert!(
            total > 0.0,
            "Total must be greater than zero!\nTotal: {}",
            total
        );
        Self { hist, cdf, total }
    }
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let r = rng.f64() * self.total;
        let bin = self.cdf.partition_point(|&x| x <= r);
        let lo = self.hist.bin_edges[bin];
        let hi = self.hist.bin_edges[bin + 1];
        lo + rng.f64() * (hi - lo)
    }
}

#[derive(Clone, Debug)]
pub enum SimpleDistribution {
    Fixed(f64),
    Uniform { min: f64, max: f64 },
    Histogram(HistogramSampler),
}
impl SimpleDistribution {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            Self::Fixed(v) => *v,
            Self::Uniform { min, max } => rng.uniform(*min, *max),
            Self::Histogram(sampler) => sampler.sample(rng),
        }
    }
}

#[derive(Clone, Debug)]
pub enum MandelstamTDistribution {
    Exponential { slope: f64 },
    Histogram(HistogramSampler),
}
impl MandelstamTDistribution {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            Self::Exponential { slope } => rng.exponential(*slope),
            Self::Histogram(sampler) => sampler.sample(rng),
        }
    }
}

pub enum Distribution {
    Fixed(f64),
    Uniform { min: f64, max: f64 },
    Normal { mu: f64, sigma: f64 },
    Exponential { slope: f64 },
    Histogram(HistogramSampler),
}
impl Distribution {
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            Self::Fixed(v) => *v,
            Self::Uniform { min, max } => rng.uniform(*min, *max),
            Self::Normal { mu, sigma } => rng.normal(*mu, *sigma),
            Self::Exponential { slope } => rng.exponential(*slope),
            Self::Histogram(hist) => hist.sample(rng),
        }
    }
}

pub trait LadduGenRngExt {
    fn uniform(&mut self, min: f64, max: f64) -> f64;
    fn normal(&mut self, mu: f64, sigma: f64) -> f64;
    fn exponential(&mut self, slope: f64) -> f64;
    fn p4(&mut self, mass: f64, energy: f64, direction: Vec3) -> Vec4;
}

impl LadduGenRngExt for Rng {
    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        self.f64_range(min..=max)
    }

    fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        self.f64_normal_approx(mu, sigma)
    }

    fn exponential(&mut self, slope: f64) -> f64 {
        -(-self.f64()).ln_1p() / slope
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
