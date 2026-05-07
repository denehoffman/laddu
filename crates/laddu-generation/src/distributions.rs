use fastrand::Rng;
use fastrand_contrib::RngExt;
use laddu_core::{math::Histogram, LadduResult, Vec3, Vec4};

/// Sampler for drawing values from a weighted histogram.
#[derive(Clone, Debug)]
pub struct HistogramSampler {
    pub(crate) hist: Histogram,
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

    /// Sample a value uniformly within a histogram bin selected by bin weight.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        let r = rng.f64() * self.total;
        let bin = self.cdf.partition_point(|&x| x <= r);
        let lo = self.hist.bin_edges()[bin];
        let hi = self.hist.bin_edges()[bin + 1];
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

#[derive(Clone, Debug)]
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
