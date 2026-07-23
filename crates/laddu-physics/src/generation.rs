//! Kinematic proposal primitives used by Monte Carlo generators.

use serde::{Deserialize, Serialize};
use std::{f64::consts::PI, sync::Arc};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    histogram::Histogram,
    quantum::ParticleProperties,
    vectors::{RealVec3, RealVec4},
};

/// Deterministic, portable random-number stream for generation proposals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalRng {
    state: u64,
}

impl ProposalRng {
    /// Construct a proposal stream from a reproducible seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next uniformly distributed 64-bit integer.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Draw a floating-point value strictly between zero and one.
    pub fn uniform(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64 + 0.5) * SCALE
    }

    fn isotropic_direction(&mut self) -> RealVec3 {
        let cos_theta = 2.0 * self.uniform() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * self.uniform();
        RealVec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
    }
}

#[derive(Clone, Copy, Debug)]
/// A named generated edge and its four-momentum.
pub struct NamedMomentum<'a> {
    /// Edge name.
    pub name: &'a str,
    /// Four-momentum in `(E, px, py, pz)` order.
    pub p4: RealVec4,
}

#[derive(Clone, Copy, Debug)]
/// A named generated edge and its proposed invariant mass.
pub struct NamedMass<'a> {
    /// Edge name.
    pub name: &'a str,
    /// Invariant mass.
    pub mass: f64,
}

#[derive(Clone, Debug)]
/// Kinematics and importance weight produced by a vertex proposal.
pub struct ProposalResult {
    /// Proposed outgoing four-momenta in edge order.
    pub outgoing: Vec<RealVec4>,
    /// The proposal correction, conventionally `dPhi / q`.
    pub weight: f64,
}

#[derive(Clone, Copy, Debug)]
/// Mass and importance weight drawn from a [`MassProposal`].
pub struct MassProposalResult {
    /// Proposed invariant mass.
    pub mass: f64,
    /// Inverse proposal-density correction.
    pub weight: f64,
}

/// Invariant-mass proposal for a generated edge.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MassProposal {
    /// Always use one invariant mass.
    Fixed {
        /// Fixed invariant mass.
        mass: f64,
    },
    /// Sample uniformly between the given bounds, clipped to kinematic support.
    Uniform {
        /// Lower proposal bound.
        low: f64,
        /// Upper proposal bound.
        high: f64,
    },
}

impl MassProposal {
    /// Construct a fixed-mass proposal.
    pub fn fixed(mass: f64) -> Self {
        Self::Fixed { mass }
    }

    /// Construct a uniform mass proposal.
    pub fn uniform(low: f64, high: f64) -> Self {
        Self::Uniform { low, high }
    }

    /// Draw a mass within the supplied kinematic interval.
    pub fn propose(
        &self,
        minimum: f64,
        maximum: f64,
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<MassProposalResult> {
        match *self {
            Self::Fixed { mass } => {
                if !mass.is_finite() || mass < minimum || mass > maximum {
                    return Err(LadduPhysicsError::invalid_value(
                        "fixed mass",
                        format!("a finite value in [{minimum}, {maximum}]"),
                        mass,
                    ));
                }
                Ok(MassProposalResult { mass, weight: 1.0 })
            }
            Self::Uniform { low, high } => {
                let (low, high) = uniform_mass_support(low, high, minimum, maximum)?;
                let width = high - low;
                Ok(MassProposalResult {
                    mass: low + rng.uniform() * width,
                    weight: width,
                })
            }
        }
    }

    /// Evaluate the proposal density when it is available.
    ///
    /// Custom proposals may retain the default `None`; density-aware generators
    /// use this hook only for optional importance adaptation.
    pub fn density(
        &self,
        minimum: f64,
        maximum: f64,
        mass: f64,
    ) -> LadduPhysicsResult<Option<f64>> {
        match *self {
            // A point mass has no ordinary continuous density.
            Self::Fixed { .. } => Ok(None),
            Self::Uniform { low, high } => {
                let (low, high) = uniform_mass_support(low, high, minimum, maximum)?;
                Ok(Some(if mass >= low && mass <= high {
                    (high - low).recip()
                } else {
                    0.0
                }))
            }
        }
    }
}

impl From<f64> for MassProposal {
    fn from(mass: f64) -> Self {
        Self::fixed(mass)
    }
}

impl From<std::ops::Range<f64>> for MassProposal {
    fn from(range: std::ops::Range<f64>) -> Self {
        Self::uniform(range.start, range.end)
    }
}

fn uniform_mass_support(
    proposal_low: f64,
    proposal_high: f64,
    minimum: f64,
    maximum: f64,
) -> LadduPhysicsResult<(f64, f64)> {
    if !proposal_low.is_finite() || !proposal_high.is_finite() || proposal_high <= proposal_low {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "uniform mass proposal requires finite low < high, got [{proposal_low}, {proposal_high}]"
        )));
    }
    let low = proposal_low.max(minimum);
    let high = proposal_high.min(maximum);
    if high <= low {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "uniform mass support [{proposal_low}, {proposal_high}] does not overlap the allowed interval [{minimum}, {maximum}]"
        )));
    }
    Ok((low, high))
}

/// Kinematic proposal attached to a channel vertex.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VertexProposal {
    /// Generate an isotropic one-to-two decay.
    #[default]
    TwoBodyDecay,
    /// Generate two-to-two scattering from a transfer distribution.
    TwoBodyScattering {
        /// Scattering proposal configuration.
        proposal: TwoBodyScattering,
    },
}

impl VertexProposal {
    /// Construct an isotropic two-body decay proposal.
    pub fn isotropic_decay() -> Self {
        Self::TwoBodyDecay
    }

    /// Construct a two-body scattering proposal distributed in momentum transfer.
    pub fn t_exchange(
        pairing: (impl Into<String>, impl Into<String>),
        distribution: TDistribution,
    ) -> Self {
        Self::TwoBodyScattering {
            proposal: TwoBodyScattering::t_exchange(pairing, distribution),
        }
    }

    /// Propose outgoing kinematics for a vertex.
    pub fn propose(
        &self,
        incoming: &[NamedMomentum<'_>],
        outgoing: &[NamedMass<'_>],
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<ProposalResult> {
        match self {
            Self::TwoBodyDecay => propose_two_body_decay(incoming, outgoing, rng),
            Self::TwoBodyScattering { proposal } => proposal.propose(incoming, outgoing, rng),
        }
    }
}

fn propose_two_body_decay(
    incoming: &[NamedMomentum<'_>],
    outgoing: &[NamedMass<'_>],
    rng: &mut ProposalRng,
) -> LadduPhysicsResult<ProposalResult> {
    if incoming.len() != 1 || outgoing.len() != 2 {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "isotropic decay requires one incoming and two outgoing edges, got {} incoming and {} outgoing",
            incoming.len(),
            outgoing.len()
        )));
    }
    let parent = incoming[0].p4;
    let mass = parent.m()?;
    let p = two_body_momentum(mass, outgoing[0].mass, outgoing[1].mass)?;
    let direction = rng.isotropic_direction();
    let first = on_shell(direction, p, outgoing[0].mass);
    let second = on_shell(-direction, p, outgoing[1].mass);
    let beta = parent.beta()?;
    Ok(ProposalResult {
        outgoing: vec![first.boost(&beta), second.boost(&beta)],
        weight: p / (4.0 * PI * mass),
    })
}

/// Density-adapted two-body decay used by the channel generator after a pilot run.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct AdaptiveTwoBodyDecay {
    counts: Arc<[f64]>,
    total: f64,
    defensive_fraction: f64,
}

impl AdaptiveTwoBodyDecay {
    /// Construct an angular proposal from nonnegative pilot-bin counts.
    pub fn new(counts: Arc<[f64]>, defensive_fraction: f64) -> LadduPhysicsResult<Self> {
        let total: f64 = counts.iter().sum();
        if counts.is_empty()
            || counts
                .iter()
                .any(|count| !count.is_finite() || *count < 0.0)
            || !total.is_finite()
            || total <= 0.0
            || !defensive_fraction.is_finite()
            || !(0.0..=1.0).contains(&defensive_fraction)
        {
            return Err(LadduPhysicsError::invalid_relation(
                "adaptive decay requires nonnegative finite counts with positive total and a defensive fraction in [0, 1]",
            ));
        }
        Ok(Self {
            counts,
            total,
            defensive_fraction,
        })
    }

    fn sample_costheta(&self, rng: &mut ProposalRng) -> (f64, f64) {
        let width = 2.0 / self.counts.len() as f64;
        let costheta = if rng.uniform() < self.defensive_fraction {
            2.0 * rng.uniform() - 1.0
        } else {
            let mut threshold = rng.uniform() * self.total;
            let mut selected = self.counts.len() - 1;
            for (bin, count) in self.counts.iter().enumerate() {
                if threshold <= *count {
                    selected = bin;
                    break;
                }
                threshold -= count;
            }
            -1.0 + (selected as f64 + rng.uniform()) * width
        };
        let bin = (((costheta + 1.0) / width) as usize).min(self.counts.len() - 1);
        let learned_density = self.counts[bin] / (self.total * width);
        let density =
            self.defensive_fraction * 0.5 + (1.0 - self.defensive_fraction) * learned_density;
        (costheta, density)
    }
}

impl AdaptiveTwoBodyDecay {
    /// Propose a two-body decay from the adapted angular density.
    pub fn propose(
        &self,
        incoming: &[NamedMomentum<'_>],
        outgoing: &[NamedMass<'_>],
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<ProposalResult> {
        if incoming.len() != 1 || outgoing.len() != 2 {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "adaptive decay requires one incoming and two outgoing edges, got {} incoming and {} outgoing",
                incoming.len(),
                outgoing.len()
            )));
        }
        let parent = incoming[0].p4;
        let mass = parent.m()?;
        let p = two_body_momentum(mass, outgoing[0].mass, outgoing[1].mass)?;
        let (cos_theta, density) = self.sample_costheta(rng);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * rng.uniform();
        let direction = RealVec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
        let first = on_shell(direction, p, outgoing[0].mass);
        let second = on_shell(-direction, p, outgoing[1].mass);
        let beta = parent.beta()?;
        Ok(ProposalResult {
            outgoing: vec![first.boost(&beta), second.boost(&beta)],
            weight: p / (4.0 * PI * mass) * 0.5 / density,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// A normalized component of a momentum-transfer proposal.
pub enum TComponent {
    /// Uniform density in `t`.
    Uniform,
    /// Density proportional to `exp(slope * t)`.
    Exponential {
        /// Exponential slope.
        slope: f64,
    },
    /// Pole-like density proportional to `(exchange_mass² - t)^(-power)`.
    Pole {
        /// Mass of the exchanged pole.
        exchange_mass: f64,
        /// Power of the pole denominator.
        power: f64,
    },
    /// Piecewise-constant density supplied by a histogram.
    Histogram {
        /// Histogram defining the piecewise density.
        histogram: Histogram,
    },
}

impl TComponent {
    fn sample(&self, low: f64, high: f64, u: f64) -> LadduPhysicsResult<f64> {
        match *self {
            Self::Uniform => Ok(low + u * (high - low)),
            Self::Exponential { slope } => {
                if !slope.is_finite() {
                    return Err(LadduPhysicsError::invalid_value(
                        "exponential t slope",
                        "finite",
                        slope,
                    ));
                }
                if slope.abs() < 1e-10 {
                    return Ok(low + u * (high - low));
                }
                let width = high - low;
                Ok(low + (1.0 + u * (slope * width).exp_m1()).ln() / slope)
            }
            Self::Pole {
                exchange_mass,
                power,
            } => {
                if !exchange_mass.is_finite()
                    || exchange_mass < 0.0
                    || !power.is_finite()
                    || power <= 0.0
                {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "pole mass and power must be finite, with nonnegative mass and positive power; got exchange_mass={exchange_mass}, power={power}"
                    )));
                }
                let a = exchange_mass * exchange_mass - high;
                let b = exchange_mass * exchange_mass - low;
                if a <= 0.0 {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "pole singularity at {} lies in the physical t interval [{low}, {high}]",
                        exchange_mass * exchange_mass
                    )));
                }
                let x = if (power - 1.0).abs() < 1e-10 {
                    a * (b / a).powf(u)
                } else {
                    let k = 1.0 - power;
                    (a.powf(k) + u * (b.powf(k) - a.powf(k))).powf(1.0 / k)
                };
                Ok(exchange_mass * exchange_mass - x)
            }
            Self::Histogram { ref histogram } => {
                let segments = Self::histogram_segments(histogram, low, high)?;
                let total: f64 = segments.iter().map(|(_, _, weight)| weight).sum();
                let mut threshold = u * total;
                for (segment_low, segment_high, weight) in &segments {
                    if threshold <= *weight {
                        return Ok(segment_low + threshold / weight * (segment_high - segment_low));
                    }
                    threshold -= weight;
                }
                Ok(segments.last().expect("segments are nonempty").1)
            }
        }
    }

    fn density(&self, low: f64, high: f64, t: f64) -> LadduPhysicsResult<f64> {
        match *self {
            Self::Uniform => Ok(1.0 / (high - low)),
            Self::Exponential { slope } => {
                if !slope.is_finite() {
                    return Err(LadduPhysicsError::invalid_value(
                        "exponential t slope",
                        "finite",
                        slope,
                    ));
                }
                if slope.abs() < 1e-10 {
                    return Ok(1.0 / (high - low));
                }
                Ok(slope * (slope * (t - low)).exp() / (slope * (high - low)).exp_m1())
            }
            Self::Pole {
                exchange_mass,
                power,
            } => {
                let a = exchange_mass * exchange_mass - high;
                let b = exchange_mass * exchange_mass - low;
                let x = exchange_mass * exchange_mass - t;
                if a <= 0.0 || power <= 0.0 {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "invalid pole component for t interval [{low}, {high}]: exchange_mass={exchange_mass}, power={power}"
                    )));
                }
                let norm = if (power - 1.0).abs() < 1e-10 {
                    (b / a).ln()
                } else {
                    (b.powf(1.0 - power) - a.powf(1.0 - power)) / (1.0 - power)
                };
                Ok(x.powf(-power) / norm)
            }
            Self::Histogram { ref histogram } => {
                let segments = Self::histogram_segments(histogram, low, high)?;
                let total: f64 = segments.iter().map(|(_, _, weight)| weight).sum();
                let Some((segment_low, segment_high, weight)) = segments
                    .iter()
                    .find(|(segment_low, segment_high, _)| t >= *segment_low && t <= *segment_high)
                else {
                    return Ok(0.0);
                };
                Ok(weight / ((segment_high - segment_low) * total))
            }
        }
    }

    fn histogram_segments(
        histogram: &Histogram,
        low: f64,
        high: f64,
    ) -> LadduPhysicsResult<Vec<(f64, f64, f64)>> {
        if histogram
            .counts()
            .iter()
            .any(|count| !count.is_finite() || *count < 0.0)
            || !histogram.total_weight().is_finite()
            || histogram.total_weight() <= 0.0
        {
            return Err(LadduPhysicsError::invalid_value(
                "histogram t-proposal counts",
                "finite and nonnegative with positive finite total weight",
                format!("{:?}", histogram.counts()),
            ));
        }
        let mut segments = Vec::new();
        for (index, &count) in histogram.counts().iter().enumerate() {
            let bin_low = histogram.bin_edges()[index];
            let bin_high = histogram.bin_edges()[index + 1];
            let segment_low = low.max(bin_low);
            let segment_high = high.min(bin_high);
            if segment_high > segment_low && count > 0.0 {
                let overlap_fraction = (segment_high - segment_low) / (bin_high - bin_low);
                segments.push((segment_low, segment_high, count * overlap_fraction));
            }
        }
        if segments.is_empty() {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "histogram support does not overlap the physical t interval [{low}, {high}]"
            )));
        }
        Ok(segments)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Source distribution for a generated scalar value.
pub enum ScalarSource {
    /// A deterministic value.
    Constant(f64),
    /// A uniform distribution on `[low, high)`.
    Uniform {
        /// Lower source bound.
        low: f64,
        /// Upper source bound.
        high: f64,
    },
    /// A piecewise-constant histogram distribution.
    Histogram(Histogram),
}

#[derive(Clone, Copy, Debug)]
/// Scalar draw and its inverse proposal-density correction.
pub struct ScalarProposalResult {
    /// Sampled scalar value.
    pub value: f64,
    /// The proposal correction `1 / q(value)`; constants use one.
    pub weight: f64,
}

impl ScalarSource {
    /// Construct a constant scalar source.
    pub fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Construct a uniform scalar source.
    pub fn uniform(low: f64, high: f64) -> Self {
        Self::Uniform { low, high }
    }

    /// Construct a histogram-backed scalar source.
    pub fn histogram(histogram: Histogram) -> Self {
        Self::Histogram(histogram)
    }

    /// Validate the source and return the smallest and largest values in its support.
    pub fn support(&self) -> LadduPhysicsResult<(f64, f64)> {
        match self {
            Self::Constant(value) if value.is_finite() => Ok((*value, *value)),
            Self::Constant(value) => Err(LadduPhysicsError::invalid_value(
                "constant scalar source",
                "finite",
                value,
            )),
            Self::Uniform { low, high } if low.is_finite() && high.is_finite() && high > low => {
                Ok((*low, *high))
            }
            Self::Uniform { low, high } => Err(LadduPhysicsError::invalid_relation(format!(
                "uniform scalar source requires finite low < high, got [{low}, {high}]"
            ))),
            Self::Histogram(histogram) => {
                if histogram
                    .counts()
                    .iter()
                    .any(|count| !count.is_finite() || *count < 0.0)
                    || !histogram.total_weight().is_finite()
                    || histogram.total_weight() <= 0.0
                {
                    return Err(LadduPhysicsError::invalid_value(
                        "histogram scalar-source counts",
                        "finite and nonnegative with positive finite total weight",
                        format!("{:?}", histogram.counts()),
                    ));
                }
                let first = histogram
                    .counts()
                    .iter()
                    .position(|count| *count > 0.0)
                    .expect("positive total weight implies a positive bin");
                let last = histogram
                    .counts()
                    .iter()
                    .rposition(|count| *count > 0.0)
                    .expect("positive total weight implies a positive bin");
                Ok((
                    histogram.bin_edges()[first],
                    histogram.bin_edges()[last + 1],
                ))
            }
        }
    }

    /// Draw a value and inverse-density weight from the source.
    pub fn sample(&self, rng: &mut ProposalRng) -> LadduPhysicsResult<ScalarProposalResult> {
        match self {
            Self::Constant(value) if value.is_finite() => Ok(ScalarProposalResult {
                value: *value,
                weight: 1.0,
            }),
            Self::Constant(value) => Err(LadduPhysicsError::invalid_value(
                "constant scalar source",
                "finite",
                value,
            )),
            Self::Uniform { low, high } if low.is_finite() && high.is_finite() && high > low => {
                Ok(ScalarProposalResult {
                    value: low + rng.uniform() * (high - low),
                    weight: high - low,
                })
            }
            Self::Uniform { low, high } => Err(LadduPhysicsError::invalid_relation(format!(
                "uniform scalar source requires finite low < high, got [{low}, {high}]"
            ))),
            Self::Histogram(histogram) => {
                let mut histogram_rng = fastrand::Rng::with_seed(rng.next_u64());
                let value = histogram.sample(&mut histogram_rng)?;
                let index = histogram.bin_index(value).ok_or_else(|| {
                    LadduPhysicsError::invalid_relation(
                        "sampled histogram value does not belong to an in-range bin",
                    )
                })?;
                let width = histogram.bin_edges()[index + 1] - histogram.bin_edges()[index];
                let probability_density =
                    histogram.counts()[index] / (histogram.total_weight() * width);
                Ok(ScalarProposalResult {
                    value,
                    weight: probability_density.recip(),
                })
            }
        }
    }
}

/// A four-momentum source attached to an initial channel edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InitialMomentum {
    /// Use a fixed four-momentum directly.
    P4(RealVec4),
    /// Use a fixed three-momentum and derive the energy from the particle mass.
    Momentum(RealVec3),
    /// Sample the energy and orient the momentum along a fixed direction.
    EnergyDirection {
        /// Energy source.
        energy: ScalarSource,
        /// Fixed direction of the initial momentum.
        direction: RealVec3,
    },
}

/// A sampled initial four-momentum and its inverse proposal density.
#[derive(Clone, Copy, Debug)]
pub struct InitialMomentumResult {
    /// Sampled on-shell four-momentum in `(E, px, py, pz)` order.
    pub p4: RealVec4,
    /// Inverse proposal-density correction.
    pub weight: f64,
}

impl InitialMomentum {
    /// Construct a fixed four-momentum source.
    pub fn p4(p4: RealVec4) -> Self {
        Self::P4(p4)
    }

    /// Construct a source from a fixed three-momentum and particle mass.
    pub fn momentum(momentum: RealVec3) -> Self {
        Self::Momentum(momentum)
    }

    /// Construct a fixed-energy source along a direction.
    pub fn energy_direction(energy: f64, direction: RealVec3) -> Self {
        Self::EnergyDirection {
            energy: ScalarSource::constant(energy),
            direction,
        }
    }

    /// Construct a sampled-energy source along a direction.
    pub fn energy_source_direction(energy: ScalarSource, direction: RealVec3) -> Self {
        Self::EnergyDirection { energy, direction }
    }

    /// Validate this source against an edge name and particle definition.
    pub fn validate(
        &self,
        edge: &str,
        properties: Option<&ParticleProperties>,
    ) -> LadduPhysicsResult<()> {
        match self {
            Self::P4(p4) => {
                let mass = particle_mass(edge, properties)?;
                if ![p4.px(), p4.py(), p4.pz(), p4.e()]
                    .into_iter()
                    .all(f64::is_finite)
                    || p4.e() < 0.0
                {
                    return Err(LadduPhysicsError::invalid_value(
                        format!("initial four-momentum for edge `{edge}`"),
                        "finite components and nonnegative energy",
                        p4,
                    ));
                }
                let tolerance = 1e-9 * (1.0 + mass * mass + p4.e() * p4.e());
                if (p4.m2() - mass * mass).abs() > tolerance {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "initial edge `{edge}` is off shell: p²={} but mass²={}",
                        p4.m2(),
                        mass * mass
                    )));
                }
            }
            Self::Momentum(momentum) => {
                particle_mass(edge, properties)?;
                if ![momentum.px(), momentum.py(), momentum.pz()]
                    .into_iter()
                    .all(f64::is_finite)
                {
                    return Err(LadduPhysicsError::invalid_value(
                        format!("initial momentum for edge `{edge}`"),
                        "finite components",
                        momentum,
                    ));
                }
            }
            Self::EnergyDirection { energy, direction } => {
                let mass = particle_mass(edge, properties)?;
                let (minimum, _) = energy.support()?;
                if minimum < mass {
                    return Err(LadduPhysicsError::invalid_value(
                        format!("energy support for initial edge `{edge}`"),
                        format!("entirely at or above its particle mass {mass}"),
                        minimum,
                    ));
                }
                direction.unit()?;
            }
        }
        Ok(())
    }

    /// Draw an initial four-momentum after validating its particle definition.
    pub fn sample(
        &self,
        edge: &str,
        properties: Option<&ParticleProperties>,
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<InitialMomentumResult> {
        self.validate(edge, properties)?;
        self.sample_prevalidated(particle_mass(edge, properties)?, rng)
    }

    /// Sample after channel validation has already established the source and
    /// particle-mass invariants.
    #[doc(hidden)]
    pub fn sample_prevalidated(
        &self,
        mass: f64,
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<InitialMomentumResult> {
        match self {
            Self::P4(p4) => Ok(InitialMomentumResult {
                p4: *p4,
                weight: 1.0,
            }),
            Self::Momentum(momentum) => Ok(InitialMomentumResult {
                p4: momentum.with_mass(mass),
                weight: 1.0,
            }),
            Self::EnergyDirection { energy, direction } => {
                let sampled = energy.sample(rng)?;
                if sampled.value < mass {
                    return Err(LadduPhysicsError::invalid_value(
                        "sampled initial-state energy",
                        format!("at or above the particle mass {mass}"),
                        sampled.value,
                    ));
                }
                let momentum =
                    direction.unit()? * (sampled.value * sampled.value - mass * mass).sqrt();
                Ok(InitialMomentumResult {
                    p4: momentum.with_energy(sampled.value),
                    weight: sampled.weight,
                })
            }
        }
    }
}

fn particle_mass(edge: &str, properties: Option<&ParticleProperties>) -> LadduPhysicsResult<f64> {
    properties
        .ok_or_else(|| {
            LadduPhysicsError::invalid_relation(format!(
                "initial edge `{edge}` has no particle properties"
            ))
        })?
        .mass()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Mixture distribution for Mandelstam `t`.
pub struct TDistribution {
    components: Vec<(f64, TComponent)>,
    #[serde(default)]
    t_min: Option<f64>,
    #[serde(default)]
    t_max: Option<f64>,
}

impl TDistribution {
    /// Construct a uniform distribution in `t`.
    pub fn uniform() -> Self {
        Self::mixture([(1.0, TComponent::Uniform)])
    }

    /// Construct an exponential distribution in `t`.
    pub fn exponential(slope: f64) -> Self {
        Self::mixture([(1.0, TComponent::Exponential { slope })])
    }

    /// Construct a pole-like distribution in `t`.
    pub fn pole(exchange_mass: f64, power: f64) -> Self {
        Self::mixture([(
            1.0,
            TComponent::Pole {
                exchange_mass,
                power,
            },
        )])
    }

    /// Construct a histogram-backed distribution in `t`.
    pub fn histogram(histogram: Histogram) -> Self {
        Self::mixture([(1.0, TComponent::Histogram { histogram })])
    }

    /// Construct a weighted mixture of transfer-density components.
    pub fn mixture(components: impl IntoIterator<Item = (f64, TComponent)>) -> Self {
        Self {
            components: components.into_iter().collect(),
            t_min: None,
            t_max: None,
        }
    }

    /// Restrict this proposal to the intersection of these limits and the
    /// event-by-event physical t interval.
    pub fn with_limits(
        mut self,
        t_min: Option<f64>,
        t_max: Option<f64>,
    ) -> LadduPhysicsResult<Self> {
        if t_min.is_some_and(|value| !value.is_finite()) {
            return Err(LadduPhysicsError::invalid_value(
                "t_min",
                "finite when specified",
                t_min.unwrap(),
            ));
        }
        if t_max.is_some_and(|value| !value.is_finite()) {
            return Err(LadduPhysicsError::invalid_value(
                "t_max",
                "finite when specified",
                t_max.unwrap(),
            ));
        }
        if let (Some(t_min), Some(t_max)) = (t_min, t_max)
            && t_max <= t_min
        {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "t limits require t_min < t_max, got [{t_min}, {t_max}]"
            )));
        }
        self.t_min = t_min;
        self.t_max = t_max;
        Ok(self)
    }

    fn normalization(&self) -> LadduPhysicsResult<f64> {
        if self.components.is_empty() {
            return Err(LadduPhysicsError::invalid_length(
                "t-distribution components",
                "at least one",
                0,
            ));
        }
        if self
            .components
            .iter()
            .any(|(weight, _)| !weight.is_finite() || *weight <= 0.0)
        {
            return Err(LadduPhysicsError::invalid_value(
                "t-distribution mixture weights",
                "finite and positive",
                format!(
                    "{:?}",
                    self.components
                        .iter()
                        .map(|(weight, _)| weight)
                        .collect::<Vec<_>>()
                ),
            ));
        }
        let sum: f64 = self.components.iter().map(|(weight, _)| weight).sum();
        Ok(sum)
    }

    fn sample(&self, low: f64, high: f64, rng: &mut ProposalRng) -> LadduPhysicsResult<(f64, f64)> {
        if !low.is_finite() || !high.is_finite() || high <= low {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "physical t interval must have finite bounds with low < high, got [{low}, {high}]"
            )));
        }
        let physical_low = low;
        let physical_high = high;
        let low = self.t_min.map_or(low, |t_min| low.max(t_min));
        let high = self.t_max.map_or(high, |t_max| high.min(t_max));
        if high <= low {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "configured t limits do not overlap the physical interval [{physical_low}, {physical_high}]"
            )));
        }
        let normalization = self.normalization()?;
        let choice = rng.uniform();
        let mut cumulative = 0.0;
        let mut selected = self.components.len() - 1;
        for (index, (weight, _)) in self.components.iter().enumerate() {
            cumulative += weight / normalization;
            if choice < cumulative {
                selected = index;
                break;
            }
        }
        let t = self.components[selected]
            .1
            .sample(low, high, rng.uniform())?;
        let mut density = 0.0;
        for (weight, component) in &self.components {
            density += weight / normalization * component.density(low, high, t)?;
        }
        if !density.is_finite() || density <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "t-proposal density",
                "finite and positive",
                density,
            ));
        }
        Ok((t, density))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Two-to-two scattering proposal based on a selected incoming/outgoing
/// momentum-transfer pairing.
pub struct TwoBodyScattering {
    incoming_edge: String,
    outgoing_edge: String,
    distribution: TDistribution,
}

impl TwoBodyScattering {
    /// Construct a `t`-exchange proposal for the named edge pairing.
    pub fn t_exchange(
        pairing: (impl Into<String>, impl Into<String>),
        distribution: TDistribution,
    ) -> Self {
        Self {
            incoming_edge: pairing.0.into(),
            outgoing_edge: pairing.1.into(),
            distribution,
        }
    }
}

impl From<TwoBodyScattering> for VertexProposal {
    fn from(proposal: TwoBodyScattering) -> Self {
        Self::TwoBodyScattering { proposal }
    }
}

impl TwoBodyScattering {
    /// Propose outgoing two-body scattering kinematics.
    pub fn propose(
        &self,
        incoming: &[NamedMomentum<'_>],
        outgoing: &[NamedMass<'_>],
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<ProposalResult> {
        if incoming.len() != 2 || outgoing.len() != 2 {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "two-body scattering requires two incoming and two outgoing edges, got {} incoming and {} outgoing",
                incoming.len(),
                outgoing.len()
            )));
        }
        let paired_in = incoming
            .iter()
            .position(|edge| edge.name == self.incoming_edge)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation(format!(
                    "unknown incoming t-pairing edge `{}`",
                    self.incoming_edge
                ))
            })?;
        let paired_out = outgoing
            .iter()
            .position(|edge| edge.name == self.outgoing_edge)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation(format!(
                    "unknown outgoing t-pairing edge `{}`",
                    self.outgoing_edge
                ))
            })?;
        let total = incoming[0].p4 + incoming[1].p4;
        let root_s = total.m()?;
        let beta = total.beta()?;
        let incoming_com = incoming[paired_in].p4.boost(&(-beta));
        // Invariant masses are best evaluated before the boost. In particular,
        // boosting a massless four-vector can leave a tiny negative m^2 from
        // floating-point cancellation.
        let m1 = incoming[paired_in].p4.m()?;
        let m2 = incoming[1 - paired_in].p4.m()?;
        let m3 = outgoing[paired_out].mass;
        let m4 = outgoing[1 - paired_out].mass;
        let p_in = two_body_momentum(root_s, m1, m2)?;
        let p_out = two_body_momentum(root_s, m3, m4)?;
        if p_in <= 0.0 {
            return Err(LadduPhysicsError::invalid_relation(
                "t exchange is undefined at the incoming threshold",
            ));
        }
        let e1 = (m1 * m1 + p_in * p_in).sqrt();
        let e3 = (m3 * m3 + p_out * p_out).sqrt();
        let center = m1 * m1 + m3 * m3 - 2.0 * e1 * e3;
        let span = 2.0 * p_in * p_out;
        let (t, q_t) = self
            .distribution
            .sample(center - span, center + span, rng)?;
        let cos_theta = ((t - center) / span).clamp(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * PI * rng.uniform();
        let z = incoming_com.vec3().unit()?;
        let seed = if z.z.abs() < 0.9 {
            RealVec3::new(0.0, 0.0, 1.0)
        } else {
            RealVec3::new(1.0, 0.0, 0.0)
        };
        let x = seed.cross(&z).unit()?;
        let y = z.cross(&x);
        let direction = z * cos_theta + x * (sin_theta * phi.cos()) + y * (sin_theta * phi.sin());
        let paired = on_shell(direction, p_out, m3).boost(&beta);
        let other = on_shell(-direction, p_out, m4).boost(&beta);
        let mut result = vec![RealVec4::new(0.0, 0.0, 0.0, 0.0); 2];
        result[paired_out] = paired;
        result[1 - paired_out] = other;
        Ok(ProposalResult {
            outgoing: result,
            weight: 1.0 / (16.0 * PI * root_s * p_in * q_t),
        })
    }
}

fn two_body_momentum(parent: f64, first: f64, second: f64) -> LadduPhysicsResult<f64> {
    if !parent.is_finite()
        || !first.is_finite()
        || !second.is_finite()
        || parent <= 0.0
        || first < 0.0
        || second < 0.0
    {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "two-body masses must be finite, with a positive parent and nonnegative daughters; got parent={parent}, first={first}, second={second}"
        )));
    }
    if parent < first + second {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "two-body threshold is closed: {parent} < {}",
            first + second
        )));
    }
    let lambda =
        (parent * parent - (first + second).powi(2)) * (parent * parent - (first - second).powi(2));
    Ok(lambda.max(0.0).sqrt() / (2.0 * parent))
}

fn on_shell(direction: RealVec3, momentum: f64, mass: f64) -> RealVec4 {
    (direction * momentum).with_mass(mass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_decay_conserves_momentum_and_mass() {
        let proposal = VertexProposal::isotropic_decay();
        let incoming = [NamedMomentum {
            name: "x",
            p4: RealVec4::new(2.0, 0.3, -0.2, 1.0),
        }];
        let outgoing = [
            NamedMass {
                name: "a",
                mass: 0.2,
            },
            NamedMass {
                name: "b",
                mass: 0.4,
            },
        ];
        let result = proposal
            .propose(&incoming, &outgoing, &mut ProposalRng::new(7))
            .unwrap();
        let sum = result.outgoing[0] + result.outgoing[1];
        for (a, b) in [sum.e, sum.px, sum.py, sum.pz]
            .into_iter()
            .zip([2.0, 0.3, -0.2, 1.0])
        {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((result.outgoing[0].m().unwrap() - 0.2).abs() < 1e-12);
        assert!((result.outgoing[1].m().unwrap() - 0.4).abs() < 1e-12);
        assert!(result.weight > 0.0);
    }

    #[test]
    fn t_mixture_samples_inside_physical_range() {
        let distribution = TDistribution::mixture([
            (1.0, TComponent::Uniform),
            (2.0, TComponent::Exponential { slope: 3.0 }),
            (
                1.0,
                TComponent::Pole {
                    exchange_mass: 1.0,
                    power: 2.0,
                },
            ),
        ]);
        let mut rng = ProposalRng::new(11);
        for _ in 0..100 {
            let (t, density) = distribution.sample(-2.0, -0.1, &mut rng).unwrap();
            assert!((-2.0..=-0.1).contains(&t));
            assert!(density.is_finite() && density > 0.0);
        }
    }

    #[test]
    fn t_distribution_limits_truncate_the_physical_interval() {
        let distribution = TDistribution::uniform()
            .with_limits(Some(-1.25), Some(-0.5))
            .unwrap();
        let mut rng = ProposalRng::new(13);
        for _ in 0..100 {
            let (t, density) = distribution.sample(-2.0, -0.1, &mut rng).unwrap();
            assert!((-1.25..=-0.5).contains(&t));
            assert!((density - 1.0 / 0.75).abs() < 1e-12);
        }
        assert!(
            TDistribution::uniform()
                .with_limits(Some(-0.5), Some(-1.0))
                .is_err()
        );
        assert!(
            distribution
                .sample(-3.0, -2.0, &mut ProposalRng::new(17))
                .is_err()
        );
    }

    #[test]
    fn t_exchange_conserves_momentum_and_is_on_shell() {
        let proposal =
            TwoBodyScattering::t_exchange(("beam", "x"), TDistribution::exponential(2.0));
        let incoming = [
            NamedMomentum {
                name: "beam",
                p4: RealVec4::new(1.5, 0.0, 0.0, 1.0),
            },
            NamedMomentum {
                name: "target",
                p4: RealVec4::new(1.5, 0.0, 0.0, -1.0),
            },
        ];
        let outgoing = [
            NamedMass {
                name: "x",
                mass: 0.5,
            },
            NamedMass {
                name: "r",
                mass: 0.7,
            },
        ];
        let result = proposal
            .propose(&incoming, &outgoing, &mut ProposalRng::new(19))
            .unwrap();
        let before = incoming[0].p4 + incoming[1].p4;
        let after = result.outgoing[0] + result.outgoing[1];
        assert!((before.e - after.e).abs() < 1e-12);
        assert!((before.px - after.px).abs() < 1e-12);
        assert!((before.py - after.py).abs() < 1e-12);
        assert!((before.pz - after.pz).abs() < 1e-12);
        assert!((result.outgoing[0].m().unwrap() - 0.5).abs() < 1e-12);
        assert!((result.outgoing[1].m().unwrap() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn adaptive_decay_preserves_the_phase_space_integral() {
        let incoming = [NamedMomentum {
            name: "parent",
            p4: RealVec4::new(2.0, 0.0, 0.0, 0.0),
        }];
        let outgoing = [
            NamedMass {
                name: "a",
                mass: 0.2,
            },
            NamedMass {
                name: "b",
                mass: 0.4,
            },
        ];
        let adaptive =
            AdaptiveTwoBodyDecay::new(Arc::from([1.0, 2.0, 8.0, 20.0, 8.0, 2.0, 1.0]), 0.2)
                .unwrap();
        let baseline = VertexProposal::isotropic_decay()
            .propose(&incoming, &outgoing, &mut ProposalRng::new(1))
            .unwrap()
            .weight;
        let mut rng = ProposalRng::new(2);
        let samples = 100_000;
        let mean = (0..samples)
            .map(|_| {
                adaptive
                    .propose(&incoming, &outgoing, &mut rng)
                    .unwrap()
                    .weight
            })
            .sum::<f64>()
            / samples as f64;
        assert!((mean / baseline - 1.0).abs() < 0.01);
    }

    #[test]
    fn proposal_failures_use_structured_physics_errors() {
        let empty = TDistribution::mixture([]);
        assert!(matches!(
            empty.normalization(),
            Err(LadduPhysicsError::InvalidLength { .. })
        ));

        assert!(matches!(
            MassProposal::fixed(2.0).propose(0.0, 1.0, &mut ProposalRng::new(0)),
            Err(LadduPhysicsError::InvalidValue { .. })
        ));

        assert!(matches!(
            VertexProposal::isotropic_decay().propose(&[], &[], &mut ProposalRng::new(0)),
            Err(LadduPhysicsError::InvalidRelation { .. })
        ));
    }

    #[test]
    fn histogram_t_component_truncates_to_the_physical_interval() {
        let histogram = Histogram::new(vec![1.0, 3.0], vec![-2.0, -1.0, 0.0]).unwrap();
        let distribution = TDistribution::histogram(histogram);
        let mut rng = ProposalRng::new(31);
        for _ in 0..100 {
            let (t, density) = distribution.sample(-1.5, -0.5, &mut rng).unwrap();
            assert!((-1.5..=-0.5).contains(&t));
            assert!(density.is_finite() && density > 0.0);
        }
    }

    #[test]
    fn scalar_sources_return_values_and_proposal_corrections() {
        let mut rng = ProposalRng::new(37);
        let constant = ScalarSource::constant(3.0).sample(&mut rng).unwrap();
        assert_eq!(constant.value, 3.0);
        assert_eq!(constant.weight, 1.0);

        let uniform = ScalarSource::uniform(-2.0, 4.0).sample(&mut rng).unwrap();
        assert!((-2.0..4.0).contains(&uniform.value));
        assert_eq!(uniform.weight, 6.0);

        let histogram = Histogram::new(vec![1.0, 2.0], vec![0.0, 1.0, 3.0]).unwrap();
        let sampled = ScalarSource::histogram(histogram).sample(&mut rng).unwrap();
        assert!((0.0..3.0).contains(&sampled.value));
        assert!(sampled.weight.is_finite() && sampled.weight > 0.0);
    }

    #[test]
    fn uniform_mass_truncates_to_the_allowed_interval() {
        let proposal = MassProposal::uniform(1.0, 2.0);
        let mut rng = ProposalRng::new(41);
        for _ in 0..100 {
            let result = proposal.propose(1.25, 1.75, &mut rng).unwrap();
            assert!((1.25..1.75).contains(&result.mass));
            assert_eq!(result.weight, 0.5);
        }
    }
}
