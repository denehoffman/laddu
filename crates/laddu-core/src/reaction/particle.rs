use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::{reaction::Endpoint, ParticleProperties, ScalarDistribution};

/// How generation should obtain a particle mass.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum MassSampler {
    /// Use the mass encoded in [`ParticleProperties`].
    #[default]
    FromProperties,
    /// Sample the mass from a distribution.
    Sampled(ScalarDistribution),
}

/// How generation should obtain an initial particle momentum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MomentumSource {
    /// Generate a particle at rest using the mass from [`ParticleProperties`].
    AtRest,
    /// Generate a particle from a sampled lab-frame energy.
    FromEnergy(ScalarDistribution),
}

/// Generation annotations attached to a particle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticleGeneration {
    #[serde(default)]
    mass: MassSampler,
    #[serde(default)]
    momentum: Option<MomentumSource>,
}

impl Default for ParticleGeneration {
    fn default() -> Self {
        Self {
            mass: MassSampler::FromProperties,
            momentum: None,
        }
    }
}

impl ParticleGeneration {
    /// Return the mass sampler for this particle.
    pub fn mass(&self) -> &MassSampler {
        &self.mass
    }

    /// Return the momentum source for this particle, if one is configured.
    pub fn momentum(&self) -> Option<&MomentumSource> {
        self.momentum.as_ref()
    }

    /// Use the given mass sampler for this particle.
    pub fn with_mass_sampler(mut self, sampler: MassSampler) -> Self {
        self.mass = sampler;
        self
    }

    /// Use a momentum source for this particle.
    pub fn with_momentum(mut self, momentum: MomentumSource) -> Self {
        self.momentum = Some(momentum);
        self
    }
}

/// How a particle four-momentum is obtained when evaluating an event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParticleSource {
    /// Infer composites from daughters and otherwise read a matching dataset column.
    Inferred,
    /// Always read a matching dataset column.
    Stored,
    /// Solve this external particle from four-momentum conservation.
    Missing,
}

/// A kinematic particle or composite system used to define a reaction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    pub(crate) label: String,
    pub(crate) source: ParticleSource,
    pub(crate) from: Endpoint,
    pub(crate) to: Endpoint,
    pub(crate) properties: ParticleProperties,
    #[serde(default)]
    pub(crate) generation: ParticleGeneration,
}

impl Particle {
    pub(crate) fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            source: ParticleSource::Inferred,
            from: Endpoint::ExternalIn,
            to: Endpoint::ExternalOut,
            properties: ParticleProperties::unknown(),
            generation: ParticleGeneration::default(),
        }
    }

    /// Return the particle label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return the particle source.
    pub fn source(&self) -> &ParticleSource {
        &self.source
    }

    /// Return the upstream endpoint.
    pub fn from(&self) -> Endpoint {
        self.from
    }

    /// Return the downstream endpoint.
    pub fn to(&self) -> Endpoint {
        self.to
    }

    /// Return the particle properties.
    pub fn properties(&self) -> &ParticleProperties {
        &self.properties
    }

    /// Return the generation annotations.
    pub fn generation(&self) -> &ParticleGeneration {
        &self.generation
    }

    /// Set the generation annotations.
    pub fn with_generation(mut self, generation: ParticleGeneration) -> Self {
        self.generation = generation;
        self
    }

    pub(crate) fn is_missing(&self) -> bool {
        matches!(self.source, ParticleSource::Missing)
    }
}

impl Display for Particle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}
