//! Event generation for [`laddu_core::Channel`] topologies.
//!
//! The current generator supports exactly one generated two-to-two production vertex with two
//! incoming particles and downstream chains of one-to-two decays. Particle masses come from
//! [`laddu_core::ParticleProperties`] unless the channel annotates a particle with an explicit
//! mass sampler.
//!
//! ```rust
//! use laddu_core::{Channel, ParticleProperties};
//! use laddu_generation::{gen, DatasetSink, EventGenerator, GenerationMode, GenerationOptions};
//!
//! let mut channel = Channel::new();
//! channel
//!     .create_production("production", ["beam", "target"], ["rho", "recoil"])?
//!     .generate(gen::t_exponential(0.1));
//! channel.create_decay("rho_decay", "rho", ["pi+", "pi-"])?;
//! channel
//!     .edit_particle("beam")?
//!     .properties(ParticleProperties::unknown().with_mass(0.0))
//!     .momentum(gen::energy(8.0));
//! channel
//!     .edit_particle("target")?
//!     .properties(ParticleProperties::unknown().with_mass(0.938272))
//!     .momentum(gen::rest());
//! channel
//!     .edit_particle("rho")?
//!     .mass_sampler(gen::uniform_mass(0.6, 0.9));
//! for label in ["pi+", "pi-"] {
//!     channel
//!         .edit_particle(label)?
//!         .properties(ParticleProperties::unknown().with_mass(0.13957));
//! }
//! channel
//!     .edit_particle("recoil")?
//!     .properties(ParticleProperties::unknown().with_mass(0.938272));
//!
//! let dataset = EventGenerator::from_channel(&channel)?
//!     .with_seed(12345)
//!     .generate(10, DatasetSink::new(), GenerationMode::Raw, GenerationOptions::default())?
//!     .output;
//! # Ok::<_, laddu_core::LadduError>(())
//! ```

pub mod generator;
pub mod plan;
pub mod sink;

/// Channel generation annotation constructors.
pub mod gen {
    use laddu_core::{
        math::Histogram, MassSampler, MomentumSource, ScalarDistribution, VertexGenerator,
    };

    /// Generate a beam-like particle with fixed lab-frame energy.
    pub fn energy(value: f64) -> MomentumSource {
        MomentumSource::FromEnergy(ScalarDistribution::Fixed(value))
    }

    /// Generate a beam-like particle with uniformly sampled lab-frame energy.
    pub fn uniform_energy(min: f64, max: f64) -> MomentumSource {
        MomentumSource::FromEnergy(ScalarDistribution::Uniform { min, max })
    }

    /// Generate a beam-like particle with histogram-sampled lab-frame energy.
    pub fn histogram_energy(histogram: Histogram) -> laddu_core::LadduResult<MomentumSource> {
        Ok(MomentumSource::FromEnergy(ScalarDistribution::Histogram(
            laddu_core::HistogramSampler::new(histogram)?,
        )))
    }

    /// Generate a particle at rest using its ParticleProperties mass.
    pub fn rest() -> MomentumSource {
        MomentumSource::AtRest
    }

    /// Generate a uniformly sampled particle mass.
    pub fn uniform_mass(min: f64, max: f64) -> MassSampler {
        MassSampler::Sampled(ScalarDistribution::Uniform { min, max })
    }

    /// Generate a histogram-sampled particle mass.
    pub fn histogram_mass(histogram: Histogram) -> laddu_core::LadduResult<MassSampler> {
        Ok(MassSampler::Sampled(ScalarDistribution::Histogram(
            laddu_core::HistogramSampler::new(histogram)?,
        )))
    }

    /// Generate a two-to-two vertex using an exponential Mandelstam-t distribution.
    pub fn t_exponential(slope: f64) -> VertexGenerator {
        VertexGenerator::TwoToTwo {
            t: ScalarDistribution::Exponential { slope },
        }
    }

    /// Generate a two-to-two vertex using a histogram-sampled Mandelstam-t distribution.
    pub fn t_histogram(histogram: Histogram) -> laddu_core::LadduResult<VertexGenerator> {
        Ok(VertexGenerator::TwoToTwo {
            t: ScalarDistribution::Histogram(laddu_core::HistogramSampler::new(histogram)?),
        })
    }
}

pub use generator::{EventGenerator, GeneratedEvent};
pub use plan::{
    DecayParticlePlan, DecayPlan, GenerationPlan, InitialParticlePlan, PlannedMass, ProductionPlan,
};
pub use sink::{
    CallbackSink, DatasetSink, Envelope, EnvelopeStats, EnvelopeViolationPolicy,
    GeneratedBatchView, GeneratedLayout, GeneratedParticleInfo, GeneratedParticleRole,
    GeneratedRecord, GeneratedSink, GenerationMode, GenerationModeKind, GenerationOptions,
    GenerationOutput, GenerationResult, GenerationStats, NullSink, SinkMpiSupport,
};
