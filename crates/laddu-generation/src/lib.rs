pub mod distributions;
pub mod topology;

/// Channel generation annotation constructors.
pub mod gen {
    use laddu_core::{ChannelMassGenerator, ChannelMomentumGenerator, ChannelVertexGenerator};

    /// Generate a beam-like particle with fixed lab-frame energy.
    pub fn energy(value: f64) -> ChannelMomentumGenerator {
        ChannelMomentumGenerator::FixedEnergy { energy: value }
    }

    /// Generate a particle at rest using its species mass.
    pub fn rest() -> ChannelMomentumGenerator {
        ChannelMomentumGenerator::Rest
    }

    /// Generate a uniformly sampled particle mass.
    pub fn uniform(low: f64, high: f64) -> ChannelMassGenerator {
        ChannelMassGenerator::Uniform { low, high }
    }

    /// Generate a fixed particle mass.
    pub fn fixed_mass(value: f64) -> ChannelMassGenerator {
        ChannelMassGenerator::Fixed { mass: value }
    }

    /// Generate a two-to-two vertex using an exponential Mandelstam-t distribution.
    pub fn t_exponential(slope: f64) -> ChannelVertexGenerator {
        ChannelVertexGenerator::TExponential { slope }
    }
}

pub use distributions::{
    Distribution, HistogramSampler, LadduGenRngExt, MandelstamTDistribution, SimpleDistribution,
};
pub use topology::{
    BatchIntensity, CompositeGenerator, EventGenerator, ExpressionIntensity, GeneratedBatch,
    GeneratedEventLayout, GeneratedParticle, GeneratedParticleLayout, GeneratedReaction,
    GeneratedReactionTopology, GeneratedStorage, GeneratedTwoToTwoReaction, GeneratedVertexKind,
    GeneratedVertexLayout, InitialGenerator, ParticleSpecies, Reconstruction, RejectionEnvelope,
    RejectionSampleIter, RejectionSampler, RejectionSamplingDiagnostics, RejectionSamplingOptions,
    StableGenerator,
};
