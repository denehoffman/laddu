//! # laddu-amplitudes
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::new_ret_no_self)] // Amplitudes should return Expressions when constructed

/// Scalar-valued amplitude components.
pub mod scalar;
pub use scalar::{
    ComplexScalar, PolarComplexScalar, Scalar, VariableExpressionExt, VariableScalar,
};

/// Amplitudes related to the K-Matrix formalism.
pub mod kmatrix;

/// Lookup-table amplitudes.
pub mod lookup;
pub use lookup::{LookupAxis, LookupBoundaryMode, LookupInterpolation, LookupTable};

/// Angular, barrier, and density-matrix factor amplitudes.
pub mod angular;
pub use angular::{
    BlattWeisskopf, PhotonHelicity, PhotonPolarization, PhotonSDME, PolPhase, WignerD, Ylm, Zlm,
};

/// Resonance line shapes and related factors.
pub mod resonance;
pub use resonance::{BreitWigner, BreitWignerNonRelativistic, Flatte, PhaseSpaceFactor, Voigt};

#[cfg(test)]
mod test_utils {
    use laddu_core::{reaction::Channel, variables::Mass, Mandelstam, MandelstamChannel};

    pub(crate) fn channel() -> Channel {
        let mut channel = Channel::new();
        channel
            .create_production("production", ["beam", "target"], ["kk", "proton"])
            .unwrap();
        channel
            .create_decay("kk_decay", "kk", ["kshort1", "kshort2"])
            .unwrap();
        channel.edit_particle("beam").unwrap().stored();
        channel.edit_particle("target").unwrap().missing().unwrap();
        channel.edit_particle("kshort1").unwrap().stored();
        channel.edit_particle("kshort2").unwrap().stored();
        channel.edit_particle("proton").unwrap().stored();
        channel
    }

    pub(crate) fn mass(particle: &str) -> Mass {
        channel().mass(particle).unwrap()
    }

    pub(crate) fn mandelstam(channel: MandelstamChannel) -> Mandelstam {
        self::channel().mandelstam("production", channel).unwrap()
    }
}
