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
    BlattWeisskopf, ClebschGordan, DecayAmplitudeExt, PhotonHelicity, PhotonPolarization,
    PhotonSDME, PolPhase, Wigner3j, WignerD, Ylm, Zlm,
};

/// Resonance line shapes and related factors.
pub mod resonance;
pub use resonance::{BreitWigner, BreitWignerNonRelativistic, Flatte, PhaseSpaceFactor, Voigt};
