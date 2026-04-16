//! # laddu-amplitudes
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::new_ret_no_self)] // Amplitudes should return Expressions when constructed

/// The Breit-Wigner amplitude.
pub mod breit_wigner;
pub use breit_wigner::{BreitWigner, BreitWignerNonRelativistic};

/// The Flatte amplitude.
pub mod flatte;
pub use flatte::Flatte;

/// The Voigt amplitude.
pub mod voigt;
pub use voigt::Voigt;

mod semantic_key;

/// Common amplitudes (like a scalar value which just contains a single free parameter).
pub mod common;
pub use common::{
    ComplexScalar, PolarComplexScalar, Scalar, VariableExpressionExt, VariableScalar,
};

/// Amplitudes related to the K-Matrix formalism.
pub mod kmatrix;

/// Lookup-table amplitudes.
pub mod lookup_table;
pub use lookup_table::{LookupAxis, LookupBoundaryMode, LookupInterpolation, LookupTable};

/// A spherical harmonic amplitude.
pub mod ylm;
pub use ylm::Ylm;

/// A polarized spherical harmonic amplitude.
pub mod zlm;
pub use zlm::Zlm;

/// A phase space factor for `$a+b\to c+d$` with `$c\to 1+2$`.
pub mod phase_space;
pub use phase_space::PhaseSpaceFactor;

/// Data structures for sequential-decay expression builders.
pub mod sequential;
