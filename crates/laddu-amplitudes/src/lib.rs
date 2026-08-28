//! Symbolic resonance and coupled-channel amplitude building blocks.
//!
//! Coupled-channel constructors take caller-supplied masses, couplings, and
//! backgrounds. Named parameterizations and their fitted data belong in downstream
//! packages rather than this experiment-neutral crate.

mod breit_wigner;
mod error;
mod kmatrix;

pub use breit_wigner::*;
pub use error::{AmplitudeError, AmplitudeResult, KMatrixError, KMatrixResult};
pub use kmatrix::*;
