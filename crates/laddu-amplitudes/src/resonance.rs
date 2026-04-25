//! Resonance line shapes and related factors.

mod breit_wigner;
mod flatte;
mod phase_space;
#[cfg(test)]
mod tests;
mod voigt;

pub use breit_wigner::{BreitWigner, BreitWignerNonRelativistic};
pub use flatte::Flatte;
pub use phase_space::PhaseSpaceFactor;
pub use voigt::Voigt;
