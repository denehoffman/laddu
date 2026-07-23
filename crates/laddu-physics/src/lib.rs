//! Physics primitives for kinematic expressions, reaction channels, event
//! generation, histograms, and particle quantum numbers.

/// Reaction-graph construction and frame-dependent kinematics.
pub mod channel;
mod error;
/// Monte Carlo proposal primitives.
pub mod generation;
/// Weighted histogram utilities.
pub mod histogram;
pub mod math;
pub mod quantum;
/// Numeric and symbolic three- and four-vector types.
pub mod vectors;

pub use error::{LadduPhysicsError, LadduPhysicsResult};
