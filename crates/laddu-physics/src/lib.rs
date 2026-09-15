//! Physics primitives for kinematic expressions, reaction channels, event
//! generation, histograms, and particle quantum numbers.

/// Validated axes and shared bin-assignment semantics.
pub mod binning;
/// Reaction-graph construction and frame-dependent kinematics.
pub mod channel;
mod error;
/// Monte Carlo proposal primitives.
pub mod generation;
/// Weighted histogram utilities.
pub mod histogram;
/// Weighted joint-histogram primitives.
pub mod joint_histogram;
pub mod math;
pub mod quantum;
/// Numeric and symbolic three- and four-vector types.
pub mod vectors;

pub use error::{LadduPhysicsError, LadduPhysicsResult};
