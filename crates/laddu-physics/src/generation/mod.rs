//! Kinematic proposal primitives used by Monte Carlo generators.

use std::{f64::consts::PI, sync::Arc};

use maryada::{Interval, IntervalOps};
use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    histogram::Histogram,
    quantum::ParticleProperties,
    vectors::{RealVec3, RealVec4},
};

mod density;

#[doc(hidden)]
pub use density::PiecewiseDensity;

include!("rng.rs");
include!("mass.rs");
include!("scalar.rs");
include!("initial.rs");
include!("two_body.rs");
include!("transfer.rs");
