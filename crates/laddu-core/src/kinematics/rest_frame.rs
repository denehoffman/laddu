use serde::{Deserialize, Serialize};

use super::support::checked_boost_vector;
use crate::{
    vectors::{Vec3, Vec4},
    LadduResult,
};

/// Rest-frame transform for a parent four-momentum.
///
/// Sequential-decay builders start from lab-frame event vectors, then apply one rest-frame
/// transform per vertex. Each child vertex receives four-vectors that have already been
/// transformed into its parent's rest frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RestFrame {
    parent: Vec4,
    boost: Vec3,
}

impl RestFrame {
    /// Construct a rest-frame transform for `parent`.
    pub fn new(parent: Vec4) -> LadduResult<Self> {
        let boost = checked_boost_vector(-parent.beta(), "parent")?;
        Ok(Self { parent, boost })
    }

    /// Return the parent four-momentum.
    pub const fn parent(self) -> Vec4 {
        self.parent
    }

    /// Return the boost vector into the parent rest frame.
    pub const fn boost(self) -> Vec3 {
        self.boost
    }

    /// Transform a four-momentum into the parent rest frame.
    pub fn transform(self, momentum: Vec4) -> Vec4 {
        momentum.boost(&self.boost)
    }

    /// Transform a four-momentum into the parent rest frame and return its three-momentum.
    pub fn momentum(self, momentum: Vec4) -> Vec3 {
        self.transform(momentum).vec3()
    }
}
