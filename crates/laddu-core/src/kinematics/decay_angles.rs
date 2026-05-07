use serde::{Deserialize, Serialize};

use crate::{vectors::Vec3, LadduError, LadduResult};

/// Spherical decay angles in a chosen frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecayAngles {
    costheta: f64,
    phi: f64,
}

impl DecayAngles {
    /// Construct angles from the components of a non-zero vector in the chosen frame.
    pub fn from_components(components: Vec3) -> LadduResult<Self> {
        let mag = components.mag();
        if !mag.is_finite() || mag <= f64::EPSILON {
            return Err(LadduError::Custom(
                "decay-angle vector must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            costheta: components.costheta().clamp(-1.0, 1.0),
            phi: components.phi(),
        })
    }

    /// Return `cos(theta)`.
    pub const fn costheta(self) -> f64 {
        self.costheta
    }

    /// Return `theta` in radians.
    pub fn theta(self) -> f64 {
        self.costheta.acos()
    }

    /// Return `phi` in radians.
    pub const fn phi(self) -> f64 {
        self.phi
    }
}
