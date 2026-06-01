use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::Variable;
use crate::{data::EventLike, reaction::AngleEvaluator};

/// A struct for obtaining the cosine of the polar angle of a decay product in a given frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CosTheta {
    pub(crate) evaluator: AngleEvaluator,
}

impl Display for CosTheta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CosTheta(particle={}, frame_origin={})",
            self.evaluator.particle(),
            self.evaluator.frame().origin()
        )
    }
}

#[typetag::serde]
impl Variable for CosTheta {
    fn value(&self, event: &dyn EventLike) -> f64 {
        self.evaluator.costheta(event).expect("TODO")
    }
}

/// A struct for obtaining the azimuthal angle of a decay product in a given frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phi {
    pub(crate) evaluator: AngleEvaluator,
}

impl Display for Phi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Phi(particle={}, frame_origin={})",
            self.evaluator.particle(),
            self.evaluator.frame().origin()
        )
    }
}

#[typetag::serde]
impl Variable for Phi {
    fn value(&self, event: &dyn EventLike) -> f64 {
        self.evaluator.phi(event).expect("TODO")
    }
}

/// A struct for obtaining both spherical angles at the same time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Angles {
    /// See [`CosTheta`].
    pub costheta: CosTheta,
    /// See [`Phi`].
    pub phi: Phi,
}

impl Display for Angles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Angles(particle={}, frame_origin={})",
            self.costheta.evaluator.particle(),
            self.costheta.evaluator.frame().origin()
        )
    }
}

impl Angles {
    /// Return the variable used for `cos(theta)`.
    pub fn costheta_variable(&self) -> Box<dyn Variable> {
        Box::new(self.costheta.clone())
    }

    /// Return the variable used for `phi`.
    pub fn phi_variable(&self) -> Box<dyn Variable> {
        Box::new(self.phi.clone())
    }
}
