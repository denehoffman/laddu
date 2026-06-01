use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::Variable;
use crate::{data::EventLike, reaction::MassEvaluator};

/// A struct for obtaining the invariant mass of a selected or reaction-defined particle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mass {
    pub(crate) evaluator: MassEvaluator,
}

impl Display for Mass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mass(particle={})", self.evaluator.particle())
    }
}

#[typetag::serde]
impl Variable for Mass {
    fn value(&self, event: &dyn EventLike) -> f64 {
        self.evaluator.mass(event).expect("TODO")
    }
}
