use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::Variable;
use crate::{data::EventLike, quantum::MandelstamChannel, reaction::MandelstamEvaluator};

/// A struct used to calculate Mandelstam variables (`s`, `t`, or `u`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mandelstam {
    pub(crate) evaluator: MandelstamEvaluator,
    pub(crate) mandelstam_channel: MandelstamChannel,
}

impl Display for Mandelstam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mandelstam(channel={})", self.mandelstam_channel)
    }
}

#[typetag::serde]
impl Variable for Mandelstam {
    fn value(&self, event: &dyn EventLike) -> f64 {
        match self.mandelstam_channel {
            MandelstamChannel::S => self.evaluator.s(event).expect("TODO"),
            MandelstamChannel::T => self.evaluator.t(event).expect("TODO"),
            MandelstamChannel::U => self.evaluator.u(event).expect("TODO"),
        }
    }
}
