use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::Variable;
use crate::{
    data::{DatasetMetadata, EventLike},
    quantum::Channel,
    reaction::Reaction,
    LadduResult,
};

/// A struct used to calculate Mandelstam variables (`s`, `t`, or `u`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mandelstam {
    reaction: Reaction,
    channel: Channel,
}

impl Display for Mandelstam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mandelstam(channel={})", self.channel)
    }
}

impl Mandelstam {
    /// Constructs the Mandelstam variable for the given `channel` using the supplied [`Reaction`].
    pub fn new(reaction: Reaction, channel: Channel) -> Self {
        Self { reaction, channel }
    }
}

#[typetag::serde]
impl Variable for Mandelstam {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let _ = metadata;
        Ok(())
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
        let resolved = self
            .reaction
            .resolve_two_to_two(event)
            .unwrap_or_else(|err| panic!("failed to evaluate reaction Mandelstam: {err}"));
        match self.channel {
            Channel::S => resolved.s(),
            Channel::T => resolved.t(),
            Channel::U => resolved.u(),
        }
    }
}
