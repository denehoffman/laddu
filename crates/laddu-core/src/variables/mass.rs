use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::{format_names, IntoP4Selection, P4Selection, Variable};
use crate::{
    data::{DatasetMetadata, Event},
    reaction::Reaction,
    vectors::Vec4,
    LadduResult,
};

/// Source for a mass variable.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum MassSource {
    Selection(P4Selection),
    Reaction {
        reaction: Box<Reaction>,
        particle: String,
    },
}

/// A struct for obtaining the invariant mass of a selected or reaction-defined particle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mass {
    source: MassSource,
}

impl Mass {
    /// Create a new [`Mass`] from the sum of the four-momenta identified by `constituents` in the
    /// [`EventData`](crate::data::EventData)'s `p4s` field.
    pub fn new<C>(constituents: C) -> Self
    where
        C: IntoP4Selection,
    {
        Self {
            source: MassSource::Selection(constituents.into_selection()),
        }
    }

    /// Create a new [`Mass`] for a particle resolved through a [`Reaction`].
    pub fn from_reaction(reaction: Reaction, particle: impl Into<String>) -> Self {
        Self {
            source: MassSource::Reaction {
                reaction: Box::new(reaction),
                particle: particle.into(),
            },
        }
    }
}

impl Display for Mass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            MassSource::Selection(constituents) => {
                write!(
                    f,
                    "Mass(constituents=[{}])",
                    format_names(constituents.names())
                )
            }
            MassSource::Reaction { particle, .. } => write!(f, "Mass(particle={})", particle),
        }
    }
}

#[typetag::serde]
impl Variable for Mass {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        match &mut self.source {
            MassSource::Selection(constituents) => constituents.bind(metadata),
            MassSource::Reaction { .. } => Ok(()),
        }
    }

    fn value(&self, event: &Event<'_>) -> f64 {
        match &self.source {
            MassSource::Selection(constituents) => constituents
                .indices()
                .iter()
                .map(|index| event.p4_at(*index))
                .sum::<Vec4>()
                .m(),
            MassSource::Reaction { reaction, particle } => reaction
                .p4(event, particle)
                .unwrap_or_else(|err| panic!("failed to evaluate reaction mass: {err}"))
                .m(),
        }
    }
}
