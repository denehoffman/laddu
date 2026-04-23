use std::{borrow::Borrow, fmt::Display};

use serde::{Deserialize, Serialize};

use crate::{
    data::NamedEventView, variables::IntoP4Selection, vectors::Vec4, LadduError, LadduResult,
};

/// A kinematic particle or composite system used to define a reaction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    label: String,
    source: ParticleSource,
}

/// Source of a particle four-momentum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParticleSource {
    /// The four-momentum is read from one or more dataset p4 columns.
    Measured {
        /// Dataset p4 column names whose momenta are summed.
        p4_names: Vec<String>,
    },
    /// The four-momentum is fixed for every event.
    Fixed {
        /// Fixed four-momentum.
        p4: Vec4,
    },
    /// The four-momentum is solved from reaction-level four-momentum conservation.
    Missing,
    /// The four-momentum is the sum of daughter particles.
    Composite {
        /// Daughter particles whose momenta are summed.
        daughters: Vec<Particle>,
    },
}

impl Particle {
    /// Construct a measured particle backed by one or more p4 column names.
    pub fn measured<S>(label: impl Into<String>, p4: S) -> Self
    where
        S: IntoP4Selection,
    {
        Self {
            label: label.into(),
            source: ParticleSource::Measured {
                p4_names: p4.into_selection().names().to_vec(),
            },
        }
    }

    /// Construct a particle with a fixed four-momentum.
    pub fn fixed(label: impl Into<String>, p4: Vec4) -> Self {
        Self {
            label: label.into(),
            source: ParticleSource::Fixed { p4 },
        }
    }

    /// Construct a missing particle solved by the enclosing reaction topology.
    pub fn missing(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            source: ParticleSource::Missing,
        }
    }

    /// Construct a composite particle from daughter particles.
    pub fn composite<I, P>(label: impl Into<String>, daughters: I) -> LadduResult<Self>
    where
        I: IntoIterator<Item = P>,
        P: Borrow<Particle>,
    {
        let daughters = daughters
            .into_iter()
            .map(|daughter| daughter.borrow().clone())
            .collect::<Vec<_>>();
        if daughters.is_empty() {
            return Err(LadduError::Custom(
                "composite particle must contain at least one daughter".to_string(),
            ));
        }
        if daughters.iter().any(Self::contains_missing) {
            return Err(LadduError::Custom(
                "missing particles cannot be used as composite daughters".to_string(),
            ));
        }
        Ok(Self {
            label: label.into(),
            source: ParticleSource::Composite { daughters },
        })
    }

    /// Return the particle label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return the particle four-momentum source.
    pub const fn source(&self) -> &ParticleSource {
        &self.source
    }

    /// Return whether this particle is missing.
    pub const fn is_missing(&self) -> bool {
        matches!(self.source, ParticleSource::Missing)
    }

    pub(super) fn contains_missing(&self) -> bool {
        self.is_missing() || self.daughters().iter().any(Self::contains_missing)
    }

    /// Return the daughters if this particle is composite.
    pub fn daughters(&self) -> &[Particle] {
        match &self.source {
            ParticleSource::Composite { daughters } => daughters,
            _ => &[],
        }
    }

    pub(super) fn contains(&self, particle: &Particle) -> bool {
        if self == particle {
            return true;
        }
        self.daughters()
            .iter()
            .any(|daughter| daughter.contains(particle))
    }

    pub(super) fn parent_of(&self, child: &Particle) -> Option<&Particle> {
        if self.daughters().iter().any(|daughter| daughter == child) {
            return Some(self);
        }
        self.daughters()
            .iter()
            .find_map(|daughter| daughter.parent_of(child))
    }
}

impl Display for Particle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

pub(super) fn resolve_particle_direct(
    event: &NamedEventView<'_>,
    particle: &Particle,
) -> LadduResult<Option<Vec4>> {
    match particle.source() {
        ParticleSource::Measured { p4_names } => {
            if p4_names.is_empty() {
                return Err(LadduError::Custom(
                    "measured particle must contain at least one p4 name".to_string(),
                ));
            }
            event
                .get_p4_sum(p4_names.iter().map(String::as_str))
                .ok_or_else(|| {
                    LadduError::Custom(format!("unknown p4 selection '{}'", p4_names.join(", ")))
                })
                .map(Some)
        }
        ParticleSource::Fixed { p4 } => Ok(Some(*p4)),
        ParticleSource::Missing => Ok(None),
        ParticleSource::Composite { daughters } => daughters
            .iter()
            .map(|daughter| {
                resolve_particle_direct(event, daughter)?.ok_or_else(|| {
                    LadduError::Custom(format!(
                        "missing daughter '{}' cannot be resolved inside composite '{}'",
                        daughter.label(),
                        particle.label()
                    ))
                })
            })
            .try_fold(Vec4::new(0.0, 0.0, 0.0, 0.0), |acc, value| {
                value.map(|value| acc + value)
            })
            .map(Some),
    }
}
