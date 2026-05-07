use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::{data::EventLike, vectors::Vec4, LadduError, LadduResult};

/// A kinematic particle or composite system used to define a reaction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    label: String,
    source: ParticleSource,
}

/// Source of a particle four-momentum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParticleSource {
    /// The four-momentum is read from a dataset p4 column with the same identifier.
    Stored {
        /// Dataset p4 column name.
        p4_name: String,
    },
    /// The four-momentum is fixed for every event.
    Fixed {
        /// Fixed four-momentum.
        p4: Vec4,
    },
    /// The four-momentum is solved from reaction-level four-momentum conservation.
    Missing,
    /// The four-momentum is the sum of two ordered daughter particles.
    Composite {
        /// Daughter particles whose momenta are summed.
        daughters: Box<[Particle; 2]>,
    },
}

impl Particle {
    /// Construct a stored particle backed by a dataset p4 column with the same identifier.
    pub fn stored(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            label: id.clone(),
            source: ParticleSource::Stored { p4_name: id },
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

    /// Construct a composite particle from exactly two ordered daughter particles.
    pub fn composite(
        label: impl Into<String>,
        daughters: (&Particle, &Particle),
    ) -> LadduResult<Self> {
        let daughters = [daughters.0.clone(), daughters.1.clone()];
        if daughters.iter().any(Self::contains_missing) {
            return Err(LadduError::Custom(
                "missing particles cannot be used as composite daughters".to_string(),
            ));
        }
        Ok(Self {
            label: label.into(),
            source: ParticleSource::Composite {
                daughters: Box::new(daughters),
            },
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
            ParticleSource::Composite { daughters } => daughters.as_slice(),
            _ => &[],
        }
    }

    pub(super) fn contains_id(&self, particle: &str) -> bool {
        if self.label() == particle {
            return true;
        }
        self.daughters()
            .iter()
            .any(|daughter| daughter.contains_id(particle))
    }

    pub(super) fn parent_of_id(&self, child: &str) -> Option<&Particle> {
        if self
            .daughters()
            .iter()
            .any(|daughter| daughter.label() == child)
        {
            return Some(self);
        }
        self.daughters()
            .iter()
            .find_map(|daughter| daughter.parent_of_id(child))
    }
}

impl Display for Particle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

pub(super) fn resolve_particle_direct(
    event: &dyn EventLike,
    particle: &Particle,
) -> LadduResult<Option<Vec4>> {
    match particle.source() {
        ParticleSource::Stored { p4_name } => event
            .p4(p4_name)
            .ok_or_else(|| LadduError::Custom(format!("unknown p4 column '{p4_name}'")))
            .map(Some),
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
