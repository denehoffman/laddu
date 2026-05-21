use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::Variable;
use crate::{
    data::{DatasetMetadata, EventLike},
    quantum::Frame,
    reaction::Reaction,
    LadduResult,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
enum AngleSource {
    Decay {
        reaction: Box<Reaction>,
        parent: String,
        daughter: String,
        frame: Frame,
    },
    Production {
        reaction: Box<Reaction>,
        produced: String,
    },
}

impl AngleSource {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let _ = metadata;
        Ok(())
    }

    fn costheta(&self, event: &dyn EventLike) -> f64 {
        match self {
            Self::Decay {
                reaction,
                parent,
                daughter,
                frame,
            } => reaction
                .angles_value(event, parent, daughter, *frame)
                .unwrap_or_else(|err| panic!("failed to evaluate reaction costheta: {err}"))
                .costheta(),
            Self::Production { reaction, produced } => reaction
                .production_angles_value(event, produced)
                .unwrap_or_else(|err| panic!("failed to evaluate production costheta: {err}"))
                .costheta(),
        }
    }

    fn phi(&self, event: &dyn EventLike) -> f64 {
        match self {
            Self::Decay {
                reaction,
                parent,
                daughter,
                frame,
            } => reaction
                .angles_value(event, parent, daughter, *frame)
                .unwrap_or_else(|err| panic!("failed to evaluate reaction phi: {err}"))
                .phi(),
            Self::Production { reaction, produced } => reaction
                .production_angles_value(event, produced)
                .unwrap_or_else(|err| panic!("failed to evaluate production phi: {err}"))
                .phi(),
        }
    }

    fn label(&self, kind: &str) -> String {
        match self {
            Self::Decay {
                parent,
                daughter,
                frame,
                ..
            } => format!("{kind}(parent={parent}, daughter={daughter}, frame={frame})"),
            Self::Production { produced, .. } => format!("{kind}(produced={produced})"),
        }
    }

    fn angles_label(&self) -> String {
        match self {
            Self::Decay {
                parent,
                daughter,
                frame,
                ..
            } => format!("Angles(parent={parent}, daughter={daughter}, frame={frame})"),
            Self::Production { produced, .. } => format!("Angles(produced={produced})"),
        }
    }
}

/// A struct for obtaining the cosine of the polar angle of a decay product in a given frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CosTheta {
    source: AngleSource,
}

impl Display for CosTheta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source.label("CosTheta"))
    }
}

impl CosTheta {
    /// Construct an angle for a reaction daughter in the specified parent frame.
    pub fn from_reaction(
        reaction: Reaction,
        parent: impl Into<String>,
        daughter: impl Into<String>,
        frame: Frame,
    ) -> Self {
        Self {
            source: AngleSource::Decay {
                reaction: Box::new(reaction),
                parent: parent.into(),
                daughter: daughter.into(),
                frame,
            },
        }
    }

    /// Construct an angle for the produced system.
    pub fn from_production(reaction: Reaction, produced: impl Into<String>) -> Self {
        Self {
            source: AngleSource::Production {
                reaction: Box::new(reaction),
                produced: produced.into(),
            },
        }
    }
}

#[typetag::serde]
impl Variable for CosTheta {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.source.bind(metadata)
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
        self.source.costheta(event)
    }
}

/// A struct for obtaining the azimuthal angle of a decay product in a given frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phi {
    source: AngleSource,
}

impl Display for Phi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source.label("Phi"))
    }
}

impl Phi {
    /// Construct an angle for a reaction daughter in the specified parent frame.
    pub fn from_reaction(
        reaction: Reaction,
        parent: impl Into<String>,
        daughter: impl Into<String>,
        frame: Frame,
    ) -> Self {
        Self {
            source: AngleSource::Decay {
                reaction: Box::new(reaction),
                parent: parent.into(),
                daughter: daughter.into(),
                frame,
            },
        }
    }

    /// Construct an angle for the produced system.
    pub fn from_production(reaction: Reaction, produced: impl Into<String>) -> Self {
        Self {
            source: AngleSource::Production {
                reaction: Box::new(reaction),
                produced: produced.into(),
            },
        }
    }
}

#[typetag::serde]
impl Variable for Phi {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.source.bind(metadata)
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
        self.source.phi(event)
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
        write!(f, "{}", self.costheta.source.angles_label())
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

    /// Construct reaction-derived angle variables for a daughter in its parent frame.
    pub fn from_reaction(
        reaction: Reaction,
        parent: impl Into<String>,
        daughter: impl Into<String>,
        frame: Frame,
    ) -> Self {
        let parent = parent.into();
        let daughter = daughter.into();
        let costheta =
            CosTheta::from_reaction(reaction.clone(), parent.clone(), daughter.clone(), frame);
        let phi = Phi::from_reaction(reaction, parent, daughter, frame);
        Self { costheta, phi }
    }

    /// Construct reaction-derived production angle variables.
    pub fn from_production(reaction: Reaction, produced: impl Into<String>) -> Self {
        let produced = produced.into();
        let costheta = CosTheta::from_production(reaction.clone(), produced.clone());
        let phi = Phi::from_production(reaction, produced);
        Self { costheta, phi }
    }
}
