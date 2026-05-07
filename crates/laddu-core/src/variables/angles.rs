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
struct AngleSource {
    reaction: Box<Reaction>,
    parent: String,
    daughter: String,
}

impl AngleSource {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let _ = metadata;
        Ok(())
    }

    fn costheta(&self, event: &dyn EventLike, frame: Frame) -> f64 {
        self.reaction
            .angles_value(event, &self.parent, &self.daughter, frame)
            .unwrap_or_else(|err| panic!("failed to evaluate reaction costheta: {err}"))
            .costheta()
    }

    fn phi(&self, event: &dyn EventLike, frame: Frame) -> f64 {
        self.reaction
            .angles_value(event, &self.parent, &self.daughter, frame)
            .unwrap_or_else(|err| panic!("failed to evaluate reaction phi: {err}"))
            .phi()
    }
}

/// A struct for obtaining the cosine of the polar angle of a decay product in a given frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CosTheta {
    source: AngleSource,
    frame: Frame,
}

impl Display for CosTheta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CosTheta(parent={}, daughter={}, frame={})",
            self.source.parent, self.source.daughter, self.frame
        )
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
            source: AngleSource {
                reaction: Box::new(reaction),
                parent: parent.into(),
                daughter: daughter.into(),
            },
            frame,
        }
    }
}

#[typetag::serde]
impl Variable for CosTheta {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.source.bind(metadata)
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
        self.source.costheta(event, self.frame)
    }
}

/// A struct for obtaining the azimuthal angle of a decay product in a given frame of its parent resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phi {
    source: AngleSource,
    frame: Frame,
}

impl Display for Phi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Phi(parent={}, daughter={}, frame={})",
            self.source.parent, self.source.daughter, self.frame
        )
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
            source: AngleSource {
                reaction: Box::new(reaction),
                parent: parent.into(),
                daughter: daughter.into(),
            },
            frame,
        }
    }
}

#[typetag::serde]
impl Variable for Phi {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.source.bind(metadata)
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
        self.source.phi(event, self.frame)
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
            "Angles(parent={}, daughter={}, frame={})",
            self.costheta.source.parent, self.costheta.source.daughter, self.costheta.frame
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
}
