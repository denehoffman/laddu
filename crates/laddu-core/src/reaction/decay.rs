use serde::{Deserialize, Serialize};

use super::Reaction;
use crate::{
    quantum::Frame,
    variables::{Angles, CosTheta, Mass, Phi},
    LadduError, LadduResult,
};

/// An isobar decay view used to derive variables and decay-local amplitudes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decay {
    pub(super) reaction: Reaction,
    pub(super) parent: String,
    pub(super) daughter_1: String,
    pub(super) daughter_2: String,
}

impl Decay {
    /// Return the enclosing reaction.
    pub const fn reaction(&self) -> &Reaction {
        &self.reaction
    }

    /// Return the parent particle identifier.
    pub fn parent(&self) -> &str {
        &self.parent
    }

    /// Return the first daughter particle identifier.
    pub fn daughter_1(&self) -> &str {
        &self.daughter_1
    }

    /// Return the second daughter particle identifier.
    pub fn daughter_2(&self) -> &str {
        &self.daughter_2
    }

    /// Return the ordered daughter particle identifiers.
    pub fn daughters(&self) -> [&str; 2] {
        [&self.daughter_1, &self.daughter_2]
    }

    /// Return the parent mass variable.
    pub fn mass(&self) -> Mass {
        self.reaction.mass(&self.parent)
    }

    /// Return the parent mass variable.
    pub fn parent_mass(&self) -> Mass {
        self.mass()
    }

    /// Return the first daughter mass variable.
    pub fn daughter_1_mass(&self) -> Mass {
        self.reaction.mass(&self.daughter_1)
    }

    /// Return the second daughter mass variable.
    pub fn daughter_2_mass(&self) -> Mass {
        self.reaction.mass(&self.daughter_2)
    }

    /// Return the mass variable for one of the ordered daughters.
    pub fn daughter_mass(&self, daughter: &str) -> LadduResult<Mass> {
        self.validate_daughter(daughter)?;
        Ok(self.reaction.mass(daughter))
    }

    /// Return the decay costheta variable.
    pub fn costheta(&self, daughter: &str, frame: Frame) -> LadduResult<CosTheta> {
        self.validate_daughter(daughter)?;
        Ok(CosTheta::from_reaction(
            self.reaction.clone(),
            self.parent.clone(),
            daughter,
            frame,
        ))
    }

    /// Return the decay phi variable.
    pub fn phi(&self, daughter: &str, frame: Frame) -> LadduResult<Phi> {
        self.validate_daughter(daughter)?;
        Ok(Phi::from_reaction(
            self.reaction.clone(),
            self.parent.clone(),
            daughter,
            frame,
        ))
    }

    /// Return both decay angle variables.
    pub fn angles(&self, daughter: &str, frame: Frame) -> LadduResult<Angles> {
        self.validate_daughter(daughter)?;
        Ok(Angles::from_reaction(
            self.reaction.clone(),
            self.parent.clone(),
            daughter,
            frame,
        ))
    }

    fn validate_daughter(&self, daughter: &str) -> LadduResult<()> {
        if daughter == self.daughter_1 || daughter == self.daughter_2 {
            Ok(())
        } else {
            Err(LadduError::Custom(format!(
                "particle '{}' is not an immediate daughter of '{}'",
                daughter, self.parent
            )))
        }
    }
}
