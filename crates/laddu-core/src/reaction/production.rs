use serde::{Deserialize, Serialize};

use super::Reaction;
use crate::{
    variables::{Angles, CosTheta, Phi},
    LadduResult,
};

/// A two-to-two production view used to derive production-local variables and amplitudes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Production {
    pub(super) reaction: Reaction,
    pub(super) produced: String,
    pub(super) recoil: String,
}

impl Production {
    /// Return the enclosing reaction.
    pub const fn reaction(&self) -> &Reaction {
        &self.reaction
    }

    /// Return the produced-system particle identifier.
    pub fn produced(&self) -> &str {
        &self.produced
    }

    /// Return the recoil particle identifier.
    pub fn recoil(&self) -> &str {
        &self.recoil
    }

    /// Return the production costheta variable.
    pub fn costheta(&self) -> LadduResult<CosTheta> {
        Ok(CosTheta::from_production(
            self.reaction.clone(),
            self.produced.clone(),
        ))
    }

    /// Return the production phi variable.
    pub fn phi(&self) -> LadduResult<Phi> {
        Ok(Phi::from_production(
            self.reaction.clone(),
            self.produced.clone(),
        ))
    }

    /// Return both production angle variables.
    pub fn angles(&self) -> LadduResult<Angles> {
        Ok(Angles::from_production(
            self.reaction.clone(),
            self.produced.clone(),
        ))
    }
}
