use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::{AuxSelection, Variable};
use crate::{
    data::{DatasetMetadata, EventLike},
    reaction::PolarizationAngleEvaluator,
    LadduResult,
};

/// A struct defining the polarization angle for a beam relative to the production plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolAngle {
    evaluator: PolarizationAngleEvaluator,
    angle_aux: AuxSelection,
}

impl Display for PolAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolAngle(vertex={}, reference={}, spectator={}, angle_aux={})",
            self.evaluator.vertex(),
            self.evaluator.reference(),
            self.evaluator.spectator(),
            self.angle_aux.name()
        )
    }
}

impl PolAngle {
    /// Constructs the polarization angle given a topology-backed evaluator and the auxiliary
    /// column storing the precomputed lab-frame polarization angle.
    pub fn new<A>(evaluator: PolarizationAngleEvaluator, angle_aux: A) -> Self
    where
        A: Into<String>,
    {
        Self {
            evaluator,
            angle_aux: AuxSelection::new(angle_aux.into()),
        }
    }
}

#[typetag::serde]
impl Variable for PolAngle {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.angle_aux.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
        let pol_angle = event.aux_at(self.angle_aux.index());
        self.evaluator.angle(event, pol_angle).expect("TODO")
    }
}

/// A struct defining the polarization magnitude for a beam relative to the production plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolMagnitude {
    magnitude_aux: AuxSelection,
}

impl Display for PolMagnitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolMagnitude(magnitude_aux={})",
            self.magnitude_aux.name()
        )
    }
}

impl PolMagnitude {
    /// Constructs the polarization magnitude given the named auxiliary column containing the
    /// magnitude value.
    pub fn new<S: Into<String>>(magnitude_aux: S) -> Self {
        Self {
            magnitude_aux: AuxSelection::new(magnitude_aux.into()),
        }
    }
}

#[typetag::serde]
impl Variable for PolMagnitude {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.magnitude_aux.bind(metadata)
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
        event.aux_at(self.magnitude_aux.index())
    }
}

/// A struct for obtaining both the polarization angle and magnitude at the same time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Polarization {
    /// See [`PolMagnitude`].
    pub pol_magnitude: PolMagnitude,
    /// See [`PolAngle`].
    pub pol_angle: PolAngle,
}

impl Display for Polarization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Polarization(vertex={}, reference={}, spectator={}, magnitude_aux={}, angle_aux={})",
            self.pol_angle.evaluator.vertex(),
            self.pol_angle.evaluator.reference(),
            self.pol_angle.evaluator.spectator(),
            self.pol_magnitude.magnitude_aux.name(),
            self.pol_angle.angle_aux.name(),
        )
    }
}

impl Polarization {
    /// Constructs the polarization angle and magnitude given a topology-backed angle evaluator and
    /// distinct auxiliary columns for magnitude and angle.
    ///
    /// # Panics
    ///
    /// Panics if `magnitude_aux` and `angle_aux` refer to the same auxiliary column name.
    pub fn new<M, A>(evaluator: PolarizationAngleEvaluator, magnitude_aux: M, angle_aux: A) -> Self
    where
        M: Into<String>,
        A: Into<String>,
    {
        let magnitude_aux = magnitude_aux.into();
        let angle_aux = angle_aux.into();
        assert!(
            magnitude_aux != angle_aux,
            "Polarization magnitude and angle must reference distinct auxiliary columns"
        );
        Self {
            pol_magnitude: PolMagnitude::new(magnitude_aux),
            pol_angle: PolAngle::new(evaluator, angle_aux),
        }
    }
}
