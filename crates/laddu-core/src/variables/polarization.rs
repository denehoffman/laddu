use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::{AuxSelection, Variable};
use crate::{
    data::{DatasetMetadata, NamedEventView},
    reaction::Reaction,
    vectors::Vec3,
    LadduResult,
};

/// A struct defining the polarization angle for a beam relative to the production plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolAngle {
    reaction: Reaction,
    angle_aux: AuxSelection,
}

impl Display for PolAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PolAngle(reaction={:?}, angle_aux={})",
            self.reaction.topology(),
            self.angle_aux.name(),
        )
    }
}

impl PolAngle {
    /// Constructs the polarization angle given a [`Reaction`] describing the production plane and
    /// the auxiliary column storing the precomputed angle.
    pub fn new<A>(reaction: Reaction, angle_aux: A) -> Self
    where
        A: Into<String>,
    {
        Self {
            reaction,
            angle_aux: AuxSelection::new(angle_aux.into()),
        }
    }
}

#[typetag::serde]
impl Variable for PolAngle {
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        let _ = metadata;
        self.angle_aux.bind(metadata)?;
        Ok(())
    }

    fn value(&self, event: &NamedEventView<'_>) -> f64 {
        let resolved = self
            .reaction
            .resolve_two_to_two(event)
            .unwrap_or_else(|err| panic!("failed to evaluate polarization angle: {err}"));
        let beam = resolved.p1();
        let recoil = resolved.p4();
        let pol_angle = event.aux_at(self.angle_aux.index());
        let polarization = Vec3::new(pol_angle.cos(), pol_angle.sin(), 0.0);
        let y = beam.vec3().cross(&-recoil.vec3()).unit();
        let numerator = y.dot(&polarization);
        let denominator = beam.vec3().unit().dot(&polarization.cross(&y));
        f64::atan2(numerator, denominator)
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
            self.magnitude_aux.name(),
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

    fn value(&self, event: &NamedEventView<'_>) -> f64 {
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
            "Polarization(reaction={:?}, magnitude_aux={}, angle_aux={})",
            self.pol_angle.reaction.topology(),
            self.pol_magnitude.magnitude_aux.name(),
            self.pol_angle.angle_aux.name(),
        )
    }
}

impl Polarization {
    /// Constructs the polarization angle and magnitude given a [`Reaction`] and distinct
    /// auxiliary columns for magnitude and angle.
    ///
    /// # Panics
    ///
    /// Panics if `magnitude_aux` and `angle_aux` refer to the same auxiliary column name.
    pub fn new<M, A>(reaction: Reaction, magnitude_aux: M, angle_aux: A) -> Self
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
            pol_angle: PolAngle::new(reaction, angle_aux),
        }
    }
}
