use nalgebra::Vector3;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{
    data::{Dataset, Event},
    utils::{
        enums::Frame,
        vectors::{FourMomentum, FourVector, ThreeVector},
    },
    Float,
};

/// Standard methods for extracting some value out of an [`Event`].
pub trait Variable: Clone + Send + Sync {
    /// This method takes an [`Event`] and extracts a single value (like the mass of a particle).
    fn value(&self, event: &Event) -> Float;
    /// This method distributes the [`Variable::value`] method over each [`Event`] in a
    /// [`Dataset`].
    #[cfg(feature = "rayon")]
    fn value_on(&self, dataset: &Arc<Dataset>) -> Vec<Float> {
        dataset.par_iter().map(|e| self.value(e)).collect()
    }
    /// This method distributes the [`Variable::value`] method over each [`Event`] in a
    /// [`Dataset`].
    #[cfg(not(feature = "rayon"))]
    fn value_on(&self, dataset: &Arc<Dataset>) -> Vec<Float> {
        dataset.iter().map(|e| self.value(e)).collect()
    }
}

/// A struct for obtaining the mass of a particle by indexing the four-momenta of an event, adding
/// together multiple four-momenta if more than one index is given.
#[derive(Clone, Debug)]
pub struct Mass(Vec<usize>);
impl Mass {
    /// Create a new [`Mass`] from the sum of the four-momenta at the given indices in the
    /// [`Event`]'s `p4s` field.
    pub fn new<T: AsRef<[usize]>>(constituents: T) -> Self {
        Self(constituents.as_ref().into())
    }
}
impl Variable for Mass {
    fn value(&self, event: &Event) -> Float {
        event.get_p4_sum(&self.0).m()
    }
}

/// A struct for obtaining the $`\cos\theta`$ (cosine of the polar angle) of a decay product in
/// a given reference frame of its parent resonance.
#[derive(Clone, Debug)]
pub struct CosTheta {
    beam: usize,
    recoil: Vec<usize>,
    daughter: Vec<usize>,
    resonance: Vec<usize>,
    frame: Frame,
}
impl CosTheta {
    /// Construct the angle given the four-momentum indices for each specified particle. Fields
    /// which can take lists of more than one index will add the relevant four-momenta to make a
    /// new particle from the constituents. See [`Frame`] for options regarding the reference
    /// frame.
    pub fn new<T: AsRef<[usize]>, U: AsRef<[usize]>, V: AsRef<[usize]>>(
        beam: usize,
        recoil: T,
        daughter: U,
        resonance: V,
        frame: Frame,
    ) -> Self {
        Self {
            beam,
            recoil: recoil.as_ref().into(),
            daughter: daughter.as_ref().into(),
            resonance: resonance.as_ref().into(),
            frame,
        }
    }
}
impl Default for CosTheta {
    fn default() -> Self {
        Self {
            beam: 0,
            recoil: vec![1],
            daughter: vec![2],
            resonance: vec![2, 3],
            frame: Frame::Helicity,
        }
    }
}
impl Variable for CosTheta {
    fn value(&self, event: &Event) -> Float {
        let beam = event.p4s[self.beam];
        let recoil = event.get_p4_sum(&self.recoil);
        let daughter = event.get_p4_sum(&self.daughter);
        let resonance = event.get_p4_sum(&self.resonance);
        let daughter_res = daughter.boost_along(&resonance);
        match self.frame {
            Frame::Helicity => {
                let recoil_res = recoil.boost_along(&resonance);
                let z = -recoil_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vector3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.costheta()
            }
            Frame::GottfriedJackson => {
                let beam_res = beam.boost_along(&resonance);
                let z = beam_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vector3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.costheta()
            }
        }
    }
}

/// A struct for obtaining the $`\phi`$ angle (azimuthal angle) of a decay product in a given
/// reference frame of its parent resonance.
#[derive(Clone, Debug)]
pub struct Phi {
    beam: usize,
    recoil: Vec<usize>,
    daughter: Vec<usize>,
    resonance: Vec<usize>,
    frame: Frame,
}
impl Phi {
    /// Construct the angle given the four-momentum indices for each specified particle. Fields
    /// which can take lists of more than one index will add the relevant four-momenta to make a
    /// new particle from the constituents. See [`Frame`] for options regarding the reference
    /// frame.
    pub fn new<T: AsRef<[usize]>, U: AsRef<[usize]>, V: AsRef<[usize]>>(
        beam: usize,
        recoil: T,
        daughter: U,
        resonance: V,
        frame: Frame,
    ) -> Self {
        Self {
            beam,
            recoil: recoil.as_ref().into(),
            daughter: daughter.as_ref().into(),
            resonance: resonance.as_ref().into(),
            frame,
        }
    }
}
impl Default for Phi {
    fn default() -> Self {
        Self {
            beam: 0,
            recoil: vec![1],
            daughter: vec![2],
            resonance: vec![2, 3],
            frame: Frame::Helicity,
        }
    }
}
impl Variable for Phi {
    fn value(&self, event: &Event) -> Float {
        let beam = event.p4s[self.beam];
        let recoil = event.get_p4_sum(&self.recoil);
        let daughter = event.get_p4_sum(&self.daughter);
        let resonance = event.get_p4_sum(&self.resonance);
        let daughter_res = daughter.boost_along(&resonance);
        match self.frame {
            Frame::Helicity => {
                let recoil_res = recoil.boost_along(&resonance);
                let z = -recoil_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vector3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.phi()
            }
            Frame::GottfriedJackson => {
                let beam_res = beam.boost_along(&resonance);
                let z = beam_res.vec3().unit();
                let y = beam.vec3().cross(&-recoil.vec3()).unit();
                let x = y.cross(&z);
                let angles = Vector3::new(
                    daughter_res.vec3().dot(&x),
                    daughter_res.vec3().dot(&y),
                    daughter_res.vec3().dot(&z),
                );
                angles.phi()
            }
        }
    }
}

/// A struct for obtaining both spherical angles at the same time.
#[derive(Clone, Debug)]
pub struct Angles {
    /// See [`CosTheta`].
    pub costheta: CosTheta,
    /// See [`Phi`].
    pub phi: Phi,
}

impl Angles {
    /// Construct the angles given the four-momentum indices for each specified particle. Fields
    /// which can take lists of more than one index will add the relevant four-momenta to make a
    /// new particle from the constituents. See [`Frame`] for options regarding the reference
    /// frame.
    pub fn new<T: AsRef<[usize]>, U: AsRef<[usize]>, V: AsRef<[usize]>>(
        beam: usize,
        recoil: T,
        daughter: U,
        resonance: V,
        frame: Frame,
    ) -> Self {
        Self {
            costheta: CosTheta::new(beam, &recoil, &daughter, &resonance, frame),
            phi: Phi {
                beam,
                recoil: recoil.as_ref().into(),
                daughter: daughter.as_ref().into(),
                resonance: resonance.as_ref().into(),
                frame,
            },
        }
    }
}

/// A struct defining the polarization angle for a beam relative to the production plane.
#[derive(Clone, Debug)]
pub struct PolAngle {
    beam: usize,
    recoil: Vec<usize>,
}
impl PolAngle {
    /// Constructs the polarization angle given the four-momentum indices for each specified
    /// particle. Fields which can take lists of more than one index will add the relevant
    /// four-momenta to make a new particle from the constituents.
    pub fn new<T: AsRef<[usize]>>(beam: usize, recoil: T) -> Self {
        Self {
            beam,
            recoil: recoil.as_ref().into(),
        }
    }
}
impl Variable for PolAngle {
    fn value(&self, event: &Event) -> Float {
        let beam = event.p4s[self.beam];
        let recoil = event.get_p4_sum(&self.recoil);
        let y = beam.vec3().cross(&-recoil.vec3()).unit();
        Float::atan2(
            y.dot(&event.eps[self.beam]),
            beam.vec3().unit().dot(&event.eps[self.beam].cross(&y)),
        )
    }
}

/// A struct defining the polarization magnitude for a beam relative to the production plane.
#[derive(Copy, Clone, Default, Debug)]
pub struct PolMagnitude {
    beam: usize,
}

impl PolMagnitude {
    /// Constructs the polarization magnitude given the four-momentum index for the beam.
    pub fn new(beam: usize) -> Self {
        Self { beam }
    }
}
impl Variable for PolMagnitude {
    fn value(&self, event: &Event) -> Float {
        event.eps[self.beam].mag()
    }
}

/// A struct for obtaining both the polarization angle and magnitude at the same time.
#[derive(Clone, Debug)]
pub struct Polarization {
    /// See [`PolMagnitude`].
    pub pol_magnitude: PolMagnitude,
    /// See [`PolAngle`].
    pub pol_angle: PolAngle,
}

impl Polarization {
    /// Constructs the polarization angle and magnitude given the four-momentum indices for
    /// the beam and target (recoil) particle. Fields which can take lists of more than one index will add
    /// the relevant four-momenta to make a new particle from the constituents.
    pub fn new<T: AsRef<[usize]>>(beam: usize, recoil: T) -> Self {
        Self {
            pol_magnitude: PolMagnitude::new(beam),
            pol_angle: PolAngle::new(beam, recoil),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::data::{test_dataset, test_event};

    use super::*;
    #[test]
    fn test_mass_single_particle() {
        let event = test_event();
        let mass = Mass::new([1]);
        assert_relative_eq!(mass.value(&event), 1.007);
    }

    #[test]
    fn test_mass_multiple_particles() {
        let event = test_event();
        let mass = Mass::new([2, 3]);
        assert_relative_eq!(mass.value(&event), 1.3743786, epsilon = 1e-7);
    }

    #[test]
    fn test_costheta_helicity() {
        let event = test_event();
        let costheta = CosTheta::new(0, [1], [2], [2, 3], Frame::Helicity);
        assert_relative_eq!(costheta.value(&event), -0.4611175, epsilon = 1e-7);
    }

    #[test]
    fn test_phi_helicity() {
        let event = test_event();
        let phi = Phi::new(0, [1], [2], [2, 3], Frame::Helicity);
        assert_relative_eq!(phi.value(&event), -2.6574625, epsilon = 1e-7);
    }

    #[test]
    fn test_costheta_gottfried_jackson() {
        let event = test_event();
        let costheta = CosTheta::new(0, [1], [2], [2, 3], Frame::GottfriedJackson);
        assert_relative_eq!(costheta.value(&event), 0.09198832, epsilon = 1e-7);
    }

    #[test]
    fn test_phi_gottfried_jackson() {
        let event = test_event();
        let phi = Phi::new(0, [1], [2], [2, 3], Frame::GottfriedJackson);
        assert_relative_eq!(phi.value(&event), -2.7139131, epsilon = 1e-7);
    }

    #[test]
    fn test_angles() {
        let event = test_event();
        let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
        assert_relative_eq!(angles.costheta.value(&event), -0.4611175, epsilon = 1e-7);
        assert_relative_eq!(angles.phi.value(&event), -2.6574625, epsilon = 1e-7);
    }

    #[test]
    fn test_pol_angle() {
        let event = test_event();
        let pol_angle = PolAngle::new(0, vec![1]);
        assert_relative_eq!(pol_angle.value(&event), 1.9359298, epsilon = 1e-7);
    }

    #[test]
    fn test_pol_magnitude() {
        let event = test_event();
        let pol_magnitude = PolMagnitude::new(0);
        assert_relative_eq!(pol_magnitude.value(&event), 0.3856280, epsilon = 1e-7);
    }

    #[test]
    fn test_polarization() {
        let event = test_event();
        let polarization = Polarization::new(0, vec![1]);
        assert_relative_eq!(
            polarization.pol_angle.value(&event),
            1.9359298,
            epsilon = 1e-7
        );
        assert_relative_eq!(
            polarization.pol_magnitude.value(&event),
            0.3856280,
            epsilon = 1e-7
        );
    }

    #[test]
    fn test_variable_value_on() {
        let dataset = Arc::new(test_dataset());
        let mass = Mass::new(vec![2, 3]);

        let values = mass.value_on(&dataset);
        assert_eq!(values.len(), 1);
        assert_relative_eq!(values[0], 1.3743786, epsilon = 1e-7);
    }
}
