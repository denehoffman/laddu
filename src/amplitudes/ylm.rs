use num::Complex;

use crate::{
    amplitudes::AmplitudeID,
    data::Event,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    utils::{
        functions::spherical_harmonic,
        variables::{Angles, Variable},
    },
    Float,
};

use super::Amplitude;

/// An [`Amplitude`] for the spherical harmonic function $`Y_\ell^m(\theta, \phi)`$.
#[derive(Clone)]
pub struct Ylm {
    name: String,
    l: usize,
    m: isize,
    angles: Angles,
    csid: ComplexScalarID,
}

impl Ylm {
    /// Construct a new [`Ylm`] with the given name, angular momentum (`l`) and moment (`m`) over
    /// the given set of [`Angles`].
    pub fn new(name: &str, l: usize, m: isize, angles: &Angles) -> Box<Self> {
        Self {
            name: name.to_string(),
            l,
            m,
            angles: angles.clone(),
            csid: ComplexScalarID::default(),
        }
        .into()
    }
}

impl Amplitude for Ylm {
    fn register(&mut self, resources: &mut Resources) -> AmplitudeID {
        self.csid = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn precompute(&self, event: &Event, cache: &mut Cache) {
        cache.store_complex_scalar(
            self.csid,
            spherical_harmonic(
                self.l,
                self.m,
                self.angles.costheta.value(event),
                self.angles.phi.value(event),
            ),
        );
    }

    fn compute(&self, _parameters: &Parameters, _event: &Event, cache: &Cache) -> Complex<Float> {
        cache.get_complex_scalar(self.csid)
    }
}
