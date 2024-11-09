use nalgebra::DVector;
use num::Complex;

use crate::{
    amplitudes::AmplitudeID,
    data::Event,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    utils::{
        functions::spherical_harmonic,
        variables::{Angles, Variable},
    },
    Float, LadduError,
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
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
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

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &Event,
        _cache: &Cache,
        _gradient: &mut DVector<Complex<Float>>,
    ) {
        // This amplitude is independent of free parameters
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::amplitudes::Manager;
    use crate::data::test_dataset;
    use crate::utils::enums::Frame;
    use approx::assert_relative_eq;

    #[test]
    fn test_ylm_evaluation() {
        let mut manager = Manager::default();
        let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
        let amp = Ylm::new("ylm", 1, 1, &angles);
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let evaluator = manager.load(&dataset, &expr);

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 0.2713394, epsilon = 1e-7);
        assert_relative_eq!(result[0].im, 0.1426897, epsilon = 1e-7);
    }

    #[test]
    fn test_ylm_gradient() {
        let mut manager = Manager::default();
        let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
        let amp = Ylm::new("ylm", 1, 1, &angles);
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let evaluator = manager.load(&dataset, &expr);

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
