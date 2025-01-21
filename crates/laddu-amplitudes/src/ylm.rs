use serde::{Deserialize, Serialize};

use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID},
    data::Event,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    utils::{
        functions::spherical_harmonic,
        variables::{Angles, Variable},
    },
    Complex, DVector, Float, LadduError,
};

#[cfg(feature = "python")]
use laddu_python::{amplitudes::PyAmplitude, utils::variables::PyAngles};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// An [`Amplitude`] for the spherical harmonic function $`Y_\ell^m(\theta, \phi)`$.
#[derive(Clone, Serialize, Deserialize)]
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

#[typetag::serde]
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

/// An spherical harmonic Amplitude
///
/// Computes a spherical harmonic (:math:`Y_{\ell}^m(\theta, \varphi)`)
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// l : int
///     The total orbital momentum (:math:`l \geq 0`)
/// m : int
///     The orbital moment (:math:`-l \leq m \leq l`)
/// angles : laddu.Angles
///     The spherical angles to use in the calculation
///     
/// Returns
/// -------
/// laddu.Amplitude
///     An Amplitude which can be registered by a laddu.Manager
///
/// See Also
/// --------
/// laddu.Manager
///
#[cfg(feature = "python")]
#[pyfunction(name = "Ylm")]
pub fn py_ylm(name: &str, l: usize, m: isize, angles: &PyAngles) -> PyAmplitude {
    PyAmplitude(Ylm::new(name, l, m, &angles.0))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, Frame, Manager};

    #[test]
    fn test_ylm_evaluation() {
        let mut manager = Manager::default();
        let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
        let amp = Ylm::new("ylm", 1, 1, &angles);
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 0.27133944, epsilon = Float::EPSILON.sqrt());
        assert_relative_eq!(result[0].im, 0.14268971, epsilon = Float::EPSILON.sqrt());
    }

    #[test]
    fn test_ylm_gradient() {
        let mut manager = Manager::default();
        let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
        let amp = Ylm::new("ylm", 1, 1, &angles);
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
