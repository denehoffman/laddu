use nalgebra::DVector;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::{
    amplitudes::AmplitudeID,
    data::Event,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    utils::{
        enums::Sign,
        functions::spherical_harmonic,
        variables::{Angles, Polarization, Variable},
    },
    Float, LadduError,
};

use super::Amplitude;

/// An [`Amplitude`] representing an extension of the [`Ylm`](crate::amplitudes::ylm::Ylm)
/// [`Amplitude`] assuming a linearly polarized beam as described in Equation (D13)
/// [here](https://arxiv.org/abs/1906.04841)[^1]
///
/// [^1]: Mathieu, V., Albaladejo, M., Fernández-Ramírez, C., Jackura, A. W., Mikhasenko, M., Pilloni, A., & Szczepaniak, A. P. (2019). Moments of angular distribution and beam asymmetries in $`\eta\pi^0`$ photoproduction at GlueX. _Physical Review D_, **100**(5). [doi:10.1103/physrevd.100.054017](https://doi.org/10.1103/PhysRevD.100.054017)
#[derive(Clone, Serialize, Deserialize)]
pub struct Zlm {
    name: String,
    l: usize,
    m: isize,
    r: Sign,
    angles: Angles,
    polarization: Polarization,
    csid: ComplexScalarID,
}

impl Zlm {
    /// Construct a new [`Zlm`] with the given name, angular momentum (`l`), moment (`m`), and
    /// reflectivity (`r`) over the given set of [`Angles`] and [`Polarization`] [`Variable`]s.
    pub fn new(
        name: &str,
        l: usize,
        m: isize,
        r: Sign,
        angles: &Angles,
        polarization: &Polarization,
    ) -> Box<Self> {
        Self {
            name: name.to_string(),
            l,
            m,
            r,
            angles: angles.clone(),
            polarization: polarization.clone(),
            csid: ComplexScalarID::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for Zlm {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        self.csid = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn precompute(&self, event: &Event, cache: &mut Cache) {
        let ylm = spherical_harmonic(
            self.l,
            self.m,
            self.angles.costheta.value(event),
            self.angles.phi.value(event),
        );
        let pol_angle = self.polarization.pol_angle.value(event);
        let pgamma = self.polarization.pol_magnitude.value(event);
        let phase = Complex::new(Float::cos(-pol_angle), Float::sin(-pol_angle));
        let zlm = ylm * phase;
        cache.store_complex_scalar(
            self.csid,
            match self.r {
                Sign::Positive => Complex::new(
                    Float::sqrt(1.0 + pgamma) * zlm.re,
                    Float::sqrt(1.0 - pgamma) * zlm.im,
                ),
                Sign::Negative => Complex::new(
                    Float::sqrt(1.0 - pgamma) * zlm.re,
                    Float::sqrt(1.0 + pgamma) * zlm.im,
                ),
            },
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
    fn test_zlm_evaluation() {
        let mut manager = Manager::default();
        let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
        let polarization = Polarization::new(0, [1]);
        let amp = Zlm::new("zlm", 1, 1, Sign::Positive, &angles, &polarization);
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 0.04284127, epsilon = Float::EPSILON.sqrt());
        assert_relative_eq!(result[0].im, -0.23859638, epsilon = Float::EPSILON.sqrt());
    }

    #[test]
    fn test_zlm_gradient() {
        let mut manager = Manager::default();
        let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
        let polarization = Polarization::new(0, [1]);
        let amp = Zlm::new("zlm", 1, 1, Sign::Positive, &angles, &polarization);
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
