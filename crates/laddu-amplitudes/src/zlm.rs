use serde::{Deserialize, Serialize};

use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID},
    data::Event,
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    utils::{
        functions::spherical_harmonic,
        variables::{Angles, Variable},
    },
    Complex, DVector, Float, LadduError, Polarization, Sign,
};

#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::PyAmplitude,
    utils::variables::{PyAngles, PyPolarization},
};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// An [`Amplitude`] representing an extension of the [`Ylm`](crate::ylm::Ylm)
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

/// An spherical harmonic Amplitude for polarized beam experiments
///
/// Computes a polarized spherical harmonic (:math:`Z_{\ell}^{(r)m}(\theta, \varphi; P_\gamma, \Phi)`) with additional
/// polarization-related factors (see notes)
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// l : int
///     The total orbital momentum (:math:`l \geq 0`)
/// m : int
///     The orbital moment (:math:`-l \leq m \leq l`)
/// r : {'+', 'plus', 'pos', 'positive', '-', 'minus', 'neg', 'negative'}
///     The reflectivity (related to naturality of parity exchange)
/// angles : laddu.Angles
///     The spherical angles to use in the calculation
/// polarization : laddu.Polarization
///     The beam polarization to use in the calculation
///
/// Returns
/// -------
/// laddu.Amplitude
///     An Amplitude which can be registered by a laddu.Manager
///
/// Raises
/// ------
/// ValueError
///     If `r` is not one of the valid options
///
/// See Also
/// --------
/// laddu.Manager
///
/// Notes
/// -----
/// This amplitude is described in [Mathieu]_
///
/// .. [Mathieu] Mathieu, V., Albaladejo, M., Fernández-Ramírez, C., Jackura, A. W., Mikhasenko, M., Pilloni, A., & Szczepaniak, A. P. (2019). Moments of angular distribution and beam asymmetries in :math:`\eta\pi^0` photoproduction at GlueX. Physical Review D, 100(5). `doi:10.1103/physrevd.100.054017 <https://doi.org/10.1103/PhysRevD.100.054017>`_
///
#[cfg(feature = "python")]
#[pyfunction(name = "Zlm")]
pub fn py_zlm(
    name: &str,
    l: usize,
    m: isize,
    r: &str,
    angles: &PyAngles,
    polarization: &PyPolarization,
) -> PyResult<PyAmplitude> {
    Ok(PyAmplitude(Zlm::new(
        name,
        l,
        m,
        r.parse()?,
        &angles.0,
        &polarization.0,
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, Frame, Manager};

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
