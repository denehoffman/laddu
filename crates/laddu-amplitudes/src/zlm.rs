use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, Expression},
    data::{DatasetMetadata, EventData},
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    utils::{
        functions::spherical_harmonic,
        variables::{Angles, Variable},
    },
    LadduResult, Polarization, Sign,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::PyExpression,
    utils::variables::{PyAngles, PyPolarization},
};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// An [`Amplitude`] representing an extension of the [`Ylm`]
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
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            l,
            m,
            r,
            angles: angles.clone(),
            polarization: polarization.clone(),
            csid: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for Zlm {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.csid = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.angles.costheta.bind(metadata)?;
        self.angles.phi.bind(metadata)?;
        self.polarization.pol_angle.bind(metadata)?;
        self.polarization.pol_magnitude.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        let ylm = spherical_harmonic(
            self.l,
            self.m,
            self.angles.costheta.value(event),
            self.angles.phi.value(event),
        );
        let pol_angle = self.polarization.pol_angle.value(event);
        let pgamma = self.polarization.pol_magnitude.value(event);
        let phase = Complex64::new(f64::cos(-pol_angle), f64::sin(-pol_angle));
        let zlm = ylm * phase;
        cache.store_complex_scalar(
            self.csid,
            match self.r {
                Sign::Positive => Complex64::new(
                    f64::sqrt(1.0 + pgamma) * zlm.re,
                    f64::sqrt(1.0 - pgamma) * zlm.im,
                ),
                Sign::Negative => Complex64::new(
                    f64::sqrt(1.0 - pgamma) * zlm.re,
                    f64::sqrt(1.0 + pgamma) * zlm.im,
                ),
            },
        );
    }

    fn compute(&self, _parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.csid)
    }

    fn compute_cached(&self, _parameters: &Parameters, cache: &Cache) -> LadduResult<Complex64> {
        Ok(cache.get_complex_scalar(self.csid))
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &EventData,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
        // This amplitude is independent of free parameters
    }

    fn compute_gradient_cached(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) -> LadduResult<()> {
        Ok(())
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
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
/// Raises
/// ------
/// ValueError
///     If `r` is not one of the valid options
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
) -> PyResult<PyExpression> {
    Ok(PyExpression(Zlm::new(
        name,
        l,
        m,
        r.parse()?,
        &angles.0,
        &polarization.0,
    )?))
}

/// An [`Amplitude`] representing the expression :math:`P_\gamma \text{Exp}(2\imath\Phi)` where
/// :math:`\P_\gamma` is the beam polarization magniutde and :math:`\Phi` is the beam
/// polarization angle. This [`Amplitude`] enocdes a polarization phase similar to Equation (3)
/// [here](https://arxiv.org/abs/1906.04841)[^1].
///
/// [^1]: Mathieu, V., Albaladejo, M., Fernández-Ramírez, C., Jackura, A. W., Mikhasenko, M., Pilloni, A., & Szczepaniak, A. P. (2019). Moments of angular distribution and beam asymmetries in $`\eta\pi^0`$ photoproduction at GlueX. _Physical Review D_, **100**(5). [doi:10.1103/physrevd.100.054017](https://doi.org/10.1103/PhysRevD.100.054017)
#[derive(Clone, Serialize, Deserialize)]
pub struct PolPhase {
    name: String,
    polarization: Polarization,
    csid: ComplexScalarID,
}

impl PolPhase {
    /// Construct a new [`PolPhase`] with the given name the given set of [`Polarization`] [`Variable`]s.
    pub fn new(name: &str, polarization: &Polarization) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            polarization: polarization.clone(),
            csid: ComplexScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PolPhase {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.csid = resources.register_complex_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.polarization.pol_angle.bind(metadata)?;
        self.polarization.pol_magnitude.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        let pol_angle = self.polarization.pol_angle.value(event);
        let pgamma = self.polarization.pol_magnitude.value(event);
        let phase = Complex64::new(f64::cos(2.0 * pol_angle), f64::sin(2.0 * pol_angle));
        cache.store_complex_scalar(self.csid, pgamma * phase);
    }

    fn compute(&self, _parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.csid)
    }

    fn compute_cached(&self, _parameters: &Parameters, cache: &Cache) -> LadduResult<Complex64> {
        Ok(cache.get_complex_scalar(self.csid))
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &EventData,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
        // This amplitude is independent of free parameters
    }

    fn compute_gradient_cached(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) -> LadduResult<()> {
        Ok(())
    }
}

/// An Amplitude representing the expression :math:`P_\gamma \text{Exp}(2\imath\Phi)` where
/// :math:`P_\gamma` is the beam polarization magniutde and :math:`\Phi` is the beam
/// polarization angle.
///
/// This Amplitude is intended to be used by its real and imaginary parts to decompose an intensity
/// function into polarized intensity components:
///
/// :math:`I = I_0 - I_1 \Re[A] - I_2 \Im[A]`
///
/// where :math:`A = P_\gamma \text{Exp}(2\imath\Phi)`.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// polarization : laddu.Polarization
///     The beam polarization to use in the calculation
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
/// Notes
/// -----
/// This amplitude is described in [Mathieu]_
///
#[cfg(feature = "python")]
#[pyfunction(name = "PolPhase")]
pub fn py_polphase(name: &str, polarization: &PyPolarization) -> PyResult<PyExpression> {
    Ok(PyExpression(PolPhase::new(name, &polarization.0)?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, utils::variables::Topology, Frame};

    fn reaction_topology() -> Topology {
        Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton")
    }

    #[test]
    fn test_zlm_evaluation() {
        let dataset = Arc::new(test_dataset());
        let topology = reaction_topology();
        let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
        let polarization = Polarization::new(topology, "pol_magnitude", "pol_angle");
        let expr = Zlm::new("zlm", 1, 1, Sign::Positive, &angles, &polarization).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 0.042841277808013944);
        assert_relative_eq!(result[0].im, -0.23859639139484332);
    }

    #[test]
    fn test_zlm_gradient() {
        let dataset = Arc::new(test_dataset());
        let topology = reaction_topology();
        let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
        let polarization = Polarization::new(topology, "pol_magnitude", "pol_angle");
        let expr = Zlm::new("zlm", 1, 1, Sign::Positive, &angles, &polarization).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }

    #[test]
    fn test_polphase_evaluation() {
        let dataset = Arc::new(test_dataset());
        let polarization = Polarization::new(reaction_topology(), "pol_magnitude", "pol_angle");
        let expr = PolPhase::new("polphase", &polarization).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, -0.2872914473575512);
        assert_relative_eq!(result[0].im, -0.2572403880070272);
    }

    #[test]
    fn test_polphase_gradient() {
        let dataset = Arc::new(test_dataset());
        let polarization = Polarization::new(reaction_topology(), "pol_magnitude", "pol_angle");
        let expr = PolPhase::new("polphase", &polarization).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
