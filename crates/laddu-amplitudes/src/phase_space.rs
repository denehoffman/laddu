use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, Expression},
    data::{DatasetMetadata, EventData, SoaNamedEventView},
    resources::{Cache, Parameters, Resources},
    utils::{functions::rho, variables::Variable},
    LadduResult, Mandelstam, Mass, ScalarID, PI,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::PyExpression,
    utils::variables::{PyMandelstam, PyMass},
};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// An [`Amplitude`] describing the phase space factor given in Equation A4 [here](https://arxiv.org/abs/1906.04841)[^1]
///
/// ```math
/// \kappa(m, s; m_1, m_2, m_{\text{recoil}}) = \frac{1}{2(4\pi)^5}
/// \frac{\sqrt{\lambda(m^2,m_1^2,m_2^2)}}{m(s-m_{\text{recoil}})^2}
/// ```
///
/// where
/// ```math
/// \lambda(a,b,c) = a^2 + b^2 + c^2 - 2(ab + bc + ca)
/// ```
///
/// Note that this amplitude actually returns `$\sqrt{\kappa}$` and is intented to be
/// used inside a coherent sum.
///
/// [^1]: Mathieu, V., Albaladejo, M., Fernández-Ramírez, C., Jackura, A. W., Mikhasenko, M., Pilloni, A., & Szczepaniak, A. P. (2019). Moments of angular distribution and beam asymmetries in $`\eta\pi^0`$ photoproduction at GlueX. _Physical Review D_, **100**(5). [doi:10.1103/physrevd.100.054017](https://doi.org/10.1103/PhysRevD.100.054017)
#[derive(Clone, Serialize, Deserialize)]
pub struct PhaseSpaceFactor {
    name: String,
    recoil_mass: Mass,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    mandelstam_s: Mandelstam,
    sid: ScalarID,
}

impl PhaseSpaceFactor {
    /// Construct a new [`PhaseSpaceFactor`] that models the two-body phase-space density.
    ///
    /// Parameters specify the recoiling particle mass, the daughter masses, the resonance
    /// mass, and the Mandelstam-s variable controlling the production kinematics.
    pub fn new(
        name: &str,
        recoil_mass: &Mass,
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
        mandelstam_s: &Mandelstam,
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            recoil_mass: recoil_mass.clone(),
            daughter_1_mass: daughter_1_mass.clone(),
            daughter_2_mass: daughter_2_mass.clone(),
            resonance_mass: resonance_mass.clone(),
            mandelstam_s: mandelstam_s.clone(),
            sid: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PhaseSpaceFactor {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.sid = resources.register_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.recoil_mass.bind(metadata)?;
        self.daughter_1_mass.bind(metadata)?;
        self.daughter_2_mass.bind(metadata)?;
        self.resonance_mass.bind(metadata)?;
        self.mandelstam_s.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        let m_recoil = self.recoil_mass.value(event);
        let m_1 = self.daughter_1_mass.value(event);
        let m_2 = self.daughter_2_mass.value(event);
        let m_res = self.resonance_mass.value(event);
        let s = self.mandelstam_s.value(event);
        let term = rho(m_res.powi(2), m_1, m_2).re * m_res
            / (s - m_recoil.powi(2)).powi(2)
            / (2.0 * (4.0 * PI).powi(5));
        cache.store_scalar(self.sid, term.sqrt());
    }

    fn precompute_soa(&self, event: &SoaNamedEventView<'_>, cache: &mut Cache) {
        let m_recoil = event.evaluate(&self.recoil_mass);
        let m_1 = event.evaluate(&self.daughter_1_mass);
        let m_2 = event.evaluate(&self.daughter_2_mass);
        let m_res = event.evaluate(&self.resonance_mass);
        let s = event.evaluate(&self.mandelstam_s);
        let term = rho(m_res.powi(2), m_1, m_2).re * m_res
            / (s - m_recoil.powi(2)).powi(2)
            / (2.0 * (4.0 * PI).powi(5));
        cache.store_scalar(self.sid, term.sqrt());
    }

    fn compute(&self, _parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        cache.get_scalar(self.sid).into()
    }

    fn compute_cached(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_scalar(self.sid).into()
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
    ) {
    }
}

/// An phase-space factor for t-channel produced particles which decay into two particles
///
/// Computes the square root of a phase-space factor for reactions
/// :math:`a+b\to c+d` with :math:`c\to 1 + 2` (see notes)
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// recoil_mass: laddu.Mass
///     The mass of the recoiling particle (:math:`d`)
/// daughter_1_mass: laddu.Mass
///     The mass of the first daughter particle of :math:`c`
/// daughter_2_mass: laddu.Mass
///     The mass of the second daughter particle of :math:`c`
/// resonance_mass: laddu.Mass
///     The mass of the resonance :math:`c`
/// mandelstam_s: laddu.Mandelstam,
///     The Mandelstam variable :math:`s`
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
/// Notes
/// -----
/// This amplitude is described in Equation A4 of [Mathieu]_
///
#[cfg(feature = "python")]
#[pyfunction(name = "PhaseSpaceFactor")]
pub fn py_phase_space_factor(
    name: &str,
    recoil_mass: &PyMass,
    daughter_1_mass: &PyMass,
    daughter_2_mass: &PyMass,
    resonance_mass: &PyMass,
    mandelstam_s: &PyMandelstam,
) -> PyResult<PyExpression> {
    Ok(PyExpression(PhaseSpaceFactor::new(
        name,
        &recoil_mass.0,
        &daughter_1_mass.0,
        &daughter_2_mass.0,
        &resonance_mass.0,
        &mandelstam_s.0,
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, utils::variables::Topology, Channel};

    #[test]
    fn test_phase_space_factor_evaluation() {
        let dataset = Arc::new(test_dataset());
        let recoil_mass = Mass::new(["proton"]);
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let topology = Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton");
        let mandelstam_s = Mandelstam::new(topology, Channel::S);
        let expr = PhaseSpaceFactor::new(
            "kappa",
            &recoil_mass,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
            &mandelstam_s,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 7.028417575882146e-5);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_phase_space_factor_gradient() {
        let dataset = Arc::new(test_dataset());
        let recoil_mass = Mass::new(["proton"]);
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let mandelstam_s = Mandelstam::new(
            Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton"),
            Channel::S,
        );
        let expr = PhaseSpaceFactor::new(
            "kappa",
            &recoil_mass,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
            &mandelstam_s,
        )
        .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[]);
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
