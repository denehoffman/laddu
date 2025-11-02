use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID},
    resources::{CacheRow, Parameters, Resources},
    utils::{functions::rho_polars, ComplexExprExt},
    ExprID, LadduError,
};
#[cfg(feature = "python")]
use laddu_python::amplitudes::PyAmplitude;
use nalgebra::DVector;
use num::complex::Complex64;
use polars::prelude::{lit, Expr};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PyExpr;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

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
    recoil_mass: Expr,
    daughter_1_mass: Expr,
    daughter_2_mass: Expr,
    resonance_mass: Expr,
    mandelstam_s: Expr,
    eid: ExprID,
}

impl PhaseSpaceFactor {
    /// Construct a new [`Zlm`] with the given name, angular momentum (`l`), moment (`m`), and
    /// reflectivity (`r`) over the given set of [`Angles`] and [`Polarization`] [`Variable`]s.
    pub fn new(
        name: &str,
        recoil_mass: &Expr,
        daughter_1_mass: &Expr,
        daughter_2_mass: &Expr,
        resonance_mass: &Expr,
        mandelstam_s: &Expr,
    ) -> Box<Self> {
        Self {
            name: name.to_string(),
            recoil_mass: recoil_mass.clone(),
            daughter_1_mass: daughter_1_mass.clone(),
            daughter_2_mass: daughter_2_mass.clone(),
            resonance_mass: resonance_mass.clone(),
            mandelstam_s: mandelstam_s.clone(),
            eid: ExprID::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for PhaseSpaceFactor {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        let m_recoil = &self.recoil_mass;
        let m_1 = &self.daughter_1_mass;
        let m_2 = &self.daughter_2_mass;
        let m_res = &self.resonance_mass;
        let s = &self.mandelstam_s;
        let term = rho_polars(m_res.clone().pow(2), m_1.clone(), m_2.clone()).real()
            * m_res.clone()
            / (s.clone() - m_recoil.clone().pow(2)).pow(2)
            / lit(2.0 * (4.0 * PI).powi(5));
        self.eid = resources.register_scalar("phase_space_factor".into(), term.sqrt())?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, _parameters: &Parameters, cache_row: &CacheRow) -> Complex64 {
        cache_row.get_scalar(self.eid).into()
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache_row: &CacheRow,
        _gradient: &mut DVector<Complex64>,
    ) {
        // This amplitude is independent of free parameters
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
/// laddu.Amplitude
///     An Amplitude which can be registered by a laddu.Manager
///
/// See Also
/// --------
/// laddu.Manager
///
/// Notes
/// -----
/// This amplitude is described in Equation A4 of [Mathieu]_
///
#[cfg(feature = "python")]
#[pyfunction(name = "PhaseSpaceFactor")]
pub fn py_phase_space_factor(
    name: &str,
    recoil_mass: &Bound<PyAny>,
    daughter_1_mass: &Bound<PyAny>,
    daughter_2_mass: &Bound<PyAny>,
    resonance_mass: &Bound<PyAny>,
    mandelstam_s: &Bound<PyAny>,
) -> PyResult<PyAmplitude> {
    Ok(PyAmplitude(PhaseSpaceFactor::new(
        name,
        &recoil_mass.extract::<PyExpr>()?.0,
        &daughter_1_mass.extract::<PyExpr>()?.0,
        &daughter_2_mass.extract::<PyExpr>()?.0,
        &resonance_mass.extract::<PyExpr>()?.0,
        &mandelstam_s.extract::<PyExpr>()?.0,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{
        data::test_dataset, mandelstam, mass, utils::enums::Topology, Channel, Manager,
    };

    #[test]
    fn test_phase_space_factor_evaluation() {
        let mut manager = Manager::default();
        let recoil_mass = mass(["proton"]);
        let daughter_1_mass = mass(["kshort1"]);
        let daughter_2_mass = mass(["kshort2"]);
        let resonance_mass = mass(["kshort1", "kshort2"]);
        let mandelstam_s = mandelstam(
            Topology::missing_p2(["beam"], ["kshort1", "kshort2"], ["proton"]),
            Channel::S,
        );
        let amp = PhaseSpaceFactor::new(
            "kappa",
            &recoil_mass,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
            &mandelstam_s,
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[]).unwrap();

        assert_relative_eq!(result[0].re, 0.0000702838, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(result[0].im, 0.0, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn test_phase_space_factor_gradient() {
        let mut manager = Manager::default();
        let recoil_mass = mass(["proton"]);
        let daughter_1_mass = mass(["kshort1"]);
        let daughter_2_mass = mass(["kshort2"]);
        let resonance_mass = mass(["kshort1", "kshort2"]);
        let mandelstam_s = mandelstam(
            Topology::missing_p2(["beam"], ["kshort1", "kshort2"], ["proton"]),
            Channel::S,
        );
        let amp = PhaseSpaceFactor::new(
            "kappa",
            &recoil_mass,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
            &mandelstam_s,
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[]).unwrap();
        assert_eq!(result[0].len(), 0); // amplitude has no parameters
    }
}
