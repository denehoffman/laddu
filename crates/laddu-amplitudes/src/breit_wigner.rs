use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, ParameterLike},
    resources::{CacheRow, ExprName, ParameterID, Parameters, Resources},
    utils::functions::{blatt_weisskopf, breakup_momentum},
    ExprID, LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::amplitudes::{PyAmplitude, PyParameterLike};
use nalgebra::DVector;
use num::complex::Complex64;
use polars::prelude::Expr;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PyExpr;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// A relativistic Breit-Wigner [`Amplitude`], parameterized as follows:
/// ```math
/// I_{\ell}(m; m_0, \Gamma_0, m_1, m_2) =  \frac{1}{\pi}\frac{m_0 \Gamma_0 B_{\ell}(m, m_1, m_2)}{(m_0^2 - m^2) - \imath m_0 \Gamma}
/// ```
/// where
/// ```math
/// \Gamma = \Gamma_0 \frac{m_0}{m} \frac{q(m, m_1, m_2)}{q(m_0, m_1, m_2)} \left(\frac{B_{\ell}(m, m_1, m_2)}{B_{\ell}(m_0, m_1, m_2)}\right)^2
/// ```
/// is the relativistic width correction, $`q(m_a, m_b, m_c)`$ is the breakup momentum of a particle with mass $`m_a`$ decaying into two particles with masses $`m_b`$ and $`m_c`$, $`B_{\ell}(m_a, m_b, m_c)`$ is the Blatt-Weisskopf barrier factor for the same decay assuming particle $`a`$ has angular momentum $`\ell`$, $`m_0`$ is the mass of the resonance, $`\Gamma_0`$ is the nominal width of the resonance, $`m_1`$ and $`m_2`$ are the masses of the decay products, and $`m`$ is the "input" mass.
#[derive(Clone, Serialize, Deserialize)]
pub struct BreitWigner {
    name: String,
    mass: ParameterLike,
    width: ParameterLike,
    pid_mass: ParameterID,
    pid_width: ParameterID,
    l: usize,
    daughter_1_mass: Expr,
    daughter_2_mass: Expr,
    resonance_mass: Expr,
    eid_daughter_1_mass: ExprID,
    eid_daughter_2_mass: ExprID,
    eid_resonance_mass: ExprID,
}
impl BreitWigner {
    /// Construct a [`BreitWigner`] with the given name, mass, width, and angular momentum (`l`).
    /// This uses the given `resonance_mass` as the "input" mass and two daughter masses of the
    /// decay products to determine phase-space and Blatt-Weisskopf factors.
    pub fn new(
        name: &str,
        mass: ParameterLike,
        width: ParameterLike,
        l: usize,
        daughter_1_mass: &Expr,
        daughter_2_mass: &Expr,
        resonance_mass: &Expr,
    ) -> Box<Self> {
        Self {
            name: name.to_string(),
            mass,
            width,
            pid_mass: ParameterID::default(),
            pid_width: ParameterID::default(),
            l,
            daughter_1_mass: daughter_1_mass.clone(),
            daughter_2_mass: daughter_2_mass.clone(),
            resonance_mass: resonance_mass.clone(),
            eid_daughter_1_mass: ExprID::default(),
            eid_daughter_2_mass: ExprID::default(),
            eid_resonance_mass: ExprID::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for BreitWigner {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass);
        self.pid_width = resources.register_parameter(&self.width);
        self.eid_daughter_1_mass =
            resources.register_scalar(ExprName::Infer, self.daughter_1_mass.clone())?;
        self.eid_daughter_2_mass =
            resources.register_scalar(ExprName::Infer, self.daughter_2_mass.clone())?;
        self.eid_resonance_mass =
            resources.register_scalar(ExprName::Infer, self.resonance_mass.clone())?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, cache_row: &CacheRow) -> Complex64 {
        let mass = cache_row.get_scalar(self.eid_resonance_mass);
        let mass0 = parameters.get(self.pid_mass).abs();
        let width0 = parameters.get(self.pid_width).abs();
        let mass1 = cache_row.get_scalar(self.eid_daughter_1_mass);
        let mass2 = cache_row.get_scalar(self.eid_daughter_2_mass);
        let q0 = breakup_momentum(mass0, mass1, mass2);
        let q = breakup_momentum(mass, mass1, mass2); // TODO: precompute
        let f0 = blatt_weisskopf(mass0, mass1, mass2, self.l);
        let f = blatt_weisskopf(mass, mass1, mass2, self.l); // TODO: precompute
        let width = width0 * (mass0 / mass) * (q / q0) * (f / f0).powi(2);
        let n = f64::sqrt(mass0 * width0 / PI);
        let d = Complex64::new(mass0.powi(2) - mass.powi(2), -(mass0 * width));
        Complex64::from(f * n) / d
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache_row: &CacheRow,
        gradient: &mut DVector<Complex64>,
    ) {
        let mut indices = Vec::with_capacity(2);
        if let ParameterID::Parameter(index) = self.pid_mass {
            indices.push(index)
        }
        if let ParameterID::Parameter(index) = self.pid_width {
            indices.push(index)
        }
        self.central_difference_with_indices(&indices, parameters, cache_row, gradient)
    }
}

/// An relativistic Breit-Wigner Amplitude
///
/// This Amplitude represents a relativistic Breit-Wigner with known angular momentum
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// mass : laddu.ParameterLike
///     The mass of the resonance
/// width : laddu.ParameterLike
///     The (nonrelativistic) width of the resonance
/// l : int
///     The total orbital momentum (:math:`l > 0`)
/// daughter_1_mass : laddu.Mass
///     The mass of the first decay product
/// daughter_2_mass : laddu.Mass
///     The mass of the second decay product
/// resonance_mass: laddu.Mass
///     The total mass of the resonance
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
#[pyfunction(name = "BreitWigner")]
pub fn py_breit_wigner(
    name: &str,
    mass: PyParameterLike,
    width: PyParameterLike,
    l: usize,
    daughter_1_mass: &Bound<PyAny>,
    daughter_2_mass: &Bound<PyAny>,
    resonance_mass: &Bound<PyAny>,
) -> PyResult<PyAmplitude> {
    Ok(PyAmplitude(BreitWigner::new(
        name,
        mass.0,
        width.0,
        l,
        &daughter_1_mass.extract::<PyExpr>()?.0,
        &daughter_2_mass.extract::<PyExpr>()?.0,
        &resonance_mass.extract::<PyExpr>()?.0,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, mass, parameter, Manager};

    #[test]
    fn test_bw_evaluation() {
        let mut manager = Manager::default();
        let amp = BreitWigner::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            2,
            &mass(["kshort1"]),
            &mass(["kshort2"]),
            &mass(["kshort1", "kshort2"]),
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[1.5, 0.3]).unwrap();

        assert_relative_eq!(result[0].re, 1.458599577038632);
        assert_relative_eq!(result[0].im, 1.4104990909599302);
    }

    #[test]
    fn test_bw_gradient() {
        let mut manager = Manager::default();
        let amp = BreitWigner::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            2,
            &mass(["kshort1"]),
            &mass(["kshort2"]),
            &mass(["kshort1", "kshort2"]),
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[1.7, 0.3]).unwrap();

        assert_relative_eq!(result[0][0].re, -2.410402487891208);
        assert_relative_eq!(result[0][0].im, -1.8877803472508647);
        assert_relative_eq!(result[0][1].re, 1.0467249913651309);
        assert_relative_eq!(result[0][1].im, 1.368233255037665);
    }
}
