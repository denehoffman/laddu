use serde::{Deserialize, Serialize};

use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, Expression, ParameterLike},
    data::{DatasetMetadata, NamedEventView},
    resources::{Cache, ParameterID, Parameters, Resources},
    utils::{
        functions::{blatt_weisskopf, breakup_momentum},
        variables::{Mass, Variable},
    },
    LadduResult, ScalarID, PI,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyExpression, PyParameterLike},
    utils::variables::PyMass,
};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;

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
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    daughter_1_mass_id: ScalarID,
    daughter_2_mass_id: ScalarID,
    resonance_mass_id: ScalarID,
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
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
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
            daughter_1_mass_id: ScalarID::default(),
            daughter_2_mass_id: ScalarID::default(),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for BreitWigner {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass)?;
        self.pid_width = resources.register_parameter(&self.width)?;
        self.daughter_1_mass_id =
            resources.register_scalar(Some(&format!("{}.daughter_1_mass", self.name)));
        self.daughter_2_mass_id =
            resources.register_scalar(Some(&format!("{}.daughter_2_mass", self.name)));
        self.resonance_mass_id =
            resources.register_scalar(Some(&format!("{}.resonance_mass", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.daughter_1_mass.bind(metadata)?;
        self.daughter_2_mass.bind(metadata)?;
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        cache.store_scalar(
            self.daughter_1_mass_id,
            event.evaluate(&self.daughter_1_mass),
        );
        cache.store_scalar(
            self.daughter_2_mass_id,
            event.evaluate(&self.daughter_2_mass),
        );
        cache.store_scalar(self.resonance_mass_id, event.evaluate(&self.resonance_mass));
    }
    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let mass = cache.get_scalar(self.resonance_mass_id);
        let mass0 = parameters.get(self.pid_mass).abs();
        let width0 = parameters.get(self.pid_width).abs();
        let mass1 = cache.get_scalar(self.daughter_1_mass_id);
        let mass2 = cache.get_scalar(self.daughter_2_mass_id);
        let q0 = breakup_momentum(mass0, mass1, mass2);
        let q = breakup_momentum(mass, mass1, mass2);
        let f0 = blatt_weisskopf(mass0, mass1, mass2, self.l);
        let f = blatt_weisskopf(mass, mass1, mass2, self.l);
        let width = width0 * (mass0 / mass) * (q / q0) * (f / f0).powi(2);
        let n = f64::sqrt(mass0 * width0 / PI);
        let d = Complex64::new(mass0.powi(2) - mass.powi(2), -(mass0 * width));
        Complex64::from(f * n) / d
    }
    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let ParameterID::Parameter(index) = self.pid_mass {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
        if let ParameterID::Parameter(index) = self.pid_width {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
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
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "BreitWigner")]
pub fn py_breit_wigner(
    name: &str,
    mass: PyParameterLike,
    width: PyParameterLike,
    l: usize,
    daughter_1_mass: &PyMass,
    daughter_2_mass: &PyMass,
    resonance_mass: &PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(BreitWigner::new(
        name,
        mass.0,
        width.0,
        l,
        &daughter_1_mass.0,
        &daughter_2_mass.0,
        &resonance_mass.0,
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Mass};

    #[test]
    fn test_bw_evaluation() {
        let dataset = Arc::new(test_dataset());
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = BreitWigner::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            2,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[1.5, 0.3]);

        assert_relative_eq!(result[0].re, 1.458569174900372);
        assert_relative_eq!(result[0].im, 1.4107341131495694);
    }

    #[test]
    fn test_bw_gradient() {
        let dataset = Arc::new(test_dataset());
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = BreitWigner::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            2,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[1.7, 0.3]);
        assert_relative_eq!(result[0][0].re, -2.4105851202988857);
        assert_relative_eq!(result[0][0].im, -1.8880913749138584);
        assert_relative_eq!(result[0][1].re, 1.0467031328673773);
        assert_relative_eq!(result[0][1].im, 1.3683612879088032);
    }
}
