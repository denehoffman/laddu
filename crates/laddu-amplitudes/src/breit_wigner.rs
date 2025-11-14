use serde::{Deserialize, Serialize};

use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, ParameterLike},
    data::{DatasetMetadata, EventData},
    resources::{Cache, ParameterID, Parameters, Resources},
    utils::{
        functions::{blatt_weisskopf, breakup_momentum},
        variables::{Mass, Variable},
    },
    LadduResult, PI,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyAmplitude, PyParameterLike},
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
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for BreitWigner {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass);
        self.pid_width = resources.register_parameter(&self.width);
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, _resources: &mut Resources, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.daughter_1_mass.bind(metadata)?;
        self.daughter_2_mass.bind(metadata)?;
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn compute(&self, parameters: &Parameters, event: &EventData, _cache: &Cache) -> Complex64 {
        let mass = self.resonance_mass.value(event);
        let mass0 = parameters.get(self.pid_mass).abs();
        let width0 = parameters.get(self.pid_width).abs();
        let mass1 = self.daughter_1_mass.value(event);
        let mass2 = self.daughter_2_mass.value(event);
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
        event: &EventData,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let mut indices = Vec::with_capacity(2);
        if let ParameterID::Parameter(index) = self.pid_mass {
            indices.push(index)
        }
        if let ParameterID::Parameter(index) = self.pid_width {
            indices.push(index)
        }
        self.central_difference_with_indices(&indices, parameters, event, cache, gradient)
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
    daughter_1_mass: &PyMass,
    daughter_2_mass: &PyMass,
    resonance_mass: &PyMass,
) -> PyAmplitude {
    PyAmplitude(BreitWigner::new(
        name,
        mass.0,
        width.0,
        l,
        &daughter_1_mass.0,
        &daughter_2_mass.0,
        &resonance_mass.0,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Manager, Mass};

    #[test]
    fn test_bw_evaluation() {
        let dataset = Arc::new(test_dataset());
        let metadata = dataset.metadata();
        let mut manager = Manager::default();
        let mut daughter_1_mass = Mass::new(["kshort1"]);
        let mut daughter_2_mass = Mass::new(["kshort2"]);
        let mut resonance_mass = Mass::new(["kshort1", "kshort2"]);
        daughter_1_mass.bind(metadata).unwrap();
        daughter_2_mass.bind(metadata).unwrap();
        resonance_mass.bind(metadata).unwrap();
        let amp = BreitWigner::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            2,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
        );
        let aid = manager.register(amp).unwrap();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate(&[1.5, 0.3]);

        assert_relative_eq!(result[0].re, 1.45856917, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(result[0].im, 1.4107341, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn test_bw_gradient() {
        let dataset = Arc::new(test_dataset());
        let metadata = dataset.metadata();
        let mut manager = Manager::default();
        let mut daughter_1_mass = Mass::new(["kshort1"]);
        let mut daughter_2_mass = Mass::new(["kshort2"]);
        let mut resonance_mass = Mass::new(["kshort1", "kshort2"]);
        daughter_1_mass.bind(metadata).unwrap();
        daughter_2_mass.bind(metadata).unwrap();
        resonance_mass.bind(metadata).unwrap();
        let amp = BreitWigner::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            2,
            &daughter_1_mass,
            &daughter_2_mass,
            &resonance_mass,
        );
        let aid = manager.register(amp).unwrap();

        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let result = evaluator.evaluate_gradient(&[1.7, 0.3]);
        assert_relative_eq!(result[0][0].re, -2.410585, epsilon = f64::EPSILON.cbrt());
        assert_relative_eq!(result[0][0].im, -1.8880913, epsilon = f64::EPSILON.cbrt());
        assert_relative_eq!(result[0][1].re, 1.0467031, epsilon = f64::EPSILON.cbrt());
        assert_relative_eq!(result[0][1].im, 1.3683612, epsilon = f64::EPSILON.cbrt());
    }
}
