use laddu_core::{
    amplitudes::AmplitudeSemanticKey,
    data::NamedEventView,
    math::{rho_m, Sheet},
    traits::{Amplitude, Variable},
    AmplitudeID, Cache, DatasetMetadata, Expression, LadduResult, Mass, Parameter, ParameterID,
    Parameters, Resources, ScalarID,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyExpression, PyParameter},
    utils::variables::PyMass,
};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::semantic_key::{debug_key, display_key, parameter_key};

/// A Flatté [`Amplitude`], parameterized as follows:
/// ```math
/// I(m; m_0, g_o, g_a, m_{o1}, m_{o2}, m_{a1}, m_{a2}) =  \frac{1}{(m_0^2 - m^2) - \imath m_0 (g_o \rho(m, m_{o1}, m_{o2}) + g_a \rho(m, m_{a1}, m_{a2})))}
/// ```
/// $`\rho(m_a, m_b, m_c)`$ is the phase-space factor of a particle with mass $`m_a`$ decaying into two particles with masses $`m_b`$ and $`m_c`$. The observed channel daughter masses are event-dependent [`Mass`] variables, while the alternate channel daughter masses are fixed threshold values.
#[derive(Clone, Serialize, Deserialize)]
pub struct Flatte {
    name: String,
    mass: Parameter,
    observed_channel_coupling: Parameter,
    alternate_channel_coupling: Parameter,
    pid_mass: ParameterID,
    pid_observed_channel_coupling: ParameterID,
    pid_alternate_channel_coupling: ParameterID,
    observed_channel_daughter_masses: (Mass, Mass),
    alternate_channel_daughter_masses: (f64, f64),
    resonance_mass: Mass,
    observed_channel_mass_ids: (ScalarID, ScalarID),
    resonance_mass_id: ScalarID,
}
impl Flatte {
    /// Construct a [`Flatte`] with the given name, mass, couplings, and channel daughter masses.
    /// This uses the given `resonance_mass` as the "input" mass. The observed channel daughter
    /// masses are event-dependent variables, while the alternate channel daughter masses are fixed.
    pub fn new(
        name: &str,
        mass: Parameter,
        observed_channel_coupling: Parameter,
        alternate_channel_coupling: Parameter,
        observed_channel_daughter_masses: (&Mass, &Mass),
        alternate_channel_daughter_masses: (f64, f64),
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            mass,
            observed_channel_coupling,
            alternate_channel_coupling,
            pid_mass: ParameterID::default(),
            pid_observed_channel_coupling: ParameterID::default(),
            pid_alternate_channel_coupling: ParameterID::default(),
            observed_channel_daughter_masses: (
                observed_channel_daughter_masses.0.clone(),
                observed_channel_daughter_masses.1.clone(),
            ),
            alternate_channel_daughter_masses,
            resonance_mass: resonance_mass.clone(),
            observed_channel_mass_ids: (ScalarID::default(), ScalarID::default()),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for Flatte {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass)?;
        self.pid_observed_channel_coupling =
            resources.register_parameter(&self.observed_channel_coupling)?;
        self.pid_alternate_channel_coupling =
            resources.register_parameter(&self.alternate_channel_coupling)?;
        self.observed_channel_mass_ids = (
            resources.register_scalar(Some(&format!(
                "{}.observed_channel.daughter_1_mass",
                self.name
            ))),
            resources.register_scalar(Some(&format!(
                "{}.observed_channel.daughter_2_mass",
                self.name
            ))),
        );
        self.resonance_mass_id =
            resources.register_scalar(Some(&format!("{}.resonance_mass", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("Flatte")
                .with_field("name", debug_key(&self.name))
                .with_field("mass", parameter_key(&self.mass))
                .with_field(
                    "observed_channel_coupling",
                    parameter_key(&self.observed_channel_coupling),
                )
                .with_field(
                    "alternate_channel_coupling",
                    parameter_key(&self.alternate_channel_coupling),
                )
                .with_field(
                    "observed_channel_m1",
                    display_key(&self.observed_channel_daughter_masses.0),
                )
                .with_field(
                    "observed_channel_m2",
                    display_key(&self.observed_channel_daughter_masses.1),
                )
                .with_field(
                    "alternate_channel_m1",
                    self.alternate_channel_daughter_masses.0.to_string(),
                )
                .with_field(
                    "alternate_channel_m2",
                    self.alternate_channel_daughter_masses.1.to_string(),
                )
                .with_field("resonance_mass", display_key(&self.resonance_mass)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.observed_channel_daughter_masses.0.bind(metadata)?;
        self.observed_channel_daughter_masses.1.bind(metadata)?;
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        cache.store_scalar(
            self.observed_channel_mass_ids.0,
            event.evaluate(&self.observed_channel_daughter_masses.0),
        );
        cache.store_scalar(
            self.observed_channel_mass_ids.1,
            event.evaluate(&self.observed_channel_daughter_masses.1),
        );
        cache.store_scalar(self.resonance_mass_id, event.evaluate(&self.resonance_mass));
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let mass = cache.get_scalar(self.resonance_mass_id);
        let mass0 = parameters.get(self.pid_mass).abs();
        let observed_channel_coupling = parameters.get(self.pid_observed_channel_coupling).abs();
        let alternate_channel_coupling = parameters.get(self.pid_alternate_channel_coupling).abs();
        let observed_m1 = cache.get_scalar(self.observed_channel_mass_ids.0);
        let observed_m2 = cache.get_scalar(self.observed_channel_mass_ids.1);
        let (alternate_m1, alternate_m2) = self.alternate_channel_daughter_masses;
        let observed_rho = rho_m(mass, observed_m1, observed_m2, Sheet::Physical);
        let alternate_rho = rho_m(mass, alternate_m1, alternate_m2, Sheet::Physical);

        1.0 / (Complex64::from(mass0.powi(2) - mass.powi(2))
            - Complex64::I
                * mass0
                * (observed_channel_coupling * observed_rho
                    + alternate_channel_coupling * alternate_rho))
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let Some(index) = parameters.free_index(self.pid_mass) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
        if let Some(index) = parameters.free_index(self.pid_observed_channel_coupling) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
        if let Some(index) = parameters.free_index(self.pid_alternate_channel_coupling) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
    }
}

/// A Flatté Amplitude
///
/// This Amplitude represents a Flatté distribution.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// mass : laddu.Parameter
///     The mass of the resonance
/// observed_channel_coupling : laddu.Parameter
///     The coupling to the observed channel
/// alternate_channel_coupling : laddu.Parameter
///     The coupling to the alternate channel
/// observed_channel_daughter_masses : tuple[laddu.Mass, laddu.Mass]
///     The event-dependent daughter masses of the observed decay channel
/// alternate_channel_daughter_masses : tuple[float, float]
///     The fixed daughter masses of the alternate decay channel
/// resonance_mass: laddu.Mass
///     The total mass of the resonance
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "Flatte")]
pub fn py_flatte(
    name: &str,
    mass: PyParameter,
    observed_channel_coupling: PyParameter,
    alternate_channel_coupling: PyParameter,
    observed_channel_daughter_masses: (PyMass, PyMass),
    alternate_channel_daughter_masses: (f64, f64),
    resonance_mass: &PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(Flatte::new(
        name,
        mass.0,
        observed_channel_coupling.0,
        alternate_channel_coupling.0,
        (
            &observed_channel_daughter_masses.0 .0,
            &observed_channel_daughter_masses.1 .0,
        ),
        alternate_channel_daughter_masses,
        &resonance_mass.0,
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Mass};

    use super::*;

    #[test]
    fn test_flatte_evaluation() {
        let dataset = Arc::new(test_dataset());
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = Flatte::new(
            "flatte",
            parameter!("mass"),
            parameter!("g_obs"),
            parameter!("g_alt"),
            (&daughter_1_mass, &daughter_2_mass),
            (0.1349768, 0.547862),
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[0.98, 0.7, 0.2]).unwrap();

        assert_relative_eq!(result[0].re, -0.7338320342780681);
        assert_relative_eq!(result[0].im, 0.5018145529787819);
    }

    #[test]
    fn test_flatte_gradient() {
        let dataset = Arc::new(test_dataset());
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = Flatte::new(
            "flatte",
            parameter!("mass"),
            parameter!("g_obs"),
            parameter!("g_alt"),
            (&daughter_1_mass, &daughter_2_mass),
            (0.1349768, 0.547862),
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[0.98, 0.7, 0.2]).unwrap();

        assert_eq!(result[0].len(), 3);
        assert_relative_eq!(result[0][0].re, -0.08473788905152731);
        assert_relative_eq!(result[0][0].im, 1.6292790093139917);
        assert_relative_eq!(result[0][1].re, 0.497349582793617);
        assert_relative_eq!(result[0][1].im, 0.19360065665801518);
        assert_relative_eq!(result[0][2].re, 0.597447011338709);
        assert_relative_eq!(result[0][2].im, 0.23256505627570476);
    }
}
