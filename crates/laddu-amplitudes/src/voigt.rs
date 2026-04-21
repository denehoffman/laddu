use std::f64::consts::SQRT_2;

use errorfunctions::ComplexErrorFunctions;
use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, Parameter},
    data::{DatasetMetadata, NamedEventView},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    LadduResult, Mass, ScalarID,
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

const SQRT_2PI: f64 = 2.5066282746310002;

/// A Voigt line-shape [`Amplitude`].
///
/// The normalized profile is
/// ```math
/// V(m; m_0, \Gamma, \sigma) =
/// \frac{\operatorname{Re}[w(z)]}{\sigma \sqrt{2\pi}},
/// \qquad
/// z = \frac{m - m_0 + i\Gamma/2}{\sigma\sqrt{2}},
/// ```
/// where `$w(z)$` is the Faddeeva function. Here `$\Gamma$` is the Lorentzian full
/// width at half maximum and `$\sigma$` is the Gaussian resolution width. This amplitude
/// returns `$\sqrt{V}$` so its norm-squared is the normalized Voigt profile.
#[derive(Clone, Serialize, Deserialize)]
pub struct Voigt {
    name: String,
    mass: Parameter,
    width: Parameter,
    sigma: Parameter,
    pid_mass: ParameterID,
    pid_width: ParameterID,
    pid_sigma: ParameterID,
    resonance_mass: Mass,
    resonance_mass_id: ScalarID,
}

impl Voigt {
    /// Construct a [`Voigt`] with the given name, mass, width, and Gaussian sigma.
    ///
    /// This uses the given `resonance_mass` as the input mass. `width` is the Lorentzian full
    /// width at half maximum and `sigma` is the Gaussian resolution width.
    pub fn new(
        name: &str,
        mass: Parameter,
        width: Parameter,
        sigma: Parameter,
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            mass,
            width,
            sigma,
            pid_mass: ParameterID::default(),
            pid_width: ParameterID::default(),
            pid_sigma: ParameterID::default(),
            resonance_mass: resonance_mass.clone(),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }

    fn profile(input_mass: f64, mass: f64, width: f64, sigma: f64) -> f64 {
        if sigma <= 0.0 || width < 0.0 {
            return 0.0;
        }
        let z = Complex64::new(input_mass - mass, 0.5 * width) / (sigma * SQRT_2);
        (z.w().re / (sigma * SQRT_2PI)).max(0.0)
    }
}

#[typetag::serde]
impl Amplitude for Voigt {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass)?;
        self.pid_width = resources.register_parameter(&self.width)?;
        self.pid_sigma = resources.register_parameter(&self.sigma)?;
        self.resonance_mass_id =
            resources.register_scalar(Some(&format!("{}.resonance_mass", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("Voigt")
                .with_field("name", debug_key(&self.name))
                .with_field("mass", parameter_key(&self.mass))
                .with_field("width", parameter_key(&self.width))
                .with_field("sigma", parameter_key(&self.sigma))
                .with_field("resonance_mass", display_key(&self.resonance_mass)),
        )
    }

    fn real_valued_hint(&self) -> bool {
        true
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        cache.store_scalar(self.resonance_mass_id, event.evaluate(&self.resonance_mass));
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let input_mass = cache.get_scalar(self.resonance_mass_id);
        let mass = parameters.get(self.pid_mass).abs();
        let width = parameters.get(self.pid_width).abs();
        let sigma = parameters.get(self.pid_sigma).abs();
        let profile = Self::profile(input_mass, mass, width, sigma);
        profile.sqrt().into()
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
        if let ParameterID::Parameter(index) = self.pid_sigma {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
    }
}

/// A Voigt line-shape amplitude
///
/// This amplitude represents the square root of a normalized Voigt profile.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// mass : laddu.Parameter
///     The central mass of the resonance
/// width : laddu.Parameter
///     The Lorentzian full width at half maximum
/// sigma : laddu.Parameter
///     The Gaussian resolution width
/// resonance_mass : laddu.Mass
///     The event-dependent input mass
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "Voigt")]
pub fn py_voigt(
    name: &str,
    mass: PyParameter,
    width: PyParameter,
    sigma: PyParameter,
    resonance_mass: &PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(Voigt::new(
        name,
        mass.0,
        width.0,
        sigma.0,
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
    fn test_voigt_sqrt_profile_evaluation() {
        let dataset = Arc::new(test_dataset());
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = Voigt::new(
            "voigt",
            parameter!("mass"),
            parameter!("width"),
            parameter!("sigma"),
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[0.98, 0.08, 0.02]);

        assert_relative_eq!(result[0].re, 0.2857389147779551);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_voigt_gradient() {
        let dataset = Arc::new(test_dataset());
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = Voigt::new(
            "voigt",
            parameter!("mass"),
            parameter!("width"),
            parameter!("sigma"),
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[0.98, 0.08, 0.02]);
        assert_relative_eq!(result[0][0].re, 0.7225730704295464);
        assert_relative_eq!(result[0][0].im, 0.0);
        assert_relative_eq!(result[0][1].re, 1.7488427782862053);
        assert_relative_eq!(result[0][1].im, 0.0);
        assert_relative_eq!(result[0][2].re, 0.10952492310922711);
        assert_relative_eq!(result[0][2].im, 0.0);
    }
}
