use serde::{Deserialize, Serialize};

use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, Parameter},
    data::{DatasetMetadata, NamedEventView},
    resources::{Cache, ParameterID, Parameters, Resources},
    utils::{
        functions::{blatt_weisskopf_m, q_m, BarrierKind, Sheet, QR_DEFAULT},
        variables::{Mass, Variable},
    },
    LadduResult, ScalarID,
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

use crate::semantic_key::{debug_key, display_key, parameter_key};

/// A relativistic Breit-Wigner [`Amplitude`], parameterized as follows:
/// ```math
/// I_{\ell}(m; m_0, \Gamma_0, m_1, m_2) =  \frac{1}{(m_0^2 - m^2) - \imath m_0 \Gamma_{\ell}(m)}
/// ```
/// where
/// ```math
/// \Gamma_{\ell}(m) = \Gamma_0 \frac{m_0}{m} \frac{q(m, m_1, m_2)}{q(m_0, m_1, m_2)} \left(\frac{B_{\ell}(m, m_1, m_2)}{B_{\ell}(m_0, m_1, m_2)}\right)^2
/// ```
/// is the relativistic width correction, $`q(m_a, m_b, m_c)`$ is the breakup momentum of a particle with mass $`m_a`$ decaying into two particles with masses $`m_b`$ and $`m_c`$, $`B_{\ell}(m_a, m_b, m_c)`$ is the Blatt-Weisskopf barrier factor for the same decay assuming particle $`a`$ has angular momentum $`\ell`$, $`m_0`$ is the mass of the resonance, $`\Gamma_0`$ is the nominal width of the resonance, $`m_1`$ and $`m_2`$ are the masses of the decay products, and $`m`$ is the "input" mass.
///
/// This amplitude can also be constructed without barrier factors, in which case,
/// ```math
/// \Gamma_{\ell}(m) = \Gamma_0 \frac{m_0}{m} \left(\frac{q(m, m_1, m_2)}{q(m_0, m_1, m_2)}\right)^{2\ell + 1}.
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct BreitWigner {
    name: String,
    mass: Parameter,
    width: Parameter,
    pid_mass: ParameterID,
    pid_width: ParameterID,
    l: usize,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    barrier_factors: bool,
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
        mass: Parameter,
        width: Parameter,
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
            barrier_factors: true,
            daughter_1_mass_id: ScalarID::default(),
            daughter_2_mass_id: ScalarID::default(),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }
    /// Construct a [`BreitWigner`] with the given name, mass, width, and angular momentum (`l`).
    /// This uses the given `resonance_mass` as the "input" mass and two daughter masses of the
    /// decay products to determine phase-space. This constructor generates an amplitude which does
    /// not use Blatt-Weisskopf barrier factors.
    pub fn new_without_barrier_factors(
        name: &str,
        mass: Parameter,
        width: Parameter,
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
            barrier_factors: false,
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

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("BreitWigner")
                .with_field("name", debug_key(&self.name))
                .with_field("mass", parameter_key(&self.mass))
                .with_field("width", parameter_key(&self.width))
                .with_field("l", self.l.to_string())
                .with_field("daughter_1_mass", display_key(&self.daughter_1_mass))
                .with_field("daughter_2_mass", display_key(&self.daughter_2_mass))
                .with_field("resonance_mass", display_key(&self.resonance_mass))
                .with_field("barrier_factors", self.barrier_factors.to_string()),
        )
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
        let q0 = q_m(mass0, mass1, mass2, Sheet::Physical);
        let q = q_m(mass, mass1, mass2, Sheet::Physical);
        let width = if self.barrier_factors {
            let f0 = blatt_weisskopf_m(
                mass0,
                mass1,
                mass2,
                self.l,
                QR_DEFAULT,
                Sheet::Physical,
                BarrierKind::Full,
            );
            let f = blatt_weisskopf_m(
                mass,
                mass1,
                mass2,
                self.l,
                QR_DEFAULT,
                Sheet::Physical,
                BarrierKind::Full,
            );
            width0 * (mass0 / mass) * (q / q0) * (f / f0).powi(2)
        } else {
            width0 * (mass0 / mass) * (q / q0).powi((2 * self.l + 1) as i32)
        };
        1.0 / (Complex64::from(mass0.powi(2) - mass.powi(2)) - Complex64::I * mass0 * width)
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
/// mass : laddu.Parameter
///     The mass of the resonance
/// width : laddu.Parameter
///     The (nonrelativistic) width of the resonance
/// l : int
///     The total orbital momentum (:math:`l > 0`)
/// daughter_1_mass : laddu.Mass
///     The mass of the first decay product
/// daughter_2_mass : laddu.Mass
///     The mass of the second decay product
/// resonance_mass: laddu.Mass
///     The total mass of the resonance
/// barrier_factors : bool, default=True
///     If true, include Blatt-Weisskopf barrier factors
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "BreitWigner", signature = (name, mass, width, l, daughter_1_mass, daughter_2_mass, resonance_mass, barrier_factors=true))]
pub fn py_breit_wigner(
    name: &str,
    mass: PyParameter,
    width: PyParameter,
    l: usize,
    daughter_1_mass: &PyMass,
    daughter_2_mass: &PyMass,
    resonance_mass: &PyMass,
    barrier_factors: bool,
) -> PyResult<PyExpression> {
    if barrier_factors {
        Ok(PyExpression(BreitWigner::new(
            name,
            mass.0,
            width.0,
            l,
            &daughter_1_mass.0,
            &daughter_2_mass.0,
            &resonance_mass.0,
        )?))
    } else {
        Ok(PyExpression(BreitWigner::new_without_barrier_factors(
            name,
            mass.0,
            width.0,
            l,
            &daughter_1_mass.0,
            &daughter_2_mass.0,
            &resonance_mass.0,
        )?))
    }
}

/// A non-relativistic Breit-Wigner [`Amplitude`], parameterized as follows:
/// ```math
/// I(m; m_0, \Gamma_0) =  \frac{1}{(m_0^2 - m^2) - \imath m_0 \Gamma_0}
/// ```
/// where $`m_0`$ is the mass of the resonance, $`\Gamma_0`$ is the nominal width of the resonance, and $`m`$ is the "input" mass.
#[derive(Clone, Serialize, Deserialize)]
pub struct BreitWignerNonRelativistic {
    name: String,
    mass: Parameter,
    width: Parameter,
    pid_mass: ParameterID,
    pid_width: ParameterID,
    resonance_mass: Mass,
    resonance_mass_id: ScalarID,
}
impl BreitWignerNonRelativistic {
    /// Construct a [`BreitWignerNonRelativistic`] with the given name, mass, and width.
    ///
    /// This uses the given `resonance_mass` as the "input" mass.
    pub fn new(
        name: &str,
        mass: Parameter,
        width: Parameter,
        resonance_mass: &Mass,
    ) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            mass,
            width,
            pid_mass: ParameterID::default(),
            pid_width: ParameterID::default(),
            resonance_mass: resonance_mass.clone(),
            resonance_mass_id: ScalarID::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for BreitWignerNonRelativistic {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_mass = resources.register_parameter(&self.mass)?;
        self.pid_width = resources.register_parameter(&self.width)?;
        self.resonance_mass_id =
            resources.register_scalar(Some(&format!("{}.resonance_mass", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("BreitWignerNonRelativistic")
                .with_field("name", debug_key(&self.name))
                .with_field("mass", parameter_key(&self.mass))
                .with_field("width", parameter_key(&self.width))
                .with_field("resonance_mass", display_key(&self.resonance_mass)),
        )
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.resonance_mass.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        cache.store_scalar(self.resonance_mass_id, event.evaluate(&self.resonance_mass));
    }
    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let mass = cache.get_scalar(self.resonance_mass_id);
        let mass0 = parameters.get(self.pid_mass).abs();
        let width0 = parameters.get(self.pid_width).abs();
        1.0 / Complex64::new(mass0.powi(2) - mass.powi(2), -(mass0 * width0))
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

/// An non-relativistic Breit-Wigner Amplitude
///
/// This Amplitude represents a non-relativistic Breit-Wigner
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// mass : laddu.Parameter
///     The mass of the resonance
/// width : laddu.Parameter
///     The (nonrelativistic) width of the resonance
/// resonance_mass: laddu.Mass
///     The total mass of the resonance
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "BreitWignerNonRelativistic")]
pub fn py_breit_wigner_non_relativistic(
    name: &str,
    mass: PyParameter,
    width: PyParameter,
    resonance_mass: &PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(BreitWignerNonRelativistic::new(
        name,
        mass.0,
        width.0,
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

        assert_relative_eq!(result[0].re, 1.4308791652435877);
        assert_relative_eq!(result[0].im, 1.3839522217669178);
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
        assert_relative_eq!(result[0][0].re, -2.4885111876303205);
        assert_relative_eq!(result[0][0].im, -1.8242624730406152);
        assert_relative_eq!(result[0][1].re, -0.5492978554232557);
        assert_relative_eq!(result[0][1].im, 0.7828010830349043);
    }

    #[test]
    fn test_bw_no_bwbf_evaluation() {
        let dataset = Arc::new(test_dataset());
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = BreitWigner::new_without_barrier_factors(
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

        assert_relative_eq!(result[0].re, 2.0654840145948143);
        assert_relative_eq!(result[0].im, 1.2058262598870584);
    }

    #[test]
    fn test_bw_no_bwbf_gradient() {
        let dataset = Arc::new(test_dataset());
        let daughter_1_mass = Mass::new(["kshort1"]);
        let daughter_2_mass = Mass::new(["kshort2"]);
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = BreitWigner::new_without_barrier_factors(
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
        assert_relative_eq!(result[0][0].re, -3.2382865275566544);
        assert_relative_eq!(result[0][0].im, -0.9544869810033058);
        assert_relative_eq!(result[0][1].re, -0.06116353148223782);
        assert_relative_eq!(result[0][1].im, 0.31318991406841384);
    }

    #[test]
    fn test_bw_nonrel_evaluation() {
        let dataset = Arc::new(test_dataset());
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = BreitWignerNonRelativistic::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[1.5, 0.3]);

        assert_relative_eq!(result[0].re, 1.084721431628924);
        assert_relative_eq!(result[0].im, 1.3518336007116172);
    }

    #[test]
    fn test_bw_nonrel_gradient() {
        let dataset = Arc::new(test_dataset());
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let amp = BreitWignerNonRelativistic::new(
            "bw",
            parameter("mass"),
            parameter("width"),
            &resonance_mass,
        )
        .unwrap();
        let evaluator = amp.load(&dataset).unwrap();

        let result = evaluator.evaluate_gradient(&[1.7, 0.3]);
        assert_relative_eq!(result[0][0].re, -1.7757650016553739);
        assert_relative_eq!(result[0][0].im, -2.0392238297998153);
        assert_relative_eq!(result[0][1].re, -1.0894724338203443);
        assert_relative_eq!(result[0][1].im, 0.7917525805669601);
    }
}
