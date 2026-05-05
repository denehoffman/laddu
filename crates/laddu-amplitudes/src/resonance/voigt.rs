use std::f64::consts::SQRT_2;

use errorfunctions::ComplexErrorFunctions;
use laddu_core::{
    amplitudes::{
        debug_key, display_key, parameter_key, Amplitude, AmplitudeID, AmplitudeSemanticKey,
        Expression, Parameter,
    },
    data::{DatasetMetadata, Event},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    variables::Mass,
    LadduResult, ScalarID,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

const SQRT_2PI: f64 = 2.5066282746310002;

/// A Voigt line-shape [`Amplitude`].
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
    /// Construct a [`Voigt`] with the given mass, width, and Gaussian sigma.
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

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
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
        if let Some(index) = parameters.free_index(self.pid_mass) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
        if let Some(index) = parameters.free_index(self.pid_width) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
        if let Some(index) = parameters.free_index(self.pid_sigma) {
            self.central_difference_with_indices(&[index], parameters, cache, gradient);
        }
    }
}
