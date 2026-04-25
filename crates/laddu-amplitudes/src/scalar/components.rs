use laddu_core::{
    amplitudes::{
        debug_key, parameter_key, Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression,
        Parameter,
    },
    parameter,
    resources::{Cache, ParameterID, Parameters, Resources},
    LadduResult,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// A scalar-valued [`Amplitude`] which just contains a single parameter as its value.
#[derive(Clone, Serialize, Deserialize)]
pub struct Scalar {
    name: String,
    value: Parameter,
    pid: ParameterID,
}

impl Scalar {
    /// Create a new [`Scalar`] with the given name and parameter value.
    pub fn new(name: &str, value: Parameter) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            value,
            pid: Default::default(),
        }
        .into_expression()
    }

    /// Create a new [`Scalar`] with the given name and a parameter value with the same name.
    pub fn new_auto(name: &str) -> LadduResult<Expression> {
        Self::new(name, parameter!(name))
    }
}

#[typetag::serde]
impl Amplitude for Scalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid = resources.register_parameter(&self.value)?;
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("Scalar")
                .with_field("name", debug_key(&self.name))
                .with_field("value", parameter_key(&self.value)),
        )
    }

    fn real_valued_hint(&self) -> bool {
        true
    }

    fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid), 0.0)
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        _cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let Some(ind) = parameters.free_index(self.pid) {
            gradient[ind] = Complex64::ONE;
        }
    }
}

/// A complex-valued [`Amplitude`] which just contains two parameters representing its real and
/// imaginary parts.
#[derive(Clone, Serialize, Deserialize)]
pub struct ComplexScalar {
    name: String,
    re: Parameter,
    pid_re: ParameterID,
    im: Parameter,
    pid_im: ParameterID,
}

impl ComplexScalar {
    /// Create a new [`ComplexScalar`] with the given name, real, and imaginary part.
    pub fn new(name: &str, re: Parameter, im: Parameter) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            re,
            pid_re: Default::default(),
            im,
            pid_im: Default::default(),
        }
        .into_expression()
    }

    /// Create a new [`ComplexScalar`] with the given name and real/imaginary parts automatically
    /// named "{name} (real)" and "{name} (imag)", respectively.
    pub fn new_auto(name: &str) -> LadduResult<Expression> {
        Self::new(
            name,
            parameter!(format!("{} (real)", name)),
            parameter!(format!("{} (imag)", name)),
        )
    }
}

#[typetag::serde]
impl Amplitude for ComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_re = resources.register_parameter(&self.re)?;
        self.pid_im = resources.register_parameter(&self.im)?;
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("ComplexScalar")
                .with_field("name", debug_key(&self.name))
                .with_field("re", parameter_key(&self.re))
                .with_field("im", parameter_key(&self.im)),
        )
    }

    fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
        Complex64::new(parameters.get(self.pid_re), parameters.get(self.pid_im))
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        _cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let Some(ind) = parameters.free_index(self.pid_re) {
            gradient[ind] = Complex64::ONE;
        }
        if let Some(ind) = parameters.free_index(self.pid_im) {
            gradient[ind] = Complex64::I;
        }
    }
}

/// A complex-valued [`Amplitude`] which just contains two parameters representing its magnitude and
/// phase.
#[derive(Clone, Serialize, Deserialize)]
pub struct PolarComplexScalar {
    name: String,
    r: Parameter,
    pid_r: ParameterID,
    theta: Parameter,
    pid_theta: ParameterID,
}

impl PolarComplexScalar {
    /// Create a new [`PolarComplexScalar`] with the given name, magnitude (`r`), and phase (`theta`).
    pub fn new(name: &str, r: Parameter, theta: Parameter) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            r,
            pid_r: Default::default(),
            theta,
            pid_theta: Default::default(),
        }
        .into_expression()
    }

    /// Create a new [`PolarComplexScalar`] with the given name and magnitude/phase parts
    /// automatically named "{name} (mag)" and "{name} (phase)", respectively.
    pub fn new_auto(name: &str) -> LadduResult<Expression> {
        Self::new(
            name,
            parameter!(format!("{} (mag)", name)),
            parameter!(format!("{} (phase)", name)),
        )
    }
}

#[typetag::serde]
impl Amplitude for PolarComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pid_r = resources.register_parameter(&self.r)?;
        self.pid_theta = resources.register_parameter(&self.theta)?;
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("PolarComplexScalar")
                .with_field("name", debug_key(&self.name))
                .with_field("r", parameter_key(&self.r))
                .with_field("theta", parameter_key(&self.theta)),
        )
    }

    fn compute(&self, parameters: &Parameters, _cache: &Cache) -> Complex64 {
        Complex64::from_polar(parameters.get(self.pid_r), parameters.get(self.pid_theta))
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        _cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let exp_i_theta = Complex64::cis(parameters.get(self.pid_theta));
        if let Some(ind) = parameters.free_index(self.pid_r) {
            gradient[ind] = exp_i_theta;
        }
        if let Some(ind) = parameters.free_index(self.pid_theta) {
            gradient[ind] = Complex64::I
                * Complex64::from_polar(parameters.get(self.pid_r), parameters.get(self.pid_theta));
        }
    }
}
