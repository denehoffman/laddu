use nalgebra::DVector;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::{
    amplitudes::{AmplitudeID, ParameterLike},
    data::Event,
    resources::{Cache, ParameterID, Parameters, Resources},
    Float, LadduError,
};

use super::Amplitude;

/// A scalar-valued [`Amplitude`] which just contains a single parameter as its value.
#[derive(Clone, Serialize, Deserialize)]
pub struct Scalar {
    name: String,
    value: ParameterLike,
    pid: ParameterID,
}

impl Scalar {
    /// Create a new [`Scalar`] with the given name and parameter value.
    pub fn new(name: &str, value: ParameterLike) -> Box<Self> {
        Self {
            name: name.to_string(),
            value,
            pid: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for Scalar {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        self.pid = resources.register_parameter(&self.value);
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, _event: &Event, _cache: &Cache) -> Complex<Float> {
        Complex::new(parameters.get(self.pid), 0.0)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &Event,
        _cache: &Cache,
        gradient: &mut DVector<Complex<Float>>,
    ) {
        if let ParameterID::Parameter(ind) = self.pid {
            gradient[ind] = Complex::ONE;
        }
    }
}

/// A complex-valued [`Amplitude`] which just contains two parameters representing its real and
/// imaginary parts.
#[derive(Clone, Serialize, Deserialize)]
pub struct ComplexScalar {
    name: String,
    re: ParameterLike,
    pid_re: ParameterID,
    im: ParameterLike,
    pid_im: ParameterID,
}

impl ComplexScalar {
    /// Create a new [`ComplexScalar`] with the given name, real, and imaginary part.
    pub fn new(name: &str, re: ParameterLike, im: ParameterLike) -> Box<Self> {
        Self {
            name: name.to_string(),
            re,
            pid_re: Default::default(),
            im,
            pid_im: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for ComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        self.pid_re = resources.register_parameter(&self.re);
        self.pid_im = resources.register_parameter(&self.im);
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, _event: &Event, _cache: &Cache) -> Complex<Float> {
        Complex::new(parameters.get(self.pid_re), parameters.get(self.pid_im))
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &Event,
        _cache: &Cache,
        gradient: &mut DVector<Complex<Float>>,
    ) {
        if let ParameterID::Parameter(ind) = self.pid_re {
            gradient[ind] = Complex::ONE;
        }
        if let ParameterID::Parameter(ind) = self.pid_im {
            gradient[ind] = Complex::I;
        }
    }
}

/// A complex-valued [`Amplitude`] which just contains two parameters representing its magnitude and
/// phase.
#[derive(Clone, Serialize, Deserialize)]
pub struct PolarComplexScalar {
    name: String,
    r: ParameterLike,
    pid_r: ParameterID,
    theta: ParameterLike,
    pid_theta: ParameterID,
}

impl PolarComplexScalar {
    /// Create a new [`PolarComplexScalar`] with the given name, magnitude (`r`), and phase (`theta`).
    pub fn new(name: &str, r: ParameterLike, theta: ParameterLike) -> Box<Self> {
        Self {
            name: name.to_string(),
            r,
            pid_r: Default::default(),
            theta,
            pid_theta: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for PolarComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        self.pid_r = resources.register_parameter(&self.r);
        self.pid_theta = resources.register_parameter(&self.theta);
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, _event: &Event, _cache: &Cache) -> Complex<Float> {
        Complex::from_polar(parameters.get(self.pid_r), parameters.get(self.pid_theta))
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        _event: &Event,
        _cache: &Cache,
        gradient: &mut DVector<Complex<Float>>,
    ) {
        let exp_i_theta = Complex::cis(parameters.get(self.pid_theta));
        if let ParameterID::Parameter(ind) = self.pid_r {
            gradient[ind] = exp_i_theta;
        }
        if let ParameterID::Parameter(ind) = self.pid_theta {
            gradient[ind] = Complex::<Float>::I
                * Complex::from_polar(parameters.get(self.pid_r), parameters.get(self.pid_theta));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amplitudes::{parameter, Manager};
    use crate::data::test_dataset;
    use crate::PI;
    use approx::assert_relative_eq;
    use std::sync::Arc;

    #[test]
    fn test_scalar_creation_and_evaluation() {
        let mut manager = Manager::default();
        let amp = Scalar::new("test_scalar", parameter("test_param"));
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into(); // Direct amplitude evaluation
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![2.5];
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 2.5);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_scalar_gradient() {
        let mut manager = Manager::default();
        let amp = Scalar::new("test_scalar", parameter("test_param"));
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.norm_sqr(); // |f(x)|^2
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![2.0];
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x)|^2 where f(x) = x, the derivative should be 2x
        assert_relative_eq!(gradient[0][0].re, 4.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
    }

    #[test]
    fn test_complex_scalar_evaluation() {
        let mut manager = Manager::default();
        let amp = ComplexScalar::new("test_complex", parameter("re_param"), parameter("im_param"));
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![1.5, 2.5]; // Real and imaginary parts
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 1.5);
        assert_relative_eq!(result[0].im, 2.5);
    }

    #[test]
    fn test_complex_scalar_gradient() {
        let mut manager = Manager::default();
        let amp = ComplexScalar::new("test_complex", parameter("re_param"), parameter("im_param"));
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.norm_sqr(); // |f(x + iy)|^2
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let params = vec![3.0, 4.0]; // Real and imaginary parts
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x + iy)|^2, partial derivatives should be 2x and 2y
        assert_relative_eq!(gradient[0][0].re, 6.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 8.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
    }

    #[test]
    fn test_polar_complex_scalar_evaluation() {
        let mut manager = Manager::default();
        let amp =
            PolarComplexScalar::new("test_polar", parameter("r_param"), parameter("theta_param"));
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let r = 2.0;
        let theta = PI / 4.0;
        let params = vec![r, theta];
        let result = evaluator.evaluate(&params);

        // r * (cos(theta) + i*sin(theta))
        assert_relative_eq!(result[0].re, r * theta.cos());
        assert_relative_eq!(result[0].im, r * theta.sin());
    }

    #[test]
    fn test_polar_complex_scalar_gradient() {
        let mut manager = Manager::default();
        let amp =
            PolarComplexScalar::new("test_polar", parameter("r_param"), parameter("theta_param"));
        let aid = manager.register(amp).unwrap();

        let dataset = Arc::new(test_dataset());
        let expr = aid.into(); // f(r,θ) = re^(iθ)
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset);

        let r = 2.0;
        let theta = PI / 4.0;
        let params = vec![r, theta];
        let gradient = evaluator.evaluate_gradient(&params);

        // d/dr re^(iθ) = e^(iθ), d/dθ re^(iθ) = ire^(iθ)
        assert_relative_eq!(gradient[0][0].re, Float::cos(theta));
        assert_relative_eq!(gradient[0][0].im, Float::sin(theta));
        assert_relative_eq!(gradient[0][1].re, -r * Float::sin(theta));
        assert_relative_eq!(gradient[0][1].im, r * Float::cos(theta));
    }
}
