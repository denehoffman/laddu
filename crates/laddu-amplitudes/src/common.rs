use laddu_core::{
    amplitudes::{
        Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, ExpressionDependence,
        ParameterLike,
    },
    data::{DatasetMetadata, NamedEventView},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    LadduResult, ScalarID,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyExpression, PyParameterLike},
    utils::variables::PyVariable,
};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::semantic_key::{debug_key, display_key, parameter_key};

/// A scalar-valued [`Amplitude`] which just contains a single parameter as its value.
#[derive(Clone, Serialize, Deserialize)]
pub struct Scalar {
    name: String,
    value: ParameterLike,
    pid: ParameterID,
}

impl Scalar {
    /// Create a new [`Scalar`] with the given name and parameter value.
    pub fn new(name: &str, value: ParameterLike) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            value,
            pid: Default::default(),
        }
        .into_expression()
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
        _parameters: &Parameters,
        _cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let ParameterID::Parameter(ind) = self.pid {
            gradient[ind] = Complex64::ONE;
        }
    }
}

/// An Amplitude which represents a single scalar value
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// value : laddu.ParameterLike
///     The scalar parameter contained in the Amplitude
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "Scalar")]
pub fn py_scalar(name: &str, value: PyParameterLike) -> PyResult<PyExpression> {
    Ok(PyExpression(Scalar::new(name, value.0)?))
}

/// A real-valued [`Amplitude`] which evaluates an event [`Variable`].
#[derive(Clone, Serialize, Deserialize)]
pub struct VariableScalar {
    name: String,
    variable: Box<dyn Variable>,
    value_id: ScalarID,
}

impl VariableScalar {
    /// Create a new [`VariableScalar`] that evaluates `variable` on each event.
    pub fn new<V: Variable + 'static>(name: &str, variable: &V) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            variable: dyn_clone::clone_box(variable),
            value_id: ScalarID::default(),
        }
        .into_expression()
    }
}

/// Extension methods for building expressions from event [`Variable`]s.
pub trait VariableExpressionExt: Variable + 'static {
    /// Convert this variable into a real-valued [`Expression`].
    fn as_expression(&self, name: &str) -> LadduResult<Expression>
    where
        Self: Sized,
    {
        VariableScalar::new(name, self)
    }
}

impl<T: Variable + 'static> VariableExpressionExt for T {}

#[typetag::serde]
impl Amplitude for VariableScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.value_id = resources.register_scalar(Some(&format!("{}.value", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("VariableScalar")
                .with_field("name", debug_key(&self.name))
                .with_field("variable", display_key(&self.variable)),
        )
    }

    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::CacheOnly
    }

    fn real_valued_hint(&self) -> bool {
        true
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.variable.bind(metadata)
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        cache.store_scalar(self.value_id, self.variable.value(event));
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_scalar(self.value_id).into()
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
}

/// An amplitude which evaluates a Variable as a real-valued Expression.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variable : laddu.Mass | laddu.CosTheta | laddu.Phi | laddu.PolAngle | laddu.PolMagnitude | laddu.Mandelstam
///     The event Variable to evaluate.
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates to the Variable value on each event.
///
#[cfg(feature = "python")]
#[pyfunction(name = "VariableScalar")]
pub fn py_variable_scalar(name: &str, variable: Bound<'_, PyAny>) -> PyResult<PyExpression> {
    let variable = variable.extract::<PyVariable>()?;
    Ok(PyExpression(VariableScalar::new(name, &variable)?))
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
    pub fn new(name: &str, re: ParameterLike, im: ParameterLike) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            re,
            pid_re: Default::default(),
            im,
            pid_im: Default::default(),
        }
        .into_expression()
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
        _parameters: &Parameters,
        _cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        if let ParameterID::Parameter(ind) = self.pid_re {
            gradient[ind] = Complex64::ONE;
        }
        if let ParameterID::Parameter(ind) = self.pid_im {
            gradient[ind] = Complex64::I;
        }
    }
}

/// An Amplitude which represents a complex value
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// re: laddu.ParameterLike
///     The real part of the complex value contained in the Amplitude
/// im: laddu.ParameterLike
///     The imaginary part of the complex value contained in the Amplitude
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "ComplexScalar")]
pub fn py_complex_scalar(
    name: &str,
    re: PyParameterLike,
    im: PyParameterLike,
) -> PyResult<PyExpression> {
    Ok(PyExpression(ComplexScalar::new(name, re.0, im.0)?))
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
    pub fn new(name: &str, r: ParameterLike, theta: ParameterLike) -> LadduResult<Expression> {
        Self {
            name: name.to_string(),
            r,
            pid_r: Default::default(),
            theta,
            pid_theta: Default::default(),
        }
        .into_expression()
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
        if let ParameterID::Parameter(ind) = self.pid_r {
            gradient[ind] = exp_i_theta;
        }
        if let ParameterID::Parameter(ind) = self.pid_theta {
            gradient[ind] = Complex64::I
                * Complex64::from_polar(parameters.get(self.pid_r), parameters.get(self.pid_theta));
        }
    }
}

/// An Amplitude which represents a complex scalar value in polar form
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// r: laddu.ParameterLike
///     The magnitude of the complex value contained in the Amplitude
/// theta: laddu.ParameterLike
///     The argument of the complex value contained in the Amplitude
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
///
#[cfg(feature = "python")]
#[pyfunction(name = "PolarComplexScalar")]
pub fn py_polar_complex_scalar(
    name: &str,
    r: PyParameterLike,
    theta: PyParameterLike,
) -> PyResult<PyExpression> {
    Ok(PyExpression(PolarComplexScalar::new(name, r.0, theta.0)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{
        amplitudes::ExpressionDependence, data::test_dataset, parameter, utils::variables::Mass, PI,
    };
    use std::f64;
    use std::sync::Arc;

    #[test]
    fn test_scalar_creation_and_evaluation() {
        let dataset = Arc::new(test_dataset());
        let expr = Scalar::new("test_scalar", parameter("test_param")).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![2.5];
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 2.5);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_scalar_gradient() {
        let dataset = Arc::new(test_dataset());
        let expr = Scalar::new("test_scalar", parameter("test_param"))
            .unwrap()
            .norm_sqr(); // |f(x)|^2
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![2.0];
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x)|^2 where f(x) = x, the derivative should be 2x
        assert_relative_eq!(gradient[0][0].re, 4.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
    }

    #[test]
    fn test_scalar_reports_real_valued_hint() {
        let scalar = Scalar {
            name: "test_scalar".to_string(),
            value: parameter("test_param"),
            pid: Default::default(),
        };
        let complex = ComplexScalar {
            name: "test_complex".to_string(),
            re: parameter("re_param"),
            pid_re: Default::default(),
            im: parameter("im_param"),
            pid_im: Default::default(),
        };

        assert!(Amplitude::real_valued_hint(&scalar));
        assert!(!Amplitude::real_valued_hint(&complex));
    }

    #[test]
    fn test_variable_scalar_evaluation() {
        let dataset = Arc::new(test_dataset());
        let mut variable = Mass::new(["kshort1", "kshort2"]);
        variable.bind(dataset.metadata()).unwrap();
        let expected = variable.value(&dataset.event_view(0));

        let expr = VariableScalar::new("mass", &variable).unwrap();
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, expected);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_variable_scalar_is_cache_only_and_real() {
        let amplitude = VariableScalar {
            name: "mass".to_string(),
            variable: Box::new(Mass::new(["kshort1", "kshort2"])),
            value_id: ScalarID::default(),
        };

        assert_eq!(amplitude.dependence_hint(), ExpressionDependence::CacheOnly);
        assert!(Amplitude::real_valued_hint(&amplitude));
    }

    #[test]
    fn test_variable_scalar_has_no_parameters() {
        let dataset = Arc::new(test_dataset());
        let variable = Mass::new(["kshort1", "kshort2"]);
        let expr = VariableScalar::new("mass", &variable).unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        assert!(evaluator.parameters().is_empty());
        assert!(evaluator.free_parameters().is_empty());
        assert!(evaluator.fixed_parameters().is_empty());
    }

    #[test]
    fn test_variable_as_expression() {
        let dataset = Arc::new(test_dataset());
        let mut variable = Mass::new(["kshort1", "kshort2"]);
        variable.bind(dataset.metadata()).unwrap();
        let expected = variable.value(&dataset.event_view(0));

        let expr = variable.as_expression("mass").unwrap();
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, expected);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_complex_scalar_evaluation() {
        let dataset = Arc::new(test_dataset());
        let expr = ComplexScalar::new("test_complex", parameter("re_param"), parameter("im_param"))
            .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![1.5, 2.5]; // Real and imaginary parts
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 1.5);
        assert_relative_eq!(result[0].im, 2.5);
    }

    #[test]
    fn test_complex_scalar_gradient() {
        let dataset = Arc::new(test_dataset());
        let expr = ComplexScalar::new("test_complex", parameter("re_param"), parameter("im_param"))
            .unwrap()
            .norm_sqr(); // |f(x + iy)|^2
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![3.0, 4.0]; // Real and imaginary parts
        let gradient = evaluator.evaluate_gradient(&params);

        // For |f(x + iy)|^2, partial derivatives should be 2x and 2y
        assert_relative_eq!(gradient[0][0].re, 6.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 8.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
    }

    #[test]
    fn test_semantic_key_deduplicates_matching_complex_scalar() {
        let dataset = Arc::new(test_dataset());
        let expr = ComplexScalar::new("same_complex", parameter("re_param"), parameter("im_param"))
            .unwrap()
            + ComplexScalar::new("same_complex", parameter("re_param"), parameter("im_param"))
                .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let result = evaluator.evaluate(&[1.5, 2.5]);

        assert_eq!(evaluator.amplitudes.len(), 1);
        assert_relative_eq!(result[0].re, 3.0);
        assert_relative_eq!(result[0].im, 5.0);
    }

    #[test]
    #[should_panic(expected = "re differs")]
    fn test_semantic_key_reports_mismatched_complex_scalar_field() {
        let _expr =
            ComplexScalar::new("same_complex", parameter("re_param"), parameter("im_param"))
                .unwrap()
                + ComplexScalar::new("same_complex", parameter("other_re"), parameter("im_param"))
                    .unwrap();
    }

    #[test]
    fn test_polar_complex_scalar_evaluation() {
        let dataset = Arc::new(test_dataset());
        let expr =
            PolarComplexScalar::new("test_polar", parameter("r_param"), parameter("theta_param"))
                .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let r = 2.0;
        let theta = PI / 4.3;
        let params = vec![r, theta];
        let result = evaluator.evaluate(&params);

        // r * (cos(theta) + i*sin(theta))
        assert_relative_eq!(result[0].re, r * theta.cos());
        assert_relative_eq!(result[0].im, r * theta.sin());
    }

    #[test]
    fn test_polar_complex_scalar_gradient() {
        let dataset = Arc::new(test_dataset());
        let expr =
            PolarComplexScalar::new("test_polar", parameter("r_param"), parameter("theta_param"))
                .unwrap();
        let evaluator = expr.load(&dataset).unwrap();

        let r = 2.0;
        let theta = PI / 4.3;
        let params = vec![r, theta];
        let gradient = evaluator.evaluate_gradient(&params);

        // d/dr re^(iθ) = e^(iθ), d/dθ re^(iθ) = ire^(iθ)
        assert_relative_eq!(gradient[0][0].re, f64::cos(theta));
        assert_relative_eq!(gradient[0][0].im, f64::sin(theta));
        assert_relative_eq!(gradient[0][1].re, -r * f64::sin(theta));
        assert_relative_eq!(gradient[0][1].im, r * f64::cos(theta));
    }
}
