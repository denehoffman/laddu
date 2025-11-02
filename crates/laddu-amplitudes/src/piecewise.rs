use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, ParameterLike},
    resources::{CacheRow, ParameterID, Parameters, Resources},
    utils::get_bin_index_polars,
    ExprID, LadduError, LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::amplitudes::{PyAmplitude, PyParameterLike};
use nalgebra::DVector;
use num::complex::Complex64;
use polars::prelude::{DataType, Expr};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PyExpr;
use serde::{Deserialize, Serialize};

/// A piecewise scalar-valued [`Amplitude`] which just contains a single parameter for each bin as its value.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewiseScalar {
    name: String,
    expr: Expr,
    bins: usize,
    limits: (f64, f64),
    values: Vec<ParameterLike>,
    pids: Vec<ParameterID>,
    bin_index: ExprID,
}
impl PiecewiseScalar {
    /// Create a new [`PiecewiseScalar`] with the given name and parameter value.
    pub fn new(
        name: &str,
        expr: &Expr,
        bins: usize,
        limits: (f64, f64),
        values: Vec<ParameterLike>,
    ) -> Box<Self> {
        assert_eq!(
            bins,
            values.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            expr: expr.clone(),
            bins,
            limits,
            values,
            pids: Default::default(),
            bin_index: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for PiecewiseScalar {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        self.pids = self
            .values
            .iter()
            .map(|value| resources.register_parameter(value))
            .collect();
        let bin_index_expr =
            get_bin_index_polars(self.expr.clone(), self.bins, self.limits).cast(DataType::Float64);
        self.bin_index = resources.register_scalar(
            format!(
                "piecewise_bin_index_{}",
                self.expr.clone().meta().output_name()?
            )
            .into(),
            bin_index_expr,
        )?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, cache_row: &CacheRow) -> Complex64 {
        let bin_index: usize = cache_row.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins {
            Complex64::ZERO
        } else {
            Complex64::from(parameters.get(self.pids[bin_index]))
        }
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        cache_row: &CacheRow,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache_row.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins {
            gradient[bin_index] = Complex64::ONE;
        }
    }
}

/// An Amplitude which represents a piecewise function of single scalar values
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
///     The variable to use for binning
/// bins: usize
///     The number of bins to use
/// range: tuple of float
///     The minimum and maximum bin edges
/// values : list of ParameterLike
///     The scalar parameters contained in each bin of the Amplitude
///
/// Returns
/// -------
/// laddu.Amplitude
///     An Amplitude which can be registered by a laddu.Manager
///
/// Raises
/// ------
/// AssertionError
///     If the number of bins does not match the number of parameters
/// TypeError
///     If the given `variable` is not a valid variable
///
/// See Also
/// --------
/// laddu.Manager
/// laddu.Mass
/// laddu.CosTheta
/// laddu.Phi
/// laddu.PolAngle
/// laddu.PolMagnitude
/// laddu.Mandelstam
///
#[cfg(feature = "python")]
#[pyfunction(name = "PiecewiseScalar")]
pub fn py_piecewise_scalar(
    name: &str,
    variable: Bound<'_, PyAny>,
    bins: usize,
    limits: (f64, f64),
    values: Vec<PyParameterLike>,
) -> PyResult<PyAmplitude> {
    let expr = variable.extract::<PyExpr>()?;
    Ok(PyAmplitude(PiecewiseScalar::new(
        name,
        &expr.0,
        bins,
        limits,
        values.into_iter().map(|value| value.0).collect(),
    )))
}

/// A piecewise complex-valued [`Amplitude`] which just contains two parameters representing its real and
/// imaginary parts.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewiseComplexScalar {
    name: String,
    expr: Expr,
    bins: usize,
    limits: (f64, f64),
    re_ims: Vec<(ParameterLike, ParameterLike)>,
    pids_re_im: Vec<(ParameterID, ParameterID)>,
    bin_index: ExprID,
}
impl PiecewiseComplexScalar {
    /// Create a new [`PiecewiseComplexScalar`] with the given name and parameter value.
    pub fn new(
        name: &str,
        expr: &Expr,
        bins: usize,
        limits: (f64, f64),
        re_ims: Vec<(ParameterLike, ParameterLike)>,
    ) -> Box<Self> {
        assert_eq!(
            bins,
            re_ims.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            expr: expr.clone(),
            bins,
            limits,
            re_ims,
            pids_re_im: Default::default(),
            bin_index: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for PiecewiseComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pids_re_im = self
            .re_ims
            .iter()
            .map(|(re, im)| {
                (
                    resources.register_parameter(re),
                    resources.register_parameter(im),
                )
            })
            .collect();
        let bin_index_expr =
            get_bin_index_polars(self.expr.clone(), self.bins, self.limits).cast(DataType::Float64);
        self.bin_index = resources.register_scalar(
            format!(
                "piecewise_bin_index_{}",
                self.expr.clone().meta().output_name()?
            )
            .into(),
            bin_index_expr,
        )?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, cache_row: &CacheRow) -> Complex64 {
        let bin_index: usize = cache_row.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins {
            Complex64::ZERO
        } else {
            let pid_re_im = self.pids_re_im[bin_index];
            Complex64::new(parameters.get(pid_re_im.0), parameters.get(pid_re_im.1))
        }
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        cache_row: &CacheRow,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache_row.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins {
            let pid_re_im = self.pids_re_im[bin_index];
            if let ParameterID::Parameter(ind) = pid_re_im.0 {
                gradient[ind] = Complex64::ONE;
            }
            if let ParameterID::Parameter(ind) = pid_re_im.1 {
                gradient[ind] = Complex64::I;
            }
        }
    }
}

/// An Amplitude which represents a piecewise function of complex values
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
///     The variable to use for binning
/// bins: usize
///     The number of bins to use
/// range: tuple of float
///     The minimum and maximum bin edges
/// values : list of tuple of ParameterLike
///     The complex parameters contained in each bin of the Amplitude (each tuple contains the
///     real and imaginary part of a single bin)
///
/// Returns
/// -------
/// laddu.Amplitude
///     An Amplitude which can be registered by a laddu.Manager
///
/// Raises
/// ------
/// AssertionError
///     If the number of bins does not match the number of parameters
/// TypeError
///     If the given `variable` is not a valid variable
///
/// See Also
/// --------
/// laddu.Manager
/// laddu.Mass
/// laddu.CosTheta
/// laddu.Phi
/// laddu.PolAngle
/// laddu.PolMagnitude
/// laddu.Mandelstam
///
#[cfg(feature = "python")]
#[pyfunction(name = "PiecewiseComplexScalar")]
pub fn py_piecewise_complex_scalar(
    name: &str,
    expr: Bound<'_, PyAny>,
    bins: usize,
    limits: (f64, f64),
    values: Vec<(PyParameterLike, PyParameterLike)>,
) -> PyResult<PyAmplitude> {
    let expr = expr.extract::<PyExpr>()?;
    Ok(PyAmplitude(PiecewiseComplexScalar::new(
        name,
        &expr.0,
        bins,
        limits,
        values
            .into_iter()
            .map(|(value_re, value_im)| (value_re.0, value_im.0))
            .collect(),
    )))
}

/// A piecewise complex-valued [`Amplitude`] which just contains two parameters representing its magnitude and
/// phase.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewisePolarComplexScalar {
    name: String,
    expr: Expr,
    bins: usize,
    range: (f64, f64),
    r_thetas: Vec<(ParameterLike, ParameterLike)>,
    pids_r_theta: Vec<(ParameterID, ParameterID)>,
    bin_index: ExprID,
}
impl PiecewisePolarComplexScalar {
    /// Create a new [`PiecewiseComplexScalar`] with the given name and parameter value.
    pub fn new(
        name: &str,
        expr: &Expr,
        bins: usize,
        range: (f64, f64),
        r_thetas: Vec<(ParameterLike, ParameterLike)>,
    ) -> Box<Self> {
        assert_eq!(
            bins,
            r_thetas.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            expr: expr.clone(),
            bins,
            range,
            r_thetas,
            pids_r_theta: Default::default(),
            bin_index: Default::default(),
        }
        .into()
    }
}

#[typetag::serde]
impl Amplitude for PiecewisePolarComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> Result<AmplitudeID, LadduError> {
        self.pids_r_theta = self
            .r_thetas
            .iter()
            .map(|(r, theta)| {
                (
                    resources.register_parameter(r),
                    resources.register_parameter(theta),
                )
            })
            .collect();
        let bin_index_expr =
            get_bin_index_polars(self.expr.clone(), self.bins, self.range).cast(DataType::Float64);
        self.bin_index = resources.register_scalar(
            format!(
                "piecewise_bin_index_{}",
                self.expr.clone().meta().output_name()?
            )
            .into(),
            bin_index_expr,
        )?;
        resources.register_amplitude(&self.name)
    }

    fn compute(&self, parameters: &Parameters, cache_row: &CacheRow) -> Complex64 {
        let bin_index: usize = cache_row.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins {
            Complex64::ZERO
        } else {
            let pid_r_theta = self.pids_r_theta[bin_index];
            Complex64::from_polar(parameters.get(pid_r_theta.0), parameters.get(pid_r_theta.1))
        }
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache_row: &CacheRow,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache_row.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins {
            let pid_r_theta = self.pids_r_theta[bin_index];
            let r = parameters.get(pid_r_theta.0);
            let theta = parameters.get(pid_r_theta.1);
            let exp_i_theta = Complex64::cis(theta);
            if let ParameterID::Parameter(ind) = pid_r_theta.0 {
                gradient[ind] = exp_i_theta;
            }
            if let ParameterID::Parameter(ind) = pid_r_theta.1 {
                gradient[ind] = Complex64::I * Complex64::from_polar(r, theta);
            }
        }
    }
}

/// An Amplitude which represents a piecewise function of polar complex values
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name
/// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
///     The variable to use for binning
/// bins: usize
///     The number of bins to use
/// range: tuple of float
///     The minimum and maximum bin edges
/// values : list of tuple of ParameterLike
///     The polar complex parameters contained in each bin of the Amplitude (each tuple contains the
///     magnitude and argument of a single bin)
///
/// Returns
/// -------
/// laddu.Amplitude
///     An Amplitude which can be registered by a laddu.Manager
///
/// Raises
/// ------
/// AssertionError
///     If the number of bins does not match the number of parameters
/// TypeError
///     If the given `variable` is not a valid variable
///
/// See Also
/// --------
/// laddu.Manager
/// laddu.Mass
/// laddu.CosTheta
/// laddu.Phi
/// laddu.PolAngle
/// laddu.PolMagnitude
/// laddu.Mandelstam
///
#[cfg(feature = "python")]
#[pyfunction(name = "PiecewisePolarComplexScalar")]
pub fn py_piecewise_polar_complex_scalar(
    name: &str,
    expr: Bound<'_, PyAny>,
    bins: usize,
    limits: (f64, f64),
    values: Vec<(PyParameterLike, PyParameterLike)>,
) -> PyResult<PyAmplitude> {
    use pyo3_polars::PyExpr;

    let expr = expr.extract::<PyExpr>()?;
    Ok(PyAmplitude(PiecewisePolarComplexScalar::new(
        name,
        &expr.0,
        bins,
        limits,
        values
            .into_iter()
            .map(|(value_re, value_im)| (value_re.0, value_im.0))
            .collect(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, mass, parameter, Manager};
    use std::f64::consts::PI;

    #[test]
    fn test_piecewise_scalar_creation_and_evaluation() {
        let mut manager = Manager::default();
        let v = mass(["kshort1"]);
        let amp = PiecewiseScalar::new(
            "test_scalar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                parameter("test_param0"),
                parameter("test_param1"),
                parameter("test_param2"),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into(); // Direct amplitude evaluation
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let params = vec![1.1, 2.2, 3.3];
        let result = evaluator.evaluate(&params).unwrap();

        assert_relative_eq!(result[0].re, 2.2);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_piecewise_scalar_gradient() {
        let mut manager = Manager::default();
        let v = mass(["kshort1"]);
        let amp = PiecewiseScalar::new(
            "test_scalar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                parameter("test_param0"),
                parameter("test_param1"),
                parameter("test_param2"),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.norm_sqr(); // |f(x)|^2
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let params = vec![1.0, 2.0, 3.0];
        let gradient = evaluator.evaluate_gradient(&params).unwrap();

        // For |f(x)|^2 where f(x) = x, the derivative should be 2x
        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 4.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 0.0);
        assert_relative_eq!(gradient[0][2].im, 0.0);
    }

    #[test]
    fn test_piecewise_complex_scalar_evaluation() {
        let mut manager = Manager::default();
        let v = mass(["kshort1"]);
        let amp = PiecewiseComplexScalar::new(
            "test_complex",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("re_param0"), parameter("im_param0")),
                (parameter("re_param1"), parameter("im_param1")),
                (parameter("re_param2"), parameter("im_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let params = vec![1.1, 1.2, 2.1, 2.2, 3.1, 3.2]; // Real and imaginary parts
        let result = evaluator.evaluate(&params).unwrap();

        assert_relative_eq!(result[0].re, 2.1);
        assert_relative_eq!(result[0].im, 2.2);
    }

    #[test]
    fn test_piecewise_complex_scalar_gradient() {
        let mut manager = Manager::default();
        let v = mass(["kshort1"]);
        let amp = PiecewiseComplexScalar::new(
            "test_complex",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("re_param0"), parameter("im_param0")),
                (parameter("re_param1"), parameter("im_param1")),
                (parameter("re_param2"), parameter("im_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.norm_sqr(); // |f(x + iy)|^2
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let params = vec![1.1, 1.2, 2.1, 2.2, 3.1, 3.2]; // Real and imaginary parts
        let gradient = evaluator.evaluate_gradient(&params).unwrap();

        // For |f(x + iy)|^2, partial derivatives should be 2x and 2y
        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 4.2);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 4.4);
        assert_relative_eq!(gradient[0][3].im, 0.0);
        assert_relative_eq!(gradient[0][4].re, 0.0);
        assert_relative_eq!(gradient[0][4].im, 0.0);
        assert_relative_eq!(gradient[0][5].re, 0.0);
        assert_relative_eq!(gradient[0][5].im, 0.0);
    }

    #[test]
    fn test_piecewise_polar_complex_scalar_evaluation() {
        let mut manager = Manager::default();
        let v = mass(["kshort1"]);
        let amp = PiecewisePolarComplexScalar::new(
            "test_polar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("r_param0"), parameter("theta_param0")),
                (parameter("r_param1"), parameter("theta_param1")),
                (parameter("r_param2"), parameter("theta_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into();
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let r = 2.0;
        let theta = PI / 4.3;
        let params = vec![
            1.1 * r,
            1.2 * theta,
            2.1 * r,
            2.2 * theta,
            3.1 * r,
            3.2 * theta,
        ];
        let result = evaluator.evaluate(&params).unwrap();

        // r * (cos(theta) + i*sin(theta))
        assert_relative_eq!(result[0].re, 2.1 * r * (2.2 * theta).cos());
        assert_relative_eq!(result[0].im, 2.1 * r * (2.2 * theta).sin());
    }

    #[test]
    fn test_piecewise_polar_complex_scalar_gradient() {
        let mut manager = Manager::default();
        let v = mass(["kshort1"]);
        let amp = PiecewisePolarComplexScalar::new(
            "test_polar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("r_param0"), parameter("theta_param0")),
                (parameter("r_param1"), parameter("theta_param1")),
                (parameter("r_param2"), parameter("theta_param2")),
            ],
        );
        let aid = manager.register(amp).unwrap();

        let dataset = test_dataset();
        let expr = aid.into(); // f(r,θ) = re^(iθ)
        let model = manager.model(&expr);
        let evaluator = model.load(&dataset).unwrap();

        let r = 2.0;
        let theta = PI / 4.3;
        let params = vec![
            1.1 * r,
            1.2 * theta,
            2.1 * r,
            2.2 * theta,
            3.1 * r,
            3.2 * theta,
        ];
        let gradient = evaluator.evaluate_gradient(&params).unwrap();

        // d/dr re^(iθ) = e^(iθ), d/dθ re^(iθ) = ire^(iθ)
        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, f64::cos(2.2 * theta));
        assert_relative_eq!(gradient[0][2].im, f64::sin(2.2 * theta));
        assert_relative_eq!(gradient[0][3].re, -2.1 * r * f64::sin(2.2 * theta));
        assert_relative_eq!(gradient[0][3].im, 2.1 * r * f64::cos(2.2 * theta));
        assert_relative_eq!(gradient[0][4].re, 0.0);
        assert_relative_eq!(gradient[0][4].im, 0.0);
        assert_relative_eq!(gradient[0][5].re, 0.0);
        assert_relative_eq!(gradient[0][5].im, 0.0);
    }
}
