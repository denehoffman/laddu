use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, Expression, ParameterLike},
    data::{DatasetMetadata, EventData},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    utils::get_bin_index,
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

/// A piecewise scalar-valued [`Amplitude`] which just contains a single parameter for each bin as its value.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewiseScalar {
    name: String,
    variable: Box<dyn Variable>,
    bins: usize,
    range: (f64, f64),
    values: Vec<ParameterLike>,
    pids: Vec<ParameterID>,
    bin_index: ScalarID,
}
impl PiecewiseScalar {
    /// Create a new [`PiecewiseScalar`] with the given name and parameter value.
    pub fn new<V: Variable + 'static>(
        name: &str,
        variable: &V,
        bins: usize,
        range: (f64, f64),
        values: Vec<ParameterLike>,
    ) -> LadduResult<Expression> {
        assert_eq!(
            bins,
            values.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            variable: dyn_clone::clone_box(variable),
            bins,
            range,
            values,
            pids: Default::default(),
            bin_index: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PiecewiseScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pids = self
            .values
            .iter()
            .map(|value| resources.register_parameter(value))
            .collect::<LadduResult<Vec<_>>>()?;
        self.bin_index = resources.register_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.variable.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        let maybe_bin_index = get_bin_index(self.variable.value(event), self.bins, self.range);
        if let Some(bin_index) = maybe_bin_index {
            cache.store_scalar(self.bin_index, bin_index as f64);
        } else {
            cache.store_scalar(self.bin_index, (self.bins + 1) as f64);
            // store ibin = nbins + 1 if outside range
        }
    }

    fn compute(&self, parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins + 1 {
            Complex64::ZERO
        } else {
            Complex64::from(parameters.get(self.pids[bin_index]))
        }
    }

    fn compute_cached(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins + 1 {
            Complex64::ZERO
        } else {
            Complex64::from(parameters.get(self.pids[bin_index]))
        }
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &EventData,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins + 1 {
            gradient[bin_index] = Complex64::ONE;
        }
    }

    fn compute_gradient_cached(
        &self,
        _parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins + 1 {
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
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
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
    range: (f64, f64),
    values: Vec<PyParameterLike>,
) -> PyResult<PyExpression> {
    let variable = variable.extract::<PyVariable>()?;
    Ok(PyExpression(PiecewiseScalar::new(
        name,
        &variable,
        bins,
        range,
        values.into_iter().map(|value| value.0).collect(),
    )?))
}

/// A piecewise complex-valued [`Amplitude`] which just contains two parameters representing its real and
/// imaginary parts.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewiseComplexScalar {
    name: String,
    variable: Box<dyn Variable>,
    bins: usize,
    range: (f64, f64),
    re_ims: Vec<(ParameterLike, ParameterLike)>,
    pids_re_im: Vec<(ParameterID, ParameterID)>,
    bin_index: ScalarID,
}
impl PiecewiseComplexScalar {
    /// Create a new [`PiecewiseComplexScalar`] with the given name and parameter value.
    pub fn new<V: Variable + 'static>(
        name: &str,
        variable: &V,
        bins: usize,
        range: (f64, f64),
        re_ims: Vec<(ParameterLike, ParameterLike)>,
    ) -> LadduResult<Expression> {
        assert_eq!(
            bins,
            re_ims.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            variable: dyn_clone::clone_box(variable),
            bins,
            range,
            re_ims,
            pids_re_im: Default::default(),
            bin_index: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PiecewiseComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pids_re_im = self
            .re_ims
            .iter()
            .map(|(re, im)| -> LadduResult<(ParameterID, ParameterID)> {
                Ok((
                    resources.register_parameter(re)?,
                    resources.register_parameter(im)?,
                ))
            })
            .collect::<LadduResult<Vec<_>>>()?;
        self.bin_index = resources.register_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.variable.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        let maybe_bin_index = get_bin_index(self.variable.value(event), self.bins, self.range);
        if let Some(bin_index) = maybe_bin_index {
            cache.store_scalar(self.bin_index, bin_index as f64);
        } else {
            cache.store_scalar(self.bin_index, (self.bins + 1) as f64);
            // store ibin = nbins + 1 if outside range
        }
    }

    fn compute(&self, parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins + 1 {
            Complex64::ZERO
        } else {
            let pid_re_im = self.pids_re_im[bin_index];
            Complex64::new(parameters.get(pid_re_im.0), parameters.get(pid_re_im.1))
        }
    }

    fn compute_cached(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins + 1 {
            Complex64::ZERO
        } else {
            let pid_re_im = self.pids_re_im[bin_index];
            Complex64::new(parameters.get(pid_re_im.0), parameters.get(pid_re_im.1))
        }
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _event: &EventData,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins + 1 {
            let pid_re_im = self.pids_re_im[bin_index];
            if let ParameterID::Parameter(ind) = pid_re_im.0 {
                gradient[ind] = Complex64::ONE;
            }
            if let ParameterID::Parameter(ind) = pid_re_im.1 {
                gradient[ind] = Complex64::I;
            }
        }
    }

    fn compute_gradient_cached(
        &self,
        _parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins + 1 {
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
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
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
    variable: Bound<'_, PyAny>,
    bins: usize,
    range: (f64, f64),
    values: Vec<(PyParameterLike, PyParameterLike)>,
) -> PyResult<PyExpression> {
    let variable = variable.extract::<PyVariable>()?;
    Ok(PyExpression(PiecewiseComplexScalar::new(
        name,
        &variable,
        bins,
        range,
        values
            .into_iter()
            .map(|(value_re, value_im)| (value_re.0, value_im.0))
            .collect(),
    )?))
}

/// A piecewise complex-valued [`Amplitude`] which just contains two parameters representing its magnitude and
/// phase.
#[derive(Clone, Serialize, Deserialize)]
pub struct PiecewisePolarComplexScalar {
    name: String,
    variable: Box<dyn Variable>,
    bins: usize,
    range: (f64, f64),
    r_thetas: Vec<(ParameterLike, ParameterLike)>,
    pids_r_theta: Vec<(ParameterID, ParameterID)>,
    bin_index: ScalarID,
}
impl PiecewisePolarComplexScalar {
    /// Create a new [`PiecewiseComplexScalar`] with the given name and parameter value.
    pub fn new<V: Variable + 'static>(
        name: &str,
        variable: &V,
        bins: usize,
        range: (f64, f64),
        r_thetas: Vec<(ParameterLike, ParameterLike)>,
    ) -> LadduResult<Expression> {
        assert_eq!(
            bins,
            r_thetas.len(),
            "Number of bins must match number of parameters!"
        );
        Self {
            name: name.to_string(),
            variable: dyn_clone::clone_box(variable),
            bins,
            range,
            r_thetas,
            pids_r_theta: Default::default(),
            bin_index: Default::default(),
        }
        .into_expression()
    }
}

#[typetag::serde]
impl Amplitude for PiecewisePolarComplexScalar {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.pids_r_theta = self
            .r_thetas
            .iter()
            .map(|(r, theta)| -> LadduResult<(ParameterID, ParameterID)> {
                Ok((
                    resources.register_parameter(r)?,
                    resources.register_parameter(theta)?,
                ))
            })
            .collect::<LadduResult<Vec<_>>>()?;
        self.bin_index = resources.register_scalar(None);
        resources.register_amplitude(&self.name)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        self.variable.bind(metadata)?;
        Ok(())
    }

    fn precompute(&self, event: &EventData, cache: &mut Cache) {
        let maybe_bin_index = get_bin_index(self.variable.value(event), self.bins, self.range);
        if let Some(bin_index) = maybe_bin_index {
            cache.store_scalar(self.bin_index, bin_index as f64);
        } else {
            cache.store_scalar(self.bin_index, (self.bins + 1) as f64);
            // store ibin = nbins + 1 if outside range
        }
    }

    fn compute(&self, parameters: &Parameters, _event: &EventData, cache: &Cache) -> Complex64 {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins + 1 {
            Complex64::ZERO
        } else {
            let pid_r_theta = self.pids_r_theta[bin_index];
            Complex64::from_polar(parameters.get(pid_r_theta.0), parameters.get(pid_r_theta.1))
        }
    }

    fn compute_cached(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index == self.bins + 1 {
            Complex64::ZERO
        } else {
            let pid_r_theta = self.pids_r_theta[bin_index];
            Complex64::from_polar(parameters.get(pid_r_theta.0), parameters.get(pid_r_theta.1))
        }
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        _event: &EventData,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins + 1 {
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

    fn compute_gradient_cached(
        &self,
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let bin_index: usize = cache.get_scalar(self.bin_index) as usize;
        if bin_index < self.bins + 1 {
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
/// laddu.Expression
///     An Expression which can be loaded and evaluated directly
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
    variable: Bound<'_, PyAny>,
    bins: usize,
    range: (f64, f64),
    values: Vec<(PyParameterLike, PyParameterLike)>,
) -> PyResult<PyExpression> {
    let variable = variable.extract::<PyVariable>()?;
    Ok(PyExpression(PiecewisePolarComplexScalar::new(
        name,
        &variable,
        bins,
        range,
        values
            .into_iter()
            .map(|(value_re, value_im)| (value_re.0, value_im.0))
            .collect(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Mass, PI};
    use std::sync::Arc;

    #[test]
    fn test_piecewise_scalar_creation_and_evaluation() {
        let v = Mass::new(["kshort1"]);
        let expr = PiecewiseScalar::new(
            "test_scalar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                parameter("test_param0"),
                parameter("test_param1"),
                parameter("test_param2"),
            ],
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![1.1, 2.2, 3.3];
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 2.2);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_piecewise_scalar_gradient() {
        let v = Mass::new(["kshort1"]);
        let expr = PiecewiseScalar::new(
            "test_scalar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                parameter("test_param0"),
                parameter("test_param1"),
                parameter("test_param2"),
            ],
        )
        .unwrap()
        .norm_sqr(); // |f(x)|^2

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![1.0, 2.0, 3.0];
        let gradient = evaluator.evaluate_gradient(&params);

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
        let v = Mass::new(["kshort1"]);
        let expr = PiecewiseComplexScalar::new(
            "test_complex",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("re_param0"), parameter("im_param0")),
                (parameter("re_param1"), parameter("im_param1")),
                (parameter("re_param2"), parameter("im_param2")),
            ],
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![1.1, 1.2, 2.1, 2.2, 3.1, 3.2]; // Real and imaginary parts
        let result = evaluator.evaluate(&params);

        assert_relative_eq!(result[0].re, 2.1);
        assert_relative_eq!(result[0].im, 2.2);
    }

    #[test]
    fn test_piecewise_complex_scalar_gradient() {
        let v = Mass::new(["kshort1"]);
        let expr = PiecewiseComplexScalar::new(
            "test_complex",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("re_param0"), parameter("im_param0")),
                (parameter("re_param1"), parameter("im_param1")),
                (parameter("re_param2"), parameter("im_param2")),
            ],
        )
        .unwrap()
        .norm_sqr(); // |f(x + iy)|^2

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();

        let params = vec![1.1, 1.2, 2.1, 2.2, 3.1, 3.2]; // Real and imaginary parts
        let gradient = evaluator.evaluate_gradient(&params);

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
        let v = Mass::new(["kshort1"]);
        let expr = PiecewisePolarComplexScalar::new(
            "test_polar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("r_param0"), parameter("theta_param0")),
                (parameter("r_param1"), parameter("theta_param1")),
                (parameter("r_param2"), parameter("theta_param2")),
            ],
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();

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
        let result = evaluator.evaluate(&params);

        // r * (cos(theta) + i*sin(theta))
        assert_relative_eq!(result[0].re, 2.1 * r * (2.2 * theta).cos());
        assert_relative_eq!(result[0].im, 2.1 * r * (2.2 * theta).sin());
    }

    #[test]
    fn test_piecewise_polar_complex_scalar_gradient() {
        let v = Mass::new(["kshort1"]);
        let expr = PiecewisePolarComplexScalar::new(
            "test_polar",
            &v,
            3,
            (0.0, 1.0),
            vec![
                (parameter("r_param0"), parameter("theta_param0")),
                (parameter("r_param1"), parameter("theta_param1")),
                (parameter("r_param2"), parameter("theta_param2")),
            ],
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();

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
        let gradient = evaluator.evaluate_gradient(&params);

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
