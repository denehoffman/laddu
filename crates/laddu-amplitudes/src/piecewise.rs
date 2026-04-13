use laddu_core::{
    amplitudes::{Expression, ParameterLike},
    traits::Variable,
    utils::get_bin_edges,
    LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::{
    amplitudes::{PyExpression, PyParameterLike},
    utils::variables::PyVariable,
};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{LookupAxis, LookupBoundaryMode, LookupInterpolation, LookupTable};

/// Compatibility constructor for 1D nearest-bin scalar lookup tables.
pub struct PiecewiseScalar;

impl PiecewiseScalar {
    /// Create a 1D nearest-bin lookup table with one scalar parameter per bin.
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
        LookupTable::new_scalar(
            name,
            vec![dyn_clone::clone_box(variable)],
            vec![LookupAxis::new(get_bin_edges(bins, range))?],
            values,
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
    }
}

/// Compatibility constructor for 1D nearest-bin cartesian-complex lookup tables.
pub struct PiecewiseComplexScalar;

impl PiecewiseComplexScalar {
    /// Create a 1D nearest-bin lookup table with real and imaginary parameters per bin.
    pub fn new<V: Variable + 'static>(
        name: &str,
        variable: &V,
        bins: usize,
        range: (f64, f64),
        values: Vec<(ParameterLike, ParameterLike)>,
    ) -> LadduResult<Expression> {
        assert_eq!(
            bins,
            values.len(),
            "Number of bins must match number of parameters!"
        );
        LookupTable::new_cartesian_complex(
            name,
            vec![dyn_clone::clone_box(variable)],
            vec![LookupAxis::new(get_bin_edges(bins, range))?],
            values,
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
    }
}

/// Compatibility constructor for 1D nearest-bin polar-complex lookup tables.
pub struct PiecewisePolarComplexScalar;

impl PiecewisePolarComplexScalar {
    /// Create a 1D nearest-bin lookup table with magnitude and phase parameters per bin.
    pub fn new<V: Variable + 'static>(
        name: &str,
        variable: &V,
        bins: usize,
        range: (f64, f64),
        values: Vec<(ParameterLike, ParameterLike)>,
    ) -> LadduResult<Expression> {
        assert_eq!(
            bins,
            values.len(),
            "Number of bins must match number of parameters!"
        );
        LookupTable::new_polar_complex(
            name,
            vec![dyn_clone::clone_box(variable)],
            vec![LookupAxis::new(get_bin_edges(bins, range))?],
            values,
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
    }
}

/// A compatibility constructor for a 1D nearest-bin scalar lookup table.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variable : laddu.Mass | laddu.CosTheta | laddu.Phi | laddu.PolAngle | laddu.PolMagnitude | laddu.Mandelstam
///     The variable to use for binning.
/// bins : int
///     The number of bins.
/// range : tuple of float
///     The minimum and maximum bin edges.
/// values : list of laddu.ParameterLike
///     The scalar parameters contained in each bin.
///
/// Returns
/// -------
/// laddu.Expression
///     A nearest-bin lookup-table Expression.
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

/// A compatibility constructor for a 1D nearest-bin cartesian-complex lookup table.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variable : laddu.Mass | laddu.CosTheta | laddu.Phi | laddu.PolAngle | laddu.PolMagnitude | laddu.Mandelstam
///     The variable to use for binning.
/// bins : int
///     The number of bins.
/// range : tuple of float
///     The minimum and maximum bin edges.
/// values : list of tuple of laddu.ParameterLike
///     The real and imaginary parameters contained in each bin.
///
/// Returns
/// -------
/// laddu.Expression
///     A nearest-bin lookup-table Expression.
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

/// A compatibility constructor for a 1D nearest-bin polar-complex lookup table.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variable : laddu.Mass | laddu.CosTheta | laddu.Phi | laddu.PolAngle | laddu.PolMagnitude | laddu.Mandelstam
///     The variable to use for binning.
/// bins : int
///     The number of bins.
/// range : tuple of float
///     The minimum and maximum bin edges.
/// values : list of tuple of laddu.ParameterLike
///     The magnitude and phase parameters contained in each bin.
///
/// Returns
/// -------
/// laddu.Expression
///     A nearest-bin lookup-table Expression.
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
            .map(|(value_r, value_theta)| (value_r.0, value_theta.0))
            .collect(),
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Mass, PI};

    use super::*;

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
        let result = evaluator.evaluate(&[1.1, 2.2, 3.3]);

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
        .norm_sqr();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let gradient = evaluator.evaluate_gradient(&[1.0, 2.0, 3.0]);

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
        let result = evaluator.evaluate(&[1.1, 1.2, 2.1, 2.2, 3.1, 3.2]);

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
        .norm_sqr();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let gradient = evaluator.evaluate_gradient(&[1.1, 1.2, 2.1, 2.2, 3.1, 3.2]);

        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 4.2);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 4.4);
        assert_relative_eq!(gradient[0][3].im, 0.0);
        assert_relative_eq!(gradient[0][4].re, 0.0);
        assert_relative_eq!(gradient[0][4].im, 0.0);
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
        let result = evaluator.evaluate(&[
            1.1 * r,
            1.2 * theta,
            2.1 * r,
            2.2 * theta,
            3.1 * r,
            3.2 * theta,
        ]);

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
        let gradient = evaluator.evaluate_gradient(&[
            1.1 * r,
            1.2 * theta,
            2.1 * r,
            2.2 * theta,
            3.1 * r,
            3.2 * theta,
        ]);

        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, (2.2 * theta).cos());
        assert_relative_eq!(gradient[0][2].im, (2.2 * theta).sin());
        assert_relative_eq!(gradient[0][3].re, -2.1 * r * (2.2 * theta).sin());
        assert_relative_eq!(gradient[0][3].im, 2.1 * r * (2.2 * theta).cos());
        assert_relative_eq!(gradient[0][4].re, 0.0);
        assert_relative_eq!(gradient[0][4].im, 0.0);
    }
}
