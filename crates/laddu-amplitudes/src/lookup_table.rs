use std::str::FromStr;

use laddu_core::{
    amplitudes::{
        Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, ExpressionDependence,
        ParameterLike,
    },
    data::{DatasetMetadata, NamedEventView},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    LadduError, LadduResult, ScalarID,
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

use crate::semantic_key::{debug_key, parameter_pair_slice_key, parameter_slice_key};

/// Interpolation scheme used by [`LookupTable`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LookupInterpolation {
    /// Select the bin containing the input value on each axis.
    Nearest,
}

impl FromStr for LookupInterpolation {
    type Err = LadduError;

    fn from_str(value: &str) -> LadduResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "nearest" | "step" | "bin" => Ok(Self::Nearest),
            _ => Err(LadduError::ParseError {
                name: value.to_string(),
                object: "LookupInterpolation".to_string(),
            }),
        }
    }
}

/// Out-of-range behavior used by [`LookupTable`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LookupBoundaryMode {
    /// Return zero when any axis is outside the tabulated range.
    Zero,
    /// Clamp out-of-range coordinates to the first or last bin.
    Clamp,
}

impl FromStr for LookupBoundaryMode {
    type Err = LadduError;

    fn from_str(value: &str) -> LadduResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "zero" | "zero_outside" | "zero-outside" => Ok(Self::Zero),
            "clamp" => Ok(Self::Clamp),
            _ => Err(LadduError::ParseError {
                name: value.to_string(),
                object: "LookupBoundaryMode".to_string(),
            }),
        }
    }
}

/// A lookup-table axis defined by monotonically increasing bin edges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookupAxis {
    edges: Vec<f64>,
}

impl LookupAxis {
    /// Create an axis from bin edges.
    pub fn new(edges: Vec<f64>) -> LadduResult<Self> {
        if edges.len() < 2 {
            return Err(LadduError::LengthMismatch {
                context: "lookup-table axis edges".to_string(),
                expected: 2,
                actual: edges.len(),
            });
        }
        if !edges.iter().all(|edge| edge.is_finite()) {
            return Err(LadduError::Custom(
                "lookup-table axis edges must be finite".to_string(),
            ));
        }
        if !edges.windows(2).all(|window| window[0] < window[1]) {
            return Err(LadduError::Custom(
                "lookup-table axis edges must be strictly increasing".to_string(),
            ));
        }
        Ok(Self { edges })
    }

    /// Return the number of bins on this axis.
    pub fn bin_count(&self) -> usize {
        self.edges.len() - 1
    }

    fn bin_index(&self, value: f64, boundary_mode: LookupBoundaryMode) -> Option<usize> {
        if !value.is_finite() {
            return None;
        }
        let bins = self.bin_count();
        if value < self.edges[0] {
            return match boundary_mode {
                LookupBoundaryMode::Zero => None,
                LookupBoundaryMode::Clamp => Some(0),
            };
        }
        if value >= self.edges[bins] {
            return match boundary_mode {
                LookupBoundaryMode::Zero => None,
                LookupBoundaryMode::Clamp => Some(bins - 1),
            };
        }
        match self
            .edges
            .binary_search_by(|edge| edge.partial_cmp(&value).expect("finite edge and value"))
        {
            Ok(index) => Some(index.min(bins - 1)),
            Err(index) => Some(index - 1),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum LookupValues {
    FixedComplex(Vec<Complex64>),
    Scalar {
        values: Vec<ParameterLike>,
        pids: Vec<ParameterID>,
    },
    CartesianComplex {
        values: Vec<(ParameterLike, ParameterLike)>,
        pids: Vec<(ParameterID, ParameterID)>,
    },
    PolarComplex {
        values: Vec<(ParameterLike, ParameterLike)>,
        pids: Vec<(ParameterID, ParameterID)>,
    },
}

impl LookupValues {
    fn fixed_complex(values: Vec<Complex64>) -> Self {
        Self::FixedComplex(values)
    }

    fn scalar(values: Vec<ParameterLike>) -> Self {
        Self::Scalar {
            values,
            pids: Vec::new(),
        }
    }

    fn cartesian_complex(values: Vec<(ParameterLike, ParameterLike)>) -> Self {
        Self::CartesianComplex {
            values,
            pids: Vec::new(),
        }
    }

    fn polar_complex(values: Vec<(ParameterLike, ParameterLike)>) -> Self {
        Self::PolarComplex {
            values,
            pids: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::FixedComplex(values) => values.len(),
            Self::Scalar { values, .. } => values.len(),
            Self::CartesianComplex { values, .. } | Self::PolarComplex { values, .. } => {
                values.len()
            }
        }
    }

    fn semantic_key(&self) -> String {
        match self {
            Self::FixedComplex(values) => format!("FixedComplex({})", debug_key(values)),
            Self::Scalar { values, .. } => format!("Scalar({})", parameter_slice_key(values)),
            Self::CartesianComplex { values, .. } => {
                format!("CartesianComplex({})", parameter_pair_slice_key(values))
            }
            Self::PolarComplex { values, .. } => {
                format!("PolarComplex({})", parameter_pair_slice_key(values))
            }
        }
    }

    fn register(&mut self, resources: &mut Resources) -> LadduResult<()> {
        match self {
            Self::FixedComplex(_) => Ok(()),
            Self::Scalar { values, pids } => {
                *pids = values
                    .iter()
                    .map(|value| resources.register_parameter(value))
                    .collect::<LadduResult<Vec<_>>>()?;
                Ok(())
            }
            Self::CartesianComplex { values, pids } | Self::PolarComplex { values, pids } => {
                *pids = values
                    .iter()
                    .map(
                        |(first, second)| -> LadduResult<(ParameterID, ParameterID)> {
                            Ok((
                                resources.register_parameter(first)?,
                                resources.register_parameter(second)?,
                            ))
                        },
                    )
                    .collect::<LadduResult<Vec<_>>>()?;
                Ok(())
            }
        }
    }

    fn real_valued_hint(&self) -> bool {
        match self {
            Self::FixedComplex(values) => values.iter().all(|value| value.im == 0.0),
            Self::Scalar { .. } => true,
            Self::CartesianComplex { .. } | Self::PolarComplex { .. } => false,
        }
    }

    fn value(&self, index: usize, parameters: &Parameters) -> Complex64 {
        match self {
            Self::FixedComplex(values) => values[index],
            Self::Scalar { pids, .. } => Complex64::new(parameters.get(pids[index]), 0.0),
            Self::CartesianComplex { pids, .. } => {
                Complex64::new(parameters.get(pids[index].0), parameters.get(pids[index].1))
            }
            Self::PolarComplex { pids, .. } => {
                Complex64::from_polar(parameters.get(pids[index].0), parameters.get(pids[index].1))
            }
        }
    }

    fn gradient(&self, index: usize, parameters: &Parameters, gradient: &mut DVector<Complex64>) {
        match self {
            Self::FixedComplex(_) => {}
            Self::Scalar { pids, .. } => {
                if let ParameterID::Parameter(ind) = pids[index] {
                    gradient[ind] = Complex64::ONE;
                }
            }
            Self::CartesianComplex { pids, .. } => {
                if let ParameterID::Parameter(ind) = pids[index].0 {
                    gradient[ind] = Complex64::ONE;
                }
                if let ParameterID::Parameter(ind) = pids[index].1 {
                    gradient[ind] = Complex64::I;
                }
            }
            Self::PolarComplex { pids, .. } => {
                let r = parameters.get(pids[index].0);
                let theta = parameters.get(pids[index].1);
                let exp_i_theta = Complex64::cis(theta);
                if let ParameterID::Parameter(ind) = pids[index].0 {
                    gradient[ind] = exp_i_theta;
                }
                if let ParameterID::Parameter(ind) = pids[index].1 {
                    gradient[ind] = Complex64::I * Complex64::from_polar(r, theta);
                }
            }
        }
    }
}

/// A lookup table over one or more event [`Variable`]s.
#[derive(Clone, Serialize, Deserialize)]
pub struct LookupTable {
    name: String,
    variables: Vec<Box<dyn Variable>>,
    axes: Vec<LookupAxis>,
    values: LookupValues,
    interpolation: LookupInterpolation,
    boundary_mode: LookupBoundaryMode,
    index_id: ScalarID,
}

impl LookupTable {
    /// Create a fixed complex lookup table.
    pub fn new(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axes: Vec<LookupAxis>,
        values: Vec<Complex64>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axes,
            LookupValues::fixed_complex(values),
            interpolation,
            boundary_mode,
        )
    }

    /// Create a parameterized scalar lookup table.
    pub fn new_scalar(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axes: Vec<LookupAxis>,
        values: Vec<ParameterLike>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axes,
            LookupValues::scalar(values),
            interpolation,
            boundary_mode,
        )
    }

    /// Create a parameterized cartesian complex lookup table.
    pub fn new_cartesian_complex(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axes: Vec<LookupAxis>,
        values: Vec<(ParameterLike, ParameterLike)>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axes,
            LookupValues::cartesian_complex(values),
            interpolation,
            boundary_mode,
        )
    }

    /// Create a parameterized polar complex lookup table.
    pub fn new_polar_complex(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axes: Vec<LookupAxis>,
        values: Vec<(ParameterLike, ParameterLike)>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axes,
            LookupValues::polar_complex(values),
            interpolation,
            boundary_mode,
        )
    }

    fn with_values(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axes: Vec<LookupAxis>,
        values: LookupValues,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        if variables.is_empty() {
            return Err(LadduError::LengthMismatch {
                context: "lookup-table variables".to_string(),
                expected: 1,
                actual: 0,
            });
        }
        if variables.len() != axes.len() {
            return Err(LadduError::LengthMismatch {
                context: "lookup-table axes".to_string(),
                expected: variables.len(),
                actual: axes.len(),
            });
        }
        let expected = axes.iter().try_fold(1usize, |acc, axis| {
            acc.checked_mul(axis.bin_count())
                .ok_or_else(|| LadduError::Custom("lookup-table shape is too large".to_string()))
        })?;
        if values.len() != expected {
            return Err(LadduError::LengthMismatch {
                context: "lookup-table values".to_string(),
                expected,
                actual: values.len(),
            });
        }
        Self {
            name: name.to_string(),
            variables,
            axes,
            values,
            interpolation,
            boundary_mode,
            index_id: ScalarID::default(),
        }
        .into_expression()
    }

    fn index_for_event(&self, event: &NamedEventView<'_>) -> Option<usize> {
        match self.interpolation {
            LookupInterpolation::Nearest => self.nearest_index(event),
        }
    }

    fn nearest_index(&self, event: &NamedEventView<'_>) -> Option<usize> {
        let mut flat_index = 0usize;
        for (variable, axis) in self.variables.iter().zip(&self.axes) {
            let bin_index = axis.bin_index(variable.value(event), self.boundary_mode)?;
            flat_index = flat_index * axis.bin_count() + bin_index;
        }
        Some(flat_index)
    }
}

#[typetag::serde]
impl Amplitude for LookupTable {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.values.register(resources)?;
        self.index_id = resources.register_scalar(Some(&format!("{}.index", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("LookupTable")
                .with_field("name", debug_key(&self.name))
                .with_field("variables", debug_key(&self.variables))
                .with_field("axes", debug_key(&self.axes))
                .with_field("values", self.values.semantic_key())
                .with_field("interpolation", debug_key(self.interpolation))
                .with_field("boundary_mode", debug_key(self.boundary_mode)),
        )
    }

    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::CacheOnly
    }

    fn real_valued_hint(&self) -> bool {
        self.values.real_valued_hint()
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        for variable in &mut self.variables {
            variable.bind(metadata)?;
        }
        Ok(())
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        let index = self.index_for_event(event).unwrap_or(self.values.len());
        cache.store_scalar(self.index_id, index as f64);
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        let index = cache.get_scalar(self.index_id) as usize;
        if index == self.values.len() {
            Complex64::ZERO
        } else {
            self.values.value(index, parameters)
        }
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        let index = cache.get_scalar(self.index_id) as usize;
        if index != self.values.len() {
            self.values.gradient(index, parameters, gradient);
        }
    }
}

#[cfg(feature = "python")]
fn py_lookup_inputs(
    variables: Vec<PyVariable>,
    axes: Vec<Vec<f64>>,
) -> LadduResult<(Vec<Box<dyn Variable>>, Vec<LookupAxis>)> {
    let axes = axes
        .into_iter()
        .map(LookupAxis::new)
        .collect::<LadduResult<Vec<_>>>()?;
    let variables = variables
        .into_iter()
        .map(|variable| Box::new(variable) as Box<dyn Variable>)
        .collect();
    Ok((variables, axes))
}

/// An amplitude which evaluates a fixed complex lookup table over event variables.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variables : list of laddu variables
///     The event variables that define the lookup coordinates.
/// axes : list of list of float
///     Per-variable bin edges. Each axis must have one more edge than bins.
/// values : list of complex
///     Flattened row-major table values.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Currently supports "nearest".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
#[cfg(feature = "python")]
#[pyfunction(name = "LookupTable", signature = (name, variables, axes, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table(
    name: &str,
    variables: Vec<PyVariable>,
    axes: Vec<Vec<f64>>,
    values: Vec<Complex64>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axes) = py_lookup_inputs(variables, axes)?;
    Ok(PyExpression(LookupTable::new(
        name,
        variables,
        axes,
        values,
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

/// An amplitude which evaluates a scalar-parameter lookup table over event variables.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variables : list of laddu variables
///     The event variables that define the lookup coordinates.
/// axes : list of list of float
///     Per-variable bin edges. Each axis must have one more edge than bins.
/// values : list of laddu.ParameterLike
///     Flattened row-major scalar parameters.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Currently supports "nearest".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
#[cfg(feature = "python")]
#[pyfunction(name = "LookupTableScalar", signature = (name, variables, axes, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table_scalar(
    name: &str,
    variables: Vec<PyVariable>,
    axes: Vec<Vec<f64>>,
    values: Vec<PyParameterLike>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axes) = py_lookup_inputs(variables, axes)?;
    Ok(PyExpression(LookupTable::new_scalar(
        name,
        variables,
        axes,
        values.into_iter().map(|value| value.0).collect(),
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

/// An amplitude which evaluates a cartesian-complex-parameter lookup table over event variables.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variables : list of laddu variables
///     The event variables that define the lookup coordinates.
/// axes : list of list of float
///     Per-variable bin edges. Each axis must have one more edge than bins.
/// values : list of tuple of laddu.ParameterLike
///     Flattened row-major real and imaginary parameter pairs.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Currently supports "nearest".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
#[cfg(feature = "python")]
#[pyfunction(name = "LookupTableComplex", signature = (name, variables, axes, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table_complex(
    name: &str,
    variables: Vec<PyVariable>,
    axes: Vec<Vec<f64>>,
    values: Vec<(PyParameterLike, PyParameterLike)>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axes) = py_lookup_inputs(variables, axes)?;
    Ok(PyExpression(LookupTable::new_cartesian_complex(
        name,
        variables,
        axes,
        values
            .into_iter()
            .map(|(value_re, value_im)| (value_re.0, value_im.0))
            .collect(),
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

/// An amplitude which evaluates a polar-complex-parameter lookup table over event variables.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variables : list of laddu variables
///     The event variables that define the lookup coordinates.
/// axes : list of list of float
///     Per-variable bin edges. Each axis must have one more edge than bins.
/// values : list of tuple of laddu.ParameterLike
///     Flattened row-major magnitude and phase parameter pairs.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Currently supports "nearest".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
#[cfg(feature = "python")]
#[pyfunction(name = "LookupTablePolar", signature = (name, variables, axes, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table_polar(
    name: &str,
    variables: Vec<PyVariable>,
    axes: Vec<Vec<f64>>,
    values: Vec<(PyParameterLike, PyParameterLike)>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axes) = py_lookup_inputs(variables, axes)?;
    Ok(PyExpression(LookupTable::new_polar_complex(
        name,
        variables,
        axes,
        values
            .into_iter()
            .map(|(value_r, value_theta)| (value_r.0, value_theta.0))
            .collect(),
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, parameter, Mass};

    use super::*;

    fn mass(name: &str) -> Box<dyn Variable> {
        Box::new(Mass::new([name]))
    }

    #[test]
    fn test_lookup_table_1d_nearest() {
        let expr = LookupTable::new(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 0.25, 0.75, 1.0]).unwrap()],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 3.0),
                Complex64::new(4.0, 0.0),
            ],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 2.0);
        assert_relative_eq!(result[0].im, 3.0);
    }

    #[test]
    fn test_lookup_table_2d_row_major() {
        let expr = LookupTable::new(
            "lookup",
            vec![mass("kshort1"), Box::new(Mass::new(["kshort1", "kshort2"]))],
            vec![
                LookupAxis::new(vec![0.0, 1.0, 2.0]).unwrap(),
                LookupAxis::new(vec![0.0, 1.0, 2.0]).unwrap(),
            ],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 2.0);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_lookup_table_zero_boundary() {
        let expr = LookupTable::new(
            "lookup",
            vec![Box::new(Mass::new(["kshort1", "kshort2"]))],
            vec![LookupAxis::new(vec![0.0, 0.5, 1.0]).unwrap()],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]);

        assert_eq!(result[0], Complex64::ZERO);
    }

    #[test]
    fn test_lookup_table_clamp_boundary() {
        let expr = LookupTable::new(
            "lookup",
            vec![Box::new(Mass::new(["kshort1", "kshort2"]))],
            vec![LookupAxis::new(vec![0.0, 0.5, 1.0]).unwrap()],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Clamp,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]);

        assert_relative_eq!(result[0].re, 2.0);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_lookup_table_scalar_parameters_and_gradient() {
        let expr = LookupTable::new_scalar(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 0.25, 0.75, 1.0]).unwrap()],
            vec![parameter("p0"), parameter("p1"), parameter("p2")],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
        .unwrap()
        .norm_sqr();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate_gradient(&[1.0, 2.0, 3.0]);

        assert_relative_eq!(result[0][0].re, 0.0);
        assert_relative_eq!(result[0][0].im, 0.0);
        assert_relative_eq!(result[0][1].re, 4.0);
        assert_relative_eq!(result[0][1].im, 0.0);
        assert_relative_eq!(result[0][2].re, 0.0);
        assert_relative_eq!(result[0][2].im, 0.0);
    }

    #[test]
    fn test_lookup_table_polar_parameters_and_gradient() {
        let expr = LookupTable::new_polar_complex(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 0.25, 0.75, 1.0]).unwrap()],
            vec![
                (parameter("r0"), parameter("theta0")),
                (parameter("r1"), parameter("theta1")),
                (parameter("r2"), parameter("theta2")),
            ],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let params = [1.1, 1.2, 2.1, 2.2, 3.1, 3.2];
        let gradient = evaluator.evaluate_gradient(&params);

        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][2].re, 2.2_f64.cos());
        assert_relative_eq!(gradient[0][2].im, 2.2_f64.sin());
        assert_relative_eq!(gradient[0][3].re, -2.1 * 2.2_f64.sin());
        assert_relative_eq!(gradient[0][3].im, 2.1 * 2.2_f64.cos());
        assert_relative_eq!(gradient[0][4].re, 0.0);
        assert_relative_eq!(gradient[0][4].im, 0.0);
    }

    #[test]
    fn test_lookup_table_rejects_shape_mismatch() {
        let result = LookupTable::new(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 0.5, 1.0]).unwrap()],
            vec![Complex64::new(1.0, 0.0)],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        );

        assert!(result.is_err());
    }
}
