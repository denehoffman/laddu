use std::str::FromStr;

use laddu_core::{
    amplitudes::{Amplitude, AmplitudeID, AmplitudeSemanticKey, Expression, ExpressionDependence},
    data::{DatasetMetadata, NamedEventView},
    resources::{Cache, ComplexScalarID, Parameters, Resources},
    traits::Variable,
    LadduError, LadduResult,
};
#[cfg(feature = "python")]
use laddu_python::{amplitudes::PyExpression, utils::variables::PyVariable};
use nalgebra::DVector;
use num::complex::Complex64;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::semantic_key::debug_key;

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

/// A fixed complex lookup table over one or more event [`Variable`]s.
#[derive(Clone, Serialize, Deserialize)]
pub struct LookupTable {
    name: String,
    variables: Vec<Box<dyn Variable>>,
    axes: Vec<LookupAxis>,
    values: Vec<Complex64>,
    interpolation: LookupInterpolation,
    boundary_mode: LookupBoundaryMode,
    value_id: ComplexScalarID,
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
            value_id: ComplexScalarID::default(),
        }
        .into_expression()
    }

    fn value_for_event(&self, event: &NamedEventView<'_>) -> Complex64 {
        match self.interpolation {
            LookupInterpolation::Nearest => self.nearest_value(event),
        }
    }

    fn nearest_value(&self, event: &NamedEventView<'_>) -> Complex64 {
        let mut flat_index = 0usize;
        for (variable, axis) in self.variables.iter().zip(&self.axes) {
            let Some(bin_index) = axis.bin_index(variable.value(event), self.boundary_mode) else {
                return Complex64::ZERO;
            };
            flat_index = flat_index * axis.bin_count() + bin_index;
        }
        self.values[flat_index]
    }
}

#[typetag::serde]
impl Amplitude for LookupTable {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.value_id = resources.register_complex_scalar(Some(&format!("{}.value", self.name)));
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("LookupTable")
                .with_field("name", debug_key(&self.name))
                .with_field("variables", debug_key(&self.variables))
                .with_field("axes", debug_key(&self.axes))
                .with_field("values", debug_key(&self.values))
                .with_field("interpolation", debug_key(self.interpolation))
                .with_field("boundary_mode", debug_key(self.boundary_mode)),
        )
    }

    fn dependence_hint(&self) -> ExpressionDependence {
        ExpressionDependence::CacheOnly
    }

    fn real_valued_hint(&self) -> bool {
        self.values.iter().all(|value| value.im == 0.0)
    }

    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        for variable in &mut self.variables {
            variable.bind(metadata)?;
        }
        Ok(())
    }

    fn precompute(&self, event: &NamedEventView<'_>, cache: &mut Cache) {
        cache.store_complex_scalar(self.value_id, self.value_for_event(event));
    }

    fn compute(&self, _parameters: &Parameters, cache: &Cache) -> Complex64 {
        cache.get_complex_scalar(self.value_id)
    }

    fn compute_gradient(
        &self,
        _parameters: &Parameters,
        _cache: &Cache,
        _gradient: &mut DVector<Complex64>,
    ) {
    }
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
    let axes = axes
        .into_iter()
        .map(LookupAxis::new)
        .collect::<LadduResult<Vec<_>>>()?;
    let variables = variables
        .into_iter()
        .map(|variable| Box::new(variable) as Box<dyn Variable>)
        .collect();
    Ok(PyExpression(LookupTable::new(
        name,
        variables,
        axes,
        values,
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_core::{data::test_dataset, Mass};

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
