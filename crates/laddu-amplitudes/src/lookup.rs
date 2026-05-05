use std::str::FromStr;

use laddu_core::{
    amplitudes::{
        debug_key, parameter_pair_slice_key, parameter_slice_key, Amplitude, AmplitudeID,
        AmplitudeSemanticKey, Expression, ExpressionDependence, Parameter,
    },
    data::{DatasetMetadata, Event},
    resources::{Cache, ParameterID, Parameters, Resources},
    traits::Variable,
    LadduError, LadduResult, ScalarID,
};
use nalgebra::DVector;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

/// Interpolation scheme used by [`LookupTable`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LookupInterpolation {
    /// Select the bin containing the input value on each axis.
    Nearest,
    /// Interpolate linearly between neighboring table values on each axis.
    Linear,
}

impl FromStr for LookupInterpolation {
    type Err = LadduError;

    fn from_str(value: &str) -> LadduResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "nearest" | "step" | "bin" => Ok(Self::Nearest),
            "linear" | "multilinear" => Ok(Self::Linear),
            _ => Err(LadduError::ParseError {
                name: value.to_string(),
                object: "LookupInterpolation".to_string(),
            }),
        }
    }
}

impl LookupInterpolation {
    fn table_len(self, axis_coordinates: &[LookupAxis]) -> LadduResult<usize> {
        axis_coordinates.iter().try_fold(1usize, |acc, axis| {
            acc.checked_mul(axis.value_count(self))
                .ok_or_else(|| LadduError::Custom("lookup-table shape is too large".to_string()))
        })
    }

    fn vertex_count(self, ndim: usize) -> LadduResult<usize> {
        match self {
            Self::Nearest => Ok(1),
            Self::Linear => (0..ndim).try_fold(1usize, |acc, _| {
                acc.checked_mul(2).ok_or_else(|| {
                    LadduError::Custom(
                        "lookup-table interpolation vertex count is too large".to_string(),
                    )
                })
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

/// A lookup-table axis defined by monotonically increasing coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookupAxis {
    coordinates: Vec<f64>,
}

impl LookupAxis {
    /// Create an axis from coordinates.
    ///
    /// For nearest interpolation these coordinates are interpreted as bin edges, so `N + 1`
    /// coordinates define `N` table values. For linear interpolation they are interpreted as grid
    /// points, so `N` coordinates define `N` table values.
    pub fn new(coordinates: Vec<f64>) -> LadduResult<Self> {
        if coordinates.len() < 2 {
            return Err(LadduError::LengthMismatch {
                context: "lookup-table axis coordinates".to_string(),
                expected: 2,
                actual: coordinates.len(),
            });
        }
        if !coordinates.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(LadduError::Custom(
                "lookup-table axis coordinates must be finite".to_string(),
            ));
        }
        if !coordinates.windows(2).all(|window| window[0] < window[1]) {
            return Err(LadduError::Custom(
                "lookup-table axis coordinates must be strictly increasing".to_string(),
            ));
        }
        Ok(Self { coordinates })
    }

    /// Return the number of bins on this axis.
    pub fn bin_count(&self) -> usize {
        self.coordinates.len() - 1
    }

    fn point_count(&self) -> usize {
        self.coordinates.len()
    }

    fn value_count(&self, interpolation: LookupInterpolation) -> usize {
        match interpolation {
            LookupInterpolation::Nearest => self.bin_count(),
            LookupInterpolation::Linear => self.point_count(),
        }
    }

    fn bin_index(&self, value: f64, boundary_mode: LookupBoundaryMode) -> Option<usize> {
        if !value.is_finite() {
            return None;
        }
        let bins = self.bin_count();
        if value < self.coordinates[0] {
            return match boundary_mode {
                LookupBoundaryMode::Zero => None,
                LookupBoundaryMode::Clamp => Some(0),
            };
        }
        if value >= self.coordinates[bins] {
            return match boundary_mode {
                LookupBoundaryMode::Zero => None,
                LookupBoundaryMode::Clamp => Some(bins - 1),
            };
        }
        match self.coordinates.binary_search_by(|coordinate| {
            coordinate
                .partial_cmp(&value)
                .expect("finite coordinate and value")
        }) {
            Ok(index) => Some(index.min(bins - 1)),
            Err(index) => Some(index - 1),
        }
    }

    fn linear_cell(&self, value: f64, boundary_mode: LookupBoundaryMode) -> Option<(usize, f64)> {
        if !value.is_finite() {
            return None;
        }
        let last = self.coordinates.len() - 1;
        if value < self.coordinates[0] {
            return match boundary_mode {
                LookupBoundaryMode::Zero => None,
                LookupBoundaryMode::Clamp => Some((0, 0.0)),
            };
        }
        if value > self.coordinates[last] {
            return match boundary_mode {
                LookupBoundaryMode::Zero => None,
                LookupBoundaryMode::Clamp => Some((last - 1, 1.0)),
            };
        }
        match self.coordinates.binary_search_by(|coordinate| {
            coordinate
                .partial_cmp(&value)
                .expect("finite coordinate and value")
        }) {
            Ok(index) if index == last => Some((last - 1, 1.0)),
            Ok(index) => Some((index, 0.0)),
            Err(index) => {
                let lower = index - 1;
                let t = (value - self.coordinates[lower])
                    / (self.coordinates[lower + 1] - self.coordinates[lower]);
                Some((lower, t))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum LookupValues {
    FixedComplex(Vec<Complex64>),
    Scalar {
        values: Vec<Parameter>,
        pids: Vec<ParameterID>,
    },
    CartesianComplex {
        values: Vec<(Parameter, Parameter)>,
        pids: Vec<(ParameterID, ParameterID)>,
    },
    PolarComplex {
        values: Vec<(Parameter, Parameter)>,
        pids: Vec<(ParameterID, ParameterID)>,
    },
}

impl LookupValues {
    fn fixed_complex(values: Vec<Complex64>) -> Self {
        Self::FixedComplex(values)
    }

    fn scalar(values: Vec<Parameter>) -> Self {
        Self::Scalar {
            values,
            pids: Vec::new(),
        }
    }

    fn cartesian_complex(values: Vec<(Parameter, Parameter)>) -> Self {
        Self::CartesianComplex {
            values,
            pids: Vec::new(),
        }
    }

    fn polar_complex(values: Vec<(Parameter, Parameter)>) -> Self {
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

    fn gradient(
        &self,
        index: usize,
        weight: f64,
        parameters: &Parameters,
        gradient: &mut DVector<Complex64>,
    ) {
        match self {
            Self::FixedComplex(_) => {}
            Self::Scalar { pids, .. } => {
                if let Some(ind) = parameters.free_index(pids[index]) {
                    gradient[ind] += weight * Complex64::ONE;
                }
            }
            Self::CartesianComplex { pids, .. } => {
                if let Some(ind) = parameters.free_index(pids[index].0) {
                    gradient[ind] += weight * Complex64::ONE;
                }
                if let Some(ind) = parameters.free_index(pids[index].1) {
                    gradient[ind] += weight * Complex64::I;
                }
            }
            Self::PolarComplex { pids, .. } => {
                let r = parameters.get(pids[index].0);
                let theta = parameters.get(pids[index].1);
                let exp_i_theta = Complex64::cis(theta);
                if let Some(ind) = parameters.free_index(pids[index].0) {
                    gradient[ind] += weight * exp_i_theta;
                }
                if let Some(ind) = parameters.free_index(pids[index].1) {
                    gradient[ind] += weight * Complex64::I * Complex64::from_polar(r, theta);
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
    axis_coordinates: Vec<LookupAxis>,
    values: LookupValues,
    interpolation: LookupInterpolation,
    boundary_mode: LookupBoundaryMode,
    vertex_ids: Vec<ScalarID>,
    weight_ids: Vec<ScalarID>,
}

impl LookupTable {
    /// Create a fixed complex lookup table.
    pub fn new(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axis_coordinates: Vec<LookupAxis>,
        values: Vec<Complex64>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axis_coordinates,
            LookupValues::fixed_complex(values),
            interpolation,
            boundary_mode,
        )
    }

    /// Create a parameterized scalar lookup table.
    pub fn new_scalar(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axis_coordinates: Vec<LookupAxis>,
        values: Vec<Parameter>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axis_coordinates,
            LookupValues::scalar(values),
            interpolation,
            boundary_mode,
        )
    }

    /// Create a parameterized cartesian complex lookup table.
    pub fn new_cartesian_complex(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axis_coordinates: Vec<LookupAxis>,
        values: Vec<(Parameter, Parameter)>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axis_coordinates,
            LookupValues::cartesian_complex(values),
            interpolation,
            boundary_mode,
        )
    }

    /// Create a parameterized polar complex lookup table.
    pub fn new_polar_complex(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axis_coordinates: Vec<LookupAxis>,
        values: Vec<(Parameter, Parameter)>,
        interpolation: LookupInterpolation,
        boundary_mode: LookupBoundaryMode,
    ) -> LadduResult<Expression> {
        Self::with_values(
            name,
            variables,
            axis_coordinates,
            LookupValues::polar_complex(values),
            interpolation,
            boundary_mode,
        )
    }

    fn with_values(
        name: &str,
        variables: Vec<Box<dyn Variable>>,
        axis_coordinates: Vec<LookupAxis>,
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
        if variables.len() != axis_coordinates.len() {
            return Err(LadduError::LengthMismatch {
                context: "lookup-table axis coordinates".to_string(),
                expected: variables.len(),
                actual: axis_coordinates.len(),
            });
        }
        let expected = interpolation.table_len(&axis_coordinates)?;
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
            axis_coordinates,
            values,
            interpolation,
            boundary_mode,
            vertex_ids: Vec::new(),
            weight_ids: Vec::new(),
        }
        .into_expression()
    }

    fn weighted_indices(&self, event: &Event<'_>) -> Option<Vec<(usize, f64)>> {
        match self.interpolation {
            LookupInterpolation::Nearest => {
                self.nearest_index(event).map(|index| vec![(index, 1.0)])
            }
            LookupInterpolation::Linear => self.linear_indices(event),
        }
    }

    fn nearest_index(&self, event: &Event<'_>) -> Option<usize> {
        let mut flat_index = 0usize;
        for (variable, axis) in self.variables.iter().zip(&self.axis_coordinates) {
            let bin_index = axis.bin_index(variable.value(event), self.boundary_mode)?;
            flat_index = flat_index * axis.bin_count() + bin_index;
        }
        Some(flat_index)
    }

    fn linear_indices(&self, event: &Event<'_>) -> Option<Vec<(usize, f64)>> {
        let mut cells = Vec::with_capacity(self.axis_coordinates.len());
        for (variable, axis) in self.variables.iter().zip(&self.axis_coordinates) {
            cells.push(axis.linear_cell(variable.value(event), self.boundary_mode)?);
        }
        let vertex_count = self
            .interpolation
            .vertex_count(self.axis_coordinates.len())
            .ok()?;
        let mut weighted_indices = Vec::with_capacity(vertex_count);
        for vertex in 0..vertex_count {
            let mut flat_index = 0usize;
            let mut weight = 1.0;
            for (axis_index, ((lower_index, t), axis)) in
                cells.iter().zip(&self.axis_coordinates).enumerate()
            {
                let high = (vertex >> (self.axis_coordinates.len() - axis_index - 1)) & 1 == 1;
                if high {
                    flat_index = flat_index * axis.point_count() + lower_index + 1;
                    weight *= *t;
                } else {
                    flat_index = flat_index * axis.point_count() + lower_index;
                    weight *= 1.0 - *t;
                }
            }
            weighted_indices.push((flat_index, weight));
        }
        Some(weighted_indices)
    }
}

#[typetag::serde]
impl Amplitude for LookupTable {
    fn register(&mut self, resources: &mut Resources) -> LadduResult<AmplitudeID> {
        self.values.register(resources)?;
        let vertex_count = self
            .interpolation
            .vertex_count(self.axis_coordinates.len())?;
        self.vertex_ids = (0..vertex_count)
            .map(|index| resources.register_scalar(Some(&format!("{}.vertex_{index}", self.name))))
            .collect();
        self.weight_ids = (0..vertex_count)
            .map(|index| resources.register_scalar(Some(&format!("{}.weight_{index}", self.name))))
            .collect();
        resources.register_amplitude(&self.name)
    }

    fn semantic_key(&self) -> Option<AmplitudeSemanticKey> {
        Some(
            AmplitudeSemanticKey::new("LookupTable")
                .with_field("name", debug_key(&self.name))
                .with_field("variables", debug_key(&self.variables))
                .with_field("axis_coordinates", debug_key(&self.axis_coordinates))
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

    fn precompute(&self, event: &Event<'_>, cache: &mut Cache) {
        let weighted_indices = self.weighted_indices(event);
        for (slot, (vertex_id, weight_id)) in
            self.vertex_ids.iter().zip(&self.weight_ids).enumerate()
        {
            let (index, weight) = weighted_indices
                .as_ref()
                .and_then(|indices| indices.get(slot).copied())
                .unwrap_or((0, 0.0));
            cache.store_scalar(*vertex_id, index as f64);
            cache.store_scalar(*weight_id, weight);
        }
    }

    fn compute(&self, parameters: &Parameters, cache: &Cache) -> Complex64 {
        self.vertex_ids
            .iter()
            .zip(&self.weight_ids)
            .map(|(vertex_id, weight_id)| {
                let weight = cache.get_scalar(*weight_id);
                if weight == 0.0 {
                    Complex64::ZERO
                } else {
                    let index = cache.get_scalar(*vertex_id) as usize;
                    weight * self.values.value(index, parameters)
                }
            })
            .sum()
    }

    fn compute_gradient(
        &self,
        parameters: &Parameters,
        cache: &Cache,
        gradient: &mut DVector<Complex64>,
    ) {
        for (vertex_id, weight_id) in self.vertex_ids.iter().zip(&self.weight_ids) {
            let weight = cache.get_scalar(*weight_id);
            if weight != 0.0 {
                let index = cache.get_scalar(*vertex_id) as usize;
                self.values.gradient(index, weight, parameters, gradient);
            }
        }
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
/// axis_coordinates : list of list of float
///     Per-variable axis coordinates. Nearest interpolation treats them as bin edges; linear
///     interpolation treats them as grid points.
/// values : list of complex
///     Flattened row-major table values.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Supports "nearest" and "linear".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
/// An amplitude which evaluates a scalar-parameter lookup table over event variables.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variables : list of laddu variables
///     The event variables that define the lookup coordinates.
/// axis_coordinates : list of list of float
///     Per-variable axis coordinates. Nearest interpolation treats them as bin edges; linear
///     interpolation treats them as grid points.
/// values : list of laddu.Parameter
///     Flattened row-major scalar parameters.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Supports "nearest" and "linear".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
/// An amplitude which evaluates a cartesian-complex-parameter lookup table over event variables.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variables : list of laddu variables
///     The event variables that define the lookup coordinates.
/// axis_coordinates : list of list of float
///     Per-variable axis coordinates. Nearest interpolation treats them as bin edges; linear
///     interpolation treats them as grid points.
/// values : list of tuple of laddu.Parameter
///     Flattened row-major real and imaginary parameter pairs.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Supports "nearest" and "linear".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
/// An amplitude which evaluates a polar-complex-parameter lookup table over event variables.
///
/// Parameters
/// ----------
/// name : str
///     The Amplitude name.
/// variables : list of laddu variables
///     The event variables that define the lookup coordinates.
/// axis_coordinates : list of list of float
///     Per-variable axis coordinates. Nearest interpolation treats them as bin edges; linear
///     interpolation treats them as grid points.
/// values : list of tuple of laddu.Parameter
///     Flattened row-major magnitude and phase parameter pairs.
/// interpolation : str, default: "nearest"
///     Interpolation mode. Supports "nearest" and "linear".
/// boundary_mode : str, default: "zero"
///     Out-of-range behavior. Currently supports "zero" and "clamp".
///
/// Returns
/// -------
/// laddu.Expression
///     An Expression which evaluates the lookup table on each event.
///
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
        let result = evaluator.evaluate(&[]).unwrap();

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
        let result = evaluator.evaluate(&[]).unwrap();

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
        let result = evaluator.evaluate(&[]).unwrap();

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
        let result = evaluator.evaluate(&[]).unwrap();

        assert_relative_eq!(result[0].re, 2.0);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_lookup_table_1d_linear() {
        let expr = LookupTable::new(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 1.0]).unwrap()],
            vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            LookupInterpolation::Linear,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]).unwrap();

        assert_relative_eq!(result[0].re, 1.0 + 2.0 * 0.498);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_lookup_table_2d_linear_row_major() {
        let expr = LookupTable::new(
            "lookup",
            vec![mass("kshort1"), mass("proton")],
            vec![
                LookupAxis::new(vec![0.0, 1.0]).unwrap(),
                LookupAxis::new(vec![1.0, 2.0]).unwrap(),
            ],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
            LookupInterpolation::Linear,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]).unwrap();

        assert_relative_eq!(result[0].re, 1.0 + 2.0 * 0.498 + 3.0 * 0.007);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_lookup_table_linear_clamp_boundary() {
        let expr = LookupTable::new(
            "lookup",
            vec![Box::new(Mass::new(["kshort1", "kshort2"]))],
            vec![LookupAxis::new(vec![0.0, 1.0]).unwrap()],
            vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            LookupInterpolation::Linear,
            LookupBoundaryMode::Clamp,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]).unwrap();

        assert_relative_eq!(result[0].re, 3.0);
        assert_relative_eq!(result[0].im, 0.0);
    }

    #[test]
    fn test_lookup_table_linear_zero_boundary() {
        let expr = LookupTable::new(
            "lookup",
            vec![Box::new(Mass::new(["kshort1", "kshort2"]))],
            vec![LookupAxis::new(vec![0.0, 1.0]).unwrap()],
            vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            LookupInterpolation::Linear,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate(&[]).unwrap();

        assert_eq!(result[0], Complex64::ZERO);
    }

    #[test]
    fn test_lookup_table_scalar_parameters_and_gradient() {
        let expr = LookupTable::new_scalar(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 0.25, 0.75, 1.0]).unwrap()],
            vec![parameter!("p0"), parameter!("p1"), parameter!("p2")],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
        .unwrap()
        .norm_sqr();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate_gradient(&[1.0, 2.0, 3.0]).unwrap();

        assert_relative_eq!(result[0][0].re, 0.0);
        assert_relative_eq!(result[0][0].im, 0.0);
        assert_relative_eq!(result[0][1].re, 4.0);
        assert_relative_eq!(result[0][1].im, 0.0);
        assert_relative_eq!(result[0][2].re, 0.0);
        assert_relative_eq!(result[0][2].im, 0.0);
    }

    #[test]
    fn test_lookup_table_linear_scalar_parameters_and_gradient() {
        let expr = LookupTable::new_scalar(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 1.0]).unwrap()],
            vec![parameter!("p0"), parameter!("p1")],
            LookupInterpolation::Linear,
            LookupBoundaryMode::Zero,
        )
        .unwrap()
        .norm_sqr();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let result = evaluator.evaluate_gradient(&[1.0, 3.0]).unwrap();
        let value = 0.502 * 1.0 + 0.498 * 3.0;

        assert_relative_eq!(result[0][0].re, 2.0 * value * 0.502, epsilon = 1e-12);
        assert_relative_eq!(result[0][0].im, 0.0);
        assert_relative_eq!(result[0][1].re, 2.0 * value * 0.498, epsilon = 1e-12);
        assert_relative_eq!(result[0][1].im, 0.0);
    }

    #[test]
    fn test_lookup_table_polar_parameters_and_gradient() {
        let expr = LookupTable::new_polar_complex(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 0.25, 0.75, 1.0]).unwrap()],
            vec![
                (parameter!("r0"), parameter!("theta0")),
                (parameter!("r1"), parameter!("theta1")),
                (parameter!("r2"), parameter!("theta2")),
            ],
            LookupInterpolation::Nearest,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let params = [1.1, 1.2, 2.1, 2.2, 3.1, 3.2];
        let gradient = evaluator.evaluate_gradient(&params).unwrap();

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
    fn test_lookup_table_linear_complex_parameters_and_gradient() {
        let expr = LookupTable::new_cartesian_complex(
            "lookup",
            vec![mass("kshort1")],
            vec![LookupAxis::new(vec![0.0, 1.0]).unwrap()],
            vec![
                (parameter!("re0"), parameter!("im0")),
                (parameter!("re1"), parameter!("im1")),
            ],
            LookupInterpolation::Linear,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let gradient = evaluator.evaluate_gradient(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_relative_eq!(gradient[0][0].re, 0.502, epsilon = 1e-12);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 0.502, epsilon = 1e-12);
        assert_relative_eq!(gradient[0][2].re, 0.498, epsilon = 1e-12);
        assert_relative_eq!(gradient[0][2].im, 0.0);
        assert_relative_eq!(gradient[0][3].re, 0.0);
        assert_relative_eq!(gradient[0][3].im, 0.498, epsilon = 1e-12);
    }

    #[test]
    fn test_lookup_table_linear_zero_boundary_has_zero_gradient() {
        let expr = LookupTable::new_scalar(
            "lookup",
            vec![Box::new(Mass::new(["kshort1", "kshort2"]))],
            vec![LookupAxis::new(vec![0.0, 1.0]).unwrap()],
            vec![parameter!("p0"), parameter!("p1")],
            LookupInterpolation::Linear,
            LookupBoundaryMode::Zero,
        )
        .unwrap();

        let dataset = Arc::new(test_dataset());
        let evaluator = expr.load(&dataset).unwrap();
        let gradient = evaluator.evaluate_gradient(&[1.0, 3.0]).unwrap();

        assert_relative_eq!(gradient[0][0].re, 0.0);
        assert_relative_eq!(gradient[0][0].im, 0.0);
        assert_relative_eq!(gradient[0][1].re, 0.0);
        assert_relative_eq!(gradient[0][1].im, 0.0);
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
