use laddu_core::data::Dataset;
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyTypeError, prelude::*};
use pyo3_polars::{PyExpr, PyLazyFrame};
use std::{path::PathBuf, sync::Arc};

/// A set of Events
///
/// Datasets can be created from lists of Events or by using the provided ``laddu.open`` function
///
/// Datasets can also be indexed directly to access individual Events
///
/// Parameters
/// ----------
/// events : list of Event
///
/// See Also
/// --------
/// laddu.open
///
#[pyclass(name = "Dataset", module = "laddu")]
#[derive(Clone)]
pub struct PyDataset(pub Dataset);

#[pymethods]
impl PyDataset {
    #[new]
    fn new(lazyframe: PyLazyFrame) -> Self {
        Self(Dataset::new(lazyframe.into()))
    }
    #[staticmethod]
    fn open(path: Bound<'_, PyAny>) -> PyResult<PyDataset> {
        let path_str = if let Ok(s) = path.extract::<String>() {
            Ok(s)
        } else if let Ok(pathbuf) = path.extract::<PathBuf>() {
            Ok(pathbuf.to_string_lossy().to_string())
        } else {
            Err(PyTypeError::new_err("Expected a str or Path"))
        }?;
        Ok(PyDataset(Dataset::open(path_str)?))
    }
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.n_events()?)
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataset> {
        if let Ok(other_ds) = other.extract::<PyRef<PyDataset>>() {
            Ok(PyDataset((&self.0 + &other_ds.0)?))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported operand type(s) for +: 'Dataset' and {}",
                other.get_type().name()?
            )))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataset> {
        if let Ok(other_ds) = other.extract::<PyRef<PyDataset>>() {
            Ok(PyDataset((&self.0 + &other_ds.0)?))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported operand type(s) for +: 'Dataset' and {}",
                other.get_type().name()?
            )))
        }
    }
    /// Get the number of Events in the Dataset
    ///
    /// Returns
    /// -------
    /// n_events : int
    ///     The number of Events
    ///
    fn n_events(&self) -> PyResult<usize> {
        Ok(self.0.n_events()?)
    }
    /// Get the weighted number of Events in the Dataset
    ///
    /// Returns
    /// -------
    /// n_events : float
    ///     The sum of all Event weights
    ///
    fn n_events_weighted(&self) -> PyResult<f64> {
        Ok(self.0.n_events_weighted()?)
    }
    /// The weights associated with the Dataset
    ///
    /// Returns
    /// -------
    /// weights : array_like
    ///     A ``numpy`` array of Event weights
    ///
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.0.weights()?.into_pyarray(py))
    }
    /// Separates a Dataset into histogram bins by a Variable value
    ///
    /// Parameters
    /// ----------
    /// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
    ///     The Variable by which each Event is binned
    /// bins : int
    ///     The number of equally-spaced bins
    /// range : tuple[float, float]
    ///     The minimum and maximum bin edges
    ///
    /// Returns
    /// -------
    /// datasets : BinnedDataset
    ///     A pub structure that holds a list of Datasets binned by the given `variable`
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
    /// Raises
    /// ------
    /// TypeError
    ///     If the given `variable` is not a valid variable
    ///
    #[pyo3(signature = (variable, bins, limits))]
    fn bin_by(
        &self,
        variable: Bound<'_, PyAny>,
        bins: usize,
        limits: (f64, f64),
    ) -> PyResult<(Vec<PyDataset>, Vec<f64>)> {
        let py_expr = variable.extract::<PyExpr>()?;
        let (datasets, edges) = self.0.bin_by(&py_expr.0, bins, limits)?;
        Ok((
            datasets.into_iter().map(|ds| PyDataset(ds)).collect(),
            edges,
        ))
    }
    /// Filter the Dataset by a given VariableExpression, selecting events for which the expression returns ``True``.
    ///
    /// Parameters
    /// ----------
    /// expression : VariableExpression
    ///     The expression with which to filter the Dataset
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     The filtered Dataset
    ///
    pub fn filter(&self, expr: &Bound<'_, PyAny>) -> PyResult<PyDataset> {
        Ok(PyDataset(self.0.filter(&expr.extract::<PyExpr>()?.0)?))
    }
    /// Generate a new bootstrapped Dataset by randomly resampling the original with replacement
    ///
    /// The new Dataset is resampled with a random generator seeded by the provided `seed`
    ///
    /// Parameters
    /// ----------
    /// seed : int
    ///     The random seed used in the resampling process
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     A bootstrapped Dataset
    ///
    fn bootstrap(&self, seed: usize) -> PyResult<PyDataset> {
        Ok(PyDataset(self.0.bootstrap(seed)?))
    }
    // /// Boost all the four-momenta in all events to the rest frame of the given set of
    // /// four-momenta by indices.
    // ///
    // /// Parameters
    // /// ----------
    // /// indices : list of int
    // ///     The indices of the four-momenta to sum
    // ///
    // /// Returns
    // /// -------
    // /// Dataset
    // ///     The boosted dataset
    // ///
    // pub fn boost_to_rest_frame_of(&self, indices: Vec<usize>) -> PyDataset {
    //     PyDataset(self.0.boost_to_rest_frame_of(indices))
    // }
    /// Get the value of a Variable over every event in the Dataset.
    ///
    /// Parameters
    /// ----------
    /// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        expr: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .0
            .evaluate(&expr.extract::<PyExpr>()?.0)?
            .into_pyarray(py))
    }
}
