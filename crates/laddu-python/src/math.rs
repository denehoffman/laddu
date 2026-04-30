use laddu_core::{math::Histogram, LadduError};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyTuple},
};

fn extract_f64_vec(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_slice()?.to_vec());
    }
    value
        .extract::<Vec<f64>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a one-dimensional float array")))
}

/// A weighted histogram with explicit bin edges.
///
/// Parameters
/// ----------
/// bin_edges : array_like
///     Strictly increasing bin edges. The length must be one greater than ``counts``.
/// counts : array_like
///     Finite nonnegative weighted counts. The total weight must be positive.
///
#[pyclass(name = "Histogram", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHistogram(pub Histogram);

#[pymethods]
impl PyHistogram {
    #[new]
    fn new(bin_edges: &Bound<'_, PyAny>, counts: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(Histogram::new(
            extract_f64_vec(bin_edges, "bin_edges")?,
            extract_f64_vec(counts, "counts")?,
        )?))
    }

    /// Construct a histogram from NumPy-compatible arrays.
    #[staticmethod]
    fn from_numpy(bin_edges: &Bound<'_, PyAny>, counts: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::new(bin_edges, counts)
    }

    /// Bin edges as a NumPy array.
    #[getter]
    fn bin_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.0.bin_edges())
    }

    /// Weighted counts as a NumPy array.
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.0.counts())
    }

    /// Total histogram weight.
    #[getter]
    fn total_weight(&self) -> f64 {
        self.0.total_weight()
    }

    /// Return ``(bin_edges, counts)`` as NumPy arrays.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [self.bin_edges(py).into_any(), self.counts(py).into_any()],
        )
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            serde_pickle::to_vec(&self.0, serde_pickle::SerOptions::new())
                .map_err(LadduError::PickleError)?
                .as_slice(),
        ))
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [self.0.bin_edges().to_vec(), self.0.counts().to_vec()])
    }

    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = Self(
            serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
                .map_err(LadduError::PickleError)?,
        );
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}
