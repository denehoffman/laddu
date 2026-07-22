use laddu_physics::histogram::Histogram;
use pyo3::prelude::*;

use super::error::to_py_err;

#[pyclass(name = "Histogram", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A weighted one-dimensional histogram with explicit flow bins.
///
/// Parameters
/// ----------
/// counts : sequence of float
///     Weight stored in each regular bin.
/// bin_edges : sequence of float
///     Strictly increasing edges; there must be one more edge than count.
/// underflow, overflow : float, default=0.0
///     Weight outside the regular bin range.
pub struct PyHistogram {
    pub(crate) inner: Histogram,
}

#[pymethods]
impl PyHistogram {
    /// Construct a histogram from precomputed bin contents.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If counts and edges have inconsistent lengths or invalid values.
    #[new]
    #[pyo3(signature = (counts, bin_edges, *, underflow=0.0, overflow=0.0))]
    fn new(counts: Vec<f64>, bin_edges: Vec<f64>, underflow: f64, overflow: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Histogram::new_with_flow(counts, bin_edges, underflow, overflow)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (values, bins, limits, *, weights=None))]
    /// Histogram samples into uniform bins.
    ///
    /// Parameters
    /// ----------
    /// values : sequence of float
    ///     Samples to bin.
    /// bins : int
    ///     Positive number of equal-width bins.
    /// limits : tuple of float
    ///     Lower and upper histogram bounds.
    /// weights : sequence of float, optional
    ///     Per-sample weights; defaults to one.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If limits, bin count, values, or weights are invalid.
    fn from_values(
        values: Vec<f64>,
        bins: usize,
        limits: (f64, f64),
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Histogram::from_values(&values, bins, limits, weights.as_deref())
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (values, bin_edges, *, weights=None))]
    /// Histogram samples using explicit bin edges.
    ///
    /// Parameters
    /// ----------
    /// values : sequence of float
    ///     Samples to bin.
    /// bin_edges : sequence of float
    ///     Strictly increasing edges.
    /// weights : sequence of float, optional
    ///     Per-sample weights.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If edges or weights are invalid.
    fn from_values_with_edges(
        values: Vec<f64>,
        bin_edges: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Histogram::from_values_with_edges(&values, bin_edges, weights.as_deref())
                .map_err(to_py_err)?,
        })
    }

    #[getter]
    /// list of float: Regular-bin contents.
    fn counts(&self) -> Vec<f64> {
        self.inner.counts().to_vec()
    }

    #[getter]
    /// list of float: Bin edges.
    fn bin_edges(&self) -> Vec<f64> {
        self.inner.bin_edges().to_vec()
    }

    #[getter]
    /// float: Accumulated weight below the first edge.
    fn underflow(&self) -> f64 {
        self.inner.underflow()
    }

    #[getter]
    /// float: Accumulated weight at or above the final edge.
    fn overflow(&self) -> f64 {
        self.inner.overflow()
    }

    #[getter]
    /// int: Number of regular bins.
    fn bins(&self) -> usize {
        self.inner.bins()
    }

    #[getter]
    /// tuple of float: Lower and upper histogram limits.
    fn limits(&self) -> (f64, f64) {
        self.inner.limits()
    }

    #[pyo3(signature = (*, include_flow=false))]
    /// Return the sum of bin weights.
    ///
    /// Parameters
    /// ----------
    /// include_flow : bool, default=False
    ///     Include underflow and overflow weights.
    fn total_weight(&self, include_flow: bool) -> f64 {
        if include_flow {
            self.inner.total_weight_with_flow()
        } else {
            self.inner.total_weight()
        }
    }

    /// Return the regular-bin index containing a value, or ``None`` for flow.
    fn bin_index(&self, value: f64) -> Option<usize> {
        self.inner.bin_index(value)
    }

    /// Return a regular bin's center, or ``None`` for an invalid index.
    fn bin_center(&self, index: usize) -> Option<f64> {
        self.inner.bin_center(index)
    }

    #[pyo3(signature = (*, include_flow=false))]
    /// Return a copy normalized to unit total weight.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the selected total weight is zero or nonfinite.
    fn normalized(&self, include_flow: bool) -> PyResult<Self> {
        let inner = if include_flow {
            self.inner.normalized_with_flow()
        } else {
            self.inner.normalized()
        };
        Ok(Self {
            inner: inner.map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (*, signed=false))]
    /// Convert bin weights to a probability density.
    ///
    /// Parameters
    /// ----------
    /// signed : bool, default=False
    ///     Preserve a signed total instead of requiring positive normalization.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If normalization is undefined or a bin width is invalid.
    fn density(&self, signed: bool) -> PyResult<Self> {
        let inner = if signed {
            self.inner.signed_density()
        } else {
            self.inner.density()
        };
        Ok(Self {
            inner: inner.map_err(to_py_err)?,
        })
    }
}
