use laddu_physics::histogram::Histogram;
use pyo3::{prelude::*, types::PyAny};

use super::{error::to_py_err, float_vec};

#[pyclass(name = "Histogram", module = "laddu", skip_from_py_object)]
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
/// errors : sequence of float, optional
///     Per-bin uncertainties. Defaults to the square root of the absolute count.
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
    #[pyo3(signature = (
        counts: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64]",
        *,
        bin_edges: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64]",
        underflow=0.0,
        overflow=0.0,
        errors: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | None" = None
    ))]
    fn new(
        counts: &Bound<'_, PyAny>,
        bin_edges: &Bound<'_, PyAny>,
        underflow: f64,
        overflow: f64,
        errors: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut hist = Histogram::new_with_flow(
            float_vec(counts)?,
            float_vec(bin_edges)?,
            underflow,
            overflow,
        )
        .map_err(to_py_err)?;
        if let Some(errors) = errors {
            let errors = float_vec(errors)?;
            hist.set_errors(&errors).map_err(to_py_err)?;
        }

        Ok(Self { inner: hist })
    }

    #[staticmethod]
    #[pyo3(signature = (
        values: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64]",
        *,
        bins,
        limits,
        weights: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | None" = None
    ))]
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
        values: &Bound<'_, PyAny>,
        bins: usize,
        limits: (f64, f64),
        weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let values = float_vec(values)?;
        let weights = weights.map(float_vec).transpose()?;
        Ok(Self {
            inner: Histogram::from_values(&values, bins, limits, weights.as_deref())
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (
        values: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64]",
        *,
        bin_edges: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64]",
        weights: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | None" = None
    ))]
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
        values: &Bound<'_, PyAny>,
        bin_edges: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let values = float_vec(values)?;
        let bin_edges = float_vec(bin_edges)?;
        let weights = weights.map(float_vec).transpose()?;
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

    #[setter(counts)]
    /// Replace all regular-bin contents without changing their uncertainties.
    fn set_counts(&mut self, counts: &Bound<'_, PyAny>) -> PyResult<()> {
        let counts = float_vec(counts)?;
        self.inner.set_counts(&counts).map_err(to_py_err)
    }

    #[getter]
    /// list of float: Uncertainties in each bin.
    fn errors(&self) -> Vec<f64> {
        self.inner.errors().to_vec()
    }

    #[setter(errors)]
    /// Replace all regular-bin uncertainties.
    fn set_errors(&mut self, errors: &Bound<'_, PyAny>) -> PyResult<()> {
        let errors = float_vec(errors)?;
        self.inner.set_errors(&errors).map_err(to_py_err)
    }

    #[pyo3(signature = (index, *, count=None, error=None))]
    /// Set the count and/or uncertainty for one regular bin.
    fn set_bin(&mut self, index: usize, count: Option<f64>, error: Option<f64>) -> PyResult<()> {
        let mut updated = self.inner.clone();
        if let Some(count) = count {
            updated.set_count(index, count).map_err(to_py_err)?;
        }
        if let Some(error) = error {
            updated.set_error(index, error).map_err(to_py_err)?;
        }
        self.inner = updated;
        Ok(())
    }

    #[pyo3(signature = (value, *, weight=1.0, error=None))]
    /// Add a value with an optional weight and explicit uncertainty.
    ///
    /// If ``error`` is omitted, the weight contributes to the bin uncertainty
    /// in quadrature.
    fn fill(&mut self, value: f64, weight: f64, error: Option<f64>) -> PyResult<()> {
        if let Some(error) = error {
            self.inner
                .fill_weighted_with_error(value, weight, error)
                .map_err(to_py_err)
        } else {
            self.inner.fill_weighted(value, weight).map_err(to_py_err)
        }
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

impl_json_methods!(PyHistogram);
