use laddu_runtime::{BinSpec, IntervalClosure, Predicate};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};

use super::{error::to_py_err, expr::PyExpr, float_vec};

#[pyclass(name = "Predicate", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
/// A symbolic boolean condition used to filter or bin datasets.
///
/// Predicates are normally created by comparing :class:`Expr` objects. Combine
/// them with ``&`` (and), ``|`` (or), and ``~`` (not).
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> mass = ld.scalar("mass")
/// >>> sideband = (mass < 0.9) | (mass > 1.1)
pub struct PyPredicate {
    pub(crate) inner: Predicate,
}

#[pymethods]
impl PyPredicate {
    fn __and__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone().and(other.inner.clone()),
        }
    }
    fn __or__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone().or(other.inner.clone()),
        }
    }
    fn __invert__(&self) -> Self {
        Self {
            inner: !self.inner.clone(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (value, *, low, high, closed="both"))]
    /// Test whether an expression lies inside an interval.
    ///
    /// Parameters
    /// ----------
    /// value, low, high : Expr
    ///     Value and interval endpoints.
    /// closed : {"both", "neither", "left", "right"}, default="both"
    ///     Select which endpoints are included.
    ///
    /// Returns
    /// -------
    /// Predicate
    ///     Symbolic interval condition.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `closed` is not a supported closure mode.
    fn between(value: &PyExpr, low: &PyExpr, high: &PyExpr, closed: &str) -> PyResult<Self> {
        let closure = match closed {
            "both" => IntervalClosure::Closed,
            "neither" => IntervalClosure::Open,
            "left" => IntervalClosure::LeftClosed,
            "right" => IntervalClosure::RightClosed,
            _ => {
                return Err(PyValueError::new_err(
                    "closed must be 'both', 'neither', 'left', or 'right'",
                ));
            }
        };
        Ok(Self {
            inner: Predicate::between_with(
                value.inner.clone(),
                low.inner.clone(),
                high.inner.clone(),
                closure,
            ),
        })
    }
}

#[pyclass(name = "Bin", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A one-dimensional bin-edge specification.
///
/// Parameters
/// ----------
/// edges : sequence of float
///     Strictly increasing bin edges. `N + 1` edges define `N` bins.
pub struct PyBin {
    pub(crate) inner: BinSpec,
}

#[pymethods]
impl PyBin {
    /// Construct bins from explicit edges.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If fewer than two edges are provided or edges are not finite and
    ///     strictly increasing.
    #[new]
    #[pyo3(signature = (
        edges: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64]"
    ))]
    fn new(edges: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: BinSpec::edges(float_vec(edges)?).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    /// Construct uniformly spaced bins.
    ///
    /// Parameters
    /// ----------
    /// count : int
    ///     Positive number of bins.
    /// low, high : float
    ///     Finite limits with ``low < high``.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the count or limits are invalid.
    #[pyo3(signature = (count, *, low, high))]
    fn uniform(count: usize, low: f64, high: f64) -> PyResult<Self> {
        Ok(Self {
            inner: BinSpec::uniform(count, low, high).map_err(to_py_err)?,
        })
    }

    #[getter]
    /// list of float: A copy of the bin edges.
    fn edges(&self) -> Vec<f64> {
        self.inner.edges_slice().to_vec()
    }
}
