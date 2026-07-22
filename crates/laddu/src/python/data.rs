use std::{path::PathBuf, sync::Arc};

use laddu_data::{
    data::{CacheStorage, Dataset, EventBatch},
    io::{
        parquet::{ParquetSink, ParquetSource},
        root::{RootSink, RootSource},
    },
    schema::Schema,
};
use laddu_physics::vectors::RealVec4;
use laddu_runtime::DatasetExprExt;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};

use super::{
    error::to_py_err,
    expr::PyExpr,
    query::{PyBin, PyPredicate},
    runtime::PyExecution,
};

fn path_string(path: &Bound<'_, PyAny>) -> PyResult<String> {
    let path: PathBuf = path.extract()?;
    Ok(path.to_string_lossy().into_owned())
}

fn configure(dataset: Dataset, chunk_size: Option<usize>, cache: &str) -> PyResult<Dataset> {
    let dataset = match chunk_size {
        Some(size) => dataset.chunked(size).map_err(to_py_err)?,
        None => dataset,
    };
    match cache.trim().to_ascii_lowercase().as_str() {
        "resident" => Ok(dataset.resident()),
        "streaming" => Ok(dataset.streaming()),
        _ => Err(PyValueError::new_err(
            "cache must be 'resident' or 'streaming'",
        )),
    }
}

#[pyclass(name = "ParquetSource", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A configurable Parquet dataset source.
///
/// Parameters
/// ----------
/// path : path-like
///     Parquet file, directory, or glob understood by the Arrow reader.
/// chunk_size : int, optional
///     Maximum number of events in each streamed batch.
/// cache : {'resident', 'streaming'}, default='resident'
///     Keep decoded batches in memory or reread them on each traversal.
/// nulls : {'error', 'nan'}, default='error'
///     Reject null scalar values or replace them with NaN.
/// validate : bool, default=True
///     Validate every matched file when constructing the source.
///
/// Notes
/// -----
/// Constructing a source records the read plan. Data are loaded when the
/// resulting :class:`Dataset` is traversed.
pub struct PyParquetSource {
    inner: Dataset,
}

#[pymethods]
impl PyParquetSource {
    /// Configure a Parquet source.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``cache`` or ``nulls`` is invalid.
    /// LadduError
    ///     If the source cannot be discovered or validated.
    #[new]
    #[pyo3(signature = (path, *, chunk_size=None, cache="resident", nulls="error", validate=true))]
    fn new(
        path: &Bound<'_, PyAny>,
        chunk_size: Option<usize>,
        cache: &str,
        nulls: &str,
        validate: bool,
    ) -> PyResult<Self> {
        let mut builder = ParquetSource::builder(path_string(path)?).validate_all_files(validate);
        builder = match nulls {
            "error" => builder.error_on_nulls(),
            "nan" => builder.nulls_as_nan(),
            _ => return Err(PyValueError::new_err("nulls must be 'error' or 'nan'")),
        };
        Ok(Self {
            inner: configure(
                Dataset::new(builder.build().map_err(to_py_err)?),
                chunk_size,
                cache,
            )?,
        })
    }
}

#[pyclass(name = "RootSource", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A configurable ROOT TTree dataset source.
///
/// Parameters
/// ----------
/// path : path-like
///     ROOT file, directory, or glob.
/// tree : str, optional
///     TTree name. If omitted, the reader discovers a compatible tree.
/// chunk_size : int, optional
///     Maximum number of events in each streamed batch.
/// cache : {'resident', 'streaming'}, default='resident'
///     Keep decoded batches in memory or reread them on each traversal.
/// validate : bool, default=True
///     Validate every matched file when constructing the source.
pub struct PyRootSource {
    inner: Dataset,
}

#[pymethods]
impl PyRootSource {
    /// Configure a ROOT source.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``cache`` is invalid.
    /// LadduError
    ///     If the source, tree, or schema is invalid.
    #[new]
    #[pyo3(signature = (path, *, tree=None, chunk_size=None, cache="resident", validate=true))]
    fn new(
        path: &Bound<'_, PyAny>,
        tree: Option<&str>,
        chunk_size: Option<usize>,
        cache: &str,
        validate: bool,
    ) -> PyResult<Self> {
        let mut builder = RootSource::builder(path_string(path)?).validate_all_files(validate);
        if let Some(tree) = tree {
            builder = builder.tree(tree);
        }
        Ok(Self {
            inner: configure(
                Dataset::new(builder.build().map_err(to_py_err)?),
                chunk_size,
                cache,
            )?,
        })
    }
}

#[pyclass(name = "ParquetSink", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A destination for writing a dataset as Parquet.
///
/// Parameters
/// ----------
/// path : path-like
///     Output file or dataset directory.
/// precision : {'f32', 'f64'}, default='f64'
///     Floating-point storage precision.
pub struct PyParquetSink {
    path: PathBuf,
    precision: String,
}

#[pymethods]
impl PyParquetSink {
    /// Configure a Parquet destination.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``precision`` is not ``'f32'`` or ``'f64'``.
    #[new]
    #[pyo3(signature = (path, *, precision="f64"))]
    fn new(path: PathBuf, precision: &str) -> PyResult<Self> {
        validate_precision(precision)?;
        Ok(Self {
            path,
            precision: precision.to_owned(),
        })
    }
}

#[pyclass(name = "RootSink", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A destination for writing a dataset as a ROOT TTree.
///
/// Parameters
/// ----------
/// path : path-like
///     Output ROOT file.
/// tree : str, default='tree'
///     Name of the output TTree.
/// precision : {'f32', 'f64'}, default='f64'
///     Floating-point storage precision.
pub struct PyRootSink {
    path: PathBuf,
    tree: String,
    precision: String,
}

#[pymethods]
impl PyRootSink {
    /// Configure a ROOT destination.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``precision`` is not ``'f32'`` or ``'f64'``.
    #[new]
    #[pyo3(signature = (path, *, tree="tree", precision="f64"))]
    fn new(path: PathBuf, tree: &str, precision: &str) -> PyResult<Self> {
        validate_precision(precision)?;
        Ok(Self {
            path,
            tree: tree.to_owned(),
            precision: precision.to_owned(),
        })
    }
}

fn validate_precision(precision: &str) -> PyResult<()> {
    match precision {
        "f32" | "f64" => Ok(()),
        _ => Err(PyValueError::new_err("precision must be 'f32' or 'f64'")),
    }
}

fn precision(value: &str) -> laddu_data::schema::Precision {
    match value {
        "f32" => laddu_data::schema::Precision::F32,
        _ => laddu_data::schema::Precision::F64,
    }
}

fn p4_array(values: &Bound<'_, PyAny>, name: &str) -> PyResult<(usize, Arc<[RealVec4]>)> {
    if let Ok(values) = values.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = values.shape();
        if shape[1] != 4 {
            return Err(PyValueError::new_err(format!(
                "p4 column {name:?} must have shape (events, 4), got ({}, {})",
                shape[0], shape[1]
            )));
        }
        let column = values
            .as_array()
            .rows()
            .into_iter()
            .map(|row| RealVec4::new(row[0], row[1], row[2], row[3]))
            .collect();
        return Ok((shape[0], column));
    }
    if let Ok(values) = values.extract::<PyReadonlyArray2<'_, f32>>() {
        let shape = values.shape();
        if shape[1] != 4 {
            return Err(PyValueError::new_err(format!(
                "p4 column {name:?} must have shape (events, 4), got ({}, {})",
                shape[0], shape[1]
            )));
        }
        let column = values
            .as_array()
            .rows()
            .into_iter()
            .map(|row| RealVec4::new(row[0] as f64, row[1] as f64, row[2] as f64, row[3] as f64))
            .collect();
        return Ok((shape[0], column));
    }
    Err(PyValueError::new_err(format!(
        "p4 column {name:?} must be a float32 or float64 NumPy array"
    )))
}

fn scalar_array(values: &Bound<'_, PyAny>, name: &str) -> PyResult<Arc<[f64]>> {
    if let Ok(values) = values.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(values.as_array().iter().copied().collect());
    }
    if let Ok(values) = values.extract::<PyReadonlyArray1<'_, f32>>() {
        return Ok(values
            .as_array()
            .iter()
            .map(|&value| value as f64)
            .collect());
    }
    Err(PyValueError::new_err(format!(
        "scalar column {name:?} must be a float32 or float64 NumPy array"
    )))
}

#[pyclass(name = "Dataset", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A reusable collection of weighted physics events.
///
/// Parameters
/// ----------
/// source : ParquetSource or RootSource
///     Lazy source that defines the dataset schema and read plan.
///
/// Examples
/// --------
/// Construct a dataset directly from NumPy arrays:
///
/// >>> import laddu as ld
/// >>> import numpy as np
/// >>> events = ld.Dataset.from_arrays(
/// ...     p4s={"beam": np.array([[5.0, 0.0, 0.0, 5.0]])},
/// ...     scalars={"run": np.array([1.0])},
/// ... )
/// >>> len(events)
/// 1
/// >>> events.p4_names()
/// ['beam']
pub struct PyDataset {
    pub(crate) inner: Dataset,
}

#[pymethods]
impl PyDataset {
    /// Create a dataset from a file source.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``source`` is not a :class:`ParquetSource` or
    ///     :class:`RootSource`.
    #[new]
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(source) = source.extract::<PyRef<'_, PyParquetSource>>() {
            return Ok(Self {
                inner: source.inner.clone(),
            });
        }
        if let Ok(source) = source.extract::<PyRef<'_, PyRootSource>>() {
            return Ok(Self {
                inner: source.inner.clone(),
            });
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Dataset source must be a ParquetSource or RootSource",
        ))
    }

    #[staticmethod]
    #[pyo3(signature = (*, p4s, scalars, weights=None))]
    /// Create an in-memory dataset from NumPy columns.
    ///
    /// Parameters
    /// ----------
    /// p4s : dict[str, numpy.ndarray]
    ///     Four-vector columns with shape ``(n_events, 4)`` and components in
    ///     ``(E, px, py, pz)`` order.
    /// scalars : dict[str, numpy.ndarray]
    ///     One-dimensional scalar columns.
    /// weights : numpy.ndarray, optional
    ///     One-dimensional event weights. Unit weights are used by default.
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     A resident, in-memory dataset.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a dtype or shape is invalid, or column lengths differ.
    /// LadduError
    ///     If the column names do not form a valid schema.
    fn from_arrays(
        p4s: &Bound<'_, PyDict>,
        scalars: &Bound<'_, PyDict>,
        weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut p4_names = Vec::with_capacity(p4s.len());
        let mut p4_columns = Vec::with_capacity(p4s.len());
        let mut expected_len = None;

        for (name, values) in p4s.iter() {
            let name = name.extract::<String>()?;
            let (len, column) = p4_array(&values, &name)?;
            if expected_len.is_some_and(|expected| expected != len) {
                return Err(PyValueError::new_err(
                    "all dataset columns must have the same number of events",
                ));
            }
            expected_len = Some(len);
            p4_names.push(name);
            p4_columns.push(column);
        }

        let mut scalar_names = Vec::with_capacity(scalars.len());
        let mut scalar_columns = Vec::with_capacity(scalars.len());
        for (name, values) in scalars.iter() {
            let name = name.extract::<String>()?;
            let values = scalar_array(&values, &name)?;
            let len = values.len();
            if expected_len.is_some_and(|expected| expected != len) {
                return Err(PyValueError::new_err(
                    "all dataset columns must have the same number of events",
                ));
            }
            expected_len = Some(len);
            scalar_names.push(name);
            scalar_columns.push(values);
        }

        let weights = weights
            .map(|weights| {
                let values = scalar_array(weights, "weights")?;
                if expected_len.is_some_and(|expected| expected != values.len()) {
                    return Err(PyValueError::new_err(
                        "weights must have the same number of events as the dataset columns",
                    ));
                }
                Ok(values)
            })
            .transpose()?;
        let schema =
            Arc::new(Schema::new(p4_names, scalar_names, weights.is_some()).map_err(to_py_err)?);
        let batch =
            EventBatch::new(schema, p4_columns, scalar_columns, weights).map_err(to_py_err)?;
        Ok(Self {
            inner: Dataset::from_batch(batch),
        })
    }

    fn __repr__(&self) -> String {
        let cache = match self.inner.cache_storage() {
            CacheStorage::Resident => "resident",
            CacheStorage::Streaming => "streaming",
        };
        format!(
            "Dataset(cache={cache:?}, chunk_size={:?})",
            self.inner.read_plan().chunk_size
        )
    }

    /// Return the names of all four-vector columns.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     Names in schema order.
    fn p4_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .schema()
            .map_err(to_py_err)?
            .p4s()
            .iter()
            .map(ToString::to_string)
            .collect())
    }

    /// Return the names of all scalar columns.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     Names in schema order.
    fn scalar_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .schema()
            .map_err(to_py_err)?
            .scalars()
            .iter()
            .map(ToString::to_string)
            .collect())
    }

    #[pyo3(signature = (expr, *, execution=None, real=false))]
    /// Evaluate an expression for every event.
    ///
    /// Parameters
    /// ----------
    /// expr : Expr
    ///     Symbolic expression to evaluate.
    /// execution : Execution, optional
    ///     Runtime configuration. Automatic local execution is used by default.
    /// real : bool, default=False
    ///     Return ``float64`` real components instead of complex values.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     One value per event.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the expression cannot be compiled or the dataset cannot be read.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        expr: &PyExpr,
        execution: Option<&PyExecution>,
        real: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        let dataset = self.inner.clone();
        let expr = expr.inner.clone();
        if real {
            let values = py
                .detach(move || dataset.evaluate_real(&expr, &execution.inner))
                .map_err(to_py_err)?;
            Ok(PyArray1::from_vec(py, values).into_any())
        } else {
            let values = py
                .detach(move || dataset.evaluate_expr(&expr, &execution.inner))
                .map_err(to_py_err)?;
            Ok(PyArray1::from_vec(py, values).into_any())
        }
    }

    /// Sum the event weights.
    ///
    /// Returns
    /// -------
    /// float
    ///     Sum over all explicit or implicit unit weights.
    fn sum_weights(&self, py: Python<'_>) -> PyResult<f64> {
        let dataset = self.inner.clone();
        py.detach(move || dataset.sum_weights()).map_err(to_py_err)
    }

    /// Materialize the event weights.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array with shape ``(n_events,)``.
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let dataset = self.inner.clone();
        let weights = py
            .detach(move || {
                dataset.try_fold_events(Vec::new(), |mut weights, event| {
                    weights.push(event.weight());
                    Ok(weights)
                })
            })
            .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, weights))
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        let dataset = self.inner.clone();
        py.detach(move || dataset.try_fold_events(0usize, |count, _| Ok(count + 1)))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (fraction, *, seed=0))]
    /// Select a reproducible random fraction of events.
    ///
    /// Parameters
    /// ----------
    /// fraction : float
    ///     Selection probability in the closed interval ``[0, 1]``.
    /// seed : int, default=0
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     A lazily transformed dataset.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If ``fraction`` is outside its valid range.
    fn subsample(&self, fraction: f64, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .subsample(fraction, seed)
                .map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (*, seed=0))]
    /// Create a Poisson-bootstrap replica of the dataset.
    ///
    /// Each event weight is multiplied by an independent Poisson(1) draw. The
    /// transformation is deterministic for a given seed.
    ///
    /// Parameters
    /// ----------
    /// seed : int, default=0
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     A lazily reweighted dataset.
    fn bootstrap(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().bootstrap(seed),
        }
    }

    #[pyo3(signature = (predicate, *, execution=None))]
    /// Retain events satisfying a symbolic predicate.
    ///
    /// Parameters
    /// ----------
    /// predicate : Predicate
    ///     Event-wise selection condition.
    /// execution : Execution, optional
    ///     Runtime configuration.
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     The selected events.
    fn select(&self, predicate: &PyPredicate, execution: Option<&PyExecution>) -> PyResult<Self> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        Ok(Self {
            inner: self
                .inner
                .select(&predicate.inner, &execution.inner)
                .map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (expr, bins, *, execution=None))]
    /// Partition events into bins of an evaluated expression.
    ///
    /// Parameters
    /// ----------
    /// expr : Expr
    ///     Real-valued binning expression.
    /// bins : Bin
    ///     Uniform or explicitly edged bin specification.
    /// execution : Execution, optional
    ///     Runtime configuration.
    ///
    /// Returns
    /// -------
    /// list of BinnedDataset
    ///     Bin metadata and the events assigned to each bin.
    fn bin_by(
        &self,
        expr: &PyExpr,
        bins: &PyBin,
        execution: Option<&PyExecution>,
    ) -> PyResult<Vec<PyBinDataset>> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        Ok(self
            .inner
            .bin_by(&expr.inner, bins.inner.clone(), &execution.inner)
            .map_err(to_py_err)?
            .into_iter()
            .map(|bin| PyBinDataset {
                index: bin.index(),
                low: bin.lower(),
                high: bin.upper(),
                dataset: PyDataset {
                    inner: bin.into_dataset(),
                },
            })
            .collect())
    }

    /// Write all events to a configured file destination.
    ///
    /// Parameters
    /// ----------
    /// sink : ParquetSink or RootSink
    ///     Output format and path.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``sink`` has an unsupported type.
    /// LadduError
    ///     If reading or writing fails.
    fn write_to(&self, py: Python<'_>, sink: &Bound<'_, PyAny>) -> PyResult<()> {
        let dataset = self.inner.clone();
        if let Ok(sink) = sink.extract::<PyRef<'_, PyParquetSink>>() {
            let mut sink = ParquetSink::builder(sink.path.clone())
                .precision(precision(&sink.precision))
                .build();
            return py
                .detach(move || dataset.write_to(&mut sink))
                .map_err(to_py_err);
        }
        if let Ok(sink) = sink.extract::<PyRef<'_, PyRootSink>>() {
            let mut sink = RootSink::builder(sink.path.clone())
                .tree(sink.tree.as_str())
                .precision(precision(&sink.precision))
                .build();
            return py
                .detach(move || dataset.write_to(&mut sink))
                .map_err(to_py_err);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Dataset sink must be a ParquetSink or RootSink",
        ))
    }
}

#[pyclass(name = "BinnedDataset", module = "laddu", frozen, skip_from_py_object)]
/// A dataset and the numeric interval that selected it.
///
/// Attributes
/// ----------
/// index : int
///     Zero-based bin index.
/// low : float
///     Inclusive lower edge.
/// high : float
///     Exclusive upper edge, except where the binning policy includes the final
///     boundary.
/// dataset : Dataset
///     Events assigned to this interval.
pub struct PyBinDataset {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    low: f64,
    #[pyo3(get)]
    high: f64,
    #[pyo3(get)]
    dataset: PyDataset,
}

#[pyfunction]
#[pyo3(signature = (path, *, chunk_size=None, cache="resident", nulls="error", validate=true))]
/// Read a Parquet dataset.
///
/// This is shorthand for ``Dataset(ParquetSource(...))``.
///
/// Parameters
/// ----------
/// path : path-like
///     Parquet file, directory, or glob.
/// chunk_size : int, optional
///     Maximum streamed batch size.
/// cache : {'resident', 'streaming'}, default='resident'
///     Dataset cache policy.
/// nulls : {'error', 'nan'}, default='error'
///     Null-value policy.
/// validate : bool, default=True
///     Validate every matched file.
///
/// Returns
/// -------
/// Dataset
///     Configured dataset.
pub fn read_parquet(
    path: &Bound<'_, PyAny>,
    chunk_size: Option<usize>,
    cache: &str,
    nulls: &str,
    validate: bool,
) -> PyResult<PyDataset> {
    let mut builder = ParquetSource::builder(path_string(path)?).validate_all_files(validate);
    builder = match nulls {
        "error" => builder.error_on_nulls(),
        "nan" => builder.nulls_as_nan(),
        _ => return Err(PyValueError::new_err("nulls must be 'error' or 'nan'")),
    };
    let dataset = Dataset::new(builder.build().map_err(to_py_err)?);
    Ok(PyDataset {
        inner: configure(dataset, chunk_size, cache)?,
    })
}

#[pyfunction]
#[pyo3(signature = (path, *, tree=None, chunk_size=None, cache="resident", validate=true))]
/// Read a ROOT TTree dataset.
///
/// This is shorthand for ``Dataset(RootSource(...))``.
///
/// Parameters
/// ----------
/// path : path-like
///     ROOT file, directory, or glob.
/// tree : str, optional
///     TTree name, or automatic discovery when omitted.
/// chunk_size : int, optional
///     Maximum streamed batch size.
/// cache : {'resident', 'streaming'}, default='resident'
///     Dataset cache policy.
/// validate : bool, default=True
///     Validate every matched file.
///
/// Returns
/// -------
/// Dataset
///     Configured dataset.
pub fn read_root(
    path: &Bound<'_, PyAny>,
    tree: Option<&str>,
    chunk_size: Option<usize>,
    cache: &str,
    validate: bool,
) -> PyResult<PyDataset> {
    let mut builder = RootSource::builder(path_string(path)?).validate_all_files(validate);
    if let Some(tree) = tree {
        builder = builder.tree(tree);
    }
    let dataset = Dataset::new(builder.build().map_err(to_py_err)?);
    Ok(PyDataset {
        inner: configure(dataset, chunk_size, cache)?,
    })
}

#[pymodule]
/// Dataset sources, sinks, transformations, and file-reading helpers.
pub mod data {
    #[pymodule_export]
    use super::{
        PyBinDataset as BinnedDataset, PyDataset as Dataset, PyParquetSink as ParquetSink,
        PyParquetSource as ParquetSource, PyRootSink as RootSink, PyRootSource as RootSource,
        read_parquet, read_root,
    };
}
