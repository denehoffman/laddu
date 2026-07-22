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
pub struct PyParquetSource {
    inner: Dataset,
}

#[pymethods]
impl PyParquetSource {
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
pub struct PyRootSource {
    inner: Dataset,
}

#[pymethods]
impl PyRootSource {
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
pub struct PyParquetSink {
    path: PathBuf,
    precision: String,
}

#[pymethods]
impl PyParquetSink {
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
pub struct PyRootSink {
    path: PathBuf,
    tree: String,
    precision: String,
}

#[pymethods]
impl PyRootSink {
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
pub struct PyDataset {
    pub(crate) inner: Dataset,
}

#[pymethods]
impl PyDataset {
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

    fn sum_weights(&self, py: Python<'_>) -> PyResult<f64> {
        let dataset = self.inner.clone();
        py.detach(move || dataset.sum_weights()).map_err(to_py_err)
    }

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
    fn bootstrap(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().bootstrap(seed),
        }
    }

    #[pyo3(signature = (predicate, *, execution=None))]
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
pub mod data {
    #[pymodule_export]
    use super::{
        PyBinDataset as BinnedDataset, PyDataset as Dataset, PyParquetSink as ParquetSink,
        PyParquetSource as ParquetSource, PyRootSink as RootSink, PyRootSource as RootSource,
        read_parquet, read_root,
    };
}
