//! Format-independent Python event I/O. Format callbacks never communicate over MPI.

use std::{path::PathBuf, sync::Arc};

use laddu_data::{
    LadduDataError, LadduDataResult,
    data::{Dataset, EventBatch},
    io::{
        EventBatchIter, EventSink, EventSource, Partitioning, ReadPlan, SourceCapabilities,
        WritePlan,
    },
    schema::Schema,
};
#[cfg(feature = "mpi")]
use laddu_physics::vectors::RealVec4;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, ndarray::Array2};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    inspect::{PyStaticConstant, PyStaticExpr},
    prelude::*,
    type_hint_identifier, type_hint_subscript, type_hint_union,
    types::{PyDict, PyIterator, PyMappingProxy, PyTuple},
};

use super::{data::PyDataset, error::to_py_err, runtime::PyExecution};

// String signature annotations are forward references and do not generate imports.
// These transparent inputs retain PyAny's runtime flexibility while supplying
// structured metadata that Maturin can resolve into imported Python types.
macro_rules! hinted_input {
    ($visibility:vis $name:ident, $hint:expr) => {
        #[doc(hidden)]
        $visibility struct $name(Py<PyAny>);
        impl<'a, 'py> FromPyObject<'a, 'py> for $name {
            type Error = PyErr;
            const INPUT_TYPE: PyStaticExpr = $hint;
            fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                Ok(Self(value.to_owned().unbind()))
            }
        }
    };
}
pub(super) use hinted_input;

const ELLIPSIS_HINT: PyStaticExpr = PyStaticExpr::Constant {
    value: PyStaticConstant::Ellipsis,
};
const NONE_HINT: PyStaticExpr = PyStaticExpr::Constant {
    value: PyStaticConstant::None,
};
hinted_input!(
    DescribeInput,
    type_hint_subscript!(
        type_hint_identifier!("collections.abc", "Callable"),
        ELLIPSIS_HINT,
        type_hint_identifier!("laddu.io", "SourceInfo")
    )
);
hinted_input!(
    ReaderInput,
    type_hint_subscript!(
        type_hint_identifier!("collections.abc", "Callable"),
        ELLIPSIS_HINT,
        type_hint_subscript!(
            type_hint_identifier!("collections.abc", "Iterable"),
            type_hint_identifier!("laddu.io", "EventBatch")
        )
    )
);
hinted_input!(
    WriterInput,
    type_hint_subscript!(
        type_hint_identifier!("collections.abc", "Callable"),
        ELLIPSIS_HINT,
        NONE_HINT
    )
);
hinted_input!(
    PartitioningInput,
    type_hint_subscript!(
        type_hint_identifier!("collections.abc", "Iterable"),
        type_hint_subscript!(
            type_hint_identifier!("typing", "Literal"),
            PyStaticExpr::Constant {
                value: PyStaticConstant::Str("contiguous")
            },
            PyStaticExpr::Constant {
                value: PyStaticConstant::Str("rows")
            },
            PyStaticExpr::Constant {
                value: PyStaticConstant::Str("file_groups")
            }
        )
    )
);
hinted_input!(pub MemoryInput, type_hint_union!(
    type_hint_identifier!("laddu", "MemoryBudget"),
    type_hint_union!(type_hint_identifier!("builtins", "int"), type_hint_identifier!("builtins", "str"))
));
const FLOAT_ARRAY_HINT: PyStaticExpr = type_hint_subscript!(
    type_hint_identifier!("numpy.typing", "NDArray"),
    type_hint_union!(
        type_hint_identifier!("numpy", "float32"),
        type_hint_identifier!("numpy", "float64")
    )
);
hinted_input!(
    ColumnsInput,
    type_hint_subscript!(
        type_hint_identifier!("builtins", "dict"),
        type_hint_identifier!("builtins", "str"),
        FLOAT_ARRAY_HINT
    )
);
hinted_input!(
    WeightsInput,
    type_hint_union!(
        type_hint_subscript!(
            type_hint_identifier!("collections.abc", "Sequence"),
            type_hint_identifier!("builtins", "float")
        ),
        FLOAT_ARRAY_HINT
    )
);

#[derive(Debug)]
struct CallbackError {
    context: String,
    error: PyErr,
}

impl std::fmt::Display for CallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.context, self.error)
    }
}
impl std::error::Error for CallbackError {}

pub(super) fn callback_error(context: impl Into<String>, error: PyErr) -> LadduDataError {
    LadduDataError::External(Arc::new(CallbackError {
        context: context.into(),
        error,
    }))
}

pub(super) fn data_error(py: Python<'_>, error: LadduDataError) -> PyErr {
    let result = to_py_err(&error);
    if let LadduDataError::External(cause) = &error
        && let Some(cause) = cause.downcast_ref::<CallbackError>()
    {
        result.set_cause(py, Some(cause.error.clone_ref(py)));
    }
    result
}

fn names(schema: &Schema) -> (Vec<String>, Vec<String>) {
    (
        schema.p4s().iter().map(ToString::to_string).collect(),
        schema.scalars().iter().map(ToString::to_string).collect(),
    )
}

#[pyclass(name = "Schema", module = "laddu.io", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Ordered logical event columns. Four-vectors use (E, px, py, pz).
pub struct PySchema {
    pub(super) inner: Arc<Schema>,
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (*, p4s=Vec::new(), scalars=Vec::new(), weights=false, columns: "dict[str, object] | None" = None))]
    fn new(
        p4s: Vec<String>,
        scalars: Vec<String>,
        weights: bool,
        columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let declarations = super::data::column_declarations(columns)?;
        let schema = Schema::new(p4s, scalars, weights)
            .and_then(|schema| schema.with_columns(declarations))
            .map_err(to_py_err)?;
        schema
            .validate_column_names(&Default::default())
            .map_err(to_py_err)?;
        Ok(Self {
            inner: Arc::new(schema),
        })
    }
    #[getter]
    fn p4s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, names(&self.inner).0)
    }
    #[getter]
    fn scalars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, names(&self.inner).1)
    }
    #[getter]
    /// Ordered exact column declarations using canonical dtype names.
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let columns = PyDict::new(py);
        for (name, dtype) in self.inner.columns() {
            columns.set_item(name.as_ref(), dtype.name())?;
        }
        Ok(columns)
    }
    #[getter]
    fn weights(&self) -> bool {
        self.inner.has_weight()
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
    fn __repr__(&self) -> String {
        let (p4s, scalars) = names(&self.inner);
        format!(
            "Schema(p4s={p4s:?}, scalars={scalars:?}, weights={}, columns={:?})",
            self.weights(),
            self.inner.columns()
        )
    }
}

#[pyclass(name = "SourceInfo", module = "laddu.io", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Source schema and an optional exact global event count.
pub struct PySourceInfo {
    #[pyo3(get)]
    schema: PySchema,
    #[pyo3(get)]
    length: Option<u64>,
}

#[pymethods]
impl PySourceInfo {
    #[new]
    #[pyo3(signature = (schema, *, length=None))]
    fn new(schema: &PySchema, length: Option<u64>) -> Self {
        Self {
            schema: schema.clone(),
            length,
        }
    }
}

#[pyclass(name = "EventBatch", module = "laddu.io", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Owned immutable float64 physics fields and exact integer row-data columns.
///
/// ``columns`` accepts one-dimensional arrays of signed or unsigned 8-, 16-,
/// 32-, or 64-bit integers. Inputs are copied; exports are independent read-only
/// NumPy arrays. Integer payload remains outside expression evaluation.
/// Input arrays may be float32 or float64. Missing weights mean unit weights.
pub struct PyEventBatch {
    pub(super) inner: EventBatch,
}

#[pymethods]
impl PyEventBatch {
    #[new]
    #[pyo3(signature = (*, p4s=None, scalars=None, weights=None, length=None, row_ids=None, columns: "dict[str, numpy.typing.ArrayLike] | None" = None))]
    fn new(
        py: Python<'_>,
        p4s: Option<ColumnsInput>,
        scalars: Option<ColumnsInput>,
        weights: Option<WeightsInput>,
        length: Option<usize>,
        row_ids: Option<Vec<u64>>,
        columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let empty_p4s = PyDict::new(py);
        let empty_scalars = PyDict::new(py);
        let p4s = p4s
            .as_ref()
            .map(|value| value.0.bind(py).cast::<PyDict>())
            .transpose()?;
        let scalars = scalars
            .as_ref()
            .map(|value| value.0.bind(py).cast::<PyDict>())
            .transpose()?;
        let batch = super::data::event_batch_from_arrays(
            p4s.unwrap_or(&empty_p4s),
            scalars.unwrap_or(&empty_scalars),
            weights.as_ref().map(|value| value.0.bind(py)),
            columns,
        )?;
        let inner = if let Some(length) = length {
            EventBatch::new_with_columns_and_len(
                Arc::clone(batch.schema()),
                (0..batch.schema().n_p4s())
                    .map(|i| Arc::from(batch.vec4_column(i)))
                    .collect(),
                (0..batch.schema().n_scalars())
                    .map(|i| Arc::from(batch.scalar_column(i)))
                    .collect(),
                (0..batch.schema().n_columns())
                    .map(|i| batch.column(i).clone())
                    .collect(),
                batch.weights_column().map(Arc::from),
                length,
            )
            .map_err(to_py_err)?
        } else {
            batch
        };
        let inner = if let Some(ids) = row_ids {
            inner.with_row_ids(ids.into()).map_err(to_py_err)?
        } else {
            inner
        };
        Ok(Self { inner })
    }
    #[getter]
    fn schema(&self) -> PySchema {
        PySchema {
            inner: Arc::clone(self.inner.schema()),
        }
    }
    #[getter]
    fn length(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn row_ids(&self) -> Option<Vec<u64>> {
        self.inner.row_ids().map(<[u64]>::to_vec)
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn p4s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyMappingProxy>> {
        let columns = PyDict::new(py);
        for (i, name) in self.inner.schema().p4s().iter().enumerate() {
            let values: Vec<f64> = self
                .inner
                .vec4_column(i)
                .iter()
                .flat_map(|p4| p4.components())
                .collect();
            let array = Array2::from_shape_vec((self.inner.len(), 4), values)
                .map_err(to_py_err)?
                .into_pyarray(py);
            array.readwrite().make_nonwriteable();
            columns.set_item(name.as_ref(), array)?;
        }
        Ok(PyMappingProxy::new(py, columns.as_mapping()))
    }
    #[getter]
    fn scalars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyMappingProxy>> {
        let columns = PyDict::new(py);
        for (i, name) in self.inner.schema().scalars().iter().enumerate() {
            let array = PyArray1::from_vec(py, self.inner.scalar_column(i).to_vec());
            array.readwrite().make_nonwriteable();
            columns.set_item(name.as_ref(), array)?;
        }
        Ok(PyMappingProxy::new(py, columns.as_mapping()))
    }
    #[getter]
    /// Exact row-data arrays in schema order, copied and initially read-only.
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyMappingProxy>> {
        let columns = PyDict::new(py);
        for (index, (name, _)) in self.inner.schema().columns().iter().enumerate() {
            columns.set_item(
                name.as_ref(),
                super::data::column_array(py, self.inner.column(index)),
            )?;
        }
        Ok(PyMappingProxy::new(py, columns.as_mapping()))
    }
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.weights_column().map(|values| {
            let array = PyArray1::from_vec(py, values.to_vec());
            array.readwrite().make_nonwriteable();
            array
        })
    }
}

fn partition_name(partition: Partitioning) -> &'static str {
    match partition {
        Partitioning::Contiguous => "contiguous",
        Partitioning::Rows => "rows",
        Partitioning::FileGroups => "file_groups",
    }
}

fn parse_partition(value: &str) -> PyResult<Partitioning> {
    match value {
        "contiguous" => Ok(Partitioning::Contiguous),
        "rows" => Ok(Partitioning::Rows),
        "file_groups" => Ok(Partitioning::FileGroups),
        _ => Err(PyValueError::new_err(
            "partitioning must be 'contiguous', 'rows', or 'file_groups'",
        )),
    }
}

fn plan_partition(plan: ReadPlan) -> Partitioning {
    #[cfg(feature = "mpi")]
    {
        plan.distribution.partitioning()
    }
    #[cfg(not(feature = "mpi"))]
    {
        let _ = plan;
        Partitioning::Contiguous
    }
}

#[pyclass(name = "ReadPlan", module = "laddu.io", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Effective batch size and assignment. A fallback reader receives a serial plan.
pub struct PyReadPlan {
    #[pyo3(get)]
    chunk_size: Option<usize>,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    nranks: usize,
    #[pyo3(get)]
    partitioning: String,
}
impl From<ReadPlan> for PyReadPlan {
    fn from(plan: ReadPlan) -> Self {
        Self {
            chunk_size: plan.chunk_size,
            rank: plan.rank(),
            nranks: plan.nranks(),
            partitioning: partition_name(plan_partition(plan)).into(),
        }
    }
}

#[pyclass(name = "WritePlan", module = "laddu.io", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Writer location and input assignment. Single-file writers run on rank zero.
pub struct PyWritePlan {
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    nranks: usize,
    #[pyo3(get)]
    partitioning: String,
    #[pyo3(get)]
    output: String,
    #[pyo3(get)]
    chunk_size: Option<usize>,
}

struct Spec {
    name: String,
    describe: Option<Py<PyAny>>,
    reader: Option<Py<PyAny>>,
    writer: Option<Py<PyAny>>,
    native: Vec<Partitioning>,
    extension: String,
}

#[pyclass(name = "FormatSpec", module = "laddu.io", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Reusable downstream format callbacks, with no registration or subclassing.
///
/// ``describe(source, *, options)`` returns SourceInfo.
/// ``read_batches(source, *, options, plan)`` returns a fresh iterator of batches.
/// ``write_batches(target, batches, *, options, schema, plan)`` consumes the stream.
/// Native partitioning is an explicit promise; undeclared strategies use fallback.
pub struct PyFormatSpec {
    spec: Arc<Spec>,
}

#[pymethods]
impl PyFormatSpec {
    #[new]
    #[pyo3(signature = (*, name, describe=None, read_batches=None, write_batches=None, extension="dat", native_partitioning=None))]
    fn new(
        py: Python<'_>,
        name: String,
        describe: Option<DescribeInput>,
        read_batches: Option<ReaderInput>,
        write_batches: Option<WriterInput>,
        extension: &str,
        native_partitioning: Option<PartitioningInput>,
    ) -> PyResult<Self> {
        let describe = describe.map(|value| value.0);
        let read_batches = read_batches.map(|value| value.0);
        let write_batches = write_batches.map(|value| value.0);
        if name.is_empty() || (read_batches.is_none() && write_batches.is_none()) {
            return Err(PyValueError::new_err(
                "a format needs a name and a reader or writer",
            ));
        }
        if read_batches.is_some() != describe.is_some() {
            return Err(PyValueError::new_err(
                "describe and read_batches must be supplied together",
            ));
        }
        for callback in [&describe, &read_batches, &write_batches]
            .into_iter()
            .flatten()
        {
            if !callback.bind(py).is_callable() {
                return Err(PyTypeError::new_err("format callbacks must be callable"));
            }
        }
        let extension = extension.trim_start_matches('.');
        if extension.is_empty() || extension.contains(['/', '\\']) {
            return Err(PyValueError::new_err(
                "extension must be a nonempty filename extension",
            ));
        }
        let mut native = native_partitioning
            .map(|values| {
                values
                    .0
                    .bind(py)
                    .try_iter()?
                    .map(|value| parse_partition(&value?.extract::<String>()?))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        native.sort_by_key(|&partition| partition_name(partition));
        native.dedup();
        Ok(Self {
            spec: Arc::new(Spec {
                name,
                describe,
                reader: read_batches,
                writer: write_batches,
                native,
                extension: extension.into(),
            }),
        })
    }
    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }
    #[getter]
    fn native_partitioning(&self) -> Vec<&'static str> {
        self.spec
            .native
            .iter()
            .copied()
            .map(partition_name)
            .collect()
    }
    #[pyo3(signature = (source: "object", *, options: "object | None" = None, memory=None, cache="fastest", execution=None))]
    fn source(
        &self,
        py: Python<'_>,
        source: Py<PyAny>,
        options: Option<Py<PyAny>>,
        memory: Option<MemoryInput>,
        cache: &str,
        execution: Option<&PyExecution>,
    ) -> PyResult<PyFormatSource> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        let options = Arc::new(options.unwrap_or_else(|| py.None()));
        let info = (|| -> PyResult<PySourceInfo> {
            let describe = self
                .spec
                .describe
                .as_ref()
                .ok_or_else(|| PyTypeError::new_err("format has no reader"))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("options", options.bind(py))?;
            let value = describe.bind(py).call((source.bind(py),), Some(&kwargs))?;
            Ok(value.extract::<PyRef<'_, PySourceInfo>>()?.clone())
        })();
        let info = coordinated(
            py,
            &execution,
            info.map_err(|error| {
                data_error(
                    py,
                    callback_error(format!("{} describe", self.spec.name), error),
                )
            }),
        )?;
        agree(
            &execution,
            &format!(
                "{:?}|{:?}|{:?}|{:?}|{}|{}",
                names(&info.schema.inner),
                info.schema.inner.columns(),
                info.schema.weights(),
                info.length,
                self.spec.name,
                self.spec
                    .native
                    .iter()
                    .map(|&p| partition_name(p))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        )?;
        let reader = FormatSource {
            spec: Arc::clone(&self.spec),
            source: Arc::new(source),
            options,
            info,
        };
        let dataset = coordinated(
            py,
            &execution,
            super::data::configure(
                Dataset::new(reader),
                memory.as_ref().map(|value| value.0.bind(py)),
                cache,
            ),
        )?;
        Ok(PyFormatSource { dataset })
    }
    #[pyo3(signature = (source: "object", *, options: "object | None" = None, memory=None, cache="fastest", execution=None))]
    fn read(
        &self,
        py: Python<'_>,
        source: Py<PyAny>,
        options: Option<Py<PyAny>>,
        memory: Option<MemoryInput>,
        cache: &str,
        execution: Option<&PyExecution>,
    ) -> PyResult<PyDataset> {
        Ok(PyDataset {
            inner: self
                .source(py, source, options, memory, cache, execution)?
                .dataset,
        })
    }
    #[pyo3(signature = (target, *, options: "object | None" = None, output="auto", chunk_size=None, execution=None))]
    fn sink(
        &self,
        py: Python<'_>,
        target: PathBuf,
        options: Option<Py<PyAny>>,
        output: &str,
        chunk_size: Option<usize>,
        execution: Option<&PyExecution>,
    ) -> PyResult<PyFormatSink> {
        if self.spec.writer.is_none() {
            return Err(PyTypeError::new_err("format has no writer"));
        }
        if !matches!(output, "auto" | "single" | "sharded") {
            return Err(PyValueError::new_err(
                "output must be 'auto', 'single', or 'sharded'",
            ));
        }
        if chunk_size == Some(0) {
            return Err(PyValueError::new_err("chunk_size must be positive"));
        }
        Ok(PyFormatSink {
            spec: Arc::clone(&self.spec),
            target,
            options: Arc::new(options.unwrap_or_else(|| py.None())),
            output: output.into(),
            chunk_size,
            execution: execution
                .cloned()
                .map(Ok)
                .unwrap_or_else(PyExecution::default_inner)?,
        })
    }
    #[pyo3(signature = (dataset, target, *, options: "object | None" = None, output="auto", chunk_size=None, execution=None))]
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        py: Python<'_>,
        dataset: &PyDataset,
        target: PathBuf,
        options: Option<Py<PyAny>>,
        output: &str,
        chunk_size: Option<usize>,
        execution: Option<&PyExecution>,
    ) -> PyResult<PyWriteResult> {
        self.sink(py, target, options, output, chunk_size, execution)?
            .write(py, dataset)
    }

    #[pyo3(signature = (source: "object", *, options: "object | None" = None, nranks=3, chunk_size=1024))]
    /// Check declared native partitioning against a serial traversal of a small fixture.
    /// This local testing helper materializes the fixture; it does not require MPI.
    fn check_partitioning(
        &self,
        py: Python<'_>,
        source: Py<PyAny>,
        options: Option<Py<PyAny>>,
        nranks: usize,
        chunk_size: usize,
    ) -> PyResult<()> {
        if nranks == 0 || chunk_size == 0 {
            return Err(PyValueError::new_err(
                "nranks and chunk_size must be positive",
            ));
        }
        let options = Arc::new(options.unwrap_or_else(|| py.None()));
        let kwargs = PyDict::new(py);
        kwargs.set_item("options", options.bind(py))?;
        let describe = self
            .spec
            .describe
            .as_ref()
            .ok_or_else(|| PyTypeError::new_err("format has no reader"))?;
        let value = describe.bind(py).call((source.bind(py),), Some(&kwargs))?;
        let info = value.extract::<PyRef<'_, PySourceInfo>>()?.clone();
        let reader = FormatSource {
            spec: Arc::clone(&self.spec),
            source: Arc::new(source),
            options,
            info,
        };
        let serial: Vec<_> = reader
            .open(ReadPlan {
                chunk_size: Some(chunk_size),
                ..ReadPlan::serial()
            })
            .map_err(|e| data_error(py, e))?
            .map(|batch| {
                batch.map(|batch| {
                    (0..batch.len())
                        .map(|i| row_key(&batch, i))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<LadduDataResult<Vec<_>>>()
            .map_err(|e| data_error(py, e))?
            .into_iter()
            .flatten()
            .collect();
        if reader.info.length.is_some_and(|n| n != serial.len() as u64) {
            return Err(PyValueError::new_err(
                "SourceInfo.length disagrees with serial traversal",
            ));
        }
        for &partition in &self.spec.native {
            let mut seen = vec![false; serial.len()];
            for rank in 0..nranks {
                let plan = PyReadPlan {
                    chunk_size: Some(chunk_size),
                    rank,
                    nranks,
                    partitioning: partition_name(partition).into(),
                };
                let expected: Vec<usize> = (0..serial.len())
                    .filter(|&i| match partition {
                        Partitioning::Rows => i % nranks == rank,
                        Partitioning::Contiguous => {
                            i >= serial.len() * rank / nranks
                                && i < serial.len() * (rank + 1) / nranks
                        }
                        Partitioning::FileGroups => true,
                    })
                    .collect();
                let mut position = 0;
                for batch in reader.open_python(plan).map_err(|e| data_error(py, e))? {
                    let batch = batch.map_err(|e| data_error(py, e))?;
                    for row in 0..batch.len() {
                        let id = if partition == Partitioning::FileGroups {
                            batch
                                .row_ids()
                                .and_then(|ids| usize::try_from(ids[row]).ok())
                        } else {
                            let expected = expected.get(position).copied();
                            if let (Some(ids), Some(id)) = (batch.row_ids(), expected)
                                && ids[row] != id as u64
                            {
                                return Err(PyValueError::new_err(
                                    "native row_ids do not match the declared assignment",
                                ));
                            }
                            expected
                        }
                        .ok_or_else(|| {
                            PyValueError::new_err(
                                "native partition has extra rows or missing row_ids",
                            )
                        })?;
                        if id >= serial.len() || seen[id] || row_key(&batch, row) != serial[id] {
                            return Err(PyValueError::new_err(format!(
                                "{} partition rank {rank} disagrees with serial event {id}",
                                partition_name(partition)
                            )));
                        }
                        seen[id] = true;
                        position += 1;
                    }
                }
                if partition != Partitioning::FileGroups && position != expected.len() {
                    return Err(PyValueError::new_err("native partition omitted events"));
                }
            }
            if seen.iter().any(|seen| !seen) {
                return Err(PyValueError::new_err(
                    "native partitions do not cover the source",
                ));
            }
        }
        Ok(())
    }
}

fn row_key(batch: &EventBatch, row: usize) -> Vec<u64> {
    let mut key = Vec::new();
    for col in 0..batch.schema().n_p4s() {
        key.extend(batch.p4_at(col, row).components().map(f64::to_bits));
    }
    for col in 0..batch.schema().n_scalars() {
        key.push(batch.scalar_at(col, row).to_bits());
    }
    for col in 0..batch.schema().n_columns() {
        key.push(match batch.column(col).at(row) {
            laddu_data::columns::ColumnValue::I8(value) => value as u64,
            laddu_data::columns::ColumnValue::U8(value) => value as u64,
            laddu_data::columns::ColumnValue::I16(value) => value as u64,
            laddu_data::columns::ColumnValue::U16(value) => value as u64,
            laddu_data::columns::ColumnValue::I32(value) => value as u64,
            laddu_data::columns::ColumnValue::U32(value) => value as u64,
            laddu_data::columns::ColumnValue::I64(value) => value as u64,
            laddu_data::columns::ColumnValue::U64(value) => value,
        });
    }
    key.push(batch.weights_at(row).to_bits());
    key
}

#[pyclass(
    name = "FormatSource",
    module = "laddu.io",
    frozen,
    skip_from_py_object
)]
/// Bound source accepted by Dataset(source).
pub struct PyFormatSource {
    pub(super) dataset: Dataset,
}

#[derive(Clone)]
struct FormatSource {
    spec: Arc<Spec>,
    source: Arc<Py<PyAny>>,
    options: Arc<Py<PyAny>>,
    info: PySourceInfo,
}

impl FormatSource {
    fn open(&self, plan: ReadPlan) -> LadduDataResult<CallbackBatches> {
        self.open_python(plan.into())
    }
    fn open_python(&self, plan: PyReadPlan) -> LadduDataResult<CallbackBatches> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("options", self.options.bind(py))?;
            kwargs.set_item("plan", plan.clone())?;
            let iterable = self
                .spec
                .reader
                .as_ref()
                .expect("validated reader")
                .bind(py)
                .call((self.source.bind(py),), Some(&kwargs))?;
            let iterator = iterable.try_iter()?.unbind();
            Ok(CallbackBatches {
                iterator: Some(iterator),
                schema: Arc::clone(&self.info.schema.inner),
                context: format!("{} read_batches", self.spec.name),
                chunk_size: plan.chunk_size,
                seen: 0,
                expected: self.info.length.and_then(|n| {
                    match (plan.nranks, plan.partitioning.as_str()) {
                        (1, _) => Some(n),
                        (_, "rows") => Some(
                            n / plan.nranks as u64
                                + u64::from((plan.rank as u64) < n % plan.nranks as u64),
                        ),
                        (_, "contiguous") => Some(
                            (u128::from(n) * (plan.rank + 1) as u128 / plan.nranks as u128
                                - u128::from(n) * plan.rank as u128 / plan.nranks as u128)
                                as u64,
                        ),
                        _ => None,
                    }
                }),
            })
        })
        .map_err(|e| callback_error(format!("{} open reader", self.spec.name), e))
    }
}

impl EventSource for FormatSource {
    fn schema(&self) -> LadduDataResult<Arc<Schema>> {
        Ok(Arc::clone(&self.info.schema.inner))
    }
    fn num_events(&self) -> LadduDataResult<Option<u64>> {
        Ok(self.info.length)
    }
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            exact_len: self.info.length.is_some(),
            deterministic_partitioning: true,
            streaming: true,
            ..SourceCapabilities::default()
        }
    }
    fn batches(&self, plan: ReadPlan) -> LadduDataResult<EventBatchIter> {
        let partition = plan_partition(plan);
        if !plan.is_distributed() {
            return Ok(Box::new(self.open(plan)?));
        }
        if self.spec.native.contains(&partition) {
            let count = if partition == Partitioning::Contiguous {
                Some(match self.info.length {
                    Some(n) => n,
                    None => self
                        .open(ReadPlan {
                            chunk_size: plan.chunk_size,
                            ..ReadPlan::serial()
                        })?
                        .try_fold(0_u64, |n, batch| Ok(n + batch?.len() as u64))?,
                })
            } else {
                None
            };
            return Ok(Box::new(NativeBatches {
                input: self.open(plan)?,
                position: 0,
                plan,
                count,
            }));
        }
        if partition == Partitioning::FileGroups {
            return Err(LadduDataError::Unsupported(
                "file_groups requires declared native partitioning",
            ));
        }
        let serial = ReadPlan {
            chunk_size: plan.chunk_size,
            ..ReadPlan::serial()
        };
        let count = if partition == Partitioning::Contiguous {
            Some(match self.info.length {
                Some(n) => n,
                None => self
                    .open(serial)?
                    .try_fold(0_u64, |n, batch| Ok(n + batch?.len() as u64))?,
            })
        } else {
            None
        };
        Ok(Box::new(PartitionedBatches {
            input: self.open(serial)?,
            position: 0,
            plan,
            count,
        }))
    }
}

struct CallbackBatches {
    iterator: Option<Py<PyIterator>>,
    schema: Arc<Schema>,
    context: String,
    chunk_size: Option<usize>,
    seen: u64,
    expected: Option<u64>,
}
impl CallbackBatches {
    fn close(&mut self) {
        if let Some(iterator) = self.iterator.take() {
            Python::attach(|py| {
                if let Ok(close) = iterator.bind(py).getattr("close") {
                    let _ = close.call0();
                }
            });
        }
    }
}
impl Drop for CallbackBatches {
    fn drop(&mut self) {
        self.close();
    }
}
impl Iterator for CallbackBatches {
    type Item = LadduDataResult<EventBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.as_ref()?;
        let result = Python::attach(|py| {
            let iterator = self.iterator.as_ref()?;
            let value = iterator.bind(py).clone().next()?;
            Some(
                value
                    .and_then(|value| {
                        let batch = super::data::event_batch_from_mapping(&value, &self.schema)?;
                        if self.chunk_size.is_some_and(|max| batch.len() > max) {
                            return Err(PyValueError::new_err(
                                "reader batch exceeds plan.chunk_size",
                            ));
                        }
                        Ok(batch)
                    })
                    .map_err(|e| callback_error(&self.context, e)),
            )
        });
        if let Some(Ok(batch)) = &result {
            self.seen += batch.len() as u64;
        }
        let result = if result.is_none() && self.expected.is_some_and(|n| n != self.seen) {
            Some(Err(LadduDataError::Source(format!(
                "{}: declared length {:?} differs from {} yielded events",
                self.context, self.expected, self.seen
            ))))
        } else {
            result
        };
        if result.as_ref().is_none_or(|r| r.is_err()) {
            self.close();
        }
        result
    }
}

struct PartitionedBatches {
    input: CallbackBatches,
    position: u64,
    plan: ReadPlan,
    count: Option<u64>,
}
impl Iterator for PartitionedBatches {
    type Item = LadduDataResult<EventBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let batch = match self.input.next()? {
                Ok(batch) => batch,
                Err(e) => return Some(Err(e)),
            };
            let base = self.position;
            self.position += batch.len() as u64;
            let rank = self.plan.rank() as u64;
            let nranks = self.plan.nranks() as u64;
            let rows: Vec<_> = (0..batch.len())
                .filter(|&i| {
                    let index = base + i as u64;
                    if let Some(count) = self.count {
                        let start =
                            (u128::from(count) * u128::from(rank) / u128::from(nranks)) as u64;
                        let end =
                            (u128::from(count) * u128::from(rank + 1) / u128::from(nranks)) as u64;
                        index >= start && index < end
                    } else {
                        index % nranks == rank
                    }
                })
                .collect();
            if !rows.is_empty() {
                let ids = rows.iter().map(|&i| base + i as u64).collect();
                return Some(batch.select(&rows).with_row_ids(ids));
            }
        }
    }
}

struct NativeBatches {
    input: CallbackBatches,
    position: u64,
    plan: ReadPlan,
    count: Option<u64>,
}
impl Iterator for NativeBatches {
    type Item = LadduDataResult<EventBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.input.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        let base = self.position;
        self.position += batch.len() as u64;
        if batch.row_ids().is_some() {
            return Some(Ok(batch));
        }
        let rank = self.plan.rank() as u64;
        let nranks = self.plan.nranks() as u64;
        let ids = match plan_partition(self.plan) {
            Partitioning::Rows => (0..batch.len())
                .map(|i| (base + i as u64) * nranks + rank)
                .collect(),
            Partitioning::Contiguous => {
                let start = (u128::from(self.count.expect("contiguous global count"))
                    * u128::from(rank)
                    / u128::from(nranks)) as u64;
                (0..batch.len()).map(|i| start + base + i as u64).collect()
            }
            Partitioning::FileGroups => {
                return Some(Err(LadduDataError::Source(
                    "native file_groups batches must include global row_ids".into(),
                )));
            }
        };
        Some(batch.with_row_ids(ids))
    }
}

pub(super) fn export_schema(dataset: &Dataset) -> LadduDataResult<Arc<Schema>> {
    let schema = dataset.schema()?;
    let (p4s, scalars) = names(&schema);
    Ok(Arc::new(
        Schema::new(p4s, scalars, true)?.with_columns(schema.columns().iter().cloned())?,
    ))
}

fn explicit_batch(batch: EventBatch, schema: Arc<Schema>) -> LadduDataResult<EventBatch> {
    EventBatch::new_with_columns_and_len(
        schema,
        (0..batch.schema().n_p4s())
            .map(|i| Arc::from(batch.vec4_column(i)))
            .collect(),
        (0..batch.schema().n_scalars())
            .map(|i| Arc::from(batch.scalar_column(i)))
            .collect(),
        (0..batch.schema().n_columns())
            .map(|i| batch.column(i).clone())
            .collect(),
        Some((0..batch.len()).map(|i| batch.weights_at(i)).collect()),
        batch.len(),
    )
}

fn read_plan(execution: &PyExecution, chunk_size: Option<usize>) -> ReadPlan {
    #[allow(unused_mut)]
    let mut plan = ReadPlan {
        chunk_size,
        ..ReadPlan::serial()
    };
    #[cfg(feature = "mpi")]
    if execution.inner.is_distributed() {
        plan.distribution = laddu_data::io::Distribution::Mpi {
            rank: execution.inner.rank(),
            nranks: execution.inner.nranks(),
            partitioning: execution.inner.partitioning(),
        };
    }
    #[cfg(not(feature = "mpi"))]
    let _ = execution;
    plan
}

pub(super) fn dataset_batches(
    py: Python<'_>,
    dataset: &Dataset,
    chunk_size: Option<usize>,
    execution: Option<&PyExecution>,
) -> PyResult<PyBatchIterator> {
    if chunk_size == Some(0) {
        return Err(PyValueError::new_err("chunk_size must be positive"));
    }
    let execution = execution
        .cloned()
        .map(Ok)
        .unwrap_or_else(PyExecution::default_inner)?;
    PyBatchIterator::open(dataset, read_plan(&execution, chunk_size)).map_err(|e| data_error(py, e))
}

#[pyclass(name = "BatchIterator", module = "laddu.io", skip_from_py_object)]
/// Closeable iterator over the logical dataset, with explicit effective weights.
pub struct PyBatchIterator {
    batches: Option<std::sync::Mutex<EventBatchIter>>,
    schema: Arc<Schema>,
    count: u64,
    exhausted: bool,
    chunk_size: usize,
    #[cfg(feature = "mpi")]
    gather: Option<GatherState>,
}
impl PyBatchIterator {
    pub(super) fn open(dataset: &Dataset, mut plan: ReadPlan) -> LadduDataResult<Self> {
        let schema = export_schema(dataset)?;
        if plan.chunk_size.is_none() {
            // Budget source, transformation, export arrays, identity metadata,
            // and the MPI wire buffer rather than only the source columns.
            let state = laddu_runtime::MemoryState::current();
            state.refresh();
            let available = dataset
                .memory_budget()
                .resolve(&state.host())
                .map_err(|e| LadduDataError::Source(format!("plan batch export: {e}")))?;
            let fields = schema
                .n_p4s()
                .checked_mul(4)
                .and_then(|n| n.checked_add(schema.n_scalars()))
                .and_then(|n| n.checked_add(2))
                .ok_or_else(|| LadduDataError::Source("export working-set overflow".into()))?;
            let integer_bytes: usize = schema
                .columns()
                .iter()
                .map(|(_, dtype)| dtype.width())
                .sum();
            let bytes = fields
                .checked_mul(8 * 6)
                .and_then(|n| {
                    integer_bytes
                        .checked_mul(6)
                        .and_then(|extra| n.checked_add(extra))
                })
                .ok_or_else(|| LadduDataError::Source("export working-set overflow".into()))?;
            let capacity = (available / bytes as u64).min(65_536) as usize;
            if capacity == 0 {
                return Err(LadduDataError::Source(
                    "memory budget cannot hold one exported event".into(),
                ));
            }
            plan.chunk_size = Some(capacity);
        }
        Ok(Self {
            batches: Some(std::sync::Mutex::new(dataset.stream_with_plan(plan)?)),
            schema,
            count: 0,
            exhausted: false,
            chunk_size: plan.chunk_size.expect("resolved export chunk size"),
            #[cfg(feature = "mpi")]
            gather: None,
        })
    }
    fn next_batch(&mut self) -> Option<LadduDataResult<EventBatch>> {
        #[cfg(feature = "mpi")]
        if self.gather.is_some() {
            return self.next_gathered();
        }
        self.batches
            .as_mut()?
            .get_mut()
            .expect("exclusive iterator access")
            .next()
            .map(|b| b.and_then(|b| explicit_batch(b, Arc::clone(&self.schema))))
    }
}
impl Drop for PyBatchIterator {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(feature = "mpi")]
struct GatherState {
    owner: i32,
    nranks: i32,
}

#[cfg(feature = "mpi")]
impl GatherState {
    fn stop(self) {
        use mpi::traits::*;
        let world = mpi::topology::SimpleCommunicator::world();
        world.process_at_rank(0).broadcast_into(&mut -1_i32);
    }
}

#[cfg(feature = "mpi")]
fn broadcast_bytes(world: &mpi::topology::SimpleCommunicator, rank: i32, bytes: &mut Vec<u8>) {
    use mpi::traits::*;
    let mut length = bytes.len() as u64;
    let root = world.process_at_rank(rank);
    root.broadcast_into(&mut length);
    bytes.resize(length as usize, 0);
    // MPI counts are signed 32-bit; segment even very large payloads.
    for chunk in bytes.chunks_mut(1024 * 1024) {
        root.broadcast_into(chunk);
    }
}

#[cfg(feature = "mpi")]
fn encode_next(iterator: &mut PyBatchIterator) -> Vec<u8> {
    match iterator.next_batch() {
        None => vec![0],
        Some(Err(error)) => {
            let mut bytes = vec![2];
            bytes.extend_from_slice(error.to_string().as_bytes());
            bytes
        }
        Some(Ok(batch)) => {
            let mut bytes = vec![1];
            bytes.extend_from_slice(&(batch.len() as u64).to_le_bytes());
            for i in 0..batch.schema().n_p4s() {
                for p4 in batch.vec4_column(i) {
                    for value in p4.components() {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
            for i in 0..batch.schema().n_scalars() {
                for value in batch.scalar_column(i) {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            for row in 0..batch.len() {
                bytes.extend_from_slice(&batch.weights_at(row).to_le_bytes());
            }
            for i in 0..batch.schema().n_columns() {
                batch.column(i).append_le_bytes(&mut bytes);
            }
            bytes
        }
    }
}

#[cfg(feature = "mpi")]
fn decode_batch(bytes: &[u8], schema: Arc<Schema>) -> LadduDataResult<EventBatch> {
    let invalid = || LadduDataError::Source("invalid distributed batch payload".into());
    let len = u64::from_le_bytes(
        bytes
            .get(..8)
            .ok_or_else(invalid)?
            .try_into()
            .map_err(|_| invalid())?,
    );
    let len = usize::try_from(len).map_err(|_| invalid())?;
    let fields = schema
        .n_p4s()
        .checked_mul(4)
        .and_then(|n| n.checked_add(schema.n_scalars()))
        .and_then(|n| n.checked_add(1))
        .ok_or_else(invalid)?;
    let float_end = len
        .checked_mul(fields)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(invalid)?;
    let integer_width: usize = schema
        .columns()
        .iter()
        .map(|(_, dtype)| dtype.width())
        .sum();
    let expected = len
        .checked_mul(integer_width)
        .and_then(|n| n.checked_add(float_end))
        .ok_or_else(invalid)?;
    if bytes.len() != expected {
        return Err(invalid());
    }
    let mut chunks = bytes[8..float_end].as_chunks::<8>().0.iter();
    let mut value = || -> f64 { f64::from_le_bytes(*chunks.next().expect("validated payload")) };
    let p4s = (0..schema.n_p4s())
        .map(|_| {
            (0..len)
                .map(|_| RealVec4::new(value(), value(), value(), value()))
                .collect()
        })
        .collect();
    let scalars = (0..schema.n_scalars())
        .map(|_| (0..len).map(|_| value()).collect())
        .collect();
    let weights = (0..len).map(|_| value()).collect();
    let mut position = float_end;
    let mut columns = Vec::with_capacity(schema.n_columns());
    for (_, dtype) in schema.columns() {
        let end = position + len * dtype.width();
        let data = &bytes[position..end];
        columns.push(laddu_data::columns::Column::from_le_bytes(*dtype, data)?);
        position = end;
    }
    EventBatch::new_with_columns_and_len(schema, p4s, scalars, columns, Some(weights), len)
}

#[cfg(feature = "mpi")]
impl PyBatchIterator {
    fn next_gathered(&mut self) -> Option<LadduDataResult<EventBatch>> {
        use mpi::traits::*;
        let world = mpi::topology::SimpleCommunicator::world();
        loop {
            let gather = self.gather.as_mut()?;
            if gather.owner >= gather.nranks {
                return None;
            }
            let mut owner = gather.owner;
            world.process_at_rank(0).broadcast_into(&mut owner);
            let mut bytes = if owner == 0 {
                // Avoid recursing into the gathered stream for the root's local input.
                let saved = self.gather.take();
                let bytes = encode_next(self);
                self.gather = saved;
                bytes
            } else {
                Vec::new()
            };
            broadcast_bytes(&world, owner, &mut bytes);
            match bytes.first() {
                Some(0) => self.gather.as_mut().expect("active gather").owner += 1,
                Some(1) => return Some(decode_batch(&bytes[1..], Arc::clone(&self.schema))),
                _ => {
                    return Some(Err(LadduDataError::Source(format!(
                        "rank {owner}: {}",
                        String::from_utf8_lossy(&bytes[1..])
                    ))));
                }
            }
        }
    }
}

#[cfg(feature = "mpi")]
fn write_gathered(
    py: Python<'_>,
    sink: &PyFormatSink,
    mut iterator: PyBatchIterator,
) -> PyResult<PyWriteResult> {
    use mpi::traits::*;
    let world = mpi::topology::SimpleCommunicator::world();
    let result = if world.rank() == 0 {
        iterator.gather = Some(GatherState {
            owner: 0,
            nranks: world.size(),
        });
        let schema = Arc::clone(&iterator.schema);
        let iterator = Py::new(py, iterator)?;
        sink.invoke(py, &sink.target, &iterator, schema, "single")
            .map(|()| iterator.borrow(py).count)
    } else {
        loop {
            let mut owner = 0_i32;
            world.process_at_rank(0).broadcast_into(&mut owner);
            if owner < 0 {
                break;
            }
            let mut bytes = if owner == world.rank() {
                encode_next(&mut iterator)
            } else {
                Vec::new()
            };
            broadcast_bytes(&world, owner, &mut bytes);
        }
        iterator.close();
        Ok(0)
    };
    let count = coordinated(py, &sink.execution, result)?;
    let mut total = count;
    world.process_at_rank(0).broadcast_into(&mut total);
    Ok(PyWriteResult {
        paths: vec![sink.target.clone()],
        counts: vec![total],
    })
}
#[pymethods]
impl PyBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<PyEventBatch> {
        if self.batches.is_none() && !self.exhausted {
            return Err(PyRuntimeError::new_err(
                "batch iterator is closed before exhaustion",
            ));
        }
        match py.detach(|| self.next_batch()) {
            Some(Ok(batch)) => {
                self.count += batch.len() as u64;
                Ok(PyEventBatch { inner: batch })
            }
            Some(Err(e)) => {
                self.close();
                Err(data_error(py, e))
            }
            None => {
                self.exhausted = true;
                self.close();
                Err(pyo3::exceptions::PyStopIteration::new_err(()))
            }
        }
    }
    fn close(&mut self) {
        self.batches = None;
        #[cfg(feature = "mpi")]
        if let Some(gather) = self.gather.take() {
            gather.stop();
        }
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __exit__(
        &mut self,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.close();
    }
}

#[pyclass]
struct BuiltinWriter {
    precision: laddu_data::schema::Precision,
    tree: Option<String>,
}

#[pymethods]
impl BuiltinWriter {
    #[pyo3(signature = (target, batches, *, options, schema, plan))]
    fn __call__(
        &self,
        py: Python<'_>,
        target: PathBuf,
        batches: Py<PyBatchIterator>,
        options: &Bound<'_, PyAny>,
        schema: &PySchema,
        plan: &PyWritePlan,
    ) -> PyResult<()> {
        let _ = (options, plan);
        let mut sink: Box<dyn EventSink> = if let Some(tree) = &self.tree {
            Box::new(
                laddu_data::io::root::RootSink::builder(target)
                    .tree(tree.as_str())
                    .precision(self.precision)
                    .build(),
            )
        } else {
            Box::new(
                laddu_data::io::parquet::ParquetSink::builder(target)
                    .precision(self.precision)
                    .build(),
            )
        };
        let result = (|| {
            sink.begin(Arc::clone(&schema.inner), WritePlan::default())
                .map_err(|e| data_error(py, e))?;
            loop {
                let batch = match batches.borrow_mut(py).__next__(py) {
                    Ok(batch) => batch,
                    Err(error) if error.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) => {
                        break;
                    }
                    Err(error) => return Err(error),
                };
                sink.write_batch(&batch.inner)
                    .map_err(|e| data_error(py, e))?;
            }
            sink.finish().map_err(|e| data_error(py, e))
        })();
        if result.is_err() {
            let _ = sink.abort();
        }
        result
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn builtin_write(
    py: Python<'_>,
    dataset: &PyDataset,
    target: PathBuf,
    precision: laddu_data::schema::Precision,
    tree: Option<String>,
    output: &str,
    chunk_size: Option<usize>,
    execution: Option<&PyExecution>,
) -> PyResult<PyWriteResult> {
    let (name, extension) = if tree.is_some() {
        ("root", "root")
    } else {
        ("parquet", "parquet")
    };
    let writer = Py::new(py, BuiltinWriter { precision, tree })?.into_any();
    let spec = PyFormatSpec::new(
        py,
        name.into(),
        None,
        None,
        Some(WriterInput(writer)),
        extension,
        None,
    )?;
    spec.write(py, dataset, target, None, output, chunk_size, execution)
}

#[pyclass(name = "WriteResult", module = "laddu.io", frozen, skip_from_py_object)]
/// Successful output paths and corresponding event counts, in rank order.
pub struct PyWriteResult {
    #[pyo3(get)]
    paths: Vec<PathBuf>,
    #[pyo3(get)]
    counts: Vec<u64>,
}
#[pymethods]
impl PyWriteResult {
    #[getter]
    fn events(&self) -> u64 {
        self.counts.iter().sum()
    }
}

#[pyclass(name = "FormatSink", module = "laddu.io", frozen, skip_from_py_object)]
/// Bound output settings accepted by Dataset.write_to(sink).
pub struct PyFormatSink {
    spec: Arc<Spec>,
    target: PathBuf,
    options: Arc<Py<PyAny>>,
    output: String,
    chunk_size: Option<usize>,
    execution: PyExecution,
}
impl PyFormatSink {
    pub(super) fn write_with_overrides(
        &self,
        py: Python<'_>,
        dataset: &PyDataset,
        output: Option<&str>,
        chunk_size: Option<usize>,
        execution: Option<&PyExecution>,
    ) -> PyResult<PyWriteResult> {
        let spec = PyFormatSpec {
            spec: Arc::clone(&self.spec),
        };
        let sink = spec.sink(
            py,
            self.target.clone(),
            Some(self.options.clone_ref(py)),
            output.unwrap_or(&self.output),
            chunk_size.or(self.chunk_size),
            Some(execution.unwrap_or(&self.execution)),
        )?;
        sink.write(py, dataset)
    }
    pub(super) fn write(&self, py: Python<'_>, dataset: &PyDataset) -> PyResult<PyWriteResult> {
        write_format(py, self, dataset)
    }
    fn invoke(
        &self,
        py: Python<'_>,
        target: &PathBuf,
        iterator: &Py<PyBatchIterator>,
        schema: Arc<Schema>,
        mode: &str,
    ) -> PyResult<()> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("options", self.options.bind(py))?;
        kwargs.set_item("schema", PySchema { inner: schema })?;
        kwargs.set_item(
            "plan",
            PyWritePlan {
                rank: self.execution.inner.rank(),
                nranks: self.execution.inner.nranks(),
                partitioning: partition_name(self.execution.inner.partitioning()).into(),
                output: mode.into(),
                chunk_size: Some(iterator.borrow(py).chunk_size),
            },
        )?;
        let result = self
            .spec
            .writer
            .as_ref()
            .expect("validated writer")
            .bind(py)
            .call((target, iterator.bind(py)), Some(&kwargs));
        let mut batches = iterator.borrow_mut(py);
        let exhausted = batches.exhausted;
        batches.close();
        result.map_err(|e| {
            let error =
                PyRuntimeError::new_err(format!("{} write_batches failed: {e}", self.spec.name));
            error.set_cause(py, Some(e));
            error
        })?;
        if !exhausted {
            return Err(PyRuntimeError::new_err(
                "writer returned before consuming the entire batch iterator",
            ));
        }
        Ok(())
    }
}

fn write_format(
    py: Python<'_>,
    sink: &PyFormatSink,
    dataset: &PyDataset,
) -> PyResult<PyWriteResult> {
    let execution = &sink.execution;
    let mode = if sink.output == "auto" {
        if execution.inner.is_distributed() {
            "sharded"
        } else {
            "single"
        }
    } else {
        &sink.output
    };
    let plan = read_plan(execution, sink.chunk_size);
    let opened = PyBatchIterator::open(&dataset.inner, plan).map_err(|e| data_error(py, e));
    let iterator = coordinated(py, execution, opened)?;
    agree(
        execution,
        &format!(
            "{:?}|{}|{}|{:?}|{:?}|{:?}|{}",
            sink.target,
            mode,
            sink.spec.name,
            sink.chunk_size,
            names(&iterator.schema),
            iterator.schema.columns(),
            partition_name(execution.inner.partitioning())
        ),
    )?;
    let schema = Arc::clone(&iterator.schema);
    let target = if mode == "sharded" {
        shard_path(
            &sink.target,
            execution.inner.rank(),
            execution.inner.nranks(),
            &sink.spec.extension,
        )
    } else {
        sink.target.clone()
    };
    let parent = target.parent().filter(|p| !p.as_os_str().is_empty());
    coordinated(
        py,
        execution,
        parent.map_or(Ok(()), |p| std::fs::create_dir_all(p).map_err(to_py_err)),
    )?;
    #[cfg(feature = "mpi")]
    if execution.inner.is_distributed() && mode == "single" {
        return write_gathered(py, sink, iterator);
    }
    let iterator = Py::new(py, iterator)?;
    let result = sink.invoke(py, &target, &iterator, schema, mode);
    coordinated(py, execution, result)?;
    let count = iterator.borrow(py).count;
    let counts = collect_counts(execution, count);
    let paths = if mode == "sharded" {
        (0..execution.inner.nranks())
            .map(|rank| {
                shard_path(
                    &sink.target,
                    rank,
                    execution.inner.nranks(),
                    &sink.spec.extension,
                )
            })
            .collect()
    } else {
        vec![target]
    };
    Ok(PyWriteResult { paths, counts })
}

fn shard_path(path: &std::path::Path, rank: usize, nranks: usize, extension: &str) -> PathBuf {
    if let Some(ext) = path.extension() {
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        path.with_file_name(format!(
            "{stem}.rank{rank:05}-of{nranks:05}.{}",
            ext.to_string_lossy()
        ))
    } else {
        path.join(format!("part-rank{rank:05}-of{nranks:05}.{extension}"))
    }
}

fn coordinated<T>(py: Python<'_>, execution: &PyExecution, result: PyResult<T>) -> PyResult<T> {
    #[cfg(feature = "mpi")]
    if execution.inner.is_distributed() {
        use mpi::traits::*;
        let world = mpi::topology::SimpleCommunicator::world();
        for rank in 0..world.size() {
            let mut message = if world.rank() == rank {
                result
                    .as_ref()
                    .err()
                    .map(ToString::to_string)
                    .unwrap_or_default()
                    .into_bytes()
            } else {
                Vec::new()
            };
            broadcast_bytes(&world, rank, &mut message);
            if !message.is_empty() {
                return match result {
                    Err(e) => Err(e),
                    Ok(_) => Err(PyRuntimeError::new_err(format!(
                        "I/O failed on rank {rank}: {}",
                        String::from_utf8_lossy(&message)
                    ))),
                };
            }
        }
    }
    let _ = (py, execution);
    result
}
fn agree(execution: &PyExecution, local: &str) -> PyResult<()> {
    #[cfg(feature = "mpi")]
    if execution.inner.is_distributed() {
        use mpi::traits::*;
        let world = mpi::topology::SimpleCommunicator::world();
        let mut mismatch = 0_i32;
        for rank in 0..world.size() {
            let mut bytes = if rank == world.rank() {
                local.as_bytes().to_vec()
            } else {
                Vec::new()
            };
            broadcast_bytes(&world, rank, &mut bytes);
            if bytes != local.as_bytes() {
                mismatch = 1;
            }
        }
        let mut global = 0_i32;
        world.all_reduce_into(
            &mismatch,
            &mut global,
            mpi::collective::SystemOperation::max(),
        );
        if global != 0 {
            return Err(PyValueError::new_err(
                "MPI ranks disagree on format metadata or output settings",
            ));
        }
    }
    let _ = (execution, local);
    Ok(())
}
fn collect_counts(execution: &PyExecution, count: u64) -> Vec<u64> {
    #[cfg(feature = "mpi")]
    if execution.inner.is_distributed() {
        use mpi::traits::*;
        let world = mpi::topology::SimpleCommunicator::world();
        let mut counts = vec![0; world.size() as usize];
        world.all_gather_into(&count, &mut counts);
        return counts;
    }
    let _ = execution;
    vec![count]
}

#[pyfunction]
#[pyo3(signature = (source: "object", target, *, reader, writer, read_options: "object | None" = None, write_options: "object | None" = None, output="auto", chunk_size=None, memory=None, execution=None))]
/// Convert between external formats through bounded logical event batches.
#[allow(clippy::too_many_arguments)]
pub fn convert(
    py: Python<'_>,
    source: Py<PyAny>,
    target: PathBuf,
    reader: &PyFormatSpec,
    writer: &PyFormatSpec,
    read_options: Option<Py<PyAny>>,
    write_options: Option<Py<PyAny>>,
    output: &str,
    chunk_size: Option<usize>,
    memory: Option<MemoryInput>,
    execution: Option<&PyExecution>,
) -> PyResult<PyWriteResult> {
    let execution = execution
        .cloned()
        .map(Ok)
        .unwrap_or_else(PyExecution::default_inner)?;
    let dataset = reader.read(
        py,
        source,
        read_options,
        memory,
        "streaming",
        Some(&execution),
    )?;
    writer.write(
        py,
        &dataset,
        target,
        write_options,
        output,
        chunk_size,
        Some(&execution),
    )
}

#[pymodule(submodule, gil_used = false)]
/// External format specifications and canonical batch types.
pub mod io {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.setattr("__name__", "laddu.io")?;
        module
            .py()
            .import("sys")?
            .getattr("modules")?
            .set_item("laddu.io", module)?;
        Ok(())
    }
    #[pymodule_export]
    use super::{
        PyBatchIterator as BatchIterator, PyEventBatch as EventBatch, PyFormatSink as FormatSink,
        PyFormatSource as FormatSource, PyFormatSpec as FormatSpec, PyReadPlan as ReadPlan,
        PySchema as Schema, PySourceInfo as SourceInfo, PyWritePlan as WritePlan,
        PyWriteResult as WriteResult, convert,
    };
}
