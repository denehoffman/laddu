use laddu_compile::CompiledModel;
use laddu_runtime::PreparedModel;
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyDict},
};

use super::{data::PyDataset, error::to_py_err, expr::PyExpr, runtime::PyExecution};

/// Resolve Python parameter values into the model's free-parameter order.
///
/// A sequence is accepted verbatim. A mapping starts from the model's default
/// values and replaces entries whose parameter names are present.
///
/// Raises
/// ------
/// TypeError
///     If `values` is neither a numeric sequence nor a mapping, or a mapped
///     value is not numeric.
/// LadduError
///     If the compiled parameter layout is inconsistent.
pub fn model_free_values(model: &CompiledModel, values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(values) = values.extract::<Vec<f64>>() {
        return Ok(values);
    }
    if let Ok(mapping) = values.cast::<PyDict>() {
        let mut out = model.params().initial_free_values();
        for (index, id) in model.params().free_params().iter().enumerate() {
            let name = model.params().name(*id).map_err(to_py_err)?;
            if let Some(value) = mapping.get_item(name)? {
                out[index] = value.extract()?;
            }
        }
        return Ok(out);
    }
    Err(PyTypeError::new_err(
        "parameters must be a numeric sequence or dict keyed by parameter name",
    ))
}

#[pyclass(name = "Model", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A compiled symbolic model ready for repeated dataset evaluation.
///
/// Parameters
/// ----------
/// expr : Expr
///     Root expression to compile. Compilation validates shapes and parameters,
///     optimizes the graph, and prepares it for the selected runtime backend.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> slope = ld.parameter("slope", initial=2.0)
/// >>> model = ld.Model(slope * ld.scalar("x"))
/// >>> model.parameter_names
/// ['slope']
pub struct PyModel {
    pub(crate) inner: CompiledModel,
}

#[pymethods]
impl PyModel {
    /// Compile a symbolic expression into a model.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If expression shapes, metadata, or parameter definitions are invalid.
    #[new]
    fn new(expr: &PyExpr) -> PyResult<Self> {
        Ok(Self {
            inner: CompiledModel::from_expr(&expr.inner).map_err(to_py_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("Model(parameters={:?})", self.parameter_names())
    }

    #[getter]
    /// list of str: Free parameter names in evaluation order.
    fn parameter_names(&self) -> Vec<String> {
        self.inner
            .params()
            .free_params()
            .iter()
            .map(|id| {
                self.inner
                    .params()
                    .name(*id)
                    .unwrap_or("<invalid>")
                    .to_owned()
            })
            .collect()
    }

    #[getter]
    /// list of float: Default values for all free parameters.
    fn default_parameters(&self) -> Vec<f64> {
        self.inner.params().initial_free_values()
    }

    #[pyo3(signature = (*, seed=0))]
    /// Sample reproducible initial values from parameter initialization ranges.
    ///
    /// Parameters
    /// ----------
    /// seed : int, default=0
    ///     Random seed.
    fn sample_parameters(&self, seed: u64) -> Vec<f64> {
        self.inner.params().sample_initial(seed)
    }

    /// Compile a model containing only expression contributions with selected tags.
    ///
    /// Parameters
    /// ----------
    /// tags : sequence of str
    ///     Projection tags to retain.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the projected expression cannot be compiled.
    fn projection(&self, tags: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .project_tags(tags.iter().map(String::as_str))
                .map_err(to_py_err)?,
        })
    }

    /// Return a recompiled model with one parameter fixed.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the parameter is unknown or the fixed value violates its bounds.
    fn fix(&self, name: &str, value: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.fix_parameter(name, value).map_err(to_py_err)?,
        })
    }

    /// Return a recompiled model with a fixed parameter made free.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the parameter is unknown or cannot be made free.
    fn free(&self, name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.free_parameter(name).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (dataset, *, parameters=None, execution=None, real=false))]
    /// Evaluate the model for every event in a dataset.
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     Events to evaluate, including any required scalar and four-vector
    ///     columns.
    /// parameters : sequence of float or dict, optional
    ///     Free values in :attr:`parameter_names` order, or a partial mapping by
    ///     name. Omitted entries use their defaults.
    /// execution : Execution, optional
    ///     Runtime backend configuration. Defaults to local automatic selection.
    /// real : bool, default=False
    ///     Return only real components as ``float64`` instead of complex values.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     One value per event.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the parameter representation is invalid.
    /// LadduError
    ///     If preparation, dataset reading, or evaluation fails.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
        parameters: Option<&Bound<'_, PyAny>>,
        execution: Option<&PyExecution>,
        real: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        let free = match parameters {
            Some(values) => model_free_values(&self.inner, values)?,
            None => self.inner.params().initial_free_values(),
        };
        let params = self.inner.params().values(&free).map_err(to_py_err)?;
        let plan = PreparedModel::prepare(&self.inner, &execution.inner).map_err(to_py_err)?;
        let dataset = dataset.inner.clone();
        let values = py
            .detach(move || {
                let mut values = Vec::new();
                for batch in dataset
                    .batches()
                    .map_err(|error| laddu_runtime::RuntimeError::Data(error.to_string()))?
                {
                    values.extend(plan.evaluate_batch(
                        &params,
                        &batch.map_err(|error| {
                            laddu_runtime::RuntimeError::Data(error.to_string())
                        })?,
                    )?);
                }
                Ok::<_, laddu_runtime::RuntimeError>(values)
            })
            .map_err(to_py_err)?;
        if real {
            Ok(
                PyArray1::from_vec(py, values.into_iter().map(|value| value.re).collect())
                    .into_any(),
            )
        } else {
            Ok(PyArray1::from_vec(py, values).into_any())
        }
    }

    #[pyo3(signature = (dataset, *, parameters=None, execution=None, real=false))]
    /// Evaluate model values and derivatives for every event.
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     Events to evaluate.
    /// parameters : sequence of float or dict, optional
    ///     Free parameter values or a partial name-to-value mapping.
    /// execution : Execution, optional
    ///     Runtime backend configuration.
    /// real : bool, default=False
    ///     Return real components instead of complex values and derivatives.
    ///
    /// Returns
    /// -------
    /// values : numpy.ndarray
    ///     Shape ``(n_events,)``.
    /// gradients : numpy.ndarray
    ///     Shape ``(n_events, n_free_parameters)`` in
    ///     :attr:`parameter_names` order.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the parameter representation is invalid.
    /// LadduError
    ///     If automatic differentiation, preparation, reading, or evaluation
    ///     fails.
    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
        parameters: Option<&Bound<'_, PyAny>>,
        execution: Option<&PyExecution>,
        real: bool,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        let free = match parameters {
            Some(values) => model_free_values(&self.inner, values)?,
            None => self.inner.params().initial_free_values(),
        };
        let params = self.inner.params().values(&free).map_err(to_py_err)?;
        let plan = PreparedModel::prepare(&self.inner, &execution.inner).map_err(to_py_err)?;
        let dataset = dataset.inner.clone();
        let evaluations = py
            .detach(move || {
                let mut evaluations = Vec::new();
                for batch in dataset
                    .batches()
                    .map_err(|error| laddu_runtime::RuntimeError::Data(error.to_string()))?
                {
                    evaluations.extend(plan.evaluate_batch_with_gradient(
                        &params,
                        &batch.map_err(|error| {
                            laddu_runtime::RuntimeError::Data(error.to_string())
                        })?,
                    )?);
                }
                Ok::<_, laddu_runtime::RuntimeError>(evaluations)
            })
            .map_err(to_py_err)?;
        if real {
            let values = evaluations.iter().map(|value| value.value().re).collect();
            let gradients = evaluations
                .iter()
                .map(|value| value.gradient().iter().map(|entry| entry.re).collect())
                .collect::<Vec<Vec<f64>>>();
            Ok((
                PyArray1::from_vec(py, values).into_any(),
                PyArray2::from_vec2(py, &gradients)?.into_any(),
            ))
        } else {
            let values = evaluations.iter().map(|value| value.value()).collect();
            let gradients = evaluations
                .iter()
                .map(|value| value.gradient().to_vec())
                .collect::<Vec<_>>();
            Ok((
                PyArray1::from_vec(py, values).into_any(),
                PyArray2::from_vec2(py, &gradients)?.into_any(),
            ))
        }
    }
}
