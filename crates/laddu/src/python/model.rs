use std::path::PathBuf;

use laddu_compile::CompiledModel;
use laddu_expr::ExprNode;
use laddu_runtime::{Device, PreparedModel};
use numpy::{PyArray1, PyArray2};
use pyo3::{
    IntoPyObjectExt,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyDict},
};

use super::{
    data::PyDataset,
    error::to_py_err,
    expr::PyExpr,
    float_vec,
    runtime::PyExecution,
    visualization::{
        PyNodeStyleRule, expression_dot, expression_equation, expression_latex, expression_svg,
        expression_tree,
    },
};

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
    if let Ok(values) = float_vec(values) {
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

impl PyModel {
    fn validate_without_dataset(&self, device: &Device) -> PyResult<()> {
        if matches!(device, Device::Gpu(_)) {
            return Err(to_py_err(
                "model evaluation without a dataset is not supported by the GPU backend",
            ));
        }
        // The parameter-only JIT entry point cannot read event inputs.
        for node in self.inner.graph().nodes() {
            let input = match node {
                ExprNode::EventScalar(name) => format!("scalar '{name}'"),
                ExprNode::EventP4Component { name, component } => {
                    format!("four-momentum '{name}.{}'", component.label())
                }
                _ => continue,
            };
            return Err(to_py_err(laddu_runtime::RuntimeError::Data(format!(
                "model requires event input {input}; provide a dataset to evaluate it"
            ))));
        }
        Ok(())
    }
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

    fn __str__(&self) -> String {
        self.inner.graph().to_string()
    }

    /// Return the optimized model as a compact mathematical equation.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Built-in ANSI color palette. The default emits no color.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    #[pyo3(signature = (*, colors=None, style_rules=None))]
    fn equation(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_equation(self.inner.graph(), colors, style_rules)
    }

    /// Return the optimized model as a LaTeX math-mode fragment.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Built-in ``\\color[RGB]`` palette. Colored output requires the
    ///     LaTeX ``xcolor`` package. The default emits no color commands.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom foreground-color rules applied after the preset.
    #[pyo3(signature = (*, colors=None, style_rules=None))]
    fn latex(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_latex(self.inner.graph(), colors, style_rules)
    }

    /// Return an indented tree representation of the optimized model graph.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Color palette for ANSI terminal output. The default emits no color.
    /// expand_repeated : bool, default=True
    ///     Expand shared subtrees at every occurrence. If false, later
    ///     occurrences are printed as references.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    #[pyo3(signature = (*, colors=None, expand_repeated=true, style_rules=None))]
    fn tree(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        expand_repeated: bool,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_tree(self.inner.graph(), colors, expand_repeated, style_rules)
    }

    /// Return the optimized model graph as Graphviz DOT source.
    ///
    /// Parameters
    /// ----------
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Color palette for graph text, fills, and borders.
    /// expand_repeated : bool, default=True
    ///     Duplicate shared subtrees at every occurrence. If false, emit a
    ///     shared directed acyclic graph.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    #[pyo3(signature = (*, colors=None, expand_repeated=true, style_rules=None))]
    fn dot(
        &self,
        colors: Option<&Bound<'_, PyAny>>,
        expand_repeated: bool,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<String> {
        expression_dot(self.inner.graph(), colors, expand_repeated, style_rules)
    }

    /// Render the optimized model graph to an SVG file.
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike
    ///     Destination SVG path. An existing file is replaced.
    /// colors : {'light', 'dark', 'none'}, optional
    ///     Color palette for graph text, fills, and borders.
    /// expand_repeated : bool, default=True
    ///     Duplicate shared subtrees at every occurrence. If false, render a
    ///     shared directed acyclic graph.
    /// style_rules : sequence of NodeStyleRule, optional
    ///     Custom rules applied after the preset; later rules take precedence.
    ///
    /// Returns
    /// -------
    /// None
    #[pyo3(signature = (path, *, colors=None, expand_repeated=true, style_rules=None))]
    fn svg(
        &self,
        path: PathBuf,
        colors: Option<&Bound<'_, PyAny>>,
        expand_repeated: bool,
        style_rules: Option<Vec<PyNodeStyleRule>>,
    ) -> PyResult<()> {
        expression_svg(
            self.inner.graph(),
            &path,
            colors,
            expand_repeated,
            style_rules,
        )
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

    #[pyo3(signature = (
        dataset=None,
        *,
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        execution=None,
        real=false
    ) -> "complex | float | numpy.typing.NDArray[numpy.complex128 | numpy.float64]")]
    /// Evaluate a scalar model at parameter values, optionally for each event.
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset or None, optional
    ///     Events containing any required scalar and four-vector columns.
    ///     Omit or pass ``None`` for expressions that need no event inputs.
    /// parameters : sequence of float or dict, optional
    ///     Free values in :attr:`parameter_names` order, or a partial mapping by
    ///     name. Omitted mapping entries use their defaults; omitting
    ///     ``parameters`` uses :attr:`default_parameters`. Fixed parameters
    ///     retain their fixed values and are not part of the ordered sequence.
    /// execution : Execution, optional
    ///     Runtime backend configuration. Without a dataset, defaults to
    ///     automatic CPU/JIT selection; explicit CPU/JIT precision and
    ///     differentiation settings are honored. GPU execution requires a
    ///     dataset and is rejected otherwise, without falling back to CPU.
    /// real : bool, default=False
    ///     Return only real components: a Python ``float`` without a dataset,
    ///     or a ``float64`` array with one.
    ///
    /// Returns
    /// -------
    /// complex or float or numpy.ndarray
    ///     Without a dataset, one Python ``complex`` (``float`` if ``real=True``).
    ///     With a dataset, an array of shape ``(n_events,)`` and dtype
    ///     ``complex128`` (``float64`` if ``real=True``). A supplied one-event
    ///     dataset retains its event dimension, returning shape ``(1,)``.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the parameter representation is invalid.
    /// LadduError
    ///     If the result is not scalar, required event inputs are missing,
    ///     GPU execution is requested without a dataset, or preparation,
    ///     parameter validation, dataset reading, or evaluation fails.
    ///
    /// Notes
    /// -----
    /// Vector and matrix operations may appear inside a scalar expression.
    /// Select an individual vector component (``vector[i]``) or matrix element
    /// (``matrix.at(i, j)``) before building the model; whole-array results
    /// are not supported.
    ///
    /// Examples
    /// --------
    /// >>> import laddu as ld
    /// >>> z = ld.complex(ld.parameter('x'), ld.parameter('y'))
    /// >>> model = ld.Model(z * z + 1.0)
    /// >>> model.evaluate(parameters={'x': 2.0, 'y': 3.0})
    /// (-4+12j)
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        dataset: Option<&PyDataset>,
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
        if dataset.is_none() {
            self.validate_without_dataset(execution.inner.requested_device())?;
        }
        let plan = PreparedModel::prepare(&self.inner, &execution.inner).map_err(to_py_err)?;
        let Some(dataset) = dataset else {
            let value = py
                .detach(move || match plan {
                    PreparedModel::Cpu(plan) => plan.evaluate(&params),
                    #[cfg(feature = "wgpu")]
                    PreparedModel::Wgpu(_) => Err(laddu_runtime::RuntimeError::Wgpu(
                        "model evaluation without a dataset is not supported by the GPU backend"
                            .into(),
                    )),
                })
                .map_err(to_py_err)?;
            return if real {
                value.re.into_bound_py_any(py)
            } else {
                value.into_bound_py_any(py)
            };
        };
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

    #[pyo3(signature = (
        dataset=None,
        *,
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        execution=None,
        real=false
    ) -> "tuple[complex | float | numpy.typing.NDArray[numpy.complex128 | numpy.float64], numpy.typing.NDArray[numpy.complex128 | numpy.float64]]")]
    /// Evaluate a scalar model and its derivatives with respect to free parameters.
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset or None, optional
    ///     Events containing any required scalar and four-vector columns.
    ///     Omit or pass ``None`` for expressions that need no event inputs.
    /// parameters : sequence of float or dict, optional
    ///     Free values in :attr:`parameter_names` order, or a partial mapping by
    ///     name. Omitted mapping entries use their defaults; omitting
    ///     ``parameters`` uses :attr:`default_parameters`. Fixed parameters
    ///     retain their fixed values and do not contribute gradient entries.
    /// execution : Execution, optional
    ///     Runtime backend configuration. Without a dataset, defaults to
    ///     automatic CPU/JIT selection; explicit CPU/JIT precision and
    ///     differentiation settings are honored. GPU execution requires a
    ///     dataset and is rejected otherwise, without falling back to CPU.
    /// real : bool, default=False
    ///     Return real components of both values and derivatives, not their
    ///     magnitudes. Arrays have dtype ``float64`` instead of ``complex128``.
    ///
    /// Returns
    /// -------
    /// values : complex or float or numpy.ndarray
    ///     Without a dataset, one Python ``complex`` (``float`` if ``real=True``).
    ///     With a dataset, an array of shape ``(n_events,)``. A supplied one-event
    ///     dataset retains shape ``(1,)``.
    /// gradients : numpy.ndarray
    ///     Shape ``(n_free_parameters,)`` without a dataset, or
    ///     ``(n_events, n_free_parameters)`` with one, including when there is
    ///     only one event. Entries follow :attr:`parameter_names` order and
    ///     differentiate with respect to the real free parameter values.
    ///     With no free parameters, these shapes are ``(0,)`` and
    ///     ``(n_events, 0)`` respectively. Value and gradient arrays have dtype
    ///     ``complex128`` (``float64`` if ``real=True``).
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the parameter representation is invalid.
    /// LadduError
    ///     If the result is not scalar, required event inputs are missing,
    ///     GPU execution is requested without a dataset, or automatic
    ///     differentiation, preparation, parameter validation, reading, or
    ///     evaluation fails.
    ///
    /// Notes
    /// -----
    /// Vector and matrix operations may appear inside a scalar expression.
    /// Select an individual component or element before building the model;
    /// whole-array results and their Jacobians are not supported.
    ///
    /// Examples
    /// --------
    /// >>> import laddu as ld
    /// >>> z = ld.complex(ld.parameter('x'), ld.parameter('y'))
    /// >>> model = ld.Model(z * z + 1.0)
    /// >>> value, gradient = model.value_and_gradient(parameters={'x': 2.0, 'y': 3.0})
    /// >>> value
    /// (-4+12j)
    /// >>> model.parameter_names
    /// ['x', 'y']
    /// >>> gradient.tolist()
    /// [(4+6j), (-6+4j)]
    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        dataset: Option<&PyDataset>,
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
        if dataset.is_none() {
            self.validate_without_dataset(execution.inner.requested_device())?;
        }
        let plan = PreparedModel::prepare(&self.inner, &execution.inner).map_err(to_py_err)?;
        let Some(dataset) = dataset else {
            let evaluation = py
                .detach(move || match plan {
                    PreparedModel::Cpu(plan) => plan.evaluate_with_gradient(&params),
                    #[cfg(feature = "wgpu")]
                    PreparedModel::Wgpu(_) => Err(laddu_runtime::RuntimeError::Wgpu(
                        "model evaluation without a dataset is not supported by the GPU backend"
                            .into(),
                    )),
                })
                .map_err(to_py_err)?;
            return if real {
                Ok((
                    evaluation.value().re.into_bound_py_any(py)?,
                    PyArray1::from_vec(
                        py,
                        evaluation.gradient().iter().map(|value| value.re).collect(),
                    )
                    .into_any(),
                ))
            } else {
                Ok((
                    evaluation.value().into_bound_py_any(py)?,
                    PyArray1::from_vec(py, evaluation.gradient().to_vec()).into_any(),
                ))
            };
        };
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

impl_json_methods!(PyModel);

#[cfg(test)]
mod tests {
    use laddu_expr::event_scalar;

    use super::*;

    #[test]
    fn visualization_methods_render_the_optimized_model_graph() {
        let expression = PyExpr::from(event_scalar("mass") + 1.0);
        let model = PyModel::new(&expression).unwrap();

        assert!(model.equation(None, None).unwrap().contains("mass"));
        assert!(model.latex(None, None).unwrap().contains("mass"));
        assert!(
            model
                .tree(None, true, None)
                .unwrap()
                .contains("ExprGraph(root=#")
        );
        assert!(
            model
                .dot(None, false, None)
                .unwrap()
                .contains("digraph ExprGraph")
        );
        let path = std::env::temp_dir().join(format!(
            "laddu-model-visualization-{}.svg",
            std::process::id()
        ));
        model.svg(path.clone(), None, false, None).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("<svg"));
        std::fs::remove_file(path).unwrap();
        assert_eq!(model.__str__(), model.equation(None, None).unwrap());
    }
}
