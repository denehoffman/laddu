use std::sync::Arc;

use laddu_compile::NormalizationStrategy;
use laddu_likelihood::{
    CrossSectionIntegrals, DatasetDiagnostics, DatasetRole, ExtendedNllTerm, LassoPenalty,
    Likelihood, LikelihoodDiagnostics, LikelihoodProjection, LikelihoodTerm, NllTerm, RidgePenalty,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyDict, PyList},
};

use super::{
    cross_section::{PyCrossSection, PyEnsemble},
    data::PyDataset,
    error::to_py_err,
    float_vec,
    model::PyModel,
    runtime::PyExecution,
};

#[pyclass(
    name = "DatasetDiagnostics",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Preparation statistics for one dataset used by a likelihood term.
pub struct PyDatasetDiagnostics {
    inner: DatasetDiagnostics,
}

#[pymethods]
impl PyDatasetDiagnostics {
    #[getter]
    fn term(&self) -> &str {
        self.inner.term()
    }

    #[getter]
    fn role(&self) -> &'static str {
        match self.inner.role() {
            DatasetRole::Observed => "observed",
            DatasetRole::AcceptedMc => "accepted_mc",
        }
    }

    #[getter]
    fn storage(&self) -> &'static str {
        match self.inner.stats().storage() {
            laddu_data::data::CacheStorage::Resident => "resident",
            laddu_data::data::CacheStorage::Streaming => "streaming",
        }
    }

    #[getter]
    fn local_events(&self) -> usize {
        self.inner.stats().local_events()
    }

    #[getter]
    fn global_events(&self) -> usize {
        self.inner.stats().global_events()
    }

    #[getter]
    fn local_batches(&self) -> usize {
        self.inner.stats().local_batches()
    }

    #[getter]
    fn sum_weights(&self) -> f64 {
        self.inner.stats().sum_weights()
    }

    #[getter]
    fn resident_bytes(&self) -> usize {
        self.inner.stats().resident_bytes()
    }

    #[getter]
    fn uses_quadratic_normalization(&self) -> bool {
        self.inner.uses_quadratic_normalization()
    }

    #[getter]
    fn normalization_strategy(&self) -> Option<&'static str> {
        self.inner
            .normalization()
            .map(|normalization| match normalization.strategy() {
                NormalizationStrategy::Hermitian => "hermitian",
                NormalizationStrategy::LinearStatistics => "linear_statistics",
                NormalizationStrategy::Hybrid => "hybrid",
                NormalizationStrategy::General => "general",
            })
    }

    #[getter]
    fn normalization_basis_count(&self) -> Option<usize> {
        self.inner
            .normalization()
            .map(|normalization| normalization.compiler().basis_count())
    }

    #[getter]
    fn normalization_coherent_group_count(&self) -> Option<usize> {
        self.inner
            .normalization()
            .map(|normalization| normalization.compiler().coherent_group_count())
    }

    #[getter]
    fn normalization_has_residual(&self) -> Option<bool> {
        self.inner
            .normalization()
            .map(|normalization| normalization.compiler().has_residual())
    }

    #[getter]
    fn normalization_retained_bytes(&self) -> Option<usize> {
        self.inner
            .normalization()
            .map(|normalization| normalization.retained_bytes())
    }

    #[getter]
    fn normalization_preparation_passes(&self) -> Option<usize> {
        self.inner
            .normalization()
            .map(|normalization| normalization.preparation_passes())
    }

    #[getter]
    fn normalization_cache_reused(&self) -> Option<bool> {
        self.inner
            .normalization()
            .map(|normalization| normalization.cache_hit())
    }

    #[getter]
    fn normalization_tag_reused_parent(&self) -> Option<bool> {
        self.inner
            .normalization()
            .map(|normalization| normalization.tag_projection_reused_parent())
    }

    #[getter]
    fn normalization_fallback_reason(&self) -> Option<String> {
        self.inner.normalization().and_then(|normalization| {
            normalization
                .compiler()
                .fallback_reason()
                .map(|reason| format!("{reason:?}"))
        })
    }

    #[getter]
    fn source_traversals(&self) -> u64 {
        self.inner.source_traversals()
    }
}

#[pyclass(
    name = "LikelihoodDiagnostics",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Snapshot of likelihood preparation, memory planning, and evaluation counts.
pub struct PyLikelihoodDiagnostics {
    inner: LikelihoodDiagnostics,
}

#[pymethods]
impl PyLikelihoodDiagnostics {
    #[getter]
    fn datasets(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDatasetDiagnostics>>> {
        self.inner
            .datasets()
            .iter()
            .cloned()
            .map(|inner| Py::new(py, PyDatasetDiagnostics { inner }))
            .collect()
    }

    #[getter]
    fn objective_evaluations(&self) -> u64 {
        self.inner.objective_evaluations()
    }

    #[getter]
    fn gradient_evaluations(&self) -> u64 {
        self.inner.gradient_evaluations()
    }

    #[getter]
    fn memory_decisions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let decisions = self
            .inner
            .memory_decisions()
            .iter()
            .map(|decision| super::runtime::memory_decision_dict(py, decision))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, decisions)
    }
}

#[pyclass(name = "NLL", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// An unbinned negative log-likelihood term.
///
/// Parameters
/// ----------
/// model : Model
///     Positive event-intensity model.
/// data : Dataset
///     Observed events contributing the logarithmic data term.
/// accepted_mc : Dataset
///     Accepted Monte Carlo events used to normalize the intensity.
/// name : str, default='nll'
///     Unique term name used by projections.
pub struct PyNll {
    inner: NllTerm,
}

#[pyclass(name = "ExtendedNLL", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// An extended unbinned negative log-likelihood term.
///
/// Unlike :class:`NLL`, this term retains the predicted-yield contribution.
///
/// Parameters
/// ----------
/// model : Model
///     Positive event-intensity model.
/// data : Dataset
///     Observed events.
/// accepted_mc : Dataset
///     Accepted Monte Carlo normalization sample.
/// name : str, default='extended_nll'
///     Unique term name used by projections.
pub struct PyExtendedNll {
    inner: ExtendedNllTerm,
}

#[pymethods]
impl PyExtendedNll {
    /// Construct an extended likelihood term.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If model preparation or dataset schema validation fails.
    #[new]
    #[pyo3(signature = (model, *, data, accepted_mc, name="extended_nll"))]
    fn new(
        model: &PyModel,
        data: &PyDataset,
        accepted_mc: &PyDataset,
        name: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ExtendedNllTerm::new(name, &model.inner, &data.inner, &accepted_mc.inner)
                .map_err(to_py_err)?,
        })
    }
}

#[pyclass(name = "RidgePenalty", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Quadratic regularization over selected parameters.
///
/// Parameters
/// ----------
/// parameters : sequence of str
///     Parameter names to penalize.
/// lambda_ : float
///     Non-negative penalty strength.
/// name : str, default='ridge'
///     Term name.
pub struct PyRidgePenalty {
    inner: RidgePenalty,
}

#[pymethods]
impl PyRidgePenalty {
    /// Construct a ridge penalty.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the name list or penalty strength is invalid.
    #[new]
    #[pyo3(signature = (parameters, *, lambda_, name="ridge"))]
    fn new(parameters: Vec<String>, lambda_: f64, name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RidgePenalty::new(name, parameters, lambda_).map_err(to_py_err)?,
        })
    }
}

#[pyclass(name = "LassoPenalty", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Absolute-value regularization over selected parameters.
///
/// Parameters
/// ----------
/// parameters : sequence of str
///     Parameter names to penalize.
/// lambda_ : float
///     Non-negative penalty strength.
/// name : str, default='lasso'
///     Term name.
pub struct PyLassoPenalty {
    inner: LassoPenalty,
}

#[pymethods]
impl PyLassoPenalty {
    /// Construct a lasso penalty.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the name list or penalty strength is invalid.
    #[new]
    #[pyo3(signature = (parameters, *, lambda_, name="lasso"))]
    fn new(parameters: Vec<String>, lambda_: f64, name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: LassoPenalty::new(name, parameters, lambda_).map_err(to_py_err)?,
        })
    }
}

#[pymethods]
impl PyNll {
    /// Construct an unbinned likelihood term.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If model preparation or dataset schema validation fails.
    #[new]
    #[pyo3(signature = (model, *, data, accepted_mc, name="nll"))]
    fn new(
        model: &PyModel,
        data: &PyDataset,
        accepted_mc: &PyDataset,
        name: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: NllTerm::new(name, &model.inner, &data.inner, &accepted_mc.inner)
                .map_err(to_py_err)?,
        })
    }

    fn __repr__(&self) -> String {
        "NLL(...)".to_owned()
    }
}

#[pyclass(name = "Likelihood", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A prepared sum of likelihood and regularization terms.
///
/// Parameters
/// ----------
/// terms : sequence of NLL, ExtendedNLL, RidgePenalty, or LassoPenalty
///     Terms to combine. Parameter definitions are reconciled by name.
/// execution : Execution, optional
///     Runtime used to prepare model-backed terms.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> # Given model, data, and accepted_mc datasets:
/// >>> # term = ld.NLL(model, data=data, accepted_mc=accepted_mc, name="signal")
/// >>> # likelihood = ld.Likelihood(\[term\])
/// >>> # value = likelihood.nll(likelihood.default_parameters)
pub struct PyLikelihood {
    pub(crate) inner: Arc<Likelihood>,
}

#[pyclass(
    name = "LikelihoodProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
/// Event weights for a tagged projection of one likelihood term.
///
/// Projection objects are created by :meth:`Likelihood.projection`.
pub struct PyLikelihoodProjection {
    inner: LikelihoodProjection,
    likelihood: Arc<Likelihood>,
}

#[pymethods]
impl PyLikelihoodProjection {
    #[getter]
    /// str: Source likelihood term name.
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the projected accepted-Monte-Carlo integral.
    fn accepted_integral(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner.accepted_integral(&values).map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the projected generated-Monte-Carlo integral.
    fn generated_integral(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner.generated_integral(&values).map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the projected accepted-to-generated integral ratio.
    fn acceptance(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner.acceptance(&values).map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the full-model accepted-Monte-Carlo integral.
    fn full_accepted_integral(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .full_accepted_integral(&values)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the projected, acceptance-corrected event yield.
    fn acceptance_corrected_yield(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .acceptance_corrected_yield(&values)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        luminosity
    ))]
    /// Evaluate the observed-yield-normalized projected cross section.
    fn observed_cross_section(
        &self,
        parameters: &Bound<'_, PyAny>,
        luminosity: f64,
    ) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .observed_cross_section(&values, luminosity)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        luminosity
    ))]
    /// Evaluate the fitted projected cross section for an absolute-rate term.
    fn fitted_cross_section(
        &self,
        parameters: &Bound<'_, PyAny>,
        luminosity: f64,
    ) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .fitted_cross_section(&values, luminosity)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        luminosity
    ))]
    /// Alias for :meth:`observed_cross_section`.
    fn cross_section(&self, parameters: &Bound<'_, PyAny>, luminosity: f64) -> PyResult<f64> {
        self.observed_cross_section(parameters, luminosity)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"
    ) -> "Sequence[float]")]
    /// Evaluate projected intensities over generated Monte Carlo.
    fn intensities<'py>(
        &self,
        py: Python<'py>,
        parameters: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = free_values(&self.likelihood, parameters)?;
        let intensities = self.inner.intensities(&values).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, intensities))
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        acceptance_corrected=true
    ) -> "Sequence[float]")]
    /// Evaluate generated-event projection weights.
    ///
    /// Parameters
    /// ----------
    /// parameters : sequence of float or dict
    ///     Free values in likelihood order, or a partial mapping by name.
    /// acceptance_corrected : bool, default=True
    ///     Include the accepted-Monte-Carlo normalization correction.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     One weight per generated event.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``parameters`` is neither a numeric sequence nor a mapping.
    /// LadduError
    ///     If parameter validation or projection evaluation fails.
    fn weights<'py>(
        &self,
        py: Python<'py>,
        parameters: &Bound<'_, PyAny>,
        acceptance_corrected: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = free_values(&self.likelihood, parameters)?;
        let weights = self
            .inner
            .weights(&values, acceptance_corrected)
            .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, weights))
    }
}

#[pyclass(
    name = "CrossSectionIntegrals",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
/// Accepted and generated Monte Carlo integrals for cross-section extraction.
///
/// Objects are created by :meth:`Likelihood.cross_section_integrals`. Passing
/// tags narrows the numerator while preserving the full-model normalization.
pub struct PyCrossSectionIntegrals {
    inner: CrossSectionIntegrals,
    likelihood: Arc<Likelihood>,
}

#[pymethods]
impl PyCrossSectionIntegrals {
    #[getter]
    /// str: Source likelihood term name.
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[getter]
    /// float: Total observed event weight used by :meth:`cross_section`.
    fn data_weight_sum(&self) -> f64 {
        self.inner.data_weight_sum()
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the selected accepted-Monte-Carlo intensity integral.
    fn accepted_integral(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner.accepted_integral(&values).map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the selected generated-Monte-Carlo intensity integral.
    fn generated_integral(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner.generated_integral(&values).map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the selected accepted-to-generated integral ratio.
    fn acceptance(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner.acceptance(&values).map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the full-model accepted-Monte-Carlo integral.
    fn full_accepted_integral(&self, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .full_accepted_integral(&values)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        accepted_yield
    ))]
    /// Correct an accepted selected-component yield for finite acceptance.
    fn acceptance_corrected_yield(
        &self,
        parameters: &Bound<'_, PyAny>,
        accepted_yield: f64,
    ) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .acceptance_corrected_yield(&values, accepted_yield)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        luminosity
    ))]
    /// Evaluate the observed-yield-normalized selected cross section.
    fn observed_cross_section(
        &self,
        parameters: &Bound<'_, PyAny>,
        luminosity: f64,
    ) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .observed_cross_section(&values, luminosity)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        luminosity
    ))]
    /// Evaluate the fitted selected cross section for an absolute-rate term.
    fn fitted_cross_section(
        &self,
        parameters: &Bound<'_, PyAny>,
        luminosity: f64,
    ) -> PyResult<f64> {
        let values = free_values(&self.likelihood, parameters)?;
        self.inner
            .fitted_cross_section(&values, luminosity)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        *,
        luminosity
    ))]
    /// Alias for :meth:`observed_cross_section`.
    fn cross_section(&self, parameters: &Bound<'_, PyAny>, luminosity: f64) -> PyResult<f64> {
        self.observed_cross_section(parameters, luminosity)
    }
}

/// Resolve Python values into a likelihood's free-parameter order.
///
/// A sequence is used directly. A mapping starts from the default values and
/// replaces entries whose names are present.
pub fn free_values(likelihood: &Likelihood, values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(values) = float_vec(values) {
        return Ok(values);
    }
    if let Ok(mapping) = values.cast::<PyDict>() {
        let mut out = likelihood.default_params();
        for (key, _) in mapping.iter() {
            let name = key
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("parameter mappings must use string keys"))?;
            if !likelihood
                .params()
                .free_params()
                .iter()
                .any(|id| likelihood.params().name(*id) == Ok(name.as_str()))
            {
                return Err(PyTypeError::new_err(format!(
                    "unknown free parameter `{name}`"
                )));
            }
        }
        for (index, id) in likelihood.params().free_params().iter().enumerate() {
            let name = likelihood.params().name(*id).map_err(to_py_err)?;
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

#[pymethods]
impl PyLikelihood {
    /// Prepare a combined likelihood.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an element of ``terms`` is not a supported likelihood term.
    /// LadduError
    ///     If parameters conflict or a model-backed term cannot be prepared.
    #[new]
    #[pyo3(signature = (
        terms: "Sequence[NLL | ExtendedNLL | RidgePenalty | LassoPenalty]",
        *,
        execution=None
    ))]
    fn new(terms: Vec<Bound<'_, PyAny>>, execution: Option<&PyExecution>) -> PyResult<Self> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        let terms = terms
            .into_iter()
            .map(|term| -> PyResult<Box<dyn LikelihoodTerm>> {
                if let Ok(term) = term.extract::<PyRef<'_, PyNll>>() {
                    return Ok(Box::new(term.inner.clone()));
                }
                if let Ok(term) = term.extract::<PyRef<'_, PyExtendedNll>>() {
                    return Ok(Box::new(term.inner.clone()));
                }
                if let Ok(term) = term.extract::<PyRef<'_, PyRidgePenalty>>() {
                    return Ok(Box::new(term.inner.clone()));
                }
                if let Ok(term) = term.extract::<PyRef<'_, PyLassoPenalty>>() {
                    return Ok(Box::new(term.inner.clone()));
                }
                Err(PyTypeError::new_err(
                    "likelihood terms must be NLL, ExtendedNLL, RidgePenalty, or LassoPenalty",
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: Arc::new(
                Likelihood::with_execution_boxed(terms, &execution.inner).map_err(to_py_err)?,
            ),
        })
    }

    fn __repr__(&self) -> String {
        format!("Likelihood(parameters={:?})", self.parameter_names())
    }

    /// Return preparation strategy, memory decisions, and objective evaluation counts.
    fn diagnostics(&self) -> PyLikelihoodDiagnostics {
        PyLikelihoodDiagnostics {
            inner: self.inner.diagnostics(),
        }
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
    /// list of float: Default free-parameter values.
    fn default_parameters(&self) -> Vec<f64> {
        self.inner.default_params()
    }

    #[pyo3(signature = (*, seed=0))]
    /// Sample reproducible values from parameter initialization ranges.
    ///
    /// Parameters
    /// ----------
    /// seed : int, default=0
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// list of float
    ///     Values in :attr:`parameter_names` order.
    fn sample_parameters(&self, seed: u64) -> Vec<f64> {
        self.inner.sample_initial(seed)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the total negative log-likelihood.
    ///
    /// Parameters
    /// ----------
    /// parameters : sequence of float or dict
    ///     Free values or a partial mapping by name.
    ///
    /// Returns
    /// -------
    /// float
    ///     Sum of all likelihood and penalty terms.
    fn value(&self, py: Python<'_>, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        let values = free_values(&self.inner, parameters)?;
        let likelihood = Arc::clone(&self.inner);
        py.detach(move || likelihood.nll(values.as_slice()))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"))]
    /// Evaluate the total negative log-likelihood.
    ///
    /// This is an alias for :meth:`value`.
    fn nll(&self, py: Python<'_>, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.value(py, parameters)
    }

    #[pyo3(signature = (
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]"
    ) -> "tuple[float, Sequence[float]]")]
    /// Evaluate the negative log-likelihood and its gradient.
    ///
    /// Parameters
    /// ----------
    /// parameters : sequence of float or dict
    ///     Free values or a partial mapping by name.
    ///
    /// Returns
    /// -------
    /// value : float
    ///     Total negative log-likelihood.
    /// gradient : numpy.ndarray
    ///     Derivatives in :attr:`parameter_names` order.
    fn value_and_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let values = free_values(&self.inner, parameters)?;
        let likelihood = Arc::clone(&self.inner);
        let evaluation = py
            .detach(move || likelihood.nll_with_gradient(values.as_slice()))
            .map_err(to_py_err)?;
        let (value, gradient) = evaluation.into_parts();
        Ok((value, PyArray1::from_vec(py, gradient)))
    }

    #[pyo3(signature = (
        term_name,
        *,
        generated_mc,
        luminosity,
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        ensemble=None
    ))]
    /// Prepare the preferred total, tagged, and differential cross-section analysis.
    fn cross_section(
        &self,
        term_name: &str,
        generated_mc: &PyDataset,
        luminosity: f64,
        parameters: &Bound<'_, PyAny>,
        ensemble: Option<&PyEnsemble>,
    ) -> PyResult<PyCrossSection> {
        let parameters = free_values(&self.inner, parameters)?;
        let inner = match ensemble {
            Some(ensemble) => self.inner.cross_section_with_ensemble(
                term_name,
                generated_mc.inner.clone(),
                luminosity,
                parameters,
                ensemble.inner.clone(),
            ),
            None => self.inner.cross_section(
                term_name,
                generated_mc.inner.clone(),
                luminosity,
                parameters,
            ),
        }
        .map_err(to_py_err)?;
        Ok(PyCrossSection { inner })
    }

    #[pyo3(signature = (term_name, *, generated_mc, tags=None))]
    /// Prepare cross-section integrals for a model-backed term.
    ///
    /// Parameters
    /// ----------
    /// term_name : str
    ///     Name assigned when constructing the likelihood term.
    /// generated_mc : Dataset
    ///     Generated Monte Carlo sample before detector acceptance.
    /// tags : sequence of str, optional
    ///     Model contribution tags to retain. By default the full model is used.
    ///
    /// Returns
    /// -------
    /// CrossSectionIntegrals
    ///     Specialized integral and cross-section evaluator.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the term is unknown, cannot be narrowed, or schemas mismatch.
    fn cross_section_integrals(
        &self,
        term_name: &str,
        generated_mc: &PyDataset,
        tags: Option<Vec<String>>,
    ) -> PyResult<PyCrossSectionIntegrals> {
        let inner = match tags {
            Some(tags) => self.inner.cross_section_integrals_with_tags(
                term_name,
                &generated_mc.inner,
                tags.iter().map(String::as_str),
            ),
            None => self
                .inner
                .cross_section_integrals(term_name, &generated_mc.inner),
        }
        .map_err(to_py_err)?;
        Ok(PyCrossSectionIntegrals {
            inner,
            likelihood: Arc::clone(&self.inner),
        })
    }

    /// Create a tagged projection for a model-backed term.
    ///
    /// Parameters
    /// ----------
    /// term_name : str
    ///     Name assigned when constructing the likelihood term.
    /// generated_mc : Dataset
    ///     Generated Monte Carlo events on which to evaluate weights.
    /// tags : sequence of str
    ///     Model contribution tags to retain.
    ///
    /// Returns
    /// -------
    /// LikelihoodProjection
    ///     Prepared projection evaluator.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the term is unknown, cannot be projected, or schemas mismatch.
    #[pyo3(signature = (term_name, *, generated_mc, tags))]
    fn projection(
        &self,
        term_name: &str,
        generated_mc: &PyDataset,
        tags: Vec<String>,
    ) -> PyResult<PyLikelihoodProjection> {
        let inner = self
            .inner
            .projection(
                term_name,
                &generated_mc.inner,
                tags.iter().map(String::as_str),
            )
            .map_err(to_py_err)?;
        Ok(PyLikelihoodProjection {
            inner,
            likelihood: Arc::clone(&self.inner),
        })
    }
}

#[pymodule]
/// Likelihood terms, regularization, evaluation, and model projections.
pub mod likelihood {
    #[pymodule_export]
    use super::{
        PyCrossSectionIntegrals as CrossSectionIntegrals, PyExtendedNll as ExtendedNLL,
        PyLassoPenalty as LassoPenalty, PyLikelihood as Likelihood,
        PyLikelihoodProjection as LikelihoodProjection, PyNll as NLL,
        PyRidgePenalty as RidgePenalty,
    };
}
