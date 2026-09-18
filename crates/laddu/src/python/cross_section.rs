//! Python wrappers for the Rust cross-section analysis API.

use std::collections::HashMap;

use laddu_likelihood::{
    Axis, BinnedEstimate, ComponentYieldProjection, CrossSection, DifferentialCrossSection,
    Ensemble, Estimate, IntegralRetentionPolicy, Projection, RateClosure, RateClosureStatus,
    ReferenceCorrectedYield, ReferenceCorrectionProvenance, TotalSet, Yield, YieldBinValidity,
    YieldHistogramView, YieldProjection,
};
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict},
};

use super::{
    data::PyDataset,
    error::to_py_err,
    expr::PyExpr,
    float_matrix, float_tensor3, float_vec,
    likelihood::{PyLikelihood, free_values},
};

#[pyclass(name = "Ensemble", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Named parameter draws used to propagate cross-section uncertainty.
pub struct PyEnsemble {
    pub(crate) inner: Ensemble,
}

#[pymethods]
impl PyEnsemble {
    #[staticmethod]
    #[pyo3(signature = (
        values: "Sequence[Sequence[float]] | numpy.typing.NDArray[numpy.float32 | numpy.float64]",
        *,
        parameter_names,
        source_id=None
    ))]
    /// Build an ensemble from a two-dimensional array of parameter draws.
    fn from_arrays(
        values: &Bound<'_, PyAny>,
        parameter_names: Vec<String>,
        source_id: Option<u64>,
    ) -> PyResult<Self> {
        let draws = float_matrix(values)?;
        let inner = match source_id {
            Some(source_id) => Ensemble::with_source_id(parameter_names, draws, source_id),
            None => Ensemble::new(parameter_names, draws),
        }
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (summary: "MCMCSummary", *, discard, thin=1))]
    /// Adapt a Ganesh MCMC summary after explicit burn-in removal and thinning.
    fn from_mcmc(summary: &Bound<'_, PyAny>, discard: usize, thin: usize) -> PyResult<Self> {
        let parameter_names = summary
            .getattr("parameter_names")?
            .extract::<Option<Vec<String>>>()?
            .ok_or_else(|| PyValueError::new_err("MCMC summary has no parameter names"))?;
        let chain = float_tensor3(&summary.getattr("chain")?)?;
        Ok(Self {
            inner: Ensemble::from_chain(parameter_names, &chain, discard, thin)
                .map_err(to_py_err)?,
        })
    }

    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().to_vec()
    }

    #[getter]
    fn source_id(&self) -> u64 {
        self.inner.source_id()
    }

    #[getter]
    fn draws<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(PyArray2::from_vec2(py, self.inner.draws())?)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(name = "Estimate", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A central scalar estimate with optional propagated draws.
pub struct PyEstimate {
    inner: Estimate,
}

impl From<Estimate> for PyEstimate {
    fn from(inner: Estimate) -> Self {
        Self { inner }
    }
}

#[pyclass(name = "TotalSet", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Full-model and named tagged scalar totals from one shared request.
pub struct PyTotalSet {
    inner: TotalSet,
}

#[pymethods]
impl PyTotalSet {
    #[getter]
    fn full(&self) -> PyEstimate {
        self.inner.full().clone().into()
    }

    #[getter]
    fn components(&self) -> HashMap<String, PyEstimate> {
        self.inner
            .components()
            .iter()
            .map(|(name, estimate)| (name.clone(), estimate.clone().into()))
            .collect()
    }

    fn __getitem__(&self, name: &str) -> PyResult<PyEstimate> {
        self.inner
            .get(name)
            .cloned()
            .map(Into::into)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_owned()))
    }
}

#[pymethods]
impl PyEstimate {
    #[new]
    #[pyo3(signature = (
        central,
        *,
        draws: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | None" = None,
        source_id=None
    ))]
    fn new(
        central: f64,
        draws: Option<&Bound<'_, PyAny>>,
        source_id: Option<u64>,
    ) -> PyResult<Self> {
        let draws = draws.map(float_vec).transpose()?.unwrap_or_default();
        let inner = match source_id {
            Some(source_id) => Estimate::with_source_id(central, draws, Some(source_id)),
            None => Estimate::new(central, draws),
        }
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn central(&self) -> f64 {
        self.inner.value()
    }

    #[getter]
    fn draws<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.draws())
    }

    #[getter]
    fn source_id(&self) -> Option<u64> {
        self.inner.source_id()
    }

    fn mean(&self) -> PyResult<f64> {
        self.inner.mean().map_err(to_py_err)
    }

    fn median(&self) -> PyResult<f64> {
        self.inner.median().map_err(to_py_err)
    }

    fn std(&self) -> PyResult<f64> {
        self.inner.std().map_err(to_py_err)
    }

    fn quantile(&self, probability: f64) -> PyResult<f64> {
        self.inner.quantile(probability).map_err(to_py_err)
    }

    #[pyo3(signature = (level=0.68))]
    fn interval(&self, level: f64) -> PyResult<(f64, f64)> {
        self.inner.interval(level).map_err(to_py_err)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        estimate_binary(self, other, |left, right| left + right)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        estimate_binary(self, other, |left, right| left - right)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        estimate_binary(self, other, |left, right| left * right)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        estimate_binary(self, other, |left, right| left / right)
    }
}

#[pyclass(name = "RateClosure", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Inspectable rate-closure diagnostic for an absolute-rate yield context.
pub struct PyRateClosure {
    #[pyo3(get)]
    selected_yield: PyEstimate,
    #[pyo3(get)]
    accepted_fitted_yield: Option<PyEstimate>,
    #[pyo3(get)]
    generated_fitted_yield: Option<PyEstimate>,
    #[pyo3(get)]
    corrected_observed_yield: Option<PyEstimate>,
    #[pyo3(get)]
    accepted_residual: Option<f64>,
    #[pyo3(get)]
    corrected_residual: Option<f64>,
    #[pyo3(get)]
    accepted_absolute_residual: Option<f64>,
    #[pyo3(get)]
    corrected_absolute_residual: Option<f64>,
    #[pyo3(get)]
    accepted_relative_residual: Option<f64>,
    #[pyo3(get)]
    corrected_relative_residual: Option<f64>,
    #[pyo3(get)]
    accepted_residual_draws: Vec<f64>,
    #[pyo3(get)]
    corrected_residual_draws: Vec<f64>,
    #[pyo3(get)]
    absolute_tolerance: f64,
    #[pyo3(get)]
    relative_tolerance: f64,
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    reason: Option<String>,
}

impl From<RateClosure> for PyRateClosure {
    fn from(inner: RateClosure) -> Self {
        let status = match inner.status() {
            RateClosureStatus::Closed => "closed",
            RateClosureStatus::Failed => "failed",
            RateClosureStatus::NotApplicable => "not_applicable",
        }
        .to_owned();
        Self {
            selected_yield: inner.selected_yield().clone().into(),
            accepted_fitted_yield: inner.accepted_fitted_yield().cloned().map(Into::into),
            generated_fitted_yield: inner.generated_fitted_yield().cloned().map(Into::into),
            corrected_observed_yield: inner.corrected_observed_yield().cloned().map(Into::into),
            accepted_residual: inner.accepted_residual(),
            corrected_residual: inner.corrected_residual(),
            accepted_absolute_residual: inner.accepted_absolute_residual(),
            corrected_absolute_residual: inner.corrected_absolute_residual(),
            accepted_relative_residual: inner.accepted_relative_residual(),
            corrected_relative_residual: inner.corrected_relative_residual(),
            accepted_residual_draws: inner.accepted_residual_draws().to_vec(),
            corrected_residual_draws: inner.corrected_residual_draws().to_vec(),
            absolute_tolerance: inner.absolute_tolerance(),
            relative_tolerance: inner.relative_tolerance(),
            status,
            reason: inner.reason().map(str::to_owned),
        }
    }
}

#[pymethods]
impl PyRateClosure {
    fn is_closed(&self) -> bool {
        self.status == "closed"
    }
}

#[pyclass(name = "Yield", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Immutable scalar selected, fitted, and acceptance-corrected yield context.
pub struct PyYield {
    pub(crate) inner: Yield,
    selected_yield: PyEstimate,
    fitted_acceptance: PyEstimate,
    corrected_observed_yield: PyEstimate,
    rate_closure: PyRateClosure,
    #[pyo3(get)]
    term_name: String,
    #[pyo3(get)]
    parameters: Vec<f64>,
    #[pyo3(get)]
    has_absolute_rate: bool,
}

#[pyclass(
    name = "ReferenceCorrectionProvenance",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Identifies every source used by a reference-acceptance correction.
pub struct PyReferenceCorrectionProvenance {
    inner: ReferenceCorrectionProvenance,
}

#[pymethods]
impl PyReferenceCorrectionProvenance {
    #[getter]
    fn reference_term_name(&self) -> &str {
        self.inner.reference_term_name()
    }

    #[getter]
    fn reference_model_digest(&self) -> u64 {
        self.inner.reference_model_digest()
    }

    #[getter]
    fn accepted_dataset_identity(&self) -> u64 {
        self.inner.accepted_dataset_identity()
    }

    #[getter]
    fn generated_dataset_identity(&self) -> u64 {
        self.inner.generated_dataset_identity()
    }

    #[getter]
    fn reference_parameters(&self) -> Vec<f64> {
        self.inner.reference_parameters().to_vec()
    }

    #[getter]
    fn reference_draw_parameters(&self) -> Vec<Vec<f64>> {
        self.inner.reference_draw_parameters().to_vec()
    }

    #[getter]
    fn reference_replica_accepted_dataset_identities(&self) -> Vec<u64> {
        self.inner
            .reference_replica_accepted_dataset_identities()
            .to_vec()
    }

    #[getter]
    fn selected_uncertainty_source(&self) -> Option<u64> {
        self.inner.selected_uncertainty_source()
    }

    #[getter]
    fn reference_uncertainty_source(&self) -> Option<u64> {
        self.inner.reference_uncertainty_source()
    }
}

#[pyclass(
    name = "ReferenceCorrectedYield",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// An observed yield corrected with an explicit reference acceptance.
pub struct PyReferenceCorrectedYield {
    inner: ReferenceCorrectedYield,
}

#[pymethods]
impl PyReferenceCorrectedYield {
    #[getter]
    fn value(&self) -> PyEstimate {
        self.inner.value().clone().into()
    }

    #[getter]
    fn acceptance(&self) -> PyEstimate {
        self.inner.acceptance().clone().into()
    }

    #[getter]
    fn provenance(&self) -> PyReferenceCorrectionProvenance {
        PyReferenceCorrectionProvenance {
            inner: self.inner.provenance().clone(),
        }
    }

    #[getter]
    fn reference_term_name(&self) -> &str {
        self.inner.reference_term_name()
    }

    #[getter]
    fn rate_closure_status(&self) -> &'static str {
        "not_applicable"
    }
}

impl From<Yield> for PyYield {
    fn from(inner: Yield) -> Self {
        Self {
            term_name: inner.term_name().to_owned(),
            parameters: inner.parameters().to_vec(),
            has_absolute_rate: inner.has_absolute_rate(),
            selected_yield: inner.selected_yield().clone().into(),
            fitted_acceptance: inner.fitted_acceptance().clone().into(),
            corrected_observed_yield: inner.corrected_observed_yield().clone().into(),
            rate_closure: inner.rate_closure().into(),
            inner,
        }
    }
}

#[pymethods]
impl PyYield {
    fn selected_yield(&self) -> PyEstimate {
        self.selected_yield.clone()
    }

    fn accepted_fitted_yield(&self) -> PyResult<PyEstimate> {
        Ok(self
            .inner
            .accepted_fitted_yield()
            .map_err(to_py_err)?
            .into())
    }

    fn generated_fitted_yield(&self) -> PyResult<PyEstimate> {
        Ok(self
            .inner
            .generated_fitted_yield()
            .map_err(to_py_err)?
            .into())
    }

    fn fitted_acceptance(&self) -> PyEstimate {
        self.fitted_acceptance.clone()
    }

    fn corrected_observed_yield(&self) -> PyEstimate {
        self.corrected_observed_yield.clone()
    }

    fn rate_closure(&self) -> PyRateClosure {
        self.rate_closure.clone()
    }

    #[pyo3(signature = (axes: "Axis | Sequence[Axis]", *, components=None))]
    /// Evaluate a central joint yield projection with optional model-only components.
    fn projection(
        &self,
        py: Python<'_>,
        axes: &Bound<'_, PyAny>,
        components: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<PyYieldProjection> {
        let axes = extract_axes(py, axes)?;
        Ok(self
            .inner
            .projection_with_components(&axes, &components.unwrap_or_default())
            .map_err(to_py_err)?
            .into())
    }

    #[pyo3(signature = (projections: "dict[str, Axis | Sequence[Axis]]", *, components=None))]
    /// Evaluate named central yield projections with optional model-only components.
    fn projection_set<'py>(
        &self,
        py: Python<'py>,
        projections: &Bound<'_, PyAny>,
        components: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let projections = extract_projections(py, projections)?;
        let results = self
            .inner
            .projection_set_with_components(&projections, &components.unwrap_or_default())
            .map_err(to_py_err)?;
        let output = PyDict::new(py);
        for (name, result) in results.iter() {
            output.set_item(
                name,
                Py::new(
                    py,
                    PyYieldProjection {
                        inner: result.clone(),
                    },
                )?,
            )?;
        }
        Ok(output)
    }

    #[pyo3(signature = (
        reference_likelihood,
        reference_term_name,
        *,
        generated_mc,
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        ensemble=None
    ))]
    /// Correct the selected observed yield using an explicit reference intensity.
    fn reference_corrected(
        &self,
        reference_likelihood: &PyLikelihood,
        reference_term_name: &str,
        generated_mc: &PyDataset,
        parameters: &Bound<'_, PyAny>,
        ensemble: Option<&PyEnsemble>,
    ) -> PyResult<PyReferenceCorrectedYield> {
        let parameters = free_values(&reference_likelihood.inner, parameters)?;
        let inner = self
            .inner
            .reference_corrected(
                std::sync::Arc::clone(&reference_likelihood.inner),
                reference_term_name,
                generated_mc.inner.clone(),
                parameters,
                ensemble.map(|value| value.inner.clone()),
            )
            .map_err(to_py_err)?;
        Ok(PyReferenceCorrectedYield { inner })
    }
}

#[pyclass(
    name = "YieldHistogramView",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Central row-major histogram view over a yield projection's axes.
pub struct PyYieldHistogramView {
    inner: YieldHistogramView,
}

#[pymethods]
impl PyYieldHistogramView {
    #[getter]
    fn axes(&self) -> Vec<Vec<f64>> {
        self.inner.axes().to_vec()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }
}

#[pyclass(
    name = "YieldProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Coherent central selected, fitted, and corrected yields on shared axes.
pub struct PyYieldProjection {
    inner: YieldProjection,
}

#[pyclass(
    name = "ComponentYieldProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Coherent model-only fitted yields for one named tag selection.
pub struct PyComponentYieldProjection {
    inner: ComponentYieldProjection,
}

#[pymethods]
impl PyComponentYieldProjection {
    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags().to_vec()
    }
    #[getter]
    fn axes(&self) -> Vec<Vec<f64>> {
        self.inner.axes().to_vec()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    #[getter]
    fn accepted(&self) -> Vec<f64> {
        self.inner.accepted().to_vec()
    }
    #[getter]
    fn generated(&self) -> Vec<f64> {
        self.inner.generated().to_vec()
    }
    #[getter]
    fn validity(&self) -> Vec<&'static str> {
        self.inner
            .validity()
            .iter()
            .copied()
            .map(validity_name)
            .collect()
    }
    fn accepted_histogram(&self) -> PyYieldHistogramView {
        PyYieldHistogramView {
            inner: self.inner.accepted_histogram(),
        }
    }
    fn generated_histogram(&self) -> PyYieldHistogramView {
        PyYieldHistogramView {
            inner: self.inner.generated_histogram(),
        }
    }
}

impl From<YieldProjection> for PyYieldProjection {
    fn from(inner: YieldProjection) -> Self {
        Self { inner }
    }
}

fn validity_name(validity: YieldBinValidity) -> &'static str {
    match validity {
        YieldBinValidity::Valid => "valid",
        YieldBinValidity::MissingGeneratedSupport => "missing_generated_support",
        YieldBinValidity::MissingAcceptedSupport => "missing_accepted_support",
        YieldBinValidity::NonPositiveAcceptedSupport => "nonpositive_accepted_support",
        YieldBinValidity::NonPositiveGeneratedSupport => "nonpositive_generated_support",
        YieldBinValidity::InvalidExposure => "invalid_exposure",
        YieldBinValidity::NonFiniteEvaluation => "nonfinite_evaluation",
    }
}

#[pymethods]
impl PyYieldProjection {
    #[getter]
    fn components(&self) -> HashMap<String, PyComponentYieldProjection> {
        self.inner
            .components()
            .iter()
            .map(|(name, inner)| {
                (
                    name.clone(),
                    PyComponentYieldProjection {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }
    #[getter]
    fn axes(&self) -> Vec<Vec<f64>> {
        self.inner.axes().to_vec()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    #[getter]
    fn selected(&self) -> Vec<f64> {
        self.inner.selected().to_vec()
    }
    #[getter]
    fn accepted(&self) -> Vec<f64> {
        self.inner.accepted().to_vec()
    }
    #[getter]
    fn generated(&self) -> Vec<f64> {
        self.inner.generated().to_vec()
    }
    #[getter]
    fn acceptance(&self) -> Vec<f64> {
        self.inner.acceptance().to_vec()
    }
    #[getter]
    fn corrected(&self) -> Vec<f64> {
        self.inner.corrected().to_vec()
    }
    #[getter]
    fn validity(&self) -> Vec<&'static str> {
        self.inner
            .validity()
            .iter()
            .copied()
            .map(validity_name)
            .collect()
    }
    #[getter]
    fn has_absolute_rate(&self) -> bool {
        self.inner.has_absolute_rate()
    }
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let value = self.inner.diagnostics();
        let out = PyDict::new(py);
        out.set_item("selected_nonfinite", value.selected_nonfinite)?;
        out.set_item("selected_out_of_range", value.selected_out_of_range)?;
        out.set_item("accepted_nonfinite", value.accepted_nonfinite)?;
        out.set_item("accepted_out_of_range", value.accepted_out_of_range)?;
        out.set_item("generated_nonfinite", value.generated_nonfinite)?;
        out.set_item("generated_out_of_range", value.generated_out_of_range)?;
        Ok(out)
    }
    fn selected_histogram(&self) -> PyYieldHistogramView {
        PyYieldHistogramView {
            inner: self.inner.selected_histogram(),
        }
    }
    fn accepted_histogram(&self) -> PyYieldHistogramView {
        PyYieldHistogramView {
            inner: self.inner.accepted_histogram(),
        }
    }
    fn generated_histogram(&self) -> PyYieldHistogramView {
        PyYieldHistogramView {
            inner: self.inner.generated_histogram(),
        }
    }
    fn corrected_histogram(&self) -> PyYieldHistogramView {
        PyYieldHistogramView {
            inner: self.inner.corrected_histogram(),
        }
    }
}

fn estimate_binary(
    left: &PyEstimate,
    right: &Bound<'_, PyAny>,
    op: impl Fn(&Estimate, &Estimate) -> Estimate,
) -> PyResult<PyEstimate> {
    if let Ok(right) = right.extract::<PyRef<'_, PyEstimate>>() {
        return Ok(op(&left.inner, &right.inner).into());
    }
    if let Ok(right) = right.extract::<f64>() {
        let right = Estimate::central(right).map_err(to_py_err)?;
        return Ok(op(&left.inner, &right).into());
    }
    Err(PyTypeError::new_err("expected Estimate or float"))
}

#[pyclass(name = "Axis", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// An expression and explicit edges defining a differential axis.
pub struct PyAxis {
    inner: Axis,
}

#[pymethods]
impl PyAxis {
    #[new]
    #[pyo3(signature = (
        expr,
        *,
        edges: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64]"
    ))]
    fn new(expr: &PyExpr, edges: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: Axis::new(expr.inner.clone(), float_vec(edges)?).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn edges(&self) -> Vec<f64> {
        self.inner.edges().to_vec()
    }
}

#[pyclass(name = "BinnedEstimate", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Central bin values with optional propagated ensemble draws.
pub struct PyBinnedEstimate {
    inner: BinnedEstimate,
}

impl From<BinnedEstimate> for PyBinnedEstimate {
    fn from(inner: BinnedEstimate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBinnedEstimate {
    #[getter]
    fn central(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        self.inner.draws().to_vec()
    }

    #[pyo3(signature = (level=0.68))]
    fn interval(&self, level: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner.interval(level).map_err(to_py_err)
    }

    fn covariance(&self) -> PyResult<Vec<Vec<f64>>> {
        self.inner.covariance().map_err(to_py_err)
    }
}

#[pyclass(name = "DifferentialCrossSection", module = "laddu", frozen)]
/// Data, model, and component differential cross sections.
pub struct PyDifferentialCrossSection {
    inner: DifferentialCrossSection,
}

impl From<DifferentialCrossSection> for PyDifferentialCrossSection {
    fn from(inner: DifferentialCrossSection) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDifferentialCrossSection {
    #[getter]
    fn edges(&self) -> PyResult<Vec<f64>> {
        if self.inner.axes().len() != 1 {
            return Err(PyValueError::new_err(
                "edges is only defined for one-dimensional results; use axes",
            ));
        }
        Ok(self.inner.axes()[0].clone())
    }

    #[getter]
    fn axes(&self) -> Vec<Vec<f64>> {
        self.inner.axes().to_vec()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn data(&self) -> PyBinnedEstimate {
        self.inner.data().clone().into()
    }

    #[getter]
    fn model(&self) -> PyBinnedEstimate {
        self.inner.model().clone().into()
    }

    #[getter]
    fn components(&self) -> HashMap<String, PyBinnedEstimate> {
        self.inner
            .components()
            .iter()
            .map(|(name, estimate)| (name.clone(), estimate.clone().into()))
            .collect()
    }
}

#[pyclass(name = "CrossSection", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Prepared total, tagged, differential, and combined cross-section analysis.
pub struct PyCrossSection {
    pub(crate) inner: CrossSection,
}

#[pymethods]
impl PyCrossSection {
    /// Return integral-cache hit, miss, count, and retained-byte diagnostics.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let diagnostics = self.inner.diagnostics();
        let out = PyDict::new(py);
        out.set_item("cache_hits", diagnostics.cache_hits())?;
        out.set_item("cache_misses", diagnostics.cache_misses())?;
        out.set_item("cached_integrals", diagnostics.cached_integrals())?;
        out.set_item("prepared_bytes", diagnostics.prepared_bytes())?;
        out.set_item("cache_evictions", diagnostics.cache_evictions())?;
        out.set_item("reserved_bytes", diagnostics.reserved_bytes())?;
        out.set_item("high_water_bytes", diagnostics.high_water_bytes())?;
        out.set_item(
            "estimated_prepared_bytes",
            diagnostics.estimated_prepared_bytes(),
        )?;
        out.set_item("full_requests", diagnostics.full_requests())?;
        out.set_item("tagged_requests", diagnostics.tagged_requests())?;
        out.set_item("central_requests", diagnostics.central_requests())?;
        out.set_item(
            "shared_bootstrap_requests",
            diagnostics.shared_bootstrap_requests(),
        )?;
        out.set_item(
            "arbitrary_replica_requests",
            diagnostics.arbitrary_replica_requests(),
        )?;
        Ok(out)
    }

    #[pyo3(signature = (*, max_bytes))]
    /// Configure bounded integral retention, or disable retention with ``None``.
    fn configure_integral_retention(&self, max_bytes: Option<usize>) {
        self.inner.set_integral_retention(match max_bytes {
            Some(max_bytes) => IntegralRetentionPolicy::Bounded { max_bytes },
            None => IntegralRetentionPolicy::None,
        });
    }

    /// Drop all eligible retained integral preparations.
    ///
    /// The full-model baseline remains owned so the cross section stays usable.
    fn clear_integral_cache(&self) {
        self.inner.clear_integral_cache();
    }

    #[staticmethod]
    #[pyo3(signature = (
        members,
        *,
        factors: "Sequence[Estimate | float] | None" = None
    ))]
    fn combine(
        py: Python<'_>,
        members: Vec<Py<PyCrossSection>>,
        factors: Option<Vec<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let members = members
            .into_iter()
            .map(|member| member.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let inner = match factors {
            Some(factors) => {
                let factors = factors
                    .into_iter()
                    .map(|factor| {
                        let factor = factor.bind(py);
                        if let Ok(value) = factor.extract::<f64>() {
                            Estimate::central(value).map_err(to_py_err)
                        } else if let Ok(value) = factor.extract::<PyRef<'_, PyEstimate>>() {
                            Ok(value.inner.clone())
                        } else {
                            Err(PyTypeError::new_err(
                                "factors must be floats or Estimate objects",
                            ))
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                CrossSection::combine_with_factors(members, factors)
            }
            None => CrossSection::combine(members),
        }
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, tags=None))]
    /// Return the observed-yield-normalized cross section.
    fn observed_total(&self, tags: Option<Vec<String>>) -> PyResult<PyEstimate> {
        match tags {
            Some(tags) => self.inner.observed_total_with_tags(&tags),
            None => self.inner.observed_total(),
        }
        .map(Into::into)
        .map_err(to_py_err)
    }

    #[pyo3(signature = (*, tags=None))]
    /// Return the central observed-yield-normalized cross section without
    /// preparing or evaluating uncertainty draws.
    fn observed_total_central(&self, tags: Option<Vec<String>>) -> PyResult<PyEstimate> {
        match tags {
            Some(tags) => self.inner.observed_total_central_with_tags(&tags),
            None => self.inner.observed_total_central(),
        }
        .map(Into::into)
        .map_err(to_py_err)
    }

    #[pyo3(signature = (*, tags=None))]
    /// Return the fitted cross section from an absolute-rate likelihood term.
    fn fitted_total(&self, tags: Option<Vec<String>>) -> PyResult<PyEstimate> {
        match tags {
            Some(tags) => self.inner.fitted_total_with_tags(&tags),
            None => self.inner.fitted_total(),
        }
        .map(Into::into)
        .map_err(to_py_err)
    }

    #[pyo3(signature = (*, tags=None))]
    /// Alias for :meth:`observed_total`.
    fn total(&self, tags: Option<Vec<String>>) -> PyResult<PyEstimate> {
        self.observed_total(tags)
    }

    /// Return the full-model and named tagged totals from one shared request.
    fn total_set(&self, components: HashMap<String, Vec<String>>) -> PyResult<PyTotalSet> {
        self.inner
            .total_set(&components)
            .map(|inner| PyTotalSet { inner })
            .map_err(to_py_err)
    }

    #[pyo3(signature = (*, tags=None))]
    fn acceptance(&self, tags: Option<Vec<String>>) -> PyResult<PyEstimate> {
        match tags {
            Some(tags) => self.inner.acceptance_with_tags(&tags),
            None => self.inner.acceptance(),
        }
        .map(Into::into)
        .map_err(to_py_err)
    }

    #[pyo3(signature = (*, tags=None))]
    fn corrected_yield(&self, tags: Option<Vec<String>>) -> PyResult<PyEstimate> {
        match tags {
            Some(tags) => self.inner.corrected_yield_with_tags(&tags),
            None => self.inner.corrected_yield(),
        }
        .map(Into::into)
        .map_err(to_py_err)
    }

    #[pyo3(signature = (
        axes: "Axis | Sequence[Axis]",
        *,
        components: "dict[str, Sequence[str]] | None" = None
    ))]
    fn differential(
        &self,
        py: Python<'_>,
        axes: &Bound<'_, PyAny>,
        components: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<PyDifferentialCrossSection> {
        let axes = extract_axes(py, axes)?;
        self.inner
            .differential(&axes, &components.unwrap_or_default())
            .map(Into::into)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (
        projections: "dict[str, Axis | Sequence[Axis]]",
        *,
        components: "dict[str, Sequence[str]] | None" = None
    ))]
    /// Return independent named differential cross sections in request order.
    fn projection_set<'py>(
        &self,
        py: Python<'py>,
        projections: &Bound<'_, PyAny>,
        components: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let projections = extract_projections(py, projections)?;
        let results = self
            .inner
            .projection_set(&projections, &components.unwrap_or_default())
            .map_err(to_py_err)?;
        let output = PyDict::new(py);
        for (name, result) in results.iter() {
            output.set_item(
                name,
                Py::new(
                    py,
                    PyDifferentialCrossSection {
                        inner: result.clone(),
                    },
                )?,
            )?;
        }
        Ok(output)
    }
}

fn extract_axes(py: Python<'_>, axes: &Bound<'_, PyAny>) -> PyResult<Vec<Axis>> {
    if let Ok(axis) = axes.extract::<PyRef<'_, PyAxis>>() {
        return Ok(vec![axis.inner.clone()]);
    }
    let axes = axes
        .extract::<Vec<Py<PyAxis>>>()
        .map_err(|_| PyTypeError::new_err("axes must be an Axis or a sequence of Axis objects"))?;
    Ok(axes
        .into_iter()
        .map(|axis| axis.borrow(py).inner.clone())
        .collect())
}

fn extract_projections(
    py: Python<'_>,
    projections: &Bound<'_, PyAny>,
) -> PyResult<Vec<Projection>> {
    let items = projections.call_method0("items").map_err(|_| {
        PyTypeError::new_err("projections must be a mapping from names to Axis values")
    })?;
    items
        .try_iter()?
        .map(|item| {
            let (name, axes) = item?.extract::<(String, Py<PyAny>)>()?;
            let axes = extract_axes(py, axes.bind(py))?;
            Projection::new(name, axes).map_err(to_py_err)
        })
        .collect()
}
