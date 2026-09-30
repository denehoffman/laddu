//! Python wrappers for the Rust cross-section analysis API.

use std::collections::HashMap;

use laddu_likelihood::{
    AreaUnit, Axis, BinnedEstimate, BinnedEstimateUnit, CombinationMember,
    ComponentYieldProjection, CrossSection, CrossSectionProjection, Ensemble, Estimate, Luminosity,
    Projection, RateClosure, RateClosureStatus, Yield, YieldBinValidity, YieldHistogramView,
    YieldProjection,
};
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict},
};

use super::{error::to_py_err, expr::PyExpr, float_matrix, float_tensor3, float_vec};

#[pyclass(
    name = "AreaUnit",
    module = "laddu",
    frozen,
    eq,
    eq_int,
    from_py_object,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Area prefix used for cross sections and reciprocal luminosity.
pub enum PyAreaUnit {
    /// Barn.
    Barn,
    /// Millibarn.
    Millibarn,
    /// Microbarn.
    Microbarn,
    /// Nanobarn.
    Nanobarn,
    /// Picobarn.
    Picobarn,
    /// Femtobarn.
    Femtobarn,
}

impl From<PyAreaUnit> for AreaUnit {
    fn from(value: PyAreaUnit) -> Self {
        match value {
            PyAreaUnit::Barn => Self::Barn,
            PyAreaUnit::Millibarn => Self::Millibarn,
            PyAreaUnit::Microbarn => Self::Microbarn,
            PyAreaUnit::Nanobarn => Self::Nanobarn,
            PyAreaUnit::Picobarn => Self::Picobarn,
            PyAreaUnit::Femtobarn => Self::Femtobarn,
        }
    }
}

impl From<AreaUnit> for PyAreaUnit {
    fn from(value: AreaUnit) -> Self {
        match value {
            AreaUnit::Barn => Self::Barn,
            AreaUnit::Millibarn => Self::Millibarn,
            AreaUnit::Microbarn => Self::Microbarn,
            AreaUnit::Nanobarn => Self::Nanobarn,
            AreaUnit::Picobarn => Self::Picobarn,
            AreaUnit::Femtobarn => Self::Femtobarn,
        }
    }
}

#[pyclass(name = "Luminosity", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Positive integrated luminosity in a typed inverse area unit.
pub struct PyLuminosity {
    pub(crate) inner: Luminosity,
}

#[pymethods]
impl PyLuminosity {
    #[new]
    fn new(value: f64, unit: PyAreaUnit) -> PyResult<Self> {
        Luminosity::new(value, unit.into())
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value()
    }
    #[getter]
    fn unit(&self) -> PyAreaUnit {
        self.inner.unit().into()
    }
    #[getter]
    fn relative_uncertainty(&self) -> Option<f64> {
        self.inner.relative_uncertainty()
    }
    #[getter]
    fn source_id(&self) -> Option<u64> {
        self.inner.source_id()
    }
    fn to_unit(&self, unit: PyAreaUnit) -> PyResult<Self> {
        self.inner
            .to_unit(unit.into())
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }
    fn with_relative_uncertainty(&self, uncertainty: f64, source_id: u64) -> PyResult<Self> {
        self.inner
            .clone()
            .with_relative_uncertainty(uncertainty, source_id)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }
}

#[pyclass(name = "CrossSection", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Fitted generated-MC intensity divided by luminosity.
pub struct PyCrossSection {
    pub(crate) inner: CrossSection,
}

#[pymethods]
impl PyCrossSection {
    #[staticmethod]
    #[pyo3(signature = (
        members: "Sequence[CrossSection | tuple[CrossSection, float | Estimate]]",
        *,
        factor_covariance: "Sequence[Sequence[float]] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | None" = None
    ))]
    fn combine(
        members: &Bound<'_, PyAny>,
        factor_covariance: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut inputs = Vec::new();
        for item in members.try_iter()? {
            let item = item?;
            if let Ok(section) = item.extract::<PyRef<'_, PyCrossSection>>() {
                inputs.push(CombinationMember::from(section.inner.clone()));
            } else {
                let (section, factor): (PyRef<'_, PyCrossSection>, Bound<'_, PyAny>) =
                    item.extract()?;
                let estimate = if let Ok(value) = factor.extract::<f64>() {
                    Estimate::central(value).map_err(to_py_err)?
                } else {
                    factor.extract::<PyRef<'_, PyEstimate>>()?.inner.clone()
                };
                inputs.push(CombinationMember::from((section.inner.clone(), estimate)));
            }
        }
        let covariance = factor_covariance.map(float_matrix).transpose()?;
        match covariance {
            Some(covariance) => CrossSection::combine_with_covariance(&inputs, covariance),
            None => CrossSection::combine(&inputs),
        }
        .map(|inner| Self { inner })
        .map_err(to_py_err)
    }

    #[getter]
    fn total(&self) -> PyEstimate {
        self.inner.total().clone().into()
    }
    #[getter]
    fn unit(&self) -> PyAreaUnit {
        self.inner.unit().into()
    }
    #[getter]
    fn total_effective_exposure(&self) -> f64 {
        self.inner.total_effective_exposure()
    }
    #[getter]
    fn accepted_integral(&self) -> Option<PyEstimate> {
        self.inner.accepted_integral().cloned().map(Into::into)
    }
    #[getter]
    fn generated_integral(&self) -> Option<PyEstimate> {
        self.inner.generated_integral().cloned().map(Into::into)
    }
    #[getter]
    fn data_yield(&self) -> Option<PyEstimate> {
        self.inner.data_yield().cloned().map(Into::into)
    }
    #[getter]
    fn rate_closure(&self) -> Option<PyRateClosure> {
        self.inner.rate_closure().map(Into::into)
    }

    /// Project one axis group, or a mapping of names to axis groups.
    /// A mapping shares preparation and returns a dictionary; other inputs
    /// return one CrossSectionProjection.
    #[pyo3(signature = (axes: "Axis | Sequence[Axis] | dict[str, Axis | Sequence[Axis]]", *, components=None))]
    fn project<'py>(
        &self,
        py: Python<'py>,
        axes: &Bound<'_, PyAny>,
        components: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let named = axes.hasattr("items")?;
        let requests = if named {
            extract_projections(py, axes)?
        } else {
            vec![Projection::new("cross_section", extract_axes(py, axes)?).map_err(to_py_err)?]
        };
        let results = self
            .inner
            .project_many(&requests, &components.unwrap_or_default())
            .map_err(to_py_err)?;
        if !named {
            let (_, inner) = results.into_iter().next().expect("one projection request");
            return Ok(Py::new(py, PyCrossSectionProjection { inner })?
                .into_bound(py)
                .into_any());
        }
        let output = PyDict::new(py);
        for (name, inner) in results {
            output.set_item(name, Py::new(py, PyCrossSectionProjection { inner })?)?;
        }
        Ok(output.into_any())
    }
}

#[pyclass(
    name = "CrossSectionProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Differential projection of fitted generated-MC intensity.
pub struct PyCrossSectionProjection {
    inner: CrossSectionProjection,
}

#[pymethods]
impl PyCrossSectionProjection {
    #[getter]
    fn axes(&self) -> Vec<Vec<f64>> {
        self.inner.axes().to_vec()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    #[getter]
    fn total(&self) -> PyBinnedEstimate {
        self.inner.total().clone().into()
    }
    #[getter]
    fn components(&self) -> HashMap<String, PyBinnedEstimate> {
        self.inner
            .components()
            .iter()
            .map(|(name, estimate)| (name.clone(), estimate.clone().into()))
            .collect()
    }
    #[getter]
    fn member_validity(&self) -> Vec<Vec<&'static str>> {
        self.inner
            .member_validity()
            .iter()
            .map(|period| period.iter().copied().map(validity_name).collect())
            .collect()
    }
}

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

#[pymethods]
impl PyEstimate {
    #[pyo3(signature = (*, data_fill=true, accepted_mc_fill=true, generated_mc_fill=true, ensemble=false, luminosity=false, branching_exposure=false))]
    fn error(
        &self,
        data_fill: bool,
        accepted_mc_fill: bool,
        generated_mc_fill: bool,
        ensemble: bool,
        luminosity: bool,
        branching_exposure: bool,
    ) -> PyResult<PyScalarErrorView> {
        let budget = laddu_likelihood::ErrorBudget {
            data_fill,
            accepted_mc_fill,
            generated_mc_fill,
            ensemble,
            luminosity,
            branching_exposure,
        };
        self.inner
            .error_with_budget(budget)
            .map(|inner| PyScalarErrorView { inner })
            .map_err(to_py_err)
    }
    #[new]
    #[pyo3(signature = (
        central,
        *,
        draws: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | None" = None,
        source_id=None,
        standard_error=None
    ))]
    fn new(
        central: f64,
        draws: Option<&Bound<'_, PyAny>>,
        source_id: Option<u64>,
        standard_error: Option<f64>,
    ) -> PyResult<Self> {
        let draws = draws.map(float_vec).transpose()?.unwrap_or_default();
        let mut inner = match source_id {
            Some(source_id) => Estimate::with_source_id(central, draws, Some(source_id)),
            None => Estimate::new(central, draws),
        }
        .map_err(to_py_err)?;
        if let Some(error) = standard_error {
            inner = inner.with_standard_error(error).map_err(to_py_err)?;
        }
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

    #[getter]
    fn standard_error(&self) -> Option<f64> {
        self.inner.standard_error()
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
        estimate_binary(self, other, Estimate::checked_add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        estimate_binary(self, other, Estimate::checked_sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        estimate_binary(self, other, Estimate::checked_mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        estimate_binary(self, other, Estimate::checked_div)
    }
}

#[pyclass(
    name = "ScalarErrorView",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Marginal scalar error with included and omitted uncertainty sources.
pub struct PyScalarErrorView {
    inner: laddu_likelihood::ScalarErrorView,
}

#[pymethods]
impl PyScalarErrorView {
    #[getter]
    fn error(&self) -> f64 {
        self.inner.error()
    }
    #[getter]
    fn included_error_components(&self) -> Vec<&'static str> {
        self.inner
            .included()
            .iter()
            .map(|component| error_component_name(*component))
            .collect()
    }
    #[getter]
    fn omitted_error_components(&self) -> Vec<(&'static str, &'static str)> {
        self.inner
            .omitted()
            .iter()
            .map(|(component, reason)| (error_component_name(*component), *reason))
            .collect()
    }
    #[getter]
    fn error_component_sources(&self) -> Vec<(&'static str, u64)> {
        self.inner
            .sources()
            .iter()
            .map(|(component, source)| (error_component_name(*component), *source))
            .collect()
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
    accepted_residual: Option<f64>,
    #[pyo3(get)]
    accepted_absolute_residual: Option<f64>,
    #[pyo3(get)]
    accepted_relative_residual: Option<f64>,
    #[pyo3(get)]
    accepted_residual_draws: Vec<f64>,
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
            accepted_residual: inner.accepted_residual(),
            accepted_absolute_residual: inner.accepted_absolute_residual(),
            accepted_relative_residual: inner.accepted_relative_residual(),
            accepted_residual_draws: inner.accepted_residual_draws().to_vec(),
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
/// Immutable scalar observed and fitted yields for one intensity term.
pub struct PyYield {
    pub(crate) inner: Yield,
    selected_yield: PyEstimate,
    rate_closure: PyRateClosure,
    #[pyo3(get)]
    term_name: String,
    #[pyo3(get)]
    parameters: Vec<f64>,
    #[pyo3(get)]
    has_absolute_rate: bool,
}

impl From<Yield> for PyYield {
    fn from(inner: Yield) -> Self {
        Self {
            term_name: inner.term_name().to_owned(),
            parameters: inner.parameters().to_vec(),
            has_absolute_rate: inner.has_absolute_rate(),
            selected_yield: inner.selected_yield().clone().into(),
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
        self.inner
            .accepted_fitted_yield()
            .map(Into::into)
            .map_err(to_py_err)
    }
    fn generated_fitted_yield(&self) -> PyResult<PyEstimate> {
        self.inner
            .generated_fitted_yield()
            .map(Into::into)
            .map_err(to_py_err)
    }
    fn rate_closure(&self) -> PyRateClosure {
        self.rate_closure.clone()
    }
    #[pyo3(signature = (axes: "Axis | Sequence[Axis]", *, components=None))]
    fn projection(
        &self,
        py: Python<'_>,
        axes: &Bound<'_, PyAny>,
        components: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<PyYieldProjection> {
        let axes = extract_axes(py, axes)?;
        self.inner
            .projection_with_components(&axes, &components.unwrap_or_default())
            .map(Into::into)
            .map_err(to_py_err)
    }
    #[pyo3(signature = (projections: "dict[str, Axis | Sequence[Axis]]", *, components=None))]
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
    #[getter]
    fn errors(&self) -> Vec<f64> {
        self.inner.errors().to_vec()
    }
    #[getter]
    fn included_error_components(&self) -> Vec<&'static str> {
        self.inner
            .included()
            .iter()
            .map(|component| error_component_name(*component))
            .collect()
    }
    #[getter]
    fn omitted_error_components(&self) -> Vec<(&'static str, &'static str)> {
        self.inner
            .omitted()
            .iter()
            .map(|(component, reason)| (error_component_name(*component), *reason))
            .collect()
    }
    #[getter]
    fn error_component_sources(&self) -> Vec<(&'static str, u64)> {
        self.inner
            .sources()
            .iter()
            .map(|(component, source)| (error_component_name(*component), *source))
            .collect()
    }
}

fn error_component_name(component: laddu_likelihood::ErrorComponent) -> &'static str {
    use laddu_likelihood::ErrorComponent;
    match component {
        ErrorComponent::DataFill => "data_fill",
        ErrorComponent::AcceptedMcFill => "accepted_mc_fill",
        ErrorComponent::GeneratedMcFill => "generated_mc_fill",
        ErrorComponent::Ensemble => "ensemble",
        ErrorComponent::Luminosity => "luminosity",
        ErrorComponent::BranchingExposure => "branching_exposure",
    }
}

#[pyclass(
    name = "YieldProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Coherent selected, fitted, and corrected yields with paired draws on shared axes.
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
    fn accepted_estimate(&self) -> PyBinnedEstimate {
        self.inner.accepted_estimate().into()
    }
    #[getter]
    fn generated_estimate(&self) -> PyBinnedEstimate {
        self.inner.generated_estimate().into()
    }
    #[getter]
    fn accepted_draws(&self) -> Vec<Vec<f64>> {
        self.inner.accepted_draws().to_vec()
    }
    #[getter]
    fn generated_draws(&self) -> Vec<Vec<f64>> {
        self.inner.generated_draws().to_vec()
    }
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
    fn has_replica_datasets(&self) -> bool {
        self.inner.has_replica_datasets()
    }
    #[getter]
    fn selected_estimate(&self) -> PyBinnedEstimate {
        self.inner.selected_estimate().into()
    }
    #[getter]
    fn accepted_estimate(&self) -> PyBinnedEstimate {
        self.inner.accepted_estimate().into()
    }
    #[getter]
    fn generated_estimate(&self) -> PyBinnedEstimate {
        self.inner.generated_estimate().into()
    }
    #[getter]
    fn source_id(&self) -> Option<u64> {
        self.inner.source_id()
    }
    #[getter]
    fn selected_draws(&self) -> Vec<Vec<f64>> {
        self.inner.selected_draws().to_vec()
    }
    #[getter]
    fn accepted_draws(&self) -> Vec<Vec<f64>> {
        self.inner.accepted_draws().to_vec()
    }
    #[getter]
    fn generated_draws(&self) -> Vec<Vec<f64>> {
        self.inner.generated_draws().to_vec()
    }
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
}

fn estimate_binary(
    left: &PyEstimate,
    right: &Bound<'_, PyAny>,
    op: impl Fn(&Estimate, &Estimate) -> laddu_likelihood::LikelihoodResult<Estimate>,
) -> PyResult<PyEstimate> {
    if let Ok(right) = right.extract::<PyRef<'_, PyEstimate>>() {
        return op(&left.inner, &right.inner)
            .map(Into::into)
            .map_err(to_py_err);
    }
    if let Ok(right) = right.extract::<f64>() {
        let right = Estimate::central(right).map_err(to_py_err)?;
        return op(&left.inner, &right).map(Into::into).map_err(to_py_err);
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
    #[pyo3(signature = (*, data_fill=true, accepted_mc_fill=true, generated_mc_fill=true, ensemble=false, luminosity=false, branching_exposure=false))]
    fn histogram(
        &self,
        data_fill: bool,
        accepted_mc_fill: bool,
        generated_mc_fill: bool,
        ensemble: bool,
        luminosity: bool,
        branching_exposure: bool,
    ) -> PyResult<PyYieldHistogramView> {
        let budget = laddu_likelihood::ErrorBudget {
            data_fill,
            accepted_mc_fill,
            generated_mc_fill,
            ensemble,
            luminosity,
            branching_exposure,
        };
        self.inner
            .histogram_with_budget(budget)
            .map(|inner| PyYieldHistogramView { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn unit(&self) -> &'static str {
        match self.inner.unit() {
            BinnedEstimateUnit::Unitless => "unitless",
            BinnedEstimateUnit::Yield => "yield",
            BinnedEstimateUnit::CrossSection => "cross_section",
        }
    }
    #[getter]
    fn source_id(&self) -> Option<u64> {
        self.inner.source_id()
    }

    #[getter]
    fn source_ids(&self) -> Vec<u64> {
        self.inner.source_ids().to_vec()
    }

    #[getter]
    fn axes(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.axes().map(<[Vec<f64>]>::to_vec)
    }

    fn __add__(&self, other: &PyBinnedEstimate) -> PyResult<Self> {
        self.inner
            .checked_add(&other.inner)
            .map(Into::into)
            .map_err(to_py_err)
    }
    fn __sub__(&self, other: &PyBinnedEstimate) -> PyResult<Self> {
        self.inner
            .checked_sub(&other.inner)
            .map(Into::into)
            .map_err(to_py_err)
    }
    fn __mul__(&self, other: &PyBinnedEstimate) -> PyResult<Self> {
        self.inner
            .checked_mul(&other.inner)
            .map(Into::into)
            .map_err(to_py_err)
    }
    fn __truediv__(&self, other: &PyBinnedEstimate) -> PyResult<Self> {
        self.inner
            .checked_div(&other.inner)
            .map(Into::into)
            .map_err(to_py_err)
    }
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
