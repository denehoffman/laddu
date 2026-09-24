//! Python wrappers for the Rust cross-section analysis API.

use std::collections::HashMap;

use laddu_likelihood::{
    AreaUnit, Axis, BinnedEstimate, BinnedEstimateUnit, ComponentCrossSectionProjection,
    ComponentYieldProjection, Ensemble, Estimate, ExposureCombinedCrossSection,
    ExposureCombinedCrossSectionProjection, ExposureCombinedReferenceCrossSection,
    ExposureCombinedReferenceCrossSectionProjection, ExposureFactor, Luminosity, Projection,
    RateClosure, RateClosureStatus, ReferenceCorrectedYield, ReferenceCorrectedYieldProjection,
    ReferenceCorrectionProvenance, ReferenceCrossSection, ReferenceCrossSectionProjection, Yield,
    YieldBinValidity, YieldCrossSection, YieldCrossSectionProjection, YieldHistogramView,
    YieldProjection,
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
    inner: Luminosity,
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

#[pyclass(
    name = "YieldCrossSection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Scalar cross sections converted from one yield context.
pub struct PyYieldCrossSection {
    inner: YieldCrossSection,
}

#[pymethods]
impl PyYieldCrossSection {
    #[staticmethod]
    fn combine(
        members: Vec<(PyRef<'_, PyYieldCrossSection>, f64)>,
    ) -> PyResult<PyExposureCombinedCrossSection> {
        let members = members
            .into_iter()
            .map(|(member, factor)| (member.inner.clone(), factor))
            .collect::<Vec<_>>();
        YieldCrossSection::combine(&members)
            .map(|inner| PyExposureCombinedCrossSection { inner })
            .map_err(to_py_err)
    }
    #[staticmethod]
    fn combine_with_factors(
        members: Vec<(PyRef<'_, PyYieldCrossSection>, PyRef<'_, PyExposureFactor>)>,
    ) -> PyResult<PyExposureCombinedCrossSection> {
        let members = members
            .into_iter()
            .map(|(member, factor)| (member.inner.clone(), factor.inner.clone()))
            .collect::<Vec<_>>();
        YieldCrossSection::combine_with_factors(&members)
            .map(|inner| PyExposureCombinedCrossSection { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn observed(&self) -> PyEstimate {
        self.inner.observed().clone().into()
    }
    #[getter]
    fn fitted(&self) -> Option<PyEstimate> {
        self.inner.fitted().cloned().map(Into::into)
    }
    #[getter]
    fn accepted_fitted(&self) -> Option<PyEstimate> {
        self.inner.accepted_fitted().cloned().map(Into::into)
    }
    #[getter]
    fn unit(&self) -> PyAreaUnit {
        self.inner.unit().into()
    }
    #[getter]
    fn luminosity(&self) -> PyLuminosity {
        PyLuminosity {
            inner: self.inner.luminosity().clone(),
        }
    }
    #[getter]
    fn rate_closure(&self) -> PyRateClosure {
        self.inner.rate_closure().clone().into()
    }
}

#[pyclass(
    name = "ExposureCombinedCrossSection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Cross sections pooled through total effective exposure.
pub struct PyExposureCombinedCrossSection {
    inner: ExposureCombinedCrossSection,
}

#[pyclass(name = "ExposureFactor", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Positive exposure factor with optional paired draws.
pub struct PyExposureFactor {
    inner: ExposureFactor,
}

#[pymethods]
impl PyExposureFactor {
    #[new]
    #[pyo3(signature = (central, *, draws=None, source_id=None))]
    fn new(
        central: f64,
        draws: Option<&Bound<'_, PyAny>>,
        source_id: Option<u64>,
    ) -> PyResult<Self> {
        let draws = draws.map(float_vec).transpose()?.unwrap_or_default();
        let estimate = Estimate::with_source_id(central, draws, source_id).map_err(to_py_err)?;
        ExposureFactor::from_estimate(estimate)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn estimate(&self) -> PyEstimate {
        self.inner.estimate().clone().into()
    }
}

#[pymethods]
impl PyExposureCombinedCrossSection {
    #[getter]
    fn observed(&self) -> PyEstimate {
        self.inner.observed().clone().into()
    }
    #[getter]
    fn fitted(&self) -> Option<PyEstimate> {
        self.inner.fitted().cloned().map(Into::into)
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
    fn member_luminosities(&self) -> Vec<PyLuminosity> {
        self.inner
            .member_luminosities()
            .iter()
            .cloned()
            .map(|inner| PyLuminosity { inner })
            .collect()
    }
    #[getter]
    fn exposure_factors(&self) -> Vec<f64> {
        self.inner.exposure_factors().to_vec()
    }
}

#[pyclass(
    name = "ReferenceCrossSection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Reference-corrected scalar cross section.
pub struct PyReferenceCrossSection {
    inner: ReferenceCrossSection,
}

#[pymethods]
impl PyReferenceCrossSection {
    #[staticmethod]
    fn combine(
        members: Vec<(PyRef<'_, PyReferenceCrossSection>, f64)>,
    ) -> PyResult<PyExposureCombinedReferenceCrossSection> {
        let members = members
            .into_iter()
            .map(|(member, factor)| (member.inner.clone(), factor))
            .collect::<Vec<_>>();
        ReferenceCrossSection::combine(&members)
            .map(|inner| PyExposureCombinedReferenceCrossSection { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn value(&self) -> PyEstimate {
        self.inner.value().clone().into()
    }
    #[getter]
    fn unit(&self) -> PyAreaUnit {
        self.inner.unit().into()
    }
    #[getter]
    fn luminosity(&self) -> PyLuminosity {
        PyLuminosity {
            inner: self.inner.luminosity().clone(),
        }
    }
    #[getter]
    fn provenance(&self) -> PyReferenceCorrectionProvenance {
        PyReferenceCorrectionProvenance {
            inner: self.inner.provenance().clone(),
        }
    }
}

#[pyclass(
    name = "ExposureCombinedReferenceCrossSection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Reference-corrected periods pooled through total effective exposure.
pub struct PyExposureCombinedReferenceCrossSection {
    inner: ExposureCombinedReferenceCrossSection,
}

#[pymethods]
impl PyExposureCombinedReferenceCrossSection {
    #[getter]
    fn value(&self) -> PyEstimate {
        self.inner.value().clone().into()
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
    fn provenances(&self) -> Vec<PyReferenceCorrectionProvenance> {
        self.inner
            .provenances()
            .iter()
            .cloned()
            .map(|inner| PyReferenceCorrectionProvenance { inner })
            .collect()
    }
}

#[pyclass(
    name = "YieldCrossSectionProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Differential cross sections converted from a yield projection.
pub struct PyYieldCrossSectionProjection {
    inner: YieldCrossSectionProjection,
}

#[pymethods]
impl PyYieldCrossSectionProjection {
    #[staticmethod]
    fn combine(
        members: Vec<(PyRef<'_, PyYieldCrossSectionProjection>, f64)>,
    ) -> PyResult<PyExposureCombinedCrossSectionProjection> {
        let members = members
            .into_iter()
            .map(|(member, factor)| (member.inner.clone(), factor))
            .collect::<Vec<_>>();
        YieldCrossSectionProjection::combine(&members)
            .map(|inner| PyExposureCombinedCrossSectionProjection { inner })
            .map_err(to_py_err)
    }
    #[staticmethod]
    fn combine_with_factors(
        members: Vec<(
            PyRef<'_, PyYieldCrossSectionProjection>,
            PyRef<'_, PyExposureFactor>,
        )>,
    ) -> PyResult<PyExposureCombinedCrossSectionProjection> {
        let members = members
            .into_iter()
            .map(|(member, factor)| (member.inner.clone(), factor.inner.clone()))
            .collect::<Vec<_>>();
        YieldCrossSectionProjection::combine_with_factors(&members)
            .map(|inner| PyExposureCombinedCrossSectionProjection { inner })
            .map_err(to_py_err)
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
    fn observed(&self) -> PyBinnedEstimate {
        self.inner.observed().clone().into()
    }
    #[getter]
    fn fitted(&self) -> Option<PyBinnedEstimate> {
        self.inner.fitted().cloned().map(Into::into)
    }
    #[getter]
    fn accepted_fitted(&self) -> Option<PyBinnedEstimate> {
        self.inner.accepted_fitted().cloned().map(Into::into)
    }
    #[getter]
    fn components(&self) -> HashMap<String, PyComponentCrossSectionProjection> {
        self.inner
            .components()
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    PyComponentCrossSectionProjection {
                        inner: value.clone(),
                    },
                )
            })
            .collect()
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
    fn luminosity(&self) -> PyLuminosity {
        PyLuminosity {
            inner: self.inner.luminosity().clone(),
        }
    }
}

#[pyclass(
    name = "ExposureCombinedCrossSectionProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Differential cross sections pooled through total effective exposure.
pub struct PyExposureCombinedCrossSectionProjection {
    inner: ExposureCombinedCrossSectionProjection,
}

#[pymethods]
impl PyExposureCombinedCrossSectionProjection {
    #[getter]
    fn axes(&self) -> Vec<Vec<f64>> {
        self.inner.axes().to_vec()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }
    #[getter]
    fn observed(&self) -> PyBinnedEstimate {
        self.inner.observed().clone().into()
    }
    #[getter]
    fn fitted(&self) -> Option<PyBinnedEstimate> {
        self.inner.fitted().cloned().map(Into::into)
    }
    #[getter]
    fn components(&self) -> HashMap<String, PyComponentCrossSectionProjection> {
        self.inner
            .components()
            .iter()
            .map(|(name, inner)| {
                (
                    name.clone(),
                    PyComponentCrossSectionProjection {
                        inner: inner.clone(),
                    },
                )
            })
            .collect()
    }
    #[getter]
    fn total_effective_exposure(&self) -> f64 {
        self.inner.total_effective_exposure()
    }
    #[getter]
    fn unit(&self) -> PyAreaUnit {
        self.inner.unit().into()
    }
}

#[pyclass(
    name = "ComponentCrossSectionProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Model-only accepted and generated differential cross sections.
pub struct PyComponentCrossSectionProjection {
    inner: ComponentCrossSectionProjection,
}

#[pymethods]
impl PyComponentCrossSectionProjection {
    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags().to_vec()
    }
    #[getter]
    fn accepted(&self) -> PyBinnedEstimate {
        self.inner.accepted().clone().into()
    }
    #[getter]
    fn generated(&self) -> PyBinnedEstimate {
        self.inner.generated().clone().into()
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
}

#[pyclass(
    name = "ReferenceCrossSectionProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Reference-corrected differential cross section.
pub struct PyReferenceCrossSectionProjection {
    inner: ReferenceCrossSectionProjection,
}

#[pymethods]
impl PyReferenceCrossSectionProjection {
    #[staticmethod]
    fn combine(
        members: Vec<(PyRef<'_, PyReferenceCrossSectionProjection>, f64)>,
    ) -> PyResult<PyExposureCombinedReferenceCrossSectionProjection> {
        let members = members
            .into_iter()
            .map(|(member, factor)| (member.inner.clone(), factor))
            .collect::<Vec<_>>();
        ReferenceCrossSectionProjection::combine(&members)
            .map(|inner| PyExposureCombinedReferenceCrossSectionProjection { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn value(&self) -> PyBinnedEstimate {
        self.inner.value().clone().into()
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
    fn provenance(&self) -> PyReferenceCorrectionProvenance {
        PyReferenceCorrectionProvenance {
            inner: self.inner.provenance().clone(),
        }
    }
    #[getter]
    fn luminosity(&self) -> PyLuminosity {
        PyLuminosity {
            inner: self.inner.luminosity().clone(),
        }
    }
}

#[pyclass(
    name = "ExposureCombinedReferenceCrossSectionProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Reference-corrected differential periods pooled by effective exposure.
pub struct PyExposureCombinedReferenceCrossSectionProjection {
    inner: ExposureCombinedReferenceCrossSectionProjection,
}

#[pymethods]
impl PyExposureCombinedReferenceCrossSectionProjection {
    #[getter]
    fn value(&self) -> PyBinnedEstimate {
        self.inner.value().clone().into()
    }
    #[getter]
    fn total_effective_exposure(&self) -> f64 {
        self.inner.total_effective_exposure()
    }
    #[getter]
    fn provenances(&self) -> Vec<PyReferenceCorrectionProvenance> {
        self.inner
            .provenances()
            .iter()
            .cloned()
            .map(|inner| PyReferenceCorrectionProvenance { inner })
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
    #[pyo3(signature = (*, data_fill=true, accepted_mc_fill=true, generated_mc_fill=true, ensemble=false, luminosity=false, branching_exposure=false, reference_model=false))]
    fn error(
        &self,
        data_fill: bool,
        accepted_mc_fill: bool,
        generated_mc_fill: bool,
        ensemble: bool,
        luminosity: bool,
        branching_exposure: bool,
        reference_model: bool,
    ) -> PyResult<PyScalarErrorView> {
        let budget = laddu_likelihood::ErrorBudget {
            data_fill,
            accepted_mc_fill,
            generated_mc_fill,
            ensemble,
            luminosity,
            branching_exposure,
            reference_model,
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
    fn to_cross_section(&self, luminosity: &PyLuminosity) -> PyReferenceCrossSection {
        PyReferenceCrossSection {
            inner: self.inner.to_cross_section(&luminosity.inner),
        }
    }
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
    fn to_cross_section(&self, luminosity: &PyLuminosity) -> PyResult<PyYieldCrossSection> {
        self.inner
            .to_cross_section(&luminosity.inner)
            .map(|inner| PyYieldCrossSection { inner })
            .map_err(to_py_err)
    }
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
        projections: "dict[str, Axis | Sequence[Axis]]",
        luminosity,
        *,
        components: "dict[str, Sequence[str]] | None" = None
    ))]
    /// Evaluate named yields once and convert them with typed luminosity.
    fn cross_section_projection_set<'py>(
        &self,
        py: Python<'py>,
        projections: &Bound<'_, PyAny>,
        luminosity: &PyLuminosity,
        components: Option<HashMap<String, Vec<String>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let projections = extract_projections(py, projections)?;
        let results = self
            .inner
            .cross_section_projection_set(
                &projections,
                &components.unwrap_or_default(),
                &luminosity.inner,
            )
            .map_err(to_py_err)?;
        let output = PyDict::new(py);
        for (name, result) in results.iter() {
            output.set_item(
                name,
                Py::new(
                    py,
                    PyYieldCrossSectionProjection {
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

    #[pyo3(signature = (
        axes: "Axis | Sequence[Axis]",
        reference_likelihood,
        reference_term_name,
        *,
        generated_mc,
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        ensemble=None
    ))]
    /// Correct a selected-data projection with an explicit reference acceptance.
    fn reference_corrected_projection(
        &self,
        py: Python<'_>,
        axes: &Bound<'_, PyAny>,
        reference_likelihood: &PyLikelihood,
        reference_term_name: &str,
        generated_mc: &PyDataset,
        parameters: &Bound<'_, PyAny>,
        ensemble: Option<&PyEnsemble>,
    ) -> PyResult<PyReferenceCorrectedYieldProjection> {
        let axes = extract_axes(py, axes)?;
        let parameters = free_values(&reference_likelihood.inner, parameters)?;
        let inner = self
            .inner
            .reference_corrected_projection(
                &axes,
                std::sync::Arc::clone(&reference_likelihood.inner),
                reference_term_name,
                generated_mc.inner.clone(),
                parameters,
                ensemble.map(|value| value.inner.clone()),
            )
            .map_err(to_py_err)?;
        Ok(PyReferenceCorrectedYieldProjection { inner })
    }
}

#[pyclass(
    name = "ReferenceCorrectedYieldProjection",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Binned selected yield corrected by an explicit reference model.
pub struct PyReferenceCorrectedYieldProjection {
    inner: ReferenceCorrectedYieldProjection,
}

#[pymethods]
impl PyReferenceCorrectedYieldProjection {
    fn to_cross_section(
        &self,
        luminosity: &PyLuminosity,
    ) -> PyResult<PyReferenceCrossSectionProjection> {
        self.inner
            .to_cross_section(&luminosity.inner)
            .map(|inner| PyReferenceCrossSectionProjection { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn value(&self) -> PyBinnedEstimate {
        self.inner.value().clone().into()
    }
    #[getter]
    fn acceptance(&self) -> PyBinnedEstimate {
        self.inner.acceptance().clone().into()
    }
    #[getter]
    fn provenance(&self) -> PyReferenceCorrectionProvenance {
        PyReferenceCorrectionProvenance {
            inner: self.inner.provenance().clone(),
        }
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
        ErrorComponent::ReferenceModel => "reference_model",
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
    fn to_cross_section(
        &self,
        luminosity: &PyLuminosity,
    ) -> PyResult<PyComponentCrossSectionProjection> {
        self.inner
            .to_cross_section(&luminosity.inner)
            .map(|inner| PyComponentCrossSectionProjection { inner })
            .map_err(to_py_err)
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
    fn to_cross_section(
        &self,
        luminosity: &PyLuminosity,
    ) -> PyResult<PyYieldCrossSectionProjection> {
        self.inner
            .to_cross_section(&luminosity.inner)
            .map(|inner| PyYieldCrossSectionProjection { inner })
            .map_err(to_py_err)
    }
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
    fn acceptance_estimate(&self) -> PyBinnedEstimate {
        self.inner.acceptance_estimate().into()
    }
    #[getter]
    fn corrected_estimate(&self) -> PyBinnedEstimate {
        self.inner.corrected_estimate().into()
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
    fn acceptance_draws(&self) -> Vec<Vec<f64>> {
        self.inner.acceptance_draws().to_vec()
    }
    #[getter]
    fn corrected_draws(&self) -> Vec<Vec<f64>> {
        self.inner.corrected_draws().to_vec()
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
    #[pyo3(signature = (*, data_fill=true, accepted_mc_fill=true, generated_mc_fill=true, ensemble=false, luminosity=false, branching_exposure=false, reference_model=false))]
    fn histogram(
        &self,
        data_fill: bool,
        accepted_mc_fill: bool,
        generated_mc_fill: bool,
        ensemble: bool,
        luminosity: bool,
        branching_exposure: bool,
        reference_model: bool,
    ) -> PyResult<PyYieldHistogramView> {
        let budget = laddu_likelihood::ErrorBudget {
            data_fill,
            accepted_mc_fill,
            generated_mc_fill,
            ensemble,
            luminosity,
            branching_exposure,
            reference_model,
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
