//! Python wrappers for the Rust cross-section analysis API.

use std::collections::HashMap;

use laddu_likelihood::{
    Axis, BinnedEstimate, CrossSection, DifferentialCrossSection, Ensemble, Estimate,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyAny,
};

use super::{error::to_py_err, expr::PyExpr};

#[pyclass(name = "Ensemble", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Named parameter draws used to propagate cross-section uncertainty.
pub struct PyEnsemble {
    pub(crate) inner: Ensemble,
}

#[pymethods]
impl PyEnsemble {
    #[staticmethod]
    #[pyo3(signature = (values, parameter_names, *, source_id=None))]
    /// Build an ensemble from a two-dimensional array of parameter draws.
    fn from_arrays(
        values: PyReadonlyArray2<'_, f64>,
        parameter_names: Vec<String>,
        source_id: Option<u64>,
    ) -> PyResult<Self> {
        let draws = values
            .as_array()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        let inner = match source_id {
            Some(source_id) => Ensemble::with_source_id(parameter_names, draws, source_id),
            None => Ensemble::new(parameter_names, draws),
        }
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (summary, *, discard, thin=1))]
    /// Adapt a Ganesh MCMC summary after explicit burn-in removal and thinning.
    fn from_mcmc(summary: &Bound<'_, PyAny>, discard: usize, thin: usize) -> PyResult<Self> {
        let parameter_names = summary
            .getattr("parameter_names")?
            .extract::<Option<Vec<String>>>()?
            .ok_or_else(|| PyValueError::new_err("MCMC summary has no parameter names"))?;
        let chain = summary
            .getattr("chain")?
            .extract::<PyReadonlyArray3<'_, f64>>()?;
        let chain = chain
            .as_array()
            .outer_iter()
            .map(|walker| walker.outer_iter().map(|step| step.to_vec()).collect())
            .collect::<Vec<Vec<Vec<f64>>>>();
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
    #[new]
    #[pyo3(signature = (central, draws=None, *, source_id=None))]
    fn new(central: f64, draws: Option<Vec<f64>>, source_id: Option<u64>) -> PyResult<Self> {
        let draws = draws.unwrap_or_default();
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
    fn new(expr: &PyExpr, edges: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Axis::new(expr.inner.clone(), edges).map_err(to_py_err)?,
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
    #[staticmethod]
    #[pyo3(signature = (members, *, factors=None))]
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
    fn total(&self, tags: Option<Vec<String>>) -> PyResult<PyEstimate> {
        match tags {
            Some(tags) => self.inner.total_with_tags(&tags),
            None => self.inner.total(),
        }
        .map(Into::into)
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

    #[pyo3(signature = (axes, *, components=None))]
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
