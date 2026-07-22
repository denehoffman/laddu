use std::sync::Arc;

use laddu_likelihood::{
    ExtendedNllTerm, LassoPenalty, Likelihood, LikelihoodProjection, LikelihoodTerm, NllTerm,
    RidgePenalty,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyDict},
};

use super::{data::PyDataset, error::to_py_err, model::PyModel, runtime::PyExecution};

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
    #[pyo3(signature = (model, data, accepted_mc, *, name="extended_nll"))]
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
    #[pyo3(signature = (parameters, lambda_, *, name="ridge"))]
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
    #[pyo3(signature = (parameters, lambda_, *, name="lasso"))]
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
    #[pyo3(signature = (model, data, accepted_mc, *, name="nll"))]
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
/// >>> # term = ld.NLL(model, data, accepted_mc, name="signal")
/// >>> # likelihood = ld.Likelihood([term])
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
    #[pyo3(signature = (parameters, *, acceptance_corrected=true))]
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

/// Resolve Python values into a likelihood's free-parameter order.
///
/// A sequence is used directly. A mapping starts from the default values and
/// replaces entries whose names are present.
pub fn free_values(likelihood: &Likelihood, values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(values) = values.extract::<Vec<f64>>() {
        return Ok(values);
    }
    if let Ok(mapping) = values.cast::<PyDict>() {
        let mut out = likelihood.default_params();
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
    #[pyo3(signature = (terms, *, execution=None))]
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

    /// Evaluate the total negative log-likelihood.
    ///
    /// This is an alias for :meth:`value`.
    fn nll(&self, py: Python<'_>, parameters: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.value(py, parameters)
    }

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
        PyExtendedNll as ExtendedNLL, PyLassoPenalty as LassoPenalty, PyLikelihood as Likelihood,
        PyLikelihoodProjection as LikelihoodProjection, PyNll as NLL,
        PyRidgePenalty as RidgePenalty,
    };
}
