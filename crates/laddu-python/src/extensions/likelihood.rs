use std::collections::HashMap;

use ganesh::python::IntoPySummary;
use laddu_core::{validate_free_parameter_len, LadduError};
use laddu_extensions::{
    likelihood::{LikelihoodTerm, StochasticNLL},
    LikelihoodExpression, LikelihoodScalar, NLL,
};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList},
    IntoPyObjectExt,
};

use crate::{
    amplitudes::{PyCompiledExpression, PyEvaluator, PyExpression, PyParameterMap},
    data::PyDataset,
    extensions::{
        install_laddu_with_threads,
        optimize::{mcmc_from_python, minimize_from_python},
    },
};

#[cfg_attr(coverage_nightly, coverage(off))]
fn extract_subset_names(subset: Option<Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    let Some(subset) = subset else {
        return Ok(None);
    };
    if let Ok(string_arg) = subset.extract::<String>() {
        Ok(Some(vec![string_arg]))
    } else if let Ok(list_arg) = subset.extract::<Vec<String>>() {
        Ok(Some(list_arg))
    } else {
        Err(PyTypeError::new_err(
            "subset must be either a string or a list of strings",
        ))
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn extract_subsets_arg(
    subsets: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<Option<Vec<String>>>>> {
    let Some(subsets) = subsets else {
        return Ok(None);
    };
    subsets
        .extract::<Vec<Option<Vec<String>>>>()
        .map(Some)
        .map_err(|_| {
            PyTypeError::new_err(
                "subsets must be a list whose items are either None or lists of strings",
            )
        })
}

/// Python wrapper for [`LikelihoodExpression`].
#[pyclass(name = "LikelihoodExpression", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyLikelihoodExpression(pub LikelihoodExpression);

/// A convenience method to sum sequences of [`LikelihoodExpression`]s or identifiers.
///
/// Parameters
/// ----------
/// terms : sequence of LikelihoodExpression
///     A non-empty sequence whose elements are summed. Single-element sequences are returned
///     unchanged while empty sequences evaluate to [`LikelihoodZero`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     A new expression representing the sum of all inputs.
///
/// See Also
/// --------
/// likelihood_product
/// LikelihoodZero
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodScalar, likelihood_sum
/// >>> expression = likelihood_sum([LikelihoodScalar('alpha')])
/// >>> expression.evaluate([0.5])
/// 0.5
/// >>> likelihood_sum([]).evaluate([])
/// 0.0
///
/// Notes
/// -----
/// When multiple inputs share the same parameter name, the value and fixed/free status from the
/// earliest term in the sequence take precedence.
#[cfg_attr(coverage_nightly, coverage(off))]
#[pyfunction(name = "likelihood_sum")]
pub fn py_likelihood_sum(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyLikelihoodExpression> {
    if terms.is_empty() {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::zero()));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyLikelihoodExpression>() {
            return Ok(expression);
        }
        return Err(PyTypeError::new_err("Item is not a PyLikelihoodExpression"));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::zero()));
    };
    let PyLikelihoodExpression(mut summation) = first_term
        .extract::<PyLikelihoodExpression>()
        .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
    for term in iter {
        let PyLikelihoodExpression(expr) = term
            .extract::<PyLikelihoodExpression>()
            .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
        summation = summation + expr;
    }
    Ok(PyLikelihoodExpression(summation))
}

/// A convenience method to multiply sequences of [`LikelihoodExpression`]s.
///
/// Parameters
/// ----------
/// terms : sequence of LikelihoodExpression
///     A non-empty sequence whose elements are multiplied. Single-element sequences are returned
///     unchanged while empty sequences evaluate to [`LikelihoodOne`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     A new expression representing the product of all inputs.
///
/// See Also
/// --------
/// likelihood_sum
/// LikelihoodOne
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodScalar, likelihood_product
/// >>> expression = likelihood_product([LikelihoodScalar('alpha'), LikelihoodScalar('beta')])
/// >>> expression.parameters
/// ['alpha', 'beta']
/// >>> expression.evaluate([2.0, 3.0])
/// 6.0
///
/// Notes
/// -----
/// When parameters overlap between inputs, the parameter definition from the earliest term is used.
#[cfg_attr(coverage_nightly, coverage(off))]
#[pyfunction(name = "likelihood_product")]
pub fn py_likelihood_product(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyLikelihoodExpression> {
    if terms.is_empty() {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::one()));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyLikelihoodExpression>() {
            return Ok(expression);
        }
        return Err(PyTypeError::new_err("Item is not a PyLikelihoodExpression"));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyLikelihoodExpression(LikelihoodExpression::one()));
    };
    let PyLikelihoodExpression(mut product) = first_term
        .extract::<PyLikelihoodExpression>()
        .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
    for term in iter {
        let PyLikelihoodExpression(expr) = term
            .extract::<PyLikelihoodExpression>()
            .map_err(|_| PyTypeError::new_err("Elements must be PyLikelihoodExpression"))?;
        product = product * expr;
    }
    Ok(PyLikelihoodExpression(product))
}

/// A convenience constructor for a zero-valued [`LikelihoodExpression`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     An expression that evaluates to ``0`` for any parameter values.
///
/// See Also
/// --------
/// LikelihoodOne
/// likelihood_sum
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodZero
/// >>> expression = LikelihoodZero()
/// >>> expression.parameters
/// []
/// >>> expression.evaluate([])
/// 0.0
#[cfg_attr(coverage_nightly, coverage(off))]
#[pyfunction(name = "LikelihoodZero")]
pub fn py_likelihood_zero() -> PyLikelihoodExpression {
    PyLikelihoodExpression(LikelihoodExpression::zero())
}

/// A convenience constructor for a unit-valued [`LikelihoodExpression`].
///
/// Returns
/// -------
/// LikelihoodExpression
///     An expression that evaluates to ``1`` for any parameter values.
///
/// See Also
/// --------
/// LikelihoodZero
/// likelihood_product
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodOne
/// >>> LikelihoodOne().evaluate([])
/// 1.0
#[cfg_attr(coverage_nightly, coverage(off))]
#[pyfunction(name = "LikelihoodOne")]
pub fn py_likelihood_one() -> PyLikelihoodExpression {
    PyLikelihoodExpression(LikelihoodExpression::one())
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[pymethods]
impl PyLikelihoodExpression {
    /// Parameters referenced by the expression.
    #[getter]
    fn parameters(&self) -> PyParameterMap {
        PyParameterMap(self.0.parameters())
    }

    /// Fix a parameter to a constant value.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the parameter.
    /// value : float
    ///     Value used during evaluation.
    ///
    fn fix_parameter(&self, name: &str, value: f64) -> PyResult<()> {
        Ok(self.0.fix_parameter(name, value)?)
    }

    /// Free a parameter that was previously fixed.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the parameter.
    ///
    fn free_parameter(&self, name: &str) -> PyResult<()> {
        Ok(self.0.free_parameter(name)?)
    }

    /// Rename a parameter.
    ///
    /// Parameters
    /// ----------
    /// old : str
    ///     Current parameter name.
    /// new : str
    ///     Desired parameter name.
    ///
    fn rename_parameter(&self, old: &str, new: &str) -> PyResult<()> {
        Ok(self.0.rename_parameter(old, new)?)
    }

    /// Rename multiple parameters at once.
    ///
    /// Parameters
    /// ----------
    /// mapping : dict[str, str]
    ///     Mapping from old names to new names.
    ///
    fn rename_parameters(&self, mapping: HashMap<String, String>) -> PyResult<()> {
        Ok(self.0.rename_parameters(&mapping)?)
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() + other_expr.0.clone(),
            ))
        } else if let Ok(int) = other.extract::<usize>() {
            if int == 0 {
                Ok(PyLikelihoodExpression(self.0.clone()))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                other_expr.0.clone() + self.0.clone(),
            ))
        } else if let Ok(int) = other.extract::<usize>() {
            if int == 0 {
                Ok(PyLikelihoodExpression(self.0.clone()))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                self.0.clone() * other_expr.0.clone(),
            ))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLikelihoodExpression> {
        if let Ok(other_expr) = other.extract::<PyLikelihoodExpression>() {
            Ok(PyLikelihoodExpression(
                other_expr.0.clone() * self.0.clone(),
            ))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    /// Number of free parameters in the expression.
    #[getter]
    fn n_free(&self) -> usize {
        self.0.n_free()
    }

    /// Number of fixed parameters in the expression.
    #[getter]
    fn n_fixed(&self) -> usize {
        self.0.n_fixed()
    }

    /// Total number of parameters (free + fixed).
    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }

    /// Evaluate the sum of all terms in the expression.
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     Parameter values for the free parameters (length ``n_free``).
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// result : float
    ///     The total negative log-likelihood summed over all terms
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate(&self, parameters: Vec<f64>, threads: Option<usize>) -> PyResult<f64> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        install_laddu_with_threads(threads, || self.0.evaluate(&parameters)).map_err(PyErr::from)
    }
    /// Evaluate the gradient of the sum of all terms in the expression.
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     Parameter values for the free parameters (length ``n_free``).
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` array representing the gradient of the sum of all terms in the
    ///     evaluator with length ``n_free``.
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool or problem creating the resulting
    ///     ``numpy`` array
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        let gradient =
            install_laddu_with_threads(threads, || self.0.evaluate_gradient(&parameters))?;
        Ok(PyArray1::from_slice(py, gradient.as_slice()))
    }
    #[cfg_attr(doctest, doc = "```ignore")]
    /// Minimize the LikelihoodTerm with respect to the free parameters in the model
    ///
    /// This method "runs the fit". Given an initial position `p0`, this
    /// method performs a minimization over the likelihood term, optimizing the model
    /// over the stored signal data and Monte Carlo.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like or ganesh.NelderMeadInit or ganesh.PSOInit or ganesh.CMAESInit or ganesh.DifferentialEvolutionInit
    ///     Initial state for the selected minimizer. Use a length-``n_free`` vector for
    ///     ``lbfgsb``, ``adam``, ``conjugate-gradient``, and ``trust-region``; either a
    ///     vector or ``ganesh.NelderMeadInit`` for ``nelder-mead``; a ``ganesh.CMAESInit``
    ///     for ``cma-es``; a ``ganesh.DifferentialEvolutionInit`` for ``differential-evolution``,
    ///     and either a 2D swarm array or ``ganesh.PSOInit`` for ``pso``.
    /// method : {'lbfgsb', 'adam', 'conjugate-gradient', 'trust-region', 'nelder-mead', 'cma-es', 'differential-evolution', 'pso'}
    ///     The minimization algorithm to use
    /// config : ganesh config object, optional
    ///     Method-specific Ganesh configuration, such as ``ganesh.LBFGSBConfig`` or
    ///     ``ganesh.PSOConfig``. Bounds are configured here when supported by the method.
    /// options : ganesh options object, optional
    ///     Method-specific Ganesh options object controlling step limits, built-in observers,
    ///     and built-in terminators.
    /// observers : MinimizerObserver or list of MinimizerObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MinimizerTerminator or list of MinimizerTerminator, optional
    ///     User-defined terminators which are called at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
    ///
    /// Returns
    /// -------
    /// MinimizationSummary
    ///     A summary of the minimization algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// Examples
    /// --------
    /// >>> import ganesh
    /// >>> expression.minimize(
    /// ...     [1.0],
    /// ...     method='lbfgsb',
    /// ...     options=ganesh.LBFGSBOptions(max_steps=150),
    /// ... )  # doctest: +SKIP
    ///
    /// Notes
    /// -----
    /// ``config`` and ``options`` use Ganesh's Python API directly. For example, pass
    /// ``ganesh.LBFGSBConfig(bounds=[...])`` for bounded L-BFGS-B, or
    /// ``ganesh.AdamOptions(max_steps=500)`` to cap the number of Adam iterations.
    ///
    /// References
    /// ----------
    /// Gao, F. & Han, L. (2010). *Implementing the Nelder-Mead simplex algorithm with adaptive
    /// parameters*. Comput. Optim. Appl. 51(1), 259–277. <https://doi.org/10.1007/s10589-010-9329-3>
    ///
    /// Lagarias, J. C., Reeds, J. A., Wright, M. H., & Wright, P. E. (1998). *Convergence Properties
    /// of the Nelder–Mead Simplex Method in Low Dimensions*. SIAM J. Optim. 9(1), 112–147.
    /// <https://doi.org/10.1137/S1052623496303470>
    ///
    /// Singer, S. & Singer, S. (2004). *Efficient Implementation of the Nelder–Mead Search Algorithm*.
    /// Appl. Numer. Anal. & Comput. 1(2), 524–534. <https://doi.org/10.1002/anac.200410015>
    ///
    #[cfg_attr(doctest, doc = "```")]
    #[pyo3(signature = (p0, *, method="lbfgsb".to_string(), config=None, options=None, observers=None, terminators=None, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        p0: Bound<'_, PyAny>,
        method: String,
        config: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parameter_names = self.0.parameters().free().names();
        minimize_from_python(
            &self.0,
            &p0,
            self.0.n_free(),
            &parameter_names,
            method,
            config.as_ref(),
            options.as_ref(),
            observers,
            terminators,
            threads,
        )?
        .to_py_class(py)
    }
    /// Run an MCMC algorithm on the free parameters of the LikelihoodTerm's model
    ///
    /// This method can be used to sample the underlying likelihood term given an initial
    /// position for each walker `p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : array_like or ganesh.AIESInit or ganesh.ESSInit
    ///     Initial sampler state. Use a 2D walker matrix with shape
    ///     ``(n_walkers, n_parameters)`` for the common case, or pass an explicit Ganesh
    ///     init object for method-specific initialization.
    /// method : {'aies', 'ess'}
    ///     The MCMC algorithm to use
    /// config : ganesh config object, optional
    ///     Method-specific Ganesh configuration, such as ``ganesh.AIESConfig`` or
    ///     ``ganesh.ESSConfig``.
    /// options : ganesh options object, optional
    ///     Method-specific Ganesh options object controlling step limits, built-in observers,
    ///     and built-in terminators.
    /// observers : MCMCObserver or list of MCMCObserver, optional
    ///     User-defined observers which are called at each step
    /// terminators : MCMCTerminator or list of MCMCTerminator, optional
    ///     User-defined terminators which are called at each step
    /// threads : int, default=0
    ///     The number of threads to use (setting this to ``0`` uses the current global or
    ///     context-managed default; any positive value overrides that default for this call
    ///     only)
    ///
    /// Returns
    /// -------
    /// MCMCSummary
    ///     The status of the MCMC algorithm at termination
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    /// See Also
    /// --------
    /// NLL.mcmc
    /// StochasticNLL.mcmc
    ///
    /// Examples
    /// --------
    /// >>> from laddu import LikelihoodScalar, likelihood_sum
    /// >>> import ganesh
    /// >>> expression = likelihood_sum([LikelihoodScalar('alpha')])
    /// >>> summary = expression.mcmc(
    /// ...     [[0.0], [0.4]],
    /// ...     method='aies',
    /// ...     options=ganesh.AIESOptions(max_steps=4),
    /// ... )
    /// >>> summary.dimension[2]
    /// 1
    /// >>> summary.chain(flat=True).shape[1]
    /// 1
    ///
    /// Notes
    /// -----
    /// ``config`` and ``options`` use Ganesh's Python API directly. For example, custom
    /// move mixes belong in ``ganesh.AIESConfig`` or ``ganesh.ESSConfig``, while
    /// ``ganesh.AIESOptions(max_steps=...)`` and ``ganesh.ESSOptions(max_steps=...)`` control
    /// run limits.
    ///
    /// References
    /// ----------
    /// Goodman, J. & Weare, J. (2010). *Ensemble samplers with affine invariance*. CAMCoS 5(1), 65–80. <https://doi.org/10.2140/camcos.2010.5.65>
    ///
    /// Karamanis, M. & Beutler, F. (2021). *Ensemble slice sampling*. Stat Comput 31(5). <https://doi.org/10.1007/s11222-021-10038-2>
    ///
    #[pyo3(signature = (p0, *, method="aies".to_string(), config=None, options=None, observers=None, terminators=None, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc<'py>(
        &self,
        py: Python<'py>,
        p0: Bound<'_, PyAny>,
        method: String,
        config: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parameter_names = self.0.parameters().free().names();
        mcmc_from_python(
            &self.0,
            &p0,
            self.0.n_free(),
            &parameter_names,
            method,
            config.as_ref(),
            options.as_ref(),
            observers,
            terminators,
            threads,
        )?
        .to_py_class(py)
    }
}

/// A (extended) negative log-likelihood evaluator.
#[pyclass(name = "NLL", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyNLL(pub Box<NLL>);

#[cfg_attr(coverage_nightly, coverage(off))]
#[pymethods]
impl PyNLL {
    #[new]
    #[pyo3(signature = (expression, ds_data, ds_accmc, *, n_mc=None))]
    fn new(
        expression: &PyExpression,
        ds_data: &PyDataset,
        ds_accmc: &PyDataset,
        n_mc: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(NLL::new(
            &expression.0,
            &ds_data.0,
            &ds_accmc.0,
            n_mc,
        )?))
    }

    #[getter]
    fn data(&self) -> PyDataset {
        PyDataset(self.0.data_evaluator.dataset.clone())
    }

    #[getter]
    fn accmc(&self) -> PyDataset {
        PyDataset(self.0.accmc_evaluator.dataset.clone())
    }

    #[getter]
    fn data_evaluator(&self) -> PyEvaluator {
        PyEvaluator(self.0.data_evaluator.clone())
    }

    #[getter]
    fn accmc_evaluator(&self) -> PyEvaluator {
        PyEvaluator(self.0.accmc_evaluator.clone())
    }

    #[getter]
    fn expression(&self) -> PyExpression {
        PyExpression(self.0.expression())
    }

    #[getter]
    fn compiled_expression(&self) -> PyCompiledExpression {
        PyCompiledExpression(self.0.compiled_expression())
    }

    #[pyo3(signature = (batch_size, *, seed=None))]
    fn to_stochastic(&self, batch_size: usize, seed: Option<usize>) -> PyResult<PyStochasticNLL> {
        Ok(PyStochasticNLL(self.0.to_stochastic(batch_size, seed)?))
    }

    fn to_expression(&self) -> PyResult<PyLikelihoodExpression> {
        Ok(PyLikelihoodExpression(self.0.clone().into_expression()?))
    }

    #[getter]
    fn parameters(&self) -> PyParameterMap {
        PyParameterMap(self.0.parameters())
    }

    #[getter]
    fn n_free(&self) -> usize {
        self.0.n_free()
    }

    #[getter]
    fn n_fixed(&self) -> usize {
        self.0.n_fixed()
    }

    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }

    fn fix_parameter(&self, name: &str, value: f64) -> PyResult<()> {
        Ok(self.0.fix_parameter(name, value)?)
    }

    fn free_parameter(&self, name: &str) -> PyResult<()> {
        Ok(self.0.free_parameter(name)?)
    }

    fn rename_parameter(&self, old: &str, new: &str) -> PyResult<()> {
        Ok(self.0.rename_parameter(old, new)?)
    }

    fn rename_parameters(&self, mapping: HashMap<String, String>) -> PyResult<()> {
        Ok(self.0.rename_parameters(&mapping)?)
    }

    #[pyo3(signature = (arg, *, strict=true))]
    fn activate(&self, arg: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            if strict {
                self.0.activate_strict(&string_arg)?;
            } else {
                self.0.activate(&string_arg);
            }
        } else if let Ok(list_arg) = arg.cast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            if strict {
                self.0.activate_many_strict(&vec)?;
            } else {
                self.0.activate_many(&vec);
            }
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        }
        Ok(())
    }

    fn activate_all(&self) {
        self.0.activate_all();
    }

    #[pyo3(signature = (arg, *, strict=true))]
    fn deactivate(&self, arg: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            if strict {
                self.0.deactivate_strict(&string_arg)?;
            } else {
                self.0.deactivate(&string_arg);
            }
        } else if let Ok(list_arg) = arg.cast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            if strict {
                self.0.deactivate_many_strict(&vec)?;
            } else {
                self.0.deactivate_many(&vec);
            }
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        }
        Ok(())
    }

    fn deactivate_all(&self) {
        self.0.deactivate_all();
    }

    #[pyo3(signature = (arg, *, strict=true))]
    fn isolate(&self, arg: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            if strict {
                self.0.isolate_strict(&string_arg)?;
            } else {
                self.0.isolate(&string_arg);
            }
        } else if let Ok(list_arg) = arg.cast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            if strict {
                self.0.isolate_many_strict(&vec)?;
            } else {
                self.0.isolate_many(&vec);
            }
        } else {
            return Err(PyTypeError::new_err(
                "Argument must be either a string or a list of strings",
            ));
        }
        Ok(())
    }

    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate(&self, parameters: Vec<f64>, threads: Option<usize>) -> PyResult<f64> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        install_laddu_with_threads(threads, || {
            LikelihoodTerm::evaluate(self.0.as_ref(), &parameters)
        })
        .map_err(PyErr::from)
    }

    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate_gradient<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        let gradient = install_laddu_with_threads(threads, || {
            LikelihoodTerm::evaluate_gradient(self.0.as_ref(), &parameters)
        })?;
        Ok(PyArray1::from_slice(py, gradient.as_slice()))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        parameters,
        *,
        subset = None,
        subsets = None,
        strict = false,
        mc_evaluator = None,
        threads = None
    ))]
    fn project_weights<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        subset: Option<Bound<'_, PyAny>>,
        subsets: Option<Bound<'_, PyAny>>,
        strict: bool,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        if subset.is_some() && subsets.is_some() {
            return Err(PyValueError::new_err(
                "subset and subsets are mutually exclusive",
            ));
        }
        let subset = extract_subset_names(subset)?;
        let subsets = extract_subsets_arg(subsets)?;
        let mc_evaluator = mc_evaluator.map(|pyeval| pyeval.0.clone());
        match (subset, subsets) {
            (Some(names), None) => {
                let projection = install_laddu_with_threads(threads, || {
                    if strict {
                        self.0.project_weights_subset_strict(
                            &parameters,
                            &names,
                            mc_evaluator.clone(),
                        )
                    } else {
                        self.0
                            .project_weights_subset(&parameters, &names, mc_evaluator.clone())
                    }
                })?;
                Ok(PyArray1::from_slice(py, projection.as_slice()).into_any())
            }
            (None, Some(subsets)) => {
                let projection = install_laddu_with_threads(threads, || {
                    let mut rows = Vec::with_capacity(subsets.len());
                    for subset in &subsets {
                        let weights = match subset {
                            Some(names) => {
                                if strict {
                                    self.0.project_weights_subset_strict(
                                        &parameters,
                                        names,
                                        mc_evaluator.clone(),
                                    )?
                                } else {
                                    self.0.project_weights_subset(
                                        &parameters,
                                        names,
                                        mc_evaluator.clone(),
                                    )?
                                }
                            }
                            None => self.0.project_weights(&parameters, mc_evaluator.clone())?,
                        };
                        rows.push(weights);
                    }
                    Ok::<_, LadduError>(rows)
                })?;
                Ok(PyArray2::from_vec2(py, &projection)
                    .map_err(LadduError::NumpyError)?
                    .into_any())
            }
            (None, None) => {
                let projection = install_laddu_with_threads(threads, || {
                    self.0.project_weights(&parameters, mc_evaluator.clone())
                })?;
                Ok(PyArray1::from_slice(py, projection.as_slice()).into_any())
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        parameters,
        *,
        subset = None,
        subsets = None,
        strict = false,
        mc_evaluator = None,
        threads = None
    ))]
    fn project_weights_and_gradients<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        subset: Option<Bound<'_, PyAny>>,
        subsets: Option<Bound<'_, PyAny>>,
        strict: bool,
        mc_evaluator: Option<PyEvaluator>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        validate_free_parameter_len(parameters.len(), self.0.n_free())?;
        if subset.is_some() && subsets.is_some() {
            return Err(PyValueError::new_err(
                "subset and subsets are mutually exclusive",
            ));
        }
        let subset = extract_subset_names(subset)?;
        let subsets = extract_subsets_arg(subsets)?;
        let mc_evaluator = mc_evaluator.map(|pyeval| pyeval.0.clone());
        match (subset, subsets) {
            (Some(names), None) => {
                let (weights, gradients) = install_laddu_with_threads(threads, || {
                    if strict {
                        self.0.project_weights_and_gradients_subset_strict(
                            &parameters,
                            &names,
                            mc_evaluator.clone(),
                        )
                    } else {
                        self.0.project_weights_and_gradients_subset(
                            &parameters,
                            &names,
                            mc_evaluator.clone(),
                        )
                    }
                })?;
                let gradients = gradients
                    .iter()
                    .map(|gradient| gradient.as_slice().to_vec())
                    .collect::<Vec<_>>();
                (
                    PyArray1::from_slice(py, weights.as_slice()),
                    PyArray2::from_vec2(py, &gradients).map_err(LadduError::NumpyError)?,
                )
                    .into_bound_py_any(py)
            }
            (None, Some(subsets)) => {
                let (weights, gradients) = install_laddu_with_threads(threads, || {
                    let mut weight_rows = Vec::with_capacity(subsets.len());
                    let mut gradient_rows = Vec::with_capacity(subsets.len());
                    for subset in &subsets {
                        let (subset_weights, subset_gradients) = match subset {
                            Some(names) => {
                                if strict {
                                    self.0.project_weights_and_gradients_subset_strict(
                                        &parameters,
                                        names,
                                        mc_evaluator.clone(),
                                    )?
                                } else {
                                    self.0.project_weights_and_gradients_subset(
                                        &parameters,
                                        names,
                                        mc_evaluator.clone(),
                                    )?
                                }
                            }
                            None => self
                                .0
                                .project_weights_and_gradients(&parameters, mc_evaluator.clone())?,
                        };
                        weight_rows.push(subset_weights);
                        gradient_rows.push(
                            subset_gradients
                                .iter()
                                .map(|gradient| gradient.as_slice().to_vec())
                                .collect::<Vec<_>>(),
                        );
                    }
                    Ok::<_, LadduError>((weight_rows, gradient_rows))
                })?;
                (
                    PyArray2::from_vec2(py, &weights).map_err(LadduError::NumpyError)?,
                    PyArray3::from_vec3(py, &gradients).map_err(LadduError::NumpyError)?,
                )
                    .into_bound_py_any(py)
            }
            (None, None) => {
                let (weights, gradients) = install_laddu_with_threads(threads, || {
                    self.0
                        .project_weights_and_gradients(&parameters, mc_evaluator.clone())
                })?;
                let gradients = gradients
                    .iter()
                    .map(|gradient| gradient.as_slice().to_vec())
                    .collect::<Vec<_>>();
                (
                    PyArray1::from_slice(py, weights.as_slice()),
                    PyArray2::from_vec2(py, &gradients).map_err(LadduError::NumpyError)?,
                )
                    .into_bound_py_any(py)
            }
            (Some(_), Some(_)) => unreachable!("checked above"),
        }
    }

    #[pyo3(signature = (p0, *, method="lbfgsb".to_string(), config=None, options=None, observers=None, terminators=None, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        p0: Bound<'_, PyAny>,
        method: String,
        config: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parameter_names = self.0.parameters().free().names();
        minimize_from_python(
            self.0.as_ref(),
            &p0,
            self.0.n_free(),
            &parameter_names,
            method,
            config.as_ref(),
            options.as_ref(),
            observers,
            terminators,
            threads,
        )?
        .to_py_class(py)
    }

    #[pyo3(signature = (p0, *, method="aies".to_string(), config=None, options=None, observers=None, terminators=None, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc<'py>(
        &self,
        py: Python<'py>,
        p0: Bound<'_, PyAny>,
        method: String,
        config: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parameter_names = self.0.parameters().free().names();
        mcmc_from_python(
            self.0.as_ref(),
            &p0,
            self.0.n_free(),
            &parameter_names,
            method,
            config.as_ref(),
            options.as_ref(),
            observers,
            terminators,
            threads,
        )?
        .to_py_class(py)
    }
}

/// A stochastic (extended) negative log-likelihood evaluator.
#[pyclass(name = "StochasticNLL", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyStochasticNLL(pub StochasticNLL);

#[cfg_attr(coverage_nightly, coverage(off))]
#[pymethods]
impl PyStochasticNLL {
    #[getter]
    fn nll(&self) -> PyNLL {
        PyNLL(Box::new(self.0.nll.clone()))
    }

    #[getter]
    fn expression(&self) -> PyExpression {
        PyExpression(self.0.expression())
    }

    #[getter]
    fn compiled_expression(&self) -> PyCompiledExpression {
        PyCompiledExpression(self.0.compiled_expression())
    }

    #[pyo3(signature = (p0, *, method="lbfgsb".to_string(), config=None, options=None, observers=None, terminators=None, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        p0: Bound<'_, PyAny>,
        method: String,
        config: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parameter_names = self.0.parameters().free().names();
        minimize_from_python(
            &self.0,
            &p0,
            self.0.n_free(),
            &parameter_names,
            method,
            config.as_ref(),
            options.as_ref(),
            observers,
            terminators,
            threads,
        )?
        .to_py_class(py)
    }

    #[pyo3(signature = (p0, *, method="aies".to_string(), config=None, options=None, observers=None, terminators=None, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn mcmc<'py>(
        &self,
        py: Python<'py>,
        p0: Bound<'_, PyAny>,
        method: String,
        config: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, PyAny>>,
        observers: Option<Bound<'_, PyAny>>,
        terminators: Option<Bound<'_, PyAny>>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let parameter_names = self.0.parameters().free().names();
        mcmc_from_python(
            &self.0,
            &p0,
            self.0.n_free(),
            &parameter_names,
            method,
            config.as_ref(),
            options.as_ref(),
            observers,
            terminators,
            threads,
        )?
        .to_py_class(py)
    }
}

/// A parameterized scalar term which can be converted into a [`LikelihoodExpression`].
///
/// Parameters
/// ----------
/// name : str
///     The name of the new scalar parameter.
///
/// Returns
/// -------
/// LikelihoodExpression
///     A [`LikelihoodExpression`] representing a single free scaling parameter.
///
/// See Also
/// --------
/// likelihood_sum
/// likelihood_product
///
/// Examples
/// --------
/// >>> from laddu import LikelihoodScalar, likelihood_sum
/// >>> expr = likelihood_sum([LikelihoodScalar('alpha')])
/// >>> expr.evaluate([1.25])
/// 1.25
#[cfg_attr(coverage_nightly, coverage(off))]
#[pyfunction(name = "LikelihoodScalar")]
pub fn py_likelihood_scalar(name: String) -> PyResult<PyLikelihoodExpression> {
    Ok(PyLikelihoodExpression(LikelihoodScalar::new(name)?))
}
