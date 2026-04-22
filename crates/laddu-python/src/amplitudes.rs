use crate::data::PyDataset;
use laddu_core::{
    amplitudes::{Evaluator, Expression, Parameter, TestAmplitude},
    CompiledExpression, LadduError, LadduResult, ReadWrite, ThreadPoolManager,
};
use num::complex::Complex64;
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyBytes, PyList, PyTuple},
};
use std::collections::HashMap;

fn install_with_threads<R: Send>(
    threads: Option<usize>,
    op: impl FnOnce() -> R + Send,
) -> LadduResult<R> {
    ThreadPoolManager::shared().install(threads, op)
}

/// A mathematical expression formed from amplitudes.
///
#[pyclass(name = "Expression", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyExpression(pub Expression);

impl<'py> FromPyObject<'_, 'py> for PyExpression {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(obj) = obj.cast::<PyExpression>() {
            Ok(obj.borrow().clone())
        } else if let Ok(obj) = obj.extract::<f64>() {
            Ok(Self(obj.into()))
        } else if let Ok(obj) = obj.extract::<Complex64>() {
            Ok(Self(obj.into()))
        } else {
            Err(PyTypeError::new_err("Failed to extract Expression"))
        }
    }
}

/// A convenience method to sum sequences of Expressions
///
#[pyfunction(name = "expr_sum")]
pub fn py_expr_sum(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyExpression> {
    if terms.is_empty() {
        return Ok(PyExpression(Expression::zero()));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyExpression>() {
            return Ok(expression);
        }
        return Err(PyTypeError::new_err("Item is not a PyExpression"));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyExpression(Expression::zero()));
    };
    let PyExpression(mut summation) = first_term
        .extract::<PyExpression>()
        .map_err(|_| PyTypeError::new_err("Elements must be PyExpression"))?;
    for term in iter {
        let PyExpression(expr) = term
            .extract::<PyExpression>()
            .map_err(|_| PyTypeError::new_err("Elements must be PyExpression"))?;
        summation = summation + expr;
    }
    Ok(PyExpression(summation))
}

/// A convenience method to multiply sequences of Expressions
///
#[pyfunction(name = "expr_product")]
pub fn py_expr_product(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyExpression> {
    if terms.is_empty() {
        return Ok(PyExpression(Expression::one()));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyExpression>() {
            return Ok(expression);
        }
        return Err(PyTypeError::new_err("Item is not a PyExpression"));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyExpression(Expression::one()));
    };
    let PyExpression(mut product) = first_term
        .extract::<PyExpression>()
        .map_err(|_| PyTypeError::new_err("Elements must be PyExpression"))?;
    for term in iter {
        let PyExpression(expr) = term
            .extract::<PyExpression>()
            .map_err(|_| PyTypeError::new_err("Elements must be PyExpression"))?;
        product = product * expr;
    }
    Ok(PyExpression(product))
}

/// A convenience class representing a zero-valued Expression
///
#[pyfunction(name = "Zero")]
pub fn py_expr_zero() -> PyExpression {
    PyExpression(Expression::zero())
}

/// A convenience class representing a unit-valued Expression
///
#[pyfunction(name = "One")]
pub fn py_expr_one() -> PyExpression {
    PyExpression(Expression::one())
}

#[pymethods]
impl PyExpression {
    /// The free parameters used by the Expression
    ///
    /// Returns
    /// -------
    /// parameters : tuple of str
    ///     The tuple of parameter names
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.parameters())
    }
    /// The free parameters used by the Expression
    #[getter]
    fn free_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.free_parameters())
    }
    /// The fixed parameters used by the Expression
    #[getter]
    fn fixed_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.fixed_parameters())
    }
    /// Number of free parameters
    #[getter]
    fn n_free(&self) -> usize {
        self.0.n_free()
    }
    /// Number of fixed parameters
    #[getter]
    fn n_fixed(&self) -> usize {
        self.0.n_fixed()
    }
    /// Total number of parameters
    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }
    /// Load an Expression by precalculating each term over the given Dataset
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     The Dataset to use in precalculation
    ///
    /// Returns
    /// -------
    /// Evaluator
    ///     An object that can be used to evaluate the `expression` over each event in the
    ///     `dataset`
    fn load(&self, dataset: &PyDataset) -> PyResult<PyEvaluator> {
        Ok(PyEvaluator(self.0.load(&dataset.0)?))
    }
    /// The real part of a complex Expression
    fn real(&self) -> PyExpression {
        PyExpression(self.0.real())
    }
    /// The imaginary part of a complex Expression
    fn imag(&self) -> PyExpression {
        PyExpression(self.0.imag())
    }
    /// The complex conjugate of a complex Expression
    fn conj(&self) -> PyExpression {
        PyExpression(self.0.conj())
    }
    /// The norm-squared of a complex Expression
    fn norm_sqr(&self) -> PyExpression {
        PyExpression(self.0.norm_sqr())
    }
    /// The square root of an Expression
    fn sqrt(&self) -> PyExpression {
        PyExpression(self.0.sqrt())
    }
    /// Raise an Expression to an int, float, or Expression power
    fn power(&self, power: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(value) = power.extract::<i32>() {
            Ok(PyExpression(self.0.powi(value)))
        } else if let Ok(value) = power.extract::<f64>() {
            Ok(PyExpression(self.0.powf(value)))
        } else if let Ok(expression) = power.extract::<PyExpression>() {
            Ok(PyExpression(self.0.pow(&expression.0)))
        } else {
            Err(PyTypeError::new_err(
                "power must be an int, float, or Expression",
            ))
        }
    }
    /// The exponential of an Expression
    fn exp(&self) -> PyExpression {
        PyExpression(self.0.exp())
    }
    /// The sine of an Expression
    fn sin(&self) -> PyExpression {
        PyExpression(self.0.sin())
    }
    /// The cosine of an Expression
    fn cos(&self) -> PyExpression {
        PyExpression(self.0.cos())
    }
    /// The natural logarithm of an Expression
    fn log(&self) -> PyExpression {
        PyExpression(self.0.log())
    }
    /// The complex phase factor exp(i * expression)
    fn cis(&self) -> PyExpression {
        PyExpression(self.0.cis())
    }
    /// Fix a parameter used by this Expression.
    fn fix_parameter(&self, name: &str, value: f64) -> PyResult<()> {
        Ok(self.0.fix_parameter(name, value)?)
    }
    /// Mark a parameter used by this Expression as free.
    fn free_parameter(&self, name: &str) -> PyResult<()> {
        Ok(self.0.free_parameter(name)?)
    }
    /// Rename a single parameter used by this Expression.
    fn rename_parameter(&mut self, old: &str, new: &str) -> PyResult<()> {
        Ok(self.0.rename_parameter(old, new)?)
    }
    /// Rename several parameters used by this Expression.
    fn rename_parameters(&mut self, mapping: HashMap<String, String>) -> PyResult<()> {
        Ok(self.0.rename_parameters(&mapping)?)
    }
    /// Return a tree-like diagnostic view of the compiled Expression.
    #[getter]
    fn compiled_expression(&self) -> PyCompiledExpression {
        PyCompiledExpression(self.0.compiled_expression())
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() + other_expr.0))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0 + self.0.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() - other_expr.0))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0 - self.0.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() * other_expr.0))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0 * self.0.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() / other_expr.0))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for /"))
        }
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0 / self.0.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for /"))
        }
    }
    fn __neg__(&self) -> PyExpression {
        PyExpression(-self.0.clone())
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    #[new]
    fn new() -> Self {
        Self(Expression::create_null())
    }
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            serde_pickle::to_vec(&self.0, serde_pickle::SerOptions::new())
                .map_err(LadduError::PickleError)?
                .as_slice(),
        ))
    }
    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = Self(
            serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
                .map_err(LadduError::PickleError)?,
        );
        Ok(())
    }
}

/// A class which can be used to evaluate a stored Expression
///
/// See Also
/// --------
/// laddu.Expression.load
///
#[pyclass(name = "Evaluator", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyEvaluator(pub Evaluator);

#[pymethods]
impl PyEvaluator {
    /// The free parameters used by the Evaluator
    ///
    /// Returns
    /// -------
    /// parameters : tuple of str
    ///     The tuple of parameter names
    ///
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.parameters())
    }
    /// The free parameters used by the Evaluator
    #[getter]
    fn free_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.free_parameters())
    }
    /// The fixed parameters used by the Evaluator
    #[getter]
    fn fixed_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.fixed_parameters())
    }
    /// Number of free parameters
    #[getter]
    fn n_free(&self) -> usize {
        self.0.n_free()
    }
    /// Number of fixed parameters
    #[getter]
    fn n_fixed(&self) -> usize {
        self.0.n_fixed()
    }
    /// Total number of parameters
    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }
    /// Fix a parameter used by this Evaluator.
    fn fix_parameter(&self, name: &str, value: f64) -> PyResult<()> {
        Ok(self.0.fix_parameter(name, value)?)
    }
    /// Mark a parameter used by this Evaluator as free.
    fn free_parameter(&self, name: &str) -> PyResult<()> {
        Ok(self.0.free_parameter(name)?)
    }
    /// Rename a single parameter used by this Evaluator.
    fn rename_parameter(&self, old: &str, new: &str) -> PyResult<()> {
        Ok(self.0.rename_parameter(old, new)?)
    }
    /// Rename several parameters used by this Evaluator.
    fn rename_parameters(&self, mapping: HashMap<String, String>) -> PyResult<()> {
        Ok(self.0.rename_parameters(&mapping)?)
    }
    /// Activates Amplitudes in the Expression by name or glob selector
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names or ``*``/``?`` glob selectors of Amplitudes to be activated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any selector matches no amplitudes. When
    ///     ``False``, silently skip selectors with no matches.
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
    /// Activates all Amplitudes in the Expression
    ///
    fn activate_all(&self) {
        self.0.activate_all();
    }
    /// Deactivates Amplitudes in the Expression by name or glob selector
    ///
    /// Deactivated Amplitudes act as zeros in the Expression
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names or ``*``/``?`` glob selectors of Amplitudes to be deactivated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any selector matches no amplitudes. When
    ///     ``False``, silently skip selectors with no matches.
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
    /// Deactivates all Amplitudes in the Expression
    ///
    fn deactivate_all(&self) {
        self.0.deactivate_all();
    }
    /// Isolates Amplitudes in the Expression by name or glob selector
    ///
    /// Activates the Amplitudes given in `arg` and deactivates the rest
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names or ``*``/``?`` glob selectors of Amplitudes to be isolated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any selector matches no amplitudes. When
    ///     ``False``, silently skip selectors with no matches.
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

    /// Return the current active-amplitude mask.
    #[getter]
    fn active_mask(&self) -> Vec<bool> {
        self.0.active_mask()
    }

    /// Apply an active-amplitude mask.
    fn set_active_mask(&self, mask: Vec<bool>) -> PyResult<()> {
        self.0.set_active_mask(&mask)?;
        Ok(())
    }

    /// Return a tree-like diagnostic view of the compiled Expression.
    #[getter]
    fn compiled_expression(&self) -> PyCompiledExpression {
        PyCompiledExpression(self.0.compiled_expression())
    }

    /// Return the Expression represented by this Evaluator.
    #[getter]
    fn expression(&self) -> PyExpression {
        PyExpression(self.0.expression())
    }

    /// Evaluate the stored Expression over the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` array of complex values for each Event in the Dataset
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    #[pyo3(signature = (parameters, *, threads=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let values = install_with_threads(threads, || self.0.evaluate(&parameters))?;
        Ok(PyArray1::from_slice(py, &values?))
    }
    /// Evaluate the stored Expression over a subset of the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// indices : list of int
    ///     The indices of events to evaluate
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` array of complex values for each indexed Event in the Dataset
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool
    ///
    #[pyo3(signature = (parameters, indices, *, threads=None))]
    fn evaluate_batch<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        indices: Vec<usize>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let values =
            install_with_threads(threads, || self.0.evaluate_batch(&parameters, &indices))?;
        Ok(PyArray1::from_slice(py, &values?))
    }
    /// Evaluate the gradient of the stored Expression over the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` 2D array of complex values for each Event in the Dataset
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
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let gradients: LadduResult<_> = install_with_threads(threads, || {
            Ok(self
                .0
                .evaluate_gradient(&parameters)?
                .iter()
                .map(|grad| grad.data.as_vec().to_vec())
                .collect::<Vec<Vec<Complex64>>>())
        })?;
        Ok(PyArray2::from_vec2(py, &gradients?).map_err(LadduError::NumpyError)?)
    }
    /// Evaluate the gradient of the stored Expression over a subset of the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// indices : list of int
    ///     The indices of events to evaluate
    /// threads : int, optional
    ///     The number of threads to use (setting this to ``None`` or ``0`` uses the current
    ///     global or context-managed default; any positive value overrides that default for
    ///     this call only)
    ///
    /// Returns
    /// -------
    /// result : array_like
    ///     A ``numpy`` 2D array of complex values for each indexed Event in the Dataset
    ///
    /// Raises
    /// ------
    /// Exception
    ///     If there was an error building the thread pool or problem creating the resulting
    ///     ``numpy`` array
    ///
    #[pyo3(signature = (parameters, indices, *, threads=None))]
    fn evaluate_gradient_batch<'py>(
        &self,
        py: Python<'py>,
        parameters: Vec<f64>,
        indices: Vec<usize>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let gradients: LadduResult<_> = install_with_threads(threads, || {
            Ok(self
                .0
                .evaluate_gradient_batch(&parameters, &indices)?
                .iter()
                .map(|grad| grad.data.as_vec().to_vec())
                .collect::<Vec<Vec<Complex64>>>())
        })?;
        Ok(PyArray2::from_vec2(py, &gradients?).map_err(LadduError::NumpyError)?)
    }
}

/// A class which can be used to display the compiled form of an Expression
///
/// Notes
/// -----
/// This should not be used for anything other than diagnostic purposes.
///
#[pyclass(name = "CompiledExpression", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyCompiledExpression(pub CompiledExpression);

#[pymethods]
impl PyCompiledExpression {
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "Parameter", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyParameter(pub Parameter);

#[pymethods]
impl PyParameter {
    #[getter]
    fn name(&self) -> String {
        self.0.name()
    }
    #[getter]
    fn fixed(&self) -> Option<f64> {
        self.0.fixed()
    }
    #[getter]
    fn initial(&self) -> Option<f64> {
        self.0.initial()
    }
    #[getter]
    fn bounds(&self) -> (Option<f64>, Option<f64>) {
        self.0.bounds()
    }
    #[getter]
    fn unit(&self) -> Option<String> {
        self.0.unit()
    }
    #[getter]
    fn latex(&self) -> Option<String> {
        self.0.latex()
    }
    #[getter]
    fn description(&self) -> Option<String> {
        self.0.description()
    }
}

/// A free parameter which floats during an optimization
///
/// Parameters
/// ----------
/// name : str
///     The name of the free parameter
/// fixed : float, optional
///     If specified, the parameter will be fixed to this value
/// initial : float, optional
///     If specified, the parameter will always be initialized to this value
/// bounds : tuple of (float or None, float or None)
///     Specify the lower and upper bounds for the parameter (None corresponds to no bound)
/// unit : str, optional
///     Optional unit string which may be used to annotate the parameter
/// latex : str, optional
///     Optional LaTeX representation of the parameter
/// description : str, optional
///     Optional description of the parameter
///
/// Returns
/// -------
/// laddu.Parameter
///     An object that can be used as the input for many Amplitude constructors
///
/// Notes
/// -----
/// Two free parameters with the same name are shared in a fit.
///
/// Attempting to set both the fixed and initial value will result in an overwrite (both will be
/// set to the "fixed" value).
///
#[pyfunction(name = "parameter", signature = (name, fixed=None, *, initial=None, bounds=(None, None), unit=None, latex=None, description=None))]
pub fn py_parameter(
    name: &str,
    fixed: Option<f64>,
    initial: Option<f64>,
    bounds: (Option<f64>, Option<f64>),
    unit: Option<&str>,
    latex: Option<&str>,
    description: Option<&str>,
) -> PyParameter {
    let par = Parameter::new(name);
    if let Some(value) = initial {
        par.set_initial(value);
    }
    if let Some(value) = fixed {
        par.set_fixed_value(Some(value)); // TODO: make this all consistent
    }
    par.set_bounds(bounds.0, bounds.1);
    if let Some(unit) = unit {
        par.set_unit(unit);
    }
    if let Some(latex) = latex {
        par.set_latex(latex);
    }
    if let Some(description) = description {
        par.set_description(description);
    }
    PyParameter(par)
}

/// An amplitude used only for internal testing which evaluates `(p0 + i * p1) * event.p4s\[0\].e`.
#[pyfunction(name = "TestAmplitude")]
pub fn py_test_amplitude(name: &str, re: PyParameter, im: PyParameter) -> PyResult<PyExpression> {
    Ok(PyExpression(TestAmplitude::new(name, re.0, im.0)?))
}
