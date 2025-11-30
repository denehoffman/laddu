use crate::data::PyDataset;
use laddu_core::{
    amplitudes::{constant, parameter, Evaluator, Expression, ParameterLike, TestAmplitude},
    f64, LadduError, ReadWrite,
};
use num::complex::Complex64;
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyBytes, PyList},
};
#[cfg(feature = "rayon")]
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;

/// A mathematical expression formed from amplitudes.
///
#[pyclass(name = "Expression", module = "laddu")]
#[derive(Clone)]
pub struct PyExpression(pub Expression);

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
    /// parameters : list of str
    ///     The list of parameter names
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }
    /// The free parameters used by the Expression
    #[getter]
    fn free_parameters(&self) -> Vec<String> {
        self.0.free_parameters()
    }
    /// The fixed parameters used by the Expression
    #[getter]
    fn fixed_parameters(&self) -> Vec<String> {
        self.0.fixed_parameters()
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
    fn n_params(&self) -> usize {
        self.0.n_params()
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
    /// Return a new Expression with the given parameter fixed
    fn fix(&self, name: &str, value: f64) -> PyResult<PyExpression> {
        Ok(PyExpression(self.0.fix(name, value)?))
    }
    /// Return a new Expression with the given parameter freed
    fn free(&self, name: &str) -> PyResult<PyExpression> {
        Ok(PyExpression(self.0.free(name)?))
    }
    /// Return a new Expression with a single parameter renamed
    fn rename_parameter(&self, old: &str, new: &str) -> PyResult<PyExpression> {
        Ok(PyExpression(self.0.rename_parameter(old, new)?))
    }
    /// Return a new Expression with several parameters renamed
    fn rename_parameters(&self, mapping: HashMap<String, String>) -> PyResult<PyExpression> {
        Ok(PyExpression(self.0.rename_parameters(&mapping)?))
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() + other_expr.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(PyExpression(self.0.clone()))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0 + self.0.clone()))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(PyExpression(self.0.clone()))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
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
            bincode::serde::encode_to_vec(&self.0, bincode::config::standard())
                .map_err(LadduError::EncodeError)?
                .as_slice(),
        ))
    }
    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = Self(
            bincode::serde::decode_from_slice(state.as_bytes(), bincode::config::standard())
                .map_err(LadduError::DecodeError)?
                .0,
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
#[pyclass(name = "Evaluator", module = "laddu")]
#[derive(Clone)]
pub struct PyEvaluator(pub Evaluator);

#[pymethods]
impl PyEvaluator {
    /// The free parameters used by the Evaluator
    ///
    /// Returns
    /// -------
    /// parameters : list of str
    ///     The list of parameter names
    ///
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.0.parameters()
    }
    /// The free parameters used by the Evaluator
    #[getter]
    fn free_parameters(&self) -> Vec<String> {
        self.0.free_parameters()
    }
    /// The fixed parameters used by the Evaluator
    #[getter]
    fn fixed_parameters(&self) -> Vec<String> {
        self.0.fixed_parameters()
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
    fn n_params(&self) -> usize {
        self.0.n_params()
    }
    /// Return a new Evaluator with the given parameter fixed
    fn fix(&self, name: &str, value: f64) -> PyResult<PyEvaluator> {
        Ok(PyEvaluator(self.0.fix(name, value)?))
    }
    /// Return a new Evaluator with the given parameter freed
    fn free(&self, name: &str) -> PyResult<PyEvaluator> {
        Ok(PyEvaluator(self.0.free(name)?))
    }
    /// Return a new Evaluator with a single parameter renamed
    fn rename_parameter(&self, old: &str, new: &str) -> PyResult<PyEvaluator> {
        Ok(PyEvaluator(self.0.rename_parameter(old, new)?))
    }
    /// Return a new Evaluator with several parameters renamed
    fn rename_parameters(&self, mapping: HashMap<String, String>) -> PyResult<PyEvaluator> {
        Ok(PyEvaluator(self.0.rename_parameters(&mapping)?))
    }
    /// Activates Amplitudes in the Expression by name
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names of Amplitudes to be activated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any amplitude is missing. When ``False``,
    ///     silently skip missing amplitudes.
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
    /// Deactivates Amplitudes in the Expression by name
    ///
    /// Deactivated Amplitudes act as zeros in the Expression
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names of Amplitudes to be deactivated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any amplitude is missing. When ``False``,
    ///     silently skip missing amplitudes.
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
    /// Isolates Amplitudes in the Expression by name
    ///
    /// Activates the Amplitudes given in `arg` and deactivates the rest
    ///
    /// Parameters
    /// ----------
    /// arg : str or list of str
    ///     Names of Amplitudes to be isolated
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `arg` is not a str or list of str
    /// ValueError
    ///     If `arg` or any items of `arg` are not registered Amplitudes
    /// strict : bool, default=True
    ///     When ``True``, raise an error if any amplitude is missing. When ``False``,
    ///     silently skip missing amplitudes.
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
    /// Evaluate the stored Expression over the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
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
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                &ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| self.0.evaluate(&parameters)),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray1::from_slice(py, &self.0.evaluate(&parameters)))
        }
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
    ///     The number of threads to use (setting this to None will use all available CPUs)
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
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray1::from_slice(
                py,
                &ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| self.0.evaluate_batch(&parameters, &indices)),
            ))
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray1::from_slice(
                py,
                &self.0.evaluate_batch(&parameters, &indices),
            ))
        }
    }
    /// Evaluate the gradient of the stored Expression over the stored Dataset
    ///
    /// Parameters
    /// ----------
    /// parameters : list of float
    ///     The values to use for the free parameters
    /// threads : int, optional
    ///     The number of threads to use (setting this to None will use all available CPUs)
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
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray2::from_vec2(
                py,
                &ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| {
                        self.0
                            .evaluate_gradient(&parameters)
                            .iter()
                            .map(|grad| grad.data.as_vec().to_vec())
                            .collect::<Vec<Vec<Complex64>>>()
                    }),
            )
            .map_err(LadduError::NumpyError)?)
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray2::from_vec2(
                py,
                &self
                    .0
                    .evaluate_gradient(&parameters)
                    .iter()
                    .map(|grad| grad.data.as_vec().to_vec())
                    .collect::<Vec<Vec<Complex64>>>(),
            )
            .map_err(LadduError::NumpyError)?)
        }
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
    ///     The number of threads to use (setting this to None will use all available CPUs)
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
        #[cfg(feature = "rayon")]
        {
            Ok(PyArray2::from_vec2(
                py,
                &ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| {
                        self.0
                            .evaluate_gradient_batch(&parameters, &indices)
                            .iter()
                            .map(|grad| grad.data.as_vec().to_vec())
                            .collect::<Vec<Vec<Complex64>>>()
                    }),
            )
            .map_err(LadduError::NumpyError)?)
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(PyArray2::from_vec2(
                py,
                &self
                    .0
                    .evaluate_gradient_batch(&parameters, &indices)
                    .iter()
                    .map(|grad| grad.data.as_vec().to_vec())
                    .collect::<Vec<Vec<Complex64>>>(),
            )
            .map_err(LadduError::NumpyError)?)
        }
    }
}

/// A class, typically used to allow Amplitudes to take either free parameters or constants as
/// inputs
///
/// See Also
/// --------
/// laddu.parameter
/// laddu.constant
///
#[pyclass(name = "ParameterLike", module = "laddu")]
#[derive(Clone)]
pub struct PyParameterLike(pub ParameterLike);

/// A free parameter which floats during an optimization
///
/// Parameters
/// ----------
/// name : str
///     The name of the free parameter
///
/// Returns
/// -------
/// laddu.ParameterLike
///     An object that can be used as the input for many Amplitude constructors
///
/// Notes
/// -----
/// Two free parameters with the same name are shared in a fit
///
#[pyfunction(name = "parameter")]
pub fn py_parameter(name: &str, value: Option<f64>) -> PyParameterLike {
    if let Some(v) = value {
        PyParameterLike(constant(name, v))
    } else {
        PyParameterLike(parameter(name))
    }
}

/// A term which stays constant during an optimization
///
/// Parameters
/// ----------
/// name : str
///     The name of the parameter
/// value : float
///     The numerical value of the constant
///
/// Returns
/// -------
/// laddu.ParameterLike
///     An object that can be used as the input for many Amplitude constructors
///
#[pyfunction(name = "constant")]
pub fn py_constant(name: &str, value: f64) -> PyParameterLike {
    PyParameterLike(constant(name, value))
}

/// An amplitude used only for internal testing which evaluates `(p0 + i * p1) * event.p4s\[0\].e`.
#[pyfunction(name = "TestAmplitude")]
pub fn py_test_amplitude(
    name: &str,
    re: PyParameterLike,
    im: PyParameterLike,
) -> PyResult<PyExpression> {
    Ok(PyExpression(TestAmplitude::new(name, re.0, im.0)?))
}
