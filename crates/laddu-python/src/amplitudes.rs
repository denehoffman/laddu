use crate::data::PyDataset;
use laddu_core::{
    amplitudes::{
        constant, parameter, Amplitude, AmplitudeID, Evaluator, Expression, Manager, Model,
        ParameterLike,
    },
    traits::ReadWrite,
    Complex, Float, LadduError,
};
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyBytes, PyList},
};
#[cfg(feature = "rayon")]
use rayon::ThreadPoolBuilder;

/// An object which holds a registered ``Amplitude``
///
/// See Also
/// --------
/// laddu.Manager.register
///
#[pyclass(name = "AmplitudeID", module = "laddu")]
#[derive(Clone)]
pub struct PyAmplitudeID(AmplitudeID);

/// A mathematical expression formed from AmplitudeIDs
///
#[pyclass(name = "Expression", module = "laddu")]
#[derive(Clone)]
pub struct PyExpression(Expression);

/// A convenience method to sum sequences of Amplitudes
///
#[pyfunction(name = "amplitude_sum")]
pub fn py_amplitude_sum(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyExpression> {
    if terms.is_empty() {
        return Ok(PyExpression(Expression::Zero));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyExpression>() {
            return Ok(expression);
        }
        if let Ok(py_amplitude_id) = term.extract::<PyAmplitudeID>() {
            return Ok(PyExpression(Expression::Amp(py_amplitude_id.0)));
        }
        return Err(PyTypeError::new_err(
            "Item is neither a PyExpression nor a PyAmplitudeID",
        ));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyExpression(Expression::Zero));
    };
    if let Ok(first_expression) = first_term.extract::<PyExpression>() {
        let mut summation = first_expression.clone();
        for term in iter {
            summation = summation.__add__(term)?;
        }
        return Ok(summation);
    }
    if let Ok(first_amplitude_id) = first_term.extract::<PyAmplitudeID>() {
        let mut summation = PyExpression(Expression::Amp(first_amplitude_id.0));
        for term in iter {
            summation = summation.__add__(term)?;
        }
        return Ok(summation);
    }
    Err(PyTypeError::new_err(
        "Elements must be PyExpression or PyAmplitudeID",
    ))
}

/// A convenience method to multiply sequences of Amplitudes
///
#[pyfunction(name = "amplitude_product")]
pub fn py_amplitude_product(terms: Vec<Bound<'_, PyAny>>) -> PyResult<PyExpression> {
    if terms.is_empty() {
        return Ok(PyExpression(Expression::One));
    }
    if terms.len() == 1 {
        let term = &terms[0];
        if let Ok(expression) = term.extract::<PyExpression>() {
            return Ok(expression);
        }
        if let Ok(py_amplitude_id) = term.extract::<PyAmplitudeID>() {
            return Ok(PyExpression(Expression::Amp(py_amplitude_id.0)));
        }
        return Err(PyTypeError::new_err(
            "Item is neither a PyExpression nor a PyAmplitudeID",
        ));
    }
    let mut iter = terms.iter();
    let Some(first_term) = iter.next() else {
        return Ok(PyExpression(Expression::One));
    };
    if let Ok(first_expression) = first_term.extract::<PyExpression>() {
        let mut product = first_expression.clone();
        for term in iter {
            product = product.__mul__(term)?;
        }
        return Ok(product);
    }
    if let Ok(first_amplitude_id) = first_term.extract::<PyAmplitudeID>() {
        let mut product = PyExpression(Expression::Amp(first_amplitude_id.0));
        for term in iter {
            product = product.__mul__(term)?;
        }
        return Ok(product);
    }
    Err(PyTypeError::new_err(
        "Elements must be PyExpression or PyAmplitudeID",
    ))
}

/// A convenience class representing a zero-valued Expression
///
#[pyfunction(name = "AmplitudeZero")]
pub fn py_amplitude_zero() -> PyExpression {
    PyExpression(Expression::Zero)
}

/// A convenience class representing a unit-valued Expression
///
#[pyfunction(name = "AmplitudeOne")]
pub fn py_amplitude_one() -> PyExpression {
    PyExpression(Expression::One)
}

#[pymethods]
impl PyAmplitudeID {
    /// The real part of a complex Amplitude
    ///
    /// Returns
    /// -------
    /// Expression
    ///     The real part of the given Amplitude
    ///
    fn real(&self) -> PyExpression {
        PyExpression(self.0.real())
    }
    /// The imaginary part of a complex Amplitude
    ///
    /// Returns
    /// -------
    /// Expression
    ///     The imaginary part of the given Amplitude
    ///
    fn imag(&self) -> PyExpression {
        PyExpression(self.0.imag())
    }
    /// The norm-squared of a complex Amplitude
    ///
    /// This is computed as :math:`AA^*` where :math:`A^*` is the complex conjugate
    ///
    /// Returns
    /// -------
    /// Expression
    ///     The norm-squared of the given Amplitude
    ///
    fn norm_sqr(&self) -> PyExpression {
        PyExpression(self.0.norm_sqr())
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(self.0.clone() + other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() + other_expr.0.clone()))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(PyExpression(Expression::Amp(self.0.clone())))
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
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(other_aid.0.clone() + self.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0.clone() + self.0.clone()))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(PyExpression(Expression::Amp(self.0.clone())))
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(self.0.clone() * other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() * other_expr.0.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(other_aid.0.clone() * self.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0.clone() * self.0.clone()))
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
}

#[pymethods]
impl PyExpression {
    /// The real part of a complex Expression
    ///
    /// Returns
    /// -------
    /// Expression
    ///     The real part of the given Expression
    ///
    fn real(&self) -> PyExpression {
        PyExpression(self.0.real())
    }
    /// The imaginary part of a complex Expression
    ///
    /// Returns
    /// -------
    /// Expression
    ///     The imaginary part of the given Expression
    ///
    fn imag(&self) -> PyExpression {
        PyExpression(self.0.imag())
    }
    /// The norm-squared of a complex Expression
    ///
    /// This is computed as :math:`AA^*` where :math:`A^*` is the complex conjugate
    ///
    /// Returns
    /// -------
    /// Expression
    ///     The norm-squared of the given Expression
    ///
    fn norm_sqr(&self) -> PyExpression {
        PyExpression(self.0.norm_sqr())
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(self.0.clone() + other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() + other_expr.0.clone()))
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
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(other_aid.0.clone() + self.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0.clone() + self.0.clone()))
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
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(self.0.clone() * other_aid.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(self.0.clone() * other_expr.0.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        if let Ok(other_aid) = other.extract::<PyRef<PyAmplitudeID>>() {
            Ok(PyExpression(other_aid.0.clone() * self.0.clone()))
        } else if let Ok(other_expr) = other.extract::<PyExpression>() {
            Ok(PyExpression(other_expr.0.clone() * self.0.clone()))
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
}

/// A class which can be used to register Amplitudes and store precalculated data
///
#[pyclass(name = "Manager", module = "laddu")]
pub struct PyManager(Manager);

#[pymethods]
impl PyManager {
    #[new]
    fn new() -> Self {
        Self(Manager::default())
    }
    /// The free parameters used by the Manager
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
    /// Register an Amplitude with the Manager
    ///
    /// Parameters
    /// ----------
    /// amplitude : Amplitude
    ///     The Amplitude to register
    ///
    /// Returns
    /// -------
    /// AmplitudeID
    ///     A reference to the registered `amplitude` that can be used to form complex
    ///     Expressions
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the name of the ``amplitude`` has already been registered
    ///
    fn register(&mut self, amplitude: &PyAmplitude) -> PyResult<PyAmplitudeID> {
        Ok(PyAmplitudeID(self.0.register(amplitude.0.clone())?))
    }
    /// Generate a Model from the given expression made of registered Amplitudes
    ///
    /// Parameters
    /// ----------
    /// expression : Expression or AmplitudeID
    ///     The expression to use in precalculation
    ///
    /// Returns
    /// -------
    /// Model
    ///     An object which represents the underlying mathematical model and can be loaded with
    ///     a Dataset
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the expression is not convertable to a Model
    ///
    /// Notes
    /// -----
    /// While the given `expression` will be the one evaluated in the end, all registered
    /// Amplitudes will be loaded, and all of their parameters will be included in the final
    /// expression. These parameters will have no effect on evaluation, but they must be
    /// included in function calls.
    ///
    fn model(&self, expression: &Bound<'_, PyAny>) -> PyResult<PyModel> {
        let expression = if let Ok(expression) = expression.extract::<PyExpression>() {
            Ok(expression.0)
        } else if let Ok(aid) = expression.extract::<PyAmplitudeID>() {
            Ok(Expression::Amp(aid.0))
        } else {
            Err(PyTypeError::new_err(
                "'expression' must either by an Expression or AmplitudeID",
            ))
        }?;
        Ok(PyModel(self.0.model(&expression)))
    }
}

/// A class which represents a model composed of registered Amplitudes
///
#[pyclass(name = "Model", module = "laddu")]
pub struct PyModel(pub Model);

#[pymethods]
impl PyModel {
    /// The free parameters used by the Manager
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
    /// Load a Model by precalculating each term over the given Dataset
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
    ///
    /// Notes
    /// -----
    /// While the given `expression` will be the one evaluated in the end, all registered
    /// Amplitudes will be loaded, and all of their parameters will be included in the final
    /// expression. These parameters will have no effect on evaluation, but they must be
    /// included in function calls.
    ///
    fn load(&self, dataset: &PyDataset) -> PyEvaluator {
        PyEvaluator(self.0.load(&dataset.0))
    }
    /// Save the Model to a file
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     The path of the new file (overwrites if the file exists!)
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If anything fails when trying to write the file
    ///
    fn save_as(&self, path: &str) -> PyResult<()> {
        self.0.save_as(path)?;
        Ok(())
    }
    /// Load a Model from a file
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     The path of the existing fit file
    ///
    /// Returns
    /// -------
    /// Model
    ///     The model contained in the file
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If anything fails when trying to read the file
    ///
    #[staticmethod]
    fn load_from(path: &str) -> PyResult<Self> {
        Ok(PyModel(Model::load_from(path)?))
    }
    #[new]
    fn new() -> Self {
        PyModel(Model::create_null())
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
        *self = PyModel(
            bincode::serde::decode_from_slice(state.as_bytes(), bincode::config::standard())
                .map_err(LadduError::DecodeError)?
                .0,
        );
        Ok(())
    }
}

/// An Amplitude which can be registered by a Manager
///
/// See Also
/// --------
/// laddu.Manager
///
#[pyclass(name = "Amplitude", module = "laddu")]
pub struct PyAmplitude(pub Box<dyn Amplitude>);

/// A class which can be used to evaluate a stored Expression
///
/// See Also
/// --------
/// laddu.Manager.load
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
    ///
    fn activate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            self.0.activate(&string_arg)?;
        } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            self.0.activate_many(&vec)?;
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
    ///
    fn deactivate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            self.0.deactivate(&string_arg)?;
        } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            self.0.deactivate_many(&vec)?;
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
    ///
    fn isolate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(string_arg) = arg.extract::<String>() {
            self.0.isolate(&string_arg)?;
        } else if let Ok(list_arg) = arg.downcast::<PyList>() {
            let vec: Vec<String> = list_arg.extract()?;
            self.0.isolate_many(&vec)?;
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
        parameters: Vec<Float>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<Complex<Float>>>> {
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
        parameters: Vec<Float>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<Complex<Float>>>> {
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
                            .collect::<Vec<Vec<Complex<Float>>>>()
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
                    .collect::<Vec<Vec<Complex<Float>>>>(),
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
pub fn py_parameter(name: &str) -> PyParameterLike {
    PyParameterLike(parameter(name))
}

/// A term which stays constant during an optimization
///
/// Parameters
/// ----------
/// value : float
///     The numerical value of the constant
///
/// Returns
/// -------
/// laddu.ParameterLike
///     An object that can be used as the input for many Amplitude constructors
///
#[pyfunction(name = "constant")]
pub fn py_constant(value: Float) -> PyParameterLike {
    PyParameterLike(constant(value))
}
