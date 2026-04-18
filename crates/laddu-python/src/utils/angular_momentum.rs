use laddu_core::{
    allowed_projections, helicity_combinations, AngularMomentum, AngularMomentumProjection,
    LadduError, LadduResult,
};
use num::rational::Ratio;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBool, PyModule},
    IntoPyObjectExt,
};

type PyQuantumNumber = Py<PyAny>;
type PyHelicityCombination = (PyQuantumNumber, PyQuantumNumber, PyQuantumNumber);

fn parse_angular_momentum(input: &Bound<'_, PyAny>) -> PyResult<AngularMomentum> {
    parse_ratio_like(input)
        .and_then(AngularMomentum::from_ratio)
        .map_err(py_value_error)
}

fn parse_ratio_like(input: &Bound<'_, PyAny>) -> LadduResult<Ratio<i32>> {
    if input.is_instance_of::<PyBool>() {
        return Err(LadduError::Custom(
            "quantum number cannot be a bool".to_string(),
        ));
    }
    if let Ok(value) = input.extract::<i32>() {
        return Ok(Ratio::from_integer(value));
    }
    if let Ok(value) = input.extract::<f64>() {
        let twice = AngularMomentumProjection::from_f64(value)?.value();
        return Ok(Ratio::new(twice, 2));
    }
    let numerator = input
        .getattr("numerator")
        .and_then(|value| value.extract::<i32>());
    let denominator = input
        .getattr("denominator")
        .and_then(|value| value.extract::<i32>());
    if let (Ok(numerator), Ok(denominator)) = (numerator, denominator) {
        if denominator == 0 {
            return Err(LadduError::Custom(
                "quantum number denominator cannot be zero".to_string(),
            ));
        }
        return Ok(Ratio::new(numerator, denominator));
    }
    Err(LadduError::Custom(
        "quantum number must be an int, float, or fractions.Fraction".to_string(),
    ))
}

fn py_value_error(err: LadduError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn projection_to_python(
    py: Python<'_>,
    projection: AngularMomentumProjection,
) -> PyResult<PyQuantumNumber> {
    let twice = projection.value();
    if twice % 2 == 0 {
        Ok((twice / 2).into_bound_py_any(py)?.unbind())
    } else {
        let fractions = PyModule::import(py, "fractions")?;
        let fraction = fractions.getattr("Fraction")?;
        Ok(fraction.call1((twice, 2))?.unbind())
    }
}

/// Enumerate allowed spin projections.
#[pyfunction(name = "allowed_projections")]
pub fn py_allowed_projections(
    py: Python<'_>,
    spin: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyQuantumNumber>> {
    allowed_projections(parse_angular_momentum(spin)?)
        .into_iter()
        .map(|projection| projection_to_python(py, projection))
        .collect()
}

/// Enumerate daughter helicity combinations.
#[pyfunction(name = "helicity_combinations")]
pub fn py_helicity_combinations(
    py: Python<'_>,
    spin_1: &Bound<'_, PyAny>,
    spin_2: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyHelicityCombination>> {
    helicity_combinations(
        parse_angular_momentum(spin_1)?,
        parse_angular_momentum(spin_2)?,
    )
    .into_iter()
    .map(|combination| {
        Ok((
            projection_to_python(py, combination.lambda_1())?,
            projection_to_python(py, combination.lambda_2())?,
            projection_to_python(py, combination.helicity())?,
        ))
    })
    .collect()
}
