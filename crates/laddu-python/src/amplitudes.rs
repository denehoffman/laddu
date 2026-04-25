use std::{array, collections::HashMap};

use laddu_amplitudes::{
    angular::{
        BlattWeisskopf, ClebschGordan, PhotonHelicity, PhotonPolarization, PhotonSDME, PolPhase,
        Wigner3j, WignerD, Ylm, Zlm,
    },
    kmatrix::{
        KopfKMatrixA0, KopfKMatrixA0Channel, KopfKMatrixA2, KopfKMatrixA2Channel, KopfKMatrixF0,
        KopfKMatrixF0Channel, KopfKMatrixF2, KopfKMatrixF2Channel, KopfKMatrixPi1,
        KopfKMatrixPi1Channel, KopfKMatrixRho, KopfKMatrixRhoChannel,
    },
    lookup::{LookupAxis, LookupTable},
    resonance::{BreitWigner, BreitWignerNonRelativistic, Flatte, PhaseSpaceFactor, Voigt},
    scalar::{ComplexScalar, PolarComplexScalar, Scalar, VariableScalar},
};
use laddu_core::{
    amplitudes::{Evaluator, Expression, Parameter, TestAmplitude},
    math::{BarrierKind, Sheet, QR_DEFAULT},
    traits::Variable,
    CompiledExpression, LadduError, LadduResult, ThreadPoolManager,
};
use num::complex::Complex64;
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBytes, PyList, PyTuple},
};

use crate::{
    data::PyDataset,
    quantum::angular_momentum::{
        parse_angular_momentum, parse_orbital_angular_momentum, parse_projection,
    },
    variables::{PyAngles, PyDecay, PyMandelstam, PyMass, PyPolarization, PyVariable},
};

type LookupInputs = (Vec<Box<dyn Variable>>, Vec<LookupAxis>);

macro_rules! py_kmatrix_channel {
    ($py_name:ident, $python_name:literal, $rust_name:path { $($variant:ident),+ $(,)? }) => {
        #[pyclass(eq, name = $python_name, module = "laddu", from_py_object)]
        #[derive(Clone, PartialEq)]
        pub enum $py_name {
            $($variant,)+
        }

        impl From<$py_name> for $rust_name {
            fn from(value: $py_name) -> Self {
                match value {
                    $( $py_name::$variant => Self::$variant, )+
                }
            }
        }
    };
}

py_kmatrix_channel!(
    PyKopfKMatrixA0Channel,
    "KopfKMatrixA0Channel",
    KopfKMatrixA0Channel { PiEta, KKbar }
);
py_kmatrix_channel!(
    PyKopfKMatrixA2Channel,
    "KopfKMatrixA2Channel",
    KopfKMatrixA2Channel {
        PiEta,
        KKbar,
        PiEtaPrime
    }
);
py_kmatrix_channel!(
    PyKopfKMatrixF0Channel,
    "KopfKMatrixF0Channel",
    KopfKMatrixF0Channel {
        PiPi,
        FourPi,
        KKbar,
        EtaEta,
        EtaEtaPrime
    }
);
py_kmatrix_channel!(
    PyKopfKMatrixF2Channel,
    "KopfKMatrixF2Channel",
    KopfKMatrixF2Channel {
        PiPi,
        FourPi,
        KKbar,
        EtaEta
    }
);
py_kmatrix_channel!(
    PyKopfKMatrixPi1Channel,
    "KopfKMatrixPi1Channel",
    KopfKMatrixPi1Channel { PiEta, PiEtaPrime }
);
py_kmatrix_channel!(
    PyKopfKMatrixRhoChannel,
    "KopfKMatrixRhoChannel",
    KopfKMatrixRhoChannel {
        PiPi,
        FourPi,
        KKbar
    }
);

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

/// Construct a scalar amplitude from one parameter.
#[pyfunction(name = "Scalar", signature = (name, value = None))]
pub fn py_scalar(name: &str, value: Option<PyParameter>) -> PyResult<PyExpression> {
    if let Some(value) = value {
        Ok(PyExpression(Scalar::new(name, value.0)?))
    } else {
        Ok(PyExpression(Scalar::new_auto(name)?))
    }
}

/// Construct a real expression from an event variable.
#[pyfunction(name = "VariableScalar")]
pub fn py_variable_scalar(name: &str, variable: Bound<'_, PyAny>) -> PyResult<PyExpression> {
    let variable = variable.extract::<PyVariable>()?;
    Ok(PyExpression(VariableScalar::new(name, &variable)?))
}

/// Construct a cartesian complex scalar amplitude.
#[pyfunction(name = "ComplexScalar", signature = (name, re_im = None))]
pub fn py_complex_scalar(
    name: &str,
    re_im: Option<(PyParameter, PyParameter)>,
) -> PyResult<PyExpression> {
    if let Some((re, im)) = re_im {
        Ok(PyExpression(ComplexScalar::new(name, re.0, im.0)?))
    } else {
        Ok(PyExpression(ComplexScalar::new_auto(name)?))
    }
}

/// Construct a polar complex scalar amplitude.
#[pyfunction(name = "PolarComplexScalar", signature = (name, r_theta = None))]
pub fn py_polar_complex_scalar(
    name: &str,
    r_theta: Option<(PyParameter, PyParameter)>,
) -> PyResult<PyExpression> {
    if let Some((r, theta)) = r_theta {
        Ok(PyExpression(PolarComplexScalar::new(name, r.0, theta.0)?))
    } else {
        Ok(PyExpression(PolarComplexScalar::new_auto(name)?))
    }
}

/// Construct a relativistic Breit-Wigner amplitude.
#[pyfunction(name = "BreitWigner", signature = (name, mass, width, l, daughter_1_mass, daughter_2_mass, resonance_mass, barrier_factors=true))]
#[allow(clippy::too_many_arguments)]
pub fn py_breit_wigner(
    name: &str,
    mass: PyParameter,
    width: PyParameter,
    l: usize,
    daughter_1_mass: &PyMass,
    daughter_2_mass: &PyMass,
    resonance_mass: &PyMass,
    barrier_factors: bool,
) -> PyResult<PyExpression> {
    if barrier_factors {
        Ok(PyExpression(BreitWigner::new(
            name,
            mass.0,
            width.0,
            l,
            &daughter_1_mass.0,
            &daughter_2_mass.0,
            &resonance_mass.0,
        )?))
    } else {
        Ok(PyExpression(BreitWigner::new_without_barrier_factors(
            name,
            mass.0,
            width.0,
            l,
            &daughter_1_mass.0,
            &daughter_2_mass.0,
            &resonance_mass.0,
        )?))
    }
}

/// Construct a non-relativistic Breit-Wigner amplitude.
#[pyfunction(name = "BreitWignerNonRelativistic")]
pub fn py_breit_wigner_non_relativistic(
    name: &str,
    mass: PyParameter,
    width: PyParameter,
    resonance_mass: &PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(BreitWignerNonRelativistic::new(
        name,
        mass.0,
        width.0,
        &resonance_mass.0,
    )?))
}

/// Construct a Flatte amplitude.
#[pyfunction(name = "Flatte")]
pub fn py_flatte(
    name: &str,
    mass: PyParameter,
    observed_channel_coupling: PyParameter,
    alternate_channel_coupling: PyParameter,
    observed_channel_daughter_masses: (PyMass, PyMass),
    alternate_channel_daughter_masses: (f64, f64),
    resonance_mass: &PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(Flatte::new(
        name,
        mass.0,
        observed_channel_coupling.0,
        alternate_channel_coupling.0,
        (
            &observed_channel_daughter_masses.0 .0,
            &observed_channel_daughter_masses.1 .0,
        ),
        alternate_channel_daughter_masses,
        &resonance_mass.0,
    )?))
}

/// Construct a Voigt amplitude.
#[pyfunction(name = "Voigt")]
pub fn py_voigt(
    name: &str,
    mass: PyParameter,
    width: PyParameter,
    sigma: PyParameter,
    resonance_mass: &PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(Voigt::new(
        name,
        mass.0,
        width.0,
        sigma.0,
        &resonance_mass.0,
    )?))
}

/// Construct a spherical-harmonic amplitude.
#[pyfunction(name = "Ylm")]
pub fn py_ylm(name: &str, l: usize, m: isize, angles: &PyAngles) -> PyResult<PyExpression> {
    Ok(PyExpression(Ylm::new(name, l, m, &angles.0)?))
}

/// Construct a polarized spherical-harmonic amplitude.
#[pyfunction(name = "Zlm")]
pub fn py_zlm(
    name: &str,
    l: usize,
    m: isize,
    r: &str,
    angles: &PyAngles,
    polarization: &PyPolarization,
) -> PyResult<PyExpression> {
    Ok(PyExpression(Zlm::new(
        name,
        l,
        m,
        r.parse()?,
        &angles.0,
        &polarization.0,
    )?))
}

/// Construct a polarization phase amplitude.
#[pyfunction(name = "PolPhase")]
pub fn py_polphase(name: &str, polarization: &PyPolarization) -> PyResult<PyExpression> {
    Ok(PyExpression(PolPhase::new(name, &polarization.0)?))
}

/// Construct a Wigner-D amplitude.
#[pyfunction(name = "WignerD")]
pub fn py_wigner_d(
    name: &str,
    spin: &Bound<'_, PyAny>,
    row_projection: &Bound<'_, PyAny>,
    column_projection: &Bound<'_, PyAny>,
    angles: &PyAngles,
) -> PyResult<PyExpression> {
    Ok(PyExpression(WignerD::new(
        name,
        parse_angular_momentum(spin)?,
        parse_projection(row_projection)?,
        parse_projection(column_projection)?,
        &angles.0,
    )?))
}

/// Construct a Blatt-Weisskopf amplitude.
#[pyfunction(name = "BlattWeisskopf", signature = (name, decay, l, reference_mass, q_r = QR_DEFAULT, sheet = "physical", kind = "full"))]
pub fn py_blatt_weisskopf(
    name: &str,
    decay: &PyDecay,
    l: &Bound<'_, PyAny>,
    reference_mass: f64,
    q_r: f64,
    sheet: &str,
    kind: &str,
) -> PyResult<PyExpression> {
    let sheet = match sheet.to_ascii_lowercase().as_str() {
        "physical" => Sheet::Physical,
        "unphysical" => Sheet::Unphysical,
        _ => {
            return Err(PyValueError::new_err(
                "sheet must be 'physical' or 'unphysical'",
            ));
        }
    };
    let kind = match kind.to_ascii_lowercase().as_str() {
        "full" => BarrierKind::Full,
        "tensor" => BarrierKind::Tensor,
        _ => {
            return Err(PyValueError::new_err("kind must be 'full' or 'tensor'"));
        }
    };
    Ok(PyExpression(BlattWeisskopf::new(
        name,
        &decay.0,
        parse_orbital_angular_momentum(l)?,
        reference_mass,
        q_r,
        sheet,
        kind,
    )?))
}

/// Construct a Clebsch-Gordan constant expression.
#[pyfunction(name = "ClebschGordan")]
pub fn py_clebsch_gordan(
    name: &str,
    j1: &Bound<'_, PyAny>,
    m1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    m2: &Bound<'_, PyAny>,
    j: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(ClebschGordan::new(
        name,
        parse_angular_momentum(j1)?,
        parse_projection(m1)?,
        parse_angular_momentum(j2)?,
        parse_projection(m2)?,
        parse_angular_momentum(j)?,
        parse_projection(m)?,
    )?))
}

/// Construct a Wigner-3j constant expression.
#[pyfunction(name = "Wigner3j")]
pub fn py_wigner_3j(
    name: &str,
    j1: &Bound<'_, PyAny>,
    m1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    m2: &Bound<'_, PyAny>,
    j3: &Bound<'_, PyAny>,
    m3: &Bound<'_, PyAny>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(Wigner3j::new(
        name,
        parse_angular_momentum(j1)?,
        parse_projection(m1)?,
        parse_angular_momentum(j2)?,
        parse_projection(m2)?,
        parse_angular_momentum(j3)?,
        parse_projection(m3)?,
    )?))
}

/// Construct a photon SDME amplitude.
#[pyfunction(name = "PhotonSDME", signature = (name, helicity, helicity_prime, polarization = None))]
pub fn py_photon_sdme(
    name: &str,
    helicity: i32,
    helicity_prime: i32,
    polarization: Option<&PyPolarization>,
) -> PyResult<PyExpression> {
    let polarization = polarization
        .map(|polarization| PhotonPolarization::Linear(Box::new(polarization.0.clone())))
        .unwrap_or(PhotonPolarization::Unpolarized);
    Ok(PyExpression(PhotonSDME::new(
        name,
        polarization,
        PhotonHelicity::new(helicity)?,
        PhotonHelicity::new(helicity_prime)?,
    )?))
}

/// Construct a phase-space factor amplitude.
#[pyfunction(name = "PhaseSpaceFactor")]
pub fn py_phase_space_factor(
    name: &str,
    recoil_mass: &PyMass,
    daughter_1_mass: &PyMass,
    daughter_2_mass: &PyMass,
    resonance_mass: &PyMass,
    mandelstam_s: &PyMandelstam,
) -> PyResult<PyExpression> {
    Ok(PyExpression(PhaseSpaceFactor::new(
        name,
        &recoil_mass.0,
        &daughter_1_mass.0,
        &daughter_2_mass.0,
        &resonance_mass.0,
        &mandelstam_s.0,
    )?))
}

fn py_lookup_inputs(
    variables: Vec<PyVariable>,
    axis_coordinates: Vec<Vec<f64>>,
) -> LadduResult<LookupInputs> {
    let axis_coordinates = axis_coordinates
        .into_iter()
        .map(LookupAxis::new)
        .collect::<LadduResult<Vec<_>>>()?;
    let variables = variables
        .into_iter()
        .map(|variable| Box::new(variable) as Box<dyn Variable>)
        .collect();
    Ok((variables, axis_coordinates))
}

/// Construct a fixed-complex lookup table amplitude.
#[pyfunction(name = "LookupTable", signature = (name, variables, axis_coordinates, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table(
    name: &str,
    variables: Vec<PyVariable>,
    axis_coordinates: Vec<Vec<f64>>,
    values: Vec<Complex64>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axis_coordinates) = py_lookup_inputs(variables, axis_coordinates)?;
    Ok(PyExpression(LookupTable::new(
        name,
        variables,
        axis_coordinates,
        values,
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

/// Construct a scalar-parameter lookup table amplitude.
#[pyfunction(name = "LookupTableScalar", signature = (name, variables, axis_coordinates, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table_scalar(
    name: &str,
    variables: Vec<PyVariable>,
    axis_coordinates: Vec<Vec<f64>>,
    values: Vec<PyParameter>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axis_coordinates) = py_lookup_inputs(variables, axis_coordinates)?;
    Ok(PyExpression(LookupTable::new_scalar(
        name,
        variables,
        axis_coordinates,
        values.into_iter().map(|value| value.0).collect(),
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

/// Construct a cartesian-complex lookup table amplitude.
#[pyfunction(name = "LookupTableComplex", signature = (name, variables, axis_coordinates, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table_complex(
    name: &str,
    variables: Vec<PyVariable>,
    axis_coordinates: Vec<Vec<f64>>,
    values: Vec<(PyParameter, PyParameter)>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axis_coordinates) = py_lookup_inputs(variables, axis_coordinates)?;
    Ok(PyExpression(LookupTable::new_cartesian_complex(
        name,
        variables,
        axis_coordinates,
        values
            .into_iter()
            .map(|(value_re, value_im)| (value_re.0, value_im.0))
            .collect(),
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

/// Construct a polar-complex lookup table amplitude.
#[pyfunction(name = "LookupTablePolar", signature = (name, variables, axis_coordinates, values, interpolation = "nearest", boundary_mode = "zero"))]
pub fn py_lookup_table_polar(
    name: &str,
    variables: Vec<PyVariable>,
    axis_coordinates: Vec<Vec<f64>>,
    values: Vec<(PyParameter, PyParameter)>,
    interpolation: &str,
    boundary_mode: &str,
) -> PyResult<PyExpression> {
    let (variables, axis_coordinates) = py_lookup_inputs(variables, axis_coordinates)?;
    Ok(PyExpression(LookupTable::new_polar_complex(
        name,
        variables,
        axis_coordinates,
        values
            .into_iter()
            .map(|(value_r, value_theta)| (value_r.0, value_theta.0))
            .collect(),
        interpolation.parse()?,
        boundary_mode.parse()?,
    )?))
}

/// Construct the fixed Kopf `a0` K-matrix amplitude.
#[pyfunction(name = "KopfKMatrixA0", signature = (name, couplings, channel, mass, seed = None))]
pub fn py_kopf_kmatrix_a0(
    name: &str,
    couplings: [[PyParameter; 2]; 2],
    channel: PyKopfKMatrixA0Channel,
    mass: PyMass,
    seed: Option<usize>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixA0::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel.into(),
        &mass.0,
        seed,
    )?))
}

/// Construct the fixed Kopf `a2` K-matrix amplitude.
#[pyfunction(name = "KopfKMatrixA2", signature = (name, couplings, channel, mass, seed = None))]
pub fn py_kopf_kmatrix_a2(
    name: &str,
    couplings: [[PyParameter; 2]; 2],
    channel: PyKopfKMatrixA2Channel,
    mass: PyMass,
    seed: Option<usize>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixA2::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel.into(),
        &mass.0,
        seed,
    )?))
}

/// Construct the fixed Kopf `f0` K-matrix amplitude.
#[pyfunction(name = "KopfKMatrixF0", signature = (name, couplings, channel, mass, seed = None))]
pub fn py_kopf_kmatrix_f0(
    name: &str,
    couplings: [[PyParameter; 2]; 5],
    channel: PyKopfKMatrixF0Channel,
    mass: PyMass,
    seed: Option<usize>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixF0::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel.into(),
        &mass.0,
        seed,
    )?))
}

/// Construct the fixed Kopf `f2` K-matrix amplitude.
#[pyfunction(name = "KopfKMatrixF2", signature = (name, couplings, channel, mass, seed = None))]
pub fn py_kopf_kmatrix_f2(
    name: &str,
    couplings: [[PyParameter; 2]; 4],
    channel: PyKopfKMatrixF2Channel,
    mass: PyMass,
    seed: Option<usize>,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixF2::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel.into(),
        &mass.0,
        seed,
    )?))
}

/// Construct the fixed Kopf `pi1` K-matrix amplitude.
#[pyfunction(name = "KopfKMatrixPi1")]
pub fn py_kopf_kmatrix_pi1(
    name: &str,
    couplings: [[PyParameter; 2]; 1],
    channel: PyKopfKMatrixPi1Channel,
    mass: PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixPi1::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel.into(),
        &mass.0,
    )?))
}

/// Construct the fixed Kopf `rho` K-matrix amplitude.
#[pyfunction(name = "KopfKMatrixRho")]
pub fn py_kopf_kmatrix_rho(
    name: &str,
    couplings: [[PyParameter; 2]; 2],
    channel: PyKopfKMatrixRhoChannel,
    mass: PyMass,
) -> PyResult<PyExpression> {
    Ok(PyExpression(KopfKMatrixRho::new(
        name,
        array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
        channel.into(),
        &mass.0,
    )?))
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
        Self(Expression::default())
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
