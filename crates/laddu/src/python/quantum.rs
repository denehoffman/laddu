use laddu_physics::quantum::{Isospin, J, L, M, MandelstamChannel, Parity, Statistics};
use num::rational::Ratio;
use pyo3::{
    class::basic::CompareOp,
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyAny,
};

use super::error::to_py_err;

fn ratio(value: &Bound<'_, PyAny>) -> PyResult<Option<Ratio<i64>>> {
    let Ok(numerator) = value.getattr("numerator") else {
        return Ok(None);
    };
    let Ok(denominator) = value.getattr("denominator") else {
        return Ok(None);
    };
    let numerator = numerator.extract::<i64>()?;
    let denominator = denominator.extract::<i64>()?;
    if denominator == 0 {
        return Err(PyValueError::new_err(
            "quantum-number denominator cannot be zero",
        ));
    }
    Ok(Some(Ratio::new(numerator, denominator)))
}

fn extract_j(value: &Bound<'_, PyAny>) -> PyResult<J> {
    if let Ok(value) = value.extract::<PyRef<'_, PyJ>>() {
        return Ok(value.inner);
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyS>>() {
        return Ok(value.inner);
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyL>>() {
        return Ok(value.inner.into());
    }
    if let Ok(value) = value.extract::<i64>() {
        return J::try_from(value).map_err(to_py_err);
    }
    if let Some(value) = ratio(value)? {
        return J::try_from(value).map_err(to_py_err);
    }
    if let Ok(value) = value.extract::<f64>() {
        return J::try_from(value).map_err(to_py_err);
    }
    Err(PyTypeError::new_err(
        "expected J, S, L, an integer, a half-integer float, or fractions.Fraction",
    ))
}

pub(crate) fn extract_l(value: &Bound<'_, PyAny>) -> PyResult<L> {
    if let Ok(value) = value.extract::<PyRef<'_, PyL>>() {
        return Ok(value.inner);
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyJ>>() {
        return L::try_from(value.inner).map_err(to_py_err);
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyS>>() {
        return L::try_from(value.inner).map_err(to_py_err);
    }
    if let Ok(value) = value.extract::<i64>() {
        return L::try_from(value).map_err(to_py_err);
    }
    if let Some(value) = ratio(value)? {
        return L::try_from(value).map_err(to_py_err);
    }
    if let Ok(value) = value.extract::<f64>() {
        return L::try_from(value).map_err(to_py_err);
    }
    Err(PyTypeError::new_err(
        "expected L, an integer, an integer-valued float, or fractions.Fraction",
    ))
}

fn extract_m(value: &Bound<'_, PyAny>) -> PyResult<M> {
    if let Ok(value) = value.extract::<PyRef<'_, PyM>>() {
        return Ok(value.inner);
    }
    if let Ok(value) = value.extract::<i64>() {
        return M::try_from(value).map_err(to_py_err);
    }
    if let Some(value) = ratio(value)? {
        return M::try_from(value).map_err(to_py_err);
    }
    if let Ok(value) = value.extract::<f64>() {
        return M::try_from(value).map_err(to_py_err);
    }
    Err(PyTypeError::new_err(
        "expected M, an integer, a half-integer float, or fractions.Fraction",
    ))
}

fn compare_i64(lhs: i64, rhs: i64, op: CompareOp) -> bool {
    match op {
        CompareOp::Lt => lhs < rhs,
        CompareOp::Le => lhs <= rhs,
        CompareOp::Eq => lhs == rhs,
        CompareOp::Ne => lhs != rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::Ge => lhs >= rhs,
    }
}

fn half_repr(name: &str, doubled: i64) -> String {
    if doubled % 2 == 0 {
        format!("{name}({})", doubled / 2)
    } else {
        format!("{name}({doubled}/2)")
    }
}

#[pyclass(name = "J", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// A nonnegative integer or half-integer total angular momentum.
///
/// Parameters
/// ----------
/// value : J, S, L, int, float, or fractions.Fraction
///     Angular momentum. Floating-point inputs must represent an exact integer
///     or half-integer.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> j = ld.J(1.5)
/// >>> j.multiplicity
/// 4
pub struct PyJ {
    pub(crate) inner: J,
}

#[pymethods]
impl PyJ {
    /// Construct a total angular momentum.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `value` is negative or not an integer or half-integer.
    #[new]
    #[pyo3(signature = (value: "J | S | L | int | float | fractions.Fraction"))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: extract_j(value)?,
        })
    }
    #[getter]
    /// float: Angular momentum as a Python floating-point value.
    fn value(&self) -> f64 {
        f64::from(self.inner)
    }
    #[getter]
    /// int: Twice the angular momentum, represented exactly.
    fn doubled(&self) -> u32 {
        self.inner.doubled()
    }
    #[getter]
    /// int: Number of magnetic substates, equal to ``2*j + 1``.
    fn multiplicity(&self) -> u32 {
        self.inner.multiplicity()
    }
    #[getter]
    /// bool: Whether the angular momentum is integral rather than half-integral.
    fn is_integer(&self) -> bool {
        self.inner.is_integer()
    }
    /// Return all magnetic projections from ``-j`` through ``+j``.
    fn projections(&self) -> Vec<PyM> {
        self.inner
            .projections()
            .into_iter()
            .map(PyM::from)
            .collect()
    }
    fn __float__(&self) -> f64 {
        f64::from(self.inner)
    }
    fn __repr__(&self) -> String {
        half_repr("J", i64::from(self.inner.doubled()))
    }
    fn __hash__(&self) -> u32 {
        self.inner.doubled()
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner + extract_j(other)?,
        })
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        Ok(compare_i64(
            i64::from(self.inner.doubled()),
            i64::from(extract_j(other)?.doubled()),
            op,
        ))
    }
}

#[pyclass(name = "S", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// A particle spin quantum number.
///
/// `S` has the same exact integer/half-integer representation and operations
/// as :class:`J`, but communicates that the value denotes intrinsic spin.
pub struct PyS {
    pub(crate) inner: J,
}

#[pymethods]
impl PyS {
    /// Construct a spin from an integer or half-integer value.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `value` is negative or not an integer or half-integer.
    #[new]
    #[pyo3(signature = (value: "J | S | L | int | float | fractions.Fraction"))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: extract_j(value)?,
        })
    }
    #[getter]
    /// float: Spin as a Python floating-point value.
    fn value(&self) -> f64 {
        f64::from(self.inner)
    }
    #[getter]
    /// int: Twice the spin, represented exactly.
    fn doubled(&self) -> u32 {
        self.inner.doubled()
    }
    #[getter]
    /// int: Spin multiplicity ``2*S + 1``.
    fn multiplicity(&self) -> u32 {
        self.inner.multiplicity()
    }
    #[getter]
    /// bool: Whether the spin is integral.
    fn is_integer(&self) -> bool {
        self.inner.is_integer()
    }
    /// Return all allowed spin projections.
    fn projections(&self) -> Vec<PyM> {
        self.inner
            .projections()
            .into_iter()
            .map(PyM::from)
            .collect()
    }
    fn __float__(&self) -> f64 {
        f64::from(self.inner)
    }
    fn __repr__(&self) -> String {
        half_repr("S", i64::from(self.inner.doubled()))
    }
    fn __hash__(&self) -> u32 {
        self.inner.doubled()
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner + extract_j(other)?,
        })
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        Ok(compare_i64(
            i64::from(self.inner.doubled()),
            i64::from(extract_j(other)?.doubled()),
            op,
        ))
    }
}

#[pyclass(name = "L", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// A nonnegative integral orbital angular momentum.
///
/// String conversion uses spectroscopic notation (S, P, D, ...), while
/// :attr:`value` gives the corresponding integer.
pub struct PyL {
    pub(crate) inner: L,
}

#[pymethods]
impl PyL {
    /// Construct an orbital angular momentum.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `value` is negative or nonintegral.
    #[new]
    #[pyo3(signature = (value: "L | J | S | int | float | fractions.Fraction"))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: extract_l(value)?,
        })
    }
    #[getter]
    /// int: Orbital angular momentum.
    fn value(&self) -> u32 {
        self.inner.value()
    }
    #[getter]
    /// int: Number of projections, equal to ``2*L + 1``.
    fn multiplicity(&self) -> u32 {
        self.inner.multiplicity()
    }
    #[getter]
    /// Parity: Orbital parity ``(-1)**L``.
    fn parity(&self) -> PyParity {
        self.inner.orbital_parity().into()
    }
    /// Return the integral projections from ``-L`` through ``+L``.
    fn projections(&self) -> Vec<PyM> {
        self.inner
            .projections()
            .into_iter()
            .map(PyM::from)
            .collect()
    }
    fn __float__(&self) -> f64 {
        f64::from(self.inner)
    }
    fn __repr__(&self) -> String {
        format!("L({})", self.inner.value())
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __hash__(&self) -> u32 {
        self.inner.value()
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        Ok(compare_i64(
            i64::from(self.inner.value()),
            i64::from(extract_l(other)?.value()),
            op,
        ))
    }
}

#[pyclass(name = "M", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// An integer or half-integer angular-momentum projection.
///
/// Unlike :class:`J`, projections may be negative.
pub struct PyM {
    pub(crate) inner: M,
}

impl From<M> for PyM {
    fn from(inner: M) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyM {
    /// Construct an angular-momentum projection.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `value` is not an integer or half-integer.
    #[new]
    #[pyo3(signature = (value: "M | int | float | fractions.Fraction"))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: extract_m(value)?,
        })
    }
    #[getter]
    /// float: Projection as a Python floating-point value.
    fn value(&self) -> f64 {
        f64::from(self.inner)
    }
    #[getter]
    /// int: Twice the projection, represented exactly.
    fn doubled(&self) -> i32 {
        self.inner.doubled()
    }
    #[getter]
    /// bool: Whether the projection is integral.
    fn is_integer(&self) -> bool {
        self.inner.is_integer()
    }
    fn __float__(&self) -> f64 {
        f64::from(self.inner)
    }
    fn __repr__(&self) -> String {
        half_repr("M", i64::from(self.inner.doubled()))
    }
    fn __hash__(&self) -> i32 {
        self.inner.doubled()
    }
    fn __neg__(&self) -> Self {
        Self { inner: -self.inner }
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner + extract_m(other)?,
        })
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner - extract_m(other)?,
        })
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        Ok(compare_i64(
            i64::from(self.inner.doubled()),
            i64::from(extract_m(other)?.doubled()),
            op,
        ))
    }
}

#[pyclass(name = "Parity", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// A positive or negative parity eigenvalue.
///
/// Parameters
/// ----------
/// value : Parity, int, or str
///     ``+1``/``-1`` or a recognized parity name.
///
/// Attributes
/// ----------
/// POSITIVE, NEGATIVE : Parity
///     Canonical parity values.
pub struct PyParity {
    pub(crate) inner: Parity,
}

impl From<Parity> for PyParity {
    fn from(inner: Parity) -> Self {
        Self { inner }
    }
}

/// Convert a Python parity representation to the Rust parity value.
///
/// Raises
/// ------
/// TypeError
///     If `value` is not a parity, integer, or string.
/// ValueError
///     If an integer is not ``+1`` or ``-1``, or a string is unrecognized.
pub fn extract_parity(value: &Bound<'_, PyAny>) -> PyResult<Parity> {
    if let Ok(value) = value.extract::<PyRef<'_, PyParity>>() {
        return Ok(value.inner);
    }
    if let Ok(value) = value.extract::<i8>() {
        return match value {
            1 => Ok(Parity::Positive),
            -1 => Ok(Parity::Negative),
            _ => Err(PyValueError::new_err("parity must be +1 or -1")),
        };
    }
    if let Ok(value) = value.extract::<String>() {
        return value.parse().map_err(to_py_err);
    }
    Err(PyTypeError::new_err(
        "expected Parity, +1, -1, or a parity name",
    ))
}

#[pymethods]
impl PyParity {
    /// Construct a parity eigenvalue.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `value` has an unsupported type.
    /// ValueError
    ///     If `value` is not a recognized parity.
    #[new]
    #[pyo3(signature = (value: "Parity | int | str"))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(extract_parity(value)?.into())
    }
    #[classattr]
    #[pyo3(name = "POSITIVE")]
    /// Positive parity, represented by ``+1``.
    fn positive() -> Self {
        Parity::Positive.into()
    }
    #[classattr]
    #[pyo3(name = "NEGATIVE")]
    /// Negative parity, represented by ``-1``.
    fn negative() -> Self {
        Parity::Negative.into()
    }
    #[getter]
    /// int: The parity eigenvalue, either ``+1`` or ``-1``.
    fn value(&self) -> i32 {
        self.inner.value()
    }
    fn __int__(&self) -> i32 {
        self.inner.value()
    }
    fn __repr__(&self) -> &'static str {
        match self.inner {
            Parity::Positive => "Parity.POSITIVE",
            Parity::Negative => "Parity.NEGATIVE",
        }
    }
    fn __hash__(&self) -> i32 {
        self.inner.value()
    }
    fn __neg__(&self) -> Self {
        (-self.inner).into()
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok((self.inner * extract_parity(other)?).into())
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        let rhs = extract_parity(other)?.value();
        Ok(compare_i64(
            i64::from(self.inner.value()),
            i64::from(rhs),
            op,
        ))
    }
}

#[pyclass(name = "Isospin", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A total isospin with an optional third-component projection.
///
/// Parameters
/// ----------
/// isospin : J, S, L, int, float, or fractions.Fraction
///     Nonnegative total isospin.
/// projection : M, int, float, or fractions.Fraction, optional
///     Third component. When supplied, it must lie in ``[-I, I]`` and have
///     compatible integer/half-integer parity.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> proton_isospin = ld.Isospin(0.5, 0.5)
pub struct PyIsospin {
    pub(crate) inner: Isospin,
}

#[pymethods]
impl PyIsospin {
    /// Construct an isospin assignment.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a quantum number cannot be converted.
    /// LadduError
    ///     If the projection is incompatible with the total isospin.
    #[new]
    #[pyo3(signature = (
        isospin: "J | S | L | int | float | fractions.Fraction",
        projection: "M | int | float | fractions.Fraction | None"=None
    ))]
    fn new(isospin: &Bound<'_, PyAny>, projection: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let isospin = extract_j(isospin)?;
        let projection = projection.map(extract_m).transpose()?;
        Ok(Self {
            inner: Isospin::new(isospin, projection).map_err(to_py_err)?,
        })
    }
    #[getter]
    /// J: Total isospin quantum number.
    fn isospin(&self) -> PyJ {
        PyJ {
            inner: self.inner.isospin,
        }
    }
    #[getter]
    /// M: Isospin projection.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If this value was created without a projection.
    fn projection(&self) -> PyResult<PyM> {
        Ok(self.inner.projection().map_err(to_py_err)?.into())
    }
    #[getter]
    /// M or None: Projection without requiring that one is present.
    fn projection_checked(&self) -> Option<PyM> {
        self.inner.projection.map(PyM::from)
    }
    fn __repr__(&self) -> String {
        match self.inner.projection {
            Some(projection) => format!(
                "Isospin({}, {})",
                half_repr("J", i64::from(self.inner.isospin.doubled())),
                half_repr("M", i64::from(projection.doubled()))
            ),
            None => format!(
                "Isospin({})",
                half_repr("J", i64::from(self.inner.isospin.doubled()))
            ),
        }
    }
    fn __hash__(&self) -> isize {
        let projection = self
            .inner
            .projection
            .map_or(i64::MIN, |value| i64::from(value.doubled()));
        (i64::from(self.inner.isospin.doubled()).wrapping_mul(31) ^ projection) as isize
    }
    fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.inner == other.inner,
            CompareOp::Ne => self.inner != other.inner,
            _ => false,
        }
    }
}

#[pyclass(name = "Statistics", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// Bose-Einstein or Fermi-Dirac particle statistics.
///
/// Attributes
/// ----------
/// BOSON, FERMION : Statistics
///     Canonical statistics values.
pub struct PyStatistics {
    pub(crate) inner: Statistics,
}

#[pyclass(
    name = "MandelstamChannel",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Copy)]
/// One of the Mandelstam ``s``, ``t``, or ``u`` scattering channels.
///
/// Parameters
/// ----------
/// value : str
///     Case-insensitive channel name.
///
/// Attributes
/// ----------
/// S, T, U : MandelstamChannel
///     Canonical channel values.
pub struct PyMandelstamChannel {
    pub(crate) inner: MandelstamChannel,
}

#[pymethods]
impl PyMandelstamChannel {
    /// Parse a Mandelstam channel name.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If `value` is not ``"s"``, ``"t"``, or ``"u"``.
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: value.parse().map_err(to_py_err)?,
        })
    }
    #[classattr]
    #[pyo3(name = "S")]
    /// Center-of-mass energy channel.
    fn s() -> Self {
        Self {
            inner: MandelstamChannel::S,
        }
    }
    #[classattr]
    #[pyo3(name = "T")]
    /// First momentum-transfer channel.
    fn t() -> Self {
        Self {
            inner: MandelstamChannel::T,
        }
    }
    #[classattr]
    #[pyo3(name = "U")]
    /// Crossed momentum-transfer channel.
    fn u() -> Self {
        Self {
            inner: MandelstamChannel::U,
        }
    }
    fn __repr__(&self) -> &'static str {
        match self.inner {
            MandelstamChannel::S => "MandelstamChannel.S",
            MandelstamChannel::T => "MandelstamChannel.T",
            MandelstamChannel::U => "MandelstamChannel.U",
        }
    }
    fn __hash__(&self) -> u8 {
        match self.inner {
            MandelstamChannel::S => 0,
            MandelstamChannel::T => 1,
            MandelstamChannel::U => 2,
        }
    }
    fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.inner == other.inner,
            CompareOp::Ne => self.inner != other.inner,
            _ => false,
        }
    }
}

impl From<Statistics> for PyStatistics {
    fn from(inner: Statistics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStatistics {
    /// Parse ``"boson"`` or ``"fermion"``.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the name is not recognized.
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        Ok(value.parse::<Statistics>().map_err(to_py_err)?.into())
    }
    #[classattr]
    #[pyo3(name = "BOSON")]
    /// Bose-Einstein statistics.
    fn boson() -> Self {
        Statistics::Boson.into()
    }
    #[classattr]
    #[pyo3(name = "FERMION")]
    /// Fermi-Dirac statistics.
    fn fermion() -> Self {
        Statistics::Fermion.into()
    }
    #[staticmethod]
    #[pyo3(signature = (spin: "J | S | L | int | float | fractions.Fraction"))]
    /// Determine statistics from the spin-statistics relation.
    ///
    /// Integral spins produce :attr:`BOSON`; half-integral spins produce
    /// :attr:`FERMION`.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `spin` cannot be converted to an angular momentum.
    fn from_spin(spin: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Statistics::from_spin(extract_j(spin)?).into())
    }
    fn __repr__(&self) -> &'static str {
        match self.inner {
            Statistics::Boson => "Statistics.BOSON",
            Statistics::Fermion => "Statistics.FERMION",
        }
    }
    fn __hash__(&self) -> u8 {
        match self.inner {
            Statistics::Boson => 0,
            Statistics::Fermion => 1,
        }
    }
    fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> bool {
        let lhs = match self.inner {
            Statistics::Boson => 0,
            Statistics::Fermion => 1,
        };
        let rhs = match other.inner {
            Statistics::Boson => 0,
            Statistics::Fermion => 1,
        };
        compare_i64(lhs, rhs, op)
    }
}

/// Convert a Python spin-like value to an exact Rust angular momentum.
///
/// Raises
/// ------
/// TypeError
///     If `value` has an unsupported type.
/// ValueError
///     If `value` is negative or not an integer or half-integer.
pub fn extract_spin(value: &Bound<'_, PyAny>) -> PyResult<J> {
    extract_j(value)
}
/// Convert a Python projection-like value to an exact Rust projection.
///
/// Raises
/// ------
/// TypeError
///     If `value` has an unsupported type.
/// ValueError
///     If `value` is not an integer or half-integer.
pub fn extract_projection(value: &Bound<'_, PyAny>) -> PyResult<M> {
    extract_m(value)
}
