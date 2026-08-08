use laddu_physics::quantum::{
    AllowedPartialWave, Isospin, J, L, M, MandelstamChannel, Parity, PartialWave, RuleCheck,
    RuleKind, RuleOutcome, RuleReport, RuleSet, SelectionRules, Statistics, UnknownPolicy,
};
use num::rational::Ratio;
use pyo3::{
    class::basic::CompareOp,
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyAny,
};

use super::error::to_py_err;
use super::particle::PyParticle;

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
    #[pyo3(signature = (value: "J | S | L | int | float"))]
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
    /// Return every total angular momentum obtainable by coupling with `other`.
    ///
    /// Examples
    /// --------
    /// >>> import laddu as ld
    /// >>> ld.J(0.5).coupled_with(ld.S(1))
    /// [J(1/2), J(3/2)]
    #[pyo3(signature = (other: "J | S | L | int | float"))]
    fn coupled_with(&self, other: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        Ok(self
            .inner
            .coupled_with(extract_j(other)?)
            .into_iter()
            .map(|inner| Self { inner })
            .collect())
    }
    /// Return whether `first` and `second` can couple to this total angular momentum.
    #[pyo3(signature = (
        first: "J | S | L | int | float",
        second: "J | S | L | int | float"
    ))]
    fn can_couple_to(&self, first: &Bound<'_, PyAny>, second: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self
            .inner
            .can_couple_to(extract_j(first)?, extract_j(second)?))
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
    #[pyo3(signature = (value: "J | S | L | int | float"))]
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
    /// Return every total angular momentum obtainable by coupling with `other`.
    #[pyo3(signature = (other: "J | S | L | int | float"))]
    fn coupled_with(&self, other: &Bound<'_, PyAny>) -> PyResult<Vec<PyJ>> {
        Ok(self
            .inner
            .coupled_with(extract_j(other)?)
            .into_iter()
            .map(|inner| PyJ { inner })
            .collect())
    }
    /// Return whether `first` and `second` can couple to this spin.
    #[pyo3(signature = (
        first: "J | S | L | int | float",
        second: "J | S | L | int | float"
    ))]
    fn can_couple_to(&self, first: &Bound<'_, PyAny>, second: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self
            .inner
            .can_couple_to(extract_j(first)?, extract_j(second)?))
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
    #[pyo3(signature = (value: "L | J | S | int | float"))]
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
    #[pyo3(signature = (value: "M | int | float"))]
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
/// >>> proton_isospin = ld.Isospin(0.5, projection=0.5)
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
        isospin: "J | S | L | int | float",
        *,
        projection: "M | int | float | None"=None
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
    #[pyo3(signature = (spin: "J | S | L | int | float"))]
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

fn extract_rule_kind(name: &str) -> PyResult<RuleKind> {
    let normalized = name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let rule = match normalized.as_str() {
        "parity" => RuleKind::Parity,
        "isospin" => RuleKind::Isospin,
        "isospinprojection" | "i3" => RuleKind::IsospinProjection,
        "cparity" => RuleKind::CParity,
        "gparity" => RuleKind::GParity,
        "charge" => RuleKind::Charge,
        "strangeness" => RuleKind::Strangeness,
        "charm" => RuleKind::Charm,
        "bottomness" => RuleKind::Bottomness,
        "topness" => RuleKind::Topness,
        "baryonnumber" => RuleKind::BaryonNumber,
        "electronleptonnumber" => RuleKind::ElectronLeptonNumber,
        "muonleptonnumber" => RuleKind::MuonLeptonNumber,
        "tauleptonnumber" => RuleKind::TauLeptonNumber,
        "leptonnumber" => RuleKind::LeptonNumber,
        "identicalparticlesymmetry" => RuleKind::IdenticalParticleSymmetry,
        "conventionalmesonjpc" => RuleKind::ConventionalMesonJpc,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown quantum rule `{name}`"
            )));
        }
    };
    Ok(rule)
}

fn rule_name(rule: RuleKind) -> String {
    format!("{rule:?}")
}

fn unknown_policy(name: &str) -> PyResult<UnknownPolicy> {
    match name.to_ascii_lowercase().as_str() {
        "allow" => Ok(UnknownPolicy::Allow),
        "reject" => Ok(UnknownPolicy::Reject),
        "warn" => Ok(UnknownPolicy::Warn),
        _ => Err(PyValueError::new_err(
            "unknown policy must be 'allow', 'reject', or 'warn'",
        )),
    }
}

/// Result of applying one quantum-number selection rule.
#[pyclass(name = "RuleCheck", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRuleCheck {
    inner: RuleCheck,
}

#[pymethods]
impl PyRuleCheck {
    #[getter]
    /// str: Name of the rule.
    fn rule(&self) -> String {
        rule_name(self.inner.rule)
    }

    #[getter]
    /// str: Outcome category.
    fn outcome(&self) -> &'static str {
        match self.inner.outcome {
            RuleOutcome::Pass { .. } => "pass",
            RuleOutcome::Fail { .. } => "fail",
            RuleOutcome::UnknownAllowed { .. } => "unknown_allowed",
            RuleOutcome::Warning { .. } => "warning",
            RuleOutcome::Ignored { .. } => "ignored",
            RuleOutcome::Diagnostic { .. } => "diagnostic",
        }
    }

    #[getter]
    /// str or None: Human-readable explanation.
    fn message(&self) -> Option<String> {
        match &self.inner.outcome {
            RuleOutcome::Pass { message }
            | RuleOutcome::Fail { message }
            | RuleOutcome::UnknownAllowed { message, .. }
            | RuleOutcome::Warning { message, .. }
            | RuleOutcome::Diagnostic { message, .. } => Some(message.clone()),
            RuleOutcome::Ignored { .. } => None,
        }
    }

    #[getter]
    /// list[str]: Inputs that were unavailable during the check.
    fn missing(&self) -> Vec<String> {
        match &self.inner.outcome {
            RuleOutcome::UnknownAllowed { missing, .. } | RuleOutcome::Warning { missing, .. } => {
                missing.clone()
            }
            _ => Vec::new(),
        }
    }

    #[getter]
    /// bool: Whether this check rejects the channel.
    fn is_failure(&self) -> bool {
        self.inner.is_failure()
    }

    fn __repr__(&self) -> String {
        format!(
            "RuleCheck(rule='{}', outcome='{}')",
            self.rule(),
            self.outcome()
        )
    }
}

/// Complete report from a quantum-number rule set.
#[pyclass(name = "RuleReport", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRuleReport {
    inner: RuleReport,
}

#[pymethods]
impl PyRuleReport {
    #[getter]
    /// bool: Whether no enforced rule failed.
    fn is_allowed(&self) -> bool {
        self.inner.is_allowed()
    }

    #[getter]
    /// list[RuleCheck]: Checks in deterministic rule order.
    fn checks(&self) -> Vec<PyRuleCheck> {
        self.inner
            .checks
            .iter()
            .cloned()
            .map(|inner| PyRuleCheck { inner })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __bool__(&self) -> bool {
        self.inner.is_allowed()
    }
}

/// Configurable quantum-number rules for a two-body decay.
#[pyclass(name = "RuleSet", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRuleSet {
    inner: RuleSet,
}

#[pymethods]
impl PyRuleSet {
    #[staticmethod]
    /// Return a rule set containing only angular-momentum coupling.
    fn angular() -> Self {
        Self {
            inner: RuleSet::angular(),
        }
    }

    #[staticmethod]
    /// Return the standard strong-interaction rule set.
    fn strong() -> Self {
        Self {
            inner: RuleSet::strong(),
        }
    }

    #[staticmethod]
    /// Return the standard electromagnetic-interaction rule set.
    fn electromagnetic() -> Self {
        Self {
            inner: RuleSet::electromagnetic(),
        }
    }

    #[staticmethod]
    /// Return the standard weak-interaction rule set.
    fn weak() -> Self {
        Self {
            inner: RuleSet::weak(),
        }
    }

    /// Return a copy with `rule` enforced.
    fn enforce(&self, rule: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().enforce(extract_rule_kind(rule)?),
        })
    }

    /// Return a copy with `rule` enforced and missing inputs rejected.
    fn enforce_strict(&self, rule: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().enforce_strict(extract_rule_kind(rule)?),
        })
    }

    /// Return a copy with `rule` disabled.
    fn disable(&self, rule: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().disable(extract_rule_kind(rule)?),
        })
    }

    /// Return a copy with the missing-input policy for `rule` changed.
    fn with_unknown_policy(&self, rule: &str, policy: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_unknown_policy(extract_rule_kind(rule)?, unknown_policy(policy)?),
        })
    }

    #[getter]
    /// list[str]: Enabled rule names.
    fn enabled_rules(&self) -> Vec<String> {
        self.inner.enabled_rules().map(rule_name).collect()
    }

    #[pyo3(signature = (
        parent,
        *,
        daughter_a,
        daughter_b,
        l: "L | J | S | int | float",
        s: "J | S | L | int | float"
    ))]
    /// Evaluate this rule set for ``parent -> daughter_a daughter_b``.
    fn evaluate(
        &self,
        parent: &PyParticle,
        daughter_a: &PyParticle,
        daughter_b: &PyParticle,
        l: &Bound<'_, PyAny>,
        s: &Bound<'_, PyAny>,
    ) -> PyResult<PyRuleReport> {
        Ok(PyRuleReport {
            inner: self.inner.evaluate(
                &parent.inner,
                (&daughter_a.inner, &daughter_b.inner),
                extract_l(l)?,
                extract_spin(s)?,
            ),
        })
    }
}

/// A coupled ``JLS`` partial wave.
#[pyclass(name = "PartialWave", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct PyPartialWave {
    inner: PartialWave,
}

#[pymethods]
impl PyPartialWave {
    #[new]
    #[pyo3(signature = (
        j: "J | S | L | int | float",
        l: "L | J | S | int | float",
        s: "J | S | L | int | float"
    ))]
    /// Construct and validate a coupled ``JLS`` partial wave.
    fn new(j: &Bound<'_, PyAny>, l: &Bound<'_, PyAny>, s: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: PartialWave::new(extract_j(j)?, extract_l(l)?, extract_spin(s)?)
                .map_err(to_py_err)?,
        })
    }

    #[getter]
    /// J: Total angular momentum.
    fn j(&self) -> PyJ {
        PyJ {
            inner: self.inner.j,
        }
    }

    #[getter]
    /// L: Orbital angular momentum.
    fn l(&self) -> PyL {
        PyL {
            inner: self.inner.l,
        }
    }

    #[getter]
    /// S: Coupled daughter spin.
    fn s(&self) -> PyS {
        PyS {
            inner: self.inner.s,
        }
    }

    #[getter]
    /// str: Spectroscopic label such as ``1S0``.
    fn label(&self) -> String {
        self.inner.label()
    }

    fn __repr__(&self) -> String {
        format!("PartialWave('{}')", self.inner.label())
    }
}

/// A partial wave with channel-dependent inferred parity quantum numbers.
#[pyclass(
    name = "AllowedPartialWave",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyAllowedPartialWave {
    inner: AllowedPartialWave,
}

#[pymethods]
impl PyAllowedPartialWave {
    #[getter]
    /// PartialWave: Coupled angular quantum numbers.
    fn wave(&self) -> PyPartialWave {
        PyPartialWave {
            inner: self.inner.wave,
        }
    }

    #[getter]
    /// Parity or None: Inferred final-state parity.
    fn parity(&self) -> Option<PyParity> {
        self.inner.parity.map(Into::into)
    }

    #[getter]
    /// Parity or None: Inferred final-state C parity.
    fn c_parity(&self) -> Option<PyParity> {
        self.inner.c_parity.map(Into::into)
    }
}

/// Generator and filter for allowed two-body partial waves.
#[pyclass(name = "SelectionRules", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PySelectionRules {
    inner: SelectionRules,
}

#[pymethods]
impl PySelectionRules {
    #[new]
    #[pyo3(signature = (
        rules,
        max_l: "L | J | S | int | float"
    ))]
    /// Construct from a rule set and maximum orbital angular momentum.
    fn new(rules: &PyRuleSet, max_l: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: SelectionRules::new(rules.inner.clone(), extract_l(max_l)?),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (max_l: "L | J | S | int | float"))]
    /// Construct strong-interaction selection rules.
    fn strong(max_l: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: SelectionRules::strong(extract_l(max_l)?),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (max_l: "L | J | S | int | float"))]
    /// Construct electromagnetic selection rules.
    fn electromagnetic(max_l: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: SelectionRules::electromagnetic(extract_l(max_l)?),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (max_l: "L | J | S | int | float"))]
    /// Construct weak-interaction selection rules.
    fn weak(max_l: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: SelectionRules::weak(extract_l(max_l)?),
        })
    }

    /// Return all allowed waves for ``parent -> daughter_a daughter_b``.
    #[pyo3(signature = (parent, *, daughter_a, daughter_b))]
    fn allowed_partial_waves(
        &self,
        parent: &PyParticle,
        daughter_a: &PyParticle,
        daughter_b: &PyParticle,
    ) -> Vec<PyAllowedPartialWave> {
        self.inner
            .allowed_partial_waves(&parent.inner, (&daughter_a.inner, &daughter_b.inner))
            .into_iter()
            .map(|inner| PyAllowedPartialWave { inner })
            .collect()
    }
}
