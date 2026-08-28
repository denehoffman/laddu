use laddu_physics::math::{
    BarrierKind, Sheet, blatt_weisskopf as rust_blatt_weisskopf,
    blatt_weisskopf_custom as rust_blatt_weisskopf_custom, chew_mandelstam as rust_chew_mandelstam,
    q as rust_q, rho as rust_rho, spherical_harmonic as rust_spherical_harmonic,
};
use pyo3::{prelude::*, types::PyAny};

use super::{
    error::to_py_err,
    expr::{PyExpr, extract_expr},
    quantum::{PyL, extract_projection},
};

#[pyclass(name = "Sheet", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// Riemann sheet used by analytically continued two-body functions.
///
/// Attributes
/// ----------
/// PHYSICAL : Sheet
///     Physical sheet with the conventional threshold branch.
/// UNPHYSICAL : Sheet
///     Adjacent unphysical sheet used for resonance continuation.
pub struct PySheet {
    inner: Sheet,
}

#[pymethods]
impl PySheet {
    #[classattr]
    #[pyo3(name = "PHYSICAL")]
    /// Physical Riemann sheet.
    fn physical() -> Self {
        Self {
            inner: Sheet::Physical,
        }
    }
    #[classattr]
    #[pyo3(name = "UNPHYSICAL")]
    /// Adjacent unphysical Riemann sheet.
    fn unphysical() -> Self {
        Self {
            inner: Sheet::Unphysical,
        }
    }
    fn __repr__(&self) -> &'static str {
        match self.inner {
            Sheet::Physical => "Sheet.PHYSICAL",
            Sheet::Unphysical => "Sheet.UNPHYSICAL",
        }
    }
}

#[pyclass(name = "BarrierKind", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
/// Normalization convention for Blatt-Weisskopf barrier factors.
///
/// Attributes
/// ----------
/// FULL : BarrierKind
///     Full centrifugal-barrier factor.
/// TENSOR : BarrierKind
///     Tensor-form-factor convention.
pub struct PyBarrierKind {
    inner: BarrierKind,
}

#[pymethods]
impl PyBarrierKind {
    #[classattr]
    #[pyo3(name = "FULL")]
    /// Full centrifugal-barrier convention.
    fn full() -> Self {
        Self {
            inner: BarrierKind::Full,
        }
    }
    #[classattr]
    #[pyo3(name = "TENSOR")]
    /// Tensor-form-factor convention.
    fn tensor() -> Self {
        Self {
            inner: BarrierKind::Tensor,
        }
    }
    fn __repr__(&self) -> &'static str {
        match self.inner {
            BarrierKind::Full => "BarrierKind.FULL",
            BarrierKind::Tensor => "BarrierKind.TENSOR",
        }
    }
}

#[pyfunction]
/// Construct a complex spherical-harmonic expression.
///
/// Parameters
/// ----------
/// l : L or int
///     Nonnegative orbital angular momentum.
/// m : M or int
///     Magnetic projection satisfying ``-l <= m <= l``.
/// costheta : Expr or number
///     Cosine of the polar angle.
/// phi : Expr or number
///     Azimuthal angle in radians.
///
/// Returns
/// -------
/// Expr
///     Complex Condon-Shortley spherical harmonic ``Y_l^m``.
///
/// Raises
/// ------
/// TypeError
///     If a quantum number or angle cannot be converted.
/// LadduError
///     If `l` and `m` are incompatible.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> costheta = ld.scalar("costheta")
/// >>> amplitude = ld.spherical_harmonic(1, 0, costheta=costheta, phi=0.0)
#[pyo3(signature = (
    l,
    m: "M | int | float",
    *,
    costheta: "Expr | int | float",
    phi: "Expr | int | float"
))]
pub fn spherical_harmonic(
    l: PyL,
    m: &Bound<'_, PyAny>,
    costheta: &Bound<'_, PyAny>,
    phi: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    rust_spherical_harmonic(
        l.inner,
        extract_projection(m)?,
        extract_expr(costheta)?,
        extract_expr(phi)?,
    )
    .map(PyExpr::from)
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    *,
    mass1: "Expr | int | float",
    mass2: "Expr | int | float",
    sheet=None
))]
/// Construct the two-body breakup momentum.
///
/// Parameters
/// ----------
/// s : Expr or number
///     Squared invariant mass of the two-body system.
/// mass1, mass2 : Expr or number
///     Daughter masses.
/// sheet : Sheet, optional
///     Analytic-continuation sheet; defaults to :attr:`Sheet.PHYSICAL`.
///
/// Returns
/// -------
/// Expr
///     Complex breakup momentum
///     ``sqrt(lambda(s, mass1**2, mass2**2) / (4*s))``.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted to an expression.
pub fn q(
    s: &Bound<'_, PyAny>,
    mass1: &Bound<'_, PyAny>,
    mass2: &Bound<'_, PyAny>,
    sheet: Option<&PySheet>,
) -> PyResult<PyExpr> {
    Ok(rust_q(
        extract_expr(s)?,
        extract_expr(mass1)?,
        extract_expr(mass2)?,
        sheet.map_or(Sheet::Physical, |sheet| sheet.inner),
    )
    .into())
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    *,
    mass1: "Expr | int | float",
    mass2: "Expr | int | float",
    sheet=None
))]
/// Construct the dimensionless two-body phase-space factor.
///
/// Parameters
/// ----------
/// s : Expr or number
///     Squared invariant mass.
/// mass1, mass2 : Expr or number
///     Daughter masses.
/// sheet : Sheet, optional
///     Analytic-continuation sheet; defaults to the physical sheet.
///
/// Returns
/// -------
/// Expr
///     Complex phase-space expression proportional to ``2*q/sqrt(s)``.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted to an expression.
pub fn rho(
    s: &Bound<'_, PyAny>,
    mass1: &Bound<'_, PyAny>,
    mass2: &Bound<'_, PyAny>,
    sheet: Option<&PySheet>,
) -> PyResult<PyExpr> {
    Ok(rust_rho(
        extract_expr(s)?,
        extract_expr(mass1)?,
        extract_expr(mass2)?,
        sheet.map_or(Sheet::Physical, |sheet| sheet.inner),
    )
    .into())
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    *,
    mass1: "Expr | int | float",
    mass2: "Expr | int | float"
))]
/// Construct the Chew-Mandelstam two-body function.
///
/// Parameters
/// ----------
/// s : Expr or number
///     Squared invariant mass.
/// mass1, mass2 : Expr or number
///     Daughter masses.
///
/// Returns
/// -------
/// Expr
///     Complex dispersive phase-space function on the physical sheet.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted to an expression.
pub fn chew_mandelstam(
    s: &Bound<'_, PyAny>,
    mass1: &Bound<'_, PyAny>,
    mass2: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    Ok(rust_chew_mandelstam(extract_expr(s)?, extract_expr(mass1)?, extract_expr(mass2)?).into())
}

#[pyfunction]
#[pyo3(signature = (
    q: "Expr | int | float",
    *,
    l,
    kind
))]
/// Construct a normalized Blatt-Weisskopf barrier factor.
///
/// Parameters
/// ----------
/// q : Expr or number
///     Breakup momentum.
/// l : L or int
///     Orbital angular momentum.
/// kind : BarrierKind
///     Barrier-factor convention.
///
/// Returns
/// -------
/// Expr
///     Symbolic barrier factor with reference momentum ``q_r = 0.1973`` GeV.
///
/// Raises
/// ------
/// TypeError
///     If `q` or `l` cannot be converted.
/// LadduError
///     If the angular momentum is unsupported.
pub fn blatt_weisskopf(q: &Bound<'_, PyAny>, l: PyL, kind: &PyBarrierKind) -> PyResult<PyExpr> {
    rust_blatt_weisskopf(extract_expr(q)?, l.inner, kind.inner)
        .map(PyExpr::from)
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    q: "Expr | int | float",
    *,
    l,
    kind,
    q_r: "Expr | int | float"
))]
/// Construct a Blatt-Weisskopf factor with a custom reference momentum.
///
/// Parameters
/// ----------
/// q : Expr or number
///     Breakup momentum.
/// l : L or int
///     Orbital angular momentum.
/// kind : BarrierKind
///     Barrier-factor convention.
/// q_r : Expr or number
///     Reference momentum or inverse barrier radius in GeV. Must remain real,
///     positive, and finite; use a bounded parameter to vary it in a fit.
///
/// Returns
/// -------
/// Expr
///     Real symbolic barrier factor.
///
/// Raises
/// ------
/// TypeError
///     If `q`, `l`, or `q_r` cannot be converted.
/// LadduError
///     If the angular momentum or reference momentum is invalid.
pub fn blatt_weisskopf_custom(
    q: &Bound<'_, PyAny>,
    l: PyL,
    kind: &PyBarrierKind,
    q_r: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    rust_blatt_weisskopf_custom(extract_expr(q)?, l.inner, kind.inner, extract_expr(q_r)?)
        .map(PyExpr::from)
        .map_err(to_py_err)
}

impl_json_methods!(PySheet);
impl_json_methods!(PyBarrierKind);
