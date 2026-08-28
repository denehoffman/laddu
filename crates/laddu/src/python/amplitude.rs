use laddu_amplitudes::{
    blatt_weisskopf_barriers as rust_blatt_weisskopf_barriers, breit_wigner as rust_breit_wigner,
    f_vector as rust_f_vector, k_matrix as rust_k_matrix,
    k_matrix_with_background as rust_k_matrix_with_background, p_vector as rust_p_vector,
    p_vector_with_background as rust_p_vector_with_background,
    relativistic_breit_wigner as rust_relativistic_breit_wigner,
    relativistic_breit_wigner_custom as rust_relativistic_breit_wigner_custom,
};
use laddu_physics::math::QR_DEFAULT;
use pyo3::{prelude::*, types::PyAny};

use super::{
    error::to_py_err,
    expr::{PyExpr, extract_expr},
    quantum::PyL,
};

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | complex",
    *,
    mass: "Expr | float",
    width: "Expr | float"
))]
/// Construct a constant-width Breit-Wigner amplitude.
///
/// Parameters
/// ----------
/// s : Expr or complex
///     Squared invariant mass.
/// mass : Expr or float
///     Resonance pole mass.
/// width : Expr or float
///     Resonance width.
///
/// Returns
/// -------
/// Expr
///     Complex amplitude ``1 / (mass**2 - s - 1j * mass * width)``.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted to an expression.
pub fn breit_wigner(
    s: &Bound<'_, PyAny>,
    mass: &Bound<'_, PyAny>,
    width: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    Ok(rust_breit_wigner(extract_expr(s)?, extract_expr(mass)?, extract_expr(width)?).into())
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | complex",
    *,
    mass: "Expr | float",
    width: "Expr | float",
    mass1: "Expr | float",
    mass2: "Expr | float",
    l = None
))]
/// Construct a relativistic two-body Breit-Wigner amplitude.
///
/// Parameters
/// ----------
/// s : Expr or complex
///     Squared invariant mass.
/// mass, width : Expr or float
///     Pole mass and nominal width.
/// mass1, mass2 : Expr or float
///     Daughter masses.
/// l : L or int, optional
///     Orbital angular momentum; the default is S-wave (``0``).
///
/// Returns
/// -------
/// Expr
///     Complex amplitude with a mass-dependent width and barrier factors.
///
/// Raises
/// ------
/// TypeError
///     If an expression or angular momentum cannot be converted.
/// LadduError
///     If the physical inputs are inconsistent.
pub fn relativistic_breit_wigner(
    s: &Bound<'_, PyAny>,
    mass: &Bound<'_, PyAny>,
    width: &Bound<'_, PyAny>,
    mass1: &Bound<'_, PyAny>,
    mass2: &Bound<'_, PyAny>,
    l: Option<PyL>,
) -> PyResult<PyExpr> {
    let l = match l {
        Some(l) => l.inner,
        None => laddu_physics::quantum::L::try_from(0).map_err(to_py_err)?,
    };
    rust_relativistic_breit_wigner(
        extract_expr(s)?,
        extract_expr(mass)?,
        extract_expr(width)?,
        extract_expr(mass1)?,
        extract_expr(mass2)?,
        l,
    )
    .map(PyExpr::from)
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | complex",
    *,
    mass: "Expr | float",
    width: "Expr | float",
    mass1: "Expr | float",
    mass2: "Expr | float",
    l,
    barrier_factors=true,
    q_r: "Expr | float | None" = None
))]
#[allow(clippy::too_many_arguments)]
/// Construct a configurable relativistic Breit-Wigner amplitude.
///
/// Parameters
/// ----------
/// s : Expr or complex
///     Squared invariant mass.
/// mass, width : Expr or float
///     Pole mass, and nominal width.
/// mass1, mass2 : Expr or float
///     Daughter masses.
/// l : L or int
///     Orbital angular momentum.
/// barrier_factors : bool, default=True
///     Include Blatt-Weisskopf factors in the running width.
/// q_r : Expr or float, optional
///     Barrier-radius momentum scale in GeV. Defaults to 0.1973 GeV.
///     Must remain real, positive, and finite; use a bounded
///     parameter to vary it in a fit.
///
/// Returns
/// -------
/// Expr
///     Configured complex resonance amplitude.
///
/// Raises
/// ------
/// TypeError
///     If an expression or angular momentum cannot be converted.
/// LadduError
///     If the physical inputs are inconsistent.
pub fn relativistic_breit_wigner_custom(
    s: &Bound<'_, PyAny>,
    mass: &Bound<'_, PyAny>,
    width: &Bound<'_, PyAny>,
    mass1: &Bound<'_, PyAny>,
    mass2: &Bound<'_, PyAny>,
    l: PyL,
    barrier_factors: bool,
    q_r: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpr> {
    let q_r = q_r
        .map(extract_expr)
        .transpose()?
        .unwrap_or_else(|| QR_DEFAULT.into());
    rust_relativistic_breit_wigner_custom(
        extract_expr(s)?,
        extract_expr(mass)?,
        extract_expr(width)?,
        extract_expr(mass1)?,
        extract_expr(mass2)?,
        l.inner,
        barrier_factors,
        q_r,
    )
    .map(PyExpr::from)
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | complex",
    *,
    channel_mass_1: "Expr",
    channel_mass_2: "Expr",
    pole_masses: "Expr",
    l,
    q_r: "Expr | float | None" = None
))]
/// Build channel-by-pole Blatt-Weisskopf barrier factors.
///
/// Parameters
/// ----------
/// s : Expr or complex
///     Squared invariant mass.
/// channel_mass_1, channel_mass_2 : Expr
///     Vectors of the two daughter masses for each channel.
/// pole_masses : Expr
///     Vector of pole masses.
/// l : L or int
///     Orbital angular momentum.
/// q_r : Expr or float, optional
///     Barrier-radius momentum scale in GeV. Defaults to 0.1973 GeV.
///     Must remain real, positive, and finite; use a bounded
///     parameter to vary it in a fit.
///
/// Returns
/// -------
/// Expr
///     Matrix whose rows are channels and columns are poles.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted.
/// LadduError
///     If expression dimensions are incompatible.
pub fn blatt_weisskopf_barriers(
    s: &Bound<'_, PyAny>,
    channel_mass_1: &Bound<'_, PyAny>,
    channel_mass_2: &Bound<'_, PyAny>,
    pole_masses: &Bound<'_, PyAny>,
    l: PyL,
    q_r: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpr> {
    let q_r = q_r
        .map(extract_expr)
        .transpose()?
        .unwrap_or_else(|| QR_DEFAULT.into());
    rust_blatt_weisskopf_barriers(
        extract_expr(s)?,
        extract_expr(channel_mass_1)?,
        extract_expr(channel_mass_2)?,
        extract_expr(pole_masses)?,
        l.inner,
        q_r,
    )
    .map(PyExpr::from)
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | complex",
    *,
    pole_masses: "Expr",
    couplings: "Expr",
    barriers: "Expr",
    background: "Expr | None" = None
))]
/// Construct a coupled-channel K-matrix expression.
///
/// Parameters
/// ----------
/// s : Expr or complex
///     Squared invariant mass.
/// pole_masses : Expr
///     Vector of pole masses.
/// couplings : Expr
///     Channel-by-pole coupling matrix.
/// barriers : Expr
///     Channel-by-pole barrier-factor matrix.
/// background : Expr, optional
///     Symmetric channel-by-channel background matrix.
///
/// Returns
/// -------
/// Expr
///     Channel-by-channel K-matrix.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted to an expression.
/// LadduError
///     If matrix and vector dimensions are incompatible.
pub fn k_matrix(
    s: &Bound<'_, PyAny>,
    pole_masses: &Bound<'_, PyAny>,
    couplings: &Bound<'_, PyAny>,
    barriers: &Bound<'_, PyAny>,
    background: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpr> {
    let result = match background {
        Some(background) => rust_k_matrix_with_background(
            extract_expr(s)?,
            extract_expr(pole_masses)?,
            extract_expr(couplings)?,
            extract_expr(barriers)?,
            extract_expr(background)?,
        ),
        None => rust_k_matrix(
            extract_expr(s)?,
            extract_expr(pole_masses)?,
            extract_expr(couplings)?,
            extract_expr(barriers)?,
        ),
    };
    result.map(PyExpr::from).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | complex",
    *,
    pole_masses: "Expr",
    production: "Expr",
    couplings: "Expr",
    barriers: "Expr",
    background: "Expr | None" = None
))]
/// Construct the production vector for a coupled-channel model.
///
/// Parameters
/// ----------
/// s : Expr or complex
///     Squared invariant mass.
/// pole_masses : Expr
///     Vector of pole masses.
/// production : Expr
///     Vector of complex production strengths for the poles.
/// couplings, barriers : Expr
///     Channel-by-pole coupling and barrier matrices.
/// background : Expr, optional
///     Nonresonant channel production vector.
///
/// Returns
/// -------
/// Expr
///     Channel production-vector expression.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted to an expression.
/// LadduError
///     If matrix and vector dimensions are incompatible.
pub fn p_vector(
    s: &Bound<'_, PyAny>,
    pole_masses: &Bound<'_, PyAny>,
    production: &Bound<'_, PyAny>,
    couplings: &Bound<'_, PyAny>,
    barriers: &Bound<'_, PyAny>,
    background: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpr> {
    let result = match background {
        Some(background) => rust_p_vector_with_background(
            extract_expr(s)?,
            extract_expr(pole_masses)?,
            extract_expr(production)?,
            extract_expr(couplings)?,
            extract_expr(barriers)?,
            extract_expr(background)?,
        ),
        None => rust_p_vector(
            extract_expr(s)?,
            extract_expr(pole_masses)?,
            extract_expr(production)?,
            extract_expr(couplings)?,
            extract_expr(barriers)?,
        ),
    };
    result.map(PyExpr::from).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | complex",
    *,
    pole_masses: "Expr",
    k: "Expr",
    p: "Expr",
    phase_space: "Expr"
))]
/// Unitarize a production vector with a K-matrix and phase space.
///
/// Parameters
/// ----------
/// s : Expr or complex
///     Squared invariant mass.
/// pole_masses : Expr
///     Pole-mass vector used to remove spurious singularities.
/// k : Expr
///     Channel-by-channel K-matrix.
/// p : Expr
///     Bare production vector.
/// phase_space : Expr
///     Channel phase-space vector or diagonal matrix.
///
/// Returns
/// -------
/// Expr
///     Unitarized channel amplitude vector.
///
/// Raises
/// ------
/// TypeError
///     If an argument cannot be converted to an expression.
/// LadduError
///     If expression dimensions are incompatible.
pub fn f_vector(
    s: &Bound<'_, PyAny>,
    pole_masses: &Bound<'_, PyAny>,
    k: &Bound<'_, PyAny>,
    p: &Bound<'_, PyAny>,
    phase_space: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    rust_f_vector(
        extract_expr(s)?,
        extract_expr(pole_masses)?,
        extract_expr(k)?,
        extract_expr(p)?,
        extract_expr(phase_space)?,
    )
    .map(PyExpr::from)
    .map_err(to_py_err)
}

#[pymodule]
/// Standard resonance and coupled-channel amplitude constructors.
pub mod amplitudes {
    #[pymodule_export]
    use super::{
        blatt_weisskopf_barriers, breit_wigner, f_vector, k_matrix, p_vector,
        relativistic_breit_wigner, relativistic_breit_wigner_custom,
    };
}
