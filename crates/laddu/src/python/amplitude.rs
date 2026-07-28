use laddu_amplitudes::{
    blatt_weisskopf_barriers as rust_blatt_weisskopf_barriers, breit_wigner as rust_breit_wigner,
    f_vector as rust_f_vector, k_matrix as rust_k_matrix,
    k_matrix_with_background as rust_k_matrix_with_background, kopf_pi1 as rust_kopf_pi1,
    kopf_rho as rust_kopf_rho, p_vector as rust_p_vector,
    p_vector_with_background as rust_p_vector_with_background,
    relativistic_breit_wigner as rust_relativistic_breit_wigner,
    relativistic_breit_wigner_custom as rust_relativistic_breit_wigner_custom,
};
use pyo3::{prelude::*, types::PyAny};

use super::{
    error::to_py_err,
    expr::{PyExpr, extract_expr},
    quantum::extract_l,
};

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    mass: "Expr | int | float",
    width: "Expr | int | float"
))]
/// Construct a constant-width Breit-Wigner amplitude.
///
/// Parameters
/// ----------
/// s : Expr or number
///     Squared invariant mass.
/// mass : Expr or number
///     Resonance pole mass.
/// width : Expr or number
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
    s: "Expr | int | float",
    mass: "Expr | int | float",
    width: "Expr | int | float",
    mass1: "Expr | int | float",
    mass2: "Expr | int | float",
    *,
    l: "int | float | None" = None
))]
/// Construct a relativistic two-body Breit-Wigner amplitude.
///
/// Parameters
/// ----------
/// s, mass, width : Expr or number
///     Squared invariant mass, pole mass, and nominal width.
/// mass1, mass2 : Expr or number
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
    l: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpr> {
    let l = match l {
        Some(l) => extract_l(l)?,
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
    s: "Expr | int | float",
    mass: "Expr | int | float",
    width: "Expr | int | float",
    mass1: "Expr | int | float",
    mass2: "Expr | int | float",
    l: "int | float",
    *,
    barrier_factors=true,
    q_r=1.0
))]
#[allow(clippy::too_many_arguments)]
/// Construct a configurable relativistic Breit-Wigner amplitude.
///
/// Parameters
/// ----------
/// s, mass, width : Expr or number
///     Squared invariant mass, pole mass, and nominal width.
/// mass1, mass2 : Expr or number
///     Daughter masses.
/// l : L or int
///     Orbital angular momentum.
/// barrier_factors : bool, default=True
///     Include Blatt-Weisskopf factors in the running width.
/// q_r : float, default=1.0
///     Barrier-radius momentum scale.
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
    l: &Bound<'_, PyAny>,
    barrier_factors: bool,
    q_r: f64,
) -> PyResult<PyExpr> {
    rust_relativistic_breit_wigner_custom(
        extract_expr(s)?,
        extract_expr(mass)?,
        extract_expr(width)?,
        extract_expr(mass1)?,
        extract_expr(mass2)?,
        extract_l(l)?,
        barrier_factors,
        q_r,
    )
    .map(PyExpr::from)
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    channel_mass_1: "Expr | int | float",
    channel_mass_2: "Expr | int | float",
    pole_masses: "Expr | int | float",
    l: "int | float",
    *,
    q_r=1.0
))]
/// Build channel-by-pole Blatt-Weisskopf barrier factors.
///
/// Parameters
/// ----------
/// s : Expr
///     Squared invariant mass.
/// channel_mass_1, channel_mass_2 : Expr
///     Vectors of the two daughter masses for each channel.
/// pole_masses : Expr
///     Vector of pole masses.
/// l : L or int
///     Orbital angular momentum.
/// q_r : float, default=1.0
///     Barrier-radius momentum scale.
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
    l: &Bound<'_, PyAny>,
    q_r: f64,
) -> PyResult<PyExpr> {
    rust_blatt_weisskopf_barriers(
        extract_expr(s)?,
        extract_expr(channel_mass_1)?,
        extract_expr(channel_mass_2)?,
        extract_expr(pole_masses)?,
        extract_l(l)?,
        q_r,
    )
    .map(PyExpr::from)
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    pole_masses: "Expr | int | float",
    couplings: "Expr | int | float",
    barriers: "Expr | int | float",
    *,
    background: "Expr | int | float | None" = None
))]
/// Construct a coupled-channel K-matrix expression.
///
/// Parameters
/// ----------
/// s : Expr
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
    s: "Expr | int | float",
    pole_masses: "Expr | int | float",
    production: "Expr | int | float",
    couplings: "Expr | int | float",
    barriers: "Expr | int | float",
    *,
    background: "Expr | int | float | None" = None
))]
/// Construct the production vector for a coupled-channel model.
///
/// Parameters
/// ----------
/// s : Expr
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
    s: "Expr | int | float",
    pole_masses: "Expr | int | float",
    k: "Expr | int | float",
    p: "Expr | int | float",
    phase_space: "Expr | int | float"
))]
/// Unitarize a production vector with a K-matrix and phase space.
///
/// Parameters
/// ----------
/// s : Expr
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

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    production: "list[Expr | int | float] | tuple[Expr | int | float, ...]"
))]
/// Construct the published two-pole Kopf rho amplitude.
///
/// Parameters
/// ----------
/// s : Expr or number
///     Squared invariant mass.
/// production : sequence of Expr
///     Exactly two complex production amplitudes.
///
/// Returns
/// -------
/// Expr
///     Coupled-channel rho amplitude vector.
///
/// Raises
/// ------
/// ValueError
///     If `production` does not contain exactly two entries.
/// TypeError
///     If an entry cannot be converted to an expression.
pub fn kopf_rho(s: &Bound<'_, PyAny>, production: Vec<Bound<'_, PyAny>>) -> PyResult<PyExpr> {
    let production: [laddu_expr::Expr; 2] = production
        .iter()
        .map(extract_expr)
        .collect::<PyResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("kopf_rho requires 2 production amplitudes")
        })?;
    rust_kopf_rho(extract_expr(s)?, production)
        .map(PyExpr::from)
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (
    s: "Expr | int | float",
    production: "list[Expr | int | float] | tuple[Expr | int | float, ...]"
))]
/// Construct the published one-pole Kopf pi1 amplitude.
///
/// Parameters
/// ----------
/// s : Expr or number
///     Squared invariant mass.
/// production : sequence of Expr
///     Exactly one complex production amplitude.
///
/// Returns
/// -------
/// Expr
///     Coupled-channel pi1 amplitude vector.
///
/// Raises
/// ------
/// ValueError
///     If `production` does not contain exactly one entry.
/// TypeError
///     If the entry cannot be converted to an expression.
pub fn kopf_pi1(s: &Bound<'_, PyAny>, production: Vec<Bound<'_, PyAny>>) -> PyResult<PyExpr> {
    let production: [laddu_expr::Expr; 1] = production
        .iter()
        .map(extract_expr)
        .collect::<PyResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("kopf_pi1 requires 1 production amplitude")
        })?;
    rust_kopf_pi1(extract_expr(s)?, production)
        .map(PyExpr::from)
        .map_err(to_py_err)
}

#[pymodule]
/// Standard resonance and coupled-channel amplitude constructors.
pub mod amplitudes {
    #[pymodule_export]
    use super::{
        blatt_weisskopf_barriers, breit_wigner, f_vector, k_matrix, kopf_pi1, kopf_rho, p_vector,
        relativistic_breit_wigner, relativistic_breit_wigner_custom,
    };
}
