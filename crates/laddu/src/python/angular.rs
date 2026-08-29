use laddu_physics::{
    math::{WignerDMatrix, clebsch_gordan as physics_clebsch_gordan},
    vectors::{Vec3, Vec4},
};
use pyo3::{prelude::*, types::PyAny};

use super::{
    error::to_py_err,
    expr::{PyExpr, extract_expr},
    quantum::{extract_projection, extract_spin},
};

#[pyclass(name = "Vec3", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A symbolic Cartesian three-vector.
///
/// Parameters
/// ----------
/// x, y, z : Expr or float
///     Cartesian components. Components may depend on event data or fit
///     parameters.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> beam_axis = ld.Vec3.z_axis()
/// >>> momentum = ld.Vec3.event("p")
/// >>> longitudinal = momentum.dot(beam_axis)
pub struct PyVec3 {
    pub(crate) inner: Vec3,
}

#[pymethods]
impl PyVec3 {
    /// Construct a symbolic three-vector from Cartesian components.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a component cannot be converted to an expression.
    #[new]
    #[pyo3(signature = (
        x: "Expr | float",
        y: "Expr | float",
        z: "Expr | float"
    ))]
    fn new(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: Vec3::new(extract_expr(x)?, extract_expr(y)?, extract_expr(z)?),
        })
    }

    #[staticmethod]
    /// Return the vector whose three components are zero.
    fn zero() -> Self {
        Self {
            inner: Vec3::zero(),
        }
    }

    #[staticmethod]
    /// Read the spatial components of a named event four-vector.
    ///
    /// Parameters
    /// ----------
    /// prefix : str
    ///     Name of the four-vector column.
    fn event(prefix: &str) -> Self {
        Self {
            inner: Vec3::event(prefix),
        }
    }

    #[staticmethod]
    /// Return the positive Cartesian x-axis unit vector.
    fn x_axis() -> Self {
        Self { inner: Vec3::x() }
    }

    #[staticmethod]
    /// Alias for :meth:`x_axis`.
    fn x() -> Self {
        Self::x_axis()
    }

    #[staticmethod]
    /// Return the positive Cartesian y-axis unit vector.
    fn y_axis() -> Self {
        Self { inner: Vec3::y() }
    }

    #[staticmethod]
    /// Alias for :meth:`y_axis`.
    fn y() -> Self {
        Self::y_axis()
    }

    #[staticmethod]
    /// Return the positive Cartesian z-axis unit vector.
    fn z_axis() -> Self {
        Self { inner: Vec3::z() }
    }

    #[staticmethod]
    /// Alias for :meth:`z_axis`.
    fn z() -> Self {
        Self::z_axis()
    }

    /// Return the symbolic cross product with another vector.
    fn cross(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.cross(&other.inner),
        }
    }

    /// Return the symbolic Euclidean dot product with another vector.
    fn dot(&self, other: &Self) -> PyExpr {
        self.inner.dot(&other.inner).into()
    }

    /// Return the symbolic Euclidean dot product with another vector.
    fn __matmul__(&self, other: &Self) -> PyExpr {
        self.dot(other)
    }

    /// Return the x component.
    fn px(&self) -> PyExpr {
        self.inner.px().into()
    }

    /// Return the y component.
    fn py(&self) -> PyExpr {
        self.inner.py().into()
    }

    /// Return the z component.
    fn pz(&self) -> PyExpr {
        self.inner.pz().into()
    }

    /// Return the squared Euclidean magnitude.
    fn mag2(&self) -> PyExpr {
        self.inner.mag2().into()
    }

    /// Return the Euclidean magnitude.
    fn mag(&self) -> PyExpr {
        self.inner.mag().into()
    }

    /// Return the polar-angle cosine relative to the positive z-axis.
    fn costheta(&self) -> PyExpr {
        self.inner.costheta().into()
    }

    /// Return the azimuthal angle in radians.
    fn phi(&self) -> PyExpr {
        self.inner.phi().into()
    }

    /// Return a vector normalized to unit magnitude.
    fn unit(&self) -> Self {
        Self {
            inner: self.inner.unit(),
        }
    }

    /// Promote the vector to a four-momentum with a specified invariant mass.
    ///
    /// The energy component is constructed as ``sqrt(|p|**2 + mass**2)``.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `mass` cannot be converted to an expression.
    #[pyo3(signature = (mass: "Expr | float"))]
    fn with_mass(&self, mass: &Bound<'_, PyAny>) -> PyResult<PyVec4> {
        Ok(PyVec4 {
            inner: self.inner.with_mass(extract_expr(mass)?),
        })
    }

    /// Promote the vector to a four-vector with a specified energy.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `energy` cannot be converted to an expression.
    #[pyo3(signature = (energy: "Expr | float"))]
    fn with_energy(&self, energy: &Bound<'_, PyAny>) -> PyResult<PyVec4> {
        Ok(PyVec4 {
            inner: self.inner.with_energy(extract_expr(energy)?),
        })
    }

    /// Return this vector as a vector-valued :class:`Expr`.
    fn as_expr(&self) -> PyExpr {
        self.inner.as_expr().into()
    }

    fn __add__(&self, other: &Self) -> Self {
        Self {
            inner: &self.inner + &other.inner,
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self {
            inner: &self.inner - &other.inner,
        }
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: -&self.inner,
        }
    }

    fn __mul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_expr(scalar)?;
        Ok(Self {
            inner: &self.inner * &scalar,
        })
    }

    fn __rmul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(scalar)
    }

    fn __truediv__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_expr(scalar)?;
        Ok(Self {
            inner: &self.inner / &scalar,
        })
    }
}

#[pyclass(name = "Vec4", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A symbolic four-vector in metric order ``(E, px, py, pz)``.
///
/// Parameters
/// ----------
/// e, px, py, pz : Expr or float
///     Energy and momentum components.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> p4 = ld.Vec4.event("proton")
/// >>> invariant_mass = p4.mass()
pub struct PyVec4 {
    pub(crate) inner: Vec4,
}

#[pymethods]
impl PyVec4 {
    /// Construct a symbolic four-vector in ``(E, px, py, pz)`` order.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a component cannot be converted to an expression.
    #[new]
    #[pyo3(signature = (
        e: "Expr | float",
        px: "Expr | float",
        py: "Expr | float",
        pz: "Expr | float"
    ))]
    fn new(
        e: &Bound<'_, PyAny>,
        px: &Bound<'_, PyAny>,
        py: &Bound<'_, PyAny>,
        pz: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Vec4::new(
                extract_expr(e)?,
                extract_expr(px)?,
                extract_expr(py)?,
                extract_expr(pz)?,
            ),
        })
    }

    #[staticmethod]
    /// Read a named four-vector from each event.
    fn event(prefix: &str) -> Self {
        Self {
            inner: Vec4::event(prefix),
        }
    }

    /// Return the x momentum component.
    fn px(&self) -> PyExpr {
        self.inner.px().into()
    }

    /// Return the y momentum component.
    fn py(&self) -> PyExpr {
        self.inner.py().into()
    }

    /// Return the z momentum component.
    fn pz(&self) -> PyExpr {
        self.inner.pz().into()
    }

    /// Return the energy component.
    fn e(&self) -> PyExpr {
        self.inner.e().into()
    }

    /// Return the spatial momentum as a three-vector.
    fn momentum(&self) -> PyVec3 {
        PyVec3 {
            inner: self.inner.momentum(),
        }
    }

    /// Alias for :meth:`momentum`.
    fn vec3(&self) -> PyVec3 {
        PyVec3 {
            inner: self.inner.vec3(),
        }
    }

    /// Return the three-velocity ``p / E``.
    fn beta(&self) -> PyVec3 {
        PyVec3 {
            inner: self.inner.beta(),
        }
    }

    /// Return the Lorentz factor ``E / mass``.
    fn gamma(&self) -> PyExpr {
        self.inner.gamma().into()
    }

    /// Return the invariant mass squared using the ``(+---)`` metric.
    fn m2(&self) -> PyExpr {
        self.inner.m2().into()
    }

    /// Return the nonnegative invariant mass.
    fn mass(&self) -> PyExpr {
        self.inner.m().into()
    }

    /// Alias for :meth:`mass`.
    fn m(&self) -> PyExpr {
        self.mass()
    }

    /// Alias for :meth:`m2`.
    fn mag2(&self) -> PyExpr {
        self.inner.mag2().into()
    }

    /// Alias for :meth:`mass`.
    fn mag(&self) -> PyExpr {
        self.inner.mag().into()
    }

    /// Return the Lorentz inner product with another four-vector.
    fn dot(&self, other: &Self) -> PyExpr {
        self.inner.dot(&other.inner).into()
    }

    /// Return the Lorentz inner product with another four-vector.
    fn __matmul__(&self, other: &Self) -> PyExpr {
        self.dot(other)
    }

    /// Apply a Lorentz boost by a three-velocity.
    ///
    /// Parameters
    /// ----------
    /// beta : Vec3
    ///     Symbolic boost velocity in units where ``c = 1``.
    fn boost(&self, beta: &PyVec3) -> Self {
        Self {
            inner: self.inner.boost(&beta.inner),
        }
    }

    /// Return this four-vector as a vector-valued :class:`Expr`.
    fn as_expr(&self) -> PyExpr {
        self.inner.as_expr().into()
    }

    fn __add__(&self, other: &Self) -> Self {
        Self {
            inner: &self.inner + &other.inner,
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self {
            inner: &self.inner - &other.inner,
        }
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: -&self.inner,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    *,
    j1: "J | S | L | float | fractions.Fraction",
    m1: "M | float | fractions.Fraction",
    j2: "J | S | L | float | fractions.Fraction",
    m2: "M | float | fractions.Fraction",
    j: "J | S | L | float | fractions.Fraction",
    m: "M | float | fractions.Fraction"
))]
/// Evaluate a Clebsch-Gordan coefficient.
///
/// Parameters
/// ----------
/// j1, j2, j : J, S, L, float, or fractions.Fraction
///     Two input angular momenta and the coupled total angular momentum. Numeric
///     values must be nonnegative integers or half-integers.
/// m1, m2, m : M, float, or fractions.Fraction
///     Corresponding integer or half-integer projections.
///
/// Returns
/// -------
/// float
///     The coefficient ``<j1 m1, j2 m2 | j m>``. Selection-rule violations
///     produce zero.
///
/// Raises
/// ------
/// TypeError
///     If an input cannot be represented as an integer or half-integer.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> ld.clebsch_gordan(j1=0.5, m1=0.5, j2=0.5, m2=-0.5, j=0, m=0)
/// 0.7071067811865476
pub fn clebsch_gordan(
    j1: &Bound<'_, PyAny>,
    m1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    m2: &Bound<'_, PyAny>,
    j: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    Ok(physics_clebsch_gordan(
        extract_spin(j1)?,
        extract_projection(m1)?,
        extract_spin(j2)?,
        extract_projection(m2)?,
        extract_spin(j)?,
        extract_projection(m)?,
    ))
}

#[pyclass(name = "WignerD", module = "laddu", frozen, skip_from_py_object)]
/// A fixed-index Wigner small-d and D-matrix element.
///
/// Parameters
/// ----------
/// j : J, S, L, float, or fractions.Fraction
///     Nonnegative integer or half-integer total angular momentum.
/// m_prime, m : M, float, or fractions.Fraction
///     Integer or half-integer output and input projections.
pub struct PyWignerD {
    inner: WignerDMatrix,
}

#[pymethods]
impl PyWignerD {
    /// Construct a Wigner matrix element with fixed quantum numbers.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a quantum number cannot be converted.
    /// LadduError
    ///     If a projection lies outside ``[-j, j]`` or has incompatible parity.
    #[new]
    #[pyo3(signature = (
        j: "J | S | L | float | fractions.Fraction",
        m_prime: "M | float | fractions.Fraction",
        m: "M | float | fractions.Fraction"
    ))]
    fn new(
        j: &Bound<'_, PyAny>,
        m_prime: &Bound<'_, PyAny>,
        m: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: WignerDMatrix::new(
                extract_spin(j)?,
                extract_projection(m_prime)?,
                extract_projection(m)?,
            )
            .map_err(to_py_err)?,
        })
    }

    /// Return the small-d element as a function of the polar angle.
    ///
    /// Parameters
    /// ----------
    /// beta : Expr or float
    ///     Polar Euler angle in radians.
    #[pyo3(signature = (beta: "Expr | float"))]
    fn d(&self, beta: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        Ok(self.inner.d(extract_expr(beta)?).into())
    }

    #[allow(non_snake_case)]
    #[pyo3(signature = (
        *,
        alpha: "Expr | float",
        beta: "Expr | float",
        gamma: "Expr | float | None" = None
    ))]
    /// Return the full complex Wigner D-matrix element.
    ///
    /// Parameters
    /// ----------
    /// alpha, beta : Expr or float
    ///     First two Euler angles in radians.
    /// gamma : Expr or float, optional
    ///     Third Euler angle; defaults to zero.
    ///
    /// Returns
    /// -------
    /// Expr
    ///     Complex symbolic rotation-matrix element.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an angle cannot be converted to an expression.
    fn D(
        &self,
        alpha: &Bound<'_, PyAny>,
        beta: &Bound<'_, PyAny>,
        gamma: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpr> {
        Ok(self
            .inner
            .D(
                extract_expr(alpha)?,
                extract_expr(beta)?,
                gamma
                    .map(extract_expr)
                    .transpose()?
                    .unwrap_or_else(|| 0.0.into()),
            )
            .into())
    }
}

impl_json_methods!(PyVec3);
impl_json_methods!(PyVec4);
impl_json_methods!(PyWignerD);
