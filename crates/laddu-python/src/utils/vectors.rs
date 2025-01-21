use laddu_core::{
    traits::{FourMomentum, FourVector, ThreeMomentum, ThreeVector},
    Float, Vector3, Vector4,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyTypeError},
    prelude::*,
};

/// A 3-momentum vector formed from Cartesian components.
///
/// Parameters
/// ----------
/// px, py, pz : float
///     The Cartesian components of the 3-vector
///
#[pyclass(name = "Vector3", module = "laddu")]
#[derive(Clone)]
pub struct PyVector3(pub Vector3<Float>);
#[pymethods]
impl PyVector3 {
    #[new]
    fn new(px: Float, py: Float, pz: Float) -> Self {
        Self(Vector3::new(px, py, pz))
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(self.0 + other_vec.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(other_vec.0 + self.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(self.0 - other_vec.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Subtraction with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(other_vec.0 - self.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Subtraction with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }
    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self(-self.0))
    }
    /// Calculate the dot product of two Vector3s.
    ///
    /// Parameters
    /// ----------
    /// other : Vector3
    ///     A vector input with which the dot product is taken
    ///
    /// Returns
    /// -------
    /// float
    ///     The dot product of this vector and `other`
    ///
    pub fn dot(&self, other: Self) -> Float {
        self.0.dot(&other.0)
    }
    /// Calculate the cross product of two Vector3s.
    ///
    /// Parameters
    /// ----------
    /// other : Vector3
    ///     A vector input with which the cross product is taken
    ///
    /// Returns
    /// -------
    /// Vector3
    ///     The cross product of this vector and `other`
    ///
    fn cross(&self, other: Self) -> Self {
        Self(self.0.cross(&other.0))
    }
    /// The magnitude of the 3-vector
    ///
    /// .. math:: |\vec{p}| = \sqrt{p_x^2 + p_y^2 + p_z^2}
    ///
    #[getter]
    fn mag(&self) -> Float {
        self.0.mag()
    }
    /// The squared magnitude of the 3-vector
    ///
    /// .. math:: |\vec{p}|^2 = p_x^2 + p_y^2 + p_z^2
    ///
    #[getter]
    fn mag2(&self) -> Float {
        self.0.mag2()
    }
    /// The cosine of the polar angle of this vector in spherical coordinates
    ///
    /// .. math:: -1 \leq \cos\theta \leq +1
    ///
    /// .. math:: \cos\theta = \frac{p_z}{|\vec{p}|}
    ///
    #[getter]
    fn costheta(&self) -> Float {
        self.0.costheta()
    }
    /// The polar angle of this vector in spherical coordinates
    ///
    /// .. math:: 0 \leq \theta \leq \pi
    ///
    /// .. math:: \theta = \arccos\left(\frac{p_z}{|\vec{p}|}\right)
    ///
    #[getter]
    fn theta(&self) -> Float {
        self.0.theta()
    }
    /// The azimuthal angle of this vector in spherical coordinates
    ///
    /// .. math:: 0 \leq \varphi \leq 2\pi
    ///
    /// .. math:: \varphi = \text{sgn}(p_y)\arccos\left(\frac{p_x}{\sqrt{p_x^2 + p_y^2}}\right)
    ///
    #[getter]
    fn phi(&self) -> Float {
        self.0.phi()
    }
    /// The normalized unit vector pointing in the direction of this vector
    ///
    #[getter]
    fn unit(&self) -> Self {
        Self(self.0.unit())
    }
    /// The x-component of this vector
    ///
    /// See Also
    /// --------
    /// Vector3.x
    ///
    #[getter]
    fn px(&self) -> Float {
        self.0.px()
    }
    /// The x-component of this vector
    ///
    /// See Also
    /// --------
    /// Vector3.px
    ///
    #[getter]
    fn x(&self) -> Float {
        self.0.x
    }

    /// The y-component of this vector
    ///
    /// See Also
    /// --------
    /// Vector3.y
    ///
    #[getter]
    fn py(&self) -> Float {
        self.0.py()
    }
    /// The y-component of this vector
    ///
    /// See Also
    /// --------
    /// Vector3.py
    ///
    #[getter]
    fn y(&self) -> Float {
        self.0.y
    }
    /// The z-component of this vector
    ///
    /// See Also
    /// --------
    /// Vector3.z
    ///
    #[getter]
    fn pz(&self) -> Float {
        self.0.pz()
    }
    /// The z-component of this vector
    ///
    /// See Also
    /// --------
    /// Vector3.pz
    ///
    #[getter]
    fn z(&self) -> Float {
        self.0.z
    }
    /// Convert a 3-vector momentum to a 4-momentum with the given mass.
    ///
    /// The mass-energy equivalence is used to compute the energy of the 4-momentum:
    ///
    /// .. math:: E = \sqrt{m^2 + p^2}
    ///
    /// Parameters
    /// ----------
    /// mass: float
    ///     The mass of the new 4-momentum
    ///
    /// Returns
    /// -------
    /// Vector4
    ///     A new 4-momentum with the given mass
    ///
    fn with_mass(&self, mass: Float) -> PyVector4 {
        PyVector4(self.0.with_mass(mass))
    }
    /// Convert a 3-vector momentum to a 4-momentum with the given energy.
    ///
    /// Parameters
    /// ----------
    /// energy: float
    ///     The mass of the new 4-momentum
    ///
    /// Returns
    /// -------
    /// Vector4
    ///     A new 4-momentum with the given energy
    ///
    fn with_energy(&self, mass: Float) -> PyVector4 {
        PyVector4(self.0.with_energy(mass))
    }
    /// Convert the 3-vector to a ``numpy`` array.
    ///
    /// Returns
    /// -------
    /// numpy_vec: array_like
    ///     A ``numpy`` array built from the components of this ``Vector3``
    ///
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, self.0.as_slice())
    }
    /// Convert an  array into a 3-vector.
    ///
    /// Parameters
    /// ----------
    /// array_like
    ///     An array containing the components of this ``Vector3``
    ///
    /// Returns
    /// -------
    /// laddu_vec: Vector3
    ///     A copy of the input array as a ``laddu`` vector
    ///
    #[staticmethod]
    fn from_array(array: Vec<Float>) -> Self {
        Self::new(array[0], array[1], array[2])
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __getitem__(&self, index: usize) -> PyResult<Float> {
        self.0
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .copied()
    }
}

/// A 4-momentum vector formed from energy and Cartesian 3-momentum components.
///
/// This vector is ordered with energy as the fourth component (:math:`[p_x, p_y, p_z, E]`) and assumes a :math:`(---+)`
/// signature
///
/// Parameters
/// ----------
/// px, py, pz : float
///     The Cartesian components of the 3-vector
/// e : float
///     The energy component
///
///
#[pyclass(name = "Vector4", module = "laddu")]
#[derive(Clone)]
pub struct PyVector4(pub Vector4<Float>);
#[pymethods]
impl PyVector4 {
    #[new]
    fn new(px: Float, py: Float, pz: Float, e: Float) -> Self {
        Self(Vector4::new(px, py, pz, e))
    }
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(self.0 + other_vec.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(other_vec.0 + self.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Addition with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(self.0 - other_vec.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Subtraction with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_vec) = other.extract::<PyRef<Self>>() {
            Ok(Self(other_vec.0 - self.0))
        } else if let Ok(other_int) = other.extract::<usize>() {
            if other_int == 0 {
                Ok(self.clone())
            } else {
                Err(PyTypeError::new_err(
                    "Subtraction with an integer for this type is only defined for 0",
                ))
            }
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }
    fn __neg__(&self) -> PyResult<Self> {
        Ok(Self(-self.0))
    }
    /// The magnitude of the 4-vector
    ///
    /// .. math:: |p| = \sqrt{E^2 - (p_x^2 + p_y^2 + p_z^2)}
    ///
    /// See Also
    /// --------
    /// Vector4.m
    ///
    #[getter]
    fn mag(&self) -> Float {
        self.0.mag()
    }
    /// The squared magnitude of the 4-vector
    ///
    /// .. math:: |p|^2 = E^2 - (p_x^2 + p_y^2 + p_z^2)
    ///
    /// See Also
    /// --------
    /// Vector4.m2
    ///
    #[getter]
    fn mag2(&self) -> Float {
        self.0.mag2()
    }
    /// The 3-vector part of this 4-vector
    ///
    /// See Also
    /// --------
    /// Vector4.momentum
    ///
    #[getter]
    fn vec3(&self) -> PyVector3 {
        PyVector3(self.0.vec3().into())
    }
    /// Boost the given 4-momentum according to a boost velocity.
    ///
    /// The resulting 4-momentum is equal to the original boosted to an inertial frame with
    /// relative velocity :math:`\beta`:
    ///
    /// .. math:: \left[\vec{p}'; E'\right] = \left[ \vec{p} + \left(\frac{(\gamma - 1) \vec{p}\cdot\vec{\beta}}{\beta^2} + \gamma E\right)\vec{\beta}; \gamma E + \vec{\beta}\cdot\vec{p} \right]
    ///
    /// Parameters
    /// ----------
    /// beta : Vector3
    ///     The relative velocity needed to get to the new frame from the current one
    ///
    /// Returns
    /// -------
    /// Vector4
    ///     The boosted 4-momentum
    ///
    /// See Also
    /// --------
    /// Vector4.beta
    /// Vector4.gamma
    ///
    fn boost(&self, beta: &PyVector3) -> Self {
        Self(self.0.boost(&beta.0))
    }
    /// The energy associated with this vector
    ///
    #[getter]
    fn e(&self) -> Float {
        self.0.e()
    }
    /// The energy associated with this vector
    ///
    #[getter]
    fn w(&self) -> Float {
        self.0.w
    }
    /// The x-component of this vector
    ///
    #[getter]
    fn px(&self) -> Float {
        self.0.px()
    }
    /// The x-component of this vector
    ///
    #[getter]
    fn x(&self) -> Float {
        self.0.x
    }

    /// The y-component of this vector
    ///
    #[getter]
    fn py(&self) -> Float {
        self.0.py()
    }
    /// The y-component of this vector
    ///
    #[getter]
    fn y(&self) -> Float {
        self.0.y
    }
    /// The z-component of this vector
    ///
    #[getter]
    fn pz(&self) -> Float {
        self.0.pz()
    }
    /// The z-component of this vector
    ///
    #[getter]
    fn z(&self) -> Float {
        self.0.z
    }
    /// The 3-momentum part of this 4-momentum
    ///
    #[getter]
    fn momentum(&self) -> PyVector3 {
        PyVector3(self.0.momentum().into())
    }
    /// The relativistic gamma factor
    ///
    /// .. math:: \gamma = \frac{1}{\sqrt{1 - \beta^2}}
    ///
    /// See Also
    /// --------
    /// Vector4.beta
    /// Vector4.boost
    ///
    #[getter]
    fn gamma(&self) -> Float {
        self.0.gamma()
    }
    /// The velocity 3-vector
    ///
    /// .. math:: \vec{\beta} = \frac{\vec{p}}{E}
    ///
    /// See Also
    /// --------
    /// Vector4.gamma
    /// Vector4.boost
    ///
    #[getter]
    fn beta(&self) -> PyVector3 {
        PyVector3(self.0.beta())
    }
    /// The invariant mass associated with the four-momentum
    ///
    /// .. math:: m = \sqrt{E^2 - (p_x^2 + p_y^2 + p_z^2)}
    ///
    /// See Also
    /// --------
    /// Vector4.mag
    ///
    #[getter]
    fn m(&self) -> Float {
        self.0.m()
    }
    /// The square of the invariant mass associated with the four-momentum
    ///
    /// .. math:: m^2 = E^2 - (p_x^2 + p_y^2 + p_z^2)
    ///
    /// See Also
    /// --------
    /// Vector4.mag2
    ///
    #[getter]
    fn m2(&self) -> Float {
        self.0.m2()
    }
    /// Convert the 4-vector to a `numpy` array.
    ///
    /// Returns
    /// -------
    /// numpy_vec: array_like
    ///     A ``numpy`` array built from the components of this ``Vector4``
    ///
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, self.0.as_slice())
    }
    /// Convert an array into a 4-vector.
    ///
    /// Parameters
    /// ----------
    /// array_like
    ///     An array containing the components of this ``Vector4``
    ///
    /// Returns
    /// -------
    /// laddu_vec: Vector4
    ///     A copy of the input array as a ``laddu`` vector
    ///
    #[staticmethod]
    fn from_array(array: Vec<Float>) -> Self {
        Self::new(array[0], array[1], array[2], array[3])
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        self.0.to_p4_string()
    }
    fn __getitem__(&self, index: usize) -> PyResult<Float> {
        self.0
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .copied()
    }
}
