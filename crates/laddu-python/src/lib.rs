use bincode::{deserialize, serialize};
use laddu_core::amplitudes::{
    constant, parameter, Amplitude, AmplitudeID, Evaluator, Expression, Manager, Model,
    ParameterLike,
};
use laddu_core::data::{open, BinnedDataset, Dataset, Event};
use laddu_core::utils::variables::{
    Angles, CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude, Polarization,
};
use laddu_core::{
    traits::{FourMomentum, FourVector, ReadWrite, ThreeMomentum, ThreeVector, Variable},
    Complex, Float, LadduError,
};
use laddu_core::{Vector3, Vector4};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "rayon")]
use rayon::ThreadPoolBuilder;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;

/// A 3-momentum vector formed from Cartesian components
///
/// Parameters
/// ----------
/// px, py, pz : float
///     The Cartesian components of the 3-vector
///
#[pyclass(name = "Vector3")]
#[derive(Clone)]
pub struct PyVector3(Vector3<Float>);
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
    /// The dot product
    ///
    /// Calculates the dot product of two Vector3s.
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
    /// The cross product
    ///
    /// Calculates the cross product of two Vector3s.
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
    /// This is calculated as:
    ///
    /// .. math:: |\vec{p}| = \sqrt{p_x^2 + p_y^2 + p_z^2}
    ///
    /// Returns
    /// -------
    /// float
    ///     The magnitude of this vector
    ///
    #[getter]
    fn mag(&self) -> Float {
        self.0.mag()
    }
    /// The squared magnitude of the 3-vector
    ///
    /// This is calculated as:
    ///
    /// .. math:: |\vec{p}|^2 = p_x^2 + p_y^2 + p_z^2
    ///
    /// Returns
    /// -------
    /// float
    ///     The squared magnitude of this vector
    ///
    #[getter]
    fn mag2(&self) -> Float {
        self.0.mag2()
    }
    /// The cosine of the polar angle of this vector in spherical coordinates
    ///
    /// The polar angle is defined in the range
    ///
    /// .. math:: 0 \leq \theta \leq \pi
    ///
    /// so the cosine falls in the range
    ///
    /// .. math:: -1 \leq \cos\theta \leq +1
    ///
    /// This is calculated as:
    ///
    /// .. math:: \cos\theta = \frac{p_z}{|\vec{p}|}
    ///
    /// Returns
    /// -------
    /// float
    ///     The cosine of the polar angle of this vector
    ///
    #[getter]
    fn costheta(&self) -> Float {
        self.0.costheta()
    }
    /// The polar angle of this vector in spherical coordinates
    ///
    /// The polar angle is defined in the range
    ///
    /// .. math:: 0 \leq \theta \leq \pi
    ///
    /// This is calculated as:
    ///
    /// .. math:: \theta = \arccos\left(\frac{p_z}{|\vec{p}|}\right)
    ///
    /// Returns
    /// -------
    /// float
    ///     The polar angle of this vector
    ///
    #[getter]
    fn theta(&self) -> Float {
        self.0.theta()
    }
    /// The azimuthal angle of this vector in spherical coordinates
    ///
    /// The azimuthal angle is defined in the range
    ///
    /// .. math:: 0 \leq \varphi \leq 2\pi
    ///
    /// This is calculated as:
    ///
    /// .. math:: \varphi = \text{sgn}(p_y)\arccos\left(\frac{p_x}{\sqrt{p_x^2 + p_y^2}}\right)
    ///
    /// although the actual calculation just uses the ``atan2`` function
    ///
    /// Returns
    /// -------
    /// float
    ///     The azimuthal angle of this vector
    ///
    #[getter]
    fn phi(&self) -> Float {
        self.0.phi()
    }
    /// The normalized unit vector pointing in the direction of this vector
    ///
    /// Returns
    /// -------
    /// Vector3
    ///     A unit vector pointing in the same direction as this vector
    ///
    #[getter]
    fn unit(&self) -> Self {
        Self(self.0.unit())
    }
    /// The x-component of this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The x-component
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
    /// Returns
    /// -------
    /// float
    ///     The x-component
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
    /// Returns
    /// -------
    /// float
    ///     The y-component
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
    /// Returns
    /// -------
    /// float
    ///     The y-component
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
    /// Returns
    /// -------
    /// float
    ///     The z-component
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
    /// Returns
    /// -------
    /// float
    ///     The z-component
    ///
    /// See Also
    /// --------
    /// Vector3.pz
    ///
    #[getter]
    fn z(&self) -> Float {
        self.0.z
    }
    /// Convert a 3-vector momentum to a 4-momentum with the given mass
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
    /// Convert a 3-vector momentum to a 4-momentum with the given energy
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
    /// Convert the 3-vector to a ``numpy`` array
    ///
    /// Returns
    /// -------
    /// numpy_vec: array_like
    ///     A ``numpy`` array built from the components of this ``Vector3``
    ///
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, self.0.as_slice())
    }
    /// Convert an  array into a 3-vector
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

/// A 4-momentum vector formed from energy and Cartesian 3-momentum components
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
#[pyclass(name = "Vector4")]
#[derive(Clone)]
pub struct PyVector4(Vector4<Float>);
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
    /// This is calculated as:
    ///
    /// .. math:: |p| = \sqrt{E^2 - (p_x^2 + p_y^2 + p_z^2)}
    ///
    /// Returns
    /// -------
    /// float
    ///     The magnitude of this vector
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
    /// This is calculated as:
    ///
    /// .. math:: |p|^2 = E^2 - (p_x^2 + p_y^2 + p_z^2)
    ///
    /// Returns
    /// -------
    /// float
    ///     The squared magnitude of this vector
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
    /// Returns
    /// -------
    /// Vector3
    ///     The internal 3-vector
    ///
    /// See Also
    /// --------
    /// Vector4.momentum
    ///
    #[getter]
    fn vec3(&self) -> PyVector3 {
        PyVector3(self.0.vec3().into())
    }
    /// Boost the given 4-momentum according to a boost velocity
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
    /// Returns
    /// -------
    /// float
    ///     The energy
    ///
    #[getter]
    fn e(&self) -> Float {
        self.0.e()
    }
    /// The energy associated with this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The energy
    ///
    #[getter]
    fn w(&self) -> Float {
        self.0.w
    }
    /// The x-component of this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The x-component
    ///
    /// See Also
    /// --------
    /// Vector4.x
    ///
    #[getter]
    fn px(&self) -> Float {
        self.0.px()
    }
    /// The x-component of this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The x-component
    ///
    /// See Also
    /// --------
    /// Vector4.px
    ///
    #[getter]
    fn x(&self) -> Float {
        self.0.x
    }

    /// The y-component of this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The y-component
    ///
    /// See Also
    /// --------
    /// Vector4.y
    ///
    #[getter]
    fn py(&self) -> Float {
        self.0.py()
    }
    /// The y-component of this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The y-component
    ///
    /// See Also
    /// --------
    /// Vector4.py
    ///
    #[getter]
    fn y(&self) -> Float {
        self.0.y
    }
    /// The z-component of this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The z-component
    ///
    /// See Also
    /// --------
    /// Vector4.z
    ///
    #[getter]
    fn pz(&self) -> Float {
        self.0.pz()
    }
    /// The z-component of this vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The z-component
    ///
    /// See Also
    /// --------
    /// Vector4.pz
    ///
    #[getter]
    fn z(&self) -> Float {
        self.0.z
    }
    /// The 3-momentum part of this 4-momentum
    ///
    /// Returns
    /// -------
    /// Vector3
    ///     The internal 3-momentum
    ///
    /// See Also
    /// --------
    /// Vector4.vec3
    ///
    #[getter]
    fn momentum(&self) -> PyVector3 {
        PyVector3(self.0.momentum().into())
    }
    /// The relativistic gamma factor
    ///
    /// The :math:`\gamma` factor is equivalent to
    ///
    /// .. math:: \gamma = \frac{1}{\sqrt{1 - \beta^2}}
    ///
    /// Returns
    /// -------
    /// float
    ///     The associated :math:`\gamma` factor
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
    /// The :math:`\beta` vector is equivalent to
    ///
    /// .. math:: \vec{\beta} = \frac{\vec{p}}{E}
    ///
    /// Returns
    /// -------
    /// Vector3
    ///     The associated velocity vector
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
    /// This is calculated as:
    ///
    /// .. math:: m = \sqrt{E^2 - (p_x^2 + p_y^2 + p_z^2)}
    ///
    /// Returns
    /// -------
    /// float
    ///     The magnitude of this vector
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
    /// This is calculated as:
    ///
    /// .. math:: m^2 = E^2 - (p_x^2 + p_y^2 + p_z^2)
    ///
    /// Returns
    /// -------
    /// float
    ///     The squared magnitude of this vector
    ///
    /// See Also
    /// --------
    /// Vector4.mag2
    ///
    #[getter]
    fn m2(&self) -> Float {
        self.0.m2()
    }
    /// Convert the 4-vector to a `numpy` array
    ///
    /// Returns
    /// -------
    /// numpy_vec: array_like
    ///     A ``numpy`` array built from the components of this ``Vector4``
    ///
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, self.0.as_slice())
    }
    /// Convert an  array into a 4-vector
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

/// A single event
///
/// Events are composed of a set of 4-momenta of particles in the overall
/// center-of-momentum frame, polarizations or helicities described by 3-vectors, and a
/// weight
///
/// Parameters
/// ----------
/// p4s : list of Vector4
///     4-momenta of each particle in the event in the overall center-of-momentum frame
/// eps : list of Vector3
///     3-vectors describing the polarization or helicity of the particles
///     given in `p4s`
/// weight : float
///     The weight associated with this event
///
#[pyclass(name = "Event")]
#[derive(Clone)]
pub struct PyEvent(Arc<Event>);

#[pymethods]
impl PyEvent {
    #[new]
    fn new(p4s: Vec<PyVector4>, eps: Vec<PyVector3>, weight: Float) -> Self {
        Self(Arc::new(Event {
            p4s: p4s.into_iter().map(|arr| arr.0).collect(),
            eps: eps.into_iter().map(|arr| arr.0).collect(),
            weight,
        }))
    }
    fn __str__(&self) -> String {
        self.0.to_string()
    }
    /// The list of 4-momenta for each particle in the event
    ///
    #[getter]
    fn get_p4s(&self) -> Vec<PyVector4> {
        self.0.p4s.iter().map(|p4| PyVector4(*p4)).collect()
    }
    /// The list of 3-vectors describing the polarization or helicity of particles in
    /// the event
    ///
    #[getter]
    fn get_eps(&self) -> Vec<PyVector3> {
        self.0
            .eps
            .iter()
            .map(|eps_vec| PyVector3(*eps_vec))
            .collect()
    }
    /// The weight of this event relative to others in a Dataset
    ///
    #[getter]
    fn get_weight(&self) -> Float {
        self.0.weight
    }
    /// Get the sum of the four-momenta within the event at the given indices
    ///
    /// Parameters
    /// ----------
    /// indices : list of int
    ///     The indices of the four-momenta to sum
    ///
    /// Returns
    /// -------
    /// Vector4
    ///     The result of summing the given four-momenta
    ///
    fn get_p4_sum(&self, indices: Vec<usize>) -> PyVector4 {
        PyVector4(self.0.get_p4_sum(indices))
    }
}

/// A set of Events
///
/// Datasets can be created from lists of Events or by using the provided ``laddu.open`` function
///
/// Datasets can also be indexed directly to access individual Events
///
/// Parameters
/// ----------
/// events : list of Event
///
/// See Also
/// --------
/// laddu.open
///
#[pyclass(name = "Dataset")]
#[derive(Clone)]
pub struct PyDataset(pub Arc<Dataset>);

#[pymethods]
impl PyDataset {
    #[new]
    fn new(events: Vec<PyEvent>) -> Self {
        Self(Arc::new(Dataset {
            events: events.into_iter().map(|event| event.0).collect(),
        }))
    }
    fn __len__(&self) -> usize {
        self.0.len()
    }
    /// Get the number of Events in the Dataset
    ///
    /// Returns
    /// -------
    /// n_events : int
    ///     The number of Events
    ///
    fn len(&self) -> usize {
        self.0.len()
    }
    /// Get the weighted number of Events in the Dataset
    ///
    /// Returns
    /// -------
    /// n_events : float
    ///     The sum of all Event weights
    ///
    fn weighted_len(&self) -> Float {
        self.0.weighted_len()
    }
    /// The weights associated with the Dataset
    ///
    /// Returns
    /// -------
    /// weights : array_like
    ///     A ``numpy`` array of Event weights
    ///
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.weights())
    }
    /// The internal list of Events stored in the Dataset
    ///
    /// Returns
    /// -------
    /// events : list of Event
    ///     The Events in the Dataset
    ///
    #[getter]
    fn events(&self) -> Vec<PyEvent> {
        self.0
            .events
            .iter()
            .map(|rust_event| PyEvent(rust_event.clone()))
            .collect()
    }
    fn __getitem__(&self, index: usize) -> PyResult<PyEvent> {
        self.0
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .map(|rust_event| PyEvent(rust_event.clone()))
    }
    /// Separates a Dataset into histogram bins by a Variable value
    ///
    /// Parameters
    /// ----------
    /// variable : {laddu.Mass, laddu.CosTheta, laddu.Phi, laddu.PolAngle, laddu.PolMagnitude, laddu.Mandelstam}
    ///     The Variable by which each Event is binned
    /// bins : int
    ///     The number of equally-spaced bins
    /// range : tuple[float, float]
    ///     The minimum and maximum bin edges
    ///
    /// Returns
    /// -------
    /// datasets : BinnedDataset
    ///     A pub structure that holds a list of Datasets binned by the given `variable`
    ///
    /// See Also
    /// --------
    /// laddu.Mass
    /// laddu.CosTheta
    /// laddu.Phi
    /// laddu.PolAngle
    /// laddu.PolMagnitude
    /// laddu.Mandelstam
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the given `variable` is not a valid variable
    ///
    #[pyo3(signature = (variable, bins, range))]
    fn bin_by(
        &self,
        variable: Bound<'_, PyAny>,
        bins: usize,
        range: (Float, Float),
    ) -> PyResult<PyBinnedDataset> {
        let py_variable = variable.extract::<PyVariable>()?;
        Ok(PyBinnedDataset(self.0.bin_by(py_variable, bins, range)))
    }
    /// Generate a new bootstrapped Dataset by randomly resampling the original with replacement
    ///
    /// The new Dataset is resampled with a random generator seeded by the provided `seed`
    ///
    /// Parameters
    /// ----------
    /// seed : int
    ///     The random seed used in the resampling process
    ///
    /// Returns
    /// -------
    /// Dataset
    ///     A bootstrapped Dataset
    ///
    fn bootstrap(&self, seed: usize) -> PyDataset {
        PyDataset(self.0.bootstrap(seed))
    }
}

/// A collection of Datasets binned by a Variable
///
/// BinnedDatasets can be indexed directly to access the underlying Datasets by bin
///
/// See Also
/// --------
/// laddu.Dataset.bin_by
///
#[pyclass(name = "BinnedDataset")]
pub struct PyBinnedDataset(BinnedDataset);

#[pymethods]
impl PyBinnedDataset {
    fn __len__(&self) -> usize {
        self.0.len()
    }
    /// Get the number of bins in the BinnedDataset
    ///
    /// Returns
    /// -------
    /// n : int
    ///     The number of bins
    fn len(&self) -> usize {
        self.0.len()
    }
    /// The number of bins in the BinnedDataset
    ///
    #[getter]
    fn bins(&self) -> usize {
        self.0.bins()
    }
    /// The minimum and maximum values of the binning Variable used to create this BinnedDataset
    ///
    #[getter]
    fn range(&self) -> (Float, Float) {
        self.0.range()
    }
    /// The edges of each bin in the BinnedDataset
    ///
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.edges())
    }
    fn __getitem__(&self, index: usize) -> PyResult<PyDataset> {
        self.0
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .map(|rust_dataset| PyDataset(rust_dataset.clone()))
    }
}

/// Open a Dataset from a file
///
/// Returns
/// -------
/// Dataset
///
/// Raises
/// ------
/// IOError
///     If the file could not be read
///
/// Warnings
/// --------
/// This method will panic/fail if the columns do not have the correct names or data types.
/// There is currently no way to make this nicer without a large performance dip (if you find a
/// way, please open a PR).
///
/// Notes
/// -----
/// Data should be stored in Parquet format with each column being filled with 32-bit floats
///
/// Valid/required column names have the following formats:
///
/// ``p4_{particle index}_{E|Px|Py|Pz}`` (four-momentum components for each particle)
///
/// ``eps_{particle index}_{x|y|z}`` (polarization/helicity vectors for each particle)
///
/// ``weight`` (the weight of the Event)
///
/// For example, the four-momentum of the 0th particle in the event would be stored in columns
/// with the names ``p4_0_E``, ``p4_0_Px``, ``p4_0_Py``, and ``p4_0_Pz``. That particle's
/// polarization could be stored in the columns ``eps_0_x``, ``eps_0_y``, and ``eps_0_z``. This
/// could continue for an arbitrary number of particles. The ``weight`` column is always
/// required.
///
#[pyfunction(name = "open")]
pub fn py_open(path: &str) -> PyResult<PyDataset> {
    Ok(PyDataset(open(path)?))
}

#[derive(FromPyObject, Clone, Serialize, Deserialize)]
pub enum PyVariable {
    #[pyo3(transparent)]
    Mass(PyMass),
    #[pyo3(transparent)]
    CosTheta(PyCosTheta),
    #[pyo3(transparent)]
    Phi(PyPhi),
    #[pyo3(transparent)]
    PolAngle(PyPolAngle),
    #[pyo3(transparent)]
    PolMagnitude(PyPolMagnitude),
    #[pyo3(transparent)]
    Mandelstam(PyMandelstam),
}

/// The invariant mass of an arbitrary combination of constituent particles in an Event
///
/// This variable is calculated by summing up the 4-momenta of each particle listed by index in
/// `constituents` and taking the invariant magnitude of the resulting 4-vector.
///
/// Parameters
/// ----------
/// constituents : list of int
///     The indices of particles to combine to create the final 4-momentum
///
/// See Also
/// --------
/// laddu.utils.vectors.Vector4.m
///
#[pyclass(name = "Mass")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyMass(pub Mass);

#[pymethods]
impl PyMass {
    #[new]
    fn new(constituents: Vec<usize>) -> Self {
        Self(Mass::new(&constituents))
    }
    /// The value of this Variable for the given Event
    ///
    /// Parameters
    /// ----------
    /// event : Event
    ///     The Event upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// value : float
    ///     The value of the Variable for the given `event`
    ///
    fn value(&self, event: &PyEvent) -> Float {
        self.0.value(&event.0)
    }
    /// All values of this Variable on the given Dataset
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     The Dataset upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///     The values of the Variable for each Event in the given `dataset`
    ///
    fn value_on<'py>(&self, py: Python<'py>, dataset: &PyDataset) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.value_on(&dataset.0))
    }
}

/// The cosine of the polar decay angle in the rest frame of the given `resonance`
///
/// This Variable is calculated by forming the given frame (helicity or Gottfried-Jackson) and
/// calculating the spherical angles according to one of the decaying `daughter` particles.
///
/// The helicity frame is defined in terms of the following Cartesian axes in the rest frame of
/// the `resonance`:
///
/// .. math:: \hat{z} \propto -\vec{p}'_{\text{recoil}}
/// .. math:: \hat{y} \propto \vec{p}_{\text{beam}} \times (-\vec{p}_{\text{recoil}})
/// .. math:: \hat{x} = \hat{y} \times \hat{z}
///
/// where primed vectors are in the rest frame of the `resonance` and unprimed vectors are in
/// the center-of-momentum frame.
///
/// The Gottfried-Jackson frame differs only in the definition of :math:`\hat{z}`:
///
/// .. math:: \hat{z} \propto \vec{p}'_{\text{beam}}
///
/// Parameters
/// ----------
/// beam : int
///     The index of the `beam` particle
/// recoil : list of int
///     Indices of particles which are combined to form the recoiling particle (particles which
///     are not `beam` or part of the `resonance`)
/// daughter : list of int
///     Indices of particles which are combined to form one of the decay products of the
///     `resonance`
/// resonance : list of int
///     Indices of particles which are combined to form the `resonance`
/// frame : {'Helicity', 'HX', 'HEL', 'GottfriedJackson', 'Gottfried Jackson', 'GJ', 'Gottfried-Jackson'}
///     The frame to use in the  calculation
///
/// Raises
/// ------
/// ValueError
///     If `frame` is not one of the valid options
///
/// See Also
/// --------
/// laddu.utils.vectors.Vector3.costheta
///
#[pyclass(name = "CosTheta")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyCosTheta(pub CosTheta);

#[pymethods]
impl PyCosTheta {
    #[new]
    #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
    fn new(
        beam: usize,
        recoil: Vec<usize>,
        daughter: Vec<usize>,
        resonance: Vec<usize>,
        frame: &str,
    ) -> PyResult<Self> {
        Ok(Self(CosTheta::new(
            beam,
            &recoil,
            &daughter,
            &resonance,
            frame.parse()?,
        )))
    }
    /// The value of this Variable for the given Event
    ///
    /// Parameters
    /// ----------
    /// event : Event
    ///     The Event upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// value : float
    ///     The value of the Variable for the given `event`
    ///
    fn value(&self, event: &PyEvent) -> Float {
        self.0.value(&event.0)
    }
    /// All values of this Variable on the given Dataset
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     The Dataset upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///     The values of the Variable for each Event in the given `dataset`
    ///
    fn value_on<'py>(&self, py: Python<'py>, dataset: &PyDataset) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.value_on(&dataset.0))
    }
}

/// The aziumuthal decay angle in the rest frame of the given `resonance`
///
/// This Variable is calculated by forming the given frame (helicity or Gottfried-Jackson) and
/// calculating the spherical angles according to one of the decaying `daughter` particles.
///
/// The helicity frame is defined in terms of the following Cartesian axes in the rest frame of
/// the `resonance`:
///
/// .. math:: \hat{z} \propto -\vec{p}'_{\text{recoil}}
/// .. math:: \hat{y} \propto \vec{p}_{\text{beam}} \times (-\vec{p}_{\text{recoil}})
/// .. math:: \hat{x} = \hat{y} \times \hat{z}
///
/// where primed vectors are in the rest frame of the `resonance` and unprimed vectors are in
/// the center-of-momentum frame.
///
/// The Gottfried-Jackson frame differs only in the definition of :math:`\hat{z}`:
///
/// .. math:: \hat{z} \propto \vec{p}'_{\text{beam}}
///
/// Parameters
/// ----------
/// beam : int
///     The index of the `beam` particle
/// recoil : list of int
///     Indices of particles which are combined to form the recoiling particle (particles which
///     are not `beam` or part of the `resonance`)
/// daughter : list of int
///     Indices of particles which are combined to form one of the decay products of the
///     `resonance`
/// resonance : list of int
///     Indices of particles which are combined to form the `resonance`
/// frame : {'Helicity', 'HX', 'HEL', 'GottfriedJackson', 'Gottfried Jackson', 'GJ', 'Gottfried-Jackson'}
///     The frame to use in the  calculation
///
/// Raises
/// ------
/// ValueError
///     If `frame` is not one of the valid options
///
///
/// See Also
/// --------
/// laddu.utils.vectors.Vector3.phi
///
#[pyclass(name = "Phi")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPhi(pub Phi);

#[pymethods]
impl PyPhi {
    #[new]
    #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
    fn new(
        beam: usize,
        recoil: Vec<usize>,
        daughter: Vec<usize>,
        resonance: Vec<usize>,
        frame: &str,
    ) -> PyResult<Self> {
        Ok(Self(Phi::new(
            beam,
            &recoil,
            &daughter,
            &resonance,
            frame.parse()?,
        )))
    }
    /// The value of this Variable for the given Event
    ///
    /// Parameters
    /// ----------
    /// event : Event
    ///     The Event upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// value : float
    ///     The value of the Variable for the given `event`
    ///
    fn value(&self, event: &PyEvent) -> Float {
        self.0.value(&event.0)
    }
    /// All values of this Variable on the given Dataset
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     The Dataset upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///     The values of the Variable for each Event in the given `dataset`
    ///
    fn value_on<'py>(&self, py: Python<'py>, dataset: &PyDataset) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.value_on(&dataset.0))
    }
}

/// A Variable used to define both spherical decay angles in the given frame
///
/// This class combines ``laddu.CosTheta`` and ``laddu.Phi`` into a single
/// object
///
/// Parameters
/// ----------
/// beam : int
///     The index of the `beam` particle
/// recoil : list of int
///     Indices of particles which are combined to form the recoiling particle (particles which
///     are not `beam` or part of the `resonance`)
/// daughter : list of int
///     Indices of particles which are combined to form one of the decay products of the
///     `resonance`
/// resonance : list of int
///     Indices of particles which are combined to form the `resonance`
/// frame : {'Helicity', 'HX', 'HEL', 'GottfriedJackson', 'Gottfried Jackson', 'GJ', 'Gottfried-Jackson'}
///     The frame to use in the  calculation
///
/// Raises
/// ------
/// ValueError
///     If `frame` is not one of the valid options
///
/// See Also
/// --------
/// laddu.CosTheta
/// laddu.Phi
///
#[pyclass(name = "Angles")]
#[derive(Clone)]
pub struct PyAngles(pub Angles);
#[pymethods]
impl PyAngles {
    #[new]
    #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
    fn new(
        beam: usize,
        recoil: Vec<usize>,
        daughter: Vec<usize>,
        resonance: Vec<usize>,
        frame: &str,
    ) -> PyResult<Self> {
        Ok(Self(Angles::new(
            beam,
            &recoil,
            &daughter,
            &resonance,
            frame.parse()?,
        )))
    }
    /// The Variable representing the cosine of the polar spherical decay angle
    ///
    /// Returns
    /// -------
    /// CosTheta
    ///
    #[getter]
    fn costheta(&self) -> PyCosTheta {
        PyCosTheta(self.0.costheta.clone())
    }
    // The Variable representing the polar azimuthal decay angle
    //
    // Returns
    // -------
    // Phi
    //
    #[getter]
    fn phi(&self) -> PyPhi {
        PyPhi(self.0.phi.clone())
    }
}

/// The polar angle of the given polarization vector with respect to the production plane
///
/// The `beam` and `recoil` particles define the plane of production, and this Variable
/// describes the polar angle of the `beam` relative to this plane
///
/// Parameters
/// ----------
/// beam : int
///     The index of the `beam` particle
/// recoil : list of int
///     Indices of particles which are combined to form the recoiling particle (particles which
///     are not `beam` or part of the `resonance`)
///
#[pyclass(name = "PolAngle")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPolAngle(pub PolAngle);

#[pymethods]
impl PyPolAngle {
    #[new]
    fn new(beam: usize, recoil: Vec<usize>) -> Self {
        Self(PolAngle::new(beam, &recoil))
    }
    /// The value of this Variable for the given Event
    ///
    /// Parameters
    /// ----------
    /// event : Event
    ///     The Event upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// value : float
    ///     The value of the Variable for the given `event`
    ///
    fn value(&self, event: &PyEvent) -> Float {
        self.0.value(&event.0)
    }
    /// All values of this Variable on the given Dataset
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     The Dataset upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///     The values of the Variable for each Event in the given `dataset`
    ///
    fn value_on<'py>(&self, py: Python<'py>, dataset: &PyDataset) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.value_on(&dataset.0))
    }
}

/// The magnitude of the given particle's polarization vector
///
/// This Variable simply represents the magnitude of the polarization vector of the particle
/// with the index `beam`
///
/// Parameters
/// ----------
/// beam : int
///     The index of the `beam` particle
///
/// See Also
/// --------
/// laddu.utils.vectors.Vector3.mag
///
#[pyclass(name = "PolMagnitude")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPolMagnitude(pub PolMagnitude);

#[pymethods]
impl PyPolMagnitude {
    #[new]
    fn new(beam: usize) -> Self {
        Self(PolMagnitude::new(beam))
    }
    /// The value of this Variable for the given Event
    ///
    /// Parameters
    /// ----------
    /// event : Event
    ///     The Event upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// value : float
    ///     The value of the Variable for the given `event`
    ///
    fn value(&self, event: &PyEvent) -> Float {
        self.0.value(&event.0)
    }
    /// All values of this Variable on the given Dataset
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     The Dataset upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///     The values of the Variable for each Event in the given `dataset`
    ///
    fn value_on<'py>(&self, py: Python<'py>, dataset: &PyDataset) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.value_on(&dataset.0))
    }
}

/// A Variable used to define both the polarization angle and magnitude of the given particle``
///
/// This class combines ``laddu.PolAngle`` and ``laddu.PolMagnitude`` into a single
/// object
///
/// Parameters
/// ----------
/// beam : int
///     The index of the `beam` particle
/// recoil : list of int
///     Indices of particles which are combined to form the recoiling particle (particles which
///     are not `beam` or part of the `resonance`)
///
/// See Also
/// --------
/// laddu.PolAngle
/// laddu.PolMagnitude
///
#[pyclass(name = "Polarization")]
#[derive(Clone)]
pub struct PyPolarization(pub Polarization);
#[pymethods]
impl PyPolarization {
    #[new]
    fn new(beam: usize, recoil: Vec<usize>) -> Self {
        PyPolarization(Polarization::new(beam, &recoil))
    }
    /// The Variable representing the magnitude of the polarization vector
    ///
    /// Returns
    /// -------
    /// PolMagnitude
    ///
    #[getter]
    fn pol_magnitude(&self) -> PyPolMagnitude {
        PyPolMagnitude(self.0.pol_magnitude)
    }
    /// The Variable representing the polar angle of the polarization vector
    ///
    /// Returns
    /// -------
    /// PolAngle
    ///
    #[getter]
    fn pol_angle(&self) -> PyPolAngle {
        PyPolAngle(self.0.pol_angle.clone())
    }
}

/// Mandelstam variables s, t, and u
///
/// By convention, the metric is chosen to be :math:`(+---)` and the variables are defined as follows
/// (ignoring factors of :math:`c`):
///
/// .. math:: s = (p_1 + p_2)^2 = (p_3 + p_4)^2
///
/// .. math:: t = (p_1 - p_3)^2 = (p_4 - p_2)^2
///
/// .. math:: u = (p_1 - p_4)^2 = (p_3 - p_2)^2
///
/// Parameters
/// ----------
/// p1: list of int
///     The indices of particles to combine to create :math:`p_1` in the diagram
/// p2: list of int
///     The indices of particles to combine to create :math:`p_2` in the diagram
/// p3: list of int
///     The indices of particles to combine to create :math:`p_3` in the diagram
/// p4: list of int
///     The indices of particles to combine to create :math:`p_4` in the diagram
/// channel: {'s', 't', 'u', 'S', 'T', 'U'}
///     The Mandelstam channel to calculate
///
/// Raises
/// ------
/// Exception
///     If more than one particle list is empty
/// ValueError
///     If `channel` is not one of the valid options
///
/// Notes
/// -----
/// At most one of the input particles may be omitted by using an empty list. This will cause
/// the calculation to use whichever equality listed above does not contain that particle.
///
/// By default, the first equality is used if no particle lists are empty.
///
#[pyclass(name = "Mandelstam")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyMandelstam(pub Mandelstam);

#[pymethods]
impl PyMandelstam {
    #[new]
    fn new(
        p1: Vec<usize>,
        p2: Vec<usize>,
        p3: Vec<usize>,
        p4: Vec<usize>,
        channel: &str,
    ) -> PyResult<Self> {
        Ok(Self(Mandelstam::new(p1, p2, p3, p4, channel.parse()?)?))
    }
    /// The value of this Variable for the given Event
    ///
    /// Parameters
    /// ----------
    /// event : Event
    ///     The Event upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// value : float
    ///     The value of the Variable for the given `event`
    ///
    fn value(&self, event: &PyEvent) -> Float {
        self.0.value(&event.0)
    }
    /// All values of this Variable on the given Dataset
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     The Dataset upon which the Variable is calculated
    ///
    /// Returns
    /// -------
    /// values : array_like
    ///     The values of the Variable for each Event in the given `dataset`
    ///
    fn value_on<'py>(&self, py: Python<'py>, dataset: &PyDataset) -> Bound<'py, PyArray1<Float>> {
        PyArray1::from_slice(py, &self.0.value_on(&dataset.0))
    }
}

/// An object which holds a registered ``Amplitude``
///
/// See Also
/// --------
/// laddu.Manager.register
///
#[pyclass(name = "AmplitudeID")]
#[derive(Clone)]
pub struct PyAmplitudeID(AmplitudeID);

/// A mathematical expression formed from AmplitudeIDs
///
#[pyclass(name = "Expression")]
#[derive(Clone)]
pub struct PyExpression(Expression);

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
#[pyclass(name = "Manager")]
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
#[pyclass(name = "Model")]
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
            serialize(&self.0)
                .map_err(LadduError::SerdeError)?
                .as_slice(),
        ))
    }
    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = PyModel(deserialize(state.as_bytes()).map_err(LadduError::SerdeError)?);
        Ok(())
    }
}

/// An Amplitude which can be registered by a Manager
///
/// See Also
/// --------
/// laddu.Manager
///
#[pyclass(name = "Amplitude")]
pub struct PyAmplitude(pub Box<dyn Amplitude>);

/// A class which can be used to evaluate a stored Expression
///
/// See Also
/// --------
/// laddu.Manager.load
///
#[pyclass(name = "Evaluator")]
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

pub trait GetStrExtractObj {
    fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>;
}

impl GetStrExtractObj for Bound<'_, PyDict> {
    fn get_extract<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        T: for<'py> FromPyObject<'py>,
    {
        self.get_item(key)?
            .map(|value| value.extract::<T>())
            .transpose()
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
#[pyclass(name = "ParameterLike")]
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

#[typetag::serde]
impl Variable for PyVariable {
    fn value_on(&self, dataset: &Arc<Dataset>) -> Vec<Float> {
        match self {
            PyVariable::Mass(mass) => mass.0.value_on(dataset),
            PyVariable::CosTheta(cos_theta) => cos_theta.0.value_on(dataset),
            PyVariable::Phi(phi) => phi.0.value_on(dataset),
            PyVariable::PolAngle(pol_angle) => pol_angle.0.value_on(dataset),
            PyVariable::PolMagnitude(pol_magnitude) => pol_magnitude.0.value_on(dataset),
            PyVariable::Mandelstam(mandelstam) => mandelstam.0.value_on(dataset),
        }
    }

    fn value(&self, event: &Event) -> Float {
        match self {
            PyVariable::Mass(mass) => mass.0.value(event),
            PyVariable::CosTheta(cos_theta) => cos_theta.0.value(event),
            PyVariable::Phi(phi) => phi.0.value(event),
            PyVariable::PolAngle(pol_angle) => pol_angle.0.value(event),
            PyVariable::PolMagnitude(pol_magnitude) => pol_magnitude.0.value(event),
            PyVariable::Mandelstam(mandelstam) => mandelstam.0.value(event),
        }
    }
}
