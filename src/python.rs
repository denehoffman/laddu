use ganesh::{mcmc::MCMCObserver, Observer};
use pyo3::{
    prelude::*,
    types::{PyTuple, PyTupleMethods},
};
#[cfg(feature = "rayon")]
use rayon::ThreadPool;

#[pymodule]
#[allow(non_snake_case, clippy::upper_case_acronyms)]
pub(crate) mod laddu {
    use std::array;
    use std::sync::Arc;

    use super::*;
    use crate as rust;
    use crate::likelihoods::LikelihoodTerm as RustLikelihoodTerm;
    use crate::likelihoods::MCMCOptions;
    use crate::likelihoods::MinimizerOptions;
    use crate::traits::ReadWrite;
    use crate::utils::variables::Variable;
    use crate::utils::vectors::{FourMomentum, FourVector, ThreeMomentum, ThreeVector};
    use crate::Float;
    #[cfg(feature = "rayon")]
    use crate::LadduError;
    use bincode::{deserialize, serialize};
    use fastrand::Rng;
    use ganesh::algorithms::lbfgsb::{LBFGSBFTerminator, LBFGSBGTerminator};
    use ganesh::algorithms::nelder_mead::{
        NelderMeadFTerminator, NelderMeadXTerminator, SimplexExpansionMethod,
    };
    use ganesh::algorithms::{NelderMead, LBFGSB};
    use ganesh::mcmc::aies::WeightedAIESMove;
    use ganesh::mcmc::ess::WeightedESSMove;
    use ganesh::mcmc::{AIESMove, ESSMove, AIES, ESS};
    use nalgebra::DVector;
    use num::Complex;
    use numpy::{PyArray1, PyArray2, PyArray3};
    use parking_lot::RwLock;
    use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
    use pyo3::types::PyBytes;
    use pyo3::types::{PyDict, PyList};
    #[cfg(feature = "rayon")]
    use rayon::ThreadPoolBuilder;

    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// A 3-momentum vector formed from Cartesian components
    ///
    /// Parameters
    /// ----------
    /// px, py, pz : float
    ///     The Cartesian components of the 3-vector
    ///
    #[pyclass]
    #[derive(Clone)]
    struct Vector3(nalgebra::Vector3<Float>);
    #[pymethods]
    impl Vector3 {
        #[new]
        fn new(px: Float, py: Float, pz: Float) -> Self {
            Self(nalgebra::Vector3::new(px, py, pz))
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
        fn with_mass(&self, mass: Float) -> Vector4 {
            Vector4(self.0.with_mass(mass))
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
        fn with_energy(&self, mass: Float) -> Vector4 {
            Vector4(self.0.with_energy(mass))
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
    #[pyclass]
    #[derive(Clone)]
    struct Vector4(nalgebra::Vector4<Float>);
    #[pymethods]
    impl Vector4 {
        #[new]
        fn new(px: Float, py: Float, pz: Float, e: Float) -> Self {
            Self(nalgebra::Vector4::new(px, py, pz, e))
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
        fn vec3(&self) -> Vector3 {
            Vector3(self.0.vec3().into())
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
        fn boost(&self, beta: &Vector3) -> Self {
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
        fn momentum(&self) -> Vector3 {
            Vector3(self.0.momentum().into())
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
        fn beta(&self) -> Vector3 {
            Vector3(self.0.beta())
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
    #[pyclass]
    #[derive(Clone)]
    struct Event(Arc<rust::data::Event>);

    #[pymethods]
    impl Event {
        #[new]
        pub(crate) fn new(p4s: Vec<Vector4>, eps: Vec<Vector3>, weight: Float) -> Self {
            Self(Arc::new(rust::data::Event {
                p4s: p4s.into_iter().map(|arr| arr.0).collect(),
                eps: eps.into_iter().map(|arr| arr.0).collect(),
                weight,
            }))
        }
        pub(crate) fn __str__(&self) -> String {
            self.0.to_string()
        }
        /// The list of 4-momenta for each particle in the event
        ///
        #[getter]
        pub(crate) fn get_p4s(&self) -> Vec<Vector4> {
            self.0.p4s.iter().map(|p4| Vector4(*p4)).collect()
        }
        /// The list of 3-vectors describing the polarization or helicity of particles in
        /// the event
        ///
        #[getter]
        pub(crate) fn get_eps(&self) -> Vec<Vector3> {
            self.0.eps.iter().map(|eps_vec| Vector3(*eps_vec)).collect()
        }
        /// The weight of this event relative to others in a Dataset
        ///
        #[getter]
        pub(crate) fn get_weight(&self) -> Float {
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
        pub(crate) fn get_p4_sum(&self, indices: Vec<usize>) -> Vector4 {
            Vector4(self.0.get_p4_sum(indices))
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
    #[pyclass]
    #[derive(Clone)]
    struct Dataset(Arc<rust::data::Dataset>);

    #[pymethods]
    impl Dataset {
        #[new]
        fn new(events: Vec<Event>) -> Self {
            Self(Arc::new(rust::data::Dataset {
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
        fn events(&self) -> Vec<Event> {
            self.0
                .events
                .iter()
                .map(|rust_event| Event(rust_event.clone()))
                .collect()
        }
        fn __getitem__(&self, index: usize) -> PyResult<Event> {
            self.0
                .get(index)
                .ok_or(PyIndexError::new_err("index out of range"))
                .map(|rust_event| Event(rust_event.clone()))
        }
        /// Separates a Dataset into histogram bins by a Variable value
        ///
        /// Currently supports ``laddu.Mass`` and ``laddu.Mandelstam`` as the binning variable.
        ///
        /// Parameters
        /// ----------
        /// variable : Mass or Mandelstam
        ///     The Variable by which each Event is binned
        /// bins : int
        ///     The number of equally-spaced bins
        /// range : tuple[float, float]
        ///     The minimum and maximum bin edges
        ///
        /// Returns
        /// -------
        /// datasets : BinnedDataset
        ///     A structure that holds a list of Datasets binned by the given `variable`
        ///
        /// See Also
        /// --------
        /// laddu.Mass
        /// laddu.Mandelstam
        ///
        #[pyo3(signature = (variable, bins, range))]
        fn bin_by(
            &self,
            variable: Bound<'_, PyAny>,
            bins: usize,
            range: (Float, Float),
        ) -> PyResult<BinnedDataset> {
            if let Ok(py_mass) = variable.extract::<PyRef<Mass>>() {
                Ok(BinnedDataset(self.0.bin_by(py_mass.0.clone(), bins, range)))
            } else if let Ok(py_mandelstam) = variable.extract::<PyRef<Mandelstam>>() {
                Ok(BinnedDataset(self.0.bin_by(
                    py_mandelstam.0.clone(),
                    bins,
                    range,
                )))
            } else {
                return Err(PyTypeError::new_err("Unsupported variable!"));
            }
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
        fn bootstrap(&self, seed: usize) -> Dataset {
            Dataset(self.0.bootstrap(seed))
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
    #[pyclass]
    struct BinnedDataset(rust::data::BinnedDataset);

    #[pymethods]
    impl BinnedDataset {
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
        fn __getitem__(&self, index: usize) -> PyResult<Dataset> {
            self.0
                .get(index)
                .ok_or(PyIndexError::new_err("index out of range"))
                .map(|rust_dataset| Dataset(rust_dataset.clone()))
        }
    }

    /// Open a Dataset from a file
    ///
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
    #[pyfunction]
    fn open(path: &str) -> PyResult<Dataset> {
        Ok(Dataset(rust::data::open(path)?))
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
    #[pyclass]
    #[derive(Clone)]
    struct Mass(rust::utils::variables::Mass);

    #[pymethods]
    impl Mass {
        #[new]
        fn new(constituents: Vec<usize>) -> Self {
            Self(rust::utils::variables::Mass::new(&constituents))
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
        fn value(&self, event: &Event) -> Float {
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
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
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
    /// See Also
    /// --------
    /// laddu.utils.vectors.Vector3.costheta
    ///
    #[pyclass]
    #[derive(Clone)]
    struct CosTheta(rust::utils::variables::CosTheta);

    #[pymethods]
    impl CosTheta {
        #[new]
        #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
        fn new(
            beam: usize,
            recoil: Vec<usize>,
            daughter: Vec<usize>,
            resonance: Vec<usize>,
            frame: &str,
        ) -> Self {
            Self(rust::utils::variables::CosTheta::new(
                beam,
                &recoil,
                &daughter,
                &resonance,
                frame.parse().unwrap(),
            ))
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
        fn value(&self, event: &Event) -> Float {
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
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
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
    /// See Also
    /// --------
    /// laddu.utils.vectors.Vector3.phi
    ///
    #[pyclass]
    #[derive(Clone)]
    struct Phi(rust::utils::variables::Phi);

    #[pymethods]
    impl Phi {
        #[new]
        #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
        fn new(
            beam: usize,
            recoil: Vec<usize>,
            daughter: Vec<usize>,
            resonance: Vec<usize>,
            frame: &str,
        ) -> Self {
            Self(rust::utils::variables::Phi::new(
                beam,
                &recoil,
                &daughter,
                &resonance,
                frame.parse().unwrap(),
            ))
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
        fn value(&self, event: &Event) -> Float {
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
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
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
    /// See Also
    /// --------
    /// laddu.CosTheta
    /// laddu.Phi
    ///
    #[pyclass]
    #[derive(Clone)]
    struct Angles(rust::utils::variables::Angles);
    #[pymethods]
    impl Angles {
        #[new]
        #[pyo3(signature=(beam, recoil, daughter, resonance, frame="Helicity"))]
        fn new(
            beam: usize,
            recoil: Vec<usize>,
            daughter: Vec<usize>,
            resonance: Vec<usize>,
            frame: &str,
        ) -> Self {
            Self(rust::utils::variables::Angles::new(
                beam,
                &recoil,
                &daughter,
                &resonance,
                frame.parse().unwrap(),
            ))
        }
        /// The Variable representing the cosine of the polar spherical decay angle
        ///
        /// Returns
        /// -------
        /// CosTheta
        ///
        #[getter]
        fn costheta(&self) -> CosTheta {
            CosTheta(self.0.costheta.clone())
        }
        // The Variable representing the polar azimuthal decay angle
        //
        // Returns
        // -------
        // Phi
        //
        #[getter]
        fn phi(&self) -> Phi {
            Phi(self.0.phi.clone())
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
    #[pyclass]
    #[derive(Clone)]
    struct PolAngle(rust::utils::variables::PolAngle);

    #[pymethods]
    impl PolAngle {
        #[new]
        fn new(beam: usize, recoil: Vec<usize>) -> Self {
            Self(rust::utils::variables::PolAngle::new(beam, &recoil))
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
        fn value(&self, event: &Event) -> Float {
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
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
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
    #[pyclass]
    #[derive(Clone)]
    struct PolMagnitude(rust::utils::variables::PolMagnitude);

    #[pymethods]
    impl PolMagnitude {
        #[new]
        fn new(beam: usize) -> Self {
            Self(rust::utils::variables::PolMagnitude::new(beam))
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
        fn value(&self, event: &Event) -> Float {
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
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
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
    #[pyclass]
    #[derive(Clone)]
    struct Polarization(rust::utils::variables::Polarization);
    #[pymethods]
    impl Polarization {
        #[new]
        fn new(beam: usize, recoil: Vec<usize>) -> Self {
            Polarization(rust::utils::variables::Polarization::new(beam, &recoil))
        }
        /// The Variable representing the magnitude of the polarization vector
        ///
        /// Returns
        /// -------
        /// PolMagnitude
        ///
        #[getter]
        fn pol_magnitude(&self) -> PolMagnitude {
            PolMagnitude(self.0.pol_magnitude)
        }
        /// The Variable representing the polar angle of the polarization vector
        ///
        /// Returns
        /// -------
        /// PolAngle
        ///
        #[getter]
        fn pol_angle(&self) -> PolAngle {
            PolAngle(self.0.pol_angle.clone())
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
    ///
    /// Notes
    /// -----
    /// At most one of the input particles may be omitted by using an empty list. This will cause
    /// the calculation to use whichever equality listed above does not contain that particle.
    ///
    /// By default, the first equality is used if no particle lists are empty.
    ///
    #[pyclass]
    #[derive(Clone)]
    struct Mandelstam(rust::utils::variables::Mandelstam);

    #[pymethods]
    impl Mandelstam {
        #[new]
        fn new(
            p1: Vec<usize>,
            p2: Vec<usize>,
            p3: Vec<usize>,
            p4: Vec<usize>,
            channel: &str,
        ) -> PyResult<Self> {
            Ok(Self(rust::utils::variables::Mandelstam::new(
                p1,
                p2,
                p3,
                p4,
                channel.parse().unwrap(),
            )?))
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
        fn value(&self, event: &Event) -> Float {
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
        fn value_on<'py>(&self, py: Python<'py>, dataset: &Dataset) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(py, &self.0.value_on(&dataset.0))
        }
    }

    /// An object which holds a registered ``Amplitude``
    ///
    /// See Also
    /// --------
    /// laddu.Manager.register
    ///
    #[pyclass]
    #[derive(Clone)]
    struct AmplitudeID(rust::amplitudes::AmplitudeID);

    /// A mathematical expression formed from AmplitudeIDs
    ///
    #[pyclass]
    #[derive(Clone)]
    pub(crate) struct Expression(pub(crate) rust::amplitudes::Expression);

    #[pymethods]
    impl AmplitudeID {
        /// The real part of a complex Amplitude
        ///
        /// Returns
        /// -------
        /// Expression
        ///     The real part of the given Amplitude
        ///
        fn real(&self) -> Expression {
            Expression(self.0.real())
        }
        /// The imaginary part of a complex Amplitude
        ///
        /// Returns
        /// -------
        /// Expression
        ///     The imaginary part of the given Amplitude
        ///
        fn imag(&self) -> Expression {
            Expression(self.0.imag())
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
        fn norm_sqr(&self) -> Expression {
            Expression(self.0.norm_sqr())
        }
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() + other_expr.0.clone()))
            } else if let Ok(other_int) = other.extract::<usize>() {
                if other_int == 0 {
                    Ok(Expression(rust::amplitudes::Expression::Amp(
                        self.0.clone(),
                    )))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(other_aid.0.clone() + self.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(other_expr.0.clone() + self.0.clone()))
            } else if let Ok(other_int) = other.extract::<usize>() {
                if other_int == 0 {
                    Ok(Expression(rust::amplitudes::Expression::Amp(
                        self.0.clone(),
                    )))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(other_aid.0.clone() * self.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(other_expr.0.clone() * self.0.clone()))
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
    impl Expression {
        /// The real part of a complex Expression
        ///
        /// Returns
        /// -------
        /// Expression
        ///     The real part of the given Expression
        ///
        fn real(&self) -> Expression {
            Expression(self.0.real())
        }
        /// The imaginary part of a complex Expression
        ///
        /// Returns
        /// -------
        /// Expression
        ///     The imaginary part of the given Expression
        ///
        fn imag(&self) -> Expression {
            Expression(self.0.imag())
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
        fn norm_sqr(&self) -> Expression {
            Expression(self.0.norm_sqr())
        }
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() + other_expr.0.clone()))
            } else if let Ok(other_int) = other.extract::<usize>() {
                if other_int == 0 {
                    Ok(Expression(self.0.clone()))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(other_aid.0.clone() + self.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(other_expr.0.clone() + self.0.clone()))
            } else if let Ok(other_int) = other.extract::<usize>() {
                if other_int == 0 {
                    Ok(Expression(self.0.clone()))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Expression> {
            if let Ok(other_aid) = other.extract::<PyRef<AmplitudeID>>() {
                Ok(Expression(other_aid.0.clone() * self.0.clone()))
            } else if let Ok(other_expr) = other.extract::<Expression>() {
                Ok(Expression(other_expr.0.clone() * self.0.clone()))
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
    #[pyclass]
    struct Manager(rust::amplitudes::Manager);

    #[pymethods]
    impl Manager {
        #[new]
        fn new() -> Self {
            Self(rust::amplitudes::Manager::default())
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
        fn register(&mut self, amplitude: &Amplitude) -> PyResult<AmplitudeID> {
            Ok(AmplitudeID(self.0.register(amplitude.0.clone())?))
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
        /// Notes
        /// -----
        /// While the given `expression` will be the one evaluated in the end, all registered
        /// Amplitudes will be loaded, and all of their parameters will be included in the final
        /// expression. These parameters will have no effect on evaluation, but they must be
        /// included in function calls.
        ///
        fn model(&self, expression: &Bound<'_, PyAny>) -> PyResult<Model> {
            let expression = if let Ok(expression) = expression.extract::<Expression>() {
                Ok(expression.0)
            } else if let Ok(aid) = expression.extract::<AmplitudeID>() {
                Ok(rust::amplitudes::Expression::Amp(aid.0))
            } else {
                Err(PyTypeError::new_err(
                    "'expression' must either by an Expression or AmplitudeID",
                ))
            }?;
            Ok(Model(self.0.model(&expression)))
        }
    }

    /// A class which represents a model composed of registered Amplitudes
    ///
    #[pyclass]
    struct Model(rust::amplitudes::Model);

    #[pymethods]
    impl Model {
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
        fn load(&self, dataset: &Dataset) -> PyResult<Evaluator> {
            Ok(Evaluator(self.0.load(&dataset.0)))
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
        /// Notes
        /// -----
        /// Valid file path names must have either the ".pickle" or ".pkl" extension
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
        /// Notes
        /// -----
        /// Valid file path names must have either the ".pickle" or ".pkl" extension
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(Model(crate::amplitudes::Model::load_from(path)?))
        }
        #[new]
        fn new() -> Self {
            Model(crate::amplitudes::Model::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(py, serialize(&self.0).unwrap().as_slice()))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = Model(deserialize(state.as_bytes()).unwrap());
            Ok(())
        }
    }

    /// An Amplitude which can be registered by a Manager
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    #[pyclass]
    struct Amplitude(Box<dyn rust::amplitudes::Amplitude>);

    /// A class which can be used to evaluate a stored Expression
    ///
    /// See Also
    /// --------
    /// laddu.Manager.load
    ///
    #[pyclass]
    #[derive(Clone)]
    struct Evaluator(rust::amplitudes::Evaluator);

    #[pymethods]
    impl Evaluator {
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
        ///
        fn activate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.activate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.activate_many(&vec);
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
        ///
        fn deactivate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.deactivate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.deactivate_many(&vec);
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
        ///
        fn isolate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.isolate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.isolate_many(&vec);
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
                .expect(
                    "Gradients do not have the same length (please make an issue report on GitHub)",
                ))
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
                .expect(
                    "Gradients do not have the same length (please make an issue report on GitHub)",
                ))
            }
        }
    }

    trait GetStrExtractObj {
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

    fn _parse_minimizer_options(
        n_parameters: usize,
        method: &str,
        max_steps: usize,
        debug: bool,
        verbose: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<MinimizerOptions> {
        let mut options = MinimizerOptions::default();
        let mut show_step = true;
        let mut show_x = true;
        let mut show_fx = true;
        if let Some(kwargs) = kwargs {
            show_step = kwargs.get_extract::<bool>("show_step")?.unwrap_or(true);
            show_x = kwargs.get_extract::<bool>("show_x")?.unwrap_or(true);
            show_fx = kwargs.get_extract::<bool>("show_fx")?.unwrap_or(true);
            let tol_x_rel = kwargs
                .get_extract::<Float>("tol_x_rel")?
                .unwrap_or(Float::EPSILON);
            let tol_x_abs = kwargs
                .get_extract::<Float>("tol_x_abs")?
                .unwrap_or(Float::EPSILON);
            let tol_f_rel = kwargs
                .get_extract::<Float>("tol_f_rel")?
                .unwrap_or(Float::EPSILON);
            let tol_f_abs = kwargs
                .get_extract::<Float>("tol_f_abs")?
                .unwrap_or(Float::EPSILON);
            let tol_g_abs = kwargs
                .get_extract::<Float>("tol_g_abs")?
                .unwrap_or(Float::cbrt(Float::EPSILON));
            let g_tolerance = kwargs.get_extract::<Float>("g_tolerance")?.unwrap_or(1e-5);
            let adaptive = kwargs.get_extract::<bool>("adaptive")?.unwrap_or(false);
            let alpha = kwargs.get_extract::<Float>("alpha")?;
            let beta = kwargs.get_extract::<Float>("beta")?;
            let gamma = kwargs.get_extract::<Float>("gamma")?;
            let delta = kwargs.get_extract::<Float>("delta")?;
            let simplex_expansion_method = kwargs
                .get_extract::<String>("simplex_expansion_method")?
                .unwrap_or("greedy minimization".into());
            let nelder_mead_f_terminator = kwargs
                .get_extract::<String>("nelder_mead_f_terminator")?
                .unwrap_or("stddev".into());
            let nelder_mead_x_terminator = kwargs
                .get_extract::<String>("nelder_mead_x_terminator")?
                .unwrap_or("singer".into());
            let threads = kwargs
                .get_extract::<usize>("threads")
                .unwrap_or(None)
                .unwrap_or_else(num_cpus::get);
            let mut observers: Vec<Arc<RwLock<PyObserver>>> = Vec::default();
            if let Ok(Some(observer_arg)) = kwargs.get_item("observers") {
                if let Ok(observer_list) = observer_arg.downcast::<PyList>() {
                    for item in observer_list.iter() {
                        let observer = item.extract::<PyObserver>()?;
                        observers.push(Arc::new(RwLock::new(observer)));
                    }
                } else if let Ok(single_observer) = observer_arg.extract::<PyObserver>() {
                    observers.push(Arc::new(RwLock::new(single_observer)));
                } else {
                    return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single Observer or a list of Observers!"));
                }
            }
            for observer in observers {
                options = options.with_observer(observer);
            }
            match method {
                "lbfgsb" => {
                    options = options.with_algorithm(
                        LBFGSB::default()
                            .with_terminator_f(LBFGSBFTerminator { tol_f_abs })
                            .with_terminator_g(LBFGSBGTerminator { tol_g_abs })
                            .with_g_tolerance(g_tolerance),
                    )
                }
                "nelder_mead" => {
                    let terminator_f = match nelder_mead_f_terminator.as_str() {
                        "amoeba" => NelderMeadFTerminator::Amoeba { tol_f_rel },
                        "absolute" => NelderMeadFTerminator::Absolute { tol_f_abs },
                        "stddev" => NelderMeadFTerminator::StdDev { tol_f_abs },
                        "none" => NelderMeadFTerminator::None,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"nelder_mead_f_terminator\": \"{}\"",
                                nelder_mead_f_terminator
                            )))
                        }
                    };
                    let terminator_x = match nelder_mead_x_terminator.as_str() {
                        "diameter" => NelderMeadXTerminator::Diameter { tol_x_abs },
                        "higham" => NelderMeadXTerminator::Higham { tol_x_rel },
                        "rowan" => NelderMeadXTerminator::Rowan { tol_x_rel },
                        "singer" => NelderMeadXTerminator::Singer { tol_x_rel },
                        "none" => NelderMeadXTerminator::None,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"nelder_mead_x_terminator\": \"{}\"",
                                nelder_mead_x_terminator
                            )))
                        }
                    };
                    let simplex_expansion_method = match simplex_expansion_method.as_str() {
                        "greedy minimization" => SimplexExpansionMethod::GreedyMinimization,
                        "greedy expansion" => SimplexExpansionMethod::GreedyExpansion,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "Invalid \"simplex_expansion_method\": \"{}\"",
                                simplex_expansion_method
                            )))
                        }
                    };
                    let mut nelder_mead = NelderMead::default()
                        .with_terminator_f(terminator_f)
                        .with_terminator_x(terminator_x)
                        .with_expansion_method(simplex_expansion_method);
                    if adaptive {
                        nelder_mead = nelder_mead.with_adaptive(n_parameters);
                    }
                    if let Some(alpha) = alpha {
                        nelder_mead = nelder_mead.with_alpha(alpha);
                    }
                    if let Some(beta) = beta {
                        nelder_mead = nelder_mead.with_beta(beta);
                    }
                    if let Some(gamma) = gamma {
                        nelder_mead = nelder_mead.with_gamma(gamma);
                    }
                    if let Some(delta) = delta {
                        nelder_mead = nelder_mead.with_delta(delta);
                    }
                    options = options.with_algorithm(nelder_mead)
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid \"method\": \"{}\"",
                        method
                    )))
                }
            }
            options = options.with_threads(threads);
        }
        if debug {
            options = options.debug();
        }
        if verbose {
            options = options.verbose(show_step, show_x, show_fx);
        }
        options = options.with_max_steps(max_steps);
        Ok(options)
    }

    fn _parse_mcmc_options(
        method: &str,
        debug: bool,
        verbose: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
        rng: Rng,
    ) -> PyResult<MCMCOptions> {
        let default_ess_moves = [ESSMove::differential(0.9), ESSMove::gaussian(0.1)];
        let default_aies_moves = [AIESMove::stretch(0.9), AIESMove::walk(0.1)];
        let mut options = MCMCOptions::new_ess(default_ess_moves, rng.clone());
        if let Some(kwargs) = kwargs {
            let n_adaptive = kwargs.get_extract::<usize>("n_adaptive")?.unwrap_or(100);
            let mu = kwargs.get_extract::<Float>("mu")?.unwrap_or(1.0);
            let max_ess_steps = kwargs
                .get_extract::<usize>("max_ess_steps")?
                .unwrap_or(10000);
            let mut ess_moves: Vec<WeightedESSMove> = Vec::default();
            if let Ok(Some(ess_move_list_arg)) = kwargs.get_item("ess_moves") {
                if let Ok(ess_move_list) = ess_move_list_arg.downcast::<PyList>() {
                    for item in ess_move_list.iter() {
                        let item_tuple = item.downcast::<PyTuple>()?;
                        let move_name = item_tuple.get_item(0)?.extract::<String>()?;
                        let move_weight = item_tuple.get_item(1)?.extract::<Float>()?;
                        match move_name.to_lowercase().as_ref() {
                            "differential" => ess_moves.push(ESSMove::differential(move_weight)),
                            "gaussian" => ess_moves.push(ESSMove::gaussian(move_weight)),
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "Unknown ESS move type: {}",
                                    move_name
                                )))
                            }
                        }
                    }
                }
            }
            if ess_moves.is_empty() {
                ess_moves = default_ess_moves.to_vec();
            }
            let mut aies_moves: Vec<WeightedAIESMove> = Vec::default();
            if let Ok(Some(aies_move_list_arg)) = kwargs.get_item("aies_moves") {
                if let Ok(aies_move_list) = aies_move_list_arg.downcast::<PyList>() {
                    for item in aies_move_list.iter() {
                        let item_tuple = item.downcast::<PyTuple>()?;
                        if let Ok(move_name) = item_tuple.get_item(0)?.extract::<String>() {
                            let move_weight = item_tuple.get_item(1)?.extract::<Float>()?;
                            match move_name.to_lowercase().as_ref() {
                                "stretch" => aies_moves.push(AIESMove::stretch(move_weight)),
                                "walk" => aies_moves.push(AIESMove::walk(move_weight)),
                                _ => {
                                    return Err(PyValueError::new_err(format!(
                                        "Unknown AIES move type: {}",
                                        move_name
                                    )))
                                }
                            }
                        } else if let Ok(move_spec) = item_tuple.get_item(0)?.downcast::<PyTuple>()
                        {
                            let move_name = move_spec.get_item(0)?.extract::<String>()?;
                            let move_weight = item_tuple.get_item(1)?.extract::<Float>()?;
                            if move_name.to_lowercase() == "stretch" {
                                let a = move_spec.get_item(1)?.extract::<Float>()?;
                                aies_moves.push((AIESMove::Stretch { a }, move_weight))
                            } else {
                                return Err(PyValueError::new_err(
                                    "Only the 'stretch' move has a hyperparameter",
                                ));
                            }
                        }
                    }
                }
            }
            if aies_moves.is_empty() {
                aies_moves = default_aies_moves.to_vec();
            }
            let threads = kwargs
                .get_extract::<usize>("threads")
                .unwrap_or(None)
                .unwrap_or_else(num_cpus::get);
            #[cfg(feature = "rayon")]
            let mut observers: Vec<
                Arc<RwLock<dyn ganesh::mcmc::MCMCObserver<ThreadPool>>>,
            > = Vec::default();
            #[cfg(not(feature = "rayon"))]
            let mut observers: Vec<Arc<RwLock<dyn ganesh::mcmc::MCMCObserver<()>>>> =
                Vec::default();
            if let Ok(Some(observer_arg)) = kwargs.get_item("observers") {
                if let Ok(observer_list) = observer_arg.downcast::<PyList>() {
                    for item in observer_list.iter() {
                        if let Ok(observer) = item.downcast::<AutocorrelationObserver>() {
                            observers.push(observer.borrow().0.clone());
                        } else if let Ok(observer) = item.extract::<PyMCMCObserver>() {
                            observers.push(Arc::new(RwLock::new(observer)));
                        }
                    }
                } else if let Ok(single_observer) =
                    observer_arg.downcast::<AutocorrelationObserver>()
                {
                    observers.push(single_observer.borrow().0.clone());
                } else if let Ok(single_observer) = observer_arg.extract::<PyMCMCObserver>() {
                    observers.push(Arc::new(RwLock::new(single_observer)));
                } else {
                    return Err(PyTypeError::new_err("The keyword argument \"observers\" must either be a single MCMCObserver or a list of MCMCObservers!"));
                }
            }
            for observer in observers {
                options = options.with_observer(observer.clone());
            }
            match method.to_lowercase().as_ref() {
                "ess" => {
                    options = options.with_algorithm(
                        ESS::new(ess_moves, rng)
                            .with_mu(mu)
                            .with_n_adaptive(n_adaptive)
                            .with_max_steps(max_ess_steps),
                    )
                }
                "aies" => options = options.with_algorithm(AIES::new(aies_moves, rng)),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid \"method\": \"{}\"",
                        method
                    )))
                }
            }
            options = options.with_threads(threads);
        }
        if debug {
            options = options.debug();
        }
        if verbose {
            options = options.verbose();
        }
        Ok(options)
    }

    /// A (extended) negative log-likelihood evaluator
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     The Model to evaluate
    /// ds_data : Dataset
    ///     A Dataset representing true signal data
    /// ds_accmc : Dataset
    ///     A Dataset of physically flat accepted Monte Carlo data used for normalization
    ///
    #[pyclass]
    #[derive(Clone)]
    struct NLL(Box<rust::likelihoods::NLL>);

    #[pymethods]
    impl NLL {
        #[new]
        #[pyo3(signature = (model, ds_data, ds_accmc))]
        fn new(model: &Model, ds_data: &Dataset, ds_accmc: &Dataset) -> PyResult<Self> {
            Ok(Self(rust::likelihoods::NLL::new(
                &model.0,
                &ds_data.0,
                &ds_accmc.0,
            )))
        }
        /// The underlying signal dataset used in calculating the NLL
        ///
        /// Returns
        /// -------
        /// Dataset
        ///
        #[getter]
        fn data(&self) -> Dataset {
            Dataset(self.0.data_evaluator.dataset.clone())
        }
        /// The underlying accepted Monte Carlo dataset used in calculating the NLL
        ///
        /// Returns
        /// -------
        /// Dataset
        ///
        #[getter]
        fn accmc(&self) -> Dataset {
            Dataset(self.0.accmc_evaluator.dataset.clone())
        }
        /// Turn an ``NLL`` into a term that can be used by a ``LikelihoodManager``
        ///
        /// Returns
        /// -------
        /// term : LikelihoodTerm
        ///     The isolated NLL which can be used to build more complex models
        ///
        fn as_term(&self) -> LikelihoodTerm {
            LikelihoodTerm(self.0.clone())
        }
        /// The names of the free parameters used to evaluate the NLL
        ///
        /// Returns
        /// -------
        /// parameters : list of str
        ///
        #[getter]
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        /// Activates Amplitudes in the NLL by name
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
        ///
        fn activate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.activate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.activate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        /// Activates all Amplitudes in the JNLL
        ///
        fn activate_all(&self) {
            self.0.activate_all();
        }
        /// Deactivates Amplitudes in the NLL by name
        ///
        /// Deactivated Amplitudes act as zeros in the NLL
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
        ///
        fn deactivate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.deactivate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.deactivate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        /// Deactivates all Amplitudes in the NLL
        ///
        fn deactivate_all(&self) {
            self.0.deactivate_all();
        }
        /// Isolates Amplitudes in the NLL by name
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
        ///
        fn isolate(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
            if let Ok(string_arg) = arg.extract::<String>() {
                self.0.isolate(&string_arg);
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                self.0.isolate_many(&vec);
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            }
            Ok(())
        }
        /// Evaluate the extended negative log-likelihood over the stored Datasets
        ///
        /// This is defined as
        ///
        /// .. math:: NLL(\vec{p}; D, MC) = -2 \left( \sum_{e \in D} (e_w \log(\mathcal{L}(e))) - \frac{1}{N_{MC}} \sum_{e \in MC} (e_w \mathcal{L}(e)) \right)
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
        /// result : float
        ///     The total negative log-likelihood
        ///
        #[pyo3(signature = (parameters, *, threads=None))]
        fn evaluate(&self, parameters: Vec<Float>, threads: Option<usize>) -> PyResult<Float> {
            #[cfg(feature = "rayon")]
            {
                Ok(ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| self.0.evaluate(&parameters)))
            }
            #[cfg(not(feature = "rayon"))]
            {
                Ok(self.0.evaluate(&parameters))
            }
        }
        /// Evaluate the gradient of the negative log-likelihood over the stored Dataset
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
        ///     A ``numpy`` array of representing the gradient of the negative log-likelihood over each parameter
        ///
        #[pyo3(signature = (parameters, *, threads=None))]
        fn evaluate_gradient<'py>(
            &self,
            py: Python<'py>,
            parameters: Vec<Float>,
            threads: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray1<Float>>> {
            #[cfg(feature = "rayon")]
            {
                Ok(PyArray1::from_slice(
                    py,
                    ThreadPoolBuilder::new()
                        .num_threads(threads.unwrap_or_else(num_cpus::get))
                        .build()
                        .map_err(LadduError::from)?
                        .install(|| self.0.evaluate_gradient(&parameters))
                        .as_slice(),
                ))
            }
            #[cfg(not(feature = "rayon"))]
            {
                Ok(PyArray1::from_slice(
                    py,
                    self.0.evaluate_gradient(&parameters).as_slice(),
                ))
            }
        }
        /// Project the model over the Monte Carlo dataset with the given parameter values
        ///
        /// This is defined as
        ///
        /// .. math:: e_w(\vec{p}) = \frac{e_w}{N_{MC}} \mathcal{L}(e)
        ///
        /// Parameters
        /// ----------
        /// parameters : list of float
        ///     The values to use for the free parameters
        /// mc_evaluator: Evaluator, optional
        ///     Project using the given Evaluator or use the stored ``accmc`` if None
        /// threads : int, optional
        ///     The number of threads to use (setting this to None will use all available CPUs)
        ///
        /// Returns
        /// -------
        /// result : array_like
        ///     Weights for every Monte Carlo event which represent the fit to data
        ///
        #[pyo3(signature = (parameters, *, mc_evaluator = None, threads=None))]
        fn project<'py>(
            &self,
            py: Python<'py>,
            parameters: Vec<Float>,
            mc_evaluator: Option<Evaluator>,
            threads: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray1<Float>>> {
            #[cfg(feature = "rayon")]
            {
                Ok(PyArray1::from_slice(
                    py,
                    ThreadPoolBuilder::new()
                        .num_threads(threads.unwrap_or_else(num_cpus::get))
                        .build()
                        .map_err(LadduError::from)?
                        .install(|| {
                            self.0
                                .project(&parameters, mc_evaluator.map(|pyeval| pyeval.0.clone()))
                        })
                        .as_slice(),
                ))
            }
            #[cfg(not(feature = "rayon"))]
            {
                Ok(PyArray1::from_slice(
                    py,
                    self.0
                        .project(&parameters, mc_evaluator.map(|pyeval| pyeval.0.clone()))
                        .as_slice(),
                ))
            }
        }

        /// Project the model over the Monte Carlo dataset with the given parameter values, first
        /// isolating the given terms by name. The NLL is then reset to its previous state of
        /// activation.
        ///
        /// This is defined as
        ///
        /// .. math:: e_w(\vec{p}) = \frac{e_w}{N_{MC}} \mathcal{L}(e)
        ///
        /// Parameters
        /// ----------
        /// parameters : list of float
        ///     The values to use for the free parameters
        /// arg : str or list of str
        ///     Names of Amplitudes to be isolated
        /// mc_evaluator: Evaluator, optional
        ///     Project using the given Evaluator or use the stored ``accmc`` if None
        /// threads : int, optional
        ///     The number of threads to use (setting this to None will use all available CPUs)
        ///
        /// Returns
        /// -------
        /// result : array_like
        ///     Weights for every Monte Carlo event which represent the fit to data
        ///
        /// Raises
        /// ------
        /// TypeError
        ///     If `arg` is not a str or list of str
        ///
        #[pyo3(signature = (parameters, arg, *, mc_evaluator = None, threads=None))]
        fn project_with<'py>(
            &self,
            py: Python<'py>,
            parameters: Vec<Float>,
            arg: &Bound<'_, PyAny>,
            mc_evaluator: Option<Evaluator>,
            threads: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray1<Float>>> {
            let names = if let Ok(string_arg) = arg.extract::<String>() {
                vec![string_arg]
            } else if let Ok(list_arg) = arg.downcast::<PyList>() {
                let vec: Vec<String> = list_arg.extract()?;
                vec
            } else {
                return Err(PyTypeError::new_err(
                    "Argument must be either a string or a list of strings",
                ));
            };
            #[cfg(feature = "rayon")]
            {
                Ok(PyArray1::from_slice(
                    py,
                    ThreadPoolBuilder::new()
                        .num_threads(threads.unwrap_or_else(num_cpus::get))
                        .build()
                        .map_err(LadduError::from)?
                        .install(|| {
                            self.0.project_with(
                                &parameters,
                                &names,
                                mc_evaluator.map(|pyeval| pyeval.0.clone()),
                            )
                        })
                        .as_slice(),
                ))
            }
            #[cfg(not(feature = "rayon"))]
            {
                Ok(PyArray1::from_slice(
                    py,
                    self.0
                        .project_with(
                            &parameters,
                            &names,
                            mc_evaluator.map(|pyeval| pyeval.0.clone()),
                        )
                        .as_slice(),
                ))
            }
        }

        /// Minimize the NLL with respect to the free parameters in the model
        ///
        /// This method "runs the fit". Given an initial position `p0` and optional `bounds`, this
        /// method performs a minimization over the negative log-likelihood, optimizing the model
        /// over the stored signal data and Monte Carlo.
        ///
        /// Parameters
        /// ----------
        /// p0 : array_like
        ///     The initial parameters at the start of optimization
        /// bounds : list of tuple of float, optional
        ///     A list of lower and upper bound pairs for each parameter (use ``None`` for no bound)
        /// method : {'lbfgsb', 'nelder-mead', 'nelder_mead'}
        ///     The minimization algorithm to use (see additional parameters for fine-tuning)
        /// max_steps : int, default=4000
        ///     The maximum number of algorithm steps to perform
        /// debug : bool, default=False
        ///     Set to ``True`` to print out debugging information at each step
        /// verbose : bool, default=False
        ///     Set to ``True`` to print verbose information at each step
        /// show_step : bool, default=True
        ///     Include step number in verbose output
        /// show_x : bool, default=True
        ///     Include current best position in verbose output
        /// show_fx : bool, default=True
        ///     Include current best NLL in verbose output
        /// observers : Observer or list of Observers
        ///     Callback functions which are applied after every algorithm step
        /// tol_x_rel : float
        ///     The relative position tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_x_abs : float
        ///     The absolute position tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_f_rel : float
        ///     The relative function tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_f_abs : float
        ///     The absolute function tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_g_abs : float
        ///     The absolute gradient tolerance used by termination methods (default is the cube
        ///     root of machine epsilon)
        /// g_tolerance : float, default=1e-5
        ///     Another gradient tolerance used by termination methods (particularly L-BFGS-B)
        /// adaptive : bool, default=False
        ///     Use adaptive values for Nelder-Mead parameters
        /// alpha : float, optional
        ///     Overwrite the default :math:`\alpha` parameter in the Nelder-Mead algorithm
        /// beta : float, optional
        ///     Overwrite the default :math:`\beta` parameter in the Nelder-Mead algorithm
        /// gamma : float, optional
        ///     Overwrite the default :math:`\gamma` parameter in the Nelder-Mead algorithm
        /// delta : float, optional
        ///     Overwrite the default :math:`\delta` parameter in the Nelder-Mead algorithm
        /// simplex_expansion_method : {'greedy_minimization', 'greedy_expansion'}
        ///     The expansion method used by the Nelder-Mead algorithm
        /// nelder_mead_f_terminator : {'stddev', 'absolute', 'stddev', 'none'}
        ///     The function terminator used by the Nelder-Mead algorithm
        /// nelder_mead_x_terminator : {'singer', 'diameter', 'rowan', 'higham', 'none'}
        ///     The positional terminator used by the Nelder-Mead algorithm
        /// threads : int, optional
        ///     The number of threads to use (setting this to None will use all available CPUs)
        ///
        #[pyo3(signature = (p0, *, bounds=None, method="lbfgsb", max_steps=4000, debug=false, verbose=false, **kwargs))]
        #[allow(clippy::too_many_arguments)]
        fn minimize(
            &self,
            p0: Vec<Float>,
            bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
            method: &str,
            max_steps: usize,
            debug: bool,
            verbose: bool,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Status> {
            let bounds = bounds.map(|bounds_vec| {
                bounds_vec
                    .iter()
                    .map(|(opt_lb, opt_ub)| {
                        (
                            opt_lb.unwrap_or(Float::NEG_INFINITY),
                            opt_ub.unwrap_or(Float::INFINITY),
                        )
                    })
                    .collect()
            });
            let n_parameters = p0.len();
            let options =
                _parse_minimizer_options(n_parameters, method, max_steps, debug, verbose, kwargs)?;
            let status = self.0.minimize(&p0, bounds, Some(options))?;
            Ok(Status(status))
        }
        /// Run an MCMC algorithm on the free parameters of the NLL's model
        ///
        /// This method can be used to sample the underlying log-likelihood given an initial
        /// position for each walker `p0`.
        ///
        /// Parameters
        /// ----------
        /// p0 : array_like
        ///     The initial parameters at the start of optimization
        /// n_steps : int,
        ///     The number of MCMC steps each walker should take
        /// method : {'ESS', 'AIES'}
        ///     The MCMC algorithm to use (see additional parameters for fine-tuning)
        /// debug : bool, default=False
        ///     Set to ``True`` to print out debugging information at each step
        /// verbose : bool, default=False
        ///     Set to ``True`` to print verbose information at each step
        /// seed : int,
        ///     The seed for the random number generator
        /// ess_moves : list of tuple
        ///     A list of moves for the ESS algorithm (see notes)
        /// aies_moves : list of tuple
        ///     A list of moves for the AIES algorithm (see notes)
        /// n_adaptive : int, default=100
        ///     Number of adaptive ESS steps to perform at the start of sampling
        /// mu : float, default=1.0
        ///     ESS adaptive parameter
        /// max_ess_steps : int, default=10000
        ///     The maximum number of slice expansions/contractions performed in the ESS algorithm
        /// threads : int, optional
        ///     The number of threads to use (setting this to None will use all available CPUs)
        ///
        /// Returns
        /// -------
        /// Ensemble
        ///     The resulting ensemble of walkers
        ///
        /// Notes
        /// -----
        /// Moves may be specified as tuples of ``(move name, usage weight)`` where the move name
        /// depends on the algorithm and the usage weight gives the proportion of time that move is
        /// used relative to the others in the list.
        ///
        /// For the Ensemble Slice Sampler (ESS) algorithm, valid move types are "differential" and
        /// "gaussian", and the default move set is ``[("differential", 0.9), ("gaussian", 0.1)]``.
        ///
        /// For the Affine Invariant Ensemble Sampler (AIES) algorithm, valid move types are
        /// "stretch" and "walk", and the default move set is ``[("stretch", 0.9), ("walk", 0.1)]``.
        ///
        /// For AIES, the "stretch" move can also be given with an adaptive parameter ``a``
        /// (default=``2``). To add a stretch move with a different value of ``a``, the "move name"
        /// can be instead given as a tuple ``(move name, a)``. For example, ``(("stretch", 2.2), 0.3)``
        /// creates a stretch move with ``a=2.2`` and usage weight of ``0.3``.
        ///
        /// Since MCMC methods are inclined to sample maxima rather than minima, the underlying
        /// function sign is automatically flipped when calling this method.
        ///
        #[pyo3(signature = (p0, n_steps, *, method="ESS", debug=false, verbose=false, seed=0, **kwargs))]
        #[allow(clippy::too_many_arguments)]
        fn mcmc(
            &self,
            p0: Vec<Vec<Float>>,
            n_steps: usize,
            method: &str,
            debug: bool,
            verbose: bool,
            seed: u64,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Ensemble> {
            let p0 = p0.into_iter().map(DVector::from_vec).collect::<Vec<_>>();
            let mut rng = Rng::new();
            rng.seed(seed);
            let options = _parse_mcmc_options(method, debug, verbose, kwargs, rng.clone())?;
            let ensemble = self.0.mcmc(&p0, n_steps, Some(options), rng)?;
            Ok(Ensemble(ensemble))
        }
    }

    /// A term in an expression with multiple likelihood components
    ///
    /// See Also
    /// --------
    /// NLL.as_term
    ///
    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodTerm(Box<dyn rust::likelihoods::LikelihoodTerm>);

    /// An object which holds a registered ``LikelihoodTerm``
    ///
    /// See Also
    /// --------
    /// laddu.LikelihoodManager.register
    ///
    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodID(rust::likelihoods::LikelihoodID);

    /// A mathematical expression formed from LikelihoodIDs
    ///
    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodExpression(rust::likelihoods::LikelihoodExpression);

    #[pymethods]
    impl LikelihoodID {
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() + other_expr.0.clone()))
            } else if let Ok(int) = other.extract::<usize>() {
                if int == 0 {
                    Ok(LikelihoodExpression(
                        rust::likelihoods::LikelihoodExpression::Term(self.0.clone()),
                    ))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(other_aid.0.clone() + self.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(other_expr.0.clone() + self.0.clone()))
            } else if let Ok(int) = other.extract::<usize>() {
                if int == 0 {
                    Ok(LikelihoodExpression(
                        rust::likelihoods::LikelihoodExpression::Term(self.0.clone()),
                    ))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(other_aid.0.clone() * self.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(other_expr.0.clone() * self.0.clone()))
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
    impl LikelihoodExpression {
        fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() + other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() + other_expr.0.clone()))
            } else if let Ok(int) = other.extract::<usize>() {
                if int == 0 {
                    Ok(LikelihoodExpression(self.0.clone()))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(other_aid.0.clone() + self.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(other_expr.0.clone() + self.0.clone()))
            } else if let Ok(int) = other.extract::<usize>() {
                if int == 0 {
                    Ok(LikelihoodExpression(self.0.clone()))
                } else {
                    Err(PyTypeError::new_err(
                        "Addition with an integer for this type is only defined for 0",
                    ))
                }
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for +"))
            }
        }
        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() * other_expr.0.clone()))
            } else {
                Err(PyTypeError::new_err("Unsupported operand type for *"))
            }
        }
        fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<LikelihoodExpression> {
            if let Ok(other_aid) = other.extract::<PyRef<LikelihoodID>>() {
                Ok(LikelihoodExpression(self.0.clone() * other_aid.0.clone()))
            } else if let Ok(other_expr) = other.extract::<LikelihoodExpression>() {
                Ok(LikelihoodExpression(self.0.clone() * other_expr.0.clone()))
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

    /// A class which can be used to register LikelihoodTerms and store precalculated data
    ///
    #[pyclass]
    #[derive(Clone)]
    struct LikelihoodManager(rust::likelihoods::LikelihoodManager);

    #[pymethods]
    impl LikelihoodManager {
        #[new]
        fn new() -> Self {
            Self(rust::likelihoods::LikelihoodManager::default())
        }
        /// Register a LikelihoodTerm with the LikelihoodManager
        ///
        /// Parameters
        /// ----------
        /// term : LikelihoodTerm
        ///     The LikelihoodTerm to register
        ///
        /// Returns
        /// -------
        /// LikelihoodID
        ///     A reference to the registered ``likelihood`` that can be used to form complex
        ///     LikelihoodExpressions
        ///
        fn register(&mut self, likelihood_term: &LikelihoodTerm) -> LikelihoodID {
            LikelihoodID(self.0.register(likelihood_term.0.clone()))
        }
        /// The free parameters used by all terms in the LikelihoodManager
        ///
        /// Returns
        /// -------
        /// parameters : list of str
        ///     The list of parameter names
        ///
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        /// Load a LikelihoodExpression by precalculating each term over their internal Datasets
        ///
        /// Parameters
        /// ----------
        /// likelihood_expression : LikelihoodExpression
        ///     The expression to use in precalculation
        ///
        /// Returns
        /// -------
        /// LikelihoodEvaluator
        ///     An object that can be used to evaluate the `likelihood_expression` over all managed
        ///     terms
        ///
        /// Notes
        /// -----
        /// While the given `likelihood_expression` will be the one evaluated in the end, all registered
        /// Amplitudes will be loaded, and all of their parameters will be included in the final
        /// expression. These parameters will have no effect on evaluation, but they must be
        /// included in function calls.
        ///
        /// See Also
        /// --------
        /// LikelihoodManager.parameters
        ///
        fn load(&self, likelihood_expression: &LikelihoodExpression) -> LikelihoodEvaluator {
            LikelihoodEvaluator(self.0.load(&likelihood_expression.0))
        }
    }

    /// A class which can be used to evaluate a collection of LikelihoodTerms managed by a
    /// LikelihoodManager
    ///
    #[pyclass]
    struct LikelihoodEvaluator(rust::likelihoods::LikelihoodEvaluator);

    #[pymethods]
    impl LikelihoodEvaluator {
        /// A list of the names of the free parameters across all terms in all models
        ///
        /// Returns
        /// -------
        /// parameters : list of str
        ///
        #[getter]
        fn parameters(&self) -> Vec<String> {
            self.0.parameters()
        }
        /// Evaluate the sum of all terms in the evaluator
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
        /// result : float
        ///     The total negative log-likelihood summed over all terms
        ///
        #[pyo3(signature = (parameters, *, threads=None))]
        fn evaluate(&self, parameters: Vec<Float>, threads: Option<usize>) -> PyResult<Float> {
            #[cfg(feature = "rayon")]
            {
                Ok(ThreadPoolBuilder::new()
                    .num_threads(threads.unwrap_or_else(num_cpus::get))
                    .build()
                    .map_err(LadduError::from)?
                    .install(|| self.0.evaluate(&parameters))?)
            }
            #[cfg(not(feature = "rayon"))]
            {
                Ok(self.0.evaluate(&parameters)?)
            }
        }
        /// Minimize all LikelihoodTerms with respect to the free parameters in the model
        ///
        /// This method "runs the fit". Given an initial position `p0` and optional `bounds`, this
        /// method performs a minimization over the tatal negative log-likelihood, optimizing the model
        /// over the stored signal data and Monte Carlo.
        ///
        /// Parameters
        /// ----------
        /// p0 : array_like
        ///     The initial parameters at the start of optimization
        /// bounds : list of tuple of float, optional
        ///     A list of lower and upper bound pairs for each parameter (use ``None`` for no bound)
        /// method : {'lbfgsb', 'nelder-mead', 'nelder_mead'}
        ///     The minimization algorithm to use (see additional parameters for fine-tuning)
        /// max_steps : int, default=4000
        ///     The maximum number of algorithm steps to perform
        /// debug : bool, default=False
        ///     Set to ``True`` to print out debugging information at each step
        /// verbose : bool, default=False
        ///     Set to ``True`` to print verbose information at each step
        /// show_step : bool, default=True
        ///     Include step number in verbose output
        /// show_x : bool, default=True
        ///     Include current best position in verbose output
        /// show_fx : bool, default=True
        ///     Include current best NLL in verbose output
        /// observers : Observer or list of Observers
        ///     Callback functions which are applied after every algorithm step
        /// tol_x_rel : float
        ///     The relative position tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_x_abs : float
        ///     The absolute position tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_f_rel : float
        ///     The relative function tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_f_abs : float
        ///     The absolute function tolerance used by termination methods (default is machine
        ///     epsilon)
        /// tol_g_abs : float
        ///     The absolute gradient tolerance used by termination methods (default is the cube
        ///     root of machine epsilon)
        /// g_tolerance : float, default=1e-5
        ///     Another gradient tolerance used by termination methods (particularly L-BFGS-B)
        /// adaptive : bool, default=False
        ///     Use adaptive values for Nelder-Mead parameters
        /// alpha : float, optional
        ///     Overwrite the default :math:`\alpha` parameter in the Nelder-Mead algorithm
        /// beta : float, optional
        ///     Overwrite the default :math:`\beta` parameter in the Nelder-Mead algorithm
        /// gamma : float, optional
        ///     Overwrite the default :math:`\gamma` parameter in the Nelder-Mead algorithm
        /// delta : float, optional
        ///     Overwrite the default :math:`\delta` parameter in the Nelder-Mead algorithm
        /// simplex_expansion_method : {'greedy_minimization', 'greedy_expansion'}
        ///     The expansion method used by the Nelder-Mead algorithm
        /// nelder_mead_f_terminator : {'stddev', 'absolute', 'stddev', 'none'}
        ///     The function terminator used by the Nelder-Mead algorithm
        /// nelder_mead_x_terminator : {'singer', 'diameter', 'rowan', 'higham', 'none'}
        ///     The positional terminator used by the Nelder-Mead algorithm
        /// threads : int, optional
        ///     The number of threads to use (setting this to None will use all available CPUs)
        ///
        #[pyo3(signature = (p0, *, bounds=None, method="lbfgsb", max_steps=4000, debug=false, verbose=false, **kwargs))]
        #[allow(clippy::too_many_arguments)]
        fn minimize(
            &self,
            p0: Vec<Float>,
            bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
            method: &str,
            max_steps: usize,
            debug: bool,
            verbose: bool,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Status> {
            let bounds = bounds.map(|bounds_vec| {
                bounds_vec
                    .iter()
                    .map(|(opt_lb, opt_ub)| {
                        (
                            opt_lb.unwrap_or(Float::NEG_INFINITY),
                            opt_ub.unwrap_or(Float::INFINITY),
                        )
                    })
                    .collect()
            });
            let n_parameters = p0.len();
            let options =
                _parse_minimizer_options(n_parameters, method, max_steps, debug, verbose, kwargs)?;
            let status = self.0.minimize(&p0, bounds, Some(options))?;
            Ok(Status(status))
        }

        /// Run an MCMC algorithm on the free parameters of the LikelihoodTerm's model
        ///
        /// This method can be used to sample the underlying log-likelihood given an initial
        /// position for each walker `p0`.
        ///
        /// Parameters
        /// ----------
        /// p0 : array_like
        ///     The initial parameters at the start of optimization
        /// n_steps : int,
        ///     The number of MCMC steps each walker should take
        /// method : {'ESS', 'AIES'}
        ///     The MCMC algorithm to use (see additional parameters for fine-tuning)
        /// debug : bool, default=False
        ///     Set to ``True`` to print out debugging information at each step
        /// verbose : bool, default=False
        ///     Set to ``True`` to print verbose information at each step
        /// seed : int,
        ///     The seed for the random number generator
        /// ess_moves : list of tuple
        ///     A list of moves for the ESS algorithm (see notes)
        /// aies_moves : list of tuple
        ///     A list of moves for the AIES algorithm (see notes)
        /// n_adaptive : int, default=100
        ///     Number of adaptive ESS steps to perform at the start of sampling
        /// mu : float, default=1.0
        ///     ESS adaptive parameter
        /// max_ess_steps : int, default=10000
        ///     The maximum number of slice expansions/contractions performed in the ESS algorithm
        /// threads : int, optional
        ///     The number of threads to use (setting this to None will use all available CPUs)
        ///
        /// Returns
        /// -------
        /// Ensemble
        ///     The resulting ensemble of walkers
        ///
        /// Notes
        /// -----
        /// Moves may be specified as tuples of ``(move name, usage weight)`` where the move name
        /// depends on the algorithm and the usage weight gives the proportion of time that move is
        /// used relative to the others in the list.
        ///
        /// For the Ensemble Slice Sampler (ESS) algorithm, valid move types are "differential" and
        /// "gaussian", and the default move set is ``[("differential", 0.9), ("gaussian", 0.1)]``.
        ///
        /// For the Affine Invariant Ensemble Sampler (AIES) algorithm, valid move types are
        /// "stretch" and "walk", and the default move set is ``[("stretch", 0.9), ("walk", 0.1)]``.
        ///
        /// For AIES, the "stretch" move can also be given with an adaptive parameter ``a``
        /// (default=``2``). To add a stretch move with a different value of ``a``, the "move name"
        /// can be instead given as a tuple ``(move name, a)``. For example, ``(("stretch", 2.2), 0.3)``
        /// creates a stretch move with ``a=2.2`` and usage weight of ``0.3``.
        ///
        /// Since MCMC methods are inclined to sample maxima rather than minima, the underlying
        /// function sign is automatically flipped when calling this method.
        ///
        #[pyo3(signature = (p0, n_steps, *, method="ESS", debug=false, verbose=false, seed=0, **kwargs))]
        #[allow(clippy::too_many_arguments)]
        fn mcmc(
            &self,
            p0: Vec<Vec<Float>>,
            n_steps: usize,
            method: &str,
            debug: bool,
            verbose: bool,
            seed: u64,
            kwargs: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<Ensemble> {
            let p0 = p0.into_iter().map(DVector::from_vec).collect::<Vec<_>>();
            let mut rng = Rng::new();
            rng.seed(seed);
            let options = _parse_mcmc_options(method, debug, verbose, kwargs, rng.clone())?;
            let ensemble = self.0.mcmc(&p0, n_steps, Some(options), rng)?;
            Ok(Ensemble(ensemble))
        }
    }

    /// A parameterized scalar term which can be added to a LikelihoodManager
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The name of the new scalar parameter
    ///
    /// Returns
    /// -------
    /// LikelihoodTerm
    ///
    #[pyfunction]
    fn LikelihoodScalar(name: String) -> LikelihoodTerm {
        LikelihoodTerm(rust::likelihoods::LikelihoodScalar::new(name))
    }

    #[pyclass]
    #[pyo3(name = "Observer")]
    pub(crate) struct PyObserver(pub(crate) Py<PyAny>);

    #[pymethods]
    impl PyObserver {
        #[new]
        pub fn new(observer: Py<PyAny>) -> Self {
            Self(observer)
        }
    }

    #[pyclass]
    #[pyo3(name = "MCMCObserver")]
    pub(crate) struct PyMCMCObserver(pub(crate) Py<PyAny>);

    #[pymethods]
    impl PyMCMCObserver {
        #[new]
        pub fn new(observer: Py<PyAny>) -> Self {
            Self(observer)
        }
    }

    /// The status/result of a minimization
    ///
    #[pyclass]
    #[derive(Clone)]
    pub(crate) struct Status(pub(crate) ganesh::Status);
    #[pymethods]
    impl Status {
        /// The current best position in parameter space
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(py, self.0.x.as_slice())
        }
        /// The uncertainty on each parameter (``None`` if it wasn't calculated)
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        #[getter]
        fn err<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<Float>>> {
            self.0
                .err
                .clone()
                .map(|err| PyArray1::from_slice(py, err.as_slice()))
        }
        /// The initial position at the start of the minimization
        ///
        /// Returns
        /// -------
        /// array_like
        ///
        #[getter]
        fn x0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(py, self.0.x0.as_slice())
        }
        /// The optimized value of the objective function
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn fx(&self) -> Float {
            self.0.fx
        }
        /// The covariance matrix (``None`` if it wasn't calculated)
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        #[getter]
        fn cov<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Float>>> {
            self.0.cov.clone().map(|cov| {
                PyArray2::from_vec2(
                    py,
                    &cov.row_iter()
                        .map(|row| row.iter().cloned().collect())
                        .collect::<Vec<Vec<Float>>>(),
                )
                .unwrap()
            })
        }
        /// The Hessian matrix (``None`` if it wasn't calculated)
        ///
        /// Returns
        /// -------
        /// array_like or None
        ///
        #[getter]
        fn hess<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Float>>> {
            self.0.hess.clone().map(|hess| {
                PyArray2::from_vec2(
                    py,
                    &hess
                        .row_iter()
                        .map(|row| row.iter().cloned().collect())
                        .collect::<Vec<Vec<Float>>>(),
                )
                .unwrap()
            })
        }
        /// A status message from the optimizer at the end of the algorithm
        ///
        /// Returns
        /// -------
        /// str
        ///
        #[getter]
        fn message(&self) -> String {
            self.0.message.clone()
        }
        /// The state of the optimizer's convergence conditions
        ///
        /// Returns
        /// -------
        /// bool
        ///
        #[getter]
        fn converged(&self) -> bool {
            self.0.converged
        }
        /// Parameter bounds which were applied to the fitting algorithm
        ///
        /// Returns
        /// -------
        /// list of Bound or None
        ///
        #[getter]
        fn bounds(&self) -> Option<Vec<ParameterBound>> {
            self.0
                .bounds
                .clone()
                .map(|bounds| bounds.iter().map(|bound| ParameterBound(*bound)).collect())
        }
        /// The number of times the objective function was evaluated
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_f_evals(&self) -> usize {
            self.0.n_f_evals
        }
        /// The number of times the gradient of the objective function was evaluated
        ///
        /// Returns
        /// -------
        /// int
        ///
        #[getter]
        fn n_g_evals(&self) -> usize {
            self.0.n_g_evals
        }
        fn __str__(&self) -> String {
            self.0.to_string()
        }
        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
        /// Save the fit result to a file
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
        /// Notes
        /// -----
        /// Valid file path names must have either the ".pickle" or ".pkl" extension
        ///
        fn save_as(&self, path: &str) -> PyResult<()> {
            self.0.save_as(path)?;
            Ok(())
        }
        /// Load a fit result from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the existing fit file
        ///
        /// Returns
        /// -------
        /// Status
        ///     The fit result contained in the file
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to read the file
        ///
        /// Notes
        /// -----
        /// Valid file path names must have either the ".pickle" or ".pkl" extension
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(Status(ganesh::Status::load_from(path)?))
        }
        #[new]
        fn new() -> Self {
            Status(ganesh::Status::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(py, serialize(&self.0).unwrap().as_slice()))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = Status(deserialize(state.as_bytes()).unwrap());
            Ok(())
        }
        /// Converts a Status into a Python dictionary
        ///
        /// Returns
        /// -------
        /// dict
        ///
        fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let dict = PyDict::new(py);
            dict.set_item("x", self.x(py))?;
            dict.set_item("err", self.err(py))?;
            dict.set_item("x0", self.x0(py))?;
            dict.set_item("fx", self.fx())?;
            dict.set_item("cov", self.cov(py))?;
            dict.set_item("hess", self.hess(py))?;
            dict.set_item("message", self.message())?;
            dict.set_item("converged", self.converged())?;
            dict.set_item("bounds", self.bounds())?;
            dict.set_item("n_f_evals", self.n_f_evals())?;
            dict.set_item("n_g_evals", self.n_g_evals())?;
            Ok(dict)
        }
    }

    /// An ensemble of MCMC walkers
    ///
    #[pyclass]
    #[derive(Clone)]
    pub(crate) struct Ensemble(pub(crate) ganesh::mcmc::Ensemble);
    #[pymethods]
    impl Ensemble {
        /// The dimension of the Ensemble ``(n_walkers, n_steps, n_variables)``
        #[getter]
        fn dimension(&self) -> (usize, usize, usize) {
            self.0.dimension()
        }
        /// Get the contents of the Ensemble
        ///
        /// Parameters
        /// ----------
        /// burn: int, default = 0
        ///     The number of steps to burn from the beginning of each walker's history
        /// thin: int, default = 1
        ///     The number of steps to discard after burn-in (``1`` corresponds to no thinning,
        ///     ``2`` discards every other step, ``3`` discards every third, and so on)
        ///
        /// Returns
        /// -------
        /// array_like
        ///     An array with dimension ``(n_walkers, n_steps, n_parameters)``
        ///
        #[pyo3(signature = (*, burn = 0, thin = 1))]
        fn get_chain<'py>(
            &self,
            py: Python<'py>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> Bound<'py, PyArray3<Float>> {
            let chain = self.0.get_chain(burn, thin);
            PyArray3::from_vec3(
                py,
                &chain
                    .iter()
                    .map(|walker| {
                        walker
                            .iter()
                            .map(|step| step.data.as_vec().to_vec())
                            .collect()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        }
        /// Get the contents of the Ensemble, flattened over walkers
        ///
        /// Parameters
        /// ----------
        /// burn: int, default = 0
        ///     The number of steps to burn from the beginning of each walker's history
        /// thin: int, default = 1
        ///     The number of steps to discard after burn-in (``1`` corresponds to no thinning,
        ///     ``2`` discards every other step, ``3`` discards every third, and so on)
        ///
        /// Returns
        /// -------
        /// array_like
        ///     An array with dimension ``(n_steps, n_parameters)``
        ///
        #[pyo3(signature = (*, burn = 0, thin = 1))]
        fn get_flat_chain<'py>(
            &self,
            py: Python<'py>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> Bound<'py, PyArray2<Float>> {
            let chain = self.0.get_flat_chain(burn, thin);
            PyArray2::from_vec2(
                py,
                &chain
                    .iter()
                    .map(|step| step.data.as_vec().to_vec())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        }
        /// Save the ensemble to a file
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
        /// Notes
        /// -----
        /// Valid file path names must have either the ".pickle" or ".pkl" extension
        ///
        fn save_as(&self, path: &str) -> PyResult<()> {
            self.0.save_as(path)?;
            Ok(())
        }
        /// Load an ensemble from a file
        ///
        /// Parameters
        /// ----------
        /// path : str
        ///     The path of the existing fit file
        ///
        /// Returns
        /// -------
        /// Ensemble
        ///     The ensemble contained in the file
        ///
        /// Raises
        /// ------
        /// IOError
        ///     If anything fails when trying to read the file
        ///
        /// Notes
        /// -----
        /// Valid file path names must have either the ".pickle" or ".pkl" extension
        ///
        #[staticmethod]
        fn load_from(path: &str) -> PyResult<Self> {
            Ok(Ensemble(ganesh::mcmc::Ensemble::load_from(path)?))
        }
        #[new]
        fn new() -> Self {
            Ensemble(ganesh::mcmc::Ensemble::create_null())
        }
        fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new(py, serialize(&self.0).unwrap().as_slice()))
        }
        fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
            *self = Ensemble(deserialize(state.as_bytes()).unwrap());
            Ok(())
        }
        /// Calculate the integrated autocorrelation time for each parameter according to
        /// [Karamanis]_
        ///
        /// Parameters
        /// ----------
        /// c : float, default = 7.0
        ///     The size of the window used in the autowindowing algorithm by [Sokal]_
        /// burn: int, default = 0
        ///     The number of steps to burn from the beginning of each walker's history
        /// thin: int, default = 1
        ///     The number of steps to discard after burn-in (``1`` corresponds to no thinning,
        ///     ``2`` discards every other step, ``3`` discards every third, and so on)
        ///
        #[pyo3(signature = (*, c=7.0, burn=0, thin=1))]
        fn get_integrated_autocorrelation_times<'py>(
            &self,
            py: Python<'py>,
            c: Option<Float>,
            burn: Option<usize>,
            thin: Option<usize>,
        ) -> Bound<'py, PyArray1<Float>> {
            PyArray1::from_slice(
                py,
                self.0
                    .get_integrated_autocorrelation_times(c, burn, thin)
                    .as_slice(),
            )
        }
    }

    /// Calculate the integrated autocorrelation time for each parameter according to
    /// [Karamanis]_
    ///
    /// Parameters
    /// ----------
    /// x : array_like
    ///     An array of dimension ``(n_walkers, n_steps, n_parameters)``
    /// c : float, default = 7.0
    ///     The size of the window used in the autowindowing algorithm by [Sokal]_
    ///
    /// .. [Karamanis] Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002. 06212.
    /// .. [Sokal] Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
    #[pyfunction]
    #[pyo3(signature = (x, *, c=7.0))]
    fn integrated_autocorrelation_times(
        py: Python<'_>,
        x: Vec<Vec<Vec<Float>>>,
        c: Option<Float>,
    ) -> Bound<'_, PyArray1<Float>> {
        let x: Vec<Vec<DVector<Float>>> = x
            .into_iter()
            .map(|y| y.into_iter().map(DVector::from_vec).collect())
            .collect();
        PyArray1::from_slice(
            py,
            ganesh::mcmc::integrated_autocorrelation_times(x, c).as_slice(),
        )
    }

    /// An obsever which can check the integrated autocorrelation time of the ensemble and
    /// terminate if convergence conditions are met
    ///
    /// Parameters
    /// ----------
    /// n_check : int, default = 50
    ///     How often (in number of steps) to check this observer
    /// n_tau_threshold : int, default = 50
    ///     The number of mean integrated autocorrelation times needed to terminate
    /// dtau_threshold : float, default = 0.01
    ///     The threshold for the absolute change in integrated autocorrelation time (Δτ/τ)
    /// discard : float, default = 0.5
    ///     The fraction of steps to discard from the beginning of the chain before analysis
    /// terminate : bool, default = True
    ///     Set to ``False`` to forego termination even if the chains converge
    /// c : float, default = 7.0
    ///     The size of the window used in the autowindowing algorithm by [Sokal]_
    /// verbose : bool, default = False
    ///     Set to ``True`` to print out details at each check
    ///
    #[pyclass]
    pub(crate) struct AutocorrelationObserver(
        pub(crate) Arc<RwLock<ganesh::observers::AutocorrelationObserver>>,
    );

    #[pymethods]
    impl AutocorrelationObserver {
        #[new]
        #[pyo3(signature = (*, n_check=50, n_taus_threshold=50, dtau_threshold=0.01, discard=0.5, terminate=true, c=7.0, verbose=false))]
        fn new(
            n_check: usize,
            n_taus_threshold: usize,
            dtau_threshold: Float,
            discard: Float,
            terminate: bool,
            c: Float,
            verbose: bool,
        ) -> Self {
            Self(
                ganesh::observers::AutocorrelationObserver::default()
                    .with_n_check(n_check)
                    .with_n_taus_threshold(n_taus_threshold)
                    .with_dtau_threshold(dtau_threshold)
                    .with_discard(discard)
                    .with_terminate(terminate)
                    .with_sokal_window(c)
                    .with_verbose(verbose)
                    .build(),
            )
        }
        /// The integrated autocorrelation times observed at each checking step
        ///
        #[getter]
        fn taus<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Float>> {
            let taus = self.0.read().taus.clone();
            PyArray1::from_vec(py, taus)
        }
    }

    /// A class representing a lower and upper bound on a free parameter
    ///
    #[pyclass]
    #[derive(Clone)]
    #[pyo3(name = "Bound")]
    pub(crate) struct ParameterBound(pub(crate) ganesh::Bound);
    #[pymethods]
    impl ParameterBound {
        /// The lower bound
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn lower(&self) -> Float {
            self.0.lower()
        }
        /// The upper bound
        ///
        /// Returns
        /// -------
        /// float
        ///
        #[getter]
        fn upper(&self) -> Float {
            self.0.upper()
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
    #[pyclass]
    #[derive(Clone)]
    struct ParameterLike(rust::amplitudes::ParameterLike);

    /// A free parameter which floats during an optimization
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The name of the free parameter
    ///
    /// Returns
    /// -------
    /// ParameterLike
    ///     An object that can be used as the input for many Amplitude constructors
    ///
    /// Notes
    /// -----
    /// Two free parameters with the same name are shared in a fit
    ///
    #[pyfunction]
    fn parameter(name: &str) -> ParameterLike {
        ParameterLike(rust::amplitudes::parameter(name))
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
    /// ParameterLike
    ///     An object that can be used as the input for many Amplitude constructors
    ///
    #[pyfunction]
    fn constant(value: Float) -> ParameterLike {
        ParameterLike(rust::amplitudes::constant(value))
    }

    /// An Amplitude which represents a single scalar value
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// value : ParameterLike
    ///     The scalar parameter contained in the Amplitude
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    #[pyfunction]
    fn Scalar(name: &str, value: ParameterLike) -> Amplitude {
        Amplitude(rust::amplitudes::common::Scalar::new(name, value.0))
    }

    /// An Amplitude which represents a complex scalar value
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// re: ParameterLike
    ///     The real part of the complex value contained in the Amplitude
    /// im: ParameterLike
    ///     The imaginary part of the complex value contained in the Amplitude
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    #[pyfunction]
    fn ComplexScalar(name: &str, re: ParameterLike, im: ParameterLike) -> Amplitude {
        Amplitude(rust::amplitudes::common::ComplexScalar::new(
            name, re.0, im.0,
        ))
    }

    /// An Amplitude which represents a complex scalar value in polar form
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// r: ParameterLike
    ///     The magnitude of the complex value contained in the Amplitude
    /// theta: ParameterLike
    ///     The argument of the complex value contained in the Amplitude
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    #[pyfunction]
    fn PolarComplexScalar(name: &str, r: ParameterLike, theta: ParameterLike) -> Amplitude {
        Amplitude(rust::amplitudes::common::PolarComplexScalar::new(
            name, r.0, theta.0,
        ))
    }

    /// An spherical harmonic Amplitude
    ///
    /// Computes a spherical harmonic (:math:`Y_{\ell}^m(\theta, \varphi)`)
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// l : int
    ///     The total orbital momentum (:math:`l \geq 0`)
    /// m : int
    ///     The orbital moment (:math:`-l \leq m \leq l`)
    /// angles : Angles
    ///     The spherical angles to use in the calculation
    ///     
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    #[pyfunction]
    fn Ylm(name: &str, l: usize, m: isize, angles: &Angles) -> Amplitude {
        Amplitude(rust::amplitudes::ylm::Ylm::new(name, l, m, &angles.0))
    }

    /// An spherical harmonic Amplitude for polarized beam experiments
    ///
    /// Computes a polarized spherical harmonic (:math:`Z_{\ell}^{(r)m}(\theta, \varphi; P_\gamma, \Phi)`) with additional
    /// polarization-related factors (see notes)
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// l : int
    ///     The total orbital momentum (:math:`l \geq 0`)
    /// m : int
    ///     The orbital moment (:math:`-l \leq m \leq l`)
    /// r : {'+', 'plus', 'pos', 'positive', '-', 'minus', 'neg', 'negative'}
    ///     The reflectivity (related to naturality of parity exchange)
    /// angles : Angles
    ///     The spherical angles to use in the calculation
    /// polarization : Polarization
    ///     The beam polarization to use in the calculation
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    /// Notes
    /// -----
    /// This amplitude is described in [Mathieu]_
    ///
    /// .. [Mathieu] Mathieu, V., Albaladejo, M., Fernández-Ramírez, C., Jackura, A. W., Mikhasenko, M., Pilloni, A., & Szczepaniak, A. P. (2019). Moments of angular distribution and beam asymmetries in :math:`\eta\pi^0` photoproduction at GlueX. Physical Review D, 100(5). `doi:10.1103/physrevd.100.054017 <https://doi.org/10.1103/PhysRevD.100.054017>`_
    ///
    #[pyfunction]
    fn Zlm(
        name: &str,
        l: usize,
        m: isize,
        r: &str,
        angles: &Angles,
        polarization: &Polarization,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::zlm::Zlm::new(
            name,
            l,
            m,
            r.parse().unwrap(),
            &angles.0,
            &polarization.0,
        ))
    }

    /// An relativistic Breit-Wigner Amplitude
    ///
    /// This Amplitude represents a relativistic Breit-Wigner with known angular momentum
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// mass : ParameterLike
    ///     The mass of the resonance
    /// width : ParameterLike
    ///     The (nonrelativistic) width of the resonance
    /// l : int
    ///     The total orbital momentum (:math:`l > 0`)
    /// daughter_1_mass : Mass
    ///     The mass of the first decay product
    /// daughter_2_mass : Mass
    ///     The mass of the second decay product
    /// resonance_mass: Mass
    ///     The total mass of the resonance
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    #[pyfunction]
    fn BreitWigner(
        name: &str,
        mass: ParameterLike,
        width: ParameterLike,
        l: usize,
        daughter_1_mass: &Mass,
        daughter_2_mass: &Mass,
        resonance_mass: &Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::breit_wigner::BreitWigner::new(
            name,
            mass.0,
            width.0,
            l,
            &daughter_1_mass.0,
            &daughter_2_mass.0,
            &resonance_mass.0,
        ))
    }

    /// A fixed K-Matrix Amplitude for :math:`f_0` mesons
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// couplings : list of list of ParameterLike
    ///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
    /// channel : int
    ///     The channel onto which the K-Matrix is projected
    /// mass: Mass
    ///     The total mass of the resonance
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    /// Notes
    /// -----
    /// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
    /// from that paper, leaving the couplings to the initial state free
    ///
    /// +---------------+-------------------+
    /// | Channel index | Channel           |
    /// +===============+===================+
    /// | 0             | :math:`\pi\pi`    |
    /// +---------------+-------------------+
    /// | 1             | :math:`2\pi 2\pi` |
    /// +---------------+-------------------+
    /// | 2             | :math:`K\bar{K}`  |
    /// +---------------+-------------------+
    /// | 3             | :math:`\eta\eta`  |
    /// +---------------+-------------------+
    /// | 4             | :math:`\eta\eta'` |
    /// +---------------+-------------------+
    ///
    /// +-------------------+
    /// | Pole names        |
    /// +===================+
    /// | :math:`f_0(500)`  |
    /// +-------------------+
    /// | :math:`f_0(980)`  |
    /// +-------------------+
    /// | :math:`f_0(1370)` |
    /// +-------------------+
    /// | :math:`f_0(1500)` |
    /// +-------------------+
    /// | :math:`f_0(1710)` |
    /// +-------------------+
    ///
    /// .. [Kopf] Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. (2021). Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of :math:`\bar{p}p`-, :math:`\pi^- p`- and :math:`\pi \pi`-Data. The European Physical Journal C, 81(12). `doi:10.1140/epjc/s10052-021-09821-2 <https://doi.org/10.1140/epjc/s10052-021-09821-2>`__
    ///
    #[pyfunction]
    fn KopfKMatrixF0(
        name: &str,
        couplings: [[ParameterLike; 2]; 5],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixF0::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    /// A fixed K-Matrix Amplitude for :math:`f_2` mesons
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// couplings : list of list of ParameterLike
    ///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
    /// channel : int
    ///     The channel onto which the K-Matrix is projected
    /// mass: Mass
    ///     The total mass of the resonance
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    /// Notes
    /// -----
    /// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
    /// from that paper, leaving the couplings to the initial state free
    ///
    /// +---------------+-------------------+
    /// | Channel index | Channel           |
    /// +===============+===================+
    /// | 0             | :math:`\pi\pi`    |
    /// +---------------+-------------------+
    /// | 1             | :math:`2\pi 2\pi` |
    /// +---------------+-------------------+
    /// | 2             | :math:`K\bar{K}`  |
    /// +---------------+-------------------+
    /// | 3             | :math:`\eta\eta`  |
    /// +---------------+-------------------+
    ///
    /// +---------------------+
    /// | Pole names          |
    /// +=====================+
    /// | :math:`f_2(1270)`   |
    /// +---------------------+
    /// | :math:`f_2'(1525)`  |
    /// +---------------------+
    /// | :math:`f_2(1810)`   |
    /// +---------------------+
    /// | :math:`f_2(1950)`   |
    /// +---------------------+
    ///
    #[pyfunction]
    fn KopfKMatrixF2(
        name: &str,
        couplings: [[ParameterLike; 2]; 4],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixF2::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    /// A fixed K-Matrix Amplitude for :math:`a_0` mesons
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// couplings : list of list of ParameterLike
    ///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
    /// channel : int
    ///     The channel onto which the K-Matrix is projected
    /// mass: Mass
    ///     The total mass of the resonance
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    /// Notes
    /// -----
    /// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
    /// from that paper, leaving the couplings to the initial state free
    ///
    /// +---------------+-------------------+
    /// | Channel index | Channel           |
    /// +===============+===================+
    /// | 0             | :math:`\pi\eta`   |
    /// +---------------+-------------------+
    /// | 1             | :math:`K\bar{K}`  |
    /// +---------------+-------------------+
    ///
    /// +-------------------+
    /// | Pole names        |
    /// +===================+
    /// | :math:`a_0(980)`  |
    /// +-------------------+
    /// | :math:`a_0(1450)` |
    /// +-------------------+
    ///
    #[pyfunction]
    fn KopfKMatrixA0(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixA0::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    /// A fixed K-Matrix Amplitude for :math:`a_2` mesons
    ///
    /// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
    /// from that paper, leaving the couplings to the initial state free
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// couplings : list of list of ParameterLike
    ///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
    /// channel : int
    ///     The channel onto which the K-Matrix is projected
    /// mass: Mass
    ///     The total mass of the resonance
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    /// Notes
    /// -----
    /// +---------------+-------------------+
    /// | Channel index | Channel           |
    /// +===============+===================+
    /// | 0             | :math:`\pi\eta`   |
    /// +---------------+-------------------+
    /// | 1             | :math:`K\bar{K}`  |
    /// +---------------+-------------------+
    /// | 2             | :math:`\pi\eta'`  |
    /// +---------------+-------------------+
    ///
    /// +-------------------+
    /// | Pole names        |
    /// +===================+
    /// | :math:`a_2(1320)` |
    /// +-------------------+
    /// | :math:`a_2(1700)` |
    /// +-------------------+
    ///
    #[pyfunction]
    fn KopfKMatrixA2(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixA2::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }

    /// A fixed K-Matrix Amplitude for :math:`\rho` mesons
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// couplings : list of list of ParameterLike
    ///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
    /// channel : int
    ///     The channel onto which the K-Matrix is projected
    /// mass: Mass
    ///     The total mass of the resonance
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    /// Notes
    /// -----
    /// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
    /// from that paper, leaving the couplings to the initial state free
    ///
    /// +---------------+-------------------+
    /// | Channel index | Channel           |
    /// +===============+===================+
    /// | 0             | :math:`\pi\pi`    |
    /// +---------------+-------------------+
    /// | 1             | :math:`2\pi 2\pi` |
    /// +---------------+-------------------+
    /// | 2             | :math:`K\bar{K}`  |
    /// +---------------+-------------------+
    ///
    /// +--------------------+
    /// | Pole names         |
    /// +====================+
    /// | :math:`\rho(770)`  |
    /// +--------------------+
    /// | :math:`\rho(1700)` |
    /// +--------------------+
    ///
    #[pyfunction]
    fn KopfKMatrixRho(
        name: &str,
        couplings: [[ParameterLike; 2]; 2],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixRho::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }
    /// A fixed K-Matrix Amplitude for the :math:`\pi_1(1600)` hybrid meson
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     The Amplitude name
    /// couplings : list of list of ParameterLike
    ///     Each initial-state coupling (as a list of pairs of real and imaginary parts)
    /// channel : int
    ///     The channel onto which the K-Matrix is projected
    /// mass: Mass
    ///     The total mass of the resonance
    ///
    /// Returns
    /// -------
    /// Amplitude
    ///     An Amplitude which can be registered by a ``Manager``
    ///
    /// See Also
    /// --------
    /// laddu.Manager
    ///
    /// Notes
    /// -----
    /// This Amplitude follows the prescription of [Kopf]_ and fixes the K-Matrix to data
    /// from that paper, leaving the couplings to the initial state free
    ///
    /// +---------------+-------------------+
    /// | Channel index | Channel           |
    /// +===============+===================+
    /// | 0             | :math:`\pi\eta`   |
    /// +---------------+-------------------+
    /// | 1             | :math:`\pi\eta'`  |
    /// +---------------+-------------------+
    ///
    /// +---------------------+
    /// | Pole names          |
    /// +=====================+
    /// | :math:`\pi_1(1600)` |
    /// +---------------------+
    ///
    #[pyfunction]
    fn KopfKMatrixPi1(
        name: &str,
        couplings: [[ParameterLike; 2]; 1],
        channel: usize,
        mass: Mass,
    ) -> Amplitude {
        Amplitude(rust::amplitudes::kmatrix::KopfKMatrixPi1::new(
            name,
            array::from_fn(|i| array::from_fn(|j| couplings[i][j].clone().0)),
            channel,
            &mass.0,
        ))
    }
}

impl Observer<()> for crate::python::laddu::PyObserver {
    fn callback(&mut self, step: usize, status: &mut ganesh::Status, _user_data: &mut ()) -> bool {
        let (new_status, result) = Python::with_gil(|py| {
            let res = self
                .0
                .bind(py)
                .call_method(
                    "callback",
                    (step, crate::python::laddu::Status(status.clone())),
                    None,
                )
                .unwrap();
            let res_tuple = res.downcast::<PyTuple>().unwrap();
            let new_status = res_tuple
                .get_item(0)
                .unwrap()
                .extract::<crate::python::laddu::Status>()
                .unwrap()
                .0;
            let result = res_tuple.get_item(1).unwrap().extract::<bool>().unwrap();
            (new_status, result)
        });
        *status = new_status;
        result
    }
}

#[cfg(feature = "rayon")]
impl Observer<ThreadPool> for crate::python::laddu::PyObserver {
    fn callback(
        &mut self,
        step: usize,
        status: &mut ganesh::Status,
        _thread_pool: &mut ThreadPool,
    ) -> bool {
        let (new_status, result) = Python::with_gil(|py| {
            let res = self
                .0
                .bind(py)
                .call_method(
                    "callback",
                    (step, crate::python::laddu::Status(status.clone())),
                    None,
                )
                .unwrap();
            let res_tuple = res.downcast::<PyTuple>().unwrap();
            let new_status = res_tuple
                .get_item(0)
                .unwrap()
                .extract::<crate::python::laddu::Status>()
                .unwrap()
                .0;
            let result = res_tuple.get_item(1).unwrap().extract::<bool>().unwrap();
            (new_status, result)
        });
        *status = new_status;
        result
    }
}
impl FromPyObject<'_> for crate::python::laddu::PyObserver {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(crate::python::laddu::PyObserver(ob.clone().into()))
    }
}
impl MCMCObserver<()> for crate::python::laddu::PyMCMCObserver {
    fn callback(
        &mut self,
        step: usize,
        ensemble: &mut ganesh::mcmc::Ensemble,
        _user_data: &mut (),
    ) -> bool {
        let (new_ensemble, result) = Python::with_gil(|py| {
            let res = self
                .0
                .bind(py)
                .call_method(
                    "callback",
                    (step, crate::python::laddu::Ensemble(ensemble.clone())),
                    None,
                )
                .unwrap();
            let res_tuple = res.downcast::<PyTuple>().unwrap();
            let new_status = res_tuple
                .get_item(0)
                .unwrap()
                .extract::<crate::python::laddu::Ensemble>()
                .unwrap()
                .0;
            let result = res_tuple.get_item(1).unwrap().extract::<bool>().unwrap();
            (new_status, result)
        });
        *ensemble = new_ensemble;
        result
    }
}
#[cfg(feature = "rayon")]
impl MCMCObserver<ThreadPool> for crate::python::laddu::PyMCMCObserver {
    fn callback(
        &mut self,
        step: usize,
        ensemble: &mut ganesh::mcmc::Ensemble,
        _thread_pool: &mut ThreadPool,
    ) -> bool {
        let (new_ensemble, result) = Python::with_gil(|py| {
            let res = self
                .0
                .bind(py)
                .call_method(
                    "callback",
                    (step, crate::python::laddu::Ensemble(ensemble.clone())),
                    None,
                )
                .unwrap();
            let res_tuple = res.downcast::<PyTuple>().unwrap();
            let new_status = res_tuple
                .get_item(0)
                .unwrap()
                .extract::<crate::python::laddu::Ensemble>()
                .unwrap()
                .0;
            let result = res_tuple.get_item(1).unwrap().extract::<bool>().unwrap();
            (new_status, result)
        });
        *ensemble = new_ensemble;
        result
    }
}

impl FromPyObject<'_> for crate::python::laddu::PyMCMCObserver {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(crate::python::laddu::PyMCMCObserver(ob.clone().into()))
    }
}
