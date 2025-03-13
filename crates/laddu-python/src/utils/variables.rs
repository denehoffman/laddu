use crate::data::{PyDataset, PyEvent};
use laddu_core::{
    data::{Dataset, Event},
    traits::Variable,
    utils::variables::{
        Angles, CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude, Polarization,
    },
    Float,
};
use numpy::PyArray1;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

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

impl Debug for PyVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mass(v) => write!(f, "{:?}", v.0),
            Self::CosTheta(v) => write!(f, "{:?}", v.0),
            Self::Phi(v) => write!(f, "{:?}", v.0),
            Self::PolAngle(v) => write!(f, "{:?}", v.0),
            Self::PolMagnitude(v) => write!(f, "{:?}", v.0),
            Self::Mandelstam(v) => write!(f, "{:?}", v.0),
        }
    }
}
impl Display for PyVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mass(v) => write!(f, "{}", v.0),
            Self::CosTheta(v) => write!(f, "{}", v.0),
            Self::Phi(v) => write!(f, "{}", v.0),
            Self::PolAngle(v) => write!(f, "{}", v.0),
            Self::PolMagnitude(v) => write!(f, "{}", v.0),
            Self::Mandelstam(v) => write!(f, "{}", v.0),
        }
    }
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
#[pyclass(name = "Mass", module = "laddu")]
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
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
#[pyclass(name = "CosTheta", module = "laddu")]
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
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
#[pyclass(name = "Phi", module = "laddu")]
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
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
#[pyclass(name = "Angles", module = "laddu")]
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
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
/// beam_polarization : int
///     The index of the auxiliary vector in storing the `beam` particle's polarization
///
#[pyclass(name = "PolAngle", module = "laddu")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPolAngle(pub PolAngle);

#[pymethods]
impl PyPolAngle {
    #[new]
    fn new(beam: usize, recoil: Vec<usize>, beam_polarization: usize) -> Self {
        Self(PolAngle::new(beam, &recoil, beam_polarization))
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

/// The magnitude of the given particle's polarization vector
///
/// This Variable simply represents the magnitude of the polarization vector of the particle
/// with the index `beam`
///
/// Parameters
/// ----------
/// beam_polarization : int
///     The index of the auxiliary vector in storing the `beam` particle's polarization
///
/// See Also
/// --------
/// laddu.utils.vectors.Vector3.mag
///
#[pyclass(name = "PolMagnitude", module = "laddu")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPolMagnitude(pub PolMagnitude);

#[pymethods]
impl PyPolMagnitude {
    #[new]
    fn new(beam_polarization: usize) -> Self {
        Self(PolMagnitude::new(beam_polarization))
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
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
/// beam_polarization : int
///     The index of the auxiliary vector in storing the `beam` particle's polarization
///
/// See Also
/// --------
/// laddu.PolAngle
/// laddu.PolMagnitude
///
#[pyclass(name = "Polarization", module = "laddu")]
#[derive(Clone)]
pub struct PyPolarization(pub Polarization);
#[pymethods]
impl PyPolarization {
    #[new]
    fn new(beam: usize, recoil: Vec<usize>, beam_polarization: usize) -> Self {
        PyPolarization(Polarization::new(beam, &recoil, beam_polarization))
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
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
#[pyclass(name = "Mandelstam", module = "laddu")]
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
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
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
