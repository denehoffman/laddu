use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    rc::Rc,
};

pub use laddu_core::variables::IntoP4Selection;
use laddu_core::{
    data::{Dataset, DatasetMetadata, EventLike, OwnedEvent},
    kinematics::{Axes, Axis, Frame},
    reaction::Channel,
    traits::Variable,
    variables::{
        Angles, CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude, Polarization,
        VariableExpression,
    },
    LadduResult,
};
use numpy::PyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

use crate::{
    data::{PyDataset, PyEvent},
    generation::{PyMassSampler, PyMomentumSource, PyVertexGenerator},
    quantum::PyParticleProperties,
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

impl PyVariable {
    pub(crate) fn bind_in_place(&mut self, metadata: &DatasetMetadata) -> PyResult<()> {
        match self {
            Self::Mass(mass) => mass.0.bind(metadata).map_err(PyErr::from),
            Self::CosTheta(cos_theta) => cos_theta.0.bind(metadata).map_err(PyErr::from),
            Self::Phi(phi) => phi.0.bind(metadata).map_err(PyErr::from),
            Self::PolAngle(pol_angle) => pol_angle.0.bind(metadata).map_err(PyErr::from),
            Self::PolMagnitude(pol_magnitude) => {
                pol_magnitude.0.bind(metadata).map_err(PyErr::from)
            }
            Self::Mandelstam(mandelstam) => mandelstam.0.bind(metadata).map_err(PyErr::from),
        }
    }

    pub(crate) fn bound(&self, metadata: &DatasetMetadata) -> PyResult<Self> {
        let mut cloned = self.clone();
        cloned.bind_in_place(metadata)?;
        Ok(cloned)
    }

    pub(crate) fn evaluate_event(&self, event: &OwnedEvent) -> PyResult<f64> {
        Ok(self.value(event))
    }
}

#[pyclass(name = "VariableExpression", module = "laddu")]
pub struct PyVariableExpression(pub VariableExpression);

#[pymethods]
impl PyVariableExpression {
    fn __and__(&self, rhs: &PyVariableExpression) -> PyVariableExpression {
        PyVariableExpression(self.0.clone() & rhs.0.clone())
    }
    fn __or__(&self, rhs: &PyVariableExpression) -> PyVariableExpression {
        PyVariableExpression(self.0.clone() | rhs.0.clone())
    }
    fn __invert__(&self) -> PyVariableExpression {
        PyVariableExpression(!self.0.clone())
    }
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
}

#[pyclass(name = "Axis", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyAxis(pub Axis);

#[pymethods]
impl PyAxis {
    #[staticmethod]
    fn particle(particle: &str) -> Self {
        Self(Axis::particle(particle))
    }

    #[staticmethod]
    fn opposite(particle: &str) -> Self {
        Self(Axis::opposite(particle))
    }

    #[staticmethod]
    fn normal(a: &str, b: &str) -> Self {
        Self(Axis::normal(a, b))
    }

    fn at(&self, vertex: &str) -> Self {
        Self(self.0.clone().at(vertex))
    }

    fn flipped(&self) -> Self {
        Self(self.0.clone().flipped())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "Axes", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyAxes(pub Axes);

#[pymethods]
impl PyAxes {
    #[staticmethod]
    fn from_y_z(y: &PyAxis, z: &PyAxis) -> Self {
        Self(Axes::from_y_z(y.0.clone(), z.0.clone()))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "Frame", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyFrame(pub Frame);

#[pymethods]
impl PyFrame {
    #[new]
    fn new(origin: &str, axes: &PyAxes) -> PyResult<Self> {
        Ok(Self(Frame::new(origin, axes.0.clone())?))
    }

    #[getter]
    fn origin(&self) -> String {
        self.0.origin().to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "Channel", module = "laddu", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyChannel {
    pub(crate) inner: Rc<RefCell<Channel>>,
}

impl PyChannel {
    pub(crate) fn channel(&self) -> Channel {
        self.inner.borrow().clone()
    }
}

#[pyclass(eq, eq_int, name = "ParticleSource", module = "laddu", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyParticleSource {
    Inferred = 0,
    Stored = 1,
    Missing = 2,
}

#[allow(clippy::too_many_arguments)]
fn apply_particle_edits(
    channel: &mut Channel,
    particle: &str,
    source: Option<PyParticleSource>,
    properties: Option<&PyParticleProperties>,
    mass: Option<f64>,
    momentum: Option<&PyMomentumSource>,
    mass_sampler: Option<&PyMassSampler>,
    name: Option<String>,
    species: Option<String>,
    self_conjugate: Option<bool>,
) -> PyResult<()> {
    let mut edit = channel.edit_particle(particle)?;
    if let Some(properties) = properties {
        edit.properties(properties.0.clone());
    }
    if let Some(mass) = mass {
        edit.mass(mass);
    }
    if let Some(momentum) = momentum {
        edit.momentum(momentum.0.clone());
    }
    if let Some(mass_sampler) = mass_sampler {
        edit.mass_sampler(mass_sampler.0.clone());
    }
    if let Some(name) = name {
        edit.name(name);
    }
    if let Some(species) = species {
        edit.species(species);
    }
    if let Some(self_conjugate) = self_conjugate {
        edit.self_conjugate(self_conjugate);
    }
    if let Some(source) = source {
        match source {
            PyParticleSource::Inferred => {
                edit.inferred();
            }
            PyParticleSource::Stored => {
                edit.stored();
            }
            PyParticleSource::Missing => {
                edit.missing()?;
            }
        }
    }
    Ok(())
}

#[pymethods]
impl PyChannel {
    #[new]
    fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(Channel::new())),
        }
    }

    #[pyo3(signature=(label, incoming, outgoing, *, generator=None))]
    fn create_vertex(
        &mut self,
        label: &str,
        incoming: Vec<String>,
        outgoing: Vec<String>,
        generator: Option<&PyVertexGenerator>,
    ) -> PyResult<()> {
        match (incoming.len(), outgoing.len()) {
            (1, 2) => {
                self.inner.borrow_mut().create_vertex(
                    label,
                    [incoming[0].as_str()],
                    [outgoing[0].as_str(), outgoing[1].as_str()],
                )?;
            }
            (2, 2) => {
                self.inner.borrow_mut().create_vertex(
                    label,
                    [incoming[0].as_str(), incoming[1].as_str()],
                    [outgoing[0].as_str(), outgoing[1].as_str()],
                )?;
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Python Channel.create_vertex currently supports 1->2 and 2->2 vertices",
                ));
            }
        }
        if let Some(generator) = generator {
            self.inner
                .borrow_mut()
                .edit_vertex(label)?
                .generate(generator.0.clone());
        }
        Ok(())
    }

    #[pyo3(signature=(label, parent, daughters, *, generator=None))]
    fn create_decay(
        &mut self,
        label: &str,
        parent: &str,
        daughters: Vec<String>,
        generator: Option<&PyVertexGenerator>,
    ) -> PyResult<()> {
        if daughters.len() != 2 {
            return Err(PyValueError::new_err(
                "decays require exactly two daughters",
            ));
        }
        self.inner.borrow_mut().create_decay(
            label,
            parent,
            [daughters[0].as_str(), daughters[1].as_str()],
        )?;
        if let Some(generator) = generator {
            self.inner
                .borrow_mut()
                .edit_vertex(label)?
                .generate(generator.0.clone());
        }
        Ok(())
    }

    #[pyo3(signature=(label, incoming, outgoing, *, generator=None))]
    fn create_production(
        &mut self,
        label: &str,
        incoming: Vec<String>,
        outgoing: Vec<String>,
        generator: Option<&PyVertexGenerator>,
    ) -> PyResult<()> {
        if incoming.len() != 2 || outgoing.len() != 2 {
            return Err(PyValueError::new_err(
                "production vertices require exactly two incoming and two outgoing particles",
            ));
        }
        self.inner.borrow_mut().create_production(
            label,
            [incoming[0].as_str(), incoming[1].as_str()],
            [outgoing[0].as_str(), outgoing[1].as_str()],
        )?;
        if let Some(generator) = generator {
            self.inner
                .borrow_mut()
                .edit_vertex(label)?
                .generate(generator.0.clone());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(particle, *, source=None, properties=None, mass=None, momentum=None, mass_sampler=None, name=None, species=None, self_conjugate=None))]
    fn edit_particle(
        &self,
        particle: &str,
        source: Option<PyParticleSource>,
        properties: Option<&PyParticleProperties>,
        mass: Option<f64>,
        momentum: Option<&PyMomentumSource>,
        mass_sampler: Option<&PyMassSampler>,
        name: Option<String>,
        species: Option<String>,
        self_conjugate: Option<bool>,
    ) -> PyResult<()> {
        apply_particle_edits(
            &mut self.inner.borrow_mut(),
            particle,
            source,
            properties,
            mass,
            momentum,
            mass_sampler,
            name,
            species,
            self_conjugate,
        )
    }

    #[pyo3(signature=(vertex, *, generator))]
    fn edit_vertex(&self, vertex: &str, generator: &PyVertexGenerator) -> PyResult<()> {
        let mut channel = self.inner.borrow_mut();
        let mut edit = channel.edit_vertex(vertex)?;
        edit.generate(generator.0.clone());
        Ok(())
    }

    fn mass(&self, particle: &str) -> PyResult<PyMass> {
        Ok(PyMass(self.inner.borrow().mass(particle)?))
    }

    fn angles(&self, particle: &str, frame: &PyFrame) -> PyResult<PyAngles> {
        Ok(PyAngles(
            self.inner.borrow().angles(particle, frame.0.clone())?,
        ))
    }

    fn mandelstam(&self, vertex: &str, channel: &str) -> PyResult<PyMandelstam> {
        Ok(PyMandelstam(
            self.inner.borrow().mandelstam(vertex, channel.parse()?)?,
        ))
    }

    fn pol_angle(&self, vertex: &str, angle_aux: String) -> PyResult<PyPolAngle> {
        Ok(PyPolAngle(
            self.inner.borrow().pol_angle(vertex, angle_aux)?,
        ))
    }

    #[pyo3(signature=(vertex, *, pol_magnitude, pol_angle))]
    fn polarization(
        &self,
        vertex: &str,
        pol_magnitude: String,
        pol_angle: String,
    ) -> PyResult<PyPolarization> {
        Ok(PyPolarization(self.inner.borrow().polarization(
            vertex,
            pol_magnitude,
            pol_angle,
        )?))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner.borrow())
    }
}

/// The invariant mass of an arbitrary combination of constituent particles in an Event
///
/// This variable is calculated by summing up the 4-momenta of each particle listed by index in
/// `constituents` and taking the invariant magnitude of the resulting 4-vector.
///
/// Parameters
/// ----------
/// constituents : str or list of str
///     Particle names to combine when constructing the final four-momentum
///
/// See Also
/// --------
/// laddu.utils.vectors.Vec4.m
///
#[pyclass(name = "Mass", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyMass(pub Mass);

#[pymethods]
impl PyMass {
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
    fn value(&self, event: &PyEvent) -> PyResult<f64> {
        let metadata = event
            .metadata_opt()
            .ok_or_else(|| PyValueError::new_err(
                "This event is not associated with metadata; supply `p4_names`/`aux_names` when constructing it or evaluate via a Dataset.",
            ))?;
        let mut variable = self.0.clone();
        variable.bind(metadata).map_err(PyErr::from)?;
        Ok(variable.value(&event.event))
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
    fn value_on<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.0.value_on(&dataset.0).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, values))
    }
    fn __eq__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.eq(value))
    }
    fn __lt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.lt(value))
    }
    fn __gt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.gt(value))
    }
    fn __le__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.le(value))
    }
    fn __ge__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.ge(value))
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
/// reaction : laddu.Reaction
///     Reaction describing the production kinematics and decay roots.
/// daughter : list of str
///     Names of particles which are combined to form one of the decay products of the
///     resonance associated with the decay parent.
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
/// laddu.utils.vectors.Vec3.costheta
///
#[pyclass(name = "CosTheta", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyCosTheta(pub CosTheta);

#[pymethods]
impl PyCosTheta {
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
    fn value(&self, event: &PyEvent) -> PyResult<f64> {
        let metadata = event
            .metadata_opt()
            .ok_or_else(|| PyValueError::new_err(
                "This event is not associated with metadata; supply `p4_names`/`aux_names` when constructing it or evaluate via a Dataset.",
            ))?;
        let mut variable = self.0.clone();
        variable.bind(metadata).map_err(PyErr::from)?;
        Ok(variable.value(&event.event))
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
    fn value_on<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.0.value_on(&dataset.0).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, values))
    }
    fn __eq__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.eq(value))
    }
    fn __lt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.lt(value))
    }
    fn __gt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.gt(value))
    }
    fn __le__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.le(value))
    }
    fn __ge__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.ge(value))
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
/// reaction : laddu.Reaction
///     Reaction describing the production kinematics and decay roots.
/// daughter : list of str
///     Names of particles which are combined to form one of the decay products of the
///     resonance associated with the decay parent.
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
/// laddu.utils.vectors.Vec3.phi
///
#[pyclass(name = "Phi", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPhi(pub Phi);

#[pymethods]
impl PyPhi {
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
    fn value(&self, event: &PyEvent) -> PyResult<f64> {
        let metadata = event
            .metadata_opt()
            .ok_or_else(|| PyValueError::new_err(
                "This event is not associated with metadata; supply `p4_names`/`aux_names` when constructing it or evaluate via a Dataset.",
            ))?;
        let mut variable = self.0.clone();
        variable.bind(metadata).map_err(PyErr::from)?;
        Ok(variable.value(&event.event))
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
    fn value_on<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.0.value_on(&dataset.0).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, values))
    }
    fn __eq__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.eq(value))
    }
    fn __lt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.lt(value))
    }
    fn __gt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.gt(value))
    }
    fn __le__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.le(value))
    }
    fn __ge__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.ge(value))
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
/// reaction : laddu.Reaction
///     Reaction describing the production kinematics and decay roots.
/// daughter : list of str
///     Names of particles which are combined to form one of the decay products of the
///     resonance associated with the decay parent.
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
#[pyclass(name = "Angles", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAngles(pub Angles);
#[pymethods]
impl PyAngles {
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
/// reaction : laddu.Reaction
///     Reaction describing the production kinematics and decay roots.
/// pol_angle : str
///     Name of the auxiliary scalar column storing the polarization angle in radians
///
#[pyclass(name = "PolAngle", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPolAngle(pub PolAngle);

#[pymethods]
impl PyPolAngle {
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
    fn value(&self, event: &PyEvent) -> PyResult<f64> {
        let metadata = event
            .metadata_opt()
            .ok_or_else(|| PyValueError::new_err(
                "This event is not associated with metadata; supply `p4_names`/`aux_names` when constructing it or evaluate via a Dataset.",
            ))?;
        let mut variable = self.0.clone();
        variable.bind(metadata).map_err(PyErr::from)?;
        Ok(variable.value(&event.event))
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
    fn value_on<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.0.value_on(&dataset.0).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, values))
    }
    fn __eq__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.eq(value))
    }
    fn __lt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.lt(value))
    }
    fn __gt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.gt(value))
    }
    fn __le__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.le(value))
    }
    fn __ge__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.ge(value))
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
/// pol_magnitude : str
///     Name of the auxiliary scalar column storing the magnitude of the polarization vector
///
/// See Also
/// --------
/// laddu.utils.vectors.Vec3.mag
///
#[pyclass(name = "PolMagnitude", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPolMagnitude(pub PolMagnitude);

#[pymethods]
impl PyPolMagnitude {
    #[new]
    fn new(pol_magnitude: String) -> Self {
        Self(PolMagnitude::new(pol_magnitude))
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
    fn value(&self, event: &PyEvent) -> PyResult<f64> {
        let metadata = event
            .metadata_opt()
            .ok_or_else(|| PyValueError::new_err(
                "This event is not associated with metadata; supply `p4_names`/`aux_names` when constructing it or evaluate via a Dataset.",
            ))?;
        let mut variable = self.0.clone();
        variable.bind(metadata).map_err(PyErr::from)?;
        Ok(variable.value(&event.event))
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
    fn value_on<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.0.value_on(&dataset.0).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, values))
    }
    fn __eq__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.eq(value))
    }
    fn __lt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.lt(value))
    }
    fn __gt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.gt(value))
    }
    fn __le__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.le(value))
    }
    fn __ge__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.ge(value))
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
/// reaction : laddu.Reaction
///     Reaction describing the production kinematics and decay roots.
/// pol_magnitude : str
///     Name of the auxiliary scalar storing the polarization magnitude
/// pol_angle : str
///     Name of the auxiliary scalar storing the polarization angle in radians
///
/// See Also
/// --------
/// laddu.PolAngle
/// laddu.PolMagnitude
///
#[pyclass(name = "Polarization", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPolarization(pub Polarization);
#[pymethods]
impl PyPolarization {
    /// The Variable representing the magnitude of the polarization vector
    ///
    /// Returns
    /// -------
    /// PolMagnitude
    ///
    #[getter]
    fn pol_magnitude(&self) -> PyPolMagnitude {
        PyPolMagnitude(self.0.pol_magnitude.clone())
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
/// reaction : laddu.Reaction
///     Reaction describing the two-to-two kinematics whose Mandelstam channels should be evaluated.
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
/// ///
#[pyclass(name = "Mandelstam", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyMandelstam(pub Mandelstam);

#[pymethods]
impl PyMandelstam {
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
    fn value(&self, event: &PyEvent) -> PyResult<f64> {
        let metadata = event
            .metadata_opt()
            .ok_or_else(|| PyValueError::new_err(
                "This event is not associated with metadata; supply `p4_names`/`aux_names` when constructing it or evaluate via a Dataset.",
            ))?;
        let mut variable = self.0.clone();
        variable.bind(metadata).map_err(PyErr::from)?;
        Ok(variable.value(&event.event))
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
    fn value_on<'py>(
        &self,
        py: Python<'py>,
        dataset: &PyDataset,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.0.value_on(&dataset.0).map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, values))
    }
    fn __eq__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.eq(value))
    }
    fn __lt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.lt(value))
    }
    fn __gt__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.gt(value))
    }
    fn __le__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.le(value))
    }
    fn __ge__(&self, value: f64) -> PyVariableExpression {
        PyVariableExpression(self.0.ge(value))
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
    fn bind(&mut self, metadata: &DatasetMetadata) -> LadduResult<()> {
        match self {
            PyVariable::Mass(mass) => mass.0.bind(metadata),
            PyVariable::CosTheta(cos_theta) => cos_theta.0.bind(metadata),
            PyVariable::Phi(phi) => phi.0.bind(metadata),
            PyVariable::PolAngle(pol_angle) => pol_angle.0.bind(metadata),
            PyVariable::PolMagnitude(pol_magnitude) => pol_magnitude.0.bind(metadata),
            PyVariable::Mandelstam(mandelstam) => mandelstam.0.bind(metadata),
        }
    }

    fn value_on(&self, dataset: &Dataset) -> LadduResult<Vec<f64>> {
        match self {
            PyVariable::Mass(mass) => mass.0.value_on(dataset),
            PyVariable::CosTheta(cos_theta) => cos_theta.0.value_on(dataset),
            PyVariable::Phi(phi) => phi.0.value_on(dataset),
            PyVariable::PolAngle(pol_angle) => pol_angle.0.value_on(dataset),
            PyVariable::PolMagnitude(pol_magnitude) => pol_magnitude.0.value_on(dataset),
            PyVariable::Mandelstam(mandelstam) => mandelstam.0.value_on(dataset),
        }
    }

    fn value(&self, event: &dyn EventLike) -> f64 {
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
