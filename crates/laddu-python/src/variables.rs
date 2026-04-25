use std::fmt::{Debug, Display};

use laddu_amplitudes::DecayAmplitudeExt;
use laddu_core::{
    data::{Dataset, DatasetMetadata, Event, NamedEventView},
    reaction::{Decay, Particle, Reaction},
    traits::Variable,
    variables::{
        Angles, CosTheta, IntoP4Selection, Mandelstam, Mass, P4Selection, Phi, PolAngle,
        PolMagnitude, Polarization, VariableExpression,
    },
    LadduResult,
};
use numpy::PyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

use crate::{
    amplitudes::PyExpression,
    data::{PyDataset, PyEvent},
    quantum::angular_momentum::{
        parse_angular_momentum, parse_orbital_angular_momentum, parse_projection,
    },
    vectors::PyVec4,
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

    pub(crate) fn evaluate_event(&self, event: &Event) -> PyResult<f64> {
        let dataset = Dataset::new_with_metadata(vec![event.data_arc()], event.metadata_arc());
        let event_view = dataset.event_view(0);
        Ok(self.value(&event_view))
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

#[derive(Clone, FromPyObject)]
pub enum PyP4SelectionInput {
    #[pyo3(transparent)]
    Name(String),
    #[pyo3(transparent)]
    Names(Vec<String>),
}

impl PyP4SelectionInput {
    fn into_selection(self) -> P4Selection {
        match self {
            PyP4SelectionInput::Name(name) => name.into_selection(),
            PyP4SelectionInput::Names(names) => names.into_selection(),
        }
    }
}

/// A kinematic particle used to define reaction-aware variables.
#[pyclass(name = "Particle", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyParticle(pub Particle);

#[pymethods]
impl PyParticle {
    /// Construct a measured particle from one or more p4 column names.
    #[staticmethod]
    fn measured(label: &str, p4: PyP4SelectionInput) -> Self {
        Self(Particle::measured(label, p4.into_selection()))
    }

    /// Construct a particle with fixed event-independent four-momentum.
    #[staticmethod]
    fn fixed(label: &str, p4: &PyVec4) -> Self {
        Self(Particle::fixed(label, p4.0))
    }

    /// Construct a missing particle solved by the reaction topology.
    #[staticmethod]
    fn missing(label: &str) -> Self {
        Self(Particle::missing(label))
    }

    /// Construct a composite particle from daughter particles.
    #[staticmethod]
    fn composite(label: &str, daughters: Vec<PyParticle>) -> PyResult<Self> {
        Ok(Self(Particle::composite(
            label,
            daughters.iter().map(|daughter| &daughter.0),
        )?))
    }

    /// The particle label.
    #[getter]
    fn label(&self) -> String {
        self.0.label().to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

/// A reaction topology with direct particle definitions.
#[pyclass(name = "Reaction", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyReaction(pub Reaction);

#[pymethods]
impl PyReaction {
    /// Construct a two-to-two reaction from `p1, p2, p3, p4`.
    #[staticmethod]
    fn two_to_two(
        p1: &PyParticle,
        p2: &PyParticle,
        p3: &PyParticle,
        p4: &PyParticle,
    ) -> PyResult<Self> {
        Ok(Self(Reaction::two_to_two(&p1.0, &p2.0, &p3.0, &p4.0)?))
    }

    /// Construct a particle mass variable.
    fn mass(&self, particle: &PyParticle) -> PyMass {
        PyMass(self.0.mass(&particle.0))
    }

    /// Construct an isobar decay view.
    fn decay(&self, parent: &PyParticle) -> PyResult<PyDecay> {
        Ok(PyDecay(self.0.decay(&parent.0)?))
    }

    /// Construct a Mandelstam variable.
    fn mandelstam(&self, channel: &str) -> PyResult<PyMandelstam> {
        Ok(PyMandelstam(self.0.mandelstam(channel.parse()?)))
    }

    /// Construct a polarization-angle variable.
    fn pol_angle(&self, angle_aux: String) -> PyPolAngle {
        PyPolAngle(self.0.pol_angle(angle_aux))
    }

    /// Construct polarization variables.
    fn polarization(&self, pol_magnitude: String, pol_angle: String) -> PyResult<PyPolarization> {
        if pol_magnitude == pol_angle {
            return Err(PyValueError::new_err(
                "`pol_magnitude` and `pol_angle` must reference distinct auxiliary columns",
            ));
        }
        Ok(PyPolarization(
            self.0.polarization(pol_magnitude, pol_angle),
        ))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A reaction-aware isobar decay view.
#[pyclass(name = "Decay", module = "laddu", from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyDecay(pub Decay);

#[pymethods]
impl PyDecay {
    /// The parent particle.
    #[getter]
    fn parent(&self) -> PyParticle {
        PyParticle(self.0.parent().clone())
    }

    /// The first daughter particle.
    #[getter]
    fn daughter_1(&self) -> PyParticle {
        PyParticle(self.0.daughter_1().clone())
    }

    /// The second daughter particle.
    #[getter]
    fn daughter_2(&self) -> PyParticle {
        PyParticle(self.0.daughter_2().clone())
    }

    /// Ordered daughter particles.
    fn daughters(&self) -> Vec<PyParticle> {
        self.0
            .daughters()
            .into_iter()
            .map(|daughter| PyParticle(daughter.clone()))
            .collect()
    }

    /// Parent mass variable.
    fn mass(&self) -> PyMass {
        PyMass(self.0.mass())
    }

    /// Parent mass variable.
    fn parent_mass(&self) -> PyMass {
        PyMass(self.0.parent_mass())
    }

    /// First daughter mass variable.
    fn daughter_1_mass(&self) -> PyMass {
        PyMass(self.0.daughter_1_mass())
    }

    /// Second daughter mass variable.
    fn daughter_2_mass(&self) -> PyMass {
        PyMass(self.0.daughter_2_mass())
    }

    /// Mass variable for a selected daughter.
    fn daughter_mass(&self, daughter: &PyParticle) -> PyResult<PyMass> {
        Ok(PyMass(self.0.daughter_mass(&daughter.0)?))
    }

    /// Decay costheta variable for the selected frame.
    #[pyo3(signature=(daughter, frame="Helicity"))]
    fn costheta(&self, daughter: &PyParticle, frame: &str) -> PyResult<PyCosTheta> {
        Ok(PyCosTheta(self.0.costheta(&daughter.0, frame.parse()?)?))
    }

    /// Decay phi variable for the selected frame.
    #[pyo3(signature=(daughter, frame="Helicity"))]
    fn phi(&self, daughter: &PyParticle, frame: &str) -> PyResult<PyPhi> {
        Ok(PyPhi(self.0.phi(&daughter.0, frame.parse()?)?))
    }

    /// Decay angle variables for the selected frame.
    #[pyo3(signature=(daughter, frame="Helicity"))]
    fn angles(&self, daughter: &PyParticle, frame: &str) -> PyResult<PyAngles> {
        Ok(PyAngles(self.0.angles(&daughter.0, frame.parse()?)?))
    }

    /// Construct the helicity-basis angular factor for one explicit helicity term.
    #[pyo3(signature=(name, spin, projection, daughter, lambda_1, lambda_2, frame="Helicity"))]
    #[allow(clippy::too_many_arguments)]
    fn helicity_factor(
        &self,
        name: &str,
        spin: &Bound<'_, PyAny>,
        projection: &Bound<'_, PyAny>,
        daughter: &PyParticle,
        lambda_1: &Bound<'_, PyAny>,
        lambda_2: &Bound<'_, PyAny>,
        frame: &str,
    ) -> PyResult<PyExpression> {
        Ok(PyExpression(self.0.helicity_factor(
            name,
            parse_angular_momentum(spin)?,
            parse_projection(projection)?,
            &daughter.0,
            parse_projection(lambda_1)?,
            parse_projection(lambda_2)?,
            frame.parse()?,
        )?))
    }

    /// Construct the canonical-basis spin-angular factor for one explicit LS/helicity term.
    #[pyo3(signature=(name, spin, projection, orbital_l, coupled_spin, daughter, daughter_1_spin, daughter_2_spin, lambda_1, lambda_2, frame="Helicity"))]
    #[allow(clippy::too_many_arguments)]
    fn canonical_factor(
        &self,
        name: &str,
        spin: &Bound<'_, PyAny>,
        projection: &Bound<'_, PyAny>,
        orbital_l: &Bound<'_, PyAny>,
        coupled_spin: &Bound<'_, PyAny>,
        daughter: &PyParticle,
        daughter_1_spin: &Bound<'_, PyAny>,
        daughter_2_spin: &Bound<'_, PyAny>,
        lambda_1: &Bound<'_, PyAny>,
        lambda_2: &Bound<'_, PyAny>,
        frame: &str,
    ) -> PyResult<PyExpression> {
        Ok(PyExpression(self.0.canonical_factor(
            name,
            parse_angular_momentum(spin)?,
            parse_projection(projection)?,
            parse_orbital_angular_momentum(orbital_l)?,
            parse_angular_momentum(coupled_spin)?,
            &daughter.0,
            parse_angular_momentum(daughter_1_spin)?,
            parse_angular_momentum(daughter_2_spin)?,
            parse_projection(lambda_1)?,
            parse_projection(lambda_2)?,
            frame.parse()?,
        )?))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
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
    #[new]
    fn new(constituents: PyP4SelectionInput) -> Self {
        Self(Mass::new(constituents.into_selection()))
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
        let dataset =
            Dataset::new_with_metadata(vec![event.event.data_arc()], event.event.metadata_arc());
        let event_view = dataset.event_view(0);
        Ok(variable.value(&event_view))
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
        let dataset =
            Dataset::new_with_metadata(vec![event.event.data_arc()], event.event.metadata_arc());
        let event_view = dataset.event_view(0);
        Ok(variable.value(&event_view))
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
        let dataset =
            Dataset::new_with_metadata(vec![event.event.data_arc()], event.event.metadata_arc());
        let event_view = dataset.event_view(0);
        Ok(variable.value(&event_view))
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
    #[new]
    fn new(reaction: PyReaction, pol_angle: String) -> Self {
        Self(PolAngle::new(reaction.0.clone(), pol_angle))
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
        let dataset =
            Dataset::new_with_metadata(vec![event.event.data_arc()], event.event.metadata_arc());
        let event_view = dataset.event_view(0);
        Ok(variable.value(&event_view))
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
        let dataset =
            Dataset::new_with_metadata(vec![event.event.data_arc()], event.event.metadata_arc());
        let event_view = dataset.event_view(0);
        Ok(variable.value(&event_view))
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
    #[new]
    #[pyo3(signature=(reaction, *, pol_magnitude, pol_angle))]
    fn new(reaction: PyReaction, pol_magnitude: String, pol_angle: String) -> PyResult<Self> {
        if pol_magnitude == pol_angle {
            return Err(PyValueError::new_err(
                "`pol_magnitude` and `pol_angle` must reference distinct auxiliary columns",
            ));
        }
        let polarization = Polarization::new(reaction.0.clone(), pol_magnitude, pol_angle);
        Ok(PyPolarization(polarization))
    }
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
    #[new]
    fn new(reaction: PyReaction, channel: &str) -> PyResult<Self> {
        Ok(Self(Mandelstam::new(reaction.0.clone(), channel.parse()?)))
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
        let dataset =
            Dataset::new_with_metadata(vec![event.event.data_arc()], event.event.metadata_arc());
        let event_view = dataset.event_view(0);
        Ok(variable.value(&event_view))
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

    fn value(&self, event: &NamedEventView<'_>) -> f64 {
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
