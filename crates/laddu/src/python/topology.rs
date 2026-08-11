use std::collections::HashSet;

use laddu_physics::{channel::Channel, vectors::Vec4};
use pyo3::{exceptions::PyValueError, prelude::*};

#[cfg(feature = "generation")]
use super::generation::{PyInitialMomentum, PyMassProposal, PyVertexProposal};
use super::{
    angular::PyVec3, error::to_py_err, expr::PyExpr, particle::PyParticle,
    quantum::PyMandelstamChannel,
};

#[pyclass(name = "Edge", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// An edge (particle line) in a reaction channel.
///
/// Parameters
/// ----------
/// name : str
///     Unique edge name.
/// p4 : str, optional
///     Dataset four-vector column used for observed events.
/// particle : Particle, optional
///     Particle properties such as mass and quantum numbers.
/// output : bool, default=True
///     Include this edge as a four-vector column in generated datasets.
/// mass_proposal : MassProposal, optional
///     Mass distribution for generation.
/// initial_momentum : InitialMomentum, optional
///     Momentum prescription for an initial-state edge.
pub struct PyEdge {
    name: String,
    p4: Option<String>,
    particle: Option<PyParticle>,
    output: bool,
    #[cfg(feature = "generation")]
    mass_proposal: Option<PyMassProposal>,
    #[cfg(feature = "generation")]
    initial_momentum: Option<PyInitialMomentum>,
}

#[pymethods]
impl PyEdge {
    /// Define a channel edge.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``name`` is empty.
    #[new]
    #[cfg(feature = "generation")]
    #[pyo3(signature = (name, *, p4=None, particle=None, output=true, mass_proposal=None, initial_momentum=None))]
    fn new(
        name: String,
        p4: Option<String>,
        particle: Option<PyRef<'_, PyParticle>>,
        output: bool,
        mass_proposal: Option<PyRef<'_, PyMassProposal>>,
        initial_momentum: Option<PyRef<'_, PyInitialMomentum>>,
    ) -> PyResult<Self> {
        if name.is_empty() {
            return Err(PyValueError::new_err("edge name cannot be empty"));
        }
        Ok(Self {
            name,
            p4,
            particle: particle.map(|particle| particle.clone()),
            output,
            mass_proposal: mass_proposal.map(|proposal| proposal.clone()),
            initial_momentum: initial_momentum.map(|momentum| momentum.clone()),
        })
    }

    fn __repr__(&self) -> String {
        format!("Edge({:?})", self.name)
    }

    #[getter]
    /// str: Unique edge name.
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    /// str or None: Dataset four-vector column.
    fn p4(&self) -> Option<&str> {
        self.p4.as_deref()
    }
    #[getter]
    /// Particle or None: Particle properties assigned to the edge.
    fn particle(&self) -> Option<PyParticle> {
        self.particle.clone()
    }
    #[getter]
    /// bool: Whether generated datasets include this edge.
    fn output(&self) -> bool {
        self.output
    }
}

#[pyclass(name = "Vertex", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A directed interaction or decay vertex.
///
/// Parameters
/// ----------
/// name : str
///     Unique vertex name.
/// incoming, outgoing : sequence of str
///     Edge names on each side of the vertex.
/// generation : VertexProposal, optional
///     Phase-space proposal used when generating this vertex.
pub struct PyVertex {
    name: String,
    incoming: Vec<String>,
    outgoing: Vec<String>,
    #[cfg(feature = "generation")]
    generation: Option<PyVertexProposal>,
}

#[pymethods]
impl PyVertex {
    /// Define a channel vertex.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the name is empty, either side is empty, or an edge is repeated.
    #[new]
    #[cfg(feature = "generation")]
    #[pyo3(signature = (name, *, incoming, outgoing, generation=None))]
    fn new(
        name: String,
        incoming: Vec<String>,
        outgoing: Vec<String>,
        generation: Option<PyRef<'_, PyVertexProposal>>,
    ) -> PyResult<Self> {
        validate_vertex_input(&name, &incoming, &outgoing)?;
        Ok(Self {
            name,
            incoming,
            outgoing,
            generation: generation.map(|proposal| proposal.clone()),
        })
    }

    fn __repr__(&self) -> String {
        format!("Vertex({:?})", self.name)
    }
    #[getter]
    /// str: Unique vertex name.
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    /// list of str: Incoming edge names.
    fn incoming(&self) -> Vec<String> {
        self.incoming.clone()
    }
    #[getter]
    /// list of str: Outgoing edge names.
    fn outgoing(&self) -> Vec<String> {
        self.outgoing.clone()
    }
}

fn validate_vertex_input(name: &str, incoming: &[String], outgoing: &[String]) -> PyResult<()> {
    if name.is_empty() {
        return Err(PyValueError::new_err("vertex name cannot be empty"));
    }
    if incoming.is_empty() || outgoing.is_empty() {
        return Err(PyValueError::new_err(
            "a vertex requires at least one incoming and one outgoing edge",
        ));
    }
    let mut seen = HashSet::new();
    if incoming
        .iter()
        .chain(outgoing)
        .any(|edge| !seen.insert(edge))
    {
        return Err(PyValueError::new_err(
            "an edge cannot appear more than once in a vertex",
        ));
    }
    Ok(())
}

#[pyclass(name = "Channel", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A validated reaction topology used to build kinematics and generators.
///
/// Parameters
/// ----------
/// name : str
///     Channel name.
/// edges : sequence of Edge
///     Particle lines with unique names.
/// vertices : sequence of Vertex
///     Directed interactions connecting the edges.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> beam = ld.Edge("beam", p4="beam", particle=ld.particles.PHOTON)
/// >>> target = ld.Edge("target", p4="target", particle=ld.particles.PROTON)
/// >>> recoil = ld.Edge("recoil", p4="recoil", particle=ld.particles.PROTON)
/// >>> production = ld.Vertex(
/// ...     "production", incoming=["beam", "target"], outgoing=["recoil"]
/// ... )
/// >>> channel = ld.Channel("gamma p", edges=\[beam, target, recoil\], vertices=\[production\])
pub struct PyChannel {
    pub(crate) inner: Channel,
}

#[pyclass(name = "VertexFrame", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Kinematic expressions evaluated in a vertex center-of-momentum frame.
///
/// Obtain a frame from :meth:`Channel.vertex`; instances cannot be constructed
/// directly.
pub struct PyVertexFrame {
    channel: Channel,
    name: String,
}

#[pymethods]
impl PyVertexFrame {
    #[getter]
    /// str: Name of the represented vertex.
    fn name(&self) -> &str {
        &self.name
    }

    /// Return an edge's three-momentum in this vertex frame.
    ///
    /// Parameters
    /// ----------
    /// edge : str
    ///     Edge incident on the vertex.
    ///
    /// Returns
    /// -------
    /// Vec3
    ///     Symbolic three-vector expression.
    fn vec3(&self, edge: &str) -> PyResult<PyVec3> {
        Ok(PyVec3 {
            inner: self
                .channel
                .get_vertex(&self.name)
                .map_err(to_py_err)?
                .vec3(edge)
                .map_err(to_py_err)?,
        })
    }

    /// Return the polar-angle expression for an edge.
    ///
    /// ``z_axis`` defines the polar axis and ``y_hint`` fixes the azimuthal
    /// orientation after orthogonalization.
    #[pyo3(signature = (edge, *, z_axis, y_hint))]
    fn theta(&self, edge: &str, z_axis: &PyVec3, y_hint: &PyVec3) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .theta(edge, z_axis.inner.clone(), y_hint.inner.clone())
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    /// Return the cosine of an edge's polar angle.
    #[pyo3(signature = (edge, *, z_axis, y_hint))]
    fn costheta(&self, edge: &str, z_axis: &PyVec3, y_hint: &PyVec3) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .costheta(edge, z_axis.inner.clone(), y_hint.inner.clone())
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    /// Return an edge's azimuthal-angle expression.
    #[pyo3(signature = (edge, *, z_axis, y_hint))]
    fn phi(&self, edge: &str, z_axis: &PyVec3, y_hint: &PyVec3) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .phi(edge, z_axis.inner.clone(), y_hint.inner.clone())
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    /// Return a Mandelstam invariant for this vertex.
    ///
    /// Parameters
    /// ----------
    /// channel : MandelstamChannel
    ///     One of the ``S``, ``T``, or ``U`` channels.
    fn mandelstam(&self, channel: &PyMandelstamChannel) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .mandelstam(channel.inner)
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    /// Return the Mandelstam-s invariant.
    fn s(&self) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .s()
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    /// Return the Mandelstam-t invariant.
    fn t(&self) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .t()
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    /// Return the Mandelstam-u invariant.
    fn u(&self) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .u()
            .map(PyExpr::from)
            .map_err(to_py_err)
    }
}

#[pymethods]
impl PyChannel {
    /// Construct and validate a reaction channel.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an edge or vertex name is duplicated.
    /// LadduError
    ///     If a vertex references invalid edges or violates topology rules.
    #[new]
    #[pyo3(signature = (name, *, edges, vertices))]
    fn new(
        name: String,
        edges: Vec<PyRef<'_, PyEdge>>,
        vertices: Vec<PyRef<'_, PyVertex>>,
    ) -> PyResult<Self> {
        let mut channel = Channel::new(name);
        let mut names = HashSet::new();
        for edge in edges {
            if !names.insert(edge.name.clone()) {
                return Err(PyValueError::new_err(format!(
                    "duplicate edge name {:?}",
                    edge.name
                )));
            }
            let mut handle = channel.edge(edge.name.clone());
            if let Some(p4) = &edge.p4 {
                handle.p4(Vec4::event(p4));
            }
            if let Some(particle) = &edge.particle {
                handle.properties(&particle.inner);
            }
            if edge.output {
                handle.output();
            } else {
                handle.generated_only();
            }
            #[cfg(feature = "generation")]
            {
                if let Some(proposal) = &edge.mass_proposal {
                    handle.mass_proposal(proposal.inner);
                }
                if let Some(momentum) = &edge.initial_momentum {
                    handle.initial(momentum.inner.clone());
                }
            }
        }
        let mut vertex_names = HashSet::new();
        for vertex in vertices {
            if !vertex_names.insert(vertex.name.clone()) {
                return Err(PyValueError::new_err(format!(
                    "duplicate vertex name {:?}",
                    vertex.name
                )));
            }
            let mut handle = channel.vertex(vertex.name.clone());
            handle.incoming(&vertex.incoming).outgoing(&vertex.outgoing);
            #[cfg(feature = "generation")]
            if let Some(proposal) = &vertex.generation {
                handle.generation(proposal.inner.clone());
            }
            handle.validate().map_err(to_py_err)?;
        }
        Ok(Self { inner: channel })
    }

    fn __repr__(&self) -> String {
        format!(
            "Channel({:?}, edges={}, vertices={})",
            self.inner.name(),
            self.inner.edges().count(),
            self.inner.vertices().count()
        )
    }

    #[getter]
    /// str: Channel name.
    fn name(&self) -> &str {
        self.inner.name()
    }
    #[getter]
    /// list of str: Edge names in channel order.
    fn edge_names(&self) -> Vec<String> {
        self.inner
            .edges()
            .map(|edge| edge.name().to_owned())
            .collect()
    }
    #[getter]
    /// list of str: Vertex names in channel order.
    fn vertex_names(&self) -> Vec<String> {
        self.inner
            .vertices()
            .map(|vertex| vertex.name().to_owned())
            .collect()
    }

    /// Return particle properties assigned to an edge.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the edge is unknown or has no particle properties.
    fn particle(&self, edge: &str) -> PyResult<PyParticle> {
        self.inner
            .particle(edge)
            .cloned()
            .map(PyParticle::from)
            .map_err(to_py_err)
    }

    /// Return the center-of-momentum frame for a named vertex.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the vertex is unknown.
    fn vertex(&self, name: &str) -> PyResult<PyVertexFrame> {
        self.inner.get_vertex(name).map_err(to_py_err)?;
        Ok(PyVertexFrame {
            channel: self.inner.clone(),
            name: name.to_owned(),
        })
    }

    /// Return the invariant-mass expression for an edge.
    fn mass(&self, edge: &str) -> PyResult<PyExpr> {
        self.inner.mass(edge).map(PyExpr::from).map_err(to_py_err)
    }

    /// Return the squared invariant-mass expression for an edge.
    fn s(&self, edge: &str) -> PyResult<PyExpr> {
        self.inner.s(edge).map(PyExpr::from).map_err(to_py_err)
    }

    /// Validate all information required for event generation.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If initial states, masses, outputs, or vertex proposals are invalid.
    fn validate_generation(&self) -> PyResult<()> {
        self.inner.validate().map_err(to_py_err)
    }
}

impl_json_methods!(PyChannel);
