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
    #[new]
    #[cfg(feature = "generation")]
    #[pyo3(signature = (name, *, p4=None, particle=None, output=false, mass_proposal=None, initial_momentum=None))]
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
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    fn p4(&self) -> Option<&str> {
        self.p4.as_deref()
    }
    #[getter]
    fn particle(&self) -> Option<PyParticle> {
        self.particle.clone()
    }
    #[getter]
    fn output(&self) -> bool {
        self.output
    }
}

#[pyclass(name = "Vertex", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyVertex {
    name: String,
    incoming: Vec<String>,
    outgoing: Vec<String>,
    #[cfg(feature = "generation")]
    generation: Option<PyVertexProposal>,
}

#[pymethods]
impl PyVertex {
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
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    fn incoming(&self) -> Vec<String> {
        self.incoming.clone()
    }
    #[getter]
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
pub struct PyChannel {
    pub(crate) inner: Channel,
}

#[pyclass(name = "VertexFrame", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyVertexFrame {
    channel: Channel,
    name: String,
}

#[pymethods]
impl PyVertexFrame {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

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

    fn theta(&self, edge: &str, z_axis: &PyVec3, y_hint: &PyVec3) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .theta(edge, z_axis.inner.clone(), y_hint.inner.clone())
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    fn costheta(&self, edge: &str, z_axis: &PyVec3, y_hint: &PyVec3) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .costheta(edge, z_axis.inner.clone(), y_hint.inner.clone())
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    fn phi(&self, edge: &str, z_axis: &PyVec3, y_hint: &PyVec3) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .phi(edge, z_axis.inner.clone(), y_hint.inner.clone())
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    fn mandelstam(&self, channel: &PyMandelstamChannel) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .mandelstam(channel.inner)
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    fn s(&self) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .s()
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

    fn t(&self) -> PyResult<PyExpr> {
        self.channel
            .get_vertex(&self.name)
            .map_err(to_py_err)?
            .t()
            .map(PyExpr::from)
            .map_err(to_py_err)
    }

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
    fn name(&self) -> &str {
        self.inner.name()
    }
    #[getter]
    fn edge_names(&self) -> Vec<String> {
        self.inner
            .edges()
            .map(|edge| edge.name().to_owned())
            .collect()
    }
    #[getter]
    fn vertex_names(&self) -> Vec<String> {
        self.inner
            .vertices()
            .map(|vertex| vertex.name().to_owned())
            .collect()
    }

    fn particle(&self, edge: &str) -> PyResult<PyParticle> {
        self.inner
            .particle(edge)
            .cloned()
            .map(PyParticle::from)
            .map_err(to_py_err)
    }

    fn vertex(&self, name: &str) -> PyResult<PyVertexFrame> {
        self.inner.get_vertex(name).map_err(to_py_err)?;
        Ok(PyVertexFrame {
            channel: self.inner.clone(),
            name: name.to_owned(),
        })
    }

    fn mass(&self, edge: &str) -> PyResult<PyExpr> {
        self.inner.mass(edge).map(PyExpr::from).map_err(to_py_err)
    }

    fn s(&self, edge: &str) -> PyResult<PyExpr> {
        self.inner.s(edge).map(PyExpr::from).map_err(to_py_err)
    }

    fn validate_generation(&self) -> PyResult<()> {
        self.inner.validate().map_err(to_py_err)
    }
}
