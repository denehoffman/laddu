use std::collections::HashSet;

use indexmap::IndexMap;
use laddu_expr::Expr;
use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    generation::{InitialMomentum, MassProposal, ScalarSource, VertexProposal},
    quantum::{MandelstamChannel, ParticleProperties},
    vectors::{RealVec3, RealVec4, Vec3, Vec4},
};

mod frame;
mod resolution;
mod topology;

#[derive(Clone, Debug, Serialize, Deserialize)]
/// A named reaction graph whose edges are particles and whose vertices are
/// production or decay processes.
pub struct Channel {
    name: String,
    edges: IndexMap<String, Edge>,
    vertices: IndexMap<String, Vertex>,
}

impl Channel {
    /// Construct an empty channel with the supplied name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            edges: IndexMap::new(),
            vertices: IndexMap::new(),
        }
    }

    /// Return the channel name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create or edit a named particle edge.
    ///
    /// # Panics
    ///
    /// Panics only if the internal edge map fails to return an entry that was
    /// just inserted.
    pub fn edge(&mut self, name: impl Into<String>) -> EdgeHandle<'_> {
        let name = name.into();
        self.edges
            .entry(name.clone())
            .or_insert_with(|| Edge::new(name.clone()));
        EdgeHandle {
            edge: self.edges.get_mut(&name).expect("edge was just inserted"),
        }
    }

    /// Create or edit a named interaction vertex.
    pub fn vertex(&mut self, name: impl Into<String>) -> VertexHandle<'_> {
        let name = name.into();
        self.vertices
            .entry(name.clone())
            .or_insert_with(|| Vertex::new(name.clone()));
        VertexHandle {
            channel: self,
            name,
        }
    }

    /// Retrieve a vertex view which evaluates quantities in that vertex's rest frame.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `name` does not identify a vertex in
    /// this channel.
    pub fn get_vertex(&self, name: &str) -> LadduPhysicsResult<VertexView<'_>> {
        self.require_vertex(name)?;
        Ok(VertexView {
            channel: self,
            name: name.to_owned(),
        })
    }

    /// Iterate over particle edges in insertion order.
    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

    /// Iterate over interaction vertices in insertion order.
    pub fn vertices(&self) -> impl Iterator<Item = &Vertex> {
        self.vertices.values()
    }

    /// Iterate over edges consumed by the graph but not produced by any vertex.
    pub fn initial_edges(&self) -> impl Iterator<Item = &Edge> {
        let consumed = self
            .vertices
            .values()
            .flat_map(|vertex| vertex.incoming.iter().map(String::as_str))
            .collect::<HashSet<_>>();
        let produced = self
            .vertices
            .values()
            .flat_map(|vertex| vertex.outgoing.iter().map(String::as_str))
            .collect::<HashSet<_>>();
        self.edges
            .values()
            .filter(move |edge| consumed.contains(edge.name()) && !produced.contains(edge.name()))
    }

    /// Validate all channel vertices.
    ///
    /// Consumers such as event generators call this automatically.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a vertex references missing or
    /// duplicate edges, the reaction graph is inconsistent, or an initial edge
    /// lacks a valid momentum source.
    pub fn validate(&self) -> LadduPhysicsResult<()> {
        let topology = topology::TopologyIndex::new(self);
        for name in self.vertices.keys() {
            topology.validate_vertex(name)?;
        }
        for edge in self.initial_edges() {
            edge.initial_momentum
                .as_ref()
                .ok_or_else(|| {
                    LadduPhysicsError::invalid_relation(format!(
                        "initial edge `{}` has no momentum source",
                        edge.name()
                    ))
                })?
                .validate(edge.name(), edge.properties())?;
        }
        Ok(())
    }

    /// Return the optional particle properties attached to an edge.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `edge` is not in this channel.
    pub fn properties(&self, edge: &str) -> LadduPhysicsResult<Option<&ParticleProperties>> {
        Ok(self.require_edge(edge)?.properties.as_ref())
    }

    /// Return the particle definition attached to an edge.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `edge` is unknown or has no particle
    /// properties.
    pub fn particle(&self, edge: &str) -> LadduPhysicsResult<&ParticleProperties> {
        self.properties(edge)?.ok_or_else(|| {
            LadduPhysicsError::invalid_relation(format!("edge `{edge}` has no particle properties"))
        })
    }

    /// Construct the symbolic four-momentum of an edge.
    ///
    /// When no explicit expression is attached, momentum conservation is used
    /// to infer a uniquely missing momentum at a connected vertex.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge is unknown or its momentum
    /// cannot be uniquely resolved from the channel.
    pub fn p4(&self, edge: &str) -> LadduPhysicsResult<Vec4> {
        self.resolved_p4(edge)
    }

    /// Construct the symbolic three-momentum of an edge.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge's four-momentum cannot be
    /// resolved.
    pub fn vec3(&self, edge: &str) -> LadduPhysicsResult<Vec3> {
        Ok(self.p4(edge)?.vec3())
    }

    /// Construct the invariant-mass expression for an edge.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge's four-momentum cannot be
    /// resolved.
    pub fn mass(&self, edge: &str) -> LadduPhysicsResult<Expr> {
        Ok(self.p4(edge)?.m())
    }

    /// Construct the invariant mass squared of an edge.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge's four-momentum cannot be
    /// resolved.
    pub fn s(&self, edge: &str) -> LadduPhysicsResult<Expr> {
        Ok(self.p4(edge)?.m2())
    }

    fn require_edge(&self, name: &str) -> LadduPhysicsResult<&Edge> {
        self.edges
            .get(name)
            .ok_or_else(|| LadduPhysicsError::invalid_relation(format!("unknown edge `{name}`")))
    }

    fn require_vertex(&self, name: &str) -> LadduPhysicsResult<&Vertex> {
        self.vertices
            .get(name)
            .ok_or_else(|| LadduPhysicsError::invalid_relation(format!("unknown vertex `{name}`")))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// A particle line in a [`Channel`].
pub struct Edge {
    name: String,
    p4: Option<Vec4>,
    properties: Option<ParticleProperties>,
    output: bool,
    mass_proposal: Option<MassProposal>,
    initial_momentum: Option<InitialMomentum>,
}

impl Edge {
    fn new(name: String) -> Self {
        Self {
            name,
            p4: None,
            properties: None,
            output: true,
            mass_proposal: None,
            initial_momentum: None,
        }
    }

    /// Return the edge name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return whether this edge has an explicitly assigned four-momentum.
    pub fn has_explicit_p4(&self) -> bool {
        self.p4.is_some()
    }

    /// Return the particle properties attached to this edge, if present.
    pub fn properties(&self) -> Option<&ParticleProperties> {
        self.properties.as_ref()
    }

    /// Return whether generated events retain this edge as an output particle.
    pub fn is_output(&self) -> bool {
        self.output
    }

    /// Return the mass proposal used to generate this edge, if configured.
    pub fn mass_proposal(&self) -> Option<&MassProposal> {
        self.mass_proposal.as_ref()
    }

    /// Return the initial-state momentum source, if configured.
    pub fn initial_momentum(&self) -> Option<&InitialMomentum> {
        self.initial_momentum.as_ref()
    }
}

/// Mutable builder handle for an [`Edge`] in a channel.
pub struct EdgeHandle<'a> {
    edge: &'a mut Edge,
}

impl EdgeHandle<'_> {
    /// Mutable access to the underlying [`Edge`].
    pub fn edge_mut(&mut self) -> &mut Edge {
        self.edge
    }
    /// Assign an explicit symbolic four-momentum.
    pub fn p4(&mut self, p4: impl Into<Vec4>) -> &mut Self {
        self.edge.p4 = Some(p4.into());
        self
    }

    /// Attach particle properties.
    pub fn properties(&mut self, properties: &ParticleProperties) -> &mut Self {
        self.edge.properties = Some(properties.clone());
        self
    }

    /// Retain this edge in generated event output.
    pub fn output(&mut self) -> &mut Self {
        self.edge.output = true;
        self
    }

    /// Use this edge internally during generation without retaining it.
    pub fn generated_only(&mut self) -> &mut Self {
        self.edge.output = false;
        self
    }

    /// Assign a generated invariant-mass proposal.
    pub fn mass_proposal(&mut self, proposal: impl Into<MassProposal>) -> &mut Self {
        self.edge.mass_proposal = Some(proposal.into());
        self
    }

    /// Assign an initial-state momentum source.
    pub fn initial(&mut self, source: InitialMomentum) -> &mut Self {
        self.edge.initial_momentum = Some(source);
        self
    }

    /// Assign a fixed initial-state four-momentum in `(E, px, py, pz)` order.
    pub fn initial_p4(&mut self, p4: RealVec4) -> &mut Self {
        self.initial(InitialMomentum::p4(p4))
    }

    /// Assign a fixed initial-state three-momentum.
    pub fn initial_momentum(&mut self, momentum: RealVec3) -> &mut Self {
        self.initial(InitialMomentum::momentum(momentum))
    }

    /// Assign a fixed energy and direction for an initial-state edge.
    pub fn initial_energy_direction(&mut self, energy: f64, direction: RealVec3) -> &mut Self {
        self.initial(InitialMomentum::energy_direction(energy, direction))
    }

    /// Assign a sampled energy source and fixed direction for an initial-state edge.
    pub fn initial_energy_source_direction(
        &mut self,
        energy: ScalarSource,
        direction: RealVec3,
    ) -> &mut Self {
        self.initial(InitialMomentum::energy_source_direction(energy, direction))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// A production, scattering, or decay vertex in a [`Channel`].
pub struct Vertex {
    name: String,
    incoming: Vec<String>,
    outgoing: Vec<String>,
    generation: Option<VertexProposal>,
}

impl Vertex {
    fn new(name: String) -> Self {
        Self {
            name,
            incoming: Vec::new(),
            outgoing: Vec::new(),
            generation: None,
        }
    }

    /// Return the vertex name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the names of incoming edges.
    pub fn incoming(&self) -> &[String] {
        &self.incoming
    }

    /// Return the names of outgoing edges.
    pub fn outgoing(&self) -> &[String] {
        &self.outgoing
    }

    /// Return the configured event-generation proposal, if any.
    pub fn generation(&self) -> Option<&VertexProposal> {
        self.generation.as_ref()
    }

    fn all_edges(&self) -> impl Iterator<Item = &String> {
        self.incoming.iter().chain(&self.outgoing)
    }

    fn matches_priority(&self, edge: &str, priority: InferencePriority) -> bool {
        match priority {
            InferencePriority::ParentFromDaughters => {
                self.incoming.len() == 1 && self.incoming.iter().any(|candidate| candidate == edge)
            }
            InferencePriority::ChildFromParents => {
                self.outgoing.len() == 1 && self.outgoing.iter().any(|candidate| candidate == edge)
            }
            InferencePriority::AnySingleMissing => true,
        }
    }
}

/// Mutable builder handle for a [`Vertex`] in a channel.
pub struct VertexHandle<'a> {
    channel: &'a mut Channel,
    name: String,
}

impl VertexHandle<'_> {
    /// Assign the proposal used to generate this vertex.
    ///
    /// # Panics
    ///
    /// Panics if the vertex associated with this live handle is missing from
    /// the channel.
    pub fn generation(&mut self, proposal: impl Into<VertexProposal>) -> &mut Self {
        self.channel
            .vertices
            .get_mut(&self.name)
            .expect("vertex handle references an existing vertex")
            .generation = Some(proposal.into());
        self
    }

    /// Replace the vertex's incoming edge list.
    ///
    /// # Panics
    ///
    /// Panics if the vertex associated with this live handle is missing from
    /// the channel.
    pub fn incoming(&mut self, edges: impl IntoIterator<Item = impl AsRef<str>>) -> &mut Self {
        let edges = edges
            .into_iter()
            .map(|edge| edge.as_ref().to_owned())
            .collect::<Vec<_>>();
        self.channel
            .vertices
            .get_mut(&self.name)
            .expect("vertex handle references an existing vertex")
            .incoming = edges;
        self
    }

    /// Replace the vertex's outgoing edge list.
    ///
    /// # Panics
    ///
    /// Panics if the vertex associated with this live handle is missing from
    /// the channel.
    pub fn outgoing(&mut self, edges: impl IntoIterator<Item = impl AsRef<str>>) -> &mut Self {
        let edges = edges
            .into_iter()
            .map(|edge| edge.as_ref().to_owned())
            .collect::<Vec<_>>();
        self.channel
            .vertices
            .get_mut(&self.name)
            .expect("vertex handle references an existing vertex")
            .outgoing = edges;
        self
    }

    /// Validate that every referenced edge exists and occurs only once.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the vertex references an unknown or
    /// duplicate edge or otherwise violates channel topology.
    pub fn validate(&self) -> LadduPhysicsResult<()> {
        self.channel.validate_vertex(&self.name)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InferencePriority {
    ParentFromDaughters,
    ChildFromParents,
    AnySingleMissing,
}

#[derive(Clone, Debug)]
/// Read-only access to channel expressions evaluated in a vertex rest frame.
pub struct VertexView<'a> {
    channel: &'a Channel,
    name: String,
}

impl<'a> VertexView<'a> {
    /// Return the underlying vertex definition.
    ///
    /// # Panics
    ///
    /// Panics if the vertex associated with this view is missing from the
    /// channel.
    pub fn vertex(&self) -> &'a Vertex {
        self.channel
            .vertices
            .get(&self.name)
            .expect("vertex view references an existing vertex")
    }

    /// Construct an edge four-momentum boosted into this vertex's rest frame.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge, frame path, or required
    /// momenta cannot be resolved.
    pub fn p4(&self, edge: &str) -> LadduPhysicsResult<Vec4> {
        self.channel.p4_in_frame(&self.name, edge)
    }

    /// Construct an edge three-momentum in this vertex's rest frame.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge's rest-frame momentum cannot
    /// be resolved.
    pub fn vec3(&self, edge: &str) -> LadduPhysicsResult<Vec3> {
        Ok(self.p4(edge)?.vec3())
    }

    /// Construct the cosine of an edge's polar angle about the supplied axis.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge's rest-frame momentum cannot
    /// be resolved.
    pub fn costheta(&self, edge: &str, z_axis: Vec3, _y_hint: Vec3) -> LadduPhysicsResult<Expr> {
        let p = self.vec3(edge)?;
        let z = z_axis.unit();
        Ok(p.dot(&z) / p.mag())
    }

    /// Construct an edge's polar angle about the supplied axis.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge's rest-frame momentum cannot
    /// be resolved.
    pub fn theta(&self, edge: &str, z_axis: Vec3, y_hint: Vec3) -> LadduPhysicsResult<Expr> {
        Ok(self.costheta(edge, z_axis, y_hint)?.acos())
    }

    /// Construct an edge's azimuthal angle in the supplied coordinate frame.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the edge's rest-frame momentum cannot
    /// be resolved.
    pub fn phi(&self, edge: &str, z_axis: Vec3, y_hint: Vec3) -> LadduPhysicsResult<Expr> {
        let p = self.vec3(edge)?;
        let z = z_axis.unit();
        let z_component = &z * &y_hint.dot(&z);
        let y = (y_hint - z_component).unit();
        let x = y.cross(&z);
        Ok(laddu_expr::atan2(p.dot(&y), p.dot(&x)))
    }

    /// Construct one of the Mandelstam invariants for a two-to-two vertex.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when this is not a two-to-two vertex or
    /// the required edge momenta cannot be resolved.
    pub fn mandelstam(&self, channel: MandelstamChannel) -> LadduPhysicsResult<Expr> {
        let vertex = self.vertex();
        if vertex.incoming.len() != 2 || vertex.outgoing.len() != 2 {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "vertex `{}` is not 2-to-2",
                vertex.name()
            )));
        }
        let pairs = match channel {
            MandelstamChannel::S => [
                (&vertex.incoming[0], &vertex.incoming[1], PairOp::Sum),
                (&vertex.outgoing[0], &vertex.outgoing[1], PairOp::Sum),
            ],
            MandelstamChannel::T => [
                (&vertex.incoming[0], &vertex.outgoing[0], PairOp::Difference),
                (&vertex.incoming[1], &vertex.outgoing[1], PairOp::Difference),
            ],
            MandelstamChannel::U => [
                (&vertex.incoming[0], &vertex.outgoing[1], PairOp::Difference),
                (&vertex.incoming[1], &vertex.outgoing[0], PairOp::Difference),
            ],
        };
        let mut best = None;
        for (lhs, rhs, op) in pairs {
            let Ok(expr) = self.pair_mandelstam(lhs, rhs, op) else {
                continue;
            };
            let score = usize::from(self.channel.edge_is_explicit(lhs))
                + usize::from(self.channel.edge_is_explicit(rhs));
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score > *best_score)
            {
                best = Some((score, expr));
            }
        }
        best.map(|(_, expr)| expr).ok_or_else(|| {
            LadduPhysicsError::invalid_relation(format!(
                "could not construct {channel} for vertex `{}`",
                vertex.name()
            ))
        })
    }

    /// Construct the Mandelstam `s` invariant.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when this is not a two-to-two vertex or
    /// the required edge momenta cannot be resolved.
    pub fn s(&self) -> LadduPhysicsResult<Expr> {
        self.mandelstam(MandelstamChannel::S)
    }

    /// Construct the Mandelstam `t` invariant.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when this is not a two-to-two vertex or
    /// the required edge momenta cannot be resolved.
    pub fn t(&self) -> LadduPhysicsResult<Expr> {
        self.mandelstam(MandelstamChannel::T)
    }

    /// Construct the Mandelstam `u` invariant.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when this is not a two-to-two vertex or
    /// the required edge momenta cannot be resolved.
    pub fn u(&self) -> LadduPhysicsResult<Expr> {
        self.mandelstam(MandelstamChannel::U)
    }

    fn pair_mandelstam(&self, lhs: &str, rhs: &str, op: PairOp) -> LadduPhysicsResult<Expr> {
        let lhs = self.channel.p4(lhs)?;
        let rhs = self.channel.p4(rhs)?;
        Ok(match op {
            PairOp::Sum => (lhs + rhs).m2(),
            PairOp::Difference => (lhs - rhs).m2(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum PairOp {
    Sum,
    Difference,
}

impl Channel {
    fn validate_vertex(&self, name: &str) -> LadduPhysicsResult<()> {
        topology::TopologyIndex::new(self).validate_vertex(name)
    }

    fn edge_is_explicit(&self, edge: &str) -> bool {
        self.edges.get(edge).is_some_and(Edge::has_explicit_p4)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_runtime::CpuBackend;

    use super::*;
    use crate::vectors::{RealVec3, RealVec4};

    fn eval(expr: Expr) -> f64 {
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = model.params().default_values();
        CpuBackend.prepare(&model).evaluate(&params).unwrap().re
    }

    fn p4(px: f64, py: f64, pz: f64, e: f64) -> Vec4 {
        RealVec4::new(e, px, py, pz).into()
    }

    fn expr_p4(value: RealVec4) -> Vec4 {
        value.into()
    }

    fn assert_vec4_close(actual: Vec4, expected: RealVec4) {
        assert_relative_eq!(eval(actual.px()), expected.px(), epsilon = 1e-12);
        assert_relative_eq!(eval(actual.py()), expected.py(), epsilon = 1e-12);
        assert_relative_eq!(eval(actual.pz()), expected.pz(), epsilon = 1e-12);
        assert_relative_eq!(eval(actual.e()), expected.e(), epsilon = 1e-12);
    }

    #[test]
    fn edges_are_outputs_by_default() {
        let mut channel = Channel::new("generated");
        channel.edge("included");
        channel.edge("excluded").generated_only();

        assert!(channel.require_edge("included").unwrap().is_output());
        assert!(!channel.require_edge("excluded").unwrap().is_output());
    }

    #[test]
    fn infers_parent_p4_from_decay_vertex() {
        let mut channel = Channel::new("KsKs");
        channel.edge("ks1").p4(p4(0.0, 0.0, 1.0, 1.0));
        channel.edge("ks2").p4(p4(0.0, 0.0, -1.0, 1.0));
        channel.edge("x");
        channel
            .vertex("x_decay")
            .incoming(["x"])
            .outgoing(["ks1", "ks2"])
            .validate()
            .unwrap();

        assert_relative_eq!(eval(channel.mass("x").unwrap()), 2.0);
    }

    #[test]
    fn infers_single_missing_vertex_edge_from_conservation() {
        let mut channel = Channel::new("KsKs");
        channel.edge("beam").p4(p4(0.0, 0.0, 3.0, 5.0));
        channel.edge("x").p4(p4(0.0, 0.0, 1.0, 3.0));
        channel.edge("recoil").p4(p4(0.0, 0.0, 2.0, 3.0));
        channel.edge("target");
        channel
            .vertex("production")
            .incoming(["beam", "target"])
            .outgoing(["x", "recoil"])
            .validate()
            .unwrap();

        let target = channel.p4("target").unwrap();
        assert_relative_eq!(eval(target.px()), 0.0);
        assert_relative_eq!(eval(target.py()), 0.0);
        assert_relative_eq!(eval(target.pz()), 0.0);
        assert_relative_eq!(eval(target.e()), 1.0);
    }

    #[test]
    fn vertex_angles_use_supplied_axes_in_vertex_rest_frame() {
        let mut channel = Channel::new("simple");
        channel.edge("x").p4(p4(0.0, 0.0, 0.0, 2.0));
        channel.edge("a").p4(p4(1.0, 0.0, 0.0, 1.0));
        channel.edge("b").p4(p4(-1.0, 0.0, 0.0, 1.0));
        channel
            .vertex("x_decay")
            .incoming(["x"])
            .outgoing(["a", "b"])
            .validate()
            .unwrap();
        let vertex = channel.get_vertex("x_decay").unwrap();

        assert_relative_eq!(
            eval(vertex.costheta("a", Vec3::z(), Vec3::y()).unwrap()),
            0.0
        );
        assert_relative_eq!(
            eval(vertex.theta("a", Vec3::z(), Vec3::y()).unwrap()),
            std::f64::consts::FRAC_PI_2
        );
        assert_relative_eq!(eval(vertex.phi("a", Vec3::z(), Vec3::y()).unwrap()), 0.0);
    }

    #[test]
    fn vertex_p4_boosts_through_overall_com_and_graph_path() {
        let beta_to_lab = RealVec3::new(0.0, 0.0, 0.6);
        let x_com = RealVec4::new(2.0, 0.5, 0.0, 0.0);
        let recoil_com = RealVec4::new(1.7, -0.5, 0.2, 0.0);
        let x_lab = x_com.boost(&beta_to_lab);
        let recoil_lab = recoil_com.boost(&beta_to_lab);

        let mut channel = Channel::new("KsKs");
        channel.edge("beam").p4(p4(0.0, 0.0, 3.0, 4.0));
        channel.edge("target").p4(p4(0.0, 0.0, 0.0, 1.0));
        channel.edge("x").p4(expr_p4(x_lab));
        channel.edge("recoil").p4(expr_p4(recoil_lab));
        channel.edge("ks1").p4(p4(0.25, 0.0, 0.0, 1.0));
        channel.edge("ks2").p4(p4(0.25, 0.0, 0.0, 1.0));
        channel
            .vertex("production")
            .incoming(["beam", "target"])
            .outgoing(["x", "recoil"])
            .validate()
            .unwrap();
        channel
            .vertex("x_decay")
            .incoming(["x"])
            .outgoing(["ks1", "ks2"])
            .validate()
            .unwrap();

        let expected = recoil_com.boost(&(-x_com.beta().unwrap()));
        assert_vec4_close(
            channel.get_vertex("x_decay").unwrap().p4("recoil").unwrap(),
            expected,
        );
        assert_vec4_close(
            channel.get_vertex("production").unwrap().p4("x").unwrap(),
            x_com,
        );
    }

    #[test]
    fn mandelstam_helpers_choose_available_explicit_formulae() {
        let mut channel = Channel::new("production");
        channel.edge("beam").p4(p4(0.0, 0.0, 3.0, 5.0));
        channel.edge("x").p4(p4(0.0, 0.0, 1.0, 3.0));
        channel.edge("recoil").p4(p4(0.0, 0.0, 2.0, 3.0));
        channel.edge("target");
        channel
            .vertex("production")
            .incoming(["beam", "target"])
            .outgoing(["x", "recoil"])
            .validate()
            .unwrap();
        let vertex = channel.get_vertex("production").unwrap();

        assert_relative_eq!(eval(vertex.s().unwrap()), 27.0);
        assert_relative_eq!(eval(vertex.t().unwrap()), 0.0);
        assert_relative_eq!(eval(vertex.u().unwrap()), 3.0);
    }

    #[test]
    fn mandelstam_helpers_reject_non_two_to_two_vertices() {
        let mut channel = Channel::new("decay");
        channel.edge("x").p4(p4(0.0, 0.0, 0.0, 2.0));
        channel.edge("a").p4(p4(1.0, 0.0, 0.0, 1.0));
        channel.edge("b").p4(p4(-1.0, 0.0, 0.0, 1.0));
        channel
            .vertex("x_decay")
            .incoming(["x"])
            .outgoing(["a", "b"])
            .validate()
            .unwrap();

        assert!(channel.get_vertex("x_decay").unwrap().s().is_err());
    }

    #[test]
    fn p4_inference_reports_cycles_and_ambiguity() {
        let mut cyclic = Channel::new("cyclic");
        cyclic.edge("a");
        cyclic.edge("b");
        cyclic.vertex("ab").incoming(["a"]).outgoing(["b"]);
        cyclic.vertex("ba").incoming(["b"]).outgoing(["a"]);
        assert!(matches!(
            cyclic.p4("a"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation.contains("cyclic p4 inference")
        ));

        let mut ambiguous = Channel::new("ambiguous");
        ambiguous.edge("x");
        ambiguous.edge("a").p4(p4(0.0, 0.0, 1.0, 1.0));
        ambiguous.edge("b").p4(p4(0.0, 0.0, -1.0, 1.0));
        ambiguous.edge("c").p4(p4(1.0, 0.0, 0.0, 1.0));
        ambiguous.edge("d").p4(p4(-1.0, 0.0, 0.0, 1.0));
        ambiguous.vertex("ab").incoming(["x"]).outgoing(["a", "b"]);
        ambiguous.vertex("cd").incoming(["x"]).outgoing(["c", "d"]);
        assert!(matches!(
            ambiguous.p4("x"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation.contains("ambiguous p4 inference")
        ));
    }

    #[test]
    fn p4_resolution_uses_typed_internal_failures_and_preserves_messages() {
        use super::resolution::ResolveFailure;
        use super::topology::TopologyIndex;

        let empty = Channel::new("empty");
        let topology = TopologyIndex::new(&empty);
        assert!(matches!(
            empty.resolve_p4(&topology, "missing", &mut Vec::new()),
            Err(ResolveFailure::UnknownEdge(edge)) if edge == "missing"
        ));
        assert!(matches!(
            empty.p4("missing"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "unknown edge `missing`"
        ));

        let mut unavailable = Channel::new("unavailable");
        unavailable.edge("x");
        let topology = TopologyIndex::new(&unavailable);
        assert!(matches!(
            unavailable.resolve_p4(&topology, "x", &mut Vec::new()),
            Err(ResolveFailure::Unavailable(edge)) if edge == "x"
        ));
        assert!(matches!(
            unavailable.p4("x"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "edge `x` has no p4 and could not be inferred"
        ));

        let mut cyclic = Channel::new("cyclic");
        cyclic.edge("a");
        cyclic.edge("b");
        cyclic.vertex("ab").incoming(["a"]).outgoing(["b"]);
        cyclic.vertex("ba").incoming(["b"]).outgoing(["a"]);
        let topology = TopologyIndex::new(&cyclic);
        assert!(matches!(
            cyclic.resolve_p4(&topology, "a", &mut Vec::new()),
            Err(ResolveFailure::Cycle(edge)) if edge == "a"
        ));
        assert!(matches!(
            cyclic.p4("a"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "cyclic p4 inference involving `a`"
        ));

        let mut ambiguous = Channel::new("ambiguous");
        ambiguous.edge("x");
        for (edge, px) in [("a", 1.0), ("b", -1.0), ("c", 2.0), ("d", -2.0)] {
            ambiguous.edge(edge).p4(p4(px, 0.0, 0.0, 2.0));
        }
        ambiguous.vertex("ab").incoming(["x"]).outgoing(["a", "b"]);
        ambiguous.vertex("cd").incoming(["x"]).outgoing(["c", "d"]);
        let topology = TopologyIndex::new(&ambiguous);
        assert!(matches!(
            ambiguous.resolve_p4(&topology, "x", &mut Vec::new()),
            Err(ResolveFailure::Ambiguous(edge)) if edge == "x"
        ));
        assert!(matches!(
            ambiguous.p4("x"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "ambiguous p4 inference for edge `x`"
        ));
    }

    #[test]
    fn frame_paths_distinguish_no_root_unreachable_and_ambiguous_topologies() {
        let mut no_root = Channel::new("no-root");
        no_root.edge("x");
        no_root.vertex("source").outgoing(["x"]);
        assert!(matches!(
            no_root.frame_path_to_vertex("source"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "could not find a root vertex for `source`"
        ));

        let mut unreachable = Channel::new("unreachable");
        for edge in ["initial", "root_out", "cycle_a", "cycle_b"] {
            unreachable.edge(edge);
        }
        unreachable
            .vertex("root")
            .incoming(["initial"])
            .outgoing(["root_out"]);
        unreachable
            .vertex("cycle_a")
            .incoming(["cycle_b"])
            .outgoing(["cycle_a"]);
        unreachable
            .vertex("cycle_b")
            .incoming(["cycle_a"])
            .outgoing(["cycle_b"]);
        assert!(matches!(
            unreachable.frame_path_to_vertex("cycle_a"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "vertex `cycle_a` is not reachable from a root vertex"
        ));

        let mut multi_root = Channel::new("multi-root");
        for edge in ["a", "b", "left", "right", "out"] {
            multi_root.edge(edge);
        }
        multi_root
            .vertex("left_root")
            .incoming(["a"])
            .outgoing(["left"]);
        multi_root
            .vertex("right_root")
            .incoming(["b"])
            .outgoing(["right"]);
        multi_root
            .vertex("target")
            .incoming(["left", "right"])
            .outgoing(["out"]);
        assert!(matches!(
            multi_root.frame_path_to_vertex("target"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "ambiguous frame path to vertex `target`"
        ));

        let mut diamond = Channel::new("diamond");
        for edge in ["initial", "left", "right", "join_left", "join_right", "out"] {
            diamond.edge(edge);
        }
        diamond
            .vertex("root")
            .incoming(["initial"])
            .outgoing(["left", "right"]);
        diamond
            .vertex("left_branch")
            .incoming(["left"])
            .outgoing(["join_left"]);
        diamond
            .vertex("right_branch")
            .incoming(["right"])
            .outgoing(["join_right"]);
        diamond
            .vertex("target")
            .incoming(["join_left", "join_right"])
            .outgoing(["out"]);
        assert!(matches!(
            diamond.frame_path_to_vertex("target"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "ambiguous frame path to vertex `target`"
        ));
    }

    #[test]
    fn frame_path_records_the_unique_root_and_ordered_edges() {
        let mut channel = Channel::new("chain");
        for edge in ["initial", "middle", "final"] {
            channel.edge(edge);
        }
        channel
            .vertex("root")
            .incoming(["initial"])
            .outgoing(["middle"]);
        channel
            .vertex("target")
            .incoming(["middle"])
            .outgoing(["final"]);

        assert_eq!(
            channel.frame_path_to_vertex("target").unwrap(),
            topology::FramePath {
                root: "root".to_owned(),
                edges: vec!["middle".to_owned()],
            }
        );
        assert!(matches!(
            channel.frame_path_to_vertex("missing"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "unknown vertex `missing`"
        ));
    }

    #[test]
    fn topology_validation_preserves_boundary_error_messages() {
        let channel = Channel::new("empty");
        assert!(matches!(
            channel.get_vertex("missing"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "unknown vertex `missing`"
        ));

        let mut unknown_edge = Channel::new("unknown-edge");
        unknown_edge.vertex("decay").incoming(["missing"]);
        assert!(matches!(
            unknown_edge.validate_vertex("decay"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "unknown edge `missing`"
        ));

        let mut duplicate = Channel::new("duplicate");
        duplicate.edge("x");
        duplicate.vertex("decay").incoming(["x"]).outgoing(["x"]);
        assert!(matches!(
            duplicate.validate_vertex("decay"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "edge `x` appears more than once in vertex `decay`"
        ));

        let mut no_incoming = Channel::new("no-incoming");
        no_incoming.edge("x");
        no_incoming.vertex("source").outgoing(["x"]);
        assert!(matches!(
            no_incoming.vertex_incoming_p4("source"),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation == "vertex `source` has no incoming edges"
        ));
    }

    #[test]
    fn edges_store_particle_properties_for_later_hypothesis_generation() {
        let mut channel = Channel::new("KsKs");
        channel
            .edge("ks1")
            .properties(&ParticleProperties::unknown().with_name("K_S"));

        assert_eq!(
            channel.properties("ks1").unwrap().unwrap().name().unwrap(),
            "K_S"
        );
    }

    #[test]
    fn channel_validation_requires_a_source_for_every_initial_edge() {
        let mut channel = Channel::new("decay");
        channel
            .edge("parent")
            .properties(&ParticleProperties::unknown().with_mass(2.0));
        channel
            .edge("a")
            .properties(&ParticleProperties::unknown().with_mass(0.2));
        channel
            .edge("b")
            .properties(&ParticleProperties::unknown().with_mass(0.4));
        channel
            .vertex("decay")
            .incoming(["parent"])
            .outgoing(["a", "b"]);

        assert!(matches!(
            channel.validate(),
            Err(LadduPhysicsError::InvalidRelation { relation })
                if relation.contains("initial edge `parent` has no momentum source")
        ));
    }

    #[test]
    fn channel_validation_accepts_annotated_initial_edges() {
        let mut channel = Channel::new("production");
        channel
            .edge("beam")
            .properties(&ParticleProperties::unknown().with_mass(0.0))
            .initial_energy_source_direction(ScalarSource::uniform(8.0, 9.0), RealVec3::z());
        channel
            .edge("target")
            .properties(&ParticleProperties::unknown().with_mass(1.0))
            .initial_momentum(RealVec3::default());
        channel
            .edge("x")
            .properties(&ParticleProperties::unknown().with_mass(1.5));
        channel
            .edge("recoil")
            .properties(&ParticleProperties::unknown().with_mass(1.0));
        channel
            .vertex("production")
            .incoming(["beam", "target"])
            .outgoing(["x", "recoil"]);

        channel.validate().unwrap();
    }

    #[test]
    fn channels_round_trip_generation_annotations_through_serde() {
        let mut channel = Channel::new("decay");
        channel
            .edge("parent")
            .properties(&ParticleProperties::unknown().with_mass(2.0))
            .initial_p4(RealVec4::new(2.0, 0.0, 0.0, 0.0));
        channel
            .edge("a")
            .properties(&ParticleProperties::unknown().with_mass(0.2))
            .mass_proposal(0.1..0.3);
        channel
            .edge("b")
            .properties(&ParticleProperties::unknown().with_mass(0.4));
        channel
            .vertex("decay")
            .incoming(["parent"])
            .outgoing(["a", "b"])
            .generation(VertexProposal::TwoBodyDecay);

        let encoded = serde_json::to_string(&channel).unwrap();
        let decoded: Channel = serde_json::from_str(&encoded).unwrap();

        decoded.validate().unwrap();
        assert!(
            decoded
                .require_vertex("decay")
                .unwrap()
                .generation()
                .is_some()
        );
        assert!(
            decoded
                .require_edge("parent")
                .unwrap()
                .initial_momentum()
                .is_some()
        );
        assert!(matches!(
            decoded.require_edge("a").unwrap().mass_proposal(),
            Some(MassProposal::Uniform {
                low: 0.1,
                high: 0.3
            })
        ));
    }
}
