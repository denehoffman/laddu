use std::collections::{HashSet, VecDeque};

use indexmap::IndexMap;
use laddu_expr::Expr;
use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    generation::{InitialMomentum, MassProposal, ScalarSource, VertexProposal},
    quantum::{MandelstamChannel, ParticleProperties},
    vectors::{RealVec3, RealVec4, Vec3, Vec4},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Channel {
    name: String,
    edges: IndexMap<String, Edge>,
    vertices: IndexMap<String, Vertex>,
}

impl Channel {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            edges: IndexMap::new(),
            vertices: IndexMap::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn edge(&mut self, name: impl Into<String>) -> EdgeHandle<'_> {
        let name = name.into();
        self.edges
            .entry(name.clone())
            .or_insert_with(|| Edge::new(name.clone()));
        EdgeHandle {
            edge: self.edges.get_mut(&name).expect("edge was just inserted"),
        }
    }

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

    pub fn get_vertex(&self, name: &str) -> LadduPhysicsResult<VertexView<'_>> {
        self.require_vertex(name)?;
        Ok(VertexView {
            channel: self,
            name: name.to_owned(),
        })
    }

    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

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
    pub fn validate(&self) -> LadduPhysicsResult<()> {
        for name in self.vertices.keys() {
            self.validate_vertex(name)?;
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

    pub fn properties(&self, edge: &str) -> LadduPhysicsResult<Option<&ParticleProperties>> {
        Ok(self.require_edge(edge)?.properties.as_ref())
    }

    /// Return the particle definition attached to an edge.
    pub fn particle(&self, edge: &str) -> LadduPhysicsResult<&ParticleProperties> {
        self.properties(edge)?.ok_or_else(|| {
            LadduPhysicsError::invalid_relation(format!("edge `{edge}` has no particle properties"))
        })
    }

    pub fn p4(&self, edge: &str) -> LadduPhysicsResult<Vec4> {
        self.resolve_p4(edge, &mut Vec::new())
    }

    pub fn vec3(&self, edge: &str) -> LadduPhysicsResult<Vec3> {
        Ok(self.p4(edge)?.vec3())
    }

    pub fn mass(&self, edge: &str) -> LadduPhysicsResult<Expr> {
        Ok(self.p4(edge)?.m())
    }

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

    fn resolve_p4(&self, edge: &str, stack: &mut Vec<String>) -> LadduPhysicsResult<Vec4> {
        let edge_def = self.require_edge(edge)?;
        if let Some(p4) = &edge_def.p4 {
            return Ok(p4.clone());
        }
        if stack.iter().any(|candidate| candidate == edge) {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "cyclic p4 inference involving `{edge}`"
            )));
        }
        stack.push(edge.to_owned());

        for priority in [
            InferencePriority::ParentFromDaughters,
            InferencePriority::ChildFromParents,
            InferencePriority::AnySingleMissing,
        ] {
            let candidates = self.inference_candidates(edge, priority, stack)?;
            if candidates.len() == 1 {
                stack.pop();
                return Ok(candidates[0].clone());
            }
            if candidates.len() > 1 {
                stack.pop();
                return Err(LadduPhysicsError::invalid_relation(format!(
                    "ambiguous p4 inference for edge `{edge}`"
                )));
            }
        }

        stack.pop();
        Err(LadduPhysicsError::invalid_relation(format!(
            "edge `{edge}` has no p4 and could not be inferred"
        )))
    }

    fn inference_candidates(
        &self,
        edge: &str,
        priority: InferencePriority,
        stack: &mut Vec<String>,
    ) -> LadduPhysicsResult<Vec<Vec4>> {
        let mut out = Vec::new();
        for vertex in self.vertices.values() {
            if !vertex.contains(edge) || !vertex.matches_priority(edge, priority) {
                continue;
            }
            if let Some(p4) = self.infer_from_vertex(edge, vertex, stack)? {
                out.push(p4);
            }
        }
        Ok(out)
    }

    fn infer_from_vertex(
        &self,
        edge: &str,
        vertex: &Vertex,
        stack: &mut Vec<String>,
    ) -> LadduPhysicsResult<Option<Vec4>> {
        if !vertex.contains(edge) {
            return Ok(None);
        }

        let incoming = match self.sum_known_except(&vertex.incoming, edge, stack) {
            Ok(incoming) => incoming,
            Err(err) if is_unresolved_inference(&err) => return Ok(None),
            Err(err) => return Err(err),
        };
        let outgoing = match self.sum_known_except(&vertex.outgoing, edge, stack) {
            Ok(outgoing) => outgoing,
            Err(err) if is_unresolved_inference(&err) => return Ok(None),
            Err(err) => return Err(err),
        };
        if vertex.incoming.iter().any(|candidate| candidate == edge) {
            Ok(Some(outgoing - incoming))
        } else {
            Ok(Some(incoming - outgoing))
        }
    }

    fn sum_known_except(
        &self,
        edges: &[String],
        except: &str,
        stack: &mut Vec<String>,
    ) -> LadduPhysicsResult<Vec4> {
        let mut sum = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for edge in edges {
            if edge == except {
                continue;
            }
            sum = sum + self.resolve_p4(edge, stack)?;
        }
        Ok(sum)
    }

    fn frame_path_to_vertex(&self, target: &str) -> LadduPhysicsResult<FramePath> {
        self.require_vertex(target)?;
        let roots = self.root_vertices();
        if roots.is_empty() {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "could not find a root vertex for `{target}`"
            )));
        }

        let mut matches = Vec::new();
        for root in roots {
            if let Some(path) = self.frame_path_from_root(&root, target) {
                matches.push(FramePath { root, edges: path });
            }
        }

        match matches.len() {
            0 => Err(LadduPhysicsError::invalid_relation(format!(
                "vertex `{target}` is not reachable from a root vertex"
            ))),
            1 => Ok(matches.remove(0)),
            _ => Err(LadduPhysicsError::invalid_relation(format!(
                "ambiguous frame path to vertex `{target}`"
            ))),
        }
    }

    fn root_vertices(&self) -> Vec<String> {
        let produced_edges = self
            .vertices
            .values()
            .flat_map(|vertex| vertex.outgoing.iter())
            .collect::<HashSet<_>>();

        self.vertices
            .values()
            .filter(|vertex| {
                !vertex.incoming.is_empty()
                    && vertex
                        .incoming
                        .iter()
                        .all(|edge| !produced_edges.contains(edge))
            })
            .map(|vertex| vertex.name.clone())
            .collect()
    }

    fn frame_path_from_root(&self, root: &str, target: &str) -> Option<Vec<String>> {
        let mut queue = VecDeque::from([(root.to_owned(), Vec::new())]);
        let mut seen = HashSet::new();
        let mut matches = Vec::new();

        while let Some((vertex_name, path)) = queue.pop_front() {
            if !seen.insert(vertex_name.clone()) {
                continue;
            }
            if vertex_name == target {
                matches.push(path);
                continue;
            }
            let vertex = self.vertices.get(&vertex_name)?;
            for edge in &vertex.outgoing {
                for child in self
                    .vertices
                    .values()
                    .filter(|candidate| candidate.incoming.iter().any(|incoming| incoming == edge))
                {
                    let mut child_path = path.clone();
                    child_path.push(edge.clone());
                    queue.push_back((child.name.clone(), child_path));
                }
            }
        }

        if matches.len() == 1 {
            matches.pop()
        } else {
            None
        }
    }

    fn vertex_incoming_p4(&self, vertex: &str) -> LadduPhysicsResult<Vec4> {
        let vertex = self.require_vertex(vertex)?;
        if vertex.incoming.is_empty() {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "vertex `{}` has no incoming edges",
                vertex.name()
            )));
        }
        self.sum_known_except(&vertex.incoming, "", &mut Vec::new())
    }
}

#[derive(Clone, Debug)]
struct FramePath {
    root: String,
    edges: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
            output: false,
            mass_proposal: None,
            initial_momentum: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn has_explicit_p4(&self) -> bool {
        self.p4.is_some()
    }

    pub fn properties(&self) -> Option<&ParticleProperties> {
        self.properties.as_ref()
    }

    pub fn is_output(&self) -> bool {
        self.output
    }

    pub fn mass_proposal(&self) -> Option<&MassProposal> {
        self.mass_proposal.as_ref()
    }

    pub fn initial_momentum(&self) -> Option<&InitialMomentum> {
        self.initial_momentum.as_ref()
    }
}

pub struct EdgeHandle<'a> {
    edge: &'a mut Edge,
}

impl EdgeHandle<'_> {
    pub fn p4(&mut self, p4: impl Into<Vec4>) -> &mut Self {
        self.edge.p4 = Some(p4.into());
        self
    }

    pub fn properties(&mut self, properties: &ParticleProperties) -> &mut Self {
        self.edge.properties = Some(properties.clone());
        self
    }

    pub fn output(&mut self) -> &mut Self {
        self.edge.output = true;
        self
    }

    pub fn generated_only(&mut self) -> &mut Self {
        self.edge.output = false;
        self
    }

    pub fn mass_proposal(&mut self, proposal: impl Into<MassProposal>) -> &mut Self {
        self.edge.mass_proposal = Some(proposal.into());
        self
    }

    pub fn initial(&mut self, source: InitialMomentum) -> &mut Self {
        self.edge.initial_momentum = Some(source);
        self
    }

    pub fn initial_p4(&mut self, p4: RealVec4) -> &mut Self {
        self.initial(InitialMomentum::p4(p4))
    }

    pub fn initial_momentum(&mut self, momentum: RealVec3) -> &mut Self {
        self.initial(InitialMomentum::momentum(momentum))
    }

    pub fn initial_energy_direction(&mut self, energy: f64, direction: RealVec3) -> &mut Self {
        self.initial(InitialMomentum::energy_direction(energy, direction))
    }

    pub fn initial_energy_source_direction(
        &mut self,
        energy: ScalarSource,
        direction: RealVec3,
    ) -> &mut Self {
        self.initial(InitialMomentum::energy_source_direction(energy, direction))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn incoming(&self) -> &[String] {
        &self.incoming
    }

    pub fn outgoing(&self) -> &[String] {
        &self.outgoing
    }

    pub fn generation(&self) -> Option<&VertexProposal> {
        self.generation.as_ref()
    }

    fn contains(&self, edge: &str) -> bool {
        self.incoming.iter().any(|candidate| candidate == edge)
            || self.outgoing.iter().any(|candidate| candidate == edge)
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

pub struct VertexHandle<'a> {
    channel: &'a mut Channel,
    name: String,
}

impl VertexHandle<'_> {
    pub fn generation(&mut self, proposal: impl Into<VertexProposal>) -> &mut Self {
        self.channel
            .vertices
            .get_mut(&self.name)
            .expect("vertex handle references an existing vertex")
            .generation = Some(proposal.into());
        self
    }

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
pub struct VertexView<'a> {
    channel: &'a Channel,
    name: String,
}

impl<'a> VertexView<'a> {
    pub fn vertex(&self) -> &'a Vertex {
        self.channel
            .vertices
            .get(&self.name)
            .expect("vertex view references an existing vertex")
    }

    pub fn p4(&self, edge: &str) -> LadduPhysicsResult<Vec4> {
        let frame_path = self.channel.frame_path_to_vertex(&self.name)?;
        let overall = self.channel.vertex_incoming_p4(&frame_path.root)?;
        let mut boosts = vec![-&overall.beta()];
        let mut p4 = self.channel.p4(edge)?;
        for beta in &boosts {
            p4 = p4.boost(beta);
        }

        for frame_edge in frame_path.edges {
            let mut frame_p4 = self.channel.p4(&frame_edge)?;
            for beta in &boosts {
                frame_p4 = frame_p4.boost(beta);
            }
            let beta = -&frame_p4.beta();
            p4 = p4.boost(&beta);
            boosts.push(beta);
        }

        Ok(p4)
    }

    pub fn vec3(&self, edge: &str) -> LadduPhysicsResult<Vec3> {
        Ok(self.p4(edge)?.vec3())
    }

    pub fn costheta(&self, edge: &str, z_axis: Vec3, _y_hint: Vec3) -> LadduPhysicsResult<Expr> {
        let p = self.vec3(edge)?;
        let z = z_axis.unit();
        Ok(p.dot(&z) / p.mag())
    }

    pub fn theta(&self, edge: &str, z_axis: Vec3, y_hint: Vec3) -> LadduPhysicsResult<Expr> {
        Ok(self.costheta(edge, z_axis, y_hint)?.acos())
    }

    pub fn phi(&self, edge: &str, z_axis: Vec3, y_hint: Vec3) -> LadduPhysicsResult<Expr> {
        let p = self.vec3(edge)?;
        let z = z_axis.unit();
        let z_component = &z * &y_hint.dot(&z);
        let y = (y_hint - z_component).unit();
        let x = y.cross(&z);
        Ok(laddu_expr::atan2(p.dot(&y), p.dot(&x)))
    }

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

    pub fn s(&self) -> LadduPhysicsResult<Expr> {
        self.mandelstam(MandelstamChannel::S)
    }

    pub fn t(&self) -> LadduPhysicsResult<Expr> {
        self.mandelstam(MandelstamChannel::T)
    }

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

fn is_unresolved_inference(err: &LadduPhysicsError) -> bool {
    matches!(err, LadduPhysicsError::InvalidRelation { relation } if relation.contains("could not be inferred"))
}

impl Channel {
    fn validate_vertex(&self, name: &str) -> LadduPhysicsResult<()> {
        let vertex = self.require_vertex(name)?;
        let mut seen = HashSet::new();
        for edge in vertex.all_edges() {
            self.require_edge(edge)?;
            if !seen.insert(edge) {
                return Err(LadduPhysicsError::invalid_relation(format!(
                    "edge `{edge}` appears more than once in vertex `{name}`"
                )));
            }
        }
        Ok(())
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
        RealVec4::new(px, py, pz, e).into()
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
        let x_com = RealVec4::new(0.5, 0.0, 0.0, 2.0);
        let recoil_com = RealVec4::new(-0.5, 0.2, 0.0, 1.7);
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
            .initial_p4(RealVec4::new(0.0, 0.0, 0.0, 2.0));
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
