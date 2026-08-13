use super::{Channel, InferencePriority, Vertex, topology::TopologyIndex};
use crate::{LadduPhysicsError, LadduPhysicsResult, vectors::Vec4};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(super) enum ResolveFailure {
    #[error("unknown edge `{0}`")]
    UnknownEdge(String),
    #[error("edge `{0}` has no p4 and could not be inferred")]
    Unavailable(String),
    #[error("cyclic p4 inference involving `{0}`")]
    Cycle(String),
    #[error("ambiguous p4 inference for edge `{0}`")]
    Ambiguous(String),
}

impl ResolveFailure {
    pub(super) fn into_public(self) -> LadduPhysicsError {
        LadduPhysicsError::invalid_relation(self.to_string())
    }
}

impl Channel {
    pub(super) fn resolved_p4(&self, edge: &str) -> LadduPhysicsResult<Vec4> {
        let topology = TopologyIndex::new(self);
        self.resolve_p4(&topology, edge, &mut Vec::new())
            .map_err(ResolveFailure::into_public)
    }

    pub(super) fn resolve_p4(
        &self,
        topology: &TopologyIndex<'_>,
        edge: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec4, ResolveFailure> {
        let edge_def = topology
            .edge(edge)
            .ok_or_else(|| ResolveFailure::UnknownEdge(edge.to_owned()))?;
        if let Some(p4) = &edge_def.p4 {
            return Ok(p4.clone());
        }
        if stack.iter().any(|candidate| candidate == edge) {
            return Err(ResolveFailure::Cycle(edge.to_owned()));
        }
        stack.push(edge.to_owned());

        let result = self.resolve_inference_candidates(topology, edge, stack);
        stack.pop();
        result
    }

    fn resolve_inference_candidates(
        &self,
        topology: &TopologyIndex<'_>,
        edge: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec4, ResolveFailure> {
        for priority in [
            InferencePriority::ParentFromDaughters,
            InferencePriority::ChildFromParents,
            InferencePriority::AnySingleMissing,
        ] {
            let candidates = self.inference_candidates(topology, edge, priority, stack)?;
            match candidates.as_slice() {
                [candidate] => return Ok(candidate.clone()),
                [_, _, ..] => return Err(ResolveFailure::Ambiguous(edge.to_owned())),
                [] => {}
            }
        }
        Err(ResolveFailure::Unavailable(edge.to_owned()))
    }

    fn inference_candidates(
        &self,
        topology: &TopologyIndex<'_>,
        edge: &str,
        priority: InferencePriority,
        stack: &mut Vec<String>,
    ) -> Result<Vec<Vec4>, ResolveFailure> {
        let mut out = Vec::new();
        for vertex in topology.vertices_for_edge(edge) {
            if !vertex.matches_priority(edge, priority) {
                continue;
            }
            if let Some(p4) = self.infer_from_vertex(topology, edge, vertex, stack)? {
                out.push(p4);
            }
        }
        Ok(out)
    }

    fn infer_from_vertex(
        &self,
        topology: &TopologyIndex<'_>,
        edge: &str,
        vertex: &Vertex,
        stack: &mut Vec<String>,
    ) -> Result<Option<Vec4>, ResolveFailure> {
        let incoming = match self.sum_known_except(topology, &vertex.incoming, edge, stack) {
            Ok(incoming) => incoming,
            Err(ResolveFailure::Unavailable(_)) => return Ok(None),
            Err(err) => return Err(err),
        };
        let outgoing = match self.sum_known_except(topology, &vertex.outgoing, edge, stack) {
            Ok(outgoing) => outgoing,
            Err(ResolveFailure::Unavailable(_)) => return Ok(None),
            Err(err) => return Err(err),
        };
        if vertex.incoming.iter().any(|candidate| candidate == edge) {
            Ok(Some(outgoing - incoming))
        } else {
            Ok(Some(incoming - outgoing))
        }
    }

    pub(super) fn sum_known_except(
        &self,
        topology: &TopologyIndex<'_>,
        edges: &[String],
        except: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec4, ResolveFailure> {
        let mut sum = Vec4::new(0.0, 0.0, 0.0, 0.0);
        for edge in edges {
            if edge != except {
                sum = sum + self.resolve_p4(topology, edge, stack)?;
            }
        }
        Ok(sum)
    }
}
