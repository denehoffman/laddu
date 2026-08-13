use super::{
    Channel,
    topology::{FramePath, TopologyIndex},
};
use crate::{LadduPhysicsError, LadduPhysicsResult, vectors::Vec4};

impl Channel {
    pub(super) fn p4_in_frame(&self, vertex: &str, edge: &str) -> LadduPhysicsResult<Vec4> {
        let frame_path = self.frame_path_to_vertex(vertex)?;
        let overall = self.vertex_incoming_p4(&frame_path.root)?;
        let mut boosts = vec![-&overall.beta()];
        let mut p4 = self.p4(edge)?;
        for beta in &boosts {
            p4 = p4.boost(beta);
        }

        for frame_edge in frame_path.edges {
            let mut frame_p4 = self.p4(&frame_edge)?;
            for beta in &boosts {
                frame_p4 = frame_p4.boost(beta);
            }
            let beta = -&frame_p4.beta();
            p4 = p4.boost(&beta);
            boosts.push(beta);
        }

        Ok(p4)
    }

    pub(super) fn frame_path_to_vertex(&self, target: &str) -> LadduPhysicsResult<FramePath> {
        let topology = TopologyIndex::new(self);
        topology
            .frame_path(target)
            .map_err(|failure| LadduPhysicsError::invalid_relation(failure.to_string()))
    }

    pub(super) fn vertex_incoming_p4(&self, vertex: &str) -> LadduPhysicsResult<Vec4> {
        let vertex = self.require_vertex(vertex)?;
        if vertex.incoming.is_empty() {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "vertex `{}` has no incoming edges",
                vertex.name()
            )));
        }
        let topology = TopologyIndex::new(self);
        self.sum_known_except(&topology, &vertex.incoming, "", &mut Vec::new())
            .map_err(super::resolution::ResolveFailure::into_public)
    }
}
