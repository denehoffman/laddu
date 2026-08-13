use std::collections::{HashMap, HashSet, VecDeque};

use super::{Channel, Vertex};
use crate::{LadduPhysicsError, LadduPhysicsResult};
use thiserror::Error;

/// A derived view of the reaction graph used by validation, momentum
/// resolution, and frame lookup.
pub(super) struct TopologyIndex<'a> {
    channel: &'a Channel,
    producers: HashMap<&'a str, Vec<&'a Vertex>>,
    consumers: HashMap<&'a str, Vec<&'a Vertex>>,
    roots: Vec<&'a Vertex>,
    children: HashMap<&'a str, Vec<(&'a str, &'a Vertex)>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FramePath {
    pub(super) root: String,
    pub(super) edges: Vec<String>,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(super) enum FrameFailure {
    #[error("unknown vertex `{0}`")]
    UnknownVertex(String),
    #[error("could not find a root vertex for `{0}`")]
    NoRoot(String),
    #[error("vertex `{0}` is not reachable from a root vertex")]
    Unreachable(String),
    #[error("ambiguous frame path to vertex `{0}`")]
    Ambiguous(String),
}

impl<'a> TopologyIndex<'a> {
    pub(super) fn new(channel: &'a Channel) -> Self {
        let mut producers: HashMap<&str, Vec<&Vertex>> = HashMap::new();
        let mut consumers: HashMap<&str, Vec<&Vertex>> = HashMap::new();

        for vertex in channel.vertices.values() {
            for edge in &vertex.incoming {
                consumers.entry(edge).or_default().push(vertex);
            }
            for edge in &vertex.outgoing {
                producers.entry(edge).or_default().push(vertex);
            }
        }

        let roots = channel
            .vertices
            .values()
            .filter(|vertex| {
                !vertex.incoming.is_empty()
                    && vertex
                        .incoming
                        .iter()
                        .all(|edge| !producers.contains_key(edge.as_str()))
            })
            .collect();

        let mut children = HashMap::new();
        for vertex in channel.vertices.values() {
            let mut links = Vec::new();
            for edge in &vertex.outgoing {
                for child in consumers.get(edge.as_str()).into_iter().flatten() {
                    links.push((edge.as_str(), *child));
                }
            }
            children.insert(vertex.name.as_str(), links);
        }

        Self {
            channel,
            producers,
            consumers,
            roots,
            children,
        }
    }

    pub(super) fn edge(&self, name: &str) -> Option<&'a super::Edge> {
        self.channel.edges.get(name)
    }

    pub(super) fn vertex(&self, name: &str) -> Option<&'a Vertex> {
        self.channel.vertices.get(name)
    }

    pub(super) fn vertices_for_edge<'b>(
        &'b self,
        edge: &'b str,
    ) -> impl Iterator<Item = &'a Vertex> + 'b {
        self.channel.vertices.values().filter(move |vertex| {
            self.producers
                .get(edge)
                .into_iter()
                .chain(self.consumers.get(edge))
                .flatten()
                .any(|connected| connected.name() == vertex.name())
        })
    }

    pub(super) fn validate_vertex(&self, name: &str) -> LadduPhysicsResult<()> {
        let vertex = self.vertex(name).ok_or_else(|| {
            LadduPhysicsError::invalid_relation(format!("unknown vertex `{name}`"))
        })?;
        let mut seen = HashSet::new();
        for edge in vertex.all_edges() {
            if self.edge(edge).is_none() {
                return Err(LadduPhysicsError::invalid_relation(format!(
                    "unknown edge `{edge}`"
                )));
            }
            if !seen.insert(edge) {
                return Err(LadduPhysicsError::invalid_relation(format!(
                    "edge `{edge}` appears more than once in vertex `{name}`"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn frame_path(&self, target: &str) -> Result<FramePath, FrameFailure> {
        if self.vertex(target).is_none() {
            return Err(FrameFailure::UnknownVertex(target.to_owned()));
        }
        if self.roots.is_empty() {
            return Err(FrameFailure::NoRoot(target.to_owned()));
        }

        let mut matches = Vec::new();
        for root in &self.roots {
            self.collect_frame_paths(root.name(), target, &mut matches);
            if matches.len() > 1 {
                return Err(FrameFailure::Ambiguous(target.to_owned()));
            }
        }

        match matches.pop() {
            Some((root, edges)) => Ok(FramePath { root, edges }),
            None => Err(FrameFailure::Unreachable(target.to_owned())),
        }
    }

    fn collect_frame_paths(
        &self,
        root: &str,
        target: &str,
        matches: &mut Vec<(String, Vec<String>)>,
    ) {
        let mut queue = VecDeque::from([(
            root.to_owned(),
            Vec::new(),
            HashSet::from([root.to_owned()]),
        )]);

        while let Some((vertex_name, path, visited)) = queue.pop_front() {
            if vertex_name == target {
                matches.push((root.to_owned(), path));
                continue;
            }
            for (edge, child) in self
                .children
                .get(vertex_name.as_str())
                .into_iter()
                .flatten()
            {
                if visited.contains(child.name()) {
                    continue;
                }
                let mut child_path = path.clone();
                child_path.push((*edge).to_owned());
                let mut child_visited = visited.clone();
                child_visited.insert(child.name().to_owned());
                queue.push_back((child.name().to_owned(), child_path, child_visited));
            }
        }
    }
}
