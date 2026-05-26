use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{Particle, Reaction, Species};
use crate::{vectors::Vec4, LadduError, LadduResult};

/// The event-level source of a channel particle's four-momentum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelP4Source {
    /// Read four-momentum from a dataset column.
    Stored {
        /// Dataset p4 column name.
        p4_name: String,
    },
    /// Use one fixed four-momentum for every event.
    Fixed {
        /// Fixed four-momentum.
        p4: Vec4,
    },
    /// Solve this occurrence from conservation in a compatible topology.
    Missing,
}

/// One named particle occurrence in a [`Channel`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelParticle {
    name: String,
    species: Option<Species>,
    p4_source: Option<ChannelP4Source>,
}

impl ChannelParticle {
    /// Return the occurrence name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the optional physical species associated with this occurrence.
    pub const fn species(&self) -> Option<&Species> {
        self.species.as_ref()
    }

    /// Return the optional event-level p4 annotation.
    pub const fn p4_source(&self) -> Option<&ChannelP4Source> {
        self.p4_source.as_ref()
    }
}

/// A named directed channel vertex.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelVertex {
    name: String,
    incoming: Vec<String>,
    outgoing: Vec<String>,
}

impl ChannelVertex {
    /// Return the vertex name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return ordered incoming occurrence names.
    pub fn incoming(&self) -> &[String] {
        &self.incoming
    }

    /// Return ordered outgoing occurrence names.
    pub fn outgoing(&self) -> &[String] {
        &self.outgoing
    }
}

/// A validating directed graph of named particle occurrences and physical vertices.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Channel {
    particles: Vec<ChannelParticle>,
    particle_indices: HashMap<String, usize>,
    vertices: Vec<ChannelVertex>,
    vertex_indices: HashMap<String, usize>,
}

impl Channel {
    /// Construct an empty channel.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare one named particle occurrence and edit its annotations.
    pub fn particle(&mut self, name: impl Into<String>) -> LadduResult<ChannelParticleEdit<'_>> {
        let name = name.into();
        if self.particle_indices.contains_key(&name) {
            return Err(LadduError::Custom(format!(
                "channel particle '{name}' is already declared"
            )));
        }
        let index = self.particles.len();
        self.particles.push(ChannelParticle {
            name: name.clone(),
            species: None,
            p4_source: None,
        });
        self.particle_indices.insert(name, index);
        Ok(ChannelParticleEdit {
            particle: &mut self.particles[index],
        })
    }

    /// Declare a vertex, validating its references and graph constraints immediately.
    pub fn vertex<const I: usize, const O: usize>(
        &mut self,
        name: impl Into<String>,
        incoming: [&str; I],
        outgoing: [&str; O],
    ) -> LadduResult<&ChannelVertex> {
        let name = name.into();
        if self.vertex_indices.contains_key(&name) {
            return Err(LadduError::Custom(format!(
                "channel vertex '{name}' is already declared"
            )));
        }
        if I == 0 || O == 0 {
            return Err(LadduError::Custom(
                "channel vertices require at least one incoming and outgoing particle".to_string(),
            ));
        }
        let incoming: Vec<String> = incoming.into_iter().map(str::to_string).collect();
        let outgoing: Vec<String> = outgoing.into_iter().map(str::to_string).collect();
        let mut members = HashSet::new();
        for particle in incoming.iter().chain(outgoing.iter()) {
            if !self.particle_indices.contains_key(particle) {
                return Err(LadduError::Custom(format!(
                    "unknown channel particle '{particle}' in vertex '{name}'"
                )));
            }
            if !members.insert(particle) {
                return Err(LadduError::Custom(format!(
                    "channel vertex '{name}' repeats particle '{particle}'"
                )));
            }
        }
        for particle in &outgoing {
            if self
                .vertices
                .iter()
                .any(|vertex| vertex.outgoing.contains(particle))
            {
                return Err(LadduError::Custom(format!(
                    "channel particle '{particle}' already has a producing vertex"
                )));
            }
        }
        for from in &outgoing {
            for to in &incoming {
                if self.has_path(from, to) {
                    return Err(LadduError::Custom(format!(
                        "vertex '{name}' introduces a channel cycle"
                    )));
                }
            }
        }
        let index = self.vertices.len();
        self.vertices.push(ChannelVertex {
            name: name.clone(),
            incoming,
            outgoing,
        });
        self.vertex_indices.insert(name, index);
        Ok(&self.vertices[index])
    }

    /// Return a declared particle occurrence.
    pub fn particle_info(&self, name: &str) -> LadduResult<&ChannelParticle> {
        self.particle_indices
            .get(name)
            .map(|index| &self.particles[*index])
            .ok_or_else(|| LadduError::Custom(format!("unknown channel particle '{name}'")))
    }

    /// Return a declared vertex.
    pub fn vertex_info(&self, name: &str) -> LadduResult<&ChannelVertex> {
        self.vertex_indices
            .get(name)
            .map(|index| &self.vertices[*index])
            .ok_or_else(|| LadduError::Custom(format!("unknown channel vertex '{name}'")))
    }

    /// Lower a two-to-two vertex into the existing reaction representation.
    ///
    /// This allows existing variables and amplitudes to consume a new channel while their
    /// constructors remain based on [`Reaction`].
    pub fn two_to_two_reaction(&self, vertex: &str) -> LadduResult<Reaction> {
        let vertex = self.vertex_info(vertex)?;
        if vertex.incoming.len() != 2 || vertex.outgoing.len() != 2 {
            return Err(LadduError::Custom(format!(
                "vertex '{}' is not a two-to-two production vertex",
                vertex.name
            )));
        }
        let mut stack = HashSet::new();
        let p1 = self.to_legacy_particle(&vertex.incoming[0], &mut stack)?;
        let p2 = self.to_legacy_particle(&vertex.incoming[1], &mut stack)?;
        let p3 = self.to_legacy_particle(&vertex.outgoing[0], &mut stack)?;
        let p4 = self.to_legacy_particle(&vertex.outgoing[1], &mut stack)?;
        Reaction::two_to_two(&p1, &p2, &p3, &p4)
    }

    fn to_legacy_particle(&self, name: &str, stack: &mut HashSet<String>) -> LadduResult<Particle> {
        if !stack.insert(name.to_string()) {
            return Err(LadduError::Custom(format!(
                "cycle encountered while lowering channel particle '{name}'"
            )));
        }
        let info = self.particle_info(name)?;
        let decay_vertices: Vec<_> = self
            .vertices
            .iter()
            .filter(|vertex| {
                vertex.incoming.len() == 1
                    && vertex.incoming[0] == name
                    && vertex.outgoing.len() == 2
            })
            .collect();
        if decay_vertices.len() > 1 {
            stack.remove(name);
            return Err(LadduError::Custom(format!(
                "channel particle '{name}' has multiple two-body decay vertices"
            )));
        }
        let result = match (&info.p4_source, decay_vertices.first()) {
            (Some(ChannelP4Source::Stored { p4_name }), None) => {
                Ok(Particle::stored_as(name, p4_name))
            }
            (Some(ChannelP4Source::Fixed { p4 }), None) => Ok(Particle::fixed(name, *p4)),
            (Some(ChannelP4Source::Missing), None) => Ok(Particle::missing(name)),
            (None, Some(decay)) => {
                let first = self.to_legacy_particle(&decay.outgoing[0], stack)?;
                let second = self.to_legacy_particle(&decay.outgoing[1], stack)?;
                Particle::composite(name, (&first, &second))
            }
            (Some(_), Some(_)) => Err(LadduError::Custom(format!(
                "channel particle '{name}' has both an explicit p4 source and an inferred decay p4"
            ))),
            (None, None) => Err(LadduError::Custom(format!(
                "channel particle '{name}' has no p4 source or two-body decay definition"
            ))),
        };
        stack.remove(name);
        result
    }

    fn has_path(&self, from: &str, to: &str) -> bool {
        let mut pending = vec![from];
        let mut visited = HashSet::new();
        while let Some(current) = pending.pop() {
            if current == to {
                return true;
            }
            if !visited.insert(current) {
                continue;
            }
            for vertex in &self.vertices {
                if vertex.incoming.iter().any(|particle| particle == current) {
                    pending.extend(vertex.outgoing.iter().map(String::as_str));
                }
            }
        }
        false
    }
}

/// Mutable annotation builder returned by [`Channel::particle`].
pub struct ChannelParticleEdit<'a> {
    particle: &'a mut ChannelParticle,
}

impl ChannelParticleEdit<'_> {
    /// Associate this occurrence with a reusable physical species.
    pub fn species(&mut self, species: impl Into<Species>) -> LadduResult<&mut Self> {
        if self.particle.species.is_some() {
            return Err(LadduError::Custom(format!(
                "channel particle '{}' already has a species",
                self.particle.name
            )));
        }
        self.particle.species = Some(species.into());
        Ok(self)
    }

    /// Read four-momentum from the dataset column with this particle's name.
    pub fn stored(&mut self) -> LadduResult<&mut Self> {
        let p4_name = self.particle.name.clone();
        self.set_p4_source(ChannelP4Source::Stored { p4_name })
    }

    /// Read four-momentum from a specified dataset column.
    pub fn stored_as(&mut self, p4_name: impl Into<String>) -> LadduResult<&mut Self> {
        self.set_p4_source(ChannelP4Source::Stored {
            p4_name: p4_name.into(),
        })
    }

    /// Use a fixed four-momentum.
    pub fn fixed(&mut self, p4: Vec4) -> LadduResult<&mut Self> {
        self.set_p4_source(ChannelP4Source::Fixed { p4 })
    }

    /// Solve this particle from four-momentum conservation.
    pub fn missing(&mut self) -> LadduResult<&mut Self> {
        self.set_p4_source(ChannelP4Source::Missing)
    }

    fn set_p4_source(&mut self, source: ChannelP4Source) -> LadduResult<&mut Self> {
        if self.particle.p4_source.is_some() {
            return Err(LadduError::Custom(format!(
                "channel particle '{}' already has a p4 source",
                self.particle.name
            )));
        }
        self.particle.p4_source = Some(source);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quantum::Frame, vectors::Vec4};

    fn ksks_channel() -> Channel {
        let mut channel = Channel::new();
        channel.particle("beam").unwrap().stored().unwrap();
        channel.particle("target").unwrap().missing().unwrap();
        channel.particle("kk").unwrap();
        channel.particle("kshort1").unwrap().stored().unwrap();
        channel.particle("kshort2").unwrap().stored().unwrap();
        channel.particle("recoil").unwrap().stored().unwrap();
        channel
            .vertex("ksks", ["kk"], ["kshort1", "kshort2"])
            .unwrap();
        channel
            .vertex("production", ["beam", "target"], ["kk", "recoil"])
            .unwrap();
        channel
    }

    #[test]
    fn lowers_into_existing_reaction_variables() {
        let reaction = ksks_channel().two_to_two_reaction("production").unwrap();
        let decay = reaction.decay("kk").unwrap();
        let _angles = decay.angles("kshort1", Frame::Helicity).unwrap();
        let _polarization = reaction.polarization("pol_mag", "pol_angle");
    }

    #[test]
    fn rejects_invalid_graph_or_p4_annotations_immediately() {
        let mut channel = Channel::new();
        channel.particle("a").unwrap().stored().unwrap();
        assert!(channel.particle("a").is_err());
        assert!(channel
            .particle("b")
            .unwrap()
            .fixed(Vec4::new(0.0, 0.0, 0.0, 0.0))
            .is_ok());
        assert!(channel
            .particle("c")
            .unwrap()
            .missing()
            .unwrap()
            .stored()
            .is_err());
        assert!(channel.vertex("unknown", ["a"], ["missing"]).is_err());
        channel.vertex("ab", ["a"], ["b"]).unwrap();
        assert!(channel.vertex("cycle", ["b"], ["a"]).is_err());
    }

    #[test]
    fn stored_as_preserves_occurrence_name_when_lowered() {
        let mut channel = Channel::new();
        channel
            .particle("initial")
            .unwrap()
            .stored_as("input_p4")
            .unwrap();
        channel.particle("fixed").unwrap().missing().unwrap();
        channel.particle("out_a").unwrap().stored().unwrap();
        channel.particle("out_b").unwrap().stored().unwrap();
        channel
            .vertex("production", ["initial", "fixed"], ["out_a", "out_b"])
            .unwrap();
        let reaction = channel.two_to_two_reaction("production").unwrap();
        assert_eq!(reaction.role("p1").unwrap().label(), "initial");
    }

    #[test]
    fn species_are_reusable_across_particle_occurrences() {
        let proton = Species::new("proton");
        let mut channel = Channel::new();
        channel
            .particle("target")
            .unwrap()
            .species(&proton)
            .unwrap();
        channel
            .particle("recoil")
            .unwrap()
            .species(&proton)
            .unwrap();
        assert_eq!(
            channel
                .particle_info("target")
                .unwrap()
                .species()
                .unwrap()
                .name(),
            "proton"
        );
    }
}
