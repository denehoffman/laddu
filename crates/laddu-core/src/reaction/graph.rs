use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::Particle;
use crate::{LadduError, LadduResult};

/// Owned particle definitions and decay edges for a reaction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParticleGraph {
    roots: Vec<Particle>,
}

impl ParticleGraph {
    /// Construct a particle graph from root particles.
    pub fn new(roots: impl IntoIterator<Item = Particle>) -> LadduResult<Self> {
        let roots = roots.into_iter().collect::<Vec<_>>();
        validate_unique_particles(roots.iter())?;
        Ok(Self { roots })
    }

    /// Return the graph root particles.
    pub fn roots(&self) -> &[Particle] {
        &self.roots
    }

    /// Return the particle with the given identifier.
    pub fn particle(&self, particle: &str) -> LadduResult<&Particle> {
        self.find_particle(particle)
            .ok_or_else(|| LadduError::Custom(format!("unknown reaction particle '{particle}'")))
    }

    /// Return whether the graph contains a particle with the given identifier.
    pub fn contains(&self, particle: &str) -> bool {
        self.find_particle(particle).is_some()
    }

    /// Return all particle definitions in graph traversal order.
    pub fn particles(&self) -> Vec<&Particle> {
        let mut particles = Vec::new();
        for root in &self.roots {
            collect_particles(root, &mut particles);
        }
        particles
    }

    /// Return the graph root containing the particle identifier.
    pub fn root_containing(&self, particle: &str) -> Option<&Particle> {
        self.roots.iter().find(|root| root.contains_id(particle))
    }

    /// Return the parent particle identifier for an immediate or nested child.
    pub fn parent_of(&self, child: &str) -> Option<&Particle> {
        self.roots.iter().find_map(|root| root.parent_of_id(child))
    }

    fn find_particle(&self, particle: &str) -> Option<&Particle> {
        self.roots
            .iter()
            .find_map(|root| find_particle(root, particle))
    }
}

fn validate_unique_particles<'a>(roots: impl IntoIterator<Item = &'a Particle>) -> LadduResult<()> {
    let mut seen = HashSet::new();
    for root in roots {
        validate_unique_particle(root, &mut seen)?;
    }
    Ok(())
}

fn validate_unique_particle<'a>(
    particle: &'a Particle,
    seen: &mut HashSet<&'a str>,
) -> LadduResult<()> {
    if !seen.insert(particle.label()) {
        return Err(LadduError::Custom(format!(
            "duplicate reaction particle identifier '{}'",
            particle.label()
        )));
    }
    for daughter in particle.daughters() {
        validate_unique_particle(daughter, seen)?;
    }
    Ok(())
}

fn collect_particles<'a>(particle: &'a Particle, particles: &mut Vec<&'a Particle>) {
    particles.push(particle);
    for daughter in particle.daughters() {
        collect_particles(daughter, particles);
    }
}

fn find_particle<'a>(particle: &'a Particle, id: &str) -> Option<&'a Particle> {
    if particle.label() == id {
        return Some(particle);
    }
    particle
        .daughters()
        .iter()
        .find_map(|daughter| find_particle(daughter, id))
}
