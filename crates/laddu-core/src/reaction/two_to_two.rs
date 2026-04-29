use serde::{Deserialize, Serialize};

use super::{graph::ParticleGraph, particle::resolve_particle_direct, Particle};
use crate::{
    data::NamedEventView,
    vectors::{Vec3, Vec4},
    LadduError, LadduResult,
};

/// A direct two-to-two reaction preserving `p1 + p2 -> p3 + p4` semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoToTwoReaction {
    p1: String,
    p2: String,
    p3: String,
    p4: String,
    missing_index: Option<usize>,
}

impl TwoToTwoReaction {
    /// Construct a two-to-two reaction.
    pub fn new(p1: &Particle, p2: &Particle, p3: &Particle, p4: &Particle) -> LadduResult<Self> {
        let particles = [p1, p2, p3, p4];
        let missing = particles
            .iter()
            .enumerate()
            .filter_map(|(index, particle)| particle.is_missing().then_some(index))
            .collect::<Vec<_>>();
        if missing.len() > 1 {
            return Err(LadduError::Custom(
                "two-to-two reaction can contain at most one missing particle".to_string(),
            ));
        }
        Ok(Self {
            p1: p1.label().to_string(),
            p2: p2.label().to_string(),
            p3: p3.label().to_string(),
            p4: p4.label().to_string(),
            missing_index: missing.first().copied(),
        })
    }

    /// Return the `p1` particle identifier.
    pub fn p1(&self) -> &str {
        &self.p1
    }

    /// Return the `p2` particle identifier.
    pub fn p2(&self) -> &str {
        &self.p2
    }

    /// Return the `p3` particle identifier.
    pub fn p3(&self) -> &str {
        &self.p3
    }

    /// Return the `p4` particle identifier.
    pub fn p4(&self) -> &str {
        &self.p4
    }

    /// Return the zero-based missing particle index, if any.
    pub const fn missing_index(&self) -> Option<usize> {
        self.missing_index
    }

    /// Resolve all four reaction momenta for one event.
    pub fn resolve(
        &self,
        graph: &ParticleGraph,
        event: &NamedEventView<'_>,
    ) -> LadduResult<ResolvedTwoToTwo> {
        let mut momenta = [
            resolve_particle_direct(event, graph.particle(&self.p1)?)?,
            resolve_particle_direct(event, graph.particle(&self.p2)?)?,
            resolve_particle_direct(event, graph.particle(&self.p3)?)?,
            resolve_particle_direct(event, graph.particle(&self.p4)?)?,
        ];
        if let Some(index) = self.missing_index {
            let missing = match index {
                0 => momenta[2].unwrap() + momenta[3].unwrap() - momenta[1].unwrap(),
                1 => momenta[2].unwrap() + momenta[3].unwrap() - momenta[0].unwrap(),
                2 => momenta[0].unwrap() + momenta[1].unwrap() - momenta[3].unwrap(),
                3 => momenta[0].unwrap() + momenta[1].unwrap() - momenta[2].unwrap(),
                _ => unreachable!("validated two-to-two slot index"),
            };
            momenta[index] = Some(missing);
        }
        Ok(ResolvedTwoToTwo {
            p1: momenta[0].unwrap(),
            p2: momenta[1].unwrap(),
            p3: momenta[2].unwrap(),
            p4: momenta[3].unwrap(),
        })
    }

    pub(super) fn missing_particle(&self) -> Option<&str> {
        self.missing_index.map(|index| self.slot_unchecked(index))
    }

    fn slot_unchecked(&self, index: usize) -> &str {
        match index {
            0 => &self.p1,
            1 => &self.p2,
            2 => &self.p3,
            3 => &self.p4,
            _ => unreachable!("validated two-to-two slot index"),
        }
    }

    /// Return the particle identifier assigned to a named role.
    pub fn role(&self, role: &str) -> LadduResult<&str> {
        match role {
            "p1" => Ok(&self.p1),
            "p2" => Ok(&self.p2),
            "p3" => Ok(&self.p3),
            "p4" => Ok(&self.p4),
            _ => Err(LadduError::Custom(format!(
                "unknown two-to-two reaction role '{role}'"
            ))),
        }
    }
}

/// Resolved event-level momenta for a two-to-two reaction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedTwoToTwo {
    pub(super) p1: Vec4,
    pub(super) p2: Vec4,
    pub(super) p3: Vec4,
    pub(super) p4: Vec4,
}

impl ResolvedTwoToTwo {
    /// Return `p1`.
    pub const fn p1(self) -> Vec4 {
        self.p1
    }

    /// Return `p2`.
    pub const fn p2(self) -> Vec4 {
        self.p2
    }

    /// Return `p3`.
    pub const fn p3(self) -> Vec4 {
        self.p3
    }

    /// Return `p4`.
    pub const fn p4(self) -> Vec4 {
        self.p4
    }

    /// Return the production center-of-momentum boost.
    pub fn com_boost(self) -> Vec3 {
        -(self.p1 + self.p2).beta()
    }

    /// Return the Mandelstam `s` invariant.
    pub fn s(self) -> f64 {
        (self.p1 + self.p2).m2()
    }

    /// Return the Mandelstam `t` invariant.
    pub fn t(self) -> f64 {
        (self.p1 - self.p3).m2()
    }

    /// Return the Mandelstam `u` invariant.
    pub fn u(self) -> f64 {
        (self.p1 - self.p4).m2()
    }
}
