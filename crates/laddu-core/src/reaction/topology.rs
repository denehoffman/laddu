use serde::{Deserialize, Serialize};

use super::{
    decay::Decay,
    particle::{resolve_particle_direct, Particle},
    two_to_two::{ResolvedTwoToTwo, TwoToTwoReaction},
};
use crate::{
    data::NamedEventView,
    kinematics::{DecayAngles, FrameAxes, RestFrame},
    quantum::{Channel, Frame},
    variables::{Mandelstam, Mass, PolAngle, Polarization},
    vectors::Vec4,
    LadduError, LadduResult,
};

/// A reaction topology.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReactionTopology {
    /// A two-to-two reaction with `p1 + p2 -> p3 + p4` semantics.
    TwoToTwo(TwoToTwoReaction),
}

/// A reaction with direct particle definitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reaction {
    topology: ReactionTopology,
}

impl Reaction {
    /// Construct a two-to-two reaction.
    pub fn two_to_two(
        p1: &Particle,
        p2: &Particle,
        p3: &Particle,
        p4: &Particle,
    ) -> LadduResult<Self> {
        Ok(Self {
            topology: ReactionTopology::TwoToTwo(TwoToTwoReaction::new(p1, p2, p3, p4)?),
        })
    }

    /// Return the reaction topology.
    pub const fn topology(&self) -> &ReactionTopology {
        &self.topology
    }

    /// Resolve a particle p4 from an event.
    pub fn p4(&self, event: &NamedEventView<'_>, particle: &Particle) -> LadduResult<Vec4> {
        match &self.topology {
            ReactionTopology::TwoToTwo(topology) => {
                if let Some(index) = topology.missing_index() {
                    if topology.particle_at(index) == particle {
                        return Ok(match index {
                            0 => topology.resolve(event)?.p1(),
                            1 => topology.resolve(event)?.p2(),
                            2 => topology.resolve(event)?.p3(),
                            3 => topology.resolve(event)?.p4(),
                            _ => unreachable!("validated two-to-two slot index"),
                        });
                    }
                }
                resolve_particle_direct(event, particle)?.ok_or_else(|| {
                    LadduError::Custom(format!(
                        "missing particle '{}' is not a missing role in this reaction",
                        particle.label()
                    ))
                })
            }
        }
    }

    /// Resolve the two-to-two topology momenta from an event.
    pub fn resolve_two_to_two(&self, event: &NamedEventView<'_>) -> LadduResult<ResolvedTwoToTwo> {
        match &self.topology {
            ReactionTopology::TwoToTwo(topology) => topology.resolve(event),
        }
    }

    /// Construct a mass variable for a particle.
    pub fn mass(&self, particle: &Particle) -> Mass {
        Mass::from_reaction(self.clone(), particle.clone())
    }

    /// Construct an isobar decay view from a composite parent.
    pub fn decay(&self, parent: &Particle) -> LadduResult<Decay> {
        self.root_particle_for(parent)?;
        let daughters = parent.daughters();
        if daughters.len() != 2 {
            return Err(LadduError::Custom(
                "isobar decays must contain exactly two ordered daughters".to_string(),
            ));
        }
        Ok(Decay {
            reaction: self.clone(),
            parent: parent.clone(),
            daughter_1: daughters[0].clone(),
            daughter_2: daughters[1].clone(),
        })
    }

    /// Construct a Mandelstam variable for this reaction.
    pub fn mandelstam(&self, channel: Channel) -> Mandelstam {
        Mandelstam::new(self.clone(), channel)
    }

    /// Construct a polarization-angle variable for this reaction.
    pub fn pol_angle<A>(&self, angle_aux: A) -> PolAngle
    where
        A: Into<String>,
    {
        PolAngle::new(self.clone(), angle_aux)
    }

    /// Construct polarization variables for this reaction.
    pub fn polarization<M, A>(&self, magnitude_aux: M, angle_aux: A) -> Polarization
    where
        M: Into<String>,
        A: Into<String>,
    {
        Polarization::new(self.clone(), magnitude_aux, angle_aux)
    }

    /// Compute axes for a particle in this reaction.
    pub fn axes(
        &self,
        event: &NamedEventView<'_>,
        particle: &Particle,
        frame: Frame,
    ) -> LadduResult<FrameAxes> {
        let (root_index, root) = self.root_particle_for(particle)?;
        if particle == root {
            let resolved = self.resolve_two_to_two(event)?;
            let (parent, spectator) = match root_index {
                2 => (resolved.p3, resolved.p4),
                3 => (resolved.p4, resolved.p3),
                _ => {
                    return Err(LadduError::Custom(
                        "only outgoing p3 and p4 decay roots have frame axes".to_string(),
                    ));
                }
            };
            return FrameAxes::from_production_frame(
                frame,
                resolved.p1,
                parent,
                spectator,
                resolved.com_boost(),
            );
        }
        let parent = self.parent_of(particle).ok_or_else(|| {
            LadduError::Custom(format!(
                "particle '{}' is not connected to the reaction root",
                particle.label()
            ))
        })?;
        let parent_axes = self.axes(event, &parent, frame)?;
        parent_axes.for_daughter(self.p4_in_rest_frame_of(event, &parent, particle)?.vec3())
    }

    /// Compute a daughter's decay angles in its parent rest frame.
    pub fn angles_value(
        &self,
        event: &NamedEventView<'_>,
        parent: &Particle,
        daughter: &Particle,
        frame: Frame,
    ) -> LadduResult<DecayAngles> {
        if !parent
            .daughters()
            .iter()
            .any(|candidate| candidate == daughter)
        {
            return Err(LadduError::Custom(
                "daughter is not an immediate child of parent".to_string(),
            ));
        }
        let axes = self.axes(event, parent, frame)?;
        axes.angles(self.p4_in_rest_frame_of(event, parent, daughter)?.vec3())
    }

    fn root_particle_for(&self, particle: &Particle) -> LadduResult<(usize, &Particle)> {
        match &self.topology {
            ReactionTopology::TwoToTwo(topology) => {
                for (index, root) in [(2, topology.p3()), (3, topology.p4())] {
                    if root.contains(particle) {
                        return Ok((index, root));
                    }
                }
            }
        }
        Err(LadduError::Custom(
            "particle is not contained in an outgoing reaction root".to_string(),
        ))
    }

    fn parent_of(&self, child: &Particle) -> Option<Particle> {
        match &self.topology {
            ReactionTopology::TwoToTwo(topology) => [topology.p3(), topology.p4()]
                .into_iter()
                .find_map(|root| root.parent_of(child).cloned()),
        }
    }

    fn p4_in_rest_frame_of(
        &self,
        event: &NamedEventView<'_>,
        frame_particle: &Particle,
        target: &Particle,
    ) -> LadduResult<Vec4> {
        let (_, root) = self.root_particle_for(frame_particle)?;
        if frame_particle == root {
            let resolved = self.resolve_two_to_two(event)?;
            let com_boost = resolved.com_boost();
            let root_rest = RestFrame::new(self.p4(event, root)?.boost(&com_boost))?;
            return Ok(root_rest.transform(self.p4(event, target)?.boost(&com_boost)));
        }
        let parent = self.parent_of(frame_particle).ok_or_else(|| {
            LadduError::Custom("frame particle is not connected to root".to_string())
        })?;
        let frame_particle_in_parent = self.p4_in_rest_frame_of(event, &parent, frame_particle)?;
        let target_in_parent = self.p4_in_rest_frame_of(event, &parent, target)?;
        Ok(RestFrame::new(frame_particle_in_parent)?.transform(target_in_parent))
    }
}
