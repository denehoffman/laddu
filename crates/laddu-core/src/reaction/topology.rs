use std::{borrow::Borrow, fmt::Display};

use serde::{Deserialize, Serialize};

use super::decay::Decay;
use crate::{
    data::NamedEventView,
    kinematics::{DecayAngles, FrameAxes, RestFrame},
    quantum::{Channel, Frame},
    variables::{IntoP4Selection, Mandelstam, Mass, PolAngle, Polarization},
    vectors::{Vec3, Vec4},
    LadduError, LadduResult,
};

/// A kinematic particle or composite system used to define a reaction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    label: String,
    source: ParticleSource,
}

/// Source of a particle four-momentum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParticleSource {
    /// The four-momentum is read from one or more dataset p4 columns.
    Measured {
        /// Dataset p4 column names whose momenta are summed.
        p4_names: Vec<String>,
    },
    /// The four-momentum is fixed for every event.
    Fixed {
        /// Fixed four-momentum.
        p4: Vec4,
    },
    /// The four-momentum is solved from reaction-level four-momentum conservation.
    Missing,
    /// The four-momentum is the sum of daughter particles.
    Composite {
        /// Daughter particles whose momenta are summed.
        daughters: Vec<Particle>,
    },
}

impl Particle {
    /// Construct a measured particle backed by one or more p4 column names.
    pub fn measured<S>(label: impl Into<String>, p4: S) -> Self
    where
        S: IntoP4Selection,
    {
        Self {
            label: label.into(),
            source: ParticleSource::Measured {
                p4_names: p4.into_selection().names().to_vec(),
            },
        }
    }

    /// Construct a particle with a fixed four-momentum.
    pub fn fixed(label: impl Into<String>, p4: Vec4) -> Self {
        Self {
            label: label.into(),
            source: ParticleSource::Fixed { p4 },
        }
    }

    /// Construct a missing particle solved by the enclosing reaction topology.
    pub fn missing(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            source: ParticleSource::Missing,
        }
    }

    /// Construct a composite particle from daughter particles.
    pub fn composite<I, P>(label: impl Into<String>, daughters: I) -> LadduResult<Self>
    where
        I: IntoIterator<Item = P>,
        P: Borrow<Particle>,
    {
        let daughters = daughters
            .into_iter()
            .map(|daughter| daughter.borrow().clone())
            .collect::<Vec<_>>();
        if daughters.is_empty() {
            return Err(LadduError::Custom(
                "composite particle must contain at least one daughter".to_string(),
            ));
        }
        if daughters.iter().any(Self::contains_missing) {
            return Err(LadduError::Custom(
                "missing particles cannot be used as composite daughters".to_string(),
            ));
        }
        Ok(Self {
            label: label.into(),
            source: ParticleSource::Composite { daughters },
        })
    }

    /// Return the particle label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return the particle four-momentum source.
    pub const fn source(&self) -> &ParticleSource {
        &self.source
    }

    /// Return whether this particle is missing.
    pub const fn is_missing(&self) -> bool {
        matches!(self.source, ParticleSource::Missing)
    }

    fn contains_missing(&self) -> bool {
        self.is_missing() || self.daughters().iter().any(Self::contains_missing)
    }

    /// Return the daughters if this particle is composite.
    pub fn daughters(&self) -> &[Particle] {
        match &self.source {
            ParticleSource::Composite { daughters } => daughters,
            _ => &[],
        }
    }

    fn contains(&self, particle: &Particle) -> bool {
        if self == particle {
            return true;
        }
        self.daughters()
            .iter()
            .any(|daughter| daughter.contains(particle))
    }

    fn parent_of(&self, child: &Particle) -> Option<&Particle> {
        if self.daughters().iter().any(|daughter| daughter == child) {
            return Some(self);
        }
        self.daughters()
            .iter()
            .find_map(|daughter| daughter.parent_of(child))
    }
}

impl Display for Particle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// A direct two-to-two reaction preserving `p1 + p2 -> p3 + p4` semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoToTwoReaction {
    p1: Particle,
    p2: Particle,
    p3: Particle,
    p4: Particle,
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
            p1: p1.clone(),
            p2: p2.clone(),
            p3: p3.clone(),
            p4: p4.clone(),
            missing_index: missing.first().copied(),
        })
    }

    /// Return `p1`.
    pub const fn p1(&self) -> &Particle {
        &self.p1
    }

    /// Return `p2`.
    pub const fn p2(&self) -> &Particle {
        &self.p2
    }

    /// Return `p3`.
    pub const fn p3(&self) -> &Particle {
        &self.p3
    }

    /// Return `p4`.
    pub const fn p4(&self) -> &Particle {
        &self.p4
    }

    /// Return the zero-based missing particle index, if any.
    pub const fn missing_index(&self) -> Option<usize> {
        self.missing_index
    }

    /// Resolve all four reaction momenta for one event.
    pub fn resolve(&self, event: &NamedEventView<'_>) -> LadduResult<ResolvedTwoToTwo> {
        let mut momenta = [
            resolve_particle_direct(event, &self.p1)?,
            resolve_particle_direct(event, &self.p2)?,
            resolve_particle_direct(event, &self.p3)?,
            resolve_particle_direct(event, &self.p4)?,
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

    fn particle_at(&self, index: usize) -> &Particle {
        match index {
            0 => &self.p1,
            1 => &self.p2,
            2 => &self.p3,
            3 => &self.p4,
            _ => unreachable!("validated two-to-two slot index"),
        }
    }
}

/// A reaction topology.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReactionTopology {
    /// A two-to-two reaction with `p1 + p2 -> p3 + p4` semantics.
    TwoToTwo(TwoToTwoReaction),
}

/// Resolved event-level momenta for a two-to-two reaction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedTwoToTwo {
    p1: Vec4,
    p2: Vec4,
    p3: Vec4,
    p4: Vec4,
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

fn resolve_particle_direct(
    event: &NamedEventView<'_>,
    particle: &Particle,
) -> LadduResult<Option<Vec4>> {
    match particle.source() {
        ParticleSource::Measured { p4_names } => {
            if p4_names.is_empty() {
                return Err(LadduError::Custom(
                    "measured particle must contain at least one p4 name".to_string(),
                ));
            }
            event
                .get_p4_sum(p4_names.iter().map(String::as_str))
                .ok_or_else(|| {
                    LadduError::Custom(format!("unknown p4 selection '{}'", p4_names.join(", ")))
                })
                .map(Some)
        }
        ParticleSource::Fixed { p4 } => Ok(Some(*p4)),
        ParticleSource::Missing => Ok(None),
        ParticleSource::Composite { daughters } => daughters
            .iter()
            .map(|daughter| {
                resolve_particle_direct(event, daughter)?.ok_or_else(|| {
                    LadduError::Custom(format!(
                        "missing daughter '{}' cannot be resolved inside composite '{}'",
                        daughter.label(),
                        particle.label()
                    ))
                })
            })
            .try_fold(Vec4::new(0.0, 0.0, 0.0, 0.0), |acc, value| {
                value.map(|value| acc + value)
            })
            .map(Some),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;

    use super::*;
    use crate::{
        quantum::{AngularMomentum, AngularMomentumProjection, OrbitalAngularMomentum},
        traits::Variable,
        Dataset, DatasetMetadata, EventData, Vec3,
    };

    fn two_body_momentum(parent_mass: f64, daughter_1_mass: f64, daughter_2_mass: f64) -> f64 {
        let sum = daughter_1_mass + daughter_2_mass;
        let difference = daughter_1_mass - daughter_2_mass;
        ((parent_mass * parent_mass - sum * sum)
            * (parent_mass * parent_mass - difference * difference))
            .sqrt()
            / (2.0 * parent_mass)
    }

    fn pion_cascade_dataset() -> (Dataset, Reaction, Particle, Particle, Particle) {
        let pion_mass = 0.139570000000000;
        let rho_mass = 0.775260000000000;
        let rho_momentum_in_x_rest = 0.450000000000000;
        let expected_costheta = 0.500000000000000;
        let expected_phi = 0.700000000000000_f64;
        let rho_in_x_rest = Vec3::new(rho_momentum_in_x_rest, 0.0, 0.0).with_mass(rho_mass);
        let bachelor_in_x_rest = Vec3::new(-rho_momentum_in_x_rest, 0.0, 0.0).with_mass(pion_mass);
        let x_rest = rho_in_x_rest + bachelor_in_x_rest;
        let lab_boost = Vec3::new(0.0, 0.0, 0.350000000000000);
        let x_lab = x_rest.boost(&lab_boost);
        let recoil_lab =
            Vec3::new(-0.200000000000000, 0.0, 0.400000000000000).with_mass(0.938000000000000);
        let beam_lab = Vec4::new(0.0, 0.0, 6.000000000000000, 6.000000000000000);
        let target_lab = x_lab + recoil_lab - beam_lab;
        let x_rest_axes = FrameAxes::from_production_frame(
            Frame::Helicity,
            beam_lab,
            x_lab,
            recoil_lab,
            -(beam_lab + target_lab).beta(),
        )
        .unwrap();
        let pion_momentum_in_rho_rest = two_body_momentum(rho_mass, pion_mass, pion_mass);
        let rho_axes = x_rest_axes.for_daughter(rho_in_x_rest.vec3()).unwrap();
        let sintheta = f64::sqrt(1.0 - expected_costheta * expected_costheta);
        let pi_plus_direction_in_rho_rest = rho_axes.x() * (sintheta * expected_phi.cos())
            + rho_axes.y() * (sintheta * expected_phi.sin())
            + rho_axes.z() * expected_costheta;
        let pi_plus_in_rho_rest =
            (pi_plus_direction_in_rho_rest * pion_momentum_in_rho_rest).with_mass(pion_mass);
        let pi_minus_in_rho_rest =
            (-pi_plus_direction_in_rho_rest * pion_momentum_in_rho_rest).with_mass(pion_mass);
        let rho_beta_in_x_rest = rho_in_x_rest.beta();
        let pi_plus_in_x_rest = pi_plus_in_rho_rest.boost(&rho_beta_in_x_rest);
        let pi_minus_in_x_rest = pi_minus_in_rho_rest.boost(&rho_beta_in_x_rest);
        let pi_plus_lab = pi_plus_in_x_rest.boost(&lab_boost);
        let pi_minus_lab = pi_minus_in_x_rest.boost(&lab_boost);
        let bachelor_lab = bachelor_in_x_rest.boost(&lab_boost);
        let metadata = Arc::new(
            DatasetMetadata::new(
                vec!["beam", "target", "pi_plus", "pi_minus", "pi0", "recoil"],
                Vec::<String>::new(),
            )
            .unwrap(),
        );
        let dataset = Dataset::new_with_metadata(
            vec![Arc::new(EventData {
                p4s: vec![
                    beam_lab,
                    target_lab,
                    pi_plus_lab,
                    pi_minus_lab,
                    bachelor_lab,
                    recoil_lab,
                ],
                aux: vec![],
                weight: 1.0,
            })],
            metadata,
        );
        let pi_plus = Particle::measured("pi+", "pi_plus");
        let pi_minus = Particle::measured("pi-", "pi_minus");
        let pi0 = Particle::measured("pi0", "pi0");
        let rho = Particle::composite("rho", [&pi_plus, &pi_minus]).unwrap();
        let x = Particle::composite("x", [&rho, &pi0]).unwrap();
        let beam = Particle::measured("beam", "beam");
        let target = Particle::measured("target", "target");
        let recoil = Particle::measured("recoil", "recoil");
        let reaction = Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
        (dataset, reaction, rho, pi_plus, x)
    }

    #[test]
    fn reaction_reconstructs_composites_from_lab_final_state() {
        let (dataset, reaction, rho, _, x) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let rho_p4 = reaction.p4(&event, &rho).unwrap();
        let x_p4 = reaction.p4(&event, &x).unwrap();

        assert_relative_eq!(rho_p4.m(), 0.775260000000000);
        assert!(x_p4.m() > rho_p4.m());
    }

    #[test]
    fn reaction_angle_variables_use_particles_not_paths() {
        let (dataset, reaction, rho, pi_plus, _) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let decay = reaction.decay(&rho);
        assert!(decay.is_ok());
        let angles = decay.unwrap().angles(&pi_plus, Frame::Helicity).unwrap();

        assert_relative_eq!(angles.costheta.value(&event), 0.500000000000000);
        assert_relative_eq!(angles.phi.value(&event), 0.700000000000000);
    }

    #[test]
    fn two_to_two_reaction_solves_missing_particle() {
        let (dataset, reaction, _, _, x) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let full = reaction.resolve_two_to_two(&event).unwrap();
        let beam = Particle::measured("beam", "beam");
        let target = Particle::missing("target");
        let recoil = Particle::measured("recoil", "recoil");
        let missing_reaction = Reaction::two_to_two(&beam, &target, &x, &recoil).unwrap();
        let resolved = missing_reaction.resolve_two_to_two(&event).unwrap();

        assert_relative_eq!(resolved.p2().px(), full.p2().px());
        assert_relative_eq!(resolved.p2().py(), full.p2().py());
        assert_relative_eq!(resolved.p2().pz(), full.p2().pz());
        assert_relative_eq!(resolved.p2().e(), full.p2().e());
    }

    #[test]
    fn fixed_particle_can_define_a_reaction_role() {
        let (dataset, _, _, _, x) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let beam = Particle::measured("beam", "beam");
        let fixed_target = Particle::fixed("target", event.p4("target").unwrap());
        let recoil = Particle::measured("recoil", "recoil");
        let reaction = Reaction::two_to_two(&beam, &fixed_target, &x, &recoil).unwrap();
        let resolved = reaction.resolve_two_to_two(&event).unwrap();

        assert_relative_eq!(resolved.p2().e(), event.p4("target").unwrap().e());
    }

    #[test]
    fn reaction_mandelstam_variables_match_resolved_values() {
        let (dataset, reaction, _, _, _) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let resolved = reaction.resolve_two_to_two(&event).unwrap();

        assert_relative_eq!(reaction.mandelstam(Channel::S).value(&event), resolved.s());
        assert_relative_eq!(reaction.mandelstam(Channel::T).value(&event), resolved.t());
        assert_relative_eq!(reaction.mandelstam(Channel::U).value(&event), resolved.u());
    }

    #[test]
    fn decay_factors_with_matching_names_deduplicate() {
        let (dataset, reaction, rho, pi_plus, _) = pion_cascade_dataset();
        let dataset = Arc::new(dataset);
        let decay = reaction.decay(&rho).unwrap();
        let factor_1 = decay
            .canonical_factor(
                "rho.factor",
                AngularMomentum::integer(1),
                AngularMomentumProjection::integer(0),
                OrbitalAngularMomentum::integer(1),
                AngularMomentum::integer(0),
                &pi_plus,
                AngularMomentum::integer(0),
                AngularMomentum::integer(0),
                AngularMomentumProjection::integer(0),
                AngularMomentumProjection::integer(0),
                Frame::Helicity,
            )
            .unwrap();
        let factor_2 = decay
            .canonical_factor(
                "rho.factor",
                AngularMomentum::integer(1),
                AngularMomentumProjection::integer(0),
                OrbitalAngularMomentum::integer(1),
                AngularMomentum::integer(0),
                &pi_plus,
                AngularMomentum::integer(0),
                AngularMomentum::integer(0),
                AngularMomentumProjection::integer(0),
                AngularMomentumProjection::integer(0),
                Frame::Helicity,
            )
            .unwrap();

        let evaluator = (&factor_1 + &factor_2).load(&dataset).unwrap();

        assert_eq!(
            evaluator.amplitudes.len(),
            factor_1.load(&dataset).unwrap().amplitudes.len()
        );
    }
}
