use serde::{Deserialize, Serialize};

use crate::{
    data::NamedEventView,
    utils::{
        enums::Frame,
        kinematics::{DecayAngles, FrameAxes, RestFrame},
        variables::IntoP4Selection,
    },
    LadduError, LadduResult, Vec3, Vec4,
};

/// A stable handle to a particle in a [`DecayChain`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ParticleHandle(usize);

impl ParticleHandle {
    /// Return the zero-based particle index in its decay chain.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A handle-based decay tree whose leaves resolve from p4 columns.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DecayChain {
    nodes: Vec<ParticleNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParticleNode {
    label: String,
    kind: ParticleNodeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ParticleNodeKind {
    Stable { p4_names: Vec<String> },
    Composite { daughters: Vec<ParticleHandle> },
}

impl DecayChain {
    /// Construct an empty decay chain.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a stable final-state particle backed by one or more p4 names.
    pub fn stable<S>(&mut self, label: impl Into<String>, p4: S) -> ParticleHandle
    where
        S: IntoP4Selection,
    {
        let selection = p4.into_selection();
        self.push_node(ParticleNode {
            label: label.into(),
            kind: ParticleNodeKind::Stable {
                p4_names: selection.names().to_vec(),
            },
        })
    }

    /// Add a composite particle whose four-momentum is the sum of its daughters.
    pub fn composite<I>(
        &mut self,
        label: impl Into<String>,
        daughters: I,
    ) -> LadduResult<ParticleHandle>
    where
        I: IntoIterator<Item = ParticleHandle>,
    {
        let daughters = daughters.into_iter().collect::<Vec<_>>();
        if daughters.is_empty() {
            return Err(LadduError::Custom(
                "composite particle must contain at least one daughter".to_string(),
            ));
        }
        for daughter in &daughters {
            self.node(*daughter)?;
        }
        Ok(self.push_node(ParticleNode {
            label: label.into(),
            kind: ParticleNodeKind::Composite { daughters },
        }))
    }

    /// Return the number of particles in the chain.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return whether this chain contains no particles.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return a particle label.
    pub fn label(&self, handle: ParticleHandle) -> LadduResult<&str> {
        Ok(&self.node(handle)?.label)
    }

    /// Resolve a particle four-momentum from an event.
    pub fn p4(&self, event: &NamedEventView<'_>, handle: ParticleHandle) -> LadduResult<Vec4> {
        match &self.node(handle)?.kind {
            ParticleNodeKind::Stable { p4_names } => {
                if p4_names.is_empty() {
                    return Err(LadduError::Custom(
                        "p4 reference must contain at least one name".to_string(),
                    ));
                }
                event
                    .get_p4_sum(p4_names.iter().map(String::as_str))
                    .ok_or_else(|| {
                        LadduError::Custom(format!(
                            "unknown p4 selection '{}'",
                            p4_names.join(", ")
                        ))
                    })
            }
            ParticleNodeKind::Composite { daughters } => daughters
                .iter()
                .map(|daughter| self.p4(event, *daughter))
                .try_fold(Vec4::new(0.0, 0.0, 0.0, 0.0), |acc, value| {
                    value.map(|value| acc + value)
                }),
        }
    }

    fn node(&self, handle: ParticleHandle) -> LadduResult<&ParticleNode> {
        self.nodes.get(handle.index()).ok_or_else(|| {
            LadduError::Custom(format!(
                "particle handle {} is not present in this decay chain",
                handle.index()
            ))
        })
    }

    fn parent_of(&self, child: ParticleHandle) -> LadduResult<Option<ParticleHandle>> {
        self.node(child)?;
        Ok(self
            .nodes
            .iter()
            .enumerate()
            .find_map(|(index, node)| match &node.kind {
                ParticleNodeKind::Stable { .. } => None,
                ParticleNodeKind::Composite { daughters } if daughters.contains(&child) => {
                    Some(ParticleHandle(index))
                }
                ParticleNodeKind::Composite { .. } => None,
            }))
    }

    fn contains(&self, root: ParticleHandle, particle: ParticleHandle) -> LadduResult<bool> {
        if root == particle {
            return Ok(true);
        }
        match &self.node(root)?.kind {
            ParticleNodeKind::Stable { .. } => Ok(false),
            ParticleNodeKind::Composite { daughters } => {
                for daughter in daughters {
                    if self.contains(*daughter, particle)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    fn push_node(&mut self, node: ParticleNode) -> ParticleHandle {
        let handle = ParticleHandle(self.nodes.len());
        self.nodes.push(node);
        handle
    }
}

/// A four-momentum reference used by reaction slots.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum KinematicRef {
    /// A p4 column name or alias selection.
    P4(Vec<String>),
    /// A handle into the reaction decay chain.
    Particle(ParticleHandle),
}

impl KinematicRef {
    /// Construct a p4 reference from a selection.
    pub fn p4<S>(selection: S) -> Self
    where
        S: IntoP4Selection,
    {
        Self::P4(selection.into_selection().names().to_vec())
    }

    /// Construct a particle reference.
    pub const fn particle(handle: ParticleHandle) -> Self {
        Self::Particle(handle)
    }
}

impl From<&str> for KinematicRef {
    fn from(value: &str) -> Self {
        Self::p4(value)
    }
}

impl From<String> for KinematicRef {
    fn from(value: String) -> Self {
        Self::p4(value)
    }
}

impl<const N: usize> From<[&str; N]> for KinematicRef {
    fn from(value: [&str; N]) -> Self {
        Self::p4(value)
    }
}

impl From<ParticleHandle> for KinematicRef {
    fn from(value: ParticleHandle) -> Self {
        Self::particle(value)
    }
}

/// A two-to-two reaction slot.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ReactionSlot {
    /// A known four-momentum source.
    Known(KinematicRef),
    /// A missing slot inferred by four-momentum conservation.
    Missing,
}

impl ReactionSlot {
    /// Construct a known slot.
    pub fn known(value: impl Into<KinematicRef>) -> Self {
        Self::Known(value.into())
    }

    /// Construct a missing slot.
    pub const fn missing() -> Self {
        Self::Missing
    }
}

impl<T> From<T> for ReactionSlot
where
    T: Into<KinematicRef>,
{
    fn from(value: T) -> Self {
        Self::known(value)
    }
}

/// A slot-based two-to-two reaction preserving `p1 + p2 -> p3 + p4` semantics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TwoToTwoReaction {
    p1: ReactionSlot,
    p2: ReactionSlot,
    p3: ReactionSlot,
    p4: ReactionSlot,
    missing_index: Option<usize>,
}

impl TwoToTwoReaction {
    /// Construct a fully specified two-to-two reaction.
    pub fn new(
        p1: impl Into<ReactionSlot>,
        p2: impl Into<ReactionSlot>,
        p3: impl Into<ReactionSlot>,
        p4: impl Into<ReactionSlot>,
    ) -> LadduResult<Self> {
        Self::from_slots(p1.into(), p2.into(), p3.into(), p4.into())
    }

    /// Construct a two-to-two reaction with `p1` inferred by conservation.
    pub fn missing_p1(
        p2: impl Into<KinematicRef>,
        p3: impl Into<KinematicRef>,
        p4: impl Into<KinematicRef>,
    ) -> LadduResult<Self> {
        Self::from_slots(
            ReactionSlot::Missing,
            ReactionSlot::known(p2),
            ReactionSlot::known(p3),
            ReactionSlot::known(p4),
        )
    }

    /// Construct a two-to-two reaction with `p2` inferred by conservation.
    pub fn missing_p2(
        p1: impl Into<KinematicRef>,
        p3: impl Into<KinematicRef>,
        p4: impl Into<KinematicRef>,
    ) -> LadduResult<Self> {
        Self::from_slots(
            ReactionSlot::known(p1),
            ReactionSlot::Missing,
            ReactionSlot::known(p3),
            ReactionSlot::known(p4),
        )
    }

    /// Construct a two-to-two reaction with `p3` inferred by conservation.
    pub fn missing_p3(
        p1: impl Into<KinematicRef>,
        p2: impl Into<KinematicRef>,
        p4: impl Into<KinematicRef>,
    ) -> LadduResult<Self> {
        Self::from_slots(
            ReactionSlot::known(p1),
            ReactionSlot::known(p2),
            ReactionSlot::Missing,
            ReactionSlot::known(p4),
        )
    }

    /// Construct a two-to-two reaction with `p4` inferred by conservation.
    pub fn missing_p4(
        p1: impl Into<KinematicRef>,
        p2: impl Into<KinematicRef>,
        p3: impl Into<KinematicRef>,
    ) -> LadduResult<Self> {
        Self::from_slots(
            ReactionSlot::known(p1),
            ReactionSlot::known(p2),
            ReactionSlot::known(p3),
            ReactionSlot::Missing,
        )
    }

    /// Resolve all four reaction momenta for one event.
    pub fn resolve(
        &self,
        event: &NamedEventView<'_>,
        chain: &DecayChain,
    ) -> LadduResult<ResolvedTwoToTwo> {
        let mut momenta = [
            self.resolve_known_slot(event, chain, 0)?,
            self.resolve_known_slot(event, chain, 1)?,
            self.resolve_known_slot(event, chain, 2)?,
            self.resolve_known_slot(event, chain, 3)?,
        ];
        if let Some(index) = self.missing_index {
            let inferred = match index {
                0 => momenta[2].unwrap() + momenta[3].unwrap() - momenta[1].unwrap(),
                1 => momenta[2].unwrap() + momenta[3].unwrap() - momenta[0].unwrap(),
                2 => momenta[0].unwrap() + momenta[1].unwrap() - momenta[3].unwrap(),
                3 => momenta[0].unwrap() + momenta[1].unwrap() - momenta[2].unwrap(),
                _ => unreachable!("validated two-to-two slot index"),
            };
            momenta[index] = Some(inferred);
        }
        let p1 = momenta[0].unwrap();
        let p2 = momenta[1].unwrap();
        let p3 = momenta[2].unwrap();
        let p4 = momenta[3].unwrap();
        Ok(ResolvedTwoToTwo { p1, p2, p3, p4 })
    }

    /// Return the zero-based missing slot index, if any.
    pub const fn missing_index(&self) -> Option<usize> {
        self.missing_index
    }

    /// Return the `p3` slot.
    pub const fn p3_slot(&self) -> &ReactionSlot {
        &self.p3
    }

    /// Return the `p4` slot.
    pub const fn p4_slot(&self) -> &ReactionSlot {
        &self.p4
    }

    fn from_slots(
        p1: ReactionSlot,
        p2: ReactionSlot,
        p3: ReactionSlot,
        p4: ReactionSlot,
    ) -> LadduResult<Self> {
        let missing = [&p1, &p2, &p3, &p4]
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| matches!(slot, ReactionSlot::Missing).then_some(index))
            .collect::<Vec<_>>();
        if missing.len() > 1 {
            return Err(LadduError::Custom(
                "two-to-two reaction can infer at most one missing slot".to_string(),
            ));
        }
        Ok(Self {
            p1,
            p2,
            p3,
            p4,
            missing_index: missing.first().copied(),
        })
    }

    fn resolve_known_slot(
        &self,
        event: &NamedEventView<'_>,
        chain: &DecayChain,
        index: usize,
    ) -> LadduResult<Option<Vec4>> {
        match self.slot(index) {
            ReactionSlot::Known(reference) => resolve_ref(event, chain, reference).map(Some),
            ReactionSlot::Missing => Ok(None),
        }
    }

    fn slot(&self, index: usize) -> &ReactionSlot {
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ReactionTopology {
    /// A two-to-two reaction with `p1 + p2 -> p3 + p4` semantics.
    TwoToTwo(TwoToTwoReaction),
}

impl From<TwoToTwoReaction> for ReactionTopology {
    fn from(value: TwoToTwoReaction) -> Self {
        Self::TwoToTwo(value)
    }
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

/// A handle-based sequential reaction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reaction {
    chain: DecayChain,
    topology: ReactionTopology,
}

impl Reaction {
    /// Construct a reaction from a decay chain and topology.
    pub fn new(chain: DecayChain, topology: impl Into<ReactionTopology>) -> Self {
        Self {
            chain,
            topology: topology.into(),
        }
    }

    /// Return the decay chain.
    pub const fn chain(&self) -> &DecayChain {
        &self.chain
    }

    /// Return the reaction topology.
    pub const fn topology(&self) -> &ReactionTopology {
        &self.topology
    }

    /// Resolve a particle p4 from an event.
    pub fn p4(&self, event: &NamedEventView<'_>, particle: ParticleHandle) -> LadduResult<Vec4> {
        self.chain.p4(event, particle)
    }

    /// Resolve the two-to-two topology momenta from an event.
    pub fn resolve_two_to_two(&self, event: &NamedEventView<'_>) -> LadduResult<ResolvedTwoToTwo> {
        match &self.topology {
            ReactionTopology::TwoToTwo(topology) => topology.resolve(event, &self.chain),
        }
    }

    /// Compute axes for a particle in this reaction.
    pub fn axes(
        &self,
        event: &NamedEventView<'_>,
        particle: ParticleHandle,
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
        let parent = self.chain.parent_of(particle)?.ok_or_else(|| {
            LadduError::Custom(format!(
                "particle {} is not connected to the reaction root",
                particle.index()
            ))
        })?;
        let parent_axes = self.axes(event, parent, frame)?;
        parent_axes.for_daughter(self.p4_in_rest_frame_of(event, parent, particle)?.vec3())
    }

    /// Compute a daughter's decay angles in its parent rest frame.
    pub fn angles(
        &self,
        event: &NamedEventView<'_>,
        parent: ParticleHandle,
        daughter: ParticleHandle,
        frame: Frame,
    ) -> LadduResult<DecayAngles> {
        if self.chain.parent_of(daughter)? != Some(parent) {
            return Err(LadduError::Custom(
                "daughter is not an immediate child of parent".to_string(),
            ));
        }
        let axes = self.axes(event, parent, frame)?;
        axes.angles(self.p4_in_rest_frame_of(event, parent, daughter)?.vec3())
    }

    fn root_particle_for(&self, particle: ParticleHandle) -> LadduResult<(usize, ParticleHandle)> {
        match &self.topology {
            ReactionTopology::TwoToTwo(topology) => {
                for (index, slot) in [(2, topology.p3_slot()), (3, topology.p4_slot())] {
                    if let ReactionSlot::Known(KinematicRef::Particle(root)) = slot {
                        if self.chain.contains(*root, particle)? {
                            return Ok((index, *root));
                        }
                    }
                }
            }
        }
        Err(LadduError::Custom(
            "particle is not contained in an outgoing decay-chain root".to_string(),
        ))
    }

    fn p4_in_rest_frame_of(
        &self,
        event: &NamedEventView<'_>,
        frame_particle: ParticleHandle,
        target: ParticleHandle,
    ) -> LadduResult<Vec4> {
        let (_, root) = self.root_particle_for(frame_particle)?;
        if frame_particle == root {
            let root_rest = RestFrame::new(self.chain.p4(event, root)?)?;
            return Ok(root_rest.transform(self.chain.p4(event, target)?));
        }
        let parent = self.chain.parent_of(frame_particle)?.ok_or_else(|| {
            LadduError::Custom("frame particle is not connected to root".to_string())
        })?;
        let frame_particle_in_parent = self.p4_in_rest_frame_of(event, parent, frame_particle)?;
        let target_in_parent = self.p4_in_rest_frame_of(event, parent, target)?;
        Ok(RestFrame::new(frame_particle_in_parent)?.transform(target_in_parent))
    }
}

fn resolve_ref(
    event: &NamedEventView<'_>,
    chain: &DecayChain,
    reference: &KinematicRef,
) -> LadduResult<Vec4> {
    match reference {
        KinematicRef::P4(names) => {
            if names.is_empty() {
                return Err(LadduError::Custom(
                    "p4 reference must contain at least one name".to_string(),
                ));
            }
            event
                .get_p4_sum(names.iter().map(String::as_str))
                .ok_or_else(|| {
                    LadduError::Custom(format!("unknown p4 selection '{}'", names.join(", ")))
                })
        }
        KinematicRef::Particle(handle) => chain.p4(event, *handle),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{Dataset, DatasetMetadata, EventData, Vec3};
    use approx::assert_relative_eq;

    use super::*;

    fn two_body_momentum(parent_mass: f64, daughter_1_mass: f64, daughter_2_mass: f64) -> f64 {
        let sum = daughter_1_mass + daughter_2_mass;
        let difference = daughter_1_mass - daughter_2_mass;
        ((parent_mass * parent_mass - sum * sum)
            * (parent_mass * parent_mass - difference * difference))
            .sqrt()
            / (2.0 * parent_mass)
    }

    fn pion_cascade_dataset() -> (
        Dataset,
        Reaction,
        ParticleHandle,
        ParticleHandle,
        ParticleHandle,
    ) {
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
        let mut chain = DecayChain::new();
        let pi_plus = chain.stable("pi+", "pi_plus");
        let pi_minus = chain.stable("pi-", "pi_minus");
        let pi0 = chain.stable("pi0", "pi0");
        let rho = chain.composite("rho", [pi_plus, pi_minus]).unwrap();
        let x = chain.composite("x", [rho, pi0]).unwrap();
        let topology = TwoToTwoReaction::new("beam", "target", x, "recoil").unwrap();
        let reaction = Reaction::new(chain, topology);
        (dataset, reaction, rho, pi_plus, x)
    }

    #[test]
    fn decay_chain_reconstructs_composites_from_lab_final_state() {
        let (dataset, reaction, rho, _, x) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let rho_p4 = reaction.p4(&event, rho).unwrap();
        let x_p4 = reaction.p4(&event, x).unwrap();

        assert_relative_eq!(rho_p4.m(), 0.775260000000000);
        assert!(x_p4.m() > rho_p4.m());
    }

    #[test]
    fn reaction_angles_use_handles_not_paths() {
        let (dataset, reaction, rho, pi_plus, _) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let angles = reaction
            .angles(&event, rho, pi_plus, Frame::Helicity)
            .unwrap();

        assert_relative_eq!(angles.costheta(), 0.500000000000000);
        assert_relative_eq!(angles.phi(), 0.700000000000000);
    }

    #[test]
    fn decay_chain_can_back_both_outgoing_two_to_two_slots() {
        let (dataset, _, _, _, _) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let mut chain = DecayChain::new();
        let pi_plus = chain.stable("pi+", "pi_plus");
        let pi_minus = chain.stable("pi-", "pi_minus");
        let pi0 = chain.stable("pi0", "pi0");
        let rho = chain.composite("rho", [pi_plus, pi_minus]).unwrap();
        let x = chain.composite("x", [rho, pi0]).unwrap();
        let recoil = chain.stable("recoil", "recoil");
        let recoil_system = chain.composite("recoil_system", [recoil]).unwrap();
        let topology = TwoToTwoReaction::new("beam", "target", x, recoil_system).unwrap();
        let reaction = Reaction::new(chain, topology);
        let resolved = reaction.resolve_two_to_two(&event).unwrap();
        let recoil_p4 = event.p4("recoil").unwrap();

        assert_relative_eq!(resolved.p4().px(), recoil_p4.px());
        assert_relative_eq!(resolved.p4().py(), recoil_p4.py());
        assert_relative_eq!(resolved.p4().pz(), recoil_p4.pz());
        assert_relative_eq!(resolved.p4().e(), recoil_p4.e());
        assert!(reaction
            .axes(&event, recoil_system, Frame::Helicity)
            .is_ok());
    }

    #[test]
    fn two_to_two_reaction_infers_any_single_missing_slot() {
        let (dataset, reaction, _, _, x) = pion_cascade_dataset();
        let event = dataset.event_view(0);
        let full = reaction.resolve_two_to_two(&event).unwrap();
        let missing_p2 =
            TwoToTwoReaction::missing_p2("beam", KinematicRef::particle(x), "recoil").unwrap();
        let resolved = missing_p2.resolve(&event, reaction.chain()).unwrap();

        assert_relative_eq!(resolved.p2().px(), full.p2().px());
        assert_relative_eq!(resolved.p2().py(), full.p2().py());
        assert_relative_eq!(resolved.p2().pz(), full.p2().pz());
        assert_relative_eq!(resolved.p2().e(), full.p2().e());
    }
}
