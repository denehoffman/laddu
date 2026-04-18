use laddu_core::{AngularMomentum, Expression, Frame, LadduError, LadduResult, Parity};
use serde::{Deserialize, Serialize};

/// Metadata for a particle used by sequential-decay builders.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ParticleDef {
    name: String,
    spin: AngularMomentum,
    parity: Option<Parity>,
}

impl ParticleDef {
    /// Construct a particle definition with no parity assignment.
    pub fn new(name: impl Into<String>, spin: AngularMomentum) -> LadduResult<Self> {
        Self::with_parity(name, spin, None)
    }

    /// Construct a particle definition with an optional parity assignment.
    pub fn with_parity(
        name: impl Into<String>,
        spin: AngularMomentum,
        parity: Option<Parity>,
    ) -> LadduResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(LadduError::Custom(
                "particle name must not be empty".to_string(),
            ));
        }
        Ok(Self { name, spin, parity })
    }

    /// Return the particle name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the particle spin.
    pub const fn spin(&self) -> AngularMomentum {
        self.spin
    }

    /// Return the optional particle parity.
    pub const fn parity(&self) -> Option<Parity> {
        self.parity
    }
}

/// Decay coupling basis used to organize amplitudes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum CouplingBasis {
    /// No spin degrees of freedom are coupled.
    Spinless,
    /// Helicity-basis couplings.
    Helicity,
    /// Canonical spin-basis couplings with orbital angular momentum and coupled spin.
    Canonical,
}

/// Basis used to represent production spin states.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ProductionStateBasis {
    /// Direct spin-projection amplitudes.
    SpinProjection,
    /// Reflectivity-adapted amplitudes.
    Reflectivity,
}

/// Production convention used to organize production amplitudes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ProductionBasis {
    frame: Frame,
    state_basis: ProductionStateBasis,
}

impl ProductionBasis {
    /// Construct a production basis from a frame and state basis.
    pub const fn new(frame: Frame, state_basis: ProductionStateBasis) -> Self {
        Self { frame, state_basis }
    }

    /// Construct a helicity-frame spin-projection basis.
    pub fn helicity() -> Self {
        Self::new(Frame::Helicity, ProductionStateBasis::SpinProjection)
    }

    /// Construct a Gottfried-Jackson-frame spin-projection basis.
    pub fn gottfried_jackson() -> Self {
        Self::new(
            Frame::GottfriedJackson,
            ProductionStateBasis::SpinProjection,
        )
    }

    /// Construct a reflectivity basis in a chosen production frame.
    pub const fn reflectivity_in(frame: Frame) -> Self {
        Self::new(frame, ProductionStateBasis::Reflectivity)
    }

    /// Return the production frame.
    pub const fn frame(self) -> Frame {
        self.frame
    }

    /// Return the production state basis.
    pub const fn state_basis(self) -> ProductionStateBasis {
        self.state_basis
    }
}

/// Optional dynamic amplitude inserted at a decay node.
#[derive(Clone, Serialize, Deserialize)]
pub enum DynamicsExpr {
    /// No dynamic amplitude is attached to the node.
    None,
    /// A user-provided expression supplies the node dynamics.
    Expression(Box<Expression>),
}

impl DynamicsExpr {
    /// Construct a dynamics expression from a user-provided expression.
    pub fn expression(expression: Expression) -> Self {
        Self::Expression(Box::new(expression))
    }
}

/// Daughter structure for a decay node.
#[derive(Clone, Serialize, Deserialize)]
pub enum DecayDaughters {
    /// A stable particle with no daughter decay.
    Stable,
    /// A two-body decay into two child nodes.
    TwoBody(Box<DecayNode>, Box<DecayNode>),
}

/// A node in a sequential-decay tree.
#[derive(Clone, Serialize, Deserialize)]
pub struct DecayNode {
    parent: ParticleDef,
    daughters: DecayDaughters,
    dynamics: DynamicsExpr,
    coupling_basis: CouplingBasis,
}

impl DecayNode {
    /// Construct a stable-particle node.
    pub fn stable(particle: ParticleDef) -> Self {
        Self {
            parent: particle,
            daughters: DecayDaughters::Stable,
            dynamics: DynamicsExpr::None,
            coupling_basis: CouplingBasis::Spinless,
        }
    }

    /// Construct a two-body decay node.
    pub fn two_body(parent: ParticleDef, daughter_1: Self, daughter_2: Self) -> Self {
        Self {
            parent,
            daughters: DecayDaughters::TwoBody(Box::new(daughter_1), Box::new(daughter_2)),
            dynamics: DynamicsExpr::None,
            coupling_basis: CouplingBasis::Spinless,
        }
    }

    /// Set the decay node dynamics expression.
    pub fn with_dynamics(mut self, dynamics: DynamicsExpr) -> Self {
        self.dynamics = dynamics;
        self
    }

    /// Set the decay node coupling basis.
    pub fn with_coupling_basis(mut self, coupling_basis: CouplingBasis) -> Self {
        self.coupling_basis = coupling_basis;
        self
    }

    /// Return the parent particle definition.
    pub const fn parent(&self) -> &ParticleDef {
        &self.parent
    }

    /// Return the daughter structure.
    pub const fn daughters(&self) -> &DecayDaughters {
        &self.daughters
    }

    /// Return the node dynamics expression.
    pub const fn dynamics(&self) -> &DynamicsExpr {
        &self.dynamics
    }

    /// Return the node coupling basis.
    pub const fn coupling_basis(&self) -> CouplingBasis {
        self.coupling_basis
    }
}

/// Root production node for a sequential-decay model.
#[derive(Clone, Serialize, Deserialize)]
pub struct ProductionNode {
    initial: Vec<ParticleDef>,
    final_state: DecayNode,
    production_basis: ProductionBasis,
}

impl ProductionNode {
    /// Construct a production node.
    pub fn new(
        initial: Vec<ParticleDef>,
        final_state: DecayNode,
        production_basis: ProductionBasis,
    ) -> LadduResult<Self> {
        if initial.is_empty() {
            return Err(LadduError::Custom(
                "production node must contain at least one initial particle".to_string(),
            ));
        }
        Ok(Self {
            initial,
            final_state,
            production_basis,
        })
    }

    /// Return the initial particles.
    pub fn initial(&self) -> &[ParticleDef] {
        &self.initial
    }

    /// Return the final-state decay tree.
    pub const fn final_state(&self) -> &DecayNode {
        &self.final_state
    }

    /// Return the production basis.
    pub const fn production_basis(&self) -> ProductionBasis {
        self.production_basis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_definition_validates_spin_parity_and_name() {
        let pion = ParticleDef::with_parity(
            "pi0",
            AngularMomentum::from_twice(0),
            Some(Parity::Negative),
        )
        .unwrap();
        assert_eq!(pion.name(), "pi0");
        assert_eq!(pion.spin().value(), 0);
        assert_eq!(pion.parity(), Some(Parity::Negative));
        assert_eq!(pion.parity().unwrap().value(), -1);

        assert!(ParticleDef::new("", AngularMomentum::from_twice(0)).is_err());
    }

    #[test]
    fn decay_tree_nodes_store_structure_and_bases() {
        let parent = ParticleDef::new("x", AngularMomentum::from_twice(0)).unwrap();
        let daughter_1 =
            DecayNode::stable(ParticleDef::new("a", AngularMomentum::from_twice(0)).unwrap());
        let daughter_2 =
            DecayNode::stable(ParticleDef::new("b", AngularMomentum::from_twice(0)).unwrap());
        let node = DecayNode::two_body(parent, daughter_1, daughter_2)
            .with_coupling_basis(CouplingBasis::Canonical);

        assert_eq!(node.parent().name(), "x");
        assert_eq!(node.coupling_basis(), CouplingBasis::Canonical);
        assert!(matches!(node.daughters(), DecayDaughters::TwoBody(_, _)));
    }

    #[test]
    fn production_node_requires_initial_particles() {
        let final_state =
            DecayNode::stable(ParticleDef::new("x", AngularMomentum::from_twice(0)).unwrap());
        assert!(
            ProductionNode::new(vec![], final_state.clone(), ProductionBasis::helicity()).is_err()
        );

        let beam = ParticleDef::new("beam", AngularMomentum::from_twice(2)).unwrap();
        let production = ProductionNode::new(
            vec![beam],
            final_state,
            ProductionBasis::gottfried_jackson(),
        )
        .unwrap();
        assert_eq!(production.initial().len(), 1);
        assert_eq!(production.final_state().parent().name(), "x");
        assert_eq!(
            production.production_basis().frame(),
            Frame::GottfriedJackson
        );
        assert_eq!(
            production.production_basis().state_basis(),
            ProductionStateBasis::SpinProjection
        );
    }

    #[test]
    fn production_basis_keeps_frame_and_state_basis_separate() {
        let basis = ProductionBasis::reflectivity_in(Frame::GottfriedJackson);
        assert_eq!(basis.frame(), Frame::GottfriedJackson);
        assert_eq!(basis.state_basis(), ProductionStateBasis::Reflectivity);
    }
}
