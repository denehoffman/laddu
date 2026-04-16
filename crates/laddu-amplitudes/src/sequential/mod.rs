use laddu_core::{Expression, LadduError, LadduResult};
use serde::{Deserialize, Serialize};

/// A non-negative angular momentum stored as twice its physical value.
///
/// This representation keeps integer and half-integer quantum numbers exact. For example,
/// `AngularMomentum::new(1)` represents `$1/2$`, and `AngularMomentum::new(2)` represents `$1$`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AngularMomentum(u32);

impl AngularMomentum {
    /// Construct a doubled non-negative angular momentum.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the doubled integer value.
    pub const fn value(self) -> u32 {
        self.0
    }

    /// Return the physical value as `f64`.
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 2.0
    }

    /// Return whether this quantum number represents an integer value.
    pub const fn is_integer(self) -> bool {
        self.0 & 1 == 0
    }

    /// Return whether this angular momentum has the same integer/half-integer parity as
    /// `projection`.
    pub const fn has_same_parity_as(self, projection: AngularMomentumProjection) -> bool {
        (self.0 & 1) as i32 == projection.value() & 1
    }
}

/// A signed angular-momentum projection stored as twice its physical value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AngularMomentumProjection(i32);

impl AngularMomentumProjection {
    /// Construct a doubled signed angular-momentum projection.
    pub const fn new(value: i32) -> Self {
        Self(value)
    }

    /// Return the doubled integer value.
    pub const fn value(self) -> i32 {
        self.0
    }

    /// Return the physical value as `f64`.
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 2.0
    }

    /// Return whether this projection represents an integer value.
    pub const fn is_integer(self) -> bool {
        self.0 & 1 == 0
    }
}

/// A validated spin state with spin and projection stored as doubled quantum numbers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SpinState {
    spin: AngularMomentum,
    projection: AngularMomentumProjection,
}

impl SpinState {
    /// Construct a spin state after validating projection bounds and parity.
    pub fn new(spin: AngularMomentum, projection: AngularMomentumProjection) -> LadduResult<Self> {
        validate_projection(spin, projection)?;
        Ok(Self { spin, projection })
    }

    /// Return the spin quantum number.
    pub const fn spin(self) -> AngularMomentum {
        self.spin
    }

    /// Return the spin projection quantum number.
    pub const fn projection(self) -> AngularMomentumProjection {
        self.projection
    }

    /// Enumerate all allowed projections for `spin`.
    pub fn allowed_projections(spin: AngularMomentum) -> Vec<Self> {
        let spin_value = spin.value() as i32;
        (-spin_value..=spin_value)
            .step_by(2)
            .map(|projection| Self {
                spin,
                projection: AngularMomentumProjection::new(projection),
            })
            .collect()
    }
}

/// A non-negative integer orbital angular momentum.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct OrbitalAngularMomentum(u32);

impl OrbitalAngularMomentum {
    /// Construct an orbital angular momentum.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Construct an orbital angular momentum from a non-negative angular momentum.
    pub fn from_angular_momentum(value: AngularMomentum) -> LadduResult<Self> {
        if !value.is_integer() {
            return Err(LadduError::Custom(
                "orbital angular momentum must be an integer".to_string(),
            ));
        }
        Ok(Self(value.value() / 2))
    }

    /// Return the integer orbital angular momentum.
    pub const fn value(self) -> u32 {
        self.0
    }

    /// Return the orbital angular momentum as a doubled non-negative angular momentum.
    pub fn angular_momentum(self) -> AngularMomentum {
        AngularMomentum::new(2 * self.0)
    }
}

/// Metadata for a particle used by sequential-decay builders.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ParticleDef {
    name: String,
    spin: AngularMomentum,
    parity: Option<Parity>,
}

/// Intrinsic parity assignment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum Parity {
    /// Positive intrinsic parity.
    Positive,
    /// Negative intrinsic parity.
    Negative,
}

impl Parity {
    /// Return the parity as `+1` or `-1`.
    pub const fn value(self) -> i8 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
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

/// Production frame used to define spin quantization axes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ProductionFrame {
    /// Helicity-frame production amplitudes.
    Helicity,
    /// Gottfried-Jackson-frame production amplitudes.
    GottfriedJackson,
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
    frame: ProductionFrame,
    state_basis: ProductionStateBasis,
}

impl ProductionBasis {
    /// Construct a production basis from a frame and state basis.
    pub const fn new(frame: ProductionFrame, state_basis: ProductionStateBasis) -> Self {
        Self { frame, state_basis }
    }

    /// Construct a helicity-frame spin-projection basis.
    pub const fn helicity() -> Self {
        Self::new(
            ProductionFrame::Helicity,
            ProductionStateBasis::SpinProjection,
        )
    }

    /// Construct a Gottfried-Jackson-frame spin-projection basis.
    pub const fn gottfried_jackson() -> Self {
        Self::new(
            ProductionFrame::GottfriedJackson,
            ProductionStateBasis::SpinProjection,
        )
    }

    /// Construct a reflectivity basis in a chosen production frame.
    pub const fn reflectivity_in(frame: ProductionFrame) -> Self {
        Self::new(frame, ProductionStateBasis::Reflectivity)
    }

    /// Return the production frame.
    pub const fn frame(self) -> ProductionFrame {
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

fn validate_projection(
    spin: AngularMomentum,
    projection: AngularMomentumProjection,
) -> LadduResult<()> {
    if projection.value().unsigned_abs() > spin.value() {
        return Err(LadduError::Custom(
            "spin projection must satisfy -J <= m <= J".to_string(),
        ));
    }
    if !spin.has_same_parity_as(projection) {
        return Err(LadduError::Custom(
            "spin projection must have the same integer or half-integer parity as spin".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_state_accepts_integer_and_half_integer_values() {
        let spin_one = AngularMomentum::new(2);
        let spin_half = AngularMomentum::new(1);
        assert_eq!(
            SpinState::new(spin_one, AngularMomentumProjection::new(0))
                .unwrap()
                .projection()
                .value(),
            0
        );
        assert_eq!(
            SpinState::new(spin_half, AngularMomentumProjection::new(-1))
                .unwrap()
                .projection()
                .value(),
            -1
        );
    }

    #[test]
    fn spin_state_rejects_invalid_projection() {
        let spin_one = AngularMomentum::new(2);
        assert!(SpinState::new(spin_one, AngularMomentumProjection::new(4)).is_err());
        assert!(SpinState::new(spin_one, AngularMomentumProjection::new(1)).is_err());
    }

    #[test]
    fn spin_state_enumerates_allowed_projections() {
        let spin_zero = SpinState::allowed_projections(AngularMomentum::new(0));
        assert_eq!(
            spin_zero
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![0]
        );

        let spin_half = SpinState::allowed_projections(AngularMomentum::new(1));
        assert_eq!(
            spin_half
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-1, 1]
        );

        let spin_one = SpinState::allowed_projections(AngularMomentum::new(2));
        assert_eq!(
            spin_one
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-2, 0, 2]
        );

        let spin_three_halves = SpinState::allowed_projections(AngularMomentum::new(3));
        assert_eq!(
            spin_three_halves
                .iter()
                .map(|state| state.projection().value())
                .collect::<Vec<_>>(),
            vec![-3, -1, 1, 3]
        );
    }

    #[test]
    fn orbital_angular_momentum_rejects_half_integer_values() {
        assert_eq!(
            OrbitalAngularMomentum::from_angular_momentum(AngularMomentum::new(4))
                .unwrap()
                .value(),
            2
        );
        assert!(OrbitalAngularMomentum::from_angular_momentum(AngularMomentum::new(3)).is_err());
    }

    #[test]
    fn particle_definition_validates_spin_parity_and_name() {
        let pion = ParticleDef::with_parity("pi0", AngularMomentum::new(0), Some(Parity::Negative))
            .unwrap();
        assert_eq!(pion.name(), "pi0");
        assert_eq!(pion.spin().value(), 0);
        assert_eq!(pion.parity(), Some(Parity::Negative));
        assert_eq!(pion.parity().unwrap().value(), -1);

        assert!(ParticleDef::new("", AngularMomentum::new(0)).is_err());
    }

    #[test]
    fn decay_tree_nodes_store_structure_and_bases() {
        let parent = ParticleDef::new("x", AngularMomentum::new(0)).unwrap();
        let daughter_1 = DecayNode::stable(ParticleDef::new("a", AngularMomentum::new(0)).unwrap());
        let daughter_2 = DecayNode::stable(ParticleDef::new("b", AngularMomentum::new(0)).unwrap());
        let node = DecayNode::two_body(parent, daughter_1, daughter_2)
            .with_coupling_basis(CouplingBasis::Canonical);

        assert_eq!(node.parent().name(), "x");
        assert_eq!(node.coupling_basis(), CouplingBasis::Canonical);
        assert!(matches!(node.daughters(), DecayDaughters::TwoBody(_, _)));
    }

    #[test]
    fn production_node_requires_initial_particles() {
        let final_state =
            DecayNode::stable(ParticleDef::new("x", AngularMomentum::new(0)).unwrap());
        assert!(
            ProductionNode::new(vec![], final_state.clone(), ProductionBasis::helicity()).is_err()
        );

        let beam = ParticleDef::new("beam", AngularMomentum::new(2)).unwrap();
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
            ProductionFrame::GottfriedJackson
        );
        assert_eq!(
            production.production_basis().state_basis(),
            ProductionStateBasis::SpinProjection
        );
    }

    #[test]
    fn production_basis_keeps_frame_and_state_basis_separate() {
        let basis = ProductionBasis::reflectivity_in(ProductionFrame::GottfriedJackson);
        assert_eq!(basis.frame(), ProductionFrame::GottfriedJackson);
        assert_eq!(basis.state_basis(), ProductionStateBasis::Reflectivity);
    }
}
