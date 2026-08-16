#[derive(Clone, Copy, Debug)]
/// A named generated edge and its four-momentum.
pub struct NamedMomentum<'a> {
    /// Edge name.
    pub name: &'a str,
    /// Four-momentum in `(E, px, py, pz)` order.
    pub p4: RealVec4,
}
#[derive(Clone, Copy, Debug)]
/// A named generated edge and its proposed invariant mass.
pub struct NamedMass<'a> {
    /// Edge name.
    pub name: &'a str,
    /// Invariant mass.
    pub mass: f64,
}
#[derive(Clone, Debug)]
/// Kinematics and importance weight produced by a vertex proposal.
pub struct ProposalResult {
    /// Proposed outgoing four-momenta in edge order.
    pub outgoing: Vec<RealVec4>,
    /// The proposal correction, conventionally `dPhi / q`.
    pub weight: f64,
}

/// Kinematic proposal attached to a channel vertex.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VertexProposal {
    /// Generate an isotropic one-to-two decay.
    #[default]
    TwoBodyDecay,
    /// Generate two-to-two scattering from a transfer distribution.
    TwoBodyScattering {
        /// Scattering proposal configuration.
        proposal: TwoBodyScattering,
    },
}

impl VertexProposal {
    /// Construct an isotropic two-body decay proposal.
    pub fn isotropic_decay() -> Self {
        Self::TwoBodyDecay
    }
    /// Construct a two-body scattering proposal distributed in momentum transfer.
    pub fn t_exchange(
        pairing: (impl Into<String>, impl Into<String>),
        distribution: TDistribution,
    ) -> Self {
        Self::TwoBodyScattering {
            proposal: TwoBodyScattering::t_exchange(pairing, distribution),
        }
    }
    /// Propose outgoing kinematics for a vertex.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the incoming or outgoing topology,
    /// masses, or kinematics are invalid for the selected proposal.
    pub fn propose(
        &self,
        incoming: &[NamedMomentum<'_>],
        outgoing: &[NamedMass<'_>],
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<ProposalResult> {
        match self {
            Self::TwoBodyDecay => propose_two_body_decay(incoming, outgoing, rng),
            Self::TwoBodyScattering { proposal } => proposal.propose(incoming, outgoing, rng),
        }
    }


    /// Return continuous-coordinate and analytical-region counts used by the
    /// proven-envelope report.
    #[doc(hidden)]
    pub fn proven_domain_metadata(&self) -> (usize, usize) {
        match self {
            Self::TwoBodyDecay => (2, 1),
            Self::TwoBodyScattering { proposal } => proposal.proven_domain_metadata(),
        }
    }
}

fn propose_two_body_decay(
    incoming: &[NamedMomentum<'_>],
    outgoing: &[NamedMass<'_>],
    rng: &mut ProposalRng,
) -> LadduPhysicsResult<ProposalResult> {
    let kinematics = TwoBodyDecayKinematics::new("isotropic", incoming, outgoing)?;
    let direction = rng.isotropic_direction();
    kinematics.finish(direction, 1.0)
}

struct TwoBodyDecayKinematics {
    parent: RealVec4,
    daughter_masses: [f64; 2],
    momentum: f64,
    phase_space: f64,
}

impl TwoBodyDecayKinematics {
    fn new(
        family: &str,
        incoming: &[NamedMomentum<'_>],
        outgoing: &[NamedMass<'_>],
    ) -> LadduPhysicsResult<Self> {
        if incoming.len() != 1 || outgoing.len() != 2 {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "{family} decay requires one incoming and two outgoing edges, got {} incoming and {} outgoing",
                incoming.len(),
                outgoing.len()
            )));
        }
        let parent = incoming[0].p4;
        let mass = parent.m()?;
        let daughter_masses = [outgoing[0].mass, outgoing[1].mass];
        let momentum = two_body_momentum(mass, daughter_masses[0], daughter_masses[1])?;
        Ok(Self {
            parent,
            daughter_masses,
            momentum,
            phase_space: momentum / (4.0 * PI * mass),
        })
    }
    fn finish(
        self,
        direction: RealVec3,
        angular_correction: f64,
    ) -> LadduPhysicsResult<ProposalResult> {
        let first = on_shell(direction, self.momentum, self.daughter_masses[0]);
        let second = on_shell(-direction, self.momentum, self.daughter_masses[1]);
        let beta = self.parent.beta()?;
        Ok(ProposalResult {
            outgoing: vec![first.boost(&beta), second.boost(&beta)],
            weight: self.phase_space * angular_correction,
        })
    }
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct AdaptiveTwoBodyDecay {
    density: PiecewiseDensity,
    defensive_fraction: f64,
}

impl AdaptiveTwoBodyDecay {
    /// Construct an angular proposal from nonnegative pilot-bin counts.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the counts are empty, negative, or
    /// non-finite, their total is not positive and finite, or
    /// `defensive_fraction` is outside `[0, 1]`.
    pub fn new(counts: Arc<[f64]>, defensive_fraction: f64) -> LadduPhysicsResult<Self> {
        let total: f64 = counts.iter().sum();
        if counts.is_empty()
            || counts
                .iter()
                .any(|count| !count.is_finite() || *count < 0.0)
            || !total.is_finite()
            || total <= 0.0
            || !defensive_fraction.is_finite()
            || !(0.0..=1.0).contains(&defensive_fraction)
        {
            return Err(LadduPhysicsError::invalid_relation(
                "adaptive decay requires nonnegative finite counts with positive total and a defensive fraction in [0, 1]",
            ));
        }
        let density = PiecewiseDensity::uniform(-1.0, 1.0, counts).map_err(|_| LadduPhysicsError::invalid_relation("adaptive decay requires nonnegative finite counts with positive total and a defensive fraction in [0, 1]"))?;
        Ok(Self {
            density,
            defensive_fraction,
        })
    }
    fn sample_costheta(&self, rng: &mut ProposalRng) -> (f64, f64) {
        let costheta = if rng.uniform() < self.defensive_fraction {
            2.0 * rng.uniform() - 1.0
        } else {
            self.density
                .sample_indexed(rng)
                .expect("validated adaptive density is sampleable")
        };
        let learned_density = self.density.density(-1.0, 1.0, costheta);
        (
            costheta,
            self.defensive_fraction * 0.5 + (1.0 - self.defensive_fraction) * learned_density,
        )
    }
    /// Propose a two-body decay from the adapted angular density.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the vertex is not a one-to-two decay
    /// or its masses and momenta are not physically valid.
    pub fn propose(
        &self,
        incoming: &[NamedMomentum<'_>],
        outgoing: &[NamedMass<'_>],
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<ProposalResult> {
        let kinematics = TwoBodyDecayKinematics::new("adaptive", incoming, outgoing)?;
        let (cos_theta, density) = self.sample_costheta(rng);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * rng.uniform();
        let direction = RealVec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
        kinematics.finish(direction, 0.5 / density)
    }
}

pub(super) fn two_body_momentum(parent: f64, first: f64, second: f64) -> LadduPhysicsResult<f64> {
    if !parent.is_finite()
        || !first.is_finite()
        || !second.is_finite()
        || parent <= 0.0
        || first < 0.0
        || second < 0.0
    {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "two-body masses must be finite, with a positive parent and nonnegative daughters; got parent={parent}, first={first}, second={second}"
        )));
    }
    if parent < first + second {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "two-body threshold is closed: {parent} < {}",
            first + second
        )));
    }
    let lambda =
        (parent * parent - (first + second).powi(2)) * (parent * parent - (first - second).powi(2));
    Ok(lambda.max(0.0).sqrt() / (2.0 * parent))
}

/// Enclose the phase-space correction for an isotropic two-body decay.
///
/// This is public only for the workspace generation crate's proven-envelope
/// path. The interval contains the correction for every kinematically defined
/// combination of masses in the supplied box.
#[doc(hidden)]
pub fn proven_two_body_decay_weight(
    parent: Interval,
    first: Interval,
    second: Interval,
) -> Interval {
    let sum = first + second;
    let difference = first - second;
    let parent_squared = parent.sqr();
    let radicand = (parent_squared - sum.sqr()) * (parent_squared - difference.sqr());
    let momentum = radicand.sqrt() / (2.0 * parent);
    let result = momentum / (4.0 * PI * parent);
    Interval::new(0.0, result.sup())
}

pub(super) fn on_shell(direction: RealVec3, momentum: f64, mass: f64) -> RealVec4 {
    (direction * momentum).with_mass(mass)
}
