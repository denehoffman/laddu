use density::{FiniteInterval, Sample};

#[derive(Clone, Copy, Debug)]
/// Mass and importance weight drawn from a [`MassProposal`].
pub struct MassProposalResult {
    /// Proposed invariant mass.
    pub mass: f64,
    /// Inverse proposal-density correction.
    pub weight: f64,
}

/// Invariant-mass proposal for a generated edge.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MassProposal {
    /// Always use one invariant mass.
    Fixed {
        /// Fixed invariant mass.
        mass: f64,
    },
    /// Sample uniformly between the given bounds, clipped to kinematic support.
    Uniform {
        /// Lower proposal bound.
        low: f64,
        /// Upper proposal bound.
        high: f64,
    },
}

impl MassProposal {
    /// Construct a fixed-mass proposal.
    pub fn fixed(mass: f64) -> Self {
        Self::Fixed { mass }
    }
    /// Construct a uniform mass proposal.
    pub fn uniform(low: f64, high: f64) -> Self {
        Self::Uniform { low, high }
    }

    /// Draw a mass within the supplied kinematic interval.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the kinematic interval or proposal
    /// bounds are non-finite, empty, or exclude a fixed mass.
    pub fn propose(
        &self,
        minimum: f64,
        maximum: f64,
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<MassProposalResult> {
        match *self {
            Self::Fixed { mass } => {
                if !mass.is_finite() || mass < minimum || mass > maximum {
                    return Err(LadduPhysicsError::invalid_value(
                        "fixed mass",
                        format!("a finite value in [{minimum}, {maximum}]"),
                        mass,
                    ));
                }
                Ok(MassProposalResult { mass, weight: 1.0 })
            }
            Self::Uniform { low, high } => {
                let interval = uniform_mass_support(low, high, minimum, maximum)?;
                let sample = Sample {
                    value: interval.low() + rng.uniform() * interval.width(),
                    inverse_density: interval.width(),
                };
                Ok(MassProposalResult {
                    mass: sample.value,
                    weight: sample.inverse_density,
                })
            }
        }
    }

    /// Evaluate the proposal density when it is available.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the kinematic interval and uniform
    /// proposal have no valid finite overlap.
    pub fn density(
        &self,
        minimum: f64,
        maximum: f64,
        mass: f64,
    ) -> LadduPhysicsResult<Option<f64>> {
        match *self {
            Self::Fixed { .. } => Ok(None),
            Self::Uniform { low, high } => {
                let interval = uniform_mass_support(low, high, minimum, maximum)?;
                Ok(Some(if interval.contains(mass) {
                    interval.width().recip()
                } else {
                    0.0
                }))
            }
        }
    }
}

impl From<f64> for MassProposal {
    fn from(mass: f64) -> Self {
        Self::fixed(mass)
    }
}

impl From<std::ops::Range<f64>> for MassProposal {
    fn from(range: std::ops::Range<f64>) -> Self {
        Self::uniform(range.start, range.end)
    }
}

fn uniform_mass_support(
    proposal_low: f64,
    proposal_high: f64,
    minimum: f64,
    maximum: f64,
) -> LadduPhysicsResult<FiniteInterval> {
    if !proposal_low.is_finite() || !proposal_high.is_finite() || proposal_high <= proposal_low {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "uniform mass proposal requires finite low < high, got [{proposal_low}, {proposal_high}]"
        )));
    }
    let proposal = FiniteInterval::new(proposal_low, proposal_high)
        .expect("proposal bounds were validated above");
    let Some(allowed) = FiniteInterval::new(minimum, maximum) else {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "uniform mass support [{proposal_low}, {proposal_high}] does not overlap the allowed interval [{minimum}, {maximum}]"
        )));
    };
    let Some(support) = proposal.intersect(allowed) else {
        return Err(LadduPhysicsError::invalid_relation(format!(
            "uniform mass support [{proposal_low}, {proposal_high}] does not overlap the allowed interval [{minimum}, {maximum}]"
        )));
    };
    Ok(support)
}
