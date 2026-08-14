/// A four-momentum source attached to an initial channel edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InitialMomentum {
    /// Use a fixed four-momentum directly.
    P4(RealVec4),
    /// Use a fixed three-momentum and derive the energy from the particle mass.
    Momentum(RealVec3),
    /// Sample the energy and orient the momentum along a fixed direction.
    EnergyDirection {
        /// Energy source.
        energy: ScalarSource,
        /// Fixed direction of the initial momentum.
        direction: RealVec3,
    },
}

/// A sampled initial four-momentum and its inverse proposal density.
#[derive(Clone, Copy, Debug)]
pub struct InitialMomentumResult {
    /// Sampled on-shell four-momentum in `(E, px, py, pz)` order.
    pub p4: RealVec4,
    /// Inverse proposal-density correction.
    pub weight: f64,
}

impl InitialMomentum {
    /// Construct a fixed four-momentum source.
    pub fn p4(p4: RealVec4) -> Self {
        Self::P4(p4)
    }
    /// Construct a source from a fixed three-momentum and particle mass.
    pub fn momentum(momentum: RealVec3) -> Self {
        Self::Momentum(momentum)
    }
    /// Construct a fixed-energy source along a direction.
    pub fn energy_direction(energy: f64, direction: RealVec3) -> Self {
        Self::EnergyDirection {
            energy: ScalarSource::constant(energy),
            direction,
        }
    }
    /// Construct a sampled-energy source along a direction.
    pub fn energy_source_direction(energy: ScalarSource, direction: RealVec3) -> Self {
        Self::EnergyDirection { energy, direction }
    }

    /// Validate this source against an edge name and particle definition.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when particle mass metadata is missing,
    /// momentum components are invalid or off shell, energy support is below
    /// threshold, or the direction cannot be normalized.
    pub fn validate(
        &self,
        edge: &str,
        properties: Option<&ParticleProperties>,
    ) -> LadduPhysicsResult<()> {
        match self {
            Self::P4(p4) => {
                let mass = particle_mass(edge, properties)?;
                if ![p4.px(), p4.py(), p4.pz(), p4.e()]
                    .into_iter()
                    .all(f64::is_finite)
                    || p4.e() < 0.0
                {
                    return Err(LadduPhysicsError::invalid_value(
                        format!("initial four-momentum for edge `{edge}`"),
                        "finite components and nonnegative energy",
                        p4,
                    ));
                }
                let tolerance = 1e-9 * (1.0 + mass * mass + p4.e() * p4.e());
                if (p4.m2() - mass * mass).abs() > tolerance {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "initial edge `{edge}` is off shell: p²={} but mass²={}",
                        p4.m2(),
                        mass * mass
                    )));
                }
            }
            Self::Momentum(momentum) => {
                particle_mass(edge, properties)?;
                if ![momentum.px(), momentum.py(), momentum.pz()]
                    .into_iter()
                    .all(f64::is_finite)
                {
                    return Err(LadduPhysicsError::invalid_value(
                        format!("initial momentum for edge `{edge}`"),
                        "finite components",
                        momentum,
                    ));
                }
            }
            Self::EnergyDirection { energy, direction } => {
                let mass = particle_mass(edge, properties)?;
                let (minimum, _) = energy.support()?;
                if minimum < mass {
                    return Err(LadduPhysicsError::invalid_value(
                        format!("energy support for initial edge `{edge}`"),
                        format!("entirely at or above its particle mass {mass}"),
                        minimum,
                    ));
                }
                direction.unit()?;
            }
        }
        Ok(())
    }

    /// Draw an initial four-momentum after validating its particle definition.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when source validation fails or a sampled
    /// value cannot produce a physical on-shell momentum.
    pub fn sample(
        &self,
        edge: &str,
        properties: Option<&ParticleProperties>,
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<InitialMomentumResult> {
        self.validate(edge, properties)?;
        self.sample_prevalidated(particle_mass(edge, properties)?, rng)
    }

    /// Sample after channel validation has already established source invariants.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a scalar source cannot be sampled or
    /// the supplied mass and sampled energy do not define a physical momentum.
    #[doc(hidden)]
    pub fn sample_prevalidated(
        &self,
        mass: f64,
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<InitialMomentumResult> {
        match self {
            Self::P4(p4) => Ok(InitialMomentumResult {
                p4: *p4,
                weight: 1.0,
            }),
            Self::Momentum(momentum) => Ok(InitialMomentumResult {
                p4: momentum.with_mass(mass),
                weight: 1.0,
            }),
            Self::EnergyDirection { energy, direction } => {
                let sampled = energy.sample(rng)?;
                if sampled.value < mass {
                    return Err(LadduPhysicsError::invalid_value(
                        "sampled initial-state energy",
                        format!("at or above the particle mass {mass}"),
                        sampled.value,
                    ));
                }
                let momentum =
                    direction.unit()? * (sampled.value * sampled.value - mass * mass).sqrt();
                Ok(InitialMomentumResult {
                    p4: momentum.with_energy(sampled.value),
                    weight: sampled.weight,
                })
            }
        }
    }
}

fn particle_mass(edge: &str, properties: Option<&ParticleProperties>) -> LadduPhysicsResult<f64> {
    properties
        .ok_or_else(|| {
            LadduPhysicsError::invalid_relation(format!(
                "initial edge `{edge}` has no particle properties"
            ))
        })?
        .mass()
}
