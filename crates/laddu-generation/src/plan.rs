use laddu_core::{
    reaction::Endpoint, Channel, LadduError, LadduResult, MassSampler, MomentumSource, Particle,
    ScalarDistribution, VertexGenerator,
};

/// A validated channel topology that can be used by the current generator.
#[derive(Clone, Debug)]
pub struct GenerationPlan {
    production: ProductionPlan,
}

impl GenerationPlan {
    /// Validate a channel and build a generation plan for the supported topology.
    pub fn from_channel(channel: &Channel) -> LadduResult<Self> {
        let generated_vertices = channel
            .vertices()
            .iter()
            .filter(|vertex| vertex.generation().is_some())
            .collect::<Vec<_>>();
        let [production_vertex] = generated_vertices.as_slice() else {
            return Err(LadduError::Custom(format!(
                "generation requires exactly one generated production vertex, found {}",
                generated_vertices.len()
            )));
        };

        let VertexGenerator::TwoToTwo { t } = production_vertex
            .generation()
            .expect("filtered to generated vertices");

        let incoming = channel.incoming_particles(production_vertex.label())?;
        let outgoing = channel.outgoing_particles(production_vertex.label())?;
        if incoming.len() != 2 || outgoing.len() != 2 {
            return Err(LadduError::Custom(format!(
                "generated production vertex '{}' must be a two-to-two vertex",
                production_vertex.label()
            )));
        }

        let initial = [
            InitialParticlePlan::from_particle(incoming[0])?,
            InitialParticlePlan::from_particle(incoming[1])?,
        ];
        let final_state = [
            DecayParticlePlan::from_channel(channel, outgoing[0])?,
            DecayParticlePlan::from_channel(channel, outgoing[1])?,
        ];

        Ok(Self {
            production: ProductionPlan {
                vertex: production_vertex.label().to_string(),
                incoming: initial,
                outgoing: final_state,
                t: t.clone(),
            },
        })
    }

    /// Return the validated production plan.
    pub fn production(&self) -> &ProductionPlan {
        &self.production
    }
}

/// A validated two-to-two production vertex.
#[derive(Clone, Debug)]
pub struct ProductionPlan {
    vertex: String,
    incoming: [InitialParticlePlan; 2],
    outgoing: [DecayParticlePlan; 2],
    t: ScalarDistribution,
}

impl ProductionPlan {
    /// Return the production vertex label.
    pub fn vertex(&self) -> &str {
        &self.vertex
    }

    /// Return the two incoming particle plans.
    pub fn incoming(&self) -> &[InitialParticlePlan; 2] {
        &self.incoming
    }

    /// Return the two outgoing particle plans.
    pub fn outgoing(&self) -> &[DecayParticlePlan; 2] {
        &self.outgoing
    }

    /// Return the Mandelstam-t distribution.
    pub fn t_distribution(&self) -> &ScalarDistribution {
        &self.t
    }
}

/// A validated initial-state particle.
#[derive(Clone, Debug)]
pub struct InitialParticlePlan {
    label: String,
    mass: f64,
    momentum: MomentumSource,
}

impl InitialParticlePlan {
    fn from_particle(particle: &Particle) -> LadduResult<Self> {
        if !matches!(particle.from(), Endpoint::ExternalIn) {
            return Err(LadduError::Custom(format!(
                "initial particle '{}' must start from ExternalIn",
                particle.label()
            )));
        }
        if !matches!(particle.generation().mass(), MassSampler::FromProperties) {
            return Err(LadduError::Custom(format!(
                "initial particle '{}' must use its ParticleProperties mass",
                particle.label()
            )));
        }
        let mass = particle.properties().mass().map_err(|_| {
            LadduError::Custom(format!(
                "initial particle '{}' needs a ParticleProperties mass for generation",
                particle.label()
            ))
        })?;
        let momentum = particle.generation().momentum().cloned().ok_or_else(|| {
            LadduError::Custom(format!(
                "initial particle '{}' needs a momentum source for generation",
                particle.label()
            ))
        })?;
        Ok(Self {
            label: particle.label().to_string(),
            mass,
            momentum,
        })
    }

    /// Return the particle label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return the particle mass from properties.
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Return the momentum source.
    pub fn momentum(&self) -> &MomentumSource {
        &self.momentum
    }
}

/// A validated generated particle downstream of production.
#[derive(Clone, Debug)]
pub struct DecayParticlePlan {
    label: String,
    mass: PlannedMass,
    decay: Option<Box<DecayPlan>>,
}

impl DecayParticlePlan {
    fn from_channel(channel: &Channel, particle: &Particle) -> LadduResult<Self> {
        if matches!(particle.from(), Endpoint::ExternalIn) {
            return Err(LadduError::Custom(format!(
                "generated particle '{}' cannot start from ExternalIn",
                particle.label()
            )));
        }
        let mass = PlannedMass::from_particle(particle)?;
        let decay_vertices = channel.decay_vertices(particle.label())?;
        let decay = match decay_vertices.as_slice() {
            [] => None,
            [vertex] => {
                let incoming = channel.incoming_particles(vertex.label())?;
                let outgoing = channel.outgoing_particles(vertex.label())?;
                if incoming.len() != 1 || outgoing.len() != 2 {
                    return Err(LadduError::Custom(format!(
                        "generated decay vertex '{}' must be one-to-two",
                        vertex.label()
                    )));
                }
                Some(Box::new(DecayPlan {
                    vertex: vertex.label().to_string(),
                    daughters: [
                        DecayParticlePlan::from_channel(channel, outgoing[0])?,
                        DecayParticlePlan::from_channel(channel, outgoing[1])?,
                    ],
                }))
            }
            _ => {
                return Err(LadduError::Custom(format!(
                    "generated particle '{}' has multiple decay vertices",
                    particle.label()
                )));
            }
        };

        Ok(Self {
            label: particle.label().to_string(),
            mass,
            decay,
        })
    }

    /// Return the particle label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return the planned mass source.
    pub fn mass(&self) -> &PlannedMass {
        &self.mass
    }

    /// Return the one-to-two decay plan, if this particle decays.
    pub fn decay(&self) -> Option<&DecayPlan> {
        self.decay.as_deref()
    }
}

/// A planned mass source after validation.
#[derive(Clone, Debug)]
pub enum PlannedMass {
    /// Use this fixed mass from ParticleProperties.
    Properties(f64),
    /// Sample this mass from an explicit distribution.
    Sampled(ScalarDistribution),
}

impl PlannedMass {
    fn from_particle(particle: &Particle) -> LadduResult<Self> {
        match particle.generation().mass() {
            MassSampler::FromProperties => particle
                .properties()
                .mass()
                .map(Self::Properties)
                .map_err(|_| {
                    LadduError::Custom(format!(
                        "generated particle '{}' needs a ParticleProperties mass or sampled mass",
                        particle.label()
                    ))
                }),
            MassSampler::Sampled(distribution) => Ok(Self::Sampled(distribution.clone())),
        }
    }
}

/// A validated one-to-two decay vertex.
#[derive(Clone, Debug)]
pub struct DecayPlan {
    vertex: String,
    daughters: [DecayParticlePlan; 2],
}

impl DecayPlan {
    /// Return the decay vertex label.
    pub fn vertex(&self) -> &str {
        &self.vertex
    }

    /// Return the daughter particle plans.
    pub fn daughters(&self) -> &[DecayParticlePlan; 2] {
        &self.daughters
    }
}
