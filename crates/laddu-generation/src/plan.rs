use std::collections::HashSet;

use laddu_core::{
    reaction::Endpoint, Channel, LadduError, LadduResult, MassSampler, MomentumSource, Particle,
    ParticleProperties, ScalarDistribution, VertexGenerator,
};

use crate::sink::GeneratedAuxInfo;

/// A validated channel topology that can be used by the current generator.
#[derive(Clone, Debug)]
pub struct GenerationPlan {
    production: ProductionPlan,
    aux: Vec<GeneratedAuxInfo>,
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
        validate_supported_topology(channel, production_vertex.label())?;

        Ok(Self {
            production: ProductionPlan {
                vertex: production_vertex.label().to_string(),
                incoming: initial,
                outgoing: final_state,
                t: t.clone(),
            },
            aux: Vec::default(),
        })
    }

    pub fn with_aux(mut self, label: &str, generator: ScalarDistribution) -> Self {
        self.aux.push(GeneratedAuxInfo {
            label: label.to_string(),
            generator,
        });
        self
    }

    /// Return the validated production plan.
    pub fn production(&self) -> &ProductionPlan {
        &self.production
    }

    /// Return the validated aux plan.
    pub fn aux_info(&self) -> &[GeneratedAuxInfo] {
        &self.aux
    }
}

fn validate_supported_topology(channel: &Channel, production_vertex: &str) -> LadduResult<()> {
    let mut vertices = HashSet::new();
    let mut particles = HashSet::new();
    collect_generated_topology(channel, production_vertex, &mut vertices, &mut particles)?;

    if let Some(vertex) = channel
        .vertices()
        .iter()
        .find(|vertex| !vertices.contains(vertex.label()))
    {
        return Err(LadduError::Custom(format!(
            "generation does not support unused or disconnected vertex '{}'",
            vertex.label()
        )));
    }
    if let Some(particle) = channel
        .particles()
        .iter()
        .find(|particle| !particles.contains(particle.label()))
    {
        return Err(LadduError::Custom(format!(
            "generation does not support unused or disconnected particle '{}'",
            particle.label()
        )));
    }
    Ok(())
}

fn collect_generated_topology(
    channel: &Channel,
    vertex: &str,
    vertices: &mut HashSet<String>,
    particles: &mut HashSet<String>,
) -> LadduResult<()> {
    vertices.insert(vertex.to_string());
    let incoming = channel.incoming_particles(vertex)?;
    let outgoing = channel.outgoing_particles(vertex)?;
    for particle in incoming.iter().chain(outgoing.iter()) {
        particles.insert(particle.label().to_string());
    }
    for particle in outgoing {
        for decay in channel.decay_vertices(particle.label())? {
            collect_generated_topology(channel, decay.label(), vertices, particles)?;
        }
    }
    Ok(())
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
    properties: ParticleProperties,
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
            properties: particle.properties().clone(),
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

    /// Return the particle properties.
    pub fn properties(&self) -> &ParticleProperties {
        &self.properties
    }
}

/// A validated generated particle downstream of production.
#[derive(Clone, Debug)]
pub struct DecayParticlePlan {
    label: String,
    mass: PlannedMass,
    properties: ParticleProperties,
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
            properties: particle.properties().clone(),
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

    /// Return the particle properties.
    pub fn properties(&self) -> &ParticleProperties {
        &self.properties
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

#[derive(Clone, Debug, Default)]
pub struct AuxPlan {}

#[cfg(test)]
mod tests {
    use laddu_core::{Channel, ParticleProperties};

    use super::*;

    fn generator() -> VertexGenerator {
        VertexGenerator::TwoToTwo {
            t: ScalarDistribution::Exponential { slope: 0.1 },
        }
    }

    fn base_channel() -> Channel {
        let mut channel = Channel::new();
        channel
            .create_production("production", ["beam", "target"], ["res", "recoil"])
            .unwrap()
            .generate(generator());
        channel
            .create_decay("res_decay", "res", ["a", "b"])
            .unwrap();
        channel
            .edit_particle("beam")
            .unwrap()
            .mass(0.0)
            .momentum(MomentumSource::FromEnergy(ScalarDistribution::Fixed(8.0)));
        channel
            .edit_particle("target")
            .unwrap()
            .mass(0.938272)
            .momentum(MomentumSource::AtRest);
        channel
            .edit_particle("res")
            .unwrap()
            .mass_sampler(MassSampler::Sampled(ScalarDistribution::Uniform {
                min: 1.1,
                max: 1.6,
            }));
        for label in ["a", "b"] {
            channel.edit_particle(label).unwrap().mass(0.497611);
        }
        channel.edit_particle("recoil").unwrap().mass(0.938272);
        channel
    }

    fn plan_error(channel: &Channel) -> String {
        GenerationPlan::from_channel(channel)
            .expect_err("channel should be rejected")
            .to_string()
    }

    #[test]
    fn accepts_single_generated_production_with_binary_decay_chain() {
        let plan = GenerationPlan::from_channel(&base_channel()).unwrap();
        assert_eq!(plan.production().vertex(), "production");
        assert_eq!(plan.production().incoming()[0].label(), "beam");
        assert_eq!(plan.production().outgoing()[0].label(), "res");
        assert_eq!(
            plan.production().outgoing()[0].decay().unwrap().daughters()[0].label(),
            "a"
        );
    }

    #[test]
    fn rejects_missing_generated_production_annotation() {
        let mut channel = Channel::new();
        channel
            .create_production("production", ["beam", "target"], ["res", "recoil"])
            .unwrap();

        let err = plan_error(&channel);
        assert!(err.contains("exactly one generated production vertex"));
    }

    #[test]
    fn rejects_multiple_generated_vertices() {
        let mut channel = base_channel();
        channel
            .edit_vertex("res_decay")
            .unwrap()
            .generate(generator());

        let err = plan_error(&channel);
        assert!(err.contains("exactly one generated production vertex"));
    }

    #[test]
    fn rejects_generated_vertex_that_is_not_two_to_two() {
        let mut channel = Channel::new();
        channel
            .create_decay("bad", "res", ["a", "b"])
            .unwrap()
            .generate(generator());

        let err = plan_error(&channel);
        assert!(err.contains("must be a two-to-two vertex"));
    }

    #[test]
    fn rejects_non_binary_decay_downstream() {
        let mut channel = Channel::new();
        channel
            .create_production("production", ["beam", "target"], ["res", "recoil"])
            .unwrap()
            .generate(generator());
        channel
            .create_vertex("res_decay", ["res"], ["a", "b", "c"])
            .unwrap();
        channel
            .edit_particle("beam")
            .unwrap()
            .mass(0.0)
            .momentum(MomentumSource::FromEnergy(ScalarDistribution::Fixed(8.0)));
        channel
            .edit_particle("target")
            .unwrap()
            .mass(0.938272)
            .momentum(MomentumSource::AtRest);
        channel
            .edit_particle("res")
            .unwrap()
            .mass_sampler(MassSampler::Sampled(ScalarDistribution::Uniform {
                min: 1.1,
                max: 1.6,
            }));
        channel.edit_particle("recoil").unwrap().mass(0.938272);

        let err = plan_error(&channel);
        assert!(err.contains("must be one-to-two"));
    }

    #[test]
    fn rejects_disconnected_topology() {
        let mut channel = base_channel();
        channel
            .create_decay("unused_decay", "unused_parent", ["unused_a", "unused_b"])
            .unwrap();

        let err = plan_error(&channel);
        assert!(err.contains("unused or disconnected vertex 'unused_decay'"));
    }

    #[test]
    fn rejects_initial_sampled_mass() {
        let mut channel = base_channel();
        channel
            .edit_particle("beam")
            .unwrap()
            .mass_sampler(MassSampler::Sampled(ScalarDistribution::Uniform {
                min: 0.0,
                max: 1.0,
            }));

        let err = plan_error(&channel);
        assert!(err.contains("initial particle 'beam' must use its ParticleProperties mass"));
    }

    #[test]
    fn rejects_initial_particle_without_mass() {
        let mut channel = base_channel();
        channel
            .edit_particle("beam")
            .unwrap()
            .properties(ParticleProperties::unknown())
            .momentum(MomentumSource::FromEnergy(ScalarDistribution::Fixed(8.0)));

        let err = plan_error(&channel);
        assert!(err.contains("initial particle 'beam' needs a ParticleProperties mass"));
    }

    #[test]
    fn rejects_initial_particle_without_momentum_source() {
        let mut channel = base_channel();
        channel
            .edit_particle("beam")
            .unwrap()
            .generation(laddu_core::ParticleGeneration::default());

        let err = plan_error(&channel);
        assert!(err.contains("initial particle 'beam' needs a momentum source"));
    }

    #[test]
    fn rejects_generated_particle_without_mass_or_sampler() {
        let mut channel = base_channel();
        channel
            .edit_particle("recoil")
            .unwrap()
            .properties(ParticleProperties::unknown());

        let err = plan_error(&channel);
        assert!(err.contains("generated particle 'recoil' needs a ParticleProperties mass"));
    }
}
