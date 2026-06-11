//! Event generation from validated channel generation plans.

use std::time::Instant;

use fastrand::Rng;
use laddu_core::{
    vectors::{Vec3, Vec4},
    LadduError, LadduResult, LadduRngExt, MomentumSource, ScalarDistribution,
};

use crate::{
    plan::{DecayParticlePlan, GenerationPlan, PlannedMass},
    sink::{
        GeneratedBatchView, GeneratedLayout, GeneratedParticleInfo, GeneratedParticleRole,
        GeneratedRecord, GeneratedSink, GenerationMode, GenerationModeKind, GenerationOptions,
        GenerationResult, GenerationStats,
    },
};

const MAX_EVENT_ATTEMPTS: usize = 10_000;
const MAX_SAMPLE_ATTEMPTS: usize = 10_000;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// A generator for the currently supported channel topology.
///
/// This generator accepts the topology validated by [`GenerationPlan`]: two incoming particles,
/// one generated two-to-two production vertex, and downstream chains of one-to-two decays.
#[derive(Clone, Debug)]
pub struct EventGenerator {
    plan: GenerationPlan,
    seed: Option<u64>,
    p4_labels: Vec<String>,
}

impl EventGenerator {
    /// Construct a generator from an already validated generation plan.
    pub fn new(plan: GenerationPlan) -> Self {
        let p4_labels = p4_labels(&plan);
        Self {
            plan,
            seed: None,
            p4_labels,
        }
    }

    /// Validate a channel and construct a generator from its generation annotations.
    pub fn from_channel(channel: &laddu_core::Channel) -> LadduResult<Self> {
        Ok(Self::new(GenerationPlan::from_channel(channel)?))
    }

    /// Return a copy of this generator with deterministic random seeding.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Return the validated generation plan.
    pub fn plan(&self) -> &GenerationPlan {
        &self.plan
    }

    /// Return generated four-momentum labels in dataset column order.
    pub fn p4_labels(&self) -> &[String] {
        &self.p4_labels
    }

    /// Generate one event using the supplied random-number generator.
    pub fn generate_event(&self, rng: &mut Rng) -> LadduResult<GeneratedEvent> {
        let mut last_error = None;
        for _ in 0..MAX_EVENT_ATTEMPTS {
            match self.try_generate_event(rng) {
                Ok(event) => return Ok(event),
                Err(err) => last_error = Some(err),
            }
        }
        Err(last_error
            .unwrap_or_else(|| LadduError::Custom("failed to generate event".to_string())))
    }

    /// Generate records into a sink.
    pub fn generate<S>(
        &self,
        target_events: usize,
        sink: S,
        mode: GenerationMode,
        options: GenerationOptions,
    ) -> LadduResult<GenerationResult<S::Output>>
    where
        S: GeneratedSink,
    {
        if options.batch_size == 0 {
            return Err(LadduError::Custom(
                "generation batch size must be greater than zero".to_string(),
            ));
        }
        match mode {
            GenerationMode::Raw => self.generate_raw(target_events, sink, options),
        }
    }

    fn rng_with_options(&self, options: &GenerationOptions) -> Rng {
        match options.seed.or(self.seed) {
            Some(seed) => Rng::with_seed(seed),
            None => Rng::new(),
        }
    }

    fn generate_raw<S>(
        &self,
        target_events: usize,
        mut sink: S,
        options: GenerationOptions,
    ) -> LadduResult<GenerationResult<S::Output>>
    where
        S: GeneratedSink,
    {
        let started = Instant::now();
        let layout = generated_layout(&self.plan);
        sink.begin(&layout)?;
        let mut rng = self.rng_with_options(&options);
        let mut records = Vec::with_capacity(options.batch_size.min(target_events));
        let mut batches_written = 0_u64;
        let mut sum_weights = 0.0;
        let mut min_weight = None::<f64>;
        let mut max_weight = None::<f64>;

        for local_index in 0..target_events {
            let event = self.generate_event(&mut rng)?;
            let record = self.record_from_event(local_index as u64, event, 1.0)?;
            sum_weights += record.weight;
            min_weight = Some(min_weight.map_or(record.weight, |value| value.min(record.weight)));
            max_weight = Some(max_weight.map_or(record.weight, |value| value.max(record.weight)));
            records.push(record);
            if records.len() == options.batch_size {
                sink.push_batch(GeneratedBatchView {
                    layout: &layout,
                    records: &records,
                })?;
                batches_written += 1;
                records.clear();
            }
        }
        if !records.is_empty() {
            sink.push_batch(GeneratedBatchView {
                layout: &layout,
                records: &records,
            })?;
            batches_written += 1;
        }

        let output = sink.finish()?;
        let target_events = target_events as u64;
        Ok(GenerationResult {
            output,
            stats: GenerationStats {
                mode: GenerationModeKind::Raw,
                target_events,
                written_events: target_events,
                proposed_events: target_events,
                accepted_events: target_events,
                rejected_events: 0,
                acceptance_rate: None,
                envelope: None,
                envelope_violations: 0,
                sum_weights,
                min_weight,
                max_weight,
                batches_written,
                elapsed: started.elapsed(),
                seed: options.seed.or(self.seed),
            },
        })
    }

    fn try_generate_event(&self, rng: &mut Rng) -> LadduResult<GeneratedEvent> {
        let production = self.plan.production();
        let incoming = production.incoming();
        let outgoing = production.outgoing();

        let p1 = initial_p4(incoming[0].mass(), incoming[0].momentum(), 1.0, rng)?;
        let p2 = initial_p4(incoming[1].mass(), incoming[1].momentum(), -1.0, rng)?;
        let m3 = sample_mass(outgoing[0].mass(), rng);
        let m4 = sample_mass(outgoing[1].mass(), rng);
        let (p3, p4) = generate_production_pair(p1, p2, m3, m4, production.t_distribution(), rng)?;

        let mut p4s = Vec::new();
        p4s.push((incoming[0].label().to_string(), p1));
        p4s.push((incoming[1].label().to_string(), p2));
        generate_decay_chain(&outgoing[0], p3, &mut p4s, rng)?;
        generate_decay_chain(&outgoing[1], p4, &mut p4s, rng)?;
        Ok(GeneratedEvent { p4s })
    }

    fn record_from_event(
        &self,
        local_index: u64,
        event: GeneratedEvent,
        weight: f64,
    ) -> LadduResult<GeneratedRecord> {
        let p4s = self
            .p4_labels
            .iter()
            .map(|label| {
                event.p4(label).ok_or_else(|| {
                    LadduError::Custom(format!("generated event is missing p4 '{label}'"))
                })
            })
            .collect::<LadduResult<Vec<_>>>()?;
        Ok(GeneratedRecord {
            local_index,
            global_index: Some(local_index),
            weight,
            p4s,
        })
    }
}

/// One generated event keyed by channel particle label.
#[derive(Clone, Debug)]
pub struct GeneratedEvent {
    p4s: Vec<(String, Vec4)>,
}

impl GeneratedEvent {
    /// Return the generated four-momentum labels in insertion order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.p4s.iter().map(|(label, _)| label.as_str())
    }

    /// Return the generated four-momentum for a label.
    pub fn p4(&self, label: &str) -> Option<Vec4> {
        self.p4s
            .iter()
            .find_map(|(candidate, p4)| (candidate == label).then_some(*p4))
    }

    /// Return all generated four-momenta.
    pub fn p4s(&self) -> &[(String, Vec4)] {
        &self.p4s
    }
}

fn p4_labels(plan: &GenerationPlan) -> Vec<String> {
    let mut labels = Vec::new();
    for particle in plan.production().incoming() {
        push_label(&mut labels, particle.label());
    }
    for particle in plan.production().outgoing() {
        collect_decay_labels(particle, &mut labels);
    }
    labels
}

fn collect_decay_labels(particle: &DecayParticlePlan, labels: &mut Vec<String>) {
    push_label(labels, particle.label());
    if let Some(decay) = particle.decay() {
        for daughter in decay.daughters() {
            collect_decay_labels(daughter, labels);
        }
    }
}

fn push_label(labels: &mut Vec<String>, label: &str) {
    if !labels.iter().any(|existing| existing == label) {
        labels.push(label.to_string());
    }
}

fn generated_layout(plan: &GenerationPlan) -> GeneratedLayout {
    let mut particles = Vec::new();
    for particle in plan.production().incoming() {
        push_particle_info(
            &mut particles,
            GeneratedParticleInfo::new(
                particle.label(),
                GeneratedParticleRole::Initial,
                particle.properties().clone(),
            ),
        );
    }
    for particle in plan.production().outgoing() {
        collect_decay_particle_info(particle, &mut particles);
    }
    for (index, particle) in particles.iter_mut().enumerate() {
        particle.output_index = Some(index);
    }
    GeneratedLayout::new(particles)
}

fn collect_decay_particle_info(
    particle: &DecayParticlePlan,
    particles: &mut Vec<GeneratedParticleInfo>,
) {
    let role = if particle.decay().is_some() {
        GeneratedParticleRole::Intermediate
    } else {
        GeneratedParticleRole::Final
    };
    push_particle_info(
        particles,
        GeneratedParticleInfo::new(particle.label(), role, particle.properties().clone()),
    );
    if let Some(decay) = particle.decay() {
        for daughter in decay.daughters() {
            collect_decay_particle_info(daughter, particles);
        }
    }
}

fn push_particle_info(particles: &mut Vec<GeneratedParticleInfo>, particle: GeneratedParticleInfo) {
    if !particles
        .iter()
        .any(|existing| existing.label == particle.label)
    {
        particles.push(particle);
    }
}

fn initial_p4(
    mass: f64,
    momentum: &MomentumSource,
    beam_direction: f64,
    rng: &mut Rng,
) -> LadduResult<Vec4> {
    match momentum {
        MomentumSource::AtRest => Ok(Vec3::zero().with_mass(mass)),
        MomentumSource::FromEnergy(distribution) => {
            let energy = distribution.sample(rng);
            if energy < mass {
                return Err(LadduError::Custom(format!(
                    "sampled initial energy {energy} is below mass {mass}"
                )));
            }
            Ok(rng.p4(mass, energy, Vec3::new(0.0, 0.0, beam_direction)))
        }
    }
}

fn generate_production_pair(
    p1_lab: Vec4,
    p2_lab: Vec4,
    m3: f64,
    m4: f64,
    t_distribution: &ScalarDistribution,
    rng: &mut Rng,
) -> LadduResult<(Vec4, Vec4)> {
    let parent_lab = p1_lab + p2_lab;
    let beta = parent_lab.beta();
    let p1_cm = p1_lab.boost(&-beta);
    let parent_mass = invariant_mass(parent_lab, "production system")?;
    let p_in = p1_cm.vec3().mag();
    let p_out = two_body_momentum(parent_mass, m3, m4)?;
    let e1 = p1_cm.e();
    let e3 = (m3 * m3 + p_out * p_out).sqrt();
    let (t_low, t_high) = t_range(p1_lab.m2(), m3 * m3, e1, e3, p_in, p_out);
    let t = sample_t(t_distribution, (t_low, t_high), rng)?;
    let denominator = 2.0 * p_in * p_out;
    let costheta = if denominator.abs() < f64::EPSILON {
        rng.uniform(-1.0, 1.0)
    } else {
        ((t - p1_lab.m2() - m3 * m3 + 2.0 * e1 * e3) / denominator).clamp(-1.0, 1.0)
    };
    let phi = rng.uniform(0.0, TWO_PI);
    let direction = direction_from_angles(costheta, phi);
    let p3_cm = (p_out * direction).with_mass(m3);
    let p4_cm = (-p_out * direction).with_mass(m4);
    Ok((p3_cm.boost(&beta), p4_cm.boost(&beta)))
}

fn generate_decay_chain(
    particle: &DecayParticlePlan,
    p4: Vec4,
    p4s: &mut Vec<(String, Vec4)>,
    rng: &mut Rng,
) -> LadduResult<()> {
    p4s.push((particle.label().to_string(), p4));
    let Some(decay) = particle.decay() else {
        return Ok(());
    };
    let daughters = decay.daughters();
    let m1 = sample_mass(daughters[0].mass(), rng);
    let m2 = sample_mass(daughters[1].mass(), rng);
    let parent_mass = invariant_mass(p4, particle.label())?;
    let q = two_body_momentum(parent_mass, m1, m2)?;
    let costheta = rng.uniform(-1.0, 1.0);
    let phi = rng.uniform(0.0, TWO_PI);
    let direction = direction_from_angles(costheta, phi);
    let beta = p4.beta();
    let d1 = (q * direction).with_mass(m1).boost(&beta);
    let d2 = (-q * direction).with_mass(m2).boost(&beta);
    generate_decay_chain(&daughters[0], d1, p4s, rng)?;
    generate_decay_chain(&daughters[1], d2, p4s, rng)?;
    Ok(())
}

fn sample_mass(mass: &PlannedMass, rng: &mut Rng) -> f64 {
    match mass {
        PlannedMass::Properties(value) => *value,
        PlannedMass::Sampled(distribution) => distribution.sample(rng),
    }
}

fn sample_t(
    distribution: &ScalarDistribution,
    range: (f64, f64),
    rng: &mut Rng,
) -> LadduResult<f64> {
    if !range.0.is_finite() || !range.1.is_finite() || range.0 >= range.1 {
        return Err(LadduError::Custom(format!(
            "invalid physical Mandelstam-t range [{}, {}]",
            range.0, range.1
        )));
    }
    if let ScalarDistribution::Exponential { slope } = distribution {
        let width = range.1 - range.0;
        return Ok(range.1 - rng.truncated_exponential(*slope, (0.0, width)));
    }
    for _ in 0..MAX_SAMPLE_ATTEMPTS {
        let value = distribution.sample(rng);
        if value > range.0 && value < range.1 {
            return Ok(value);
        }
    }
    Err(LadduError::Custom(format!(
        "failed to sample Mandelstam-t inside physical range [{}, {}]",
        range.0, range.1
    )))
}

fn t_range(m1_sq: f64, m3_sq: f64, e1: f64, e3: f64, p_in: f64, p_out: f64) -> (f64, f64) {
    let center = m1_sq + m3_sq - 2.0 * e1 * e3;
    let span = 2.0 * p_in * p_out;
    let low = center - span;
    let high = center + span;
    if low <= high {
        (low, high)
    } else {
        (high, low)
    }
}

fn two_body_momentum(parent_mass: f64, m1: f64, m2: f64) -> LadduResult<f64> {
    if parent_mass < m1 + m2 {
        return Err(LadduError::Custom(format!(
            "two-body system with parent mass {parent_mass} cannot produce daughter masses {m1} and {m2}"
        )));
    }
    let s = parent_mass * parent_mass;
    let lambda = (s - (m1 + m2).powi(2)) * (s - (m1 - m2).powi(2));
    if !lambda.is_finite() || lambda < -1.0e-12 {
        return Err(LadduError::Custom(format!(
            "invalid two-body phase-space factor {lambda}"
        )));
    }
    Ok(lambda.max(0.0).sqrt() / (2.0 * parent_mass))
}

fn invariant_mass(p4: Vec4, label: &str) -> LadduResult<f64> {
    let mass_sq = p4.m2();
    if !mass_sq.is_finite() || mass_sq < -1.0e-12 {
        return Err(LadduError::Custom(format!(
            "generated four-momentum '{label}' has invalid mass squared {mass_sq}"
        )));
    }
    Ok(mass_sq.max(0.0).sqrt())
}

fn direction_from_angles(costheta: f64, phi: f64) -> Vec3 {
    let sintheta = (1.0 - costheta * costheta).max(0.0).sqrt();
    Vec3::new(sintheta * phi.cos(), sintheta * phi.sin(), costheta)
}

#[cfg(test)]
mod tests {
    use laddu_core::{Channel, ParticleProperties, VertexGenerator};

    use super::*;
    use crate::{DatasetSink, NullSink};

    fn demo_generator() -> EventGenerator {
        let mut channel = Channel::new();
        channel
            .create_production("production", ["beam", "target"], ["res", "recoil"])
            .unwrap()
            .generate(VertexGenerator::TwoToTwo {
                t: ScalarDistribution::Exponential { slope: 0.1 },
            });
        channel
            .create_decay("res_decay", "res", ["a", "b"])
            .unwrap();
        channel
            .edit_particle("beam")
            .unwrap()
            .properties(ParticleProperties::unknown().with_mass(0.0))
            .momentum(MomentumSource::FromEnergy(ScalarDistribution::Fixed(8.0)));
        channel
            .edit_particle("target")
            .unwrap()
            .properties(ParticleProperties::unknown().with_mass(0.938272))
            .momentum(MomentumSource::AtRest);
        channel
            .edit_particle("res")
            .unwrap()
            .mass_sampler(crate::gen::uniform_mass(1.1, 1.6));
        for label in ["a", "b"] {
            channel
                .edit_particle(label)
                .unwrap()
                .properties(ParticleProperties::unknown().with_mass(0.497611));
        }
        channel
            .edit_particle("recoil")
            .unwrap()
            .properties(ParticleProperties::unknown().with_mass(0.938272));
        EventGenerator::from_channel(&channel)
            .unwrap()
            .with_seed(12345)
    }

    #[test]
    fn generates_named_dataset() {
        let generator = demo_generator();
        let dataset = generator
            .generate(
                4,
                DatasetSink::new(),
                GenerationMode::Raw,
                GenerationOptions::default(),
            )
            .unwrap()
            .output;
        assert_eq!(dataset.n_events(), 4);
        assert_eq!(
            dataset.p4_names(),
            ["beam", "target", "res", "a", "b", "recoil"]
        );
    }

    #[test]
    fn seeded_generation_is_deterministic() {
        let generator = demo_generator();
        let first = generator
            .generate(
                2,
                DatasetSink::new(),
                GenerationMode::Raw,
                GenerationOptions::default(),
            )
            .unwrap()
            .output;
        let second = generator
            .generate(
                2,
                DatasetSink::new(),
                GenerationMode::Raw,
                GenerationOptions::default(),
            )
            .unwrap()
            .output;
        assert_eq!(
            first.event_local(0).unwrap().p4_at(0),
            second.event_local(0).unwrap().p4_at(0)
        );
        assert_eq!(
            first.event_local(1).unwrap().p4_at(2),
            second.event_local(1).unwrap().p4_at(2)
        );
    }

    #[test]
    fn raw_generation_can_write_dataset_sink_with_stats() {
        let generator = demo_generator();
        let result = generator
            .generate(
                4,
                DatasetSink::new(),
                GenerationMode::Raw,
                GenerationOptions::default().batch_size(2),
            )
            .unwrap();
        assert_eq!(result.output.n_events(), 4);
        assert_eq!(
            result.output.p4_names(),
            ["beam", "target", "res", "a", "b", "recoil"]
        );
        assert_eq!(result.stats.target_events, 4);
        assert_eq!(result.stats.written_events, 4);
        assert_eq!(result.stats.proposed_events, 4);
        assert_eq!(result.stats.rejected_events, 0);
        assert_eq!(result.stats.batches_written, 2);
        assert_eq!(result.stats.sum_weights, 4.0);
        assert!(result.stats.audit().contains("Generation audit"));
    }

    #[test]
    fn null_sink_counts_generated_records() {
        let generator = demo_generator();
        let result = generator
            .generate(
                5,
                NullSink::new(),
                GenerationMode::Raw,
                GenerationOptions::default().batch_size(3),
            )
            .unwrap();
        assert_eq!(result.output, 5);
        assert_eq!(result.stats.batches_written, 2);
        assert_eq!(result.stats.accepted_events, 5);
    }

    #[test]
    fn dataset_sink_can_select_final_state() {
        let generator = demo_generator();
        let dataset = generator
            .generate(
                2,
                DatasetSink::new().output(crate::GenerationOutput::final_state()),
                GenerationMode::Raw,
                GenerationOptions::default(),
            )
            .unwrap()
            .output;
        assert_eq!(dataset.p4_names(), ["a", "b", "recoil"]);
    }
}
