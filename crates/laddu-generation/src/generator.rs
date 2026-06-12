//! Event generation from validated channel generation plans.

use std::{sync::Arc, time::Instant};

use fastrand::Rng;
use laddu_core::{
    data::{Dataset, DatasetMetadata, EventData},
    vectors::{Vec3, Vec4},
    Expression, LadduError, LadduResult, LadduRngExt, MomentumSource, ScalarDistribution,
};

use crate::{
    plan::{DecayParticlePlan, GenerationPlan, PlannedMass},
    sink::{
        validate_envelope_value, Envelope, EnvelopeStats, EnvelopeViolationPolicy,
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
            GenerationMode::Weighted {
                expression,
                parameters,
            } => self.generate_weighted(target_events, sink, options, *expression, parameters),
            GenerationMode::Accepted {
                expression,
                parameters,
                envelope,
            } => self.generate_accepted(
                target_events,
                sink,
                options,
                *expression,
                parameters,
                envelope,
            ),
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
                envelope_stats: None,
                sum_weights,
                min_weight,
                max_weight,
                batches_written,
                elapsed: started.elapsed(),
                seed: options.seed.or(self.seed),
            },
        })
    }

    fn generate_weighted<S>(
        &self,
        target_events: usize,
        mut sink: S,
        options: GenerationOptions,
        expression: Expression,
        parameters: Vec<f64>,
    ) -> LadduResult<GenerationResult<S::Output>>
    where
        S: GeneratedSink,
    {
        let started = Instant::now();
        let layout = generated_layout(&self.plan);
        sink.begin(&layout)?;
        let mut rng = self.rng_with_options(&options);
        let mut records = Vec::with_capacity(options.batch_size.min(target_events));
        let mut stats = WeightStats::default();

        for local_index in 0..target_events {
            let event = self.generate_event(&mut rng)?;
            records.push(self.record_from_event(local_index as u64, event, 1.0)?);
            if records.len() == options.batch_size {
                evaluate_record_weights(&layout, &mut records, &expression, &parameters)?;
                stats.observe_batch(&records);
                sink.push_batch(GeneratedBatchView {
                    layout: &layout,
                    records: &records,
                })?;
                records.clear();
            }
        }
        if !records.is_empty() {
            evaluate_record_weights(&layout, &mut records, &expression, &parameters)?;
            stats.observe_batch(&records);
            sink.push_batch(GeneratedBatchView {
                layout: &layout,
                records: &records,
            })?;
        }

        let output = sink.finish()?;
        let target_events = target_events as u64;
        Ok(GenerationResult {
            output,
            stats: GenerationStats {
                mode: GenerationModeKind::Weighted,
                target_events,
                written_events: target_events,
                proposed_events: target_events,
                accepted_events: target_events,
                rejected_events: 0,
                acceptance_rate: None,
                envelope_stats: None,
                sum_weights: stats.sum_weights,
                min_weight: stats.min_weight,
                max_weight: stats.max_weight,
                batches_written: stats.batches_written,
                elapsed: started.elapsed(),
                seed: options.seed.or(self.seed),
            },
        })
    }

    fn generate_accepted<S>(
        &self,
        target_events: usize,
        mut sink: S,
        options: GenerationOptions,
        expression: Expression,
        parameters: Vec<f64>,
        envelope: Envelope,
    ) -> LadduResult<GenerationResult<S::Output>>
    where
        S: GeneratedSink,
    {
        envelope.validate()?;
        let started = Instant::now();
        let layout = generated_layout(&self.plan);
        let mut rng = self.rng_with_options(&options);
        let max_trials = options.max_trials.unwrap_or(u64::MAX);
        let EnvelopeConfiguration {
            active_envelope,
            envelope_stats,
            proposed_events,
        } = self.configure_envelope(
            EnvelopeEstimationContext {
                layout: &layout,
                expression: &expression,
                parameters: &parameters,
                max_trials,
                batch_size: options.batch_size,
            },
            envelope,
            &mut rng,
        )?;
        sink.begin(&layout)?;
        let mut proposal_batch = Vec::with_capacity(options.batch_size);
        let mut accepted_batch = Vec::with_capacity(options.batch_size.min(target_events));
        let mut stats = WeightStats::default();
        let mut active_envelope = active_envelope;
        let mut envelope_stats = envelope_stats;
        let mut proposed_events = proposed_events;
        let mut accepted_events = 0_u64;
        let mut rejected_events = 0_u64;

        while accepted_events < target_events as u64 {
            if proposed_events >= max_trials {
                return Err(LadduError::Custom(format!(
                    "accepted generation reached max_trials {max_trials} before writing {target_events} accepted events"
                )));
            }
            let event = self.generate_event(&mut rng)?;
            proposal_batch.push(self.record_from_event(proposed_events, event, 1.0)?);
            proposed_events += 1;
            if proposal_batch.len() == options.batch_size {
                AcceptedBatchContext {
                    layout: &layout,
                    expression: &expression,
                    parameters: &parameters,
                    active_envelope: &mut active_envelope,
                    envelope_stats: &mut envelope_stats,
                    envelope_violation_policy: options.envelope_violation_policy,
                    rng: &mut rng,
                    accepted_batch: &mut accepted_batch,
                    stats: &mut stats,
                    accepted_events: &mut accepted_events,
                    rejected_events: &mut rejected_events,
                    target_events: target_events as u64,
                    sink: &mut sink,
                }
                .accept_weighted_batch(&mut proposal_batch)?;
            }
        }
        if !proposal_batch.is_empty() {
            AcceptedBatchContext {
                layout: &layout,
                expression: &expression,
                parameters: &parameters,
                active_envelope: &mut active_envelope,
                envelope_stats: &mut envelope_stats,
                envelope_violation_policy: options.envelope_violation_policy,
                rng: &mut rng,
                accepted_batch: &mut accepted_batch,
                stats: &mut stats,
                accepted_events: &mut accepted_events,
                rejected_events: &mut rejected_events,
                target_events: target_events as u64,
                sink: &mut sink,
            }
            .accept_weighted_batch(&mut proposal_batch)?;
        }
        flush_records(&layout, &mut accepted_batch, &mut sink, &mut stats)?;

        let output = sink.finish()?;
        let target_events = target_events as u64;
        let pilot_events = envelope_stats.pilot_events;
        let sampled_proposals = proposed_events.saturating_sub(pilot_events);
        Ok(GenerationResult {
            output,
            stats: GenerationStats {
                mode: GenerationModeKind::Accepted,
                target_events,
                written_events: accepted_events,
                proposed_events,
                accepted_events,
                rejected_events,
                acceptance_rate: (sampled_proposals > 0)
                    .then_some(accepted_events as f64 / sampled_proposals as f64),
                envelope_stats: Some(envelope_stats),
                sum_weights: stats.sum_weights,
                min_weight: stats.min_weight,
                max_weight: stats.max_weight,
                batches_written: stats.batches_written,
                elapsed: started.elapsed(),
                seed: options.seed.or(self.seed),
            },
        })
    }

    fn configure_envelope(
        &self,
        context: EnvelopeEstimationContext<'_>,
        envelope: Envelope,
        rng: &mut Rng,
    ) -> LadduResult<EnvelopeConfiguration> {
        match envelope {
            Envelope::Initial(value) => {
                validate_envelope_value(value)?;
                Ok(EnvelopeConfiguration {
                    active_envelope: value,
                    envelope_stats: EnvelopeStats::initial(value),
                    proposed_events: 0,
                })
            }
            Envelope::Estimate {
                pilot_events,
                safety_factor,
            } => {
                if pilot_events as u64 > context.max_trials {
                    return Err(LadduError::Custom(format!(
                        "envelope estimation requires {pilot_events} pilot events but max_trials is {}",
                        context.max_trials
                    )));
                }
                let pilot_observed_max = self.estimate_envelope_max(
                    context.layout,
                    context.expression,
                    context.parameters,
                    rng,
                    pilot_events,
                    context.batch_size,
                )?;
                let active_envelope = pilot_observed_max * safety_factor;
                validate_envelope_value(active_envelope)?;
                Ok(EnvelopeConfiguration {
                    active_envelope,
                    envelope_stats: EnvelopeStats::estimated(
                        pilot_events as u64,
                        pilot_observed_max,
                        safety_factor,
                        active_envelope,
                    ),
                    proposed_events: pilot_events as u64,
                })
            }
        }
    }

    fn estimate_envelope_max(
        &self,
        layout: &GeneratedLayout,
        expression: &Expression,
        parameters: &[f64],
        rng: &mut Rng,
        pilot_events: usize,
        batch_size: usize,
    ) -> LadduResult<f64> {
        let mut records = Vec::with_capacity(batch_size.min(pilot_events));
        let mut observed_max = None::<f64>;
        for local_index in 0..pilot_events {
            let event = self.generate_event(rng)?;
            records.push(self.record_from_event(local_index as u64, event, 1.0)?);
            if records.len() == batch_size {
                observe_estimation_batch(
                    layout,
                    &mut records,
                    expression,
                    parameters,
                    &mut observed_max,
                )?;
            }
        }
        observe_estimation_batch(
            layout,
            &mut records,
            expression,
            parameters,
            &mut observed_max,
        )?;
        let observed_max = observed_max.ok_or_else(|| {
            LadduError::Custom("envelope estimation did not produce any pilot weights".to_string())
        })?;
        if observed_max <= 0.0 {
            return Err(LadduError::Custom(format!(
                "envelope estimation requires a positive pilot maximum, got {observed_max}"
            )));
        }
        Ok(observed_max)
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

#[derive(Default)]
struct WeightStats {
    sum_weights: f64,
    min_weight: Option<f64>,
    max_weight: Option<f64>,
    batches_written: u64,
}

impl WeightStats {
    fn observe_batch(&mut self, records: &[GeneratedRecord]) {
        self.batches_written += 1;
        for record in records {
            self.sum_weights += record.weight;
            self.min_weight = Some(
                self.min_weight
                    .map_or(record.weight, |value| value.min(record.weight)),
            );
            self.max_weight = Some(
                self.max_weight
                    .map_or(record.weight, |value| value.max(record.weight)),
            );
        }
    }
}

struct AcceptedBatchContext<'a, S> {
    layout: &'a GeneratedLayout,
    expression: &'a Expression,
    parameters: &'a [f64],
    active_envelope: &'a mut f64,
    envelope_stats: &'a mut EnvelopeStats,
    envelope_violation_policy: EnvelopeViolationPolicy,
    rng: &'a mut Rng,
    accepted_batch: &'a mut Vec<GeneratedRecord>,
    stats: &'a mut WeightStats,
    accepted_events: &'a mut u64,
    rejected_events: &'a mut u64,
    target_events: u64,
    sink: &'a mut S,
}

struct EnvelopeConfiguration {
    active_envelope: f64,
    envelope_stats: EnvelopeStats,
    proposed_events: u64,
}

struct EnvelopeEstimationContext<'a> {
    layout: &'a GeneratedLayout,
    expression: &'a Expression,
    parameters: &'a [f64],
    max_trials: u64,
    batch_size: usize,
}

impl<S> AcceptedBatchContext<'_, S>
where
    S: GeneratedSink,
{
    fn accept_weighted_batch(
        &mut self,
        proposal_batch: &mut Vec<GeneratedRecord>,
    ) -> LadduResult<()> {
        evaluate_record_weights(
            self.layout,
            proposal_batch,
            self.expression,
            self.parameters,
        )?;
        for mut record in proposal_batch.drain(..) {
            let weight = record.weight;
            if weight < 0.0 {
                return Err(LadduError::Custom(format!(
                    "accepted generation produced a negative event weight {weight}"
                )));
            }
            self.envelope_stats.observe(weight, *self.active_envelope);
            if weight > *self.active_envelope {
                match self.envelope_violation_policy {
                    EnvelopeViolationPolicy::Error => {
                        return Err(LadduError::Custom(format!(
                            "accepted generation weight {weight} exceeded envelope {}",
                            *self.active_envelope
                        )));
                    }
                    EnvelopeViolationPolicy::WarnAndContinue => {}
                    EnvelopeViolationPolicy::Grow => {
                        *self.active_envelope = weight;
                        self.envelope_stats.update_final_max(weight);
                    }
                }
            }
            if *self.accepted_events >= self.target_events {
                *self.rejected_events += 1;
                continue;
            }
            if self.rng.f64() * *self.active_envelope < weight {
                record.weight = 1.0;
                self.accepted_batch.push(record);
                *self.accepted_events += 1;
                if self.accepted_batch.len() == self.accepted_batch.capacity() {
                    flush_records(self.layout, self.accepted_batch, self.sink, self.stats)?;
                }
            } else {
                *self.rejected_events += 1;
            }
        }
        Ok(())
    }
}

fn observe_estimation_batch(
    layout: &GeneratedLayout,
    records: &mut Vec<GeneratedRecord>,
    expression: &Expression,
    parameters: &[f64],
    observed_max: &mut Option<f64>,
) -> LadduResult<()> {
    if records.is_empty() {
        return Ok(());
    }
    evaluate_record_weights(layout, records, expression, parameters)?;
    for record in records.iter() {
        if record.weight < 0.0 {
            return Err(LadduError::Custom(format!(
                "envelope estimation produced a negative event weight {}",
                record.weight
            )));
        }
        *observed_max = Some(observed_max.map_or(record.weight, |value| value.max(record.weight)));
    }
    records.clear();
    Ok(())
}

fn flush_records<S>(
    layout: &GeneratedLayout,
    records: &mut Vec<GeneratedRecord>,
    sink: &mut S,
    stats: &mut WeightStats,
) -> LadduResult<()>
where
    S: GeneratedSink,
{
    if records.is_empty() {
        return Ok(());
    }
    stats.observe_batch(records);
    sink.push_batch(GeneratedBatchView { layout, records })?;
    records.clear();
    Ok(())
}

fn evaluate_record_weights(
    layout: &GeneratedLayout,
    records: &mut [GeneratedRecord],
    expression: &Expression,
    parameters: &[f64],
) -> LadduResult<()> {
    if records.is_empty() {
        return Ok(());
    }
    let dataset = Arc::new(records_dataset(layout, records)?);
    let values = expression.load(&dataset)?.evaluate(parameters)?;
    if values.len() != records.len() {
        return Err(LadduError::Custom(format!(
            "weighted generation expected {} weights but expression returned {}",
            records.len(),
            values.len()
        )));
    }
    for (record, value) in records.iter_mut().zip(values) {
        if !value.re.is_finite() {
            return Err(LadduError::Custom(format!(
                "weighted generation produced a non-finite event weight {}",
                value.re
            )));
        }
        record.weight = value.re;
    }
    Ok(())
}

fn records_dataset(layout: &GeneratedLayout, records: &[GeneratedRecord]) -> LadduResult<Dataset> {
    let metadata = Arc::new(DatasetMetadata::new(layout.labels(), Vec::<String>::new())?);
    let events = records
        .iter()
        .map(|record| {
            Arc::new(EventData {
                p4s: record.p4s.clone(),
                aux: Vec::new(),
                weight: 1.0,
            })
        })
        .collect();
    Ok(Dataset::new_with_metadata(events, metadata))
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
    use laddu_core::{Channel, Expression, ParticleProperties, VertexGenerator};

    use super::*;
    use crate::{CallbackSink, DatasetSink, NullSink};

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
    fn callback_sink_receives_generated_batches() {
        let generator = demo_generator();
        let mut labels = Vec::new();
        let mut batch_sizes = Vec::new();
        let result = generator
            .generate(
                5,
                CallbackSink::new(|batch| {
                    labels = batch.layout.labels();
                    batch_sizes.push(batch.records.len());
                    Ok(())
                }),
                GenerationMode::Raw,
                GenerationOptions::default().batch_size(2),
            )
            .unwrap();
        assert_eq!(result.output, 5);
        assert_eq!(batch_sizes, [2, 2, 1]);
        assert_eq!(labels, ["beam", "target", "res", "a", "b", "recoil"]);
    }

    #[test]
    fn callback_sink_errors_are_propagated() {
        let generator = demo_generator();
        let err = generator
            .generate(
                1,
                CallbackSink::new(|_| Err(LadduError::Custom("callback failed".to_string()))),
                GenerationMode::Raw,
                GenerationOptions::default(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("callback failed"));
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

    #[test]
    fn weighted_generation_assigns_expression_weights() {
        let generator = demo_generator();
        let result = generator
            .generate(
                4,
                DatasetSink::new(),
                GenerationMode::Weighted {
                    expression: Box::new(Expression::one() * 2.5),
                    parameters: Vec::new(),
                },
                GenerationOptions::default().batch_size(2),
            )
            .unwrap();
        assert_eq!(result.output.n_events(), 4);
        assert_eq!(result.stats.mode, GenerationModeKind::Weighted);
        assert_eq!(result.stats.proposed_events, 4);
        assert_eq!(result.stats.accepted_events, 4);
        assert_eq!(result.stats.batches_written, 2);
        assert_eq!(result.stats.sum_weights, 10.0);
        assert_eq!(result.stats.min_weight, Some(2.5));
        assert_eq!(result.stats.max_weight, Some(2.5));
        for event in result.output.events_global() {
            assert_eq!(event.weight(), 2.5);
        }
    }

    #[test]
    fn accepted_generation_writes_target_unit_weight_events() {
        let generator = demo_generator();
        let result = generator
            .generate(
                4,
                DatasetSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one()),
                    parameters: Vec::new(),
                    envelope: Envelope::initial(1.0),
                },
                GenerationOptions::default().batch_size(2),
            )
            .unwrap();
        assert_eq!(result.output.n_events(), 4);
        assert_eq!(result.stats.mode, GenerationModeKind::Accepted);
        assert_eq!(result.stats.written_events, 4);
        assert_eq!(result.stats.proposed_events, 4);
        assert_eq!(result.stats.accepted_events, 4);
        assert_eq!(result.stats.rejected_events, 0);
        assert_eq!(result.stats.acceptance_rate, Some(1.0));
        assert_eq!(result.stats.envelope(), Some(1.0));
        assert_eq!(result.stats.envelope_violations(), 0);
        let envelope_stats = result.stats.envelope_stats.as_ref().unwrap();
        assert_eq!(envelope_stats.configured_max, Some(1.0));
        assert_eq!(envelope_stats.final_max, Some(1.0));
        assert_eq!(envelope_stats.observed_max, Some(1.0));
        assert_eq!(envelope_stats.violations, 0);
        assert_eq!(result.stats.sum_weights, 4.0);
        for event in result.output.events_global() {
            assert_eq!(event.weight(), 1.0);
        }
    }

    #[test]
    fn accepted_generation_rejects_envelope_violations() {
        let generator = demo_generator();
        let err = generator
            .generate(
                1,
                NullSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one() * 2.0),
                    parameters: Vec::new(),
                    envelope: Envelope::initial(1.0),
                },
                GenerationOptions::default().batch_size(1),
            )
            .unwrap_err();
        assert!(err.to_string().contains("exceeded envelope"));
    }

    #[test]
    fn accepted_generation_can_continue_after_envelope_violation() {
        let generator = demo_generator();
        let result = generator
            .generate(
                1,
                NullSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one() * 2.0),
                    parameters: Vec::new(),
                    envelope: Envelope::initial(1.0),
                },
                GenerationOptions::default()
                    .batch_size(1)
                    .envelope_violation_policy(EnvelopeViolationPolicy::WarnAndContinue),
            )
            .unwrap();
        let envelope_stats = result.stats.envelope_stats.as_ref().unwrap();
        assert_eq!(result.stats.accepted_events, 1);
        assert_eq!(envelope_stats.violations, 1);
        assert_eq!(envelope_stats.final_max, Some(1.0));
        assert_eq!(envelope_stats.largest_violation_ratio, Some(2.0));
    }

    #[test]
    fn accepted_generation_can_grow_envelope_after_violation() {
        let generator = demo_generator();
        let result = generator
            .generate(
                1,
                NullSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one() * 2.0),
                    parameters: Vec::new(),
                    envelope: Envelope::initial(1.0),
                },
                GenerationOptions::default()
                    .batch_size(1)
                    .envelope_violation_policy(EnvelopeViolationPolicy::Grow),
            )
            .unwrap();
        let envelope_stats = result.stats.envelope_stats.as_ref().unwrap();
        assert_eq!(result.stats.accepted_events, 1);
        assert_eq!(envelope_stats.violations, 1);
        assert_eq!(envelope_stats.updates, 1);
        assert_eq!(envelope_stats.final_max, Some(2.0));
    }

    #[test]
    fn accepted_generation_can_estimate_envelope_from_pilot_events() {
        let generator = demo_generator();
        let result = generator
            .generate(
                4,
                NullSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one() * 2.0),
                    parameters: Vec::new(),
                    envelope: Envelope::estimate(3, 1.0),
                },
                GenerationOptions::default().batch_size(2),
            )
            .unwrap();
        let envelope_stats = result.stats.envelope_stats.as_ref().unwrap();
        assert_eq!(result.output, 4);
        assert_eq!(result.stats.proposed_events, 7);
        assert_eq!(result.stats.accepted_events, 4);
        assert_eq!(result.stats.acceptance_rate, Some(1.0));
        assert_eq!(envelope_stats.pilot_events, 3);
        assert_eq!(envelope_stats.pilot_observed_max, Some(2.0));
        assert_eq!(envelope_stats.safety_factor, Some(1.0));
        assert_eq!(envelope_stats.configured_max, Some(2.0));
        assert_eq!(envelope_stats.final_max, Some(2.0));
    }

    #[test]
    fn estimated_envelope_rejects_invalid_configuration() {
        let generator = demo_generator();
        let err = generator
            .generate(
                1,
                NullSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one()),
                    parameters: Vec::new(),
                    envelope: Envelope::estimate(0, 1.0),
                },
                GenerationOptions::default(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("at least one pilot event"));

        let err = generator
            .generate(
                1,
                NullSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one()),
                    parameters: Vec::new(),
                    envelope: Envelope::estimate(2, 0.0),
                },
                GenerationOptions::default(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("safety factor"));
    }

    #[test]
    fn estimated_envelope_respects_max_trials() {
        let generator = demo_generator();
        let err = generator
            .generate(
                1,
                NullSink::new(),
                GenerationMode::Accepted {
                    expression: Box::new(Expression::one()),
                    parameters: Vec::new(),
                    envelope: Envelope::estimate(3, 1.0),
                },
                GenerationOptions::default().max_trials(2),
            )
            .unwrap_err();
        assert!(err.to_string().contains("pilot events"));
    }
}
