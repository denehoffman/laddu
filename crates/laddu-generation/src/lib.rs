//! Channel-aware Monte Carlo event generation.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::{
    data::{Dataset, EventBatch},
    io::{EventSink, WritePlan, memory::MemorySink},
    schema::Schema,
};
use laddu_expr::{ExprNode, parameters::ParamValues};
use laddu_physics::{LadduPhysicsError, channel::Channel, vectors::RealVec4};
use laddu_runtime::{Execution, PreparedModel};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use thiserror::Error;

pub use laddu_physics::generation::{
    AdaptiveTwoBodyDecay, InitialMomentum, InitialMomentumResult, MassProposal, MassProposalResult,
    NamedMass, NamedMomentum, ProposalResult, ProposalRng, ScalarProposalResult, ScalarSource,
    TComponent, TDistribution, TwoBodyScattering, VertexProposal,
};

pub type GenerationResult<T> = Result<T, GenerationError>;

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("invalid generation channel: {0}")]
    InvalidChannel(String),
    #[error("channel validation failed: {source}")]
    ChannelValidation {
        #[source]
        source: LadduPhysicsError,
    },
    #[error("invalid generation configuration: {0}")]
    InvalidConfiguration(String),
    #[error("initial-state proposal {index} failed: {source}")]
    InitialState {
        index: u64,
        #[source]
        source: LadduPhysicsError,
    },
    #[error("mass proposal for edge `{edge}` at proposal {index} failed: {source}")]
    MassProposal {
        index: u64,
        edge: String,
        #[source]
        source: LadduPhysicsError,
    },
    #[error("vertex `{vertex}` at proposal {index} failed: {source}")]
    VertexProposal {
        index: u64,
        vertex: String,
        #[source]
        source: LadduPhysicsError,
    },
    #[error("scalar column `{column}` at proposal {index} failed: {source}")]
    ScalarProposal {
        index: u64,
        column: String,
        #[source]
        source: LadduPhysicsError,
    },
    #[error("kinematic validation at proposal {index} failed: {source}")]
    Kinematics {
        index: u64,
        #[source]
        source: LadduPhysicsError,
    },
    #[error("model evaluation failed: {0}")]
    Model(String),
    #[error("target weight {weight} exceeds envelope {envelope} at proposal {index}")]
    EnvelopeOverflow {
        index: u64,
        weight: f64,
        envelope: f64,
    },
    #[error(
        "accepted {accepted} events after exhausting {proposals} proposals (requested {requested})"
    )]
    Exhausted {
        requested: usize,
        accepted: usize,
        proposals: usize,
    },
    #[error(transparent)]
    Data(#[from] laddu_data::LadduDataError),
    #[error(transparent)]
    Runtime(#[from] laddu_runtime::RuntimeError),
    #[error(transparent)]
    Physics(#[from] LadduPhysicsError),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WeightedConfig {
    pub events: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub diagnostics: bool,
}

impl WeightedConfig {
    pub fn new(events: usize) -> Self {
        Self {
            events,
            batch_size: 1024,
            seed: 0,
            diagnostics: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct UnweightedConfig {
    pub events: usize,
    /// Optional safeguard limiting production proposals.
    ///
    /// `None` allows generation to continue until the requested event count is
    /// reached. Pilot proposals are not included in this limit.
    pub max_proposals: Option<usize>,
    pub batch_size: usize,
    pub seed: u64,
    pub diagnostics: bool,
    /// Strategy used to establish the rejection-sampling envelope.
    pub envelope: EnvelopeMode,
    pub envelope_overflow: EnvelopeOverflow,
}

impl UnweightedConfig {
    /// Create an unweighted-generation configuration without a proposal limit.
    pub fn new(events: usize) -> Self {
        Self {
            events,
            max_proposals: None,
            batch_size: 1024,
            seed: 0,
            diagnostics: false,
            envelope: EnvelopeMode::default(),
            envelope_overflow: EnvelopeOverflow::Error,
        }
    }

    /// Stop with [`GenerationError::Exhausted`] after at most `max_proposals`
    /// production proposals.
    pub fn with_max_proposals(mut self, max_proposals: usize) -> Self {
        self.max_proposals = Some(max_proposals);
        self
    }
}

/// Policy used when an unweighting proposal exceeds the current envelope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum EnvelopeOverflow {
    /// Stop immediately and report [`GenerationError::EnvelopeOverflow`].
    #[default]
    Error,
    /// Grow the envelope and retrospectively thin previously accepted events.
    ///
    /// Adaptive generation buffers accepted events until the run completes,
    /// because events already written to a sink cannot be withdrawn safely.
    Grow { safety_factor: f64 },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum EnvelopeMode {
    Strict {
        max_weight: f64,
    },
    /// Estimate an envelope from pilot proposals.
    ///
    /// Density-aware built-in proposals may use one deterministic pilot pass
    /// for importance adaptation and a second pass for the final envelope.
    Pilot {
        proposals: usize,
        safety_factor: f64,
    },
}

impl Default for EnvelopeMode {
    fn default() -> Self {
        Self::Pilot {
            proposals: 10_000,
            safety_factor: 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvelopeKind {
    Strict,
    Pilot,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenerationReport {
    pub requested: usize,
    pub produced: usize,
    pub proposals: usize,
    pub pilot_proposals: usize,
    pub rejected: usize,
    pub envelope: Option<f64>,
    pub envelope_kind: Option<EnvelopeKind>,
    pub envelope_updates: usize,
    pub maximum_weight: f64,
    pub minimum_weight: f64,
    pub sum_weights: f64,
    pub sum_squared_weights: f64,
    pub seed: u64,
    pub batch_size: usize,
}

impl GenerationReport {
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.produced as f64 / self.proposals as f64
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelEvaluator {
    prepared: PreparedModel,
    params: ParamValues,
    required_scalars: HashSet<String>,
}

impl ModelEvaluator {
    pub fn prepare(
        model: &CompiledModel,
        params: ParamValues,
        execution: &Execution,
    ) -> GenerationResult<Self> {
        let required_scalars = model
            .graph()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::EventScalar(name) => Some(name.to_string()),
                _ => None,
            })
            .collect();
        Ok(Self {
            prepared: PreparedModel::prepare(model, execution)?,
            params,
            required_scalars,
        })
    }

    /// Evaluate the positive-real model value for every event in a batch.
    ///
    /// This is useful for projecting a fitted model over weighted Monte Carlo
    /// without regenerating events.
    pub fn evaluate_batch(&self, batch: &EventBatch) -> GenerationResult<Vec<f64>> {
        let reduction = ReductionPlan::weighted_positive_real();
        self.prepared
            .evaluate_batch(&self.params, batch)?
            .into_iter()
            .map(|value| {
                if !value.re.is_finite() {
                    return Err(GenerationError::Model(format!(
                        "positive-real model produced nonfinite value {}",
                        value.re
                    )));
                }
                reduction
                    .apply(value)
                    .map(|out| out.value())
                    .map_err(|err| GenerationError::Model(err.to_string()))
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct ChannelGenerator {
    edges: Vec<EdgePlan>,
    vertices: Vec<VertexPlan>,
    edge_names: Vec<String>,
    output_indices: Vec<usize>,
    output_names: Vec<String>,
    root_indices: Vec<usize>,
    scalar_sources: Vec<(String, ScalarSource)>,
}

#[derive(Clone, Debug)]
struct EdgePlan {
    name: String,
    initial: Option<InitialMomentum>,
    mass: EdgeMassPlan,
}

#[derive(Clone, Debug)]
enum EdgeMassPlan {
    Fixed(f64),
    Proposed(MassProposal),
}

#[derive(Clone, Debug)]
struct AdaptiveMassProposal {
    base: MassProposal,
    low: f64,
    high: f64,
    counts: Arc<[f64]>,
    defensive_fraction: f64,
}

impl AdaptiveMassProposal {
    fn bin_width(&self) -> f64 {
        (self.high - self.low) / self.counts.len() as f64
    }

    fn truncated_total(&self, minimum: f64, maximum: f64) -> f64 {
        let width = self.bin_width();
        self.counts
            .iter()
            .enumerate()
            .map(|(bin, count)| {
                let bin_low = self.low + bin as f64 * width;
                let bin_high = bin_low + width;
                let overlap = maximum.min(bin_high) - minimum.max(bin_low);
                if overlap > 0.0 {
                    count * overlap / width
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn adaptive_density(&self, minimum: f64, maximum: f64, mass: f64) -> f64 {
        if mass < self.low || mass > self.high || mass < minimum || mass > maximum {
            return 0.0;
        }
        let width = self.bin_width();
        let bin = (((mass - self.low) / width) as usize).min(self.counts.len() - 1);
        let total = self.truncated_total(minimum, maximum);
        if total > 0.0 {
            self.counts[bin] / (width * total)
        } else {
            0.0
        }
    }

    fn sample_adaptive(&self, minimum: f64, maximum: f64, rng: &mut ProposalRng) -> Option<f64> {
        let width = self.bin_width();
        let total = self.truncated_total(minimum, maximum);
        if total <= 0.0 {
            return None;
        }
        let mut threshold = rng.uniform() * total;
        for (bin, count) in self.counts.iter().enumerate() {
            let bin_low = self.low + bin as f64 * width;
            let bin_high = bin_low + width;
            let low = minimum.max(bin_low);
            let high = maximum.min(bin_high);
            let weight = if high > low {
                count * (high - low) / width
            } else {
                0.0
            };
            if threshold <= weight && weight > 0.0 {
                return Some(low + rng.uniform() * (high - low));
            }
            threshold -= weight;
        }
        None
    }
}

impl AdaptiveMassProposal {
    fn propose(
        &self,
        minimum: f64,
        maximum: f64,
        rng: &mut ProposalRng,
    ) -> laddu_physics::LadduPhysicsResult<MassProposalResult> {
        let adaptive_available = self.truncated_total(minimum, maximum) > 0.0;
        let use_base = !adaptive_available || rng.uniform() < self.defensive_fraction;
        let mass = if use_base {
            self.base.propose(minimum, maximum, rng)?.mass
        } else {
            self.sample_adaptive(minimum, maximum, rng)
                .expect("positive adaptive mass support must be sampleable")
        };
        let Some(base_density) = self.base.density(minimum, maximum, mass)? else {
            return Err(LadduPhysicsError::invalid_relation(
                "adaptive mass proposal lost access to its base density",
            ));
        };
        let density = if adaptive_available {
            self.defensive_fraction * base_density
                + (1.0 - self.defensive_fraction) * self.adaptive_density(minimum, maximum, mass)
        } else {
            base_density
        };
        if !density.is_finite() || density <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "adaptive mass-proposal density",
                "finite and positive",
                density,
            ));
        }
        Ok(MassProposalResult {
            mass,
            weight: density.recip(),
        })
    }

    fn density(
        &self,
        minimum: f64,
        maximum: f64,
        mass: f64,
    ) -> laddu_physics::LadduPhysicsResult<Option<f64>> {
        let Some(base_density) = self.base.density(minimum, maximum, mass)? else {
            return Ok(None);
        };
        let adaptive_available = self.truncated_total(minimum, maximum) > 0.0;
        Ok(Some(if adaptive_available {
            self.defensive_fraction * base_density
                + (1.0 - self.defensive_fraction) * self.adaptive_density(minimum, maximum, mass)
        } else {
            base_density
        }))
    }
}

#[derive(Clone, Debug)]
struct VertexPlan {
    name: String,
    incoming: Vec<usize>,
    outgoing: Vec<usize>,
    proposal: VertexProposal,
    adaptive_decay: bool,
}

#[derive(Clone, Debug)]
struct ProposalAdaptations {
    masses: Vec<Option<AdaptiveMassProposal>>,
    vertices: Vec<Option<AdaptiveTwoBodyDecay>>,
}

#[derive(Clone, Debug)]
struct GeneratedEvent {
    p4s: Vec<RealVec4>,
    scalars: Vec<f64>,
    proposal_weight: f64,
    model_weight: f64,
    target_weight: f64,
    index: u64,
}

impl ChannelGenerator {
    pub fn new(channel: Channel) -> GenerationResult<Self> {
        channel
            .validate()
            .map_err(|source| GenerationError::ChannelValidation { source })?;
        let edge_names = channel
            .edges()
            .map(|edge| edge.name().to_owned())
            .collect::<Vec<_>>();
        let output_names = channel
            .edges()
            .filter(|edge| edge.is_output())
            .map(|edge| edge.name().to_owned())
            .collect::<Vec<_>>();
        if output_names.is_empty() {
            return Err(GenerationError::InvalidChannel(
                "at least one edge must be marked as output".into(),
            ));
        }
        const DIAGNOSTICS: [&str; 3] = [
            "__laddu_proposal_weight",
            "__laddu_model_weight",
            "__laddu_target_weight",
        ];
        if output_names
            .iter()
            .any(|name| DIAGNOSTICS.contains(&name.as_str()))
        {
            return Err(GenerationError::InvalidChannel(
                "an output edge uses a reserved generation-diagnostic name".into(),
            ));
        }
        let root_edges = channel
            .initial_edges()
            .map(|edge| edge.name().to_owned())
            .collect::<HashSet<_>>();
        let plan = topological_plan(&channel, &root_edges)?;
        let edge_indices = edge_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<HashMap<_, _>>();
        let output_indices = output_names
            .iter()
            .map(|name| edge_indices[name.as_str()])
            .collect::<Vec<_>>();
        let root_indices = channel
            .initial_edges()
            .map(|edge| edge_indices[edge.name()])
            .collect::<Vec<_>>();
        let edges = channel
            .edges()
            .map(|edge| {
                let mass = if let Some(proposal) = edge.mass_proposal() {
                    EdgeMassPlan::Proposed(*proposal)
                } else {
                    let properties = edge.properties().ok_or_else(|| {
                        GenerationError::InvalidChannel(format!(
                            "edge `{}` has neither particle properties nor a mass proposal",
                            edge.name()
                        ))
                    })?;
                    EdgeMassPlan::Fixed(
                        properties
                            .mass()
                            .map_err(|source| GenerationError::ChannelValidation { source })?,
                    )
                };
                Ok(EdgePlan {
                    name: edge.name().to_owned(),
                    initial: edge.initial_momentum().cloned(),
                    mass,
                })
            })
            .collect::<GenerationResult<Vec<_>>>()?;
        let channel_vertices = channel.vertices().collect::<Vec<_>>();
        let vertices = plan
            .into_iter()
            .map(|vertex_index| {
                let vertex = channel_vertices[vertex_index];
                let incoming = vertex
                    .incoming()
                    .iter()
                    .map(|name| edge_indices[name.as_str()])
                    .collect::<Vec<_>>();
                let outgoing = vertex
                    .outgoing()
                    .iter()
                    .map(|name| edge_indices[name.as_str()])
                    .collect::<Vec<_>>();
                let (proposal, adaptive_decay) = match vertex.generation() {
                    Some(proposal) => (proposal.clone(), false),
                    None if incoming.len() == 1 && outgoing.len() == 2 => {
                        (VertexProposal::TwoBodyDecay, true)
                    }
                    None => {
                        return Err(GenerationError::InvalidChannel(format!(
                            "vertex `{}` has no generation proposal",
                            vertex.name()
                        )));
                    }
                };
                Ok(VertexPlan {
                    name: vertex.name().to_owned(),
                    incoming,
                    outgoing,
                    proposal,
                    adaptive_decay,
                })
            })
            .collect::<GenerationResult<Vec<_>>>()?;
        Ok(Self {
            edges,
            vertices,
            edge_names,
            output_indices,
            output_names,
            root_indices,
            scalar_sources: Vec::new(),
        })
    }

    pub fn with_scalar(
        mut self,
        name: impl Into<String>,
        source: ScalarSource,
    ) -> GenerationResult<Self> {
        self.add_scalar(name, source)?;
        Ok(self)
    }

    pub fn add_scalar(
        &mut self,
        name: impl Into<String>,
        source: ScalarSource,
    ) -> GenerationResult<&mut Self> {
        let name = name.into();
        const DIAGNOSTICS: [&str; 3] = [
            "__laddu_proposal_weight",
            "__laddu_model_weight",
            "__laddu_target_weight",
        ];
        if name.is_empty()
            || DIAGNOSTICS.contains(&name.as_str())
            || self.edge_names.contains(&name)
            || self
                .scalar_sources
                .iter()
                .any(|(existing, _)| existing == &name)
        {
            return Err(GenerationError::InvalidConfiguration(format!(
                "scalar column name `{name}` is empty, reserved, or duplicated"
            )));
        }
        self.scalar_sources.push((name, source));
        Ok(self)
    }

    pub fn generate_weighted_to(
        &self,
        config: WeightedConfig,
        model: Option<&ModelEvaluator>,
        sink: &mut dyn EventSink,
    ) -> GenerationResult<GenerationReport> {
        validate_common(config.events, config.batch_size)?;
        let schema = self.output_schema(true, config.diagnostics)?;
        sink.begin(Arc::clone(&schema), WritePlan::default())?;
        let mut report = report(config.events, config.seed, config.batch_size);
        let work_batch = config.batch_size.max(16_384);
        for start in (0..config.events).step_by(work_batch) {
            let count = work_batch.min(config.events - start);
            let mut events = self.propose_range(start as u64, count, config.seed, 0)?;
            self.apply_model(&mut events, model)?;
            update_report(&mut report, &events);
            report.proposals += events.len();
            report.produced += events.len();
            for chunk in events.chunks(config.batch_size) {
                let batch =
                    self.output_batch(chunk, Arc::clone(&schema), true, config.diagnostics)?;
                sink.write_batch(&batch)?;
            }
        }
        sink.finish()?;
        Ok(report)
    }

    pub fn generate_unweighted_to(
        &self,
        config: UnweightedConfig,
        model: &ModelEvaluator,
        sink: &mut dyn EventSink,
    ) -> GenerationResult<GenerationReport> {
        validate_common(config.events, config.batch_size)?;
        if config
            .max_proposals
            .is_some_and(|max_proposals| max_proposals < config.events)
        {
            return Err(GenerationError::InvalidConfiguration(
                "max_proposals must be at least the requested event count".into(),
            ));
        }
        let mut adaptations = None;
        let (mut bound, kind, pilot_count) = match config.envelope {
            EnvelopeMode::Strict { max_weight } => {
                validate_bound(max_weight)?;
                (max_weight, EnvelopeKind::Strict, 0)
            }
            EnvelopeMode::Pilot {
                proposals,
                safety_factor,
            } => {
                if proposals == 0 || !safety_factor.is_finite() || safety_factor <= 1.0 {
                    return Err(GenerationError::InvalidConfiguration("pilot proposals must be nonzero and safety_factor must be finite and greater than one".into()));
                }
                let mut adaptation_pilot = self.propose_range(0, proposals, config.seed, 1)?;
                self.apply_model(&mut adaptation_pilot, Some(model))?;
                let learned = self.learn_mass_adaptations(&adaptation_pilot)?;
                let has_adaptation = learned.masses.iter().any(Option::is_some)
                    || learned.vertices.iter().any(Option::is_some);
                let mut envelope_pilot = if has_adaptation {
                    self.propose_range_with_adaptation(
                        0,
                        proposals,
                        config.seed,
                        2,
                        Some(&learned),
                    )?
                } else {
                    adaptation_pilot
                };
                if has_adaptation {
                    self.apply_model(&mut envelope_pilot, Some(model))?;
                    adaptations = Some(learned);
                }
                let observed = envelope_pilot
                    .iter()
                    .map(|event| event.target_weight)
                    .fold(0.0, f64::max);
                (
                    observed * safety_factor,
                    EnvelopeKind::Pilot,
                    proposals * if has_adaptation { 2 } else { 1 },
                )
            }
        };
        if let EnvelopeOverflow::Grow { safety_factor } = config.envelope_overflow
            && (!safety_factor.is_finite() || safety_factor <= 1.0)
        {
            return Err(GenerationError::InvalidConfiguration(
                "envelope growth safety_factor must be finite and greater than one".into(),
            ));
        }
        let schema = self.output_schema(false, config.diagnostics)?;
        sink.begin(Arc::clone(&schema), WritePlan::default())?;
        let mut report = report(config.events, config.seed, config.batch_size);
        report.envelope = Some(bound);
        report.envelope_kind = Some(kind);
        report.pilot_proposals = pilot_count;
        let mut proposal_index = 0_usize;
        let mut buffered = Vec::new();
        let work_batch = config.batch_size.max(32_768);
        while report.produced < config.events
            && config
                .max_proposals
                .is_none_or(|max_proposals| proposal_index < max_proposals)
        {
            let count = config.max_proposals.map_or(work_batch, |max_proposals| {
                work_batch.min(max_proposals - proposal_index)
            });
            let mut events = self.propose_range_with_adaptation(
                proposal_index as u64,
                count,
                config.seed,
                0,
                adaptations.as_ref(),
            )?;
            self.apply_model(&mut events, Some(model))?;
            let remaining_before_overflow = config.events - report.produced;
            if let Some(last_needed) = events
                .iter()
                .enumerate()
                .filter(|(_, event)| {
                    acceptance_uniform(config.seed, event.index) * bound <= event.target_weight
                })
                .nth(remaining_before_overflow - 1)
                .map(|(position, _)| position + 1)
                && !events[..last_needed]
                    .iter()
                    .any(|event| event.target_weight > bound)
            {
                events.truncate(last_needed);
            }
            update_report(&mut report, &events);
            if let Some(overflow) = events
                .iter()
                .filter(|event| event.target_weight > bound)
                .max_by(|a, b| a.target_weight.total_cmp(&b.target_weight))
            {
                match config.envelope_overflow {
                    EnvelopeOverflow::Error => {
                        return Err(GenerationError::EnvelopeOverflow {
                            index: overflow.index,
                            weight: overflow.target_weight,
                            envelope: bound,
                        });
                    }
                    EnvelopeOverflow::Grow { safety_factor } => {
                        bound = overflow.target_weight * safety_factor;
                        validate_bound(bound)?;
                        report.envelope = Some(bound);
                        report.envelope_updates += 1;
                        buffered.retain(|event: &GeneratedEvent| {
                            acceptance_uniform(config.seed, event.index) * bound
                                <= event.target_weight
                        });
                        report.produced = buffered.len();
                    }
                }
            }
            let remaining = config.events - report.produced;
            let proposal_count = events.len();
            let accepted = events
                .into_iter()
                .filter(|event| {
                    acceptance_uniform(config.seed, event.index) * bound <= event.target_weight
                })
                .take(remaining)
                .collect::<Vec<_>>();
            report.proposals += proposal_count;
            report.produced += accepted.len();
            report.rejected = report.proposals - report.produced;
            proposal_index += proposal_count;
            match config.envelope_overflow {
                EnvelopeOverflow::Error if !accepted.is_empty() => {
                    for chunk in accepted.chunks(config.batch_size) {
                        let batch = self.output_batch(
                            chunk,
                            Arc::clone(&schema),
                            false,
                            config.diagnostics,
                        )?;
                        sink.write_batch(&batch)?;
                    }
                }
                EnvelopeOverflow::Grow { .. } => {
                    buffered.extend(accepted);
                    report.produced = buffered.len();
                    report.rejected = report.proposals - report.produced;
                }
                EnvelopeOverflow::Error => {}
            }
        }
        if report.produced != config.events {
            return Err(GenerationError::Exhausted {
                requested: config.events,
                accepted: report.produced,
                proposals: report.proposals,
            });
        }
        if matches!(config.envelope_overflow, EnvelopeOverflow::Grow { .. }) {
            for events in buffered.chunks(config.batch_size) {
                let batch =
                    self.output_batch(events, Arc::clone(&schema), false, config.diagnostics)?;
                sink.write_batch(&batch)?;
            }
        }
        sink.finish()?;
        Ok(report)
    }

    pub fn generate_weighted_dataset(
        &self,
        config: WeightedConfig,
        model: Option<&ModelEvaluator>,
    ) -> GenerationResult<(Dataset, GenerationReport)> {
        let mut sink = MemorySink::new();
        let report = self.generate_weighted_to(config, model, &mut sink)?;
        Ok((Dataset::from_batches(sink.into_batches())?, report))
    }

    pub fn generate_unweighted_dataset(
        &self,
        config: UnweightedConfig,
        model: &ModelEvaluator,
    ) -> GenerationResult<(Dataset, GenerationReport)> {
        let mut sink = MemorySink::new();
        let report = self.generate_unweighted_to(config, model, &mut sink)?;
        Ok((Dataset::from_batches(sink.into_batches())?, report))
    }

    fn propose_range(
        &self,
        start: u64,
        count: usize,
        seed: u64,
        stream: u64,
    ) -> GenerationResult<Vec<GeneratedEvent>> {
        self.propose_range_with_adaptation(start, count, seed, stream, None)
    }

    fn propose_range_with_adaptation(
        &self,
        start: u64,
        count: usize,
        seed: u64,
        stream: u64,
        adaptations: Option<&ProposalAdaptations>,
    ) -> GenerationResult<Vec<GeneratedEvent>> {
        (0..count)
            .into_par_iter()
            .map(|offset| self.propose(start + offset as u64, seed, stream, adaptations))
            .collect()
    }

    fn propose(
        &self,
        index: u64,
        seed: u64,
        stream: u64,
        adaptations: Option<&ProposalAdaptations>,
    ) -> GenerationResult<GeneratedEvent> {
        let mut rng = ProposalRng::new(derive_seed(seed, stream, index, 0));
        let mut p4s = vec![RealVec4::new(0.0, 0.0, 0.0, 0.0); self.edges.len()];
        let mut proposal_weight = 1.0;
        for &edge_index in &self.root_indices {
            let edge = &self.edges[edge_index];
            let source = edge
                .initial
                .as_ref()
                .ok_or_else(|| GenerationError::InitialState {
                    index,
                    source: LadduPhysicsError::invalid_relation(format!(
                        "initial edge `{}` has no momentum source",
                        edge.name
                    )),
                })?;
            let sampled = source
                .sample_prevalidated(
                    match edge.mass {
                        EdgeMassPlan::Fixed(mass) => mass,
                        EdgeMassPlan::Proposed(_) => {
                            return Err(GenerationError::InitialState {
                                index,
                                source: LadduPhysicsError::invalid_relation(format!(
                                    "initial edge `{}` cannot use a generated mass",
                                    edge.name
                                )),
                            });
                        }
                    },
                    &mut rng,
                )
                .map_err(|source| GenerationError::InitialState { index, source })?;
            p4s[edge_index] = sampled.p4;
            proposal_weight *= sampled.weight;
        }
        if !proposal_weight.is_finite() || proposal_weight <= 0.0 {
            return Err(GenerationError::InitialState {
                index,
                source: LadduPhysicsError::invalid_value(
                    "initial-state proposal weight",
                    "finite and positive",
                    proposal_weight,
                ),
            });
        }
        let total_initial: RealVec4 = self.root_indices.iter().map(|&edge| p4s[edge]).sum();
        let maximum_mass = total_initial
            .m()
            .map_err(|source| GenerationError::Kinematics { index, source })?;
        let mut masses = Vec::with_capacity(self.edges.len());
        for (edge_index, edge) in self.edges.iter().enumerate() {
            let mass = if let EdgeMassPlan::Proposed(base_proposal) = &edge.mass {
                let mut mass_rng =
                    ProposalRng::new(derive_seed(seed, stream, index, 1 + edge_index as u64));
                let result = if let Some(proposal) =
                    adaptations.and_then(|adaptations| adaptations.masses[edge_index].as_ref())
                {
                    proposal.propose(0.0, maximum_mass, &mut mass_rng)
                } else {
                    base_proposal.propose(0.0, maximum_mass, &mut mass_rng)
                }
                .map_err(|source| GenerationError::MassProposal {
                    index,
                    edge: edge.name.clone(),
                    source,
                })?;
                proposal_weight *= result.weight;
                result.mass
            } else if let EdgeMassPlan::Fixed(mass) = edge.mass {
                mass
            } else {
                unreachable!()
            };
            if !mass.is_finite() || mass < 0.0 {
                return Err(GenerationError::MassProposal {
                    index,
                    edge: edge.name.clone(),
                    source: LadduPhysicsError::invalid_value(
                        "mass",
                        "finite and nonnegative",
                        mass,
                    ),
                });
            }
            masses.push(mass);
        }
        for &edge in &self.root_indices {
            validate_p4(&self.edges[edge].name, p4s[edge], masses[edge], index)?;
        }
        for (step, vertex) in self.vertices.iter().enumerate() {
            let incoming = vertex
                .incoming
                .iter()
                .map(|&edge| NamedMomentum {
                    name: &self.edges[edge].name,
                    p4: p4s[edge],
                })
                .collect::<SmallVec<[_; 2]>>();
            let outgoing = vertex
                .outgoing
                .iter()
                .map(|&edge| NamedMass {
                    name: &self.edges[edge].name,
                    mass: masses[edge],
                })
                .collect::<SmallVec<[_; 2]>>();
            let mut vertex_rng =
                ProposalRng::new(derive_seed(seed, stream, index, 10_000 + step as u64));
            let result = if let Some(proposal) =
                adaptations.and_then(|adaptations| adaptations.vertices[step].as_ref())
            {
                proposal.propose(&incoming, &outgoing, &mut vertex_rng)
            } else {
                vertex
                    .proposal
                    .propose(&incoming, &outgoing, &mut vertex_rng)
            }
            .map_err(|source| GenerationError::VertexProposal {
                index,
                vertex: vertex.name.clone(),
                source,
            })?;
            if result.outgoing.len() != vertex.outgoing.len()
                || !result.weight.is_finite()
                || result.weight <= 0.0
            {
                return Err(GenerationError::VertexProposal {
                    index,
                    vertex: vertex.name.clone(),
                    source: LadduPhysicsError::invalid_relation(format!(
                        "proposal returned {} outgoing momenta and weight {}",
                        result.outgoing.len(),
                        result.weight
                    )),
                });
            }
            proposal_weight *= result.weight;
            for (&edge, p4) in vertex.outgoing.iter().zip(result.outgoing) {
                validate_p4(&self.edges[edge].name, p4, masses[edge], index)?;
                p4s[edge] = p4;
            }
            validate_indexed_conservation(vertex, &p4s, index)?;
        }
        let mut scalars = Vec::with_capacity(self.scalar_sources.len());
        for (scalar_index, (column, source)) in self.scalar_sources.iter().enumerate() {
            let mut scalar_rng = ProposalRng::new(derive_seed(
                seed,
                stream,
                index,
                20_000 + scalar_index as u64,
            ));
            let result = source.sample(&mut scalar_rng).map_err(|source| {
                GenerationError::ScalarProposal {
                    index,
                    column: column.clone(),
                    source,
                }
            })?;
            proposal_weight *= result.weight;
            scalars.push(result.value);
        }
        if !proposal_weight.is_finite() || proposal_weight <= 0.0 {
            return Err(GenerationError::Kinematics {
                index,
                source: LadduPhysicsError::invalid_value(
                    "accumulated proposal weight",
                    "finite and positive",
                    proposal_weight,
                ),
            });
        }
        Ok(GeneratedEvent {
            p4s,
            scalars,
            proposal_weight,
            model_weight: 1.0,
            target_weight: proposal_weight,
            index,
        })
    }

    fn apply_model(
        &self,
        events: &mut [GeneratedEvent],
        model: Option<&ModelEvaluator>,
    ) -> GenerationResult<()> {
        if let Some(model) = model {
            let scalar_names = self
                .scalar_sources
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>();
            let available = scalar_names.iter().copied().collect::<HashSet<_>>();
            if let Some(missing) = model
                .required_scalars
                .iter()
                .find(|name| !available.contains(name.as_str()))
            {
                return Err(GenerationError::Model(format!(
                    "model requires scalar column `{missing}`, but the generator has no source for it"
                )));
            }
            let schema = Arc::new(Schema::new(
                self.edge_names.iter().map(String::as_str),
                scalar_names,
                false,
            )?);
            events
                .par_chunks_mut(4_096)
                .try_for_each(|events| -> GenerationResult<()> {
                    let columns = (0..self.edge_names.len())
                        .map(|edge| {
                            Arc::<[RealVec4]>::from(
                                events
                                    .iter()
                                    .map(|event| event.p4s[edge])
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    let scalar_columns = (0..self.scalar_sources.len())
                        .map(|column| {
                            Arc::<[f64]>::from(
                                events
                                    .iter()
                                    .map(|event| event.scalars[column])
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    let batch =
                        EventBatch::new(Arc::clone(&schema), columns, scalar_columns, None)?;
                    let weights = model.evaluate_batch(&batch)?;
                    for (event, weight) in events.iter_mut().zip(weights) {
                        event.model_weight = weight;
                        event.target_weight = event.proposal_weight * weight;
                    }
                    Ok(())
                })?;
        }
        Ok(())
    }

    fn learn_mass_adaptations(
        &self,
        pilot: &[GeneratedEvent],
    ) -> GenerationResult<ProposalAdaptations> {
        const BINS: usize = 64;
        const DEFENSIVE_FRACTION: f64 = 0.2;
        const MINIMUM_GAIN: f64 = 1.02;

        let mut best: Option<(usize, f64, AdaptiveMassProposal)> = None;
        let old_sum: f64 = pilot.iter().map(|event| event.target_weight).sum();
        let old_max = pilot
            .iter()
            .map(|event| event.target_weight)
            .fold(0.0, f64::max);
        if pilot.is_empty() || old_sum <= 0.0 || old_max <= 0.0 {
            return Ok(ProposalAdaptations {
                masses: vec![None; self.edges.len()],
                vertices: vec![None; self.vertices.len()],
            });
        }
        let old_efficiency = old_sum / (pilot.len() as f64 * old_max);

        for (edge_index, edge) in self.edges.iter().enumerate() {
            let EdgeMassPlan::Proposed(base) = &edge.mass else {
                continue;
            };
            let masses = pilot
                .iter()
                .map(|event| {
                    event.p4s[edge_index]
                        .m()
                        .map_err(|source| GenerationError::Kinematics {
                            index: event.index,
                            source,
                        })
                })
                .collect::<GenerationResult<Vec<_>>>()?;
            let low = masses.iter().copied().fold(f64::INFINITY, f64::min);
            let high = masses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !low.is_finite() || !high.is_finite() || high <= low {
                continue;
            }
            let width = (high - low) / BINS as f64;
            let mut counts = vec![0.0; BINS];
            for (&mass, event) in masses.iter().zip(pilot) {
                let bin = (((mass - low) / width) as usize).min(BINS - 1);
                counts[bin] += event.target_weight;
            }
            if counts.iter().filter(|count| **count > 0.0).count() < 2 {
                continue;
            }
            let candidate = AdaptiveMassProposal {
                base: *base,
                low,
                high,
                counts: counts.into(),
                defensive_fraction: DEFENSIVE_FRACTION,
            };
            let mut new_sum = 0.0;
            let mut new_max: f64 = 0.0;
            let mut density_available = true;
            for (event, &mass) in pilot.iter().zip(&masses) {
                let total_initial: RealVec4 =
                    self.root_indices.iter().map(|&edge| event.p4s[edge]).sum();
                let maximum = total_initial
                    .m()
                    .map_err(|source| GenerationError::Kinematics {
                        index: event.index,
                        source,
                    })?;
                let Some(base_density) = base.density(0.0, maximum, mass)? else {
                    density_available = false;
                    break;
                };
                let Some(new_density) = candidate.density(0.0, maximum, mass)? else {
                    density_available = false;
                    break;
                };
                if base_density <= 0.0 || new_density <= 0.0 {
                    density_available = false;
                    break;
                }
                let adjusted = event.target_weight * base_density / new_density;
                new_sum += adjusted;
                new_max = new_max.max(adjusted);
            }
            if !density_available || new_sum <= 0.0 || new_max <= 0.0 {
                continue;
            }
            let new_efficiency = new_sum / (pilot.len() as f64 * new_max);
            let gain = new_efficiency / old_efficiency;
            if gain >= MINIMUM_GAIN
                && best
                    .as_ref()
                    .is_none_or(|(_, best_gain, _)| gain > *best_gain)
            {
                best = Some((edge_index, gain, candidate));
            }
        }

        let mut masses = vec![None; self.edges.len()];
        if let Some((edge, _, proposal)) = best {
            masses[edge] = Some(proposal);
        }
        let mut vertices = vec![None; self.vertices.len()];
        for (vertex_index, vertex) in self.vertices.iter().enumerate() {
            if !vertex.adaptive_decay {
                continue;
            }
            let mut counts = vec![0.0; 32];
            let mut costhetas = Vec::with_capacity(pilot.len());
            for event in pilot {
                let parent = event.p4s[vertex.incoming[0]];
                let inverse_beta =
                    -parent
                        .beta()
                        .map_err(|source| GenerationError::Kinematics {
                            index: event.index,
                            source,
                        })?;
                let rest = event.p4s[vertex.outgoing[0]].boost(&inverse_beta);
                let momentum = rest.vec3();
                let magnitude = momentum.mag();
                if magnitude <= 0.0 || !magnitude.is_finite() {
                    costhetas.push(None);
                    continue;
                }
                let costheta = (momentum.pz() / magnitude).clamp(-1.0, 1.0);
                costhetas.push(Some(costheta));
                let bin =
                    (((costheta + 1.0) * 0.5 * counts.len() as f64) as usize).min(counts.len() - 1);
                counts[bin] += event.target_weight;
            }
            let total: f64 = counts.iter().sum();
            let width = 2.0 / counts.len() as f64;
            let mut new_sum = 0.0;
            let mut new_max: f64 = 0.0;
            for (event, costheta) in pilot.iter().zip(&costhetas) {
                let Some(costheta) = costheta else {
                    continue;
                };
                let bin = (((costheta + 1.0) / width) as usize).min(counts.len() - 1);
                let learned_density = counts[bin] / (total * width);
                let density =
                    DEFENSIVE_FRACTION * 0.5 + (1.0 - DEFENSIVE_FRACTION) * learned_density;
                let adjusted = event.target_weight * 0.5 / density;
                new_sum += adjusted;
                new_max = new_max.max(adjusted);
            }
            let new_efficiency = new_sum / (pilot.len() as f64 * new_max);
            if counts.iter().filter(|count| **count > 0.0).count() >= 2
                && new_efficiency / old_efficiency >= MINIMUM_GAIN
            {
                vertices[vertex_index] = Some(
                    AdaptiveTwoBodyDecay::new(counts.into(), DEFENSIVE_FRACTION)
                        .map_err(GenerationError::Physics)?,
                );
            }
        }
        Ok(ProposalAdaptations { masses, vertices })
    }

    fn output_schema(&self, weighted: bool, diagnostics: bool) -> GenerationResult<Arc<Schema>> {
        let mut scalars = self
            .scalar_sources
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        if diagnostics {
            scalars.extend([
                "__laddu_proposal_weight",
                "__laddu_model_weight",
                "__laddu_target_weight",
            ]);
        }
        Ok(Arc::new(Schema::new(
            self.output_names.iter().map(String::as_str),
            scalars,
            weighted,
        )?))
    }

    fn output_batch(
        &self,
        events: &[GeneratedEvent],
        schema: Arc<Schema>,
        weighted: bool,
        diagnostics: bool,
    ) -> GenerationResult<EventBatch> {
        let p4s = self
            .output_indices
            .iter()
            .map(|&edge| {
                Arc::<[RealVec4]>::from(
                    events
                        .iter()
                        .map(|event| event.p4s[edge])
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut scalars = (0..self.scalar_sources.len())
            .map(|column| {
                Arc::<[f64]>::from(
                    events
                        .iter()
                        .map(|event| event.scalars[column])
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        if diagnostics {
            scalars.extend([
                Arc::<[f64]>::from(
                    events
                        .iter()
                        .map(|event| event.proposal_weight)
                        .collect::<Vec<_>>(),
                ),
                Arc::<[f64]>::from(
                    events
                        .iter()
                        .map(|event| event.model_weight)
                        .collect::<Vec<_>>(),
                ),
                Arc::<[f64]>::from(
                    events
                        .iter()
                        .map(|event| event.target_weight)
                        .collect::<Vec<_>>(),
                ),
            ]);
        }
        let weights = weighted.then(|| {
            Arc::<[f64]>::from(
                events
                    .iter()
                    .map(|event| event.target_weight)
                    .collect::<Vec<_>>(),
            )
        });
        Ok(EventBatch::new(schema, p4s, scalars, weights)?)
    }
}

impl TryFrom<Channel> for ChannelGenerator {
    type Error = GenerationError;

    fn try_from(channel: Channel) -> Result<Self, Self::Error> {
        Self::new(channel)
    }
}

fn topological_plan(channel: &Channel, roots: &HashSet<String>) -> GenerationResult<Vec<usize>> {
    let vertices = channel.vertices().collect::<Vec<_>>();
    let declared = channel
        .edges()
        .map(|edge| edge.name())
        .collect::<HashSet<_>>();
    let mut consumers = HashMap::<&str, usize>::new();
    for vertex in &vertices {
        for edge in vertex.incoming().iter().chain(vertex.outgoing()) {
            if !declared.contains(edge.as_str()) {
                return Err(GenerationError::InvalidChannel(format!(
                    "vertex `{}` references undeclared edge `{edge}`",
                    vertex.name()
                )));
            }
        }
        for edge in vertex.incoming() {
            *consumers.entry(edge).or_default() += 1;
        }
    }
    if let Some((edge, count)) = consumers.iter().find(|(_, count)| **count > 1) {
        return Err(GenerationError::InvalidChannel(format!(
            "edge `{edge}` is consumed by {count} vertices"
        )));
    }
    let mut available = roots.clone();
    let mut remaining = (0..vertices.len()).collect::<Vec<_>>();
    let mut plan = Vec::with_capacity(vertices.len());
    while !remaining.is_empty() {
        let Some(position) = remaining.iter().position(|index| {
            vertices[*index]
                .incoming()
                .iter()
                .all(|edge| available.contains(edge))
        }) else {
            return Err(GenerationError::InvalidChannel(
                "channel is cyclic or has an edge without a producer".into(),
            ));
        };
        let index = remaining.remove(position);
        let vertex = vertices[index];
        if vertex.incoming().is_empty() || vertex.outgoing().is_empty() {
            return Err(GenerationError::InvalidChannel(format!(
                "vertex `{}` must have incoming and outgoing edges",
                vertex.name()
            )));
        }
        if vertex.generation().is_none()
            && !(vertex.incoming().len() == 1 && vertex.outgoing().len() == 2)
        {
            return Err(GenerationError::InvalidChannel(format!(
                "vertex `{}` has no generation proposal and is not a two-body decay",
                vertex.name()
            )));
        }
        for edge in vertex.outgoing() {
            if !available.insert(edge.clone()) {
                return Err(GenerationError::InvalidChannel(format!(
                    "edge `{edge}` is produced more than once"
                )));
            }
        }
        plan.push(index);
    }
    Ok(plan)
}

fn validate_common(events: usize, batch_size: usize) -> GenerationResult<()> {
    if events == 0 || batch_size == 0 {
        return Err(GenerationError::InvalidConfiguration(
            "events and batch_size must be nonzero".into(),
        ));
    }
    Ok(())
}
fn validate_bound(bound: f64) -> GenerationResult<()> {
    if !bound.is_finite() || bound <= 0.0 {
        return Err(GenerationError::InvalidConfiguration(
            "envelope must be finite and positive".into(),
        ));
    }
    Ok(())
}
fn validate_p4(name: &str, p4: RealVec4, mass: f64, index: u64) -> GenerationResult<()> {
    if ![p4.x, p4.y, p4.z, p4.t].into_iter().all(f64::is_finite) || p4.t <= 0.0 {
        return Err(GenerationError::Kinematics {
            index,
            source: LadduPhysicsError::invalid_value(
                format!("four-momentum for edge `{name}`"),
                "finite components and positive energy",
                p4,
            ),
        });
    }
    let tolerance = 1e-9 * (1.0 + mass * mass + p4.t * p4.t);
    if (p4.m2() - mass * mass).abs() > tolerance {
        return Err(GenerationError::Kinematics {
            index,
            source: LadduPhysicsError::invalid_relation(format!(
                "edge is off shell: p²={} but mass²={}",
                p4.m2(),
                mass * mass
            )),
        });
    }
    Ok(())
}
fn validate_indexed_conservation(
    vertex: &VertexPlan,
    p4s: &[RealVec4],
    index: u64,
) -> GenerationResult<()> {
    let incoming: RealVec4 = vertex.incoming.iter().map(|&edge| p4s[edge]).sum();
    let outgoing: RealVec4 = vertex.outgoing.iter().map(|&edge| p4s[edge]).sum();
    let residual = incoming - outgoing;
    let scale = incoming.t.abs().max(1.0);
    if residual.t.abs() > 1e-9 * scale
        || residual.x.abs() > 1e-9 * scale
        || residual.y.abs() > 1e-9 * scale
        || residual.z.abs() > 1e-9 * scale
    {
        return Err(GenerationError::Kinematics {
            index,
            source: LadduPhysicsError::invalid_relation(format!(
                "vertex `{}` violates four-momentum conservation by {residual:?}",
                vertex.name
            )),
        });
    }
    Ok(())
}
fn derive_seed(seed: u64, stream: u64, index: u64, object: u64) -> u64 {
    let mut rng = ProposalRng::new(
        seed ^ stream.wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ index.wrapping_mul(0xa076_1d64_78bd_642f)
            ^ object.wrapping_mul(0xe703_7ed1_a0b4_28db),
    );
    rng.next_u64()
}
fn acceptance_uniform(seed: u64, index: u64) -> f64 {
    ProposalRng::new(derive_seed(seed, 2, index, 0)).uniform()
}
fn report(requested: usize, seed: u64, batch_size: usize) -> GenerationReport {
    GenerationReport {
        requested,
        seed,
        batch_size,
        minimum_weight: f64::INFINITY,
        ..GenerationReport::default()
    }
}
fn update_report(report: &mut GenerationReport, events: &[GeneratedEvent]) {
    for event in events {
        report.maximum_weight = report.maximum_weight.max(event.target_weight);
        report.minimum_weight = report.minimum_weight.min(event.target_weight);
        report.sum_weights += event.target_weight;
        report.sum_squared_weights += event.target_weight * event.target_weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laddu_physics::{quantum::ParticleProperties, vectors::RealVec3};

    fn decay_generator() -> ChannelGenerator {
        let mut channel = Channel::new("decay");
        channel
            .edge("parent")
            .properties(&ParticleProperties::unknown().with_mass(2.0))
            .initial_p4(RealVec4::new(0.0, 0.0, 0.0, 2.0));
        channel
            .edge("a")
            .properties(&ParticleProperties::unknown().with_mass(0.2))
            .output();
        channel
            .edge("b")
            .properties(&ParticleProperties::unknown().with_mass(0.4))
            .output();
        channel
            .vertex("decay")
            .incoming(["parent"])
            .outgoing(["a", "b"]);
        ChannelGenerator::new(channel).unwrap()
    }

    #[test]
    fn weighted_generation_is_batch_size_independent() {
        let generator = decay_generator();
        let mut first = WeightedConfig::new(32);
        first.seed = 9;
        first.batch_size = 3;
        let mut second = first;
        second.batch_size = 11;
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (a, _) = one_thread
            .install(|| generator.generate_weighted_dataset(first, None))
            .unwrap();
        let (b, _) = four_threads
            .install(|| generator.generate_weighted_dataset(second, None))
            .unwrap();
        let mut av = Vec::new();
        a.for_each_event(|event| av.push(event.p4(0))).unwrap();
        let mut bv = Vec::new();
        b.for_each_event(|event| bv.push(event.p4(0))).unwrap();
        assert_eq!(av, bv);
    }

    #[test]
    fn unweighted_generation_is_thread_and_batch_size_independent() {
        let generator = decay_generator();
        let model = CompiledModel::from_expr(&laddu_expr::Expr::from(1.0)).unwrap();
        let evaluator = ModelEvaluator::prepare(
            &model,
            model.params().default_values(),
            &Execution::default(),
        )
        .unwrap();
        let mut first = UnweightedConfig::new(32).with_max_proposals(20_000);
        first.seed = 91;
        first.batch_size = 7;
        first.envelope = EnvelopeMode::Strict { max_weight: 1.0 };
        let mut second = first;
        second.batch_size = 29;
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (a, _) = one_thread
            .install(|| generator.generate_unweighted_dataset(first, &evaluator))
            .unwrap();
        let (b, _) = four_threads
            .install(|| generator.generate_unweighted_dataset(second, &evaluator))
            .unwrap();
        let mut av = Vec::new();
        a.for_each_event(|event| av.push(event.p4(0))).unwrap();
        let mut bv = Vec::new();
        b.for_each_event(|event| bv.push(event.p4(0))).unwrap();
        assert_eq!(av, bv);
    }

    #[test]
    fn adaptive_mass_density_is_normalized_and_matches_returned_weight() {
        let proposal = AdaptiveMassProposal {
            base: MassProposal::uniform(0.0, 4.0),
            low: 1.0,
            high: 3.0,
            counts: Arc::from([1.0, 3.0, 7.0, 1.0]),
            defensive_fraction: 0.2,
        };
        let steps = 20_000;
        let dx = 4.0 / steps as f64;
        let integral: f64 = (0..steps)
            .map(|step| {
                let mass = (step as f64 + 0.5) * dx;
                proposal.density(0.0, 4.0, mass).unwrap().unwrap() * dx
            })
            .sum();
        assert!((integral - 1.0).abs() < 1e-10);

        let mut rng = ProposalRng::new(123);
        for _ in 0..1_000 {
            let sampled = proposal.propose(0.0, 4.0, &mut rng).unwrap();
            let density = proposal.density(0.0, 4.0, sampled.mass).unwrap().unwrap();
            assert!((sampled.weight * density - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn strict_envelope_rejects_overflow() {
        let generator = decay_generator();
        let model = CompiledModel::from_expr(&laddu_expr::Expr::from(1.0)).unwrap();
        let evaluator = ModelEvaluator::prepare(
            &model,
            model.params().default_values(),
            &Execution::default(),
        )
        .unwrap();
        assert!(matches!(
            generator.generate_unweighted_dataset(
                UnweightedConfig {
                    envelope: EnvelopeMode::Strict { max_weight: 1e-12 },
                    ..UnweightedConfig::new(2).with_max_proposals(10)
                },
                &evaluator,
            ),
            Err(GenerationError::EnvelopeOverflow { .. })
        ));
    }

    #[test]
    fn unweighted_generation_is_unlimited_by_default() {
        let generator = decay_generator();
        let model = CompiledModel::from_expr(&laddu_expr::Expr::from(1.0)).unwrap();
        let evaluator = ModelEvaluator::prepare(
            &model,
            model.params().default_values(),
            &Execution::default(),
        )
        .unwrap();
        let config = UnweightedConfig {
            envelope: EnvelopeMode::Strict { max_weight: 1.0 },
            ..UnweightedConfig::new(32)
        };
        assert_eq!(config.max_proposals, None);
        let (_, report) = generator
            .generate_unweighted_dataset(config, &evaluator)
            .unwrap();
        assert_eq!(report.produced, 32);
        assert!(report.proposals >= 32);
    }

    #[test]
    fn explicit_proposal_limit_can_exhaust_generation() {
        let generator = decay_generator();
        let model = CompiledModel::from_expr(&laddu_expr::Expr::from(1.0)).unwrap();
        let evaluator = ModelEvaluator::prepare(
            &model,
            model.params().default_values(),
            &Execution::default(),
        )
        .unwrap();
        assert!(matches!(
            generator.generate_unweighted_dataset(
                UnweightedConfig {
                    envelope: EnvelopeMode::Strict { max_weight: 1.0 },
                    ..UnweightedConfig::new(16).with_max_proposals(16)
                },
                &evaluator,
            ),
            Err(GenerationError::Exhausted {
                requested: 16,
                proposals: 16,
                ..
            })
        ));
    }

    #[test]
    fn adaptive_envelope_grows_and_rethins_buffered_events() {
        let generator = decay_generator();
        let model = CompiledModel::from_expr(&laddu_expr::Expr::from(1.0)).unwrap();
        let evaluator = ModelEvaluator::prepare(
            &model,
            model.params().default_values(),
            &Execution::default(),
        )
        .unwrap();
        let mut config = UnweightedConfig::new(16).with_max_proposals(10_000);
        config.envelope = EnvelopeMode::Strict { max_weight: 1e-12 };
        config.envelope_overflow = EnvelopeOverflow::Grow { safety_factor: 1.5 };
        let (dataset, report) = generator
            .generate_unweighted_dataset(config, &evaluator)
            .unwrap();

        let mut events = 0;
        dataset.for_each_event(|_| events += 1).unwrap();
        assert_eq!(events, 16);
        assert_eq!(report.produced, 16);
        assert!(report.envelope_updates >= 1);
        assert!(report.envelope.unwrap() >= report.maximum_weight);
    }

    #[test]
    fn channel_validation_preserves_initial_source_error() {
        let mut channel = Channel::new("decay");
        channel
            .edge("parent")
            .properties(&ParticleProperties::unknown().with_mass(2.0))
            .initial_energy_direction(2.0, RealVec3::default());
        channel
            .edge("a")
            .properties(&ParticleProperties::unknown().with_mass(0.2))
            .output();
        channel
            .edge("b")
            .properties(&ParticleProperties::unknown().with_mass(0.4))
            .output();
        channel
            .vertex("decay")
            .incoming(["parent"])
            .outgoing(["a", "b"])
            .generation(VertexProposal::isotropic_decay());
        assert!(matches!(
            ChannelGenerator::new(channel),
            Err(GenerationError::ChannelValidation {
                source: LadduPhysicsError::InvalidValue { .. },
            })
        ));
    }

    #[test]
    fn channel_is_validated_when_consumed_by_the_generator() {
        let mut channel = Channel::new("invalid");
        channel
            .edge("same")
            .properties(&ParticleProperties::unknown().with_mass(1.0))
            .initial_p4(RealVec4::new(0.0, 0.0, 0.0, 1.0))
            .output();
        channel
            .vertex("duplicate")
            .incoming(["same"])
            .outgoing(["same"]);

        assert!(matches!(
            ChannelGenerator::new(channel),
            Err(GenerationError::ChannelValidation {
                source: LadduPhysicsError::InvalidRelation { .. },
            })
        ));
    }

    #[test]
    fn momentum_initial_state_requires_particle_mass_metadata() {
        let mut channel = Channel::new("decay");
        channel
            .edge("parent")
            .initial_momentum(RealVec3::new(0.0, 0.0, 0.0));
        channel
            .edge("a")
            .properties(&ParticleProperties::unknown().with_mass(0.2))
            .output();
        channel
            .edge("b")
            .properties(&ParticleProperties::unknown().with_mass(0.4))
            .output();
        channel
            .vertex("decay")
            .incoming(["parent"])
            .outgoing(["a", "b"])
            .generation(VertexProposal::isotropic_decay());
        assert!(matches!(
            ChannelGenerator::new(channel),
            Err(GenerationError::ChannelValidation {
                source: LadduPhysicsError::InvalidRelation { .. },
            })
        ));
    }

    #[test]
    fn initial_energy_sources_sample_energy_and_apply_the_proposal_correction() {
        let mut channel = Channel::new("initial state");
        channel
            .edge("beam")
            .properties(&ParticleProperties::unknown().with_mass(0.5));
        let source = InitialMomentum::energy_source_direction(
            ScalarSource::uniform(1.0, 3.0),
            RealVec3::z(),
        );
        let properties = channel.particle("beam").unwrap();
        source.validate("beam", Some(properties)).unwrap();
        let sampled = source
            .sample("beam", Some(properties), &mut ProposalRng::new(43))
            .unwrap();

        assert!((1.0..3.0).contains(&sampled.p4.e()));
        assert_eq!(sampled.weight, 2.0);
    }

    #[test]
    fn generated_scalar_columns_are_emitted_and_available_to_models() {
        let generator = decay_generator()
            .with_scalar("aux", ScalarSource::constant(2.0))
            .unwrap();
        let model = CompiledModel::from_expr(&laddu_expr::event_scalar("aux")).unwrap();
        let evaluator = ModelEvaluator::prepare(
            &model,
            model.params().default_values(),
            &Execution::default(),
        )
        .unwrap();
        let (dataset, _) = generator
            .generate_weighted_dataset(WeightedConfig::new(8), Some(&evaluator))
            .unwrap();

        assert_eq!(dataset.schema().unwrap().scalar_index("aux"), Some(0));
        dataset
            .for_each_event(|event| {
                assert_eq!(event.scalar_named("aux"), Some(2.0));
                assert!(event.weight() > 0.0);
            })
            .unwrap();
    }
}
