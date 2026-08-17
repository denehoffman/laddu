//! Channel-aware Monte Carlo event generation.

use std::{
    collections::{HashMap, HashSet},
    mem::size_of,
    sync::Arc,
};

use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::{
    BatchLayout,
    data::{Dataset, EventBatch},
    io::{EventSink, WritePlan, memory::MemorySink},
    schema::{Precision as DataPrecision, Schema},
};
use laddu_expr::{ExprNode, parameters::ParamValues};
use laddu_memory::{MemoryFitRequest, MemoryFootprint};
use laddu_physics::{
    LadduPhysicsError,
    channel::Channel,
    generation::{PiecewiseDensity, proven_two_body_decay_weight},
    vectors::RealVec4,
};
use laddu_runtime::{
    Execution, MemoryBudget, MemoryDecision, MemoryLease, MemoryState, PreparedModel,
};
use maryada::{EnclosureOps, GlobalMinimizer, GlobalMinimizerOptions, Interval, IntervalOps};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use thiserror::Error;

pub use laddu_physics::generation::{
    AdaptiveTwoBodyDecay, InitialMomentum, InitialMomentumResult, MassProposal, MassProposalResult,
    NamedMass, NamedMomentum, ProposalResult, ProposalRng, ScalarProposalResult, ScalarSource,
    TComponent, TDistribution, TwoBodyScattering, VertexProposal,
};

/// Result type returned by event-generation operations.
pub type GenerationResult<T> = Result<T, GenerationError>;

/// Error produced while configuring or running event generation.
#[derive(Debug, Error)]
pub enum GenerationError {
    /// The channel topology or output definition is invalid.
    #[error("invalid generation channel: {0}")]
    InvalidChannel(String),
    /// Physics-level validation of the channel failed.
    #[error("channel validation failed: {source}")]
    ChannelValidation {
        /// Underlying channel validation error.
        #[source]
        source: LadduPhysicsError,
    },
    /// A generation option is inconsistent or outside its valid range.
    #[error("invalid generation configuration: {0}")]
    InvalidConfiguration(String),
    /// Initial-state momentum generation failed for a proposal.
    #[error("initial-state proposal {index} failed: {source}")]
    InitialState {
        /// Global proposal index.
        index: u64,
        /// Underlying physics error.
        #[source]
        source: LadduPhysicsError,
    },
    /// An intermediate mass proposal failed.
    #[error("mass proposal for edge `{edge}` at proposal {index} failed: {source}")]
    MassProposal {
        /// Global proposal index.
        index: u64,
        /// Channel edge being sampled.
        edge: String,
        /// Underlying physics error.
        #[source]
        source: LadduPhysicsError,
    },
    /// A vertex kinematics proposal failed.
    #[error("vertex `{vertex}` at proposal {index} failed: {source}")]
    VertexProposal {
        /// Global proposal index.
        index: u64,
        /// Channel vertex being sampled.
        vertex: String,
        /// Underlying physics error.
        #[source]
        source: LadduPhysicsError,
    },
    /// A derived scalar proposal failed.
    #[error("scalar column `{column}` at proposal {index} failed: {source}")]
    ScalarProposal {
        /// Global proposal index.
        index: u64,
        /// Scalar column being sampled.
        column: String,
        /// Underlying physics error.
        #[source]
        source: LadduPhysicsError,
    },
    /// Final-state kinematic validation failed.
    #[error("kinematic validation at proposal {index} failed: {source}")]
    Kinematics {
        /// Global proposal index.
        index: u64,
        /// Underlying physics error.
        #[source]
        source: LadduPhysicsError,
    },
    /// Target-model evaluation failed.
    #[error("model evaluation failed: {0}")]
    Model(String),
    /// A proposal exceeded the strict rejection-sampling envelope.
    #[error("target weight {weight} exceeds envelope {envelope} at proposal {index}")]
    EnvelopeOverflow {
        /// Global proposal index.
        index: u64,
        /// Target weight of the overflowing proposal.
        weight: f64,
        /// Active envelope bound.
        envelope: f64,
    },
    /// The proposal limit was reached before enough events were accepted.
    #[error(
        "accepted {accepted} events after exhausting {proposals} proposals (requested {requested})"
    )]
    Exhausted {
        /// Requested event count.
        requested: usize,
        /// Number of accepted events.
        accepted: usize,
        /// Number of production proposals attempted.
        proposals: usize,
    },
    /// Dataset input or output failed.
    #[error(transparent)]
    Data(#[from] laddu_data::LadduDataError),
    /// Model preparation or execution failed.
    #[error(transparent)]
    Runtime(#[from] laddu_runtime::RuntimeError),
    /// A physics-level proposal or validation failed.
    #[error(transparent)]
    Physics(#[from] LadduPhysicsError),
}

/// Configuration for weighted event generation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WeightedConfig {
    /// Number of events to generate.
    pub events: usize,
    /// Memory available to proposal and output staging.
    pub memory: MemoryBudget,
    /// Deterministic random seed.
    pub seed: u64,
    /// Whether to include diagnostic weight columns.
    pub diagnostics: bool,
}

impl WeightedConfig {
    /// Creates a configuration for `events` with default batching and seed.
    pub fn new(events: usize) -> Self {
        Self {
            events,
            memory: MemoryBudget::Auto,
            seed: 0,
            diagnostics: false,
        }
    }
}

/// Configuration for rejection-sampled unweighted event generation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct UnweightedConfig {
    /// Number of accepted events to generate.
    pub events: usize,
    /// Optional safeguard limiting production proposals.
    ///
    /// `None` allows generation to continue until the requested event count is
    /// reached. Pilot proposals are not included in this limit.
    pub max_proposals: Option<usize>,
    /// Memory available to proposal and output staging.
    pub memory: MemoryBudget,
    /// Deterministic random seed.
    pub seed: u64,
    /// Whether to include diagnostic weight columns.
    pub diagnostics: bool,
    /// Strategy used to establish the rejection-sampling envelope.
    pub envelope: EnvelopeMode,
    /// Policy applied when a proposal exceeds the active envelope.
    pub envelope_overflow: EnvelopeOverflow,
}

impl UnweightedConfig {
    /// Create an unweighted-generation configuration without a proposal limit.
    pub fn new(events: usize) -> Self {
        Self {
            events,
            max_proposals: None,
            memory: MemoryBudget::Auto,
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
    Grow {
        /// Factor by which the observed overflow expands the envelope.
        safety_factor: f64,
    },
}

/// Strategy used to establish a rejection-sampling envelope.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum EnvelopeMode {
    /// Use a caller-supplied fixed maximum weight.
    Strict {
        /// Fixed upper bound for target weights.
        max_weight: f64,
    },
    /// Estimate an envelope from pilot proposals.
    ///
    /// Density-aware built-in proposals may use one deterministic pilot pass
    /// for importance adaptation and a second pass for the final envelope.
    Pilot {
        /// Number of pilot proposals used to estimate the maximum.
        proposals: usize,
        /// Multiplier applied to the maximum pilot weight.
        safety_factor: f64,
    },
    /// Prove a fixed envelope for the phase-space proposal weight using
    /// outward-rounded interval arithmetic.
    ///
    /// This mode is valid only for unit-model generation.
    ProvenPhaseSpace,
}

impl Default for EnvelopeMode {
    fn default() -> Self {
        Self::Pilot {
            proposals: 10_000,
            safety_factor: 2.0,
        }
    }
}

/// Source from which the final rejection envelope was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvelopeKind {
    /// A caller-supplied strict bound.
    Strict,
    /// A bound estimated from pilot proposals.
    Pilot,
    /// A maryada interval enclosure of the unit-model phase-space weight.
    ProvenPhaseSpace,
}

/// Diagnostics produced while proving a unit-model phase-space envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProvenEnvelopeReport {
    /// Full nonnegative enclosure of every phase-space proposal weight.
    pub weight_interval: Interval,
    /// Upper endpoint used as the rejection-sampling maximum.
    pub maximum_weight: f64,
    /// Number of continuous proposal coordinates represented by the domain.
    pub continuous_dimensions: usize,
    /// Number of analytical piecewise regions represented by the enclosure.
    pub piecewise_regions: usize,
    /// Number of branch-and-bound subdivisions used to tighten the enclosure.
    pub subdivisions: usize,
}

impl ProvenEnvelopeReport {
    /// Return the outward-rounded lower and upper weight endpoints.
    pub fn weight_bounds(&self) -> (f64, f64) {
        self.weight_interval.bounds()
    }
}

/// Diagnostics and aggregate statistics from an event-generation run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenerationReport {
    /// Requested event count.
    pub requested: usize,
    /// Produced event count.
    pub produced: usize,
    /// Number of production proposals.
    pub proposals: usize,
    /// Number of pilot proposals.
    pub pilot_proposals: usize,
    /// Number of rejected production proposals.
    pub rejected: usize,
    /// Final rejection envelope, if unweighting was used.
    pub envelope: Option<f64>,
    /// Method used to establish the envelope.
    pub envelope_kind: Option<EnvelopeKind>,
    /// Number of adaptive envelope expansions.
    pub envelope_updates: usize,
    /// Proven interval endpoints when the phase-space envelope was established analytically.
    #[serde(default)]
    pub proven_weight_interval: Option<(f64, f64)>,
    /// Continuous proposal-coordinate count for a proven phase-space envelope.
    #[serde(default)]
    pub proven_continuous_dimensions: Option<usize>,
    /// Analytical piecewise-region count for a proven phase-space envelope.
    #[serde(default)]
    pub proven_piecewise_regions: Option<usize>,
    /// Adaptive subdivision count for a proven phase-space envelope.
    #[serde(default)]
    pub proven_subdivisions: Option<usize>,
    /// Maximum target weight encountered.
    pub maximum_weight: f64,
    /// Minimum target weight encountered.
    pub minimum_weight: f64,
    /// Sum of generated target weights.
    pub sum_weights: f64,
    /// Sum of squared generated target weights.
    pub sum_squared_weights: f64,
    /// Random seed used for the run.
    pub seed: u64,
    /// Memory-derived internal event chunk size.
    pub chunk_events: usize,
    /// Estimated peak tracked bytes for one generation chunk.
    pub estimated_peak_bytes: u64,
    /// Actual tracked high-water bytes when generation used an execution pool.
    pub actual_high_water_bytes: Option<u64>,
}

impl GenerationReport {
    /// Returns the fraction of production proposals that were accepted.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.produced as f64 / self.proposals as f64
        }
    }
}

/// A prepared model and parameters used to weight generated proposals.
#[derive(Clone, Debug)]
pub struct ModelEvaluator {
    prepared: PreparedModel,
    params: ParamValues,
    required_scalars: HashSet<String>,
    execution: Execution,
}

impl ModelEvaluator {
    /// Prepares a compiled model for generation-time batch evaluation.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when model preparation for the selected
    /// execution backend fails.
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
            execution: execution.clone(),
        })
    }

    /// Evaluate the positive-real model value for every event in a batch.
    ///
    /// This is useful for projecting a fitted model over weighted Monte Carlo
    /// without regenerating events.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when batch evaluation fails or a model value
    /// is non-finite or not strictly positive.
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

/// Generates kinematically valid events for a physics channel.
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
    density: PiecewiseDensity,
    defensive_fraction: f64,
}

impl AdaptiveMassProposal {
    fn truncated_total(&self, minimum: f64, maximum: f64) -> f64 {
        self.density.truncated_total(minimum, maximum)
    }

    fn adaptive_density(&self, minimum: f64, maximum: f64, mass: f64) -> f64 {
        self.density.density(minimum, maximum, mass)
    }

    fn sample_adaptive(&self, minimum: f64, maximum: f64, rng: &mut ProposalRng) -> Option<f64> {
        self.density.sample(minimum, maximum, rng)
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

#[derive(Clone, Copy, Debug)]
struct IntervalInitialMomentum {
    energy: Interval,
    momentum: [Interval; 3],
    mass: f64,
    weight: Interval,
    continuous_dimensions: usize,
    piecewise_regions: usize,
}

#[derive(Clone, Copy, Debug)]
struct IntervalSample {
    value: Interval,
    weight: Interval,
    continuous_dimensions: usize,
    piecewise_regions: usize,
}

#[derive(Clone, Copy, Debug)]
struct GenerationMemoryUse {
    event_limit: usize,
    resident_events: usize,
    retained_output_events: usize,
}

impl ChannelGenerator {
    fn generation_memory(
        &self,
        schema: &Schema,
        budget: MemoryBudget,
        model: Option<&ModelEvaluator>,
        usage: GenerationMemoryUse,
        label: &str,
    ) -> GenerationResult<(MemoryDecision, MemoryLease)> {
        let generated_layout = BatchLayout::new(
            schema.n_p4s(),
            schema.n_scalars(),
            schema.has_weight(),
            false,
        );
        let generated = generated_layout
            .footprint(DataPrecision::F64)
            .and_then(|footprint| {
                footprint.checked_add(MemoryFootprint::per_event(
                    u64::try_from(size_of::<GeneratedEvent>())
                        .map_err(|_| laddu_memory::FootprintOverflow::Conversion)?,
                ))
            })
            .and_then(|footprint| {
                footprint.checked_add(MemoryFootprint::per_event(
                    u64::try_from(4 * size_of::<f64>())
                        .map_err(|_| laddu_memory::FootprintOverflow::Conversion)?,
                ))
            })
            .map_err(|error| {
                GenerationError::Runtime(laddu_runtime::RuntimeError::Data(format!(
                    "generation working-set overflow: {error}"
                )))
            })?;
        let output = generated_layout
            .schema_footprint(DataPrecision::F64)
            .map_err(|error| {
                GenerationError::Runtime(laddu_runtime::RuntimeError::Data(format!(
                    "output working-set overflow: {error}"
                )))
            })?;
        let bytes_per_event = generated.checked_add(output).map_err(|error| {
            GenerationError::Runtime(laddu_runtime::RuntimeError::Data(format!(
                "generation working-set overflow: {error}"
            )))
        })?;
        let fixed_bytes = generated
            .checked_peak_bytes(usage.resident_events)
            .and_then(|resident| {
                output
                    .checked_peak_bytes(usage.retained_output_events)
                    .and_then(|retained| {
                        resident
                            .checked_add(retained)
                            .ok_or(laddu_memory::FootprintOverflow::Addition)
                    })
            })
            .map_err(|error| {
                GenerationError::Runtime(laddu_runtime::RuntimeError::Data(format!(
                    "generation working-set overflow: {error}"
                )))
            })?;
        let state = model
            .map(|model| model.execution.memory_state().clone())
            .unwrap_or_else(MemoryState::current);
        state.refresh();
        let operation_cap = budget
            .resolve(&state.host())
            .map_err(laddu_runtime::RuntimeError::from)?;
        let owned_pool;
        let pool = if let Some(model) = model {
            model.execution.host_memory()
        } else {
            owned_pool = state
                .pool("host", budget)
                .map_err(laddu_runtime::RuntimeError::from)?;
            &owned_pool
        };
        let available = pool.remaining().min(operation_cap);
        let decision = MemoryFitRequest {
            label: label.into(),
            footprint: MemoryFootprint::new(fixed_bytes, bytes_per_event.bytes_per_event),
            available_bytes: available,
            event_limit: usage.event_limit,
            strategy: "memory-derived generation".into(),
        }
        .evaluate()
        .map_err(laddu_runtime::RuntimeError::from)?;
        let lease = pool
            .reserve(decision.estimated_peak_bytes)
            .map_err(laddu_runtime::RuntimeError::from)?;
        if let Some(model) = model {
            model.execution.record_memory_decision(decision.clone());
        }
        Ok((decision, lease))
    }

    /// Validates a channel and constructs its topological generation plan.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when channel topology, edge metadata,
    /// particle masses, output names, or vertex proposals are invalid.
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

    /// Adds a generated scalar column and returns the updated generator.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when `name` is empty, reserved, duplicated,
    /// or conflicts with an edge name.
    pub fn with_scalar(
        mut self,
        name: impl Into<String>,
        source: ScalarSource,
    ) -> GenerationResult<Self> {
        self.add_scalar(name, source)?;
        Ok(self)
    }

    /// Adds a generated scalar column.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when `name` is empty, reserved, duplicated,
    /// or conflicts with an edge name.
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

    /// Prove an upper envelope for the model-less phase-space proposal weight.
    ///
    /// The returned interval is outward-rounded by maryada. A finite upper
    /// endpoint is required before it can be used for rejection sampling.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when a proposal domain is invalid, a
    /// scattering vertex does not consume initial edges, or interval
    /// propagation cannot establish a finite positive upper endpoint.
    pub fn phase_space_envelope(&self) -> GenerationResult<ProvenEnvelopeReport> {
        let mut initials = HashMap::new();
        let mut weight = Interval::ONE;
        let mut continuous_dimensions = 0_usize;
        let mut piecewise_regions = 1_usize;

        for &edge_index in &self.root_indices {
            let edge = &self.edges[edge_index];
            let mass = match edge.mass {
                EdgeMassPlan::Fixed(mass) => mass,
                EdgeMassPlan::Proposed(_) => {
                    return Err(GenerationError::InvalidConfiguration(format!(
                        "initial edge `{}` cannot use a generated mass",
                        edge.name
                    )));
                }
            };
            let source = edge.initial.as_ref().ok_or_else(|| {
                GenerationError::InvalidConfiguration(format!(
                    "initial edge `{}` has no momentum source",
                    edge.name
                ))
            })?;
            let initial = interval_initial_momentum(source, mass)?;
            weight *= initial.weight;
            continuous_dimensions =
                continuous_dimensions.saturating_add(initial.continuous_dimensions);
            piecewise_regions = piecewise_regions.saturating_mul(initial.piecewise_regions);
            initials.insert(edge_index, initial);
        }

        let root_s = interval_invariant_mass(
            self.root_indices
                .iter()
                .map(|edge| initials[edge])
                .collect::<Vec<_>>()
                .as_slice(),
        );
        if root_s.is_empty() || !root_s.sup().is_finite() || root_s.sup() <= 0.0 {
            return Err(GenerationError::InvalidConfiguration(format!(
                "initial-state invariant-mass enclosure {root_s} is not finite and positive"
            )));
        }

        // Validate the mass supports once against the root domain. The
        // branch-and-bound evaluator below narrows these supports for each
        // root-invariant-mass and generated-mass subdomain.
        let mut mass_dimensions = vec![None; self.edges.len()];
        let mut domain = vec![root_s];
        for edge in &self.edges {
            match edge.mass {
                EdgeMassPlan::Fixed(mass)
                | EdgeMassPlan::Proposed(MassProposal::Fixed { mass }) => {
                    if mass > root_s.sup() {
                        return Err(GenerationError::InvalidConfiguration(format!(
                            "fixed generated mass {mass} for edge `{}` exceeds the initial invariant-mass enclosure {root_s}",
                            edge.name
                        )));
                    }
                }
                EdgeMassPlan::Proposed(MassProposal::Uniform { low, high }) => {
                    let support_low = low.max(0.0);
                    let support_high = high.min(root_s.sup());
                    if !support_low.is_finite()
                        || !support_high.is_finite()
                        || support_high <= support_low
                    {
                        return Err(GenerationError::InvalidConfiguration(format!(
                            "uniform generated mass for edge `{}` has no finite support inside [0, {}]",
                            edge.name,
                            root_s.sup()
                        )));
                    }
                    continuous_dimensions = continuous_dimensions.saturating_add(1);
                }
            }
        }

        // Re-index generated-mass coordinates by edge position. Keeping the
        // mapping separate from the physical mass intervals prevents the
        // latent variables from being mistaken for independent masses.
        for (edge_index, edge) in self.edges.iter().enumerate() {
            if matches!(
                edge.mass,
                EdgeMassPlan::Proposed(MassProposal::Uniform { .. })
            ) {
                mass_dimensions[edge_index] = Some(domain.len());
                domain.push(Interval::new(0.0, 1.0));
            }
        }

        let mut transfer_dimensions = vec![None; self.vertices.len()];
        for vertex in &self.vertices {
            let (dimensions, regions) = vertex.proposal.proven_domain_metadata();
            continuous_dimensions = continuous_dimensions.saturating_add(dimensions);
            piecewise_regions = piecewise_regions.saturating_mul(regions);
        }
        for (vertex_index, vertex) in self.vertices.iter().enumerate() {
            if matches!(vertex.proposal, VertexProposal::TwoBodyScattering { .. }) {
                transfer_dimensions[vertex_index] = Some(domain.len());
                // The transfer coordinate is normalized to the configured
                // physical support. Angular coordinates remain analytically
                // enclosed by the physics interval formulas.
                domain.push(Interval::new(0.0, 1.0));
            }
        }
        for (_, source) in &self.scalar_sources {
            let sample = interval_scalar_sample(source)?;
            weight *= sample.weight;
            continuous_dimensions =
                continuous_dimensions.saturating_add(sample.continuous_dimensions);
            piecewise_regions = piecewise_regions.saturating_mul(sample.piecewise_regions);
        }

        let static_weight = weight;
        // Search the derived root invariant mass together with normalized
        // generated-mass and transfer coordinates. Dependent physical masses
        // and transfer values are reconstructed inside each box so the search
        // does not treat them as independent kinematic quantities.
        let evaluate = |domain: &Vec<Interval>| -> GenerationResult<Interval> {
            let root_s = domain.first().copied().ok_or_else(|| {
                GenerationError::InvalidConfiguration(
                    "branch-and-bound domain has no root invariant-mass coordinate".into(),
                )
            })?;
            let mut weight = static_weight;
            let mut masses = Vec::with_capacity(self.edges.len());
            for (edge_index, edge) in self.edges.iter().enumerate() {
                match edge.mass {
                    EdgeMassPlan::Fixed(mass)
                    | EdgeMassPlan::Proposed(MassProposal::Fixed { mass }) => {
                        masses.push(Interval::from(mass));
                    }
                    EdgeMassPlan::Proposed(MassProposal::Uniform { low, high }) => {
                        let support_low = low.max(0.0);
                        let support_high = high.min(root_s.sup());
                        let minimum_width = (high.min(root_s.inf()) - support_low).max(0.0);
                        let maximum_width = support_high - support_low;
                        if maximum_width <= 0.0 {
                            return Ok(Interval::EMPTY);
                        }
                        let support_width = Interval::new(minimum_width, maximum_width);
                        weight *= support_width;
                        let fraction = domain[mass_dimensions[edge_index].ok_or_else(|| {
                            GenerationError::InvalidConfiguration(format!(
                                "missing branch coordinate for generated mass edge `{}`",
                                edge.name
                            ))
                        })?];
                        masses.push(Interval::from(support_low) + fraction * support_width);
                    }
                }
            }

            for (vertex_index, vertex) in self.vertices.iter().enumerate() {
                let vertex_weight = match &vertex.proposal {
                    VertexProposal::TwoBodyDecay => {
                        if vertex.incoming.len() != 1 || vertex.outgoing.len() != 2 {
                            return Err(GenerationError::InvalidConfiguration(format!(
                                "vertex `{}` is not a one-to-two decay",
                                vertex.name
                            )));
                        }
                        proven_two_body_decay_weight(
                            masses[vertex.incoming[0]],
                            masses[vertex.outgoing[0]],
                            masses[vertex.outgoing[1]],
                        )
                    }
                    VertexProposal::TwoBodyScattering { proposal } => {
                        if vertex.incoming.len() != 2
                            || vertex.outgoing.len() != 2
                            || vertex
                                .incoming
                                .iter()
                                .any(|edge| !self.root_indices.contains(edge))
                        {
                            return Err(GenerationError::InvalidConfiguration(format!(
                                "proven two-body scattering at vertex `{}` currently requires two initial incoming edges",
                                vertex.name
                            )));
                        }
                        proposal.proven_weight_bound_for_transfer(
                            root_s,
                            [
                                (
                                    self.edges[vertex.incoming[0]].name.as_str(),
                                    masses[vertex.incoming[0]],
                                ),
                                (
                                    self.edges[vertex.incoming[1]].name.as_str(),
                                    masses[vertex.incoming[1]],
                                ),
                            ],
                            [
                                (
                                    self.edges[vertex.outgoing[0]].name.as_str(),
                                    masses[vertex.outgoing[0]],
                                ),
                                (
                                    self.edges[vertex.outgoing[1]].name.as_str(),
                                    masses[vertex.outgoing[1]],
                                ),
                            ],
                            domain[transfer_dimensions[vertex_index].ok_or_else(|| {
                                GenerationError::InvalidConfiguration(format!(
                                    "missing branch coordinate for transfer vertex `{}`",
                                    vertex.name
                                ))
                            })?],
                        )?
                    }
                };
                weight *= vertex_weight;
            }
            Ok(weight)
        };

        let coarse_weight = evaluate(&domain)?;
        let scale = coarse_weight.sup().abs().max(1.0);
        let domain_tolerance = 1.0 / 1_024.0;
        let options = GlobalMinimizerOptions {
            value_tolerance: Some(scale * 1.0e-8),
            domain_tolerance: Some(domain_tolerance),
            gap_tolerance: Some(scale * 1.0e-8),
            max_steps: Some(1_024),
        };
        let mut minimizer = GlobalMinimizer::with_options(
            domain,
            move |domain: &Vec<Interval>| evaluate(domain).map(|value| -value),
            options,
        );
        let result = minimizer.solve().map_err(|error| {
            GenerationError::InvalidConfiguration(format!(
                "branch-and-bound phase-space envelope failed: {error}"
            ))
        })?;
        let upper = -result.minimum.inf();
        debug_assert!(upper <= coarse_weight.sup() * (1.0 + 1.0e-12));
        if result.minimum.is_empty() || !upper.is_finite() || upper <= 0.0 {
            return Err(GenerationError::InvalidConfiguration(format!(
                "phase-space proposal-weight enclosure has no finite positive upper endpoint: {}",
                result.minimum
            )));
        }
        let weight_interval = Interval::new(0.0, upper);
        Ok(ProvenEnvelopeReport {
            weight_interval,
            maximum_weight: upper,
            continuous_dimensions,
            piecewise_regions,
            subdivisions: result.branched,
        })
    }

    /// Generates weighted events and writes them to `sink`.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when configuration, proposal generation,
    /// model evaluation, batch construction, or sink I/O fails.
    pub fn generate_weighted_to(
        &self,
        config: WeightedConfig,
        model: Option<&ModelEvaluator>,
        sink: &mut dyn EventSink,
    ) -> GenerationResult<GenerationReport> {
        validate_common(config.events)?;
        let schema = self.output_schema(true, config.diagnostics)?;
        let (decision, _memory) = self.generation_memory(
            &schema,
            config.memory,
            model,
            GenerationMemoryUse {
                event_limit: config.events,
                resident_events: 0,
                retained_output_events: if sink.retains_batches() {
                    config.events
                } else {
                    0
                },
            },
            "weighted generation",
        )?;
        sink.begin(Arc::clone(&schema), WritePlan::default())?;
        let result = (|| -> GenerationResult<GenerationReport> {
            let mut report = report(config.events, config.seed, &decision);
            let work_batch = decision.chunk_events.max(1);
            for start in (0..config.events).step_by(work_batch) {
                let count = work_batch.min(config.events - start);
                let mut events = self.propose_range(start as u64, count, config.seed, 0)?;
                self.apply_model(&mut events, model)?;
                update_report(&mut report, &events);
                report.proposals += events.len();
                report.produced += events.len();
                for chunk in events.chunks(decision.chunk_events.max(1)) {
                    let batch =
                        self.output_batch(chunk, Arc::clone(&schema), true, config.diagnostics)?;
                    sink.write_batch(&batch)?;
                }
            }
            sink.finish()?;
            Ok(report)
        })();

        if result.is_err() {
            // The generation error is authoritative even if backend cleanup
            // also fails. Aborted files are intentionally left in place.
            let _ = sink.abort();
        }

        result
    }

    /// Generates rejection-sampled unweighted events and writes them to `sink`.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when configuration, envelope estimation,
    /// proposal generation, model evaluation, rejection sampling, or sink I/O
    /// fails.
    pub fn generate_unweighted_to(
        &self,
        config: UnweightedConfig,
        model: Option<&ModelEvaluator>,
        sink: &mut dyn EventSink,
    ) -> GenerationResult<GenerationReport> {
        validate_common(config.events)?;
        if matches!(config.envelope, EnvelopeMode::ProvenPhaseSpace) && model.is_some() {
            return Err(GenerationError::InvalidConfiguration(
                "the proven phase-space envelope is valid only when no model is supplied".into(),
            ));
        }
        if matches!(config.envelope, EnvelopeMode::ProvenPhaseSpace)
            && !matches!(config.envelope_overflow, EnvelopeOverflow::Error)
        {
            return Err(GenerationError::InvalidConfiguration(
                "the proven phase-space envelope requires envelope_overflow=Error".into(),
            ));
        }
        if config
            .max_proposals
            .is_some_and(|max_proposals| max_proposals < config.events)
        {
            return Err(GenerationError::InvalidConfiguration(
                "max_proposals must be at least the requested event count".into(),
            ));
        }
        let mut adaptations = None;
        let schema = self.output_schema(false, config.diagnostics)?;
        let pilot_limit = match config.envelope {
            EnvelopeMode::Pilot { proposals, .. } => proposals,
            EnvelopeMode::Strict { .. } | EnvelopeMode::ProvenPhaseSpace => 0,
        };
        let (decision, _memory) = self.generation_memory(
            &schema,
            config.memory,
            model,
            GenerationMemoryUse {
                event_limit: config.events.max(pilot_limit),
                resident_events: if matches!(
                    config.envelope_overflow,
                    EnvelopeOverflow::Grow { .. }
                ) {
                    config.events
                } else {
                    0
                },
                retained_output_events: if sink.retains_batches() {
                    config.events
                } else {
                    0
                },
            },
            "unweighted generation",
        )?;
        if pilot_limit > decision.chunk_events {
            return Err(GenerationError::InvalidConfiguration(format!(
                "pilot sample requires {pilot_limit} simultaneously resident proposals, but the \
                 memory budget fits {}; increase the budget or reduce pilot_proposals",
                decision.chunk_events
            )));
        }
        let mut proven_report = None;
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
                self.apply_model(&mut adaptation_pilot, model)?;
                let learned = self.learn_mass_adaptations(&adaptation_pilot)?;
                let has_adaptation = learned.masses.iter().any(Option::is_some)
                    || learned.vertices.iter().any(Option::is_some);
                let mut envelope_pilot = if has_adaptation {
                    drop(adaptation_pilot);
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
                    self.apply_model(&mut envelope_pilot, model)?;
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
            EnvelopeMode::ProvenPhaseSpace => {
                let report = self.phase_space_envelope()?;
                let maximum = report.maximum_weight;
                proven_report = Some(report);
                (maximum, EnvelopeKind::ProvenPhaseSpace, 0)
            }
        };
        if let EnvelopeOverflow::Grow { safety_factor } = config.envelope_overflow
            && (!safety_factor.is_finite() || safety_factor <= 1.0)
        {
            return Err(GenerationError::InvalidConfiguration(
                "envelope growth safety_factor must be finite and greater than one".into(),
            ));
        }
        sink.begin(Arc::clone(&schema), WritePlan::default())?;
        let result = (|| -> GenerationResult<GenerationReport> {
            let mut report = report(config.events, config.seed, &decision);
            report.envelope = Some(bound);
            report.envelope_kind = Some(kind);
            report.pilot_proposals = pilot_count;
            if let Some(proven) = proven_report {
                report.proven_weight_interval = Some(proven.weight_interval.bounds());
                report.proven_continuous_dimensions = Some(proven.continuous_dimensions);
                report.proven_piecewise_regions = Some(proven.piecewise_regions);
                report.proven_subdivisions = Some(proven.subdivisions);
            }
            let mut proposal_index = 0_usize;
            let mut buffered = Vec::new();
            let work_batch = decision.chunk_events.max(1);
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
                self.apply_model(&mut events, model)?;
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
                        for chunk in accepted.chunks(decision.chunk_events.max(1)) {
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
                for events in buffered.chunks(decision.chunk_events.max(1)) {
                    let batch =
                        self.output_batch(events, Arc::clone(&schema), false, config.diagnostics)?;
                    sink.write_batch(&batch)?;
                }
            }
            sink.finish()?;
            Ok(report)
        })();

        if result.is_err() {
            let _ = sink.abort();
        }

        result
    }

    /// Generates weighted events into an in-memory dataset.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when weighted generation fails or generated
    /// batches cannot form a valid dataset.
    pub fn generate_weighted_dataset(
        &self,
        config: WeightedConfig,
        model: Option<&ModelEvaluator>,
    ) -> GenerationResult<(Dataset, GenerationReport)> {
        let mut sink = MemorySink::new();
        let report = self.generate_weighted_to(config, model, &mut sink)?;
        Ok((Dataset::from_batches(sink.into_batches())?, report))
    }

    /// Generates unweighted events into an in-memory dataset.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError`] when unweighted generation fails or
    /// generated batches cannot form a valid dataset.
    pub fn generate_unweighted_dataset(
        &self,
        config: UnweightedConfig,
        model: Option<&ModelEvaluator>,
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
                density: PiecewiseDensity::uniform(low, high, counts.into())
                    .map_err(GenerationError::Physics)?,
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

fn interval_scalar_sample(source: &ScalarSource) -> GenerationResult<IntervalSample> {
    match source {
        ScalarSource::Constant(value) => {
            source.support()?;
            Ok(IntervalSample {
                value: Interval::from(*value),
                weight: Interval::ONE,
                continuous_dimensions: 0,
                piecewise_regions: 1,
            })
        }
        ScalarSource::Uniform { low, high } => {
            source.support()?;
            Ok(IntervalSample {
                value: Interval::new(*low, *high),
                weight: Interval::from(high - low),
                continuous_dimensions: 1,
                piecewise_regions: 1,
            })
        }
        ScalarSource::Histogram(histogram) => {
            let (low, high) = source.support()?;
            let total: f64 = histogram.counts().iter().sum();
            let mut minimum_inverse = f64::INFINITY;
            let mut maximum_inverse = 0.0_f64;
            let mut regions = 0_usize;
            for (count, edges) in histogram
                .counts()
                .iter()
                .zip(histogram.bin_edges().windows(2))
            {
                if *count <= 0.0 {
                    continue;
                }
                let width = Interval::from(edges[1]) - Interval::from(edges[0]);
                let density = Interval::from(*count) / (width * total);
                let inverse_density = density.recip();
                minimum_inverse = minimum_inverse.min(inverse_density.inf());
                maximum_inverse = maximum_inverse.max(inverse_density.sup());
                regions += 1;
            }
            if !minimum_inverse.is_finite()
                || !maximum_inverse.is_finite()
                || maximum_inverse <= 0.0
            {
                return Err(GenerationError::InvalidConfiguration(
                    "histogram source has no finite positive-density region".into(),
                ));
            }
            Ok(IntervalSample {
                value: Interval::new(low, high),
                weight: Interval::new(minimum_inverse, maximum_inverse),
                continuous_dimensions: 1,
                piecewise_regions: regions,
            })
        }
    }
}

fn interval_initial_momentum(
    source: &InitialMomentum,
    mass: f64,
) -> GenerationResult<IntervalInitialMomentum> {
    let (energy, momentum, weight, continuous_dimensions, piecewise_regions) = match source {
        InitialMomentum::P4(p4) => (
            Interval::from(p4.e()),
            [
                Interval::from(p4.px()),
                Interval::from(p4.py()),
                Interval::from(p4.pz()),
            ],
            Interval::ONE,
            0,
            1,
        ),
        InitialMomentum::Momentum(momentum) => {
            let p2 = momentum.px() * momentum.px()
                + momentum.py() * momentum.py()
                + momentum.pz() * momentum.pz();
            (
                Interval::from((p2 + mass * mass).sqrt()),
                [
                    Interval::from(momentum.px()),
                    Interval::from(momentum.py()),
                    Interval::from(momentum.pz()),
                ],
                Interval::ONE,
                0,
                1,
            )
        }
        InitialMomentum::EnergyDirection { energy, direction } => {
            let sample = interval_scalar_sample(energy)?;
            let direction = direction.unit()?;
            let magnitude = (sample.value.sqr() - mass * mass).sqrt();
            (
                sample.value,
                [
                    magnitude * direction.px(),
                    magnitude * direction.py(),
                    magnitude * direction.pz(),
                ],
                sample.weight,
                sample.continuous_dimensions,
                sample.piecewise_regions,
            )
        }
    };
    Ok(IntervalInitialMomentum {
        energy,
        momentum,
        mass,
        weight,
        continuous_dimensions,
        piecewise_regions,
    })
}

fn interval_invariant_mass(initials: &[IntervalInitialMomentum]) -> Interval {
    let mut invariant_squared = Interval::ZERO;
    for initial in initials {
        invariant_squared += initial.mass * initial.mass;
    }
    for first in 0..initials.len() {
        for second in (first + 1)..initials.len() {
            let lhs = initials[first];
            let rhs = initials[second];
            let spatial_dot = lhs.momentum[0] * rhs.momentum[0]
                + lhs.momentum[1] * rhs.momentum[1]
                + lhs.momentum[2] * rhs.momentum[2];
            invariant_squared += 2.0 * (lhs.energy * rhs.energy - spatial_dot);
        }
    }
    invariant_squared.sqrt()
}

fn validate_common(events: usize) -> GenerationResult<()> {
    if events == 0 {
        return Err(GenerationError::InvalidConfiguration(
            "events must be nonzero".into(),
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
    if ![p4.e, p4.px, p4.py, p4.pz].into_iter().all(f64::is_finite) || p4.e <= 0.0 {
        return Err(GenerationError::Kinematics {
            index,
            source: LadduPhysicsError::invalid_value(
                format!("four-momentum for edge `{name}`"),
                "finite components and positive energy",
                p4,
            ),
        });
    }
    let tolerance = 1e-9 * (1.0 + mass * mass + p4.e * p4.e);
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
    let scale = incoming.e.abs().max(1.0);
    if residual.e.abs() > 1e-9 * scale
        || residual.px.abs() > 1e-9 * scale
        || residual.py.abs() > 1e-9 * scale
        || residual.pz.abs() > 1e-9 * scale
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
fn report(requested: usize, seed: u64, decision: &MemoryDecision) -> GenerationReport {
    GenerationReport {
        requested,
        seed,
        chunk_events: decision.chunk_events,
        estimated_peak_bytes: decision.estimated_peak_bytes,
        actual_high_water_bytes: Some(decision.estimated_peak_bytes),
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
    use laddu_physics::{histogram::Histogram, quantum::ParticleProperties, vectors::RealVec3};

    struct FailingSink {
        aborted: bool,
    }

    impl EventSink for FailingSink {
        fn begin(
            &mut self,
            _schema: Arc<Schema>,
            _plan: WritePlan,
        ) -> laddu_data::LadduDataResult<()> {
            Ok(())
        }

        fn write_batch(&mut self, _batch: &EventBatch) -> laddu_data::LadduDataResult<()> {
            Err(laddu_data::LadduDataError::Sink(
                "injected write failure".into(),
            ))
        }

        fn finish(&mut self) -> laddu_data::LadduDataResult<()> {
            Ok(())
        }

        fn abort(&mut self) -> laddu_data::LadduDataResult<()> {
            self.aborted = true;
            Ok(())
        }
    }

    fn decay_generator() -> ChannelGenerator {
        let mut channel = Channel::new("decay");
        channel
            .edge("parent")
            .properties(&ParticleProperties::unknown().with_mass(2.0))
            .initial_p4(RealVec4::new(2.0, 0.0, 0.0, 0.0));
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

    fn closure_generator() -> ChannelGenerator {
        let photon = ParticleProperties::unknown().with_mass(0.0);
        let proton = ParticleProperties::unknown().with_mass(0.938_272_088_16);
        let kaon = ParticleProperties::unknown().with_mass(0.497_611);
        let mut channel = Channel::new("closure envelope");
        channel
            .edge("gamma")
            .properties(&photon)
            .initial_energy_source_direction(
                ScalarSource::uniform(8.0, 9.0),
                RealVec3::new(0.0, 0.0, 1.0),
            );
        channel
            .edge("target")
            .properties(&proton)
            .initial_momentum(RealVec3::new(0.0, 0.0, 0.0));
        channel
            .edge("x")
            .mass_proposal(MassProposal::uniform(2.0 * 0.497_611, 2.0))
            .generated_only();
        channel.edge("recoil").properties(&proton);
        channel.edge("ks1").properties(&kaon);
        channel.edge("ks2").properties(&kaon);
        channel
            .vertex("production")
            .incoming(["gamma", "target"])
            .outgoing(["x", "recoil"])
            .generation(VertexProposal::t_exchange(
                ("gamma", "x"),
                TDistribution::mixture([
                    (0.2, TComponent::Uniform),
                    (0.8, TComponent::Exponential { slope: 4.0 }),
                ]),
            ));
        channel
            .vertex("decay")
            .incoming(["x"])
            .outgoing(["ks1", "ks2"]);
        ChannelGenerator::new(channel).unwrap()
    }

    #[test]
    fn interval_scalar_and_initial_bounds_cover_builtin_samples() {
        let histogram = Histogram::new(vec![1.0, 0.0, 3.0], vec![8.0, 8.25, 8.75, 9.0]).unwrap();
        let sources = [
            ScalarSource::constant(8.5),
            ScalarSource::uniform(8.0, 9.0),
            ScalarSource::histogram(histogram),
        ];
        for (index, source) in sources.into_iter().enumerate() {
            let scalar_bound = interval_scalar_sample(&source).unwrap();
            let initial_source = InitialMomentum::energy_source_direction(
                source.clone(),
                RealVec3::new(0.0, 0.0, 1.0),
            );
            let initial_bound = interval_initial_momentum(&initial_source, 0.5).unwrap();
            let mut rng = ProposalRng::new(500 + index as u64);
            for _ in 0..2_000 {
                let scalar = source.sample(&mut rng).unwrap();
                assert!(scalar_bound.value.contains(scalar.value));
                assert!(scalar_bound.weight.contains(scalar.weight));
                let initial = initial_source.sample_prevalidated(0.5, &mut rng).unwrap();
                assert!(initial_bound.energy.contains(initial.p4.e()));
                assert!(initial_bound.momentum[2].contains(initial.p4.pz()));
                assert!(initial_bound.weight.contains(initial.weight));
            }
        }
    }

    #[test]
    fn proven_decay_envelope_contains_proposals_and_unweights_without_a_model() {
        let generator = decay_generator();
        let proven = generator.phase_space_envelope().unwrap();
        let proposals = generator.propose_range(0, 2_000, 41, 0).unwrap();
        assert!(
            proposals
                .iter()
                .all(|event| proven.weight_interval.contains(event.proposal_weight))
        );

        let mut config = UnweightedConfig::new(64).with_max_proposals(10_000);
        config.seed = 41;
        config.envelope = EnvelopeMode::ProvenPhaseSpace;
        let (_, report) = generator.generate_unweighted_dataset(config, None).unwrap();
        assert_eq!(report.envelope_kind, Some(EnvelopeKind::ProvenPhaseSpace));
        assert_eq!(report.proven_weight_interval, Some(proven.weight_bounds()));
        assert!(report.maximum_weight <= proven.maximum_weight);
    }

    #[test]
    fn proven_unweighting_is_thread_and_memory_budget_independent() {
        let generator = decay_generator();
        let mut first = UnweightedConfig::new(32).with_max_proposals(1_000);
        first.seed = 57;
        first.memory = MemoryBudget::Bytes(4_096);
        first.envelope = EnvelopeMode::ProvenPhaseSpace;
        let mut second = first;
        second.memory = MemoryBudget::Bytes(16_384);
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (a, _) = one_thread
            .install(|| generator.generate_unweighted_dataset(first, None))
            .unwrap();
        let (b, _) = four_threads
            .install(|| generator.generate_unweighted_dataset(second, None))
            .unwrap();
        let mut av = Vec::new();
        a.for_each_event(|event| av.push(event.p4(0))).unwrap();
        let mut bv = Vec::new();
        b.for_each_event(|event| bv.push(event.p4(0))).unwrap();
        assert_eq!(av, bv);
    }

    #[test]
    fn closure_phase_space_bound_contains_sample_and_reports_efficiency() {
        let generator = closure_generator();
        let proven = generator.phase_space_envelope().unwrap();
        let mut weights = generator
            .propose_range(0, 20_000, 73, 0)
            .unwrap()
            .into_iter()
            .map(|event| event.proposal_weight)
            .collect::<Vec<_>>();
        assert!(
            weights
                .iter()
                .all(|weight| proven.weight_interval.contains(*weight))
        );
        weights.sort_by(f64::total_cmp);
        let quantile = |fraction: f64| {
            weights[((weights.len() - 1) as f64 * fraction).round() as usize]
                / proven.maximum_weight
        };
        let mean = weights.iter().sum::<f64>() / weights.len() as f64;
        eprintln!(
            "closure proven-envelope ratios: max={:.6e}, mean={:.6e}, p50={:.6e}, p90={:.6e}, p99={:.6e}",
            weights[weights.len() - 1] / proven.maximum_weight,
            mean / proven.maximum_weight,
            quantile(0.50),
            quantile(0.90),
            quantile(0.99),
        );
        assert!(proven.maximum_weight.is_finite());
        assert!(
            proven.subdivisions > 0,
            "a non-singleton root invariant-mass domain should be subdivided"
        );
        eprintln!(
            "closure branch-and-bound envelope: subdivisions={}, proposal_dimensions={}, regions={}, maximum={:.6e}",
            proven.subdivisions,
            proven.continuous_dimensions,
            proven.piecewise_regions,
            proven.maximum_weight,
        );
        assert!(
            weights[weights.len() - 1] / proven.maximum_weight > 0.5,
            "latent mass and transfer refinement should materially tighten the closure envelope"
        );
    }

    #[test]
    fn weighted_generation_is_memory_budget_independent() {
        let generator = decay_generator();
        let mut first = WeightedConfig::new(32);
        first.seed = 9;
        first.memory = MemoryBudget::Bytes(4_096);
        let mut second = first;
        second.memory = MemoryBudget::Bytes(16_384);
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
    fn unweighted_generation_is_thread_and_memory_budget_independent() {
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
        first.memory = MemoryBudget::Bytes(4_096);
        first.envelope = EnvelopeMode::Strict { max_weight: 1.0 };
        let mut second = first;
        second.memory = MemoryBudget::Bytes(16_384);
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (a, _) = one_thread
            .install(|| generator.generate_unweighted_dataset(first, Some(&evaluator)))
            .unwrap();
        let (b, _) = four_threads
            .install(|| generator.generate_unweighted_dataset(second, Some(&evaluator)))
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
            density: PiecewiseDensity::uniform(1.0, 3.0, Arc::from([1.0, 3.0, 7.0, 1.0])).unwrap(),
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
                Some(&evaluator),
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
            .generate_unweighted_dataset(config, Some(&evaluator))
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
                Some(&evaluator),
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
            .generate_unweighted_dataset(config, Some(&evaluator))
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
            .initial_p4(RealVec4::new(1.0, 0.0, 0.0, 0.0))
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

    #[test]
    fn weighted_generation_aborts_sink_after_post_begin_failure() {
        let generator = decay_generator();
        let mut sink = FailingSink { aborted: false };

        let result = generator.generate_weighted_to(WeightedConfig::new(2), None, &mut sink);

        assert!(matches!(
            result,
            Err(GenerationError::Data(laddu_data::LadduDataError::Sink(message)))
                if message.contains("injected write failure")
        ));
        assert!(sink.aborted);
    }
}
