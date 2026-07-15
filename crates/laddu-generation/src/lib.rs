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
use laddu_physics::{
    LadduPhysicsError,
    channel::{Channel, Vertex},
    vectors::RealVec4,
};
use laddu_runtime::{Execution, PreparedModel};
use rayon::prelude::*;
use thiserror::Error;

pub use laddu_physics::generation::{
    FixedMass, InitialMomentum, InitialMomentumResult, MassProposal, MassProposalResult, NamedMass,
    NamedMomentum, ProposalResult, ProposalRng, ScalarProposalResult, ScalarSource, TComponent,
    TDistribution, TwoBodyDecay, TwoBodyScattering, UniformMass, VertexProposal,
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

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub struct UnweightedConfig {
    pub events: usize,
    pub max_proposals: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub diagnostics: bool,
    pub envelope_overflow: EnvelopeOverflow,
}

impl UnweightedConfig {
    pub fn new(events: usize, max_proposals: usize) -> Self {
        Self {
            events,
            max_proposals,
            batch_size: 1024,
            seed: 0,
            diagnostics: false,
            envelope_overflow: EnvelopeOverflow::Error,
        }
    }
}

/// Policy used when an unweighting proposal exceeds the current envelope.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

#[derive(Clone, Copy, Debug)]
pub enum EnvelopeMode {
    Strict {
        max_weight: f64,
    },
    Pilot {
        proposals: usize,
        safety_factor: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeKind {
    Strict,
    Pilot,
}

#[derive(Clone, Debug, Default)]
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

    fn evaluate(&self, batch: &EventBatch) -> GenerationResult<Vec<f64>> {
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
    channel: Channel,
    plan: Vec<usize>,
    edge_names: Vec<String>,
    output_names: Vec<String>,
    root_edges: HashSet<String>,
    scalar_sources: Vec<(String, ScalarSource)>,
}

#[derive(Clone, Debug)]
struct GeneratedEvent {
    p4s: HashMap<String, RealVec4>,
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
        Ok(Self {
            channel,
            plan,
            edge_names,
            output_names,
            root_edges,
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
        for start in (0..config.events).step_by(config.batch_size) {
            let count = config.batch_size.min(config.events - start);
            let mut events = self.propose_range(start as u64, count, config.seed, 0)?;
            self.apply_model(&mut events, model)?;
            update_report(&mut report, &events);
            report.proposals += events.len();
            report.produced += events.len();
            let batch =
                self.output_batch(&events, Arc::clone(&schema), true, config.diagnostics)?;
            sink.write_batch(&batch)?;
        }
        sink.finish()?;
        Ok(report)
    }

    pub fn generate_unweighted_to(
        &self,
        config: UnweightedConfig,
        model: &ModelEvaluator,
        envelope: EnvelopeMode,
        sink: &mut dyn EventSink,
    ) -> GenerationResult<GenerationReport> {
        validate_common(config.events, config.batch_size)?;
        if config.max_proposals < config.events {
            return Err(GenerationError::InvalidConfiguration(
                "max_proposals must be at least the requested event count".into(),
            ));
        }
        let (mut bound, kind, pilot_count) = match envelope {
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
                let mut pilot = self.propose_range(0, proposals, config.seed, 1)?;
                self.apply_model(&mut pilot, Some(model))?;
                let observed = pilot
                    .iter()
                    .map(|event| event.target_weight)
                    .fold(0.0, f64::max);
                (observed * safety_factor, EnvelopeKind::Pilot, proposals)
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
        while report.produced < config.events && proposal_index < config.max_proposals {
            let count = config.batch_size.min(config.max_proposals - proposal_index);
            let mut events = self.propose_range(proposal_index as u64, count, config.seed, 0)?;
            self.apply_model(&mut events, Some(model))?;
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
            let accepted = events
                .into_iter()
                .filter(|event| {
                    acceptance_uniform(config.seed, event.index) * bound <= event.target_weight
                })
                .take(remaining)
                .collect::<Vec<_>>();
            report.proposals += count;
            report.produced += accepted.len();
            report.rejected = report.proposals - report.produced;
            proposal_index += count;
            match config.envelope_overflow {
                EnvelopeOverflow::Error if !accepted.is_empty() => {
                    let batch = self.output_batch(
                        &accepted,
                        Arc::clone(&schema),
                        false,
                        config.diagnostics,
                    )?;
                    sink.write_batch(&batch)?;
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
        envelope: EnvelopeMode,
    ) -> GenerationResult<(Dataset, GenerationReport)> {
        let mut sink = MemorySink::new();
        let report = self.generate_unweighted_to(config, model, envelope, &mut sink)?;
        Ok((Dataset::from_batches(sink.into_batches())?, report))
    }

    fn propose_range(
        &self,
        start: u64,
        count: usize,
        seed: u64,
        stream: u64,
    ) -> GenerationResult<Vec<GeneratedEvent>> {
        (0..count)
            .into_par_iter()
            .map(|offset| self.propose(start + offset as u64, seed, stream))
            .collect()
    }

    fn propose(&self, index: u64, seed: u64, stream: u64) -> GenerationResult<GeneratedEvent> {
        let mut rng = ProposalRng::new(derive_seed(seed, stream, index, 0));
        let mut p4s = HashMap::with_capacity(self.root_edges.len());
        let mut proposal_weight = 1.0;
        for edge in self.channel.initial_edges() {
            let source = edge
                .initial_momentum()
                .ok_or_else(|| GenerationError::InitialState {
                    index,
                    source: LadduPhysicsError::invalid_relation(format!(
                        "initial edge `{}` has no momentum source",
                        edge.name()
                    )),
                })?;
            let sampled = source
                .sample(edge.name(), edge.properties(), &mut rng)
                .map_err(|source| GenerationError::InitialState { index, source })?;
            p4s.insert(edge.name().to_owned(), sampled.p4);
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
        let total_initial: RealVec4 = p4s.values().copied().sum();
        let maximum_mass = total_initial
            .m()
            .map_err(|source| GenerationError::Kinematics { index, source })?;
        let mut masses = HashMap::new();
        for (edge_index, edge) in self.channel.edges().enumerate() {
            let mass = if let Some(proposal) = edge.mass_proposal() {
                let mut mass_rng =
                    ProposalRng::new(derive_seed(seed, stream, index, 1 + edge_index as u64));
                let result =
                    proposal
                        .propose(0.0, maximum_mass, &mut mass_rng)
                        .map_err(|source| GenerationError::MassProposal {
                            index,
                            edge: edge.name().to_owned(),
                            source,
                        })?;
                proposal_weight *= result.weight;
                result.mass
            } else {
                edge.properties()
                    .ok_or_else(|| GenerationError::MassProposal {
                        index,
                        edge: edge.name().to_owned(),
                        source: LadduPhysicsError::invalid_relation(format!(
                            "edge `{}` has neither particle properties nor a mass proposal",
                            edge.name()
                        )),
                    })?
                    .mass()
                    .map_err(|source| GenerationError::MassProposal {
                        index,
                        edge: edge.name().to_owned(),
                        source,
                    })?
            };
            if !mass.is_finite() || mass < 0.0 {
                return Err(GenerationError::MassProposal {
                    index,
                    edge: edge.name().to_owned(),
                    source: LadduPhysicsError::invalid_value(
                        "mass",
                        "finite and nonnegative",
                        mass,
                    ),
                });
            }
            masses.insert(edge.name().to_owned(), mass);
        }
        for edge in &self.root_edges {
            validate_p4(edge, p4s[edge], masses[edge], index)?;
        }
        let vertices = self.channel.vertices().collect::<Vec<_>>();
        for (step, vertex_index) in self.plan.iter().copied().enumerate() {
            let vertex = vertices[vertex_index];
            let incoming = vertex
                .incoming()
                .iter()
                .map(|name| {
                    p4s.get(name)
                        .copied()
                        .map(|p4| NamedMomentum { name, p4 })
                        .ok_or_else(|| GenerationError::VertexProposal {
                            index,
                            vertex: vertex.name().to_owned(),
                            source: LadduPhysicsError::invalid_relation(format!(
                                "missing generated momentum for `{name}`"
                            )),
                        })
                })
                .collect::<GenerationResult<Vec<_>>>()?;
            let outgoing = vertex
                .outgoing()
                .iter()
                .map(|name| NamedMass {
                    name,
                    mass: masses[name],
                })
                .collect::<Vec<_>>();
            let mut vertex_rng =
                ProposalRng::new(derive_seed(seed, stream, index, 10_000 + step as u64));
            let result = match vertex.generation() {
                Some(proposal) => proposal.propose(&incoming, &outgoing, &mut vertex_rng),
                None if vertex.incoming().len() == 1 && vertex.outgoing().len() == 2 => {
                    TwoBodyDecay::isotropic().propose(&incoming, &outgoing, &mut vertex_rng)
                }
                None => Err(LadduPhysicsError::invalid_relation(format!(
                    "vertex `{}` has no generation proposal",
                    vertex.name()
                ))),
            }
            .map_err(|source| GenerationError::VertexProposal {
                index,
                vertex: vertex.name().to_owned(),
                source,
            })?;
            if result.outgoing.len() != vertex.outgoing().len()
                || !result.weight.is_finite()
                || result.weight <= 0.0
            {
                return Err(GenerationError::VertexProposal {
                    index,
                    vertex: vertex.name().to_owned(),
                    source: LadduPhysicsError::invalid_relation(format!(
                        "proposal returned {} outgoing momenta and weight {}",
                        result.outgoing.len(),
                        result.weight
                    )),
                });
            }
            proposal_weight *= result.weight;
            for (name, p4) in vertex.outgoing().iter().cloned().zip(result.outgoing) {
                validate_p4(name.as_str(), p4, masses[&name], index)?;
                p4s.insert(name, p4);
            }
            validate_conservation(vertex, &p4s, index)?;
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
            let columns = self
                .edge_names
                .iter()
                .map(|name| {
                    Arc::<[RealVec4]>::from(
                        events
                            .iter()
                            .map(|event| event.p4s[name])
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
            let batch = EventBatch::new(schema, columns, scalar_columns, None)?;
            let weights = model.evaluate(&batch)?;
            for (event, weight) in events.iter_mut().zip(weights) {
                event.model_weight = weight;
                event.target_weight = event.proposal_weight * weight;
            }
        }
        Ok(())
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
            .output_names
            .iter()
            .map(|name| {
                Arc::<[RealVec4]>::from(
                    events
                        .iter()
                        .map(|event| event.p4s[name])
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
fn validate_conservation(
    vertex: &Vertex,
    p4s: &HashMap<String, RealVec4>,
    index: u64,
) -> GenerationResult<()> {
    let incoming: RealVec4 = vertex.incoming().iter().map(|name| p4s[name]).sum();
    let outgoing: RealVec4 = vertex.outgoing().iter().map(|name| p4s[name]).sum();
    let scale = 1.0 + incoming.t.abs();
    if [
        incoming.x - outgoing.x,
        incoming.y - outgoing.y,
        incoming.z - outgoing.z,
        incoming.t - outgoing.t,
    ]
    .into_iter()
    .any(|delta| delta.abs() > 1e-9 * scale)
    {
        return Err(GenerationError::VertexProposal {
            index,
            vertex: vertex.name().to_owned(),
            source: LadduPhysicsError::invalid_relation("four-momentum is not conserved"),
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
                UnweightedConfig::new(2, 10),
                &evaluator,
                EnvelopeMode::Strict { max_weight: 1e-12 },
            ),
            Err(GenerationError::EnvelopeOverflow { .. })
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
        let mut config = UnweightedConfig::new(16, 10_000);
        config.envelope_overflow = EnvelopeOverflow::Grow { safety_factor: 1.5 };
        let (dataset, report) = generator
            .generate_unweighted_dataset(
                config,
                &evaluator,
                EnvelopeMode::Strict { max_weight: 1e-12 },
            )
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
            .generation(TwoBodyDecay::isotropic());
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
            .generation(TwoBodyDecay::isotropic());
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
