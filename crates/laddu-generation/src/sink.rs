use std::{sync::Arc, time::Duration};

use laddu_core::{
    data::{Dataset, DatasetMetadata, EventData},
    vectors::Vec4,
    LadduError, LadduResult, ParticleProperties,
};

/// Which generated particles should be passed to a sink.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum GenerationOutput {
    /// Pass every generated particle.
    #[default]
    All,
    /// Pass only generated particles that do not decay further.
    FinalState,
    /// Pass only the listed particle labels.
    Only(Vec<String>),
    /// Pass every generated particle except the listed particle labels.
    Exclude(Vec<String>),
}

impl GenerationOutput {
    /// Select every generated particle.
    pub fn all() -> Self {
        Self::All
    }

    /// Select only generated particles that do not decay further.
    pub fn final_state() -> Self {
        Self::FinalState
    }

    /// Select only the listed generated particle labels.
    pub fn only(labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::Only(labels.into_iter().map(Into::into).collect())
    }

    /// Select every generated particle except the listed particle labels.
    pub fn exclude(labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::Exclude(labels.into_iter().map(Into::into).collect())
    }

    pub(crate) fn includes(&self, particle: &GeneratedParticleInfo) -> bool {
        match self {
            Self::All => true,
            Self::FinalState => matches!(particle.role, GeneratedParticleRole::Final),
            Self::Only(labels) => labels.iter().any(|label| label == &particle.label),
            Self::Exclude(labels) => !labels.iter().any(|label| label == &particle.label),
        }
    }
}

/// Metadata describing the generated particles available to sinks.
#[derive(Clone, Debug)]
pub struct GeneratedLayout {
    particles: Vec<GeneratedParticleInfo>,
}

impl GeneratedLayout {
    /// Construct a generated layout from particle metadata.
    pub fn new(particles: Vec<GeneratedParticleInfo>) -> Self {
        Self { particles }
    }

    /// Return the generated particle metadata in sink output order.
    pub fn particles(&self) -> &[GeneratedParticleInfo] {
        &self.particles
    }

    /// Return the generated particle labels in sink output order.
    pub fn labels(&self) -> Vec<String> {
        self.particles
            .iter()
            .map(|particle| particle.label.clone())
            .collect()
    }

    /// Return a copy of this layout with an output selection applied.
    pub fn select(&self, output: &GenerationOutput) -> Self {
        let particles = self
            .particles
            .iter()
            .filter(|particle| output.includes(particle))
            .enumerate()
            .map(|(output_index, particle)| {
                let mut particle = particle.clone();
                particle.output_index = Some(output_index);
                particle
            })
            .collect();
        Self { particles }
    }
}

/// Metadata for one generated particle.
#[derive(Clone, Debug)]
pub struct GeneratedParticleInfo {
    /// Channel particle label.
    pub label: String,
    /// Role in the generated topology.
    pub role: GeneratedParticleRole,
    /// Particle properties copied from the validated generation plan.
    pub properties: ParticleProperties,
    /// Output p4 index for selected layouts.
    pub output_index: Option<usize>,
}

impl GeneratedParticleInfo {
    /// Construct generated particle metadata.
    pub fn new(
        label: impl Into<String>,
        role: GeneratedParticleRole,
        properties: ParticleProperties,
    ) -> Self {
        Self {
            label: label.into(),
            role,
            properties,
            output_index: None,
        }
    }
}

/// Role of a generated particle in the topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratedParticleRole {
    /// Initial-state particle.
    Initial,
    /// Generated particle that decays further.
    Intermediate,
    /// Generated particle with no downstream generated decay.
    Final,
}

/// One generated record passed to sinks.
#[derive(Clone, Debug)]
pub struct GeneratedRecord {
    /// Rank-local generated record index.
    pub local_index: u64,
    /// Global generated record index, if known.
    pub global_index: Option<u64>,
    /// Event weight.
    pub weight: f64,
    /// Generated p4 values in layout order.
    pub p4s: Vec<Vec4>,
}

/// Borrowed batch passed to sinks.
#[derive(Clone, Copy, Debug)]
pub struct GeneratedBatchView<'a> {
    /// Layout describing the p4 order in each record.
    pub layout: &'a GeneratedLayout,
    /// Generated records in this batch.
    pub records: &'a [GeneratedRecord],
}

/// MPI behavior supported by a sink.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkMpiSupport {
    /// Each rank writes its own independent output.
    RankLocal,
    /// The sink expects all records on the root rank.
    RootOnly,
    /// The sink performs collective output.
    Collective,
}

/// Sink for generated event records.
pub trait GeneratedSink {
    /// Output returned after generation finishes.
    type Output;

    /// Called once before records are pushed.
    fn begin(&mut self, _layout: &GeneratedLayout) -> LadduResult<()> {
        Ok(())
    }

    /// Push one batch of generated records.
    fn push_batch(&mut self, batch: GeneratedBatchView<'_>) -> LadduResult<()>;

    /// Finish output and return the sink-specific result.
    fn finish(self) -> LadduResult<Self::Output>;

    /// Return this sink's MPI output support.
    fn mpi_support(&self) -> SinkMpiSupport {
        SinkMpiSupport::RankLocal
    }
}

/// Sink that materializes generated records as a [`Dataset`].
#[derive(Clone, Debug, Default)]
pub struct DatasetSink {
    output: GenerationOutput,
    metadata: Option<Arc<DatasetMetadata>>,
    p4_indices: Vec<usize>,
    events: Vec<Arc<EventData>>,
}

impl DatasetSink {
    /// Construct a dataset sink that writes all generated p4 values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set which generated particles should appear in the output dataset.
    pub fn output(mut self, output: GenerationOutput) -> Self {
        self.output = output;
        self
    }
}

impl GeneratedSink for DatasetSink {
    type Output = Dataset;

    fn begin(&mut self, layout: &GeneratedLayout) -> LadduResult<()> {
        self.p4_indices = layout
            .particles()
            .iter()
            .enumerate()
            .filter_map(|(index, particle)| self.output.includes(particle).then_some(index))
            .collect();
        let labels = self
            .p4_indices
            .iter()
            .map(|index| layout.particles()[*index].label.clone())
            .collect();
        self.metadata = Some(Arc::new(DatasetMetadata::new(
            labels,
            Vec::<String>::new(),
        )?));
        Ok(())
    }

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>) -> LadduResult<()> {
        if self.metadata.is_none() {
            self.begin(batch.layout)?;
        }
        for record in batch.records {
            let p4s = self
                .p4_indices
                .iter()
                .map(|index| record.p4s[*index])
                .collect();
            self.events.push(Arc::new(EventData {
                p4s,
                aux: Vec::new(),
                weight: record.weight,
            }));
        }
        Ok(())
    }

    fn finish(self) -> LadduResult<Self::Output> {
        let metadata = self.metadata.ok_or_else(|| {
            LadduError::Custom("dataset sink was not initialized before finish".to_string())
        })?;
        Ok(Dataset::new_with_metadata(self.events, metadata))
    }
}

/// Sink that discards records and returns the number of records it received.
#[derive(Clone, Debug, Default)]
pub struct NullSink {
    count: usize,
}

impl NullSink {
    /// Construct a null sink.
    pub fn new() -> Self {
        Self::default()
    }
}

impl GeneratedSink for NullSink {
    type Output = usize;

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>) -> LadduResult<()> {
        self.count += batch.records.len();
        Ok(())
    }

    fn finish(self) -> LadduResult<Self::Output> {
        Ok(self.count)
    }
}

/// Generation mode.
#[derive(Clone, Debug, Default)]
pub enum GenerationMode {
    /// Generate raw phase-space proposal events with unit weights.
    #[default]
    Raw,
    /// Generate proposal events and assign weights from the real part of an expression.
    Weighted {
        /// Expression evaluated on generated events.
        expression: Box<laddu_core::Expression>,
        /// Free-parameter values passed to the expression evaluator.
        parameters: Vec<f64>,
    },
}

/// Options controlling generation execution.
#[derive(Clone, Debug)]
pub struct GenerationOptions {
    /// Number of records to accumulate before pushing to the sink.
    pub batch_size: usize,
    /// Optional upper bound on proposal trials.
    pub max_trials: Option<u64>,
    /// Optional run seed overriding the generator's seed.
    pub seed: Option<u64>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            batch_size: 10_000,
            max_trials: None,
            seed: None,
        }
    }
}

impl GenerationOptions {
    /// Set the batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set a maximum number of proposal trials.
    pub fn max_trials(mut self, max_trials: impl Into<Option<u64>>) -> Self {
        self.max_trials = max_trials.into();
        self
    }

    /// Set a run seed overriding the generator seed.
    pub fn seed(mut self, seed: impl Into<Option<u64>>) -> Self {
        self.seed = seed.into();
        self
    }
}

/// Coarse generation mode used in statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationModeKind {
    /// Raw phase-space proposal generation.
    Raw,
    /// Weighted proposal generation.
    Weighted,
}

/// Statistics for a generation run.
#[derive(Clone, Debug)]
pub struct GenerationStats {
    /// Generation mode.
    pub mode: GenerationModeKind,
    /// Requested number of output events.
    pub target_events: u64,
    /// Number of records written to the sink.
    pub written_events: u64,
    /// Number of proposal events generated.
    pub proposed_events: u64,
    /// Number of accepted records.
    pub accepted_events: u64,
    /// Number of rejected records.
    pub rejected_events: u64,
    /// Acceptance rate for modes with rejection.
    pub acceptance_rate: Option<f64>,
    /// Envelope used by rejection modes, when applicable.
    pub envelope: Option<f64>,
    /// Number of proposal weights that exceeded the active rejection envelope.
    pub envelope_violations: u64,
    /// Sum of output weights.
    pub sum_weights: f64,
    /// Minimum output weight.
    pub min_weight: Option<f64>,
    /// Maximum output weight.
    pub max_weight: Option<f64>,
    /// Number of batches pushed to the sink.
    pub batches_written: u64,
    /// Wall-clock time spent in the generation call.
    pub elapsed: Duration,
    /// Seed used by this generation run, if deterministic.
    pub seed: Option<u64>,
}

impl GenerationStats {
    /// Return a concise human-readable audit report.
    pub fn audit(&self) -> String {
        let acceptance_rate = self
            .acceptance_rate
            .map_or_else(|| "n/a".to_string(), |rate| format!("{rate:.6}"));
        format!(
            "Generation audit\nmode: {:?}\ntarget events: {}\nwritten events: {}\nproposed events: {}\naccepted events: {}\nrejected events: {}\nacceptance rate: {}\nenvelope: {}\nenvelope violations: {}\nsum weights: {:.6}\nmin weight: {}\nmax weight: {}\nbatches written: {}\nelapsed: {:.3} s",
            self.mode,
            self.target_events,
            self.written_events,
            self.proposed_events,
            self.accepted_events,
            self.rejected_events,
            acceptance_rate,
            self.envelope
                .map_or_else(|| "n/a".to_string(), |value| format!("{value:.6}")),
            self.envelope_violations,
            self.sum_weights,
            self.min_weight
                .map_or_else(|| "n/a".to_string(), |value| format!("{value:.6}")),
            self.max_weight
                .map_or_else(|| "n/a".to_string(), |value| format!("{value:.6}")),
            self.batches_written,
            self.elapsed.as_secs_f64(),
        )
    }
}

/// Result of a generation call.
#[derive(Clone, Debug)]
pub struct GenerationResult<T> {
    /// Sink-specific output.
    pub output: T,
    /// Statistics collected during generation.
    pub stats: GenerationStats,
}
