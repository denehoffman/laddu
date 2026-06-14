use std::{sync::Arc, time::Duration};

use fastrand::Rng;
use laddu_core::{
    data::{
        io::ParquetBatchWriter, write_root, Dataset, DatasetMetadata, DatasetWriteOptions,
        EventData,
    },
    vectors::Vec4,
    LadduError, LadduResult, ParticleProperties, ScalarDistribution,
};
#[cfg(feature = "mpi")]
use mpi::{collective::SystemOperation, topology::SimpleCommunicator, traits::*};

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

    /// Check whether the output includes the given particle.
    pub fn includes(&self, particle: &GeneratedParticleInfo) -> bool {
        match self {
            Self::All => true,
            Self::FinalState => matches!(particle.role, GeneratedParticleRole::Final),
            Self::Only(labels) => labels.iter().any(|label| label == &particle.label),
            Self::Exclude(labels) => !labels.iter().any(|label| label == &particle.label),
        }
    }

    /// Get the indices corresponding to the particles selected by the [`GenerationOutput`].
    pub fn selected_p4_indices(&self, layout: &GeneratedLayout) -> Vec<usize> {
        layout
            .particles()
            .iter()
            .enumerate()
            .filter_map(|(index, particle)| self.includes(particle).then_some(index))
            .collect()
    }
}

/// Metadata describing the generated particles available to sinks.
#[derive(Clone, Debug)]
pub struct GeneratedLayout {
    particles: Vec<GeneratedParticleInfo>,
    aux: Vec<GeneratedAuxInfo>,
}

impl GeneratedLayout {
    /// Construct a generated layout from particle metadata.
    pub fn new(particles: Vec<GeneratedParticleInfo>, aux: Vec<GeneratedAuxInfo>) -> Self {
        Self { particles, aux }
    }

    /// Return the generated particle metadata in sink output order.
    pub fn particles(&self) -> &[GeneratedParticleInfo] {
        &self.particles
    }

    /// Return the generated auxiliary column metadata in sink output order.
    pub fn aux_info(&self) -> &[GeneratedAuxInfo] {
        &self.aux
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
        Self {
            particles,
            aux: self.aux.clone(),
        }
    }

    pub fn particle(&self, label: &str) -> Option<&GeneratedParticleInfo> {
        self.particles.iter().find(|p| p.label == label)
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

#[derive(Clone, Debug)]
pub struct GeneratedAuxInfo {
    pub label: String,
    pub generator: ScalarDistribution,
}

/// One generated record passed to sinks.
#[derive(Clone, Debug)]
pub struct GeneratedRecord {
    /// Rank-local generated record index.
    pub local_index: u64,
    /// Global generated record index.
    pub global_index: u64,
    /// Event weight.
    pub weight: f64,
    /// Generated p4 values in layout order.
    pub p4s: Vec<Vec4>,
    /// Generated aux values in layout order.
    pub aux: Vec<f64>,
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
    fn push_batch(&mut self, batch: GeneratedBatchView<'_>, rng: &mut Rng) -> LadduResult<()>;

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
        let aux_labels = layout.aux.iter().map(|aux| aux.label.clone()).collect();
        self.metadata = Some(Arc::new(DatasetMetadata::new(labels, aux_labels)?));
        Ok(())
    }

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>, _rng: &mut Rng) -> LadduResult<()> {
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
                aux: record.aux.clone(),
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

fn metadata_for_indices(
    layout: &GeneratedLayout,
    p4_indices: &[usize],
) -> LadduResult<Arc<DatasetMetadata>> {
    let labels = p4_indices
        .iter()
        .map(|index| layout.particles()[*index].label.clone())
        .collect();
    let aux_labels = layout.aux.iter().map(|aux| aux.label.clone()).collect();
    Ok(Arc::new(DatasetMetadata::new(labels, aux_labels)?))
}

fn records_to_dataset(
    records: &[GeneratedRecord],
    p4_indices: &[usize],
    metadata: Arc<DatasetMetadata>,
) -> Dataset {
    let events = records
        .iter()
        .map(|record| {
            let p4s = p4_indices.iter().map(|index| record.p4s[*index]).collect();
            Arc::new(EventData {
                p4s,
                aux: record.aux.clone(),
                weight: record.weight,
            })
        })
        .collect();
    Dataset::new_local(events, metadata)
}

/// Sink that streams generated records directly to a Parquet file.
pub struct ParquetSink {
    writer: ParquetBatchWriter,
    output: GenerationOutput,
    metadata: Option<Arc<DatasetMetadata>>,
    p4_indices: Vec<usize>,
    count: usize,
}

impl ParquetSink {
    /// Construct a Parquet sink with default write options.
    pub fn new(file_path: &str) -> LadduResult<Self> {
        Self::with_options(file_path, DatasetWriteOptions::default())
    }

    /// Construct a Parquet sink with explicit write options.
    pub fn with_options(file_path: &str, options: DatasetWriteOptions) -> LadduResult<Self> {
        let file_path = mpi_ranked_path(file_path);
        Ok(Self {
            writer: ParquetBatchWriter::new(&file_path, options)?,
            output: GenerationOutput::default(),
            metadata: None,
            p4_indices: Vec::new(),
            count: 0,
        })
    }

    /// Set which generated particles should appear in the output file.
    pub fn output(mut self, output: GenerationOutput) -> Self {
        self.output = output;
        self
    }
}

impl GeneratedSink for ParquetSink {
    type Output = usize;

    fn begin(&mut self, layout: &GeneratedLayout) -> LadduResult<()> {
        self.p4_indices = self.output.selected_p4_indices(layout);
        self.metadata = Some(metadata_for_indices(layout, &self.p4_indices)?);
        Ok(())
    }

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>, _rng: &mut Rng) -> LadduResult<()> {
        if self.metadata.is_none() {
            self.begin(batch.layout)?;
        }
        let metadata = self.metadata.as_ref().expect("metadata set above").clone();
        let dataset = records_to_dataset(batch.records, &self.p4_indices, metadata);
        self.writer.write(&dataset)?;
        self.count += batch.records.len();
        Ok(())
    }

    fn finish(mut self) -> LadduResult<Self::Output> {
        self.writer.close()?;
        Ok(self.count)
    }
}

/// Sink that writes generated records to a ROOT file on finish.
pub struct RootSink {
    file_path: String,
    options: DatasetWriteOptions,
    output: GenerationOutput,
    metadata: Option<Arc<DatasetMetadata>>,
    p4_indices: Vec<usize>,
    records: Vec<GeneratedRecord>,
}

impl RootSink {
    /// Construct a ROOT sink with default write options.
    pub fn new(file_path: impl Into<String>) -> Self {
        Self::with_options(file_path, DatasetWriteOptions::default())
    }

    /// Construct a ROOT sink with explicit write options.
    pub fn with_options(file_path: impl Into<String>, options: DatasetWriteOptions) -> Self {
        Self {
            file_path: mpi_ranked_path(file_path.into()),
            options,
            output: GenerationOutput::default(),
            metadata: None,
            p4_indices: Vec::new(),
            records: Vec::new(),
        }
    }

    /// Set which generated particles should appear in the output file.
    pub fn output(mut self, output: GenerationOutput) -> Self {
        self.output = output;
        self
    }
}

fn mpi_ranked_path(file_path: impl AsRef<str>) -> String {
    let file_path = file_path.as_ref();
    #[cfg(feature = "mpi")]
    {
        if let Some(world) = laddu_core::mpi::get_world() {
            let path = std::path::Path::new(file_path);
            let rank = world.rank();
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or(file_path);
            let file_name = match path.extension().and_then(|extension| extension.to_str()) {
                Some(extension) => format!("{stem}.rank{rank}.{extension}"),
                None => format!("{stem}.rank{rank}"),
            };
            return match path.parent() {
                Some(parent) => parent.join(file_name).to_string_lossy().into_owned(),
                None => file_name,
            };
        }
    }
    file_path.to_string()
}

impl GeneratedSink for RootSink {
    type Output = usize;

    fn begin(&mut self, layout: &GeneratedLayout) -> LadduResult<()> {
        self.p4_indices = self.output.selected_p4_indices(layout);
        self.metadata = Some(metadata_for_indices(layout, &self.p4_indices)?);
        Ok(())
    }

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>, _rng: &mut Rng) -> LadduResult<()> {
        if self.metadata.is_none() {
            self.begin(batch.layout)?;
        }
        self.records.extend_from_slice(batch.records);
        Ok(())
    }

    fn finish(self) -> LadduResult<Self::Output> {
        let count = self.records.len();
        let metadata = self.metadata.ok_or_else(|| {
            LadduError::Custom("root sink was not initialized before finish".to_string())
        })?;
        let dataset = records_to_dataset(&self.records, &self.p4_indices, metadata);
        write_root(&dataset, &self.file_path, &self.options)?;
        Ok(count)
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

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>, _rng: &mut Rng) -> LadduResult<()> {
        self.count += batch.records.len();
        Ok(())
    }

    fn finish(self) -> LadduResult<Self::Output> {
        Ok(self.count)
    }
}

/// Sink that forwards each generated batch to a Rust callback.
pub struct CallbackSink<F> {
    callback: F,
    count: usize,
    mpi_support: SinkMpiSupport,
}

impl<F> CallbackSink<F>
where
    for<'a> F: FnMut(GeneratedBatchView<'a>) -> LadduResult<()>,
{
    /// Construct a callback sink.
    pub fn new(callback: F) -> Self {
        Self {
            callback,
            count: 0,
            mpi_support: SinkMpiSupport::RankLocal,
        }
    }

    /// Set this sink's MPI output support.
    pub fn mpi_support(mut self, support: SinkMpiSupport) -> Self {
        self.mpi_support = support;
        self
    }
}

impl<F> GeneratedSink for CallbackSink<F>
where
    for<'a> F: FnMut(GeneratedBatchView<'a>) -> LadduResult<()>,
{
    type Output = usize;

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>, _rng: &mut Rng) -> LadduResult<()> {
        self.count += batch.records.len();
        (self.callback)(batch)
    }

    fn finish(self) -> LadduResult<Self::Output> {
        Ok(self.count)
    }

    fn mpi_support(&self) -> SinkMpiSupport {
        self.mpi_support
    }
}

/// Rejection-envelope configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Envelope {
    /// Start rejection sampling from this maximum event weight.
    Initial(f64),
    /// Estimate the starting envelope from a pilot sample.
    Estimate {
        /// Number of pilot proposal events used to estimate the envelope.
        pilot_events: usize,
        /// Factor multiplied into the largest observed pilot weight.
        safety_factor: f64,
    },
    /// Start from an initial envelope and grow it when violations are observed.
    Adaptive {
        /// Initial rejection envelope.
        initial: f64,
        /// Multiplicative factor used when growing the active envelope.
        growth_factor: f64,
    },
}

impl Envelope {
    /// Construct an initial rejection envelope.
    pub fn initial(value: f64) -> Self {
        Self::Initial(value)
    }

    /// Construct an envelope estimated from a pilot sample.
    pub fn estimate(pilot_events: usize, safety_factor: f64) -> Self {
        Self::Estimate {
            pilot_events,
            safety_factor,
        }
    }

    /// Construct an adaptive rejection envelope.
    pub fn adaptive(initial: f64, growth_factor: f64) -> Self {
        Self::Adaptive {
            initial,
            growth_factor,
        }
    }

    pub(crate) fn validate(self) -> LadduResult<()> {
        match self {
            Self::Initial(value) => validate_envelope_value(value),
            Self::Estimate {
                pilot_events,
                safety_factor,
            } => {
                if pilot_events == 0 {
                    return Err(LadduError::Custom(
                        "envelope estimation requires at least one pilot event".to_string(),
                    ));
                }
                if !safety_factor.is_finite() || safety_factor <= 0.0 {
                    return Err(LadduError::Custom(format!(
                        "envelope safety factor must be finite and positive, got {safety_factor}"
                    )));
                }
                Ok(())
            }
            Self::Adaptive {
                initial,
                growth_factor,
            } => {
                validate_envelope_value(initial)?;
                if !growth_factor.is_finite() || growth_factor <= 1.0 {
                    return Err(LadduError::Custom(format!(
                        "adaptive envelope growth factor must be finite and greater than one, got {growth_factor}"
                    )));
                }
                Ok(())
            }
        }
    }
}

impl From<f64> for Envelope {
    fn from(value: f64) -> Self {
        Self::initial(value)
    }
}

/// Behavior when a rejection-sampling weight exceeds the active envelope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnvelopeViolationPolicy {
    /// Return an error immediately.
    #[default]
    Error,
    /// Count the violation and accept the proposal with probability one.
    WarnAndContinue,
    /// Count the violation, grow the active envelope to the observed weight, and continue.
    Grow,
}

/// Statistics about rejection-envelope usage.
#[derive(Clone, Debug, Default)]
pub struct EnvelopeStats {
    /// Configured envelope value at the beginning of the run.
    pub configured_max: Option<f64>,
    /// Number of pilot proposal events used to estimate the envelope.
    pub pilot_events: u64,
    /// Largest event weight observed during envelope estimation.
    pub pilot_observed_max: Option<f64>,
    /// Safety factor applied to the pilot maximum.
    pub safety_factor: Option<f64>,
    /// Growth factor used by adaptive envelopes.
    pub growth_factor: Option<f64>,
    /// Largest proposal weight observed by the run.
    pub observed_max: Option<f64>,
    /// Number of proposal weights that exceeded the active rejection envelope.
    pub violations: u64,
    /// Largest ratio of observed weight to active envelope for violating events.
    pub largest_violation_ratio: Option<f64>,
    /// Number of times the active envelope was updated.
    pub updates: u64,
    /// Active envelope value at the end of the run.
    pub final_max: Option<f64>,
}

impl EnvelopeStats {
    /// Construct envelope statistics for an initial envelope.
    pub fn initial(value: f64) -> Self {
        Self {
            configured_max: Some(value),
            final_max: Some(value),
            ..Self::default()
        }
    }

    /// Construct envelope statistics for an estimated envelope.
    pub fn estimated(
        pilot_events: u64,
        pilot_observed_max: f64,
        safety_factor: f64,
        configured_max: f64,
    ) -> Self {
        Self {
            configured_max: Some(configured_max),
            pilot_events,
            pilot_observed_max: Some(pilot_observed_max),
            safety_factor: Some(safety_factor),
            observed_max: Some(pilot_observed_max),
            final_max: Some(configured_max),
            ..Self::default()
        }
    }

    pub(crate) fn observe(&mut self, weight: f64, active_envelope: f64) {
        self.observed_max = Some(self.observed_max.map_or(weight, |value| value.max(weight)));
        if weight > active_envelope {
            self.violations += 1;
            let ratio = weight / active_envelope;
            self.largest_violation_ratio = Some(
                self.largest_violation_ratio
                    .map_or(ratio, |value| value.max(ratio)),
            );
        }
    }

    pub(crate) fn update_final_max(&mut self, value: f64) {
        if self.final_max != Some(value) {
            self.updates += 1;
        }
        self.final_max = Some(value);
    }

    /// Construct envelope statistics for an adaptive envelope.
    pub fn adaptive(initial: f64, growth_factor: f64) -> Self {
        Self {
            configured_max: Some(initial),
            growth_factor: Some(growth_factor),
            final_max: Some(initial),
            ..Self::default()
        }
    }
}

pub(crate) fn validate_envelope_value(value: f64) -> LadduResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(LadduError::Custom(format!(
            "rejection envelope must be finite and positive, got {value}"
        )));
    }
    Ok(())
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
    /// Rejection sample proposal events using a configured initial envelope.
    Accepted {
        /// Expression evaluated on generated proposal events.
        expression: Box<laddu_core::Expression>,
        /// Free-parameter values passed to the expression evaluator.
        parameters: Vec<f64>,
        /// Rejection-envelope configuration.
        envelope: Envelope,
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
    /// Behavior when rejection sampling observes weights above the active envelope.
    pub envelope_violation_policy: EnvelopeViolationPolicy,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            batch_size: 10_000,
            max_trials: None,
            seed: None,
            envelope_violation_policy: EnvelopeViolationPolicy::Error,
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

    /// Set envelope-violation behavior for rejection sampling.
    pub fn envelope_violation_policy(mut self, policy: EnvelopeViolationPolicy) -> Self {
        self.envelope_violation_policy = policy;
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
    /// Accepted rejection sampling.
    Accepted,
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
    /// Envelope statistics for rejection modes, when applicable.
    pub envelope_stats: Option<EnvelopeStats>,
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
    /// Return the final rejection envelope, when applicable.
    pub fn envelope(&self) -> Option<f64> {
        self.envelope_stats
            .as_ref()
            .and_then(|stats| stats.final_max)
    }

    /// Return the number of rejection-envelope violations.
    pub fn envelope_violations(&self) -> u64 {
        self.envelope_stats
            .as_ref()
            .map_or(0, |stats| stats.violations)
    }

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
            self.envelope_stats
                .as_ref()
                .and_then(|stats| stats.final_max)
                .map_or_else(|| "n/a".to_string(), |value| format!("{value:.6}")),
            self.envelope_stats
                .as_ref()
                .map_or(0, |stats| stats.violations),
            self.sum_weights,
            self.min_weight
                .map_or_else(|| "n/a".to_string(), |value| format!("{value:.6}")),
            self.max_weight
                .map_or_else(|| "n/a".to_string(), |value| format!("{value:.6}")),
            self.batches_written,
            self.elapsed.as_secs_f64(),
        )
    }

    #[cfg(feature = "mpi")]
    pub(crate) fn reduce_mpi(&mut self, world: &SimpleCommunicator) {
        self.written_events = mpi_sum_u64(world, self.written_events);
        self.proposed_events = mpi_sum_u64(world, self.proposed_events);
        self.accepted_events = mpi_sum_u64(world, self.accepted_events);
        self.rejected_events = mpi_sum_u64(world, self.rejected_events);
        self.sum_weights = mpi_sum_f64(world, self.sum_weights);
        self.batches_written = mpi_sum_u64(world, self.batches_written);
        self.elapsed = Duration::from_secs_f64(mpi_max_f64(world, self.elapsed.as_secs_f64()));

        let global_min = mpi_min_f64(world, self.min_weight.unwrap_or(f64::INFINITY));
        self.min_weight = global_min.is_finite().then_some(global_min);
        let global_max = mpi_max_f64(world, self.max_weight.unwrap_or(f64::NEG_INFINITY));
        self.max_weight = global_max.is_finite().then_some(global_max);

        if let Some(envelope_stats) = &mut self.envelope_stats {
            envelope_stats.reduce_mpi(world);
        }
        self.acceptance_rate = match self.mode {
            GenerationModeKind::Accepted => {
                let pilot_events = self
                    .envelope_stats
                    .as_ref()
                    .map_or(0, |stats| stats.pilot_events);
                let sampled_proposals = self.proposed_events.saturating_sub(pilot_events);
                (sampled_proposals > 0)
                    .then_some(self.accepted_events as f64 / sampled_proposals as f64)
            }
            GenerationModeKind::Raw | GenerationModeKind::Weighted => None,
        };
    }
}

#[cfg(feature = "mpi")]
impl EnvelopeStats {
    pub(crate) fn reduce_mpi(&mut self, world: &SimpleCommunicator) {
        self.pilot_events = mpi_sum_u64(world, self.pilot_events);
        self.violations = mpi_sum_u64(world, self.violations);
        self.updates = mpi_sum_u64(world, self.updates);
        self.configured_max = mpi_reduce_optional_max(world, self.configured_max);
        self.pilot_observed_max = mpi_reduce_optional_max(world, self.pilot_observed_max);
        self.safety_factor = mpi_reduce_optional_max(world, self.safety_factor);
        self.growth_factor = mpi_reduce_optional_max(world, self.growth_factor);
        self.observed_max = mpi_reduce_optional_max(world, self.observed_max);
        self.largest_violation_ratio = mpi_reduce_optional_max(world, self.largest_violation_ratio);
        self.final_max = mpi_reduce_optional_max(world, self.final_max);
    }
}

#[cfg(feature = "mpi")]
fn mpi_sum_u64(world: &SimpleCommunicator, value: u64) -> u64 {
    let mut reduced = 0;
    world.all_reduce_into(&value, &mut reduced, SystemOperation::sum());
    reduced
}

#[cfg(feature = "mpi")]
fn mpi_sum_f64(world: &SimpleCommunicator, value: f64) -> f64 {
    let mut reduced = 0.0;
    world.all_reduce_into(&value, &mut reduced, SystemOperation::sum());
    reduced
}

#[cfg(feature = "mpi")]
fn mpi_min_f64(world: &SimpleCommunicator, value: f64) -> f64 {
    let mut reduced = f64::INFINITY;
    world.all_reduce_into(&value, &mut reduced, SystemOperation::min());
    reduced
}

#[cfg(feature = "mpi")]
fn mpi_max_f64(world: &SimpleCommunicator, value: f64) -> f64 {
    let mut reduced = f64::NEG_INFINITY;
    world.all_reduce_into(&value, &mut reduced, SystemOperation::max());
    reduced
}

#[cfg(feature = "mpi")]
fn mpi_reduce_optional_max(world: &SimpleCommunicator, value: Option<f64>) -> Option<f64> {
    let reduced = mpi_max_f64(world, value.unwrap_or(f64::NEG_INFINITY));
    reduced.is_finite().then_some(reduced)
}

/// Result of a generation call.
#[derive(Clone, Debug)]
pub struct GenerationResult<T> {
    /// Sink-specific output.
    pub output: T,
    /// Statistics collected during generation.
    pub stats: GenerationStats,
}
