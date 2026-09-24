//! ganesh-backed minimization and sampling adapters for laddu objectives.
//!
//! [`FitProblem`] implements ganesh's cost, gradient, and log-density traits
//! for both `f32` and `f64`. The adapter leaves algorithm choice and callbacks
//! fully exposed while centralizing laddu parameter conversion, metadata, and
//! stochastic batching.

pub use ganesh;

use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use ganesh::error::GaneshError;
use ganesh::traits::{
    Bounds as GaneshBounds, CostFunction, Gradient, IdentityTransform, LogDensity,
    PeriodicTransform, ScaleTransform, Transform,
};
use ganesh::{LinearAlgebra, RealScalar, Vector};
use ganesh::{NalgebraProvider, core::MinimizationSummary};
use laddu_expr::parameters::ParamLayout;
use laddu_likelihood::{LikelihoodError, LikelihoodEvaluation, Objective, StochasticObjective};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors raised while adapting a laddu objective to ganesh.
#[derive(Debug, Error)]
pub enum FitError {
    /// The likelihood could not be evaluated or prepared.
    #[error(transparent)]
    Likelihood(#[from] LikelihoodError),
    /// The underlying ganesh optimizer or sampler failed.
    #[error(transparent)]
    Optimizer(#[from] GaneshError),
    /// A backend scalar could not be represented as an `f64`.
    #[error("ganesh scalar value cannot be represented as f64")]
    ScalarConversion,
    /// A fit artifact could not be created or bound safely.
    #[error("fit artifact mismatch: {0}")]
    Artifact(String),
}

/// A result produced by a laddu fitting operation.
pub type FitResult<T> = Result<T, FitError>;

/// Terminal outcome of a deterministic fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitOutcome {
    /// The optimizer reported successful convergence.
    Converged,
    /// The optimizer stopped without reporting convergence.
    NotConverged,
}

/// Stable optimizer-independent evaluation counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitDiagnostics {
    function_evaluations: usize,
    gradient_evaluations: usize,
    hessian_evaluations: usize,
}

impl FitDiagnostics {
    /// Number of objective evaluations requested by the optimizer.
    pub const fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }
    /// Number of gradient evaluations requested by the optimizer.
    pub const fn gradient_evaluations(&self) -> usize {
        self.gradient_evaluations
    }
    /// Number of Hessian evaluations requested by the optimizer.
    pub const fn hessian_evaluations(&self) -> usize {
        self.hessian_evaluations
    }
}

/// laddu-owned façade for a deterministic minimization result.
#[derive(Clone)]
pub struct MinimizationResult {
    parameter_names: Vec<String>,
    values: Vec<f64>,
    objective: f64,
    outcome: FitOutcome,
    terminal_message: String,
    covariance: Option<Vec<Vec<f64>>>,
    standard_errors: Option<Vec<f64>>,
    diagnostics: FitDiagnostics,
    raw_ganesh_summary: MinimizationSummary<f64, NalgebraProvider>,
    likelihood_fingerprint: Option<String>,
    strong_compatibility: bool,
}

impl MinimizationResult {
    /// Build a stable result from a complete Ganesh summary.
    pub fn from_ganesh(
        summary: MinimizationSummary<f64, NalgebraProvider>,
        fallback_parameter_names: &[String],
    ) -> Self {
        let values = summary.x.to_vec();
        let parameter_names = summary
            .parameter_names
            .clone()
            .filter(|names| names.len() == values.len())
            .unwrap_or_else(|| fallback_parameter_names.to_vec());
        let standard_errors = (summary.std.len() == values.len())
            .then(|| summary.std.to_vec())
            .filter(|errors| errors.iter().all(|value| value.is_finite()));
        let covariance = (standard_errors.is_some()
            && summary.covariance.rows() == values.len()
            && summary.covariance.cols() == values.len())
        .then(|| {
            (0..values.len())
                .map(|row| {
                    (0..values.len())
                        .map(|column| summary.covariance.get(row, column))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .filter(|matrix| matrix.iter().flatten().all(|value| value.is_finite()));
        let outcome = if summary.message.success() {
            FitOutcome::Converged
        } else {
            FitOutcome::NotConverged
        };
        let diagnostics = FitDiagnostics {
            function_evaluations: summary.evals.f(),
            gradient_evaluations: summary.evals.g(),
            hessian_evaluations: summary.evals.h(),
        };
        Self {
            parameter_names,
            values,
            objective: summary.fx,
            outcome,
            terminal_message: summary.message.to_string(),
            covariance,
            standard_errors,
            diagnostics,
            raw_ganesh_summary: summary,
            likelihood_fingerprint: None,
            strong_compatibility: false,
        }
    }

    /// Attach the versioned structural identity of the fitted likelihood.
    pub fn with_likelihood_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.likelihood_fingerprint = Some(fingerprint.into());
        self
    }
    /// Record whether every objective component supplied a stable identity.
    pub const fn with_strong_compatibility(mut self, strong: bool) -> Self {
        self.strong_compatibility = strong;
        self
    }

    /// Create a dataset-free, inspectable artifact from this terminal result.
    pub fn artifact(&self) -> FitResult<FitArtifact> {
        let fingerprint = self
            .likelihood_fingerprint
            .clone()
            .ok_or_else(|| FitError::Artifact("result has no likelihood fingerprint".to_owned()))?;
        Ok(FitArtifact {
            artifact_kind: "laddu.fit".to_owned(),
            schema_version: 1,
            laddu_version: env!("CARGO_PKG_VERSION").to_owned(),
            fingerprint_version: 1,
            likelihood_fingerprint: fingerprint,
            parameter_names: self.parameter_names.clone(),
            values: self.values.clone(),
            objective: self.objective,
            outcome: self.outcome,
            terminal_message: self.terminal_message.clone(),
            diagnostics: self.diagnostics,
            covariance: self.covariance.clone(),
            standard_errors: self.standard_errors.clone(),
            ensemble: None,
            strong_compatibility: self.strong_compatibility,
        })
    }

    /// Parameter names in canonical free-parameter order.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }
    /// Central free-parameter values in canonical order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Final objective value.
    pub const fn objective(&self) -> f64 {
        self.objective
    }
    /// Terminal fit outcome.
    pub const fn outcome(&self) -> FitOutcome {
        self.outcome
    }
    /// Whether the optimizer reported successful convergence.
    pub const fn converged(&self) -> bool {
        matches!(self.outcome, FitOutcome::Converged)
    }
    /// Complete terminal status text.
    pub fn terminal_message(&self) -> &str {
        &self.terminal_message
    }
    /// Covariance matrix when the optimizer computed a finite matrix.
    pub fn covariance(&self) -> Option<&[Vec<f64>]> {
        self.covariance.as_deref()
    }
    /// Parameter standard errors when the optimizer computed finite values.
    pub fn standard_errors(&self) -> Option<&[f64]> {
        self.standard_errors.as_deref()
    }
    /// Stable optimizer-independent diagnostics.
    pub const fn diagnostics(&self) -> FitDiagnostics {
        self.diagnostics
    }
    /// Complete dependency-specific optimizer summary.
    pub const fn raw_ganesh_summary(&self) -> &MinimizationSummary<f64, NalgebraProvider> {
        &self.raw_ganesh_summary
    }
}

/// Dataset-free deterministic-fit artifact.
#[derive(Clone, Debug)]
pub struct FitArtifact {
    artifact_kind: String,
    schema_version: u32,
    laddu_version: String,
    fingerprint_version: u32,
    likelihood_fingerprint: String,
    parameter_names: Vec<String>,
    values: Vec<f64>,
    objective: f64,
    outcome: FitOutcome,
    terminal_message: String,
    diagnostics: FitDiagnostics,
    covariance: Option<Vec<Vec<f64>>>,
    standard_errors: Option<Vec<f64>>,
    ensemble: Option<FitEnsembleArtifact>,
    strong_compatibility: bool,
}

impl FitArtifact {
    /// Stable artifact kind discriminator.
    pub fn artifact_kind(&self) -> &str {
        &self.artifact_kind
    }
    /// Artifact schema version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }
    /// Producing laddu crate version.
    pub fn laddu_version(&self) -> &str {
        &self.laddu_version
    }
    /// Structural fingerprint algorithm version.
    pub const fn fingerprint_version(&self) -> u32 {
        self.fingerprint_version
    }
    /// Dataset-free likelihood structure fingerprint.
    pub fn likelihood_fingerprint(&self) -> &str {
        &self.likelihood_fingerprint
    }
    /// Original canonical parameter names.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }
    /// Central values in original canonical order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Final objective value.
    pub const fn objective(&self) -> f64 {
        self.objective
    }
    /// Preserved terminal outcome.
    pub const fn outcome(&self) -> FitOutcome {
        self.outcome
    }
    /// Preserved terminal status text.
    pub fn terminal_message(&self) -> &str {
        &self.terminal_message
    }
    /// Stable evaluation diagnostics.
    pub const fn diagnostics(&self) -> FitDiagnostics {
        self.diagnostics
    }
    /// Optional finite covariance matrix.
    pub fn covariance(&self) -> Option<&[Vec<f64>]> {
        self.covariance.as_deref()
    }
    /// Optional finite standard errors.
    pub fn standard_errors(&self) -> Option<&[f64]> {
        self.standard_errors.as_deref()
    }
    /// Ordered paired ensemble, when attached.
    pub const fn ensemble(&self) -> Option<&FitEnsembleArtifact> {
        self.ensemble.as_ref()
    }
    /// Whether strict structural binding is provable for all components.
    pub const fn strong_compatibility(&self) -> bool {
        self.strong_compatibility
    }
    /// Return a copy with ordered ensemble metadata and draws attached.
    pub fn with_ensemble(mut self, ensemble: &laddu_likelihood::Ensemble) -> FitResult<Self> {
        if ensemble.parameter_names() != self.parameter_names {
            return Err(FitError::Artifact(
                "ensemble parameter schema differs from central fit".to_owned(),
            ));
        }
        self.ensemble = Some(FitEnsembleArtifact {
            source_id: ensemble.source_id(),
            parameter_names: ensemble.parameter_names().to_vec(),
            draws: ensemble.draws().to_vec(),
            draw_ids: (0..ensemble.len()).map(|index| index as u64).collect(),
            bootstrap_seed: ensemble.bootstrap_seed(),
            requires_external_datasets: ensemble.requires_external_replica_datasets(),
            failures: Vec::new(),
        });
        Ok(self)
    }
    /// Return a copy with structured failed-replica records attached.
    pub fn with_failed_replicas(mut self, failures: Vec<FailedReplica>) -> FitResult<Self> {
        let ensemble = self.ensemble.as_mut().ok_or_else(|| {
            FitError::Artifact("failed replicas require an attached ensemble".to_owned())
        })?;
        let mut indices = std::collections::HashSet::new();
        if let Some(duplicate) = failures
            .iter()
            .find(|failure| !indices.insert(failure.index))
        {
            return Err(FitError::Artifact(format!(
                "duplicate failed replica index {}",
                duplicate.index
            )));
        }
        let replica_count = ensemble
            .draws
            .len()
            .checked_add(failures.len())
            .ok_or_else(|| {
                FitError::Artifact("replica count overflows platform limits".to_owned())
            })?;
        if let Some(failure) = failures
            .iter()
            .find(|failure| failure.index >= replica_count)
        {
            return Err(FitError::Artifact(format!(
                "failed replica index {} is outside the reconstructed sequence",
                failure.index
            )));
        }
        ensemble.draw_ids = (0..replica_count)
            .filter(|index| !indices.contains(index))
            .map(|index| index as u64)
            .collect();
        ensemble.failures = failures;
        Ok(self)
    }

    /// Bind after exact structure and unique-name schema validation.
    pub fn bind(&self, likelihood: &laddu_likelihood::Likelihood) -> FitResult<BoundFitState> {
        if !self.strong_compatibility || !likelihood.artifact_compatibility_is_strong() {
            return Err(FitError::Artifact(
                "unidentified custom component prevents strong compatibility binding".to_owned(),
            ));
        }
        if self.artifact_kind != "laddu.fit"
            || self.schema_version != 1
            || self.fingerprint_version != 1
        {
            return Err(FitError::Artifact(
                "unsupported artifact kind or schema version".to_owned(),
            ));
        }
        let target_names = likelihood
            .params()
            .free_params()
            .iter()
            .map(|id| {
                likelihood
                    .params()
                    .name(*id)
                    .unwrap_or("<invalid>")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        let mut artifact_names = std::collections::HashSet::new();
        if let Some(duplicate) = self
            .parameter_names
            .iter()
            .find(|name| !artifact_names.insert((*name).clone()))
        {
            return Err(FitError::Artifact(format!(
                "artifact parameter `{duplicate}` is duplicated"
            )));
        }
        let missing = target_names
            .iter()
            .filter(|name| !artifact_names.contains(*name))
            .cloned()
            .collect::<Vec<_>>();
        let added = self
            .parameter_names
            .iter()
            .filter(|name| !target_names.contains(name))
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() || !added.is_empty() {
            return Err(FitError::Artifact(format!(
                "parameter schema differs; missing={missing:?}, added_or_renamed={added:?}"
            )));
        }
        if self.likelihood_fingerprint != likelihood.artifact_fingerprint_v1() {
            return Err(FitError::Artifact(
                "likelihood structural fingerprint differs".to_owned(),
            ));
        }
        let mut values = Vec::with_capacity(target_names.len());
        for name in &target_names {
            let matches = self
                .parameter_names
                .iter()
                .enumerate()
                .filter(|(_, candidate)| *candidate == name)
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(FitError::Artifact(format!(
                    "parameter `{name}` is missing or duplicated"
                )));
            }
            values.push(self.values[matches[0].0]);
        }
        if target_names.len() != self.parameter_names.len() {
            return Err(FitError::Artifact(
                "artifact has added or renamed parameters".to_owned(),
            ));
        }
        Ok(BoundFitState {
            parameter_names: target_names,
            values,
            objective: self.objective,
            outcome: self.outcome,
            migration: Vec::new(),
        })
    }

    /// Bind with an explicit one-to-one artifact-name to target-name migration.
    pub fn bind_with_parameter_map(
        &self,
        likelihood: &laddu_likelihood::Likelihood,
        mapping: &std::collections::HashMap<String, String>,
    ) -> FitResult<BoundFitState> {
        if !self.strong_compatibility || !likelihood.artifact_compatibility_is_strong() {
            return Err(FitError::Artifact(
                "unidentified custom component prevents migrated binding".to_owned(),
            ));
        }
        let target_names = likelihood
            .params()
            .free_params()
            .iter()
            .map(|id| {
                likelihood
                    .params()
                    .name(*id)
                    .unwrap_or("<invalid>")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        if mapping.len() != self.parameter_names.len()
            || mapping
                .keys()
                .any(|name| !self.parameter_names.contains(name))
        {
            return Err(FitError::Artifact(
                "parameter migration must map every artifact name exactly once".to_owned(),
            ));
        }
        let targets = mapping
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        if targets.len() != mapping.len()
            || targets.len() != target_names.len()
            || targets.iter().any(|name| !target_names.contains(name))
        {
            return Err(FitError::Artifact(
                "parameter migration targets must be unique and cover the target schema".to_owned(),
            ));
        }
        let target_to_artifact = mapping
            .iter()
            .map(|(artifact, target)| (target.clone(), artifact.clone()))
            .collect::<std::collections::HashMap<_, _>>();
        if self.likelihood_fingerprint
            != likelihood.artifact_fingerprint_v1_with_parameter_map(&target_to_artifact)
        {
            return Err(FitError::Artifact(
                "likelihood structural fingerprint differs after parameter migration".to_owned(),
            ));
        }
        let mut values = Vec::with_capacity(target_names.len());
        for target in &target_names {
            let source = mapping
                .iter()
                .find_map(|(source, candidate)| (candidate == target).then_some(source))
                .expect("validated target coverage");
            let index = self
                .parameter_names
                .iter()
                .position(|name| name == source)
                .expect("validated source coverage");
            values.push(self.values[index]);
        }
        let mut migration = mapping
            .iter()
            .map(|(source, target)| (source.clone(), target.clone()))
            .collect::<Vec<_>>();
        migration.sort();
        Ok(BoundFitState {
            parameter_names: target_names,
            values,
            objective: self.objective,
            outcome: self.outcome,
            migration,
        })
    }
}

/// One failed replica retained without event data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailedReplica {
    /// Replica index in the requested sequence.
    pub index: usize,
    /// Deterministic reconstruction seed, when applicable.
    pub seed: Option<u64>,
    /// Structured failure message.
    pub message: String,
}

/// Ordered parameter draws and pairing identity stored in an artifact.
#[derive(Clone, Debug)]
pub struct FitEnsembleArtifact {
    source_id: u64,
    parameter_names: Vec<String>,
    draws: Vec<Vec<f64>>,
    draw_ids: Vec<u64>,
    bootstrap_seed: Option<u64>,
    requires_external_datasets: bool,
    failures: Vec<FailedReplica>,
}
impl FitEnsembleArtifact {
    /// Correlation identity retained for paired arithmetic.
    pub const fn source_id(&self) -> u64 {
        self.source_id
    }
    /// Draw-column names.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }
    /// Ordered successful parameter rows.
    pub fn draws(&self) -> &[Vec<f64>] {
        &self.draws
    }
    /// Stable ordered draw identifiers.
    pub fn draw_ids(&self) -> &[u64] {
        &self.draw_ids
    }
    /// Base deterministic bootstrap seed.
    pub const fn bootstrap_seed(&self) -> Option<u64> {
        self.bootstrap_seed
    }
    /// Whether arbitrary replica datasets must be supplied externally.
    pub const fn requires_external_datasets(&self) -> bool {
        self.requires_external_datasets
    }
    /// Failed replica records.
    pub fn failures(&self) -> &[FailedReplica] {
        &self.failures
    }
    /// Rebuild a parameter-only ensemble with the original source identity.
    pub fn to_parameter_ensemble(&self) -> FitResult<laddu_likelihood::Ensemble> {
        laddu_likelihood::Ensemble::with_source_id(
            self.parameter_names.clone(),
            self.draws.clone(),
            self.source_id,
        )
        .map_err(FitError::from)
    }
    /// Reconstruct deterministic bootstrap replicas against a compatible likelihood.
    pub fn to_ensemble_for(
        &self,
        likelihood: &std::sync::Arc<laddu_likelihood::Likelihood>,
    ) -> FitResult<laddu_likelihood::Ensemble> {
        if self.requires_external_datasets {
            return Err(FitError::Artifact(
                "ensemble requires caller-supplied external replica datasets".to_owned(),
            ));
        }
        let Some(seed) = self.bootstrap_seed else {
            return self.to_parameter_ensemble();
        };
        let replicas = self
            .draw_ids
            .iter()
            .map(|draw_id| {
                likelihood
                    .bootstrap(seed.wrapping_add(*draw_id))
                    .map(std::sync::Arc::new)
            })
            .collect::<Result<Vec<_>, _>>()?;
        laddu_likelihood::Ensemble::with_replicas_and_source_id(
            self.parameter_names.clone(),
            self.draws.clone(),
            replicas,
            self.source_id,
        )
        .map_err(FitError::from)
    }
}

/// Fit values validated and reordered for one reconstructed likelihood.
#[derive(Clone, Debug)]
pub struct BoundFitState {
    parameter_names: Vec<String>,
    values: Vec<f64>,
    objective: f64,
    outcome: FitOutcome,
    migration: Vec<(String, String)>,
}

/// Passive snapshot component registry that preserves shared fit-artifact identity.
#[derive(Clone, Debug, Default)]
pub struct AnalysisSnapshot {
    fits: std::collections::HashMap<String, std::sync::Arc<FitArtifact>>,
}
impl AnalysisSnapshot {
    /// Create an empty passive snapshot.
    pub fn new() -> Self {
        Self::default()
    }
    /// Insert one artifact object under a unique path.
    pub fn insert_fit(
        &mut self,
        path: impl Into<String>,
        artifact: std::sync::Arc<FitArtifact>,
    ) -> FitResult<()> {
        let path = path.into();
        if path.is_empty() || path.contains("..") {
            return Err(FitError::Artifact(format!(
                "invalid snapshot object path `{path}`"
            )));
        }
        if self.fits.contains_key(&path) {
            return Err(FitError::Artifact(format!(
                "duplicate snapshot object path `{path}`"
            )));
        }
        self.fits.insert(path, artifact);
        Ok(())
    }
    /// Add another path referencing the exact same artifact object.
    pub fn alias_fit(&mut self, path: impl Into<String>, existing: &str) -> FitResult<()> {
        let artifact = self.fits.get(existing).cloned().ok_or_else(|| {
            FitError::Artifact(format!("snapshot fit path `{existing}` is missing"))
        })?;
        self.insert_fit(path, artifact)
    }
    /// Resolve an embedded artifact without binding or evaluation.
    pub fn fit(&self, path: &str) -> Option<&std::sync::Arc<FitArtifact>> {
        self.fits.get(path)
    }

    /// Return sorted embedded fit paths.
    pub fn fit_paths(&self) -> Vec<&str> {
        let mut paths = self.fits.keys().map(String::as_str).collect::<Vec<_>>();
        paths.sort_unstable();
        paths
    }

    /// Save a passive snapshot with each shared artifact embedded once.
    pub fn save(&self, path: impl AsRef<std::path::Path>, overwrite: bool) -> FitResult<()> {
        use std::io::Write;
        let mut artifact_ids = std::collections::HashMap::<usize, usize>::new();
        let mut artifacts = Vec::<Vec<u8>>::new();
        let mut entries = Vec::new();
        for object_path in self.fit_paths() {
            let artifact = self.fits.get(object_path).expect("path came from map");
            let pointer = std::sync::Arc::as_ptr(artifact) as usize;
            let artifact_id = if let Some(id) = artifact_ids.get(&pointer) {
                *id
            } else {
                let id = artifacts.len();
                let temporary = unique_temporary_path("embedded-fit");
                artifact.save(&temporary, false)?;
                let bytes = std::fs::read(&temporary).map_err(|error| {
                    FitError::Artifact(format!("embedded artifact read failed: {error}"))
                });
                let _ = std::fs::remove_file(&temporary);
                artifacts.push(bytes?);
                artifact_ids.insert(pointer, id);
                id
            };
            entries.push(SnapshotEntry {
                path: object_path.to_owned(),
                artifact_id,
            });
        }
        let manifest = SnapshotManifest {
            kind: "laddu.analysis-snapshot".to_owned(),
            schema_version: 1,
            entries,
            artifacts: artifacts
                .iter()
                .map(|bytes| SnapshotArtifact {
                    byte_length: bytes.len(),
                    checksum: payload_checksum(bytes),
                })
                .collect(),
        };
        let manifest = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            FitError::Artifact(format!("snapshot manifest serialization failed: {error}"))
        })?;
        let destination = path.as_ref();
        if destination.exists() && !overwrite {
            return Err(FitError::Artifact(format!(
                "snapshot already exists: {}",
                destination.display()
            )));
        }
        let temporary = unique_sibling_path(destination, "snapshot");
        let result = (|| -> FitResult<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|error| FitError::Artifact(format!("snapshot create failed: {error}")))?;
            file.write_all(b"LADDUSNAPSHOT\n")
                .and_then(|_| file.write_all(&(manifest.len() as u64).to_le_bytes()))
                .and_then(|_| file.write_all(&manifest))
                .map_err(|error| FitError::Artifact(format!("snapshot write failed: {error}")))?;
            for bytes in &artifacts {
                file.write_all(bytes).map_err(|error| {
                    FitError::Artifact(format!("embedded artifact write failed: {error}"))
                })?;
            }
            file.sync_all()
                .map_err(|error| FitError::Artifact(format!("snapshot sync failed: {error}")))?;
            publish_temporary(&temporary, destination, overwrite, "snapshot")
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }

    /// Load a passive snapshot and preserve aliases to shared artifact objects.
    pub fn load(path: impl AsRef<std::path::Path>) -> FitResult<Self> {
        const MAX: usize = 256 * 1024 * 1024;
        let bytes = std::fs::read(path.as_ref())
            .map_err(|error| FitError::Artifact(format!("snapshot read failed: {error}")))?;
        if bytes.len() > MAX || !bytes.starts_with(b"LADDUSNAPSHOT\n") {
            return Err(FitError::Artifact(
                "invalid or oversized analysis snapshot".to_owned(),
            ));
        }
        let mut cursor = b"LADDUSNAPSHOT\n".len();
        let length_bytes: [u8; 8] = bytes
            .get(cursor..cursor + 8)
            .ok_or_else(|| FitError::Artifact("truncated snapshot header".to_owned()))?
            .try_into()
            .expect("checked length");
        cursor += 8;
        let manifest_length = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| FitError::Artifact("snapshot manifest is too large".to_owned()))?;
        let manifest_end = cursor
            .checked_add(manifest_length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| FitError::Artifact("truncated snapshot manifest".to_owned()))?;
        let manifest: SnapshotManifest = serde_json::from_slice(&bytes[cursor..manifest_end])
            .map_err(|error| FitError::Artifact(format!("invalid snapshot manifest: {error}")))?;
        if manifest.kind != "laddu.analysis-snapshot" || manifest.schema_version != 1 {
            return Err(FitError::Artifact(
                "unsupported analysis snapshot schema".to_owned(),
            ));
        }
        cursor = manifest_end;
        let first_paths = manifest.entries.iter().fold(
            std::collections::HashMap::<usize, &str>::new(),
            |mut paths, entry| {
                paths.entry(entry.artifact_id).or_insert(&entry.path);
                paths
            },
        );
        let mut artifacts = Vec::with_capacity(manifest.artifacts.len());
        for (artifact_id, embedded) in manifest.artifacts.iter().enumerate() {
            let object_path = first_paths
                .get(&artifact_id)
                .copied()
                .unwrap_or("<unreferenced>");
            let end = cursor
                .checked_add(embedded.byte_length)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| {
                    FitError::Artifact(format!(
                        "snapshot path `{object_path}` has a truncated embedded artifact"
                    ))
                })?;
            let archive = &bytes[cursor..end];
            cursor = end;
            if payload_checksum(archive) != embedded.checksum {
                return Err(FitError::Artifact(format!(
                    "snapshot path `{object_path}` embedded artifact checksum mismatch"
                )));
            }
            let temporary = unique_temporary_path("embedded-fit-load");
            std::fs::write(&temporary, archive).map_err(|error| {
                FitError::Artifact(format!("embedded artifact staging failed: {error}"))
            })?;
            let artifact = FitArtifact::load(&temporary);
            let _ = std::fs::remove_file(&temporary);
            artifacts.push(std::sync::Arc::new(artifact.map_err(|error| {
                FitError::Artifact(format!(
                    "snapshot path `{object_path}` embedded artifact: {error}"
                ))
            })?));
        }
        if cursor != bytes.len() {
            return Err(FitError::Artifact(
                "snapshot contains undeclared trailing bytes".to_owned(),
            ));
        }
        let mut snapshot = Self::new();
        for entry in manifest.entries {
            let artifact = artifacts.get(entry.artifact_id).cloned().ok_or_else(|| {
                FitError::Artifact(format!(
                    "snapshot path `{}` references a missing artifact",
                    entry.path
                ))
            })?;
            snapshot.insert_fit(entry.path, artifact)?;
        }
        Ok(snapshot)
    }
}

#[derive(Serialize, Deserialize)]
struct SnapshotManifest {
    kind: String,
    schema_version: u32,
    entries: Vec<SnapshotEntry>,
    artifacts: Vec<SnapshotArtifact>,
}
#[derive(Serialize, Deserialize)]
struct SnapshotEntry {
    path: String,
    artifact_id: usize,
}
#[derive(Serialize, Deserialize)]
struct SnapshotArtifact {
    byte_length: usize,
    checksum: String,
}

#[derive(Serialize, Deserialize)]
struct ArtifactManifest {
    artifact_kind: String,
    schema_version: u32,
    laddu_version: String,
    fingerprint_version: u32,
    likelihood_fingerprint: String,
    parameter_names: Vec<String>,
    objective: f64,
    outcome: String,
    terminal_message: String,
    diagnostics: [usize; 3],
    payloads: Vec<PayloadManifest>,
    ensemble: Option<EnsembleManifest>,
    #[serde(default)]
    strong_compatibility: bool,
}
#[derive(Serialize, Deserialize)]
struct EnsembleManifest {
    source_id: u64,
    parameter_names: Vec<String>,
    draw_ids: Vec<u64>,
    bootstrap_seed: Option<u64>,
    requires_external_datasets: bool,
    failures: Vec<(usize, Option<u64>, String)>,
}
#[derive(Serialize, Deserialize)]
struct PayloadManifest {
    role: String,
    dtype: String,
    shape: Vec<usize>,
    byte_length: usize,
    checksum: String,
}

fn decode_artifact_manifest(bytes: &[u8]) -> FitResult<ArtifactManifest> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| FitError::Artifact(format!("malformed manifest: {error}")))?;
    let version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| FitError::Artifact("manifest has no schema version".to_owned()))?;
    match version {
        1 => serde_json::from_value(value)
            .map_err(|error| FitError::Artifact(format!("malformed v1 manifest: {error}"))),
        _ => Err(FitError::Artifact(format!(
            "unsupported fit artifact schema version {version}"
        ))),
    }
}

fn payload_checksum(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("{hash:016x}")
}
fn unique_temporary_path(label: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("laddu-{label}-{}-{nonce}", std::process::id()))
}
fn unique_sibling_path(path: &std::path::Path, label: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    path.with_extension(format!("laddu-{label}-{}-{nonce}", std::process::id()))
}
fn publish_temporary(
    temporary: &std::path::Path,
    destination: &std::path::Path,
    overwrite: bool,
    label: &str,
) -> FitResult<()> {
    if overwrite {
        std::fs::rename(temporary, destination)
            .map_err(|error| FitError::Artifact(format!("atomic {label} rename failed: {error}")))
    } else {
        std::fs::hard_link(temporary, destination).map_err(|error| {
            FitError::Artifact(format!("atomic {label} create failed: {error}"))
        })?;
        std::fs::remove_file(temporary).map_err(|error| {
            FitError::Artifact(format!("temporary {label} cleanup failed: {error}"))
        })
    }
}
fn encode_f64(values: impl IntoIterator<Item = f64>) -> Vec<u8> {
    values.into_iter().flat_map(f64::to_le_bytes).collect()
}
fn decode_f64(bytes: &[u8], role: &str) -> FitResult<Vec<f64>> {
    if bytes.len() % 8 != 0 {
        return Err(FitError::Artifact(format!(
            "payload `{role}` has invalid byte length"
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect())
}

impl FitArtifact {
    /// Save one passive manifest-plus-binary archive with an atomic rename.
    pub fn save(&self, path: impl AsRef<std::path::Path>, overwrite: bool) -> FitResult<()> {
        use std::io::Write;
        let path = path.as_ref();
        if path.exists() && !overwrite {
            return Err(FitError::Artifact(format!(
                "artifact already exists: {}",
                path.display()
            )));
        }
        let mut payloads = Vec::<(PayloadManifest, Vec<u8>)>::new();
        let mut add = |role: &str, shape: Vec<usize>, bytes: Vec<u8>| {
            payloads.push((
                PayloadManifest {
                    role: role.to_owned(),
                    dtype: "<f8".to_owned(),
                    shape,
                    byte_length: bytes.len(),
                    checksum: payload_checksum(&bytes),
                },
                bytes,
            ))
        };
        add(
            "central_parameters",
            vec![self.values.len()],
            encode_f64(self.values.iter().copied()),
        );
        if let Some(covariance) = &self.covariance {
            add(
                "covariance",
                vec![covariance.len(), covariance.len()],
                encode_f64(covariance.iter().flatten().copied()),
            );
        }
        if let Some(errors) = &self.standard_errors {
            add(
                "standard_errors",
                vec![errors.len()],
                encode_f64(errors.iter().copied()),
            );
        }
        if let Some(ensemble) = &self.ensemble {
            add(
                "ensemble_draws",
                vec![ensemble.draws.len(), ensemble.parameter_names.len()],
                encode_f64(ensemble.draws.iter().flatten().copied()),
            );
        }
        let manifest = ArtifactManifest {
            artifact_kind: self.artifact_kind.clone(),
            schema_version: self.schema_version,
            laddu_version: self.laddu_version.clone(),
            fingerprint_version: self.fingerprint_version,
            likelihood_fingerprint: self.likelihood_fingerprint.clone(),
            parameter_names: self.parameter_names.clone(),
            objective: self.objective,
            outcome: match self.outcome {
                FitOutcome::Converged => "converged",
                FitOutcome::NotConverged => "not_converged",
            }
            .to_owned(),
            terminal_message: self.terminal_message.clone(),
            diagnostics: [
                self.diagnostics.function_evaluations,
                self.diagnostics.gradient_evaluations,
                self.diagnostics.hessian_evaluations,
            ],
            payloads: payloads
                .iter()
                .map(|(entry, _)| PayloadManifest {
                    role: entry.role.clone(),
                    dtype: entry.dtype.clone(),
                    shape: entry.shape.clone(),
                    byte_length: entry.byte_length,
                    checksum: entry.checksum.clone(),
                })
                .collect(),
            ensemble: self.ensemble.as_ref().map(|ensemble| EnsembleManifest {
                source_id: ensemble.source_id,
                parameter_names: ensemble.parameter_names.clone(),
                draw_ids: ensemble.draw_ids.clone(),
                bootstrap_seed: ensemble.bootstrap_seed,
                requires_external_datasets: ensemble.requires_external_datasets,
                failures: ensemble
                    .failures
                    .iter()
                    .map(|failure| (failure.index, failure.seed, failure.message.clone()))
                    .collect(),
            }),
            strong_compatibility: self.strong_compatibility,
        };
        let manifest = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            FitError::Artifact(format!("manifest serialization failed: {error}"))
        })?;
        static TEMPORARY_COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let nonce = TEMPORARY_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temporary = path.with_extension(format!("laddu-tmp-{}-{nonce}", std::process::id()));
        let result = (|| -> FitResult<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|error| {
                    FitError::Artifact(format!("temporary artifact creation failed: {error}"))
                })?;
            file.write_all(b"LADDUFIT\n")
                .and_then(|_| file.write_all(&(manifest.len() as u64).to_le_bytes()))
                .and_then(|_| file.write_all(&manifest))
                .map_err(|error| FitError::Artifact(format!("artifact write failed: {error}")))?;
            for (_, bytes) in &payloads {
                file.write_all(bytes).map_err(|error| {
                    FitError::Artifact(format!("artifact payload write failed: {error}"))
                })?;
            }
            file.sync_all()
                .map_err(|error| FitError::Artifact(format!("artifact sync failed: {error}")))?;
            if overwrite {
                std::fs::rename(&temporary, path).map_err(|error| {
                    FitError::Artifact(format!("atomic artifact rename failed: {error}"))
                })
            } else {
                std::fs::hard_link(&temporary, path).map_err(|error| {
                    FitError::Artifact(format!("atomic artifact create failed: {error}"))
                })?;
                std::fs::remove_file(&temporary).map_err(|error| {
                    FitError::Artifact(format!("temporary artifact cleanup failed: {error}"))
                })
            }
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }

    /// Load and validate a passive archive without constructing a likelihood.
    pub fn load(path: impl AsRef<std::path::Path>) -> FitResult<Self> {
        use std::io::Read;
        const MAX: u64 = 256 * 1024 * 1024;
        let mut file = std::fs::File::open(path.as_ref())
            .map_err(|error| FitError::Artifact(format!("artifact open failed: {error}")))?;
        let size = file
            .metadata()
            .map_err(|error| FitError::Artifact(error.to_string()))?
            .len();
        if size > MAX {
            return Err(FitError::Artifact(format!(
                "artifact size {size} exceeds supported maximum {MAX}"
            )));
        }
        let mut bytes = Vec::with_capacity(size as usize);
        file.read_to_end(&mut bytes)
            .map_err(|error| FitError::Artifact(format!("artifact read failed: {error}")))?;
        if bytes.len() < 17 || &bytes[..9] != b"LADDUFIT\n" {
            return Err(FitError::Artifact(
                "invalid or truncated artifact header".to_owned(),
            ));
        }
        let manifest_len =
            u64::from_le_bytes(bytes[9..17].try_into().expect("header length")) as usize;
        let manifest_end = 17usize
            .checked_add(manifest_len)
            .ok_or_else(|| FitError::Artifact("manifest length overflow".to_owned()))?;
        if manifest_end > bytes.len() {
            return Err(FitError::Artifact("truncated artifact manifest".to_owned()));
        }
        let manifest = decode_artifact_manifest(&bytes[17..manifest_end])?;
        if manifest.artifact_kind != "laddu.fit"
            || manifest.schema_version != 1
            || manifest.fingerprint_version != 1
        {
            return Err(FitError::Artifact(format!(
                "unsupported artifact kind/version: {}/{} (supported laddu.fit/1)",
                manifest.artifact_kind, manifest.schema_version
            )));
        }
        let mut cursor = manifest_end;
        let mut values = None;
        let mut covariance = None;
        let mut standard_errors = None;
        let mut ensemble_draws = None;
        let mut roles = std::collections::HashSet::new();
        let parameter_count = manifest.parameter_names.len();
        for entry in &manifest.payloads {
            if !roles.insert(&entry.role) {
                return Err(FitError::Artifact(format!(
                    "duplicate payload role `{}`",
                    entry.role
                )));
            }
            if entry.dtype != "<f8" {
                return Err(FitError::Artifact(format!(
                    "payload `{}` has unsupported dtype {}",
                    entry.role, entry.dtype
                )));
            }
            let elements = entry
                .shape
                .iter()
                .try_fold(1usize, |total, value| total.checked_mul(*value))
                .ok_or_else(|| {
                    FitError::Artifact(format!("payload `{}` shape overflows", entry.role))
                })?;
            if elements.checked_mul(8) != Some(entry.byte_length) {
                return Err(FitError::Artifact(format!(
                    "payload `{}` shape and byte length disagree",
                    entry.role
                )));
            }
            let shape_is_valid = match entry.role.as_str() {
                "central_parameters" | "standard_errors" => {
                    entry.shape.as_slice() == [parameter_count]
                }
                "covariance" => entry.shape.as_slice() == [parameter_count, parameter_count],
                "ensemble_draws" => {
                    let Some(metadata) = manifest.ensemble.as_ref() else {
                        return Err(FitError::Artifact(
                            "ensemble draw payload has no metadata".to_owned(),
                        ));
                    };
                    entry.shape.len() == 2
                        && entry.shape[1] > 0
                        && entry.shape[0] == metadata.draw_ids.len()
                        && entry.shape[1] == metadata.parameter_names.len()
                }
                _ => false,
            };
            if !shape_is_valid {
                return Err(FitError::Artifact(format!(
                    "payload `{}` has an invalid scientific shape {:?}",
                    entry.role, entry.shape
                )));
            }
            let end = cursor.checked_add(entry.byte_length).ok_or_else(|| {
                FitError::Artifact(format!("payload `{}` length overflows", entry.role))
            })?;
            if end > bytes.len() {
                return Err(FitError::Artifact(format!(
                    "payload `{}` is truncated",
                    entry.role
                )));
            }
            let payload = &bytes[cursor..end];
            cursor = end;
            if payload_checksum(payload) != entry.checksum {
                return Err(FitError::Artifact(format!(
                    "payload `{}` checksum mismatch",
                    entry.role
                )));
            }
            let decoded = decode_f64(payload, &entry.role)?;
            match entry.role.as_str() {
                "central_parameters" => values = Some(decoded),
                "covariance" => {
                    covariance = Some(
                        decoded
                            .chunks(entry.shape[1])
                            .map(<[f64]>::to_vec)
                            .collect(),
                    )
                }
                "standard_errors" => standard_errors = Some(decoded),
                "ensemble_draws" => ensemble_draws = Some((entry.shape.clone(), decoded)),
                _ => unreachable!("payload roles were validated above"),
            }
        }
        if cursor != bytes.len() {
            return Err(FitError::Artifact(
                "archive contains undeclared trailing bytes".to_owned(),
            ));
        }
        let values = values
            .ok_or_else(|| FitError::Artifact("missing central_parameters payload".to_owned()))?;
        if values.len() != manifest.parameter_names.len() {
            return Err(FitError::Artifact(
                "parameter names and values differ in length".to_owned(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        if let Some(name) = manifest
            .parameter_names
            .iter()
            .find(|name| !names.insert((*name).clone()))
        {
            return Err(FitError::Artifact(format!(
                "duplicate parameter name `{name}`"
            )));
        }
        let ensemble = match manifest.ensemble {
            Some(metadata) => {
                let (shape, flat) = ensemble_draws.ok_or_else(|| {
                    FitError::Artifact("ensemble metadata has no draw payload".to_owned())
                })?;
                if shape.len() != 2
                    || shape[1] != metadata.parameter_names.len()
                    || shape[0] != metadata.draw_ids.len()
                {
                    return Err(FitError::Artifact(
                        "ensemble draw shape or identity count differs".to_owned(),
                    ));
                }
                if metadata.parameter_names != manifest.parameter_names {
                    return Err(FitError::Artifact(
                        "ensemble parameter schema differs from the artifact".to_owned(),
                    ));
                }
                let failed_ids = metadata
                    .failures
                    .iter()
                    .map(|(index, _, _)| *index as u64)
                    .collect::<std::collections::HashSet<_>>();
                let replica_count = metadata
                    .draw_ids
                    .len()
                    .checked_add(failed_ids.len())
                    .ok_or_else(|| FitError::Artifact("replica count overflows".to_owned()))?;
                let mut observed_ids = metadata.draw_ids.clone();
                observed_ids.extend(failed_ids.iter().copied());
                observed_ids.sort_unstable();
                if failed_ids.len() != metadata.failures.len()
                    || metadata.draw_ids.windows(2).any(|ids| ids[0] >= ids[1])
                    || observed_ids
                        .iter()
                        .enumerate()
                        .any(|(index, id)| *id != index as u64)
                    || observed_ids.len() != replica_count
                {
                    return Err(FitError::Artifact(
                        "replica identifiers are missing, duplicated, or reordered".to_owned(),
                    ));
                }
                let draws = flat.chunks(shape[1]).map(<[f64]>::to_vec).collect();
                Some(FitEnsembleArtifact {
                    source_id: metadata.source_id,
                    parameter_names: metadata.parameter_names,
                    draws,
                    draw_ids: metadata.draw_ids,
                    bootstrap_seed: metadata.bootstrap_seed,
                    requires_external_datasets: metadata.requires_external_datasets,
                    failures: metadata
                        .failures
                        .into_iter()
                        .map(|(index, seed, message)| FailedReplica {
                            index,
                            seed,
                            message,
                        })
                        .collect(),
                })
            }
            None if ensemble_draws.is_some() => {
                return Err(FitError::Artifact(
                    "ensemble draw payload has no metadata".to_owned(),
                ));
            }
            None => None,
        };
        Ok(Self {
            artifact_kind: manifest.artifact_kind,
            schema_version: manifest.schema_version,
            laddu_version: manifest.laddu_version,
            fingerprint_version: manifest.fingerprint_version,
            likelihood_fingerprint: manifest.likelihood_fingerprint,
            parameter_names: manifest.parameter_names,
            values,
            objective: manifest.objective,
            outcome: if manifest.outcome == "converged" {
                FitOutcome::Converged
            } else if manifest.outcome == "not_converged" {
                FitOutcome::NotConverged
            } else {
                return Err(FitError::Artifact(format!(
                    "unknown terminal outcome `{}`",
                    manifest.outcome
                )));
            },
            terminal_message: manifest.terminal_message,
            diagnostics: FitDiagnostics {
                function_evaluations: manifest.diagnostics[0],
                gradient_evaluations: manifest.diagnostics[1],
                hessian_evaluations: manifest.diagnostics[2],
            },
            covariance,
            standard_errors,
            ensemble,
            strong_compatibility: manifest.strong_compatibility,
        })
    }
}
impl BoundFitState {
    /// Target likelihood's canonical parameter names.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }
    /// Values reordered for the target likelihood.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Preserved objective value.
    pub const fn objective(&self) -> f64 {
        self.objective
    }
    /// Preserved terminal outcome.
    pub const fn outcome(&self) -> FitOutcome {
        self.outcome
    }
    /// Explicit parameter migrations applied while binding.
    pub fn migration(&self) -> &[(String, String)] {
        &self.migration
    }
}

/// Scalar-generic ganesh view of a laddu objective.
#[derive(Clone, Copy, Debug)]
pub struct FitProblem<'a, O: ?Sized, T = f64, B = ganesh::NalgebraProvider> {
    objective: &'a O,
    _numeric: PhantomData<(T, B)>,
}

/// Adam-oriented adapter that draws a deterministic event batch per point.
///
/// ganesh asks for the value and gradient separately. This adapter caches the
/// paired stochastic evaluation so both requests see exactly the same rows.
#[derive(Debug)]
pub struct StochasticFitProblem<'a, O: ?Sized, T = f64, B = ganesh::NalgebraProvider> {
    objective: &'a O,
    fraction: f64,
    next_seed: AtomicU64,
    cache: Mutex<Option<(Vec<f64>, LikelihoodEvaluation)>>,
    _numeric: PhantomData<(T, B)>,
}

impl<'a, O: StochasticObjective + ?Sized, T, B> StochasticFitProblem<'a, O, T, B> {
    /// Create a stochastic adapter that samples `fraction` of events per evaluation.
    ///
    /// `seed` initializes the deterministic sequence of batch seeds.
    ///
    /// # Errors
    ///
    /// Returns [`FitError`] when `fraction` is outside `(0, 1]`.
    pub fn new(objective: &'a O, fraction: f64, seed: u64) -> FitResult<Self> {
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(LikelihoodError::InvalidBatchFraction(fraction).into());
        }
        Ok(Self {
            objective,
            fraction,
            next_seed: AtomicU64::new(seed),
            cache: Mutex::new(None),
            _numeric: PhantomData,
        })
    }

    /// Return the adapted stochastic objective.
    pub const fn objective(&self) -> &'a O {
        self.objective
    }

    /// Return free parameter names in optimizer-vector order.
    pub fn parameter_names(&self) -> Vec<String> {
        FitProblem::<O, T, B>::new(self.objective).parameter_names()
    }

    /// Convert user-facing `f64` values to the backend vector type.
    pub fn vector(&self, values: &[f64]) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).vector(values)
    }

    /// Build the parameter transform for minimizers without native bounds.
    ///
    /// # Errors
    ///
    /// Returns [`FitError`] when parameter scaling or bound metadata cannot be
    /// represented by the optimizer transform.
    pub fn minimizer_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).minimizer_transform()
    }

    /// Build the parameter transform for minimizers that enforce native bounds.
    ///
    /// # Errors
    ///
    /// Returns [`FitError`] when parameter scaling or periodic metadata cannot
    /// be represented by the optimizer transform.
    pub fn native_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).native_transform()
    }

    /// Return native optimizer bounds in transformed coordinates.
    pub fn native_bounds(&self) -> Vec<(T, T)>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        FitProblem::<O, T, B>::new(self.objective).native_bounds()
    }

    fn external(&self, x: &Vector<T, B>) -> FitResult<Vec<f64>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        (0..x.len())
            .map(|index| x.get(index).to_f64().ok_or(FitError::ScalarConversion))
            .collect()
    }

    fn evaluation(&self, x: &Vector<T, B>) -> FitResult<LikelihoodEvaluation>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        let external = self.external(x)?;
        if let Some((_, evaluation)) = self
            .cache
            .lock()
            .as_ref()
            .filter(|(cached, _)| cached == &external)
        {
            return Ok(evaluation.clone());
        }
        let seed = self.next_seed.fetch_add(1, Ordering::Relaxed);
        let evaluation =
            self.objective
                .stochastic_value_gradient(&external, self.fraction, seed)?;
        *self.cache.lock() = Some((external, evaluation.clone()));
        Ok(evaluation)
    }
}

impl<O, T, B> CostFunction<T, B, (), FitError> for StochasticFitProblem<'_, O, T, B>
where
    O: StochasticObjective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> FitResult<T> {
        Ok(T::literal(self.evaluation(x)?.value()))
    }
}

impl<O, T, B> Gradient<T, B, (), FitError> for StochasticFitProblem<'_, O, T, B>
where
    O: StochasticObjective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<Vector<T, B>> {
        Ok(Vector::from_vec(
            self.evaluation(x)?
                .gradient()
                .iter()
                .copied()
                .map(T::literal)
                .collect(),
        ))
    }

    fn evaluate_with_gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<(T, Vector<T, B>)> {
        let evaluation = self.evaluation(x)?;
        Ok((
            T::literal(evaluation.value()),
            Vector::from_vec(
                evaluation
                    .gradient()
                    .iter()
                    .copied()
                    .map(T::literal)
                    .collect(),
            ),
        ))
    }
}

impl<'a, O: Objective + ?Sized, T, B> FitProblem<'a, O, T, B> {
    /// Adapt an objective to ganesh's scalar-generic optimization traits.
    pub const fn new(objective: &'a O) -> Self {
        Self {
            objective,
            _numeric: PhantomData,
        }
    }

    /// Return the adapted objective.
    pub const fn objective(&self) -> &'a O {
        self.objective
    }

    /// Convert user-facing f64 parameter values to this problem's ganesh scalar.
    pub fn vector(&self, values: &[f64]) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Vector::from_vec(values.iter().copied().map(T::literal).collect())
    }

    /// Return free parameter names in optimizer-vector order.
    pub fn parameter_names(&self) -> Vec<String> {
        self.objective
            .parameter_layout()
            .free_parameters()
            .map(|parameter| parameter.name().to_owned())
            .collect()
    }

    /// Build the metadata transform for minimizers without native bounds.
    /// Scaling, periodic wrapping, and smooth bounds are applied automatically.
    ///
    /// # Errors
    ///
    /// Returns [`FitError`] when parameter scaling, bounds, or periodic
    /// metadata cannot form a valid transform.
    pub fn minimizer_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Ok(
            minimizer_transform(self.objective.parameter_layout(), true)?
                .unwrap_or_else(|| Box::new(IdentityTransform)),
        )
    }

    /// Build the metadata transform for algorithms with native bounds, such as
    /// L-BFGS-B. Bounds themselves are returned by [`Self::native_bounds`].
    ///
    /// # Errors
    ///
    /// Returns [`FitError`] when parameter scaling or periodic metadata cannot
    /// form a valid transform.
    pub fn native_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Ok(
            minimizer_transform(self.objective.parameter_layout(), false)?
                .unwrap_or_else(|| Box::new(IdentityTransform)),
        )
    }

    /// Native optimizer bounds in the coordinates produced by
    /// [`Self::native_transform`]. Periodic parameters are deliberately
    /// unbounded in optimizer space.
    pub fn native_bounds(&self) -> Vec<(T, T)>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        self.objective
            .parameter_layout()
            .free_parameters()
            .map(|parameter| {
                if parameter.is_periodic() {
                    (T::literal(f64::NEG_INFINITY), T::infinity())
                } else {
                    let scale = parameter.scale().unwrap_or(1.0);
                    (
                        T::literal(
                            parameter.bounds_spec().min.unwrap_or(f64::NEG_INFINITY) / scale,
                        ),
                        T::literal(parameter.bounds_spec().max.unwrap_or(f64::INFINITY) / scale),
                    )
                }
            })
            .collect()
    }

    /// Build the only metadata transform that is posterior-safe without a
    /// Jacobian correction: linear scaling (whose Jacobian is constant).
    /// Bounds are enforced by [`LogDensity::log_density`] as support, and
    /// periodic values remain in their single canonical domain.
    ///
    /// # Errors
    ///
    /// Returns [`FitError`] when parameter scale metadata cannot form a valid
    /// linear transform.
    pub fn sampler_transform(&self) -> FitResult<Box<dyn Transform<T, B>>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Ok(scale_transform(self.objective.parameter_layout())?
            .unwrap_or_else(|| Box::new(IdentityTransform)))
    }

    fn external(&self, x: &Vector<T, B>) -> FitResult<Vec<f64>>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        (0..x.len())
            .map(|index| x.get(index).to_f64().ok_or(FitError::ScalarConversion))
            .collect()
    }
}

impl<O, T, B> CostFunction<T, B, (), FitError> for FitProblem<'_, O, T, B>
where
    O: Objective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> FitResult<T> {
        Ok(T::literal(self.objective.value(&self.external(x)?)?))
    }
}

impl<O, T, B> Gradient<T, B, (), FitError> for FitProblem<'_, O, T, B>
where
    O: Objective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<Vector<T, B>> {
        let evaluation = self.objective.value_gradient(&self.external(x)?)?;
        Ok(Vector::from_vec(
            evaluation
                .gradient()
                .iter()
                .copied()
                .map(T::literal)
                .collect(),
        ))
    }

    fn evaluate_with_gradient(&self, x: &Vector<T, B>, _args: &()) -> FitResult<(T, Vector<T, B>)> {
        let evaluation = self.objective.value_gradient(&self.external(x)?)?;
        Ok((
            T::literal(evaluation.value()),
            Vector::from_vec(
                evaluation
                    .gradient()
                    .iter()
                    .copied()
                    .map(T::literal)
                    .collect(),
            ),
        ))
    }
}

impl<O, T, B> LogDensity<T, B, (), FitError> for FitProblem<'_, O, T, B>
where
    O: Objective + ?Sized,
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn log_density(&self, x: &Vector<T, B>, _args: &()) -> FitResult<T> {
        let external = self.external(x)?;
        if self
            .objective
            .parameter_layout()
            .validate_free_values(&external)
            .is_err()
        {
            return Ok(-T::infinity());
        }
        Ok(-T::literal(self.objective.value(&external)?))
    }
}

fn append_transform<T, B, X>(
    current: Option<Box<dyn Transform<T, B>>>,
    next: X,
) -> Option<Box<dyn Transform<T, B>>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    X: Transform<T, B> + 'static,
{
    Some(match current {
        Some(current) => Box::new(current.then(next)),
        None => Box::new(next),
    })
}

fn scale_transform<T, B>(layout: &ParamLayout) -> FitResult<Option<Box<dyn Transform<T, B>>>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    let parameters = layout.free_parameters().collect::<Vec<_>>();
    if parameters
        .iter()
        .all(|parameter| parameter.scale().is_none())
    {
        return Ok(None);
    }
    let scales = parameters
        .into_iter()
        .map(|parameter| T::literal(parameter.scale().unwrap_or(1.0)));
    Ok(Some(Box::new(
        ScaleTransform::<T, B>::from_parameter_scales(scales)?,
    )))
}

fn minimizer_transform<T, B>(
    layout: &ParamLayout,
    include_bounds: bool,
) -> FitResult<Option<Box<dyn Transform<T, B>>>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    let parameters = layout.free_parameters().collect::<Vec<_>>();
    let mut transform = scale_transform(layout)?;

    if parameters.iter().any(|parameter| parameter.is_periodic()) {
        let intervals = parameters.iter().map(|parameter| {
            parameter
                .periodic_bounds()
                .map(|(min, max)| (T::literal(min), T::literal(max)))
        });
        transform = append_transform(transform, PeriodicTransform::<T, B>::new(intervals)?);
    }

    if include_bounds
        && parameters.iter().any(|parameter| {
            !parameter.is_periodic()
                && (parameter.bounds_spec().min.is_some() || parameter.bounds_spec().max.is_some())
        })
    {
        let bounds = parameters.iter().map(|parameter| {
            if parameter.is_periodic() {
                (T::literal(f64::NEG_INFINITY), T::infinity())
            } else {
                (
                    T::literal(parameter.bounds_spec().min.unwrap_or(f64::NEG_INFINITY)),
                    T::literal(parameter.bounds_spec().max.unwrap_or(f64::INFINITY)),
                )
            }
        });
        transform = append_transform(transform, GaneshBounds::<T, B>::new(bounds)?);
    }
    Ok(transform)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ganesh::{
        algorithms::{
            gradient::{Adam, AdamConfig, ConjugateGradientConfig, LBFGSB, LBFGSBConfig},
            mcmc::{AIES, AIESConfig, AIESInit, ESS, ESSConfig, ESSInit},
        },
        core::{Callbacks, MaxSteps},
        traits::{Algorithm, SupportsParameterNames},
    };
    use laddu_expr::parameters::Parameter;
    use laddu_likelihood::{LikelihoodEvaluation, LikelihoodResult};
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug)]
    struct Quadratic {
        layout: ParamLayout,
    }

    impl Objective for Quadratic {
        fn parameter_layout(&self) -> &ParamLayout {
            &self.layout
        }

        fn value(&self, parameters: &[f64]) -> LikelihoodResult<f64> {
            Ok(parameters.iter().map(|value| value * value).sum())
        }

        fn value_gradient(&self, parameters: &[f64]) -> LikelihoodResult<LikelihoodEvaluation> {
            let value = self.value(parameters)?;
            let gradient = parameters.iter().map(|value| 2.0 * value).collect();
            Ok(LikelihoodEvaluation::new(value, gradient))
        }
    }

    #[test]
    fn adapter_is_native_in_both_precisions() {
        let objective = Quadratic {
            layout: ParamLayout::new([Parameter::free("x").with_initial(2.0)]).unwrap(),
        };
        let f64_problem = FitProblem::<_, f64>::new(&objective);
        let f32_problem = FitProblem::<_, f32>::new(&objective);
        assert_eq!(
            f64_problem
                .evaluate(&f64_problem.vector(&[2.0]), &())
                .unwrap(),
            4.0
        );
        assert_eq!(
            f32_problem
                .evaluate(&f32_problem.vector(&[2.0]), &())
                .unwrap(),
            4.0
        );
    }

    #[test]
    fn ensemble_samplers_are_native_in_both_precisions() {
        let objective = Quadratic {
            layout: ParamLayout::new([Parameter::free("x").with_initial(0.0)]).unwrap(),
        };

        let f32_problem = FitProblem::<_, f32>::new(&objective);
        let f32_walkers = [-0.3_f32, -0.1, 0.1, 0.3]
            .into_iter()
            .map(|value| Vector::from_vec(vec![value]))
            .collect::<Vec<_>>();
        let aies_config = AIESConfig::<f32>::default()
            .with_parameter_names(f32_problem.parameter_names())
            .with_transform(f32_problem.sampler_transform().unwrap());
        let aies = AIES::<f32>::new(Some(7))
            .process(
                &f32_problem,
                &(),
                AIESInit::new(f32_walkers.clone()).unwrap(),
                aies_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(aies.dimension, (4, 5, 1));

        let ess_config = ESSConfig::<f32>::default()
            .with_parameter_names(f32_problem.parameter_names())
            .with_transform(f32_problem.sampler_transform().unwrap());
        let ess = ESS::<f32>::new(Some(11))
            .process(
                &f32_problem,
                &(),
                ESSInit::new(f32_walkers).unwrap(),
                ess_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(ess.dimension, (4, 5, 1));

        let f64_problem = FitProblem::<_, f64>::new(&objective);
        let f64_walkers = [-0.3_f64, -0.1, 0.1, 0.3]
            .into_iter()
            .map(|value| Vector::from_vec(vec![value]))
            .collect::<Vec<_>>();
        let aies_config = AIESConfig::<f64>::default()
            .with_parameter_names(f64_problem.parameter_names())
            .with_transform(f64_problem.sampler_transform().unwrap());
        let aies = AIES::<f64>::new(Some(13))
            .process(
                &f64_problem,
                &(),
                AIESInit::new(f64_walkers.clone()).unwrap(),
                aies_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(aies.dimension, (4, 5, 1));

        let ess_config = ESSConfig::<f64>::default()
            .with_parameter_names(f64_problem.parameter_names())
            .with_transform(f64_problem.sampler_transform().unwrap());
        let ess = ESS::<f64>::new(Some(17))
            .process(
                &f64_problem,
                &(),
                ESSInit::new(f64_walkers).unwrap(),
                ess_config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert_eq!(ess.dimension, (4, 5, 1));
    }

    #[test]
    fn lbfgsb_accepts_mixed_bounded_and_periodic_parameters() {
        let objective = Quadratic {
            layout: ParamLayout::new([
                Parameter::free("magnitude")
                    .with_initial(0.5)
                    .with_scale(0.25)
                    .with_bounds(0.0, 2.0),
                Parameter::free("phase")
                    .with_initial(0.0)
                    .with_bounds(-std::f64::consts::PI, std::f64::consts::PI)
                    .with_periodic(),
            ])
            .unwrap(),
        };
        let problem = FitProblem::<_, f64>::new(&objective);

        let config = LBFGSBConfig::<f64>::default()
            .with_parameter_names(problem.parameter_names())
            .with_transform(problem.native_transform().unwrap())
            .unwrap()
            .with_bounds(problem.native_bounds())
            .unwrap();
        let result = LBFGSB::<f64>::default()
            .process(
                &problem,
                &(),
                problem.vector(&[0.5, 0.0]),
                config,
                LBFGSB::<f64>::default_callbacks().with_terminator(MaxSteps(20)),
            )
            .unwrap();
        assert!((0.0..=2.0).contains(&result.x.get(0)));
        assert!(result.x.get(1).abs() <= std::f64::consts::PI);
    }

    #[test]
    fn typed_configuration_preserves_custom_line_searches() {
        use ganesh::algorithms::line_search::HagerZhangLineSearch;

        let objective = Quadratic {
            layout: ParamLayout::new([Parameter::free("x").with_initial(1.0)]).unwrap(),
        };
        let problem = FitProblem::<_, f64>::new(&objective);
        let _config = ConjugateGradientConfig::<f64>::default()
            .with_line_search(HagerZhangLineSearch::<f64>::default())
            .with_parameter_names(problem.parameter_names())
            .with_transform(problem.minimizer_transform().unwrap());
    }

    #[derive(Debug)]
    struct StochasticQuadratic {
        inner: Quadratic,
        calls: AtomicUsize,
    }

    impl Objective for StochasticQuadratic {
        fn parameter_layout(&self) -> &ParamLayout {
            self.inner.parameter_layout()
        }

        fn value(&self, parameters: &[f64]) -> LikelihoodResult<f64> {
            self.inner.value(parameters)
        }

        fn value_gradient(&self, parameters: &[f64]) -> LikelihoodResult<LikelihoodEvaluation> {
            self.inner.value_gradient(parameters)
        }
    }

    impl StochasticObjective for StochasticQuadratic {
        fn stochastic_value_gradient(
            &self,
            parameters: &[f64],
            _fraction: f64,
            _seed: u64,
        ) -> LikelihoodResult<LikelihoodEvaluation> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.value_gradient(parameters)
        }
    }

    #[test]
    fn stochastic_adapter_pairs_value_and_gradient_on_one_batch() {
        let objective = StochasticQuadratic {
            inner: Quadratic {
                layout: ParamLayout::new([Parameter::free("x").with_initial(2.0)]).unwrap(),
            },
            calls: AtomicUsize::new(0),
        };
        let problem = StochasticFitProblem::<_, f64>::new(&objective, 0.5, 7).unwrap();
        let initial = problem.vector(&[2.0]);
        assert_eq!(problem.evaluate(&initial, &()).unwrap(), 4.0);
        assert_eq!(problem.gradient(&initial, &()).unwrap().get(0), 4.0);
        assert_eq!(objective.calls.load(Ordering::Relaxed), 1);

        let config = AdamConfig::<f64>::default()
            .with_parameter_names(problem.parameter_names())
            .with_transform(problem.minimizer_transform().unwrap());
        let result = Adam::<f64>::default()
            .process(
                &problem,
                &(),
                initial,
                config,
                Adam::<f64>::default_callbacks().with_terminator(MaxSteps(4)),
            )
            .unwrap();
        assert!(result.fx.is_finite());
    }

    #[test]
    fn minimization_result_distinguishes_outcomes_and_optional_inference() {
        use ganesh::core::{EvalCounts, Matrix};
        use ganesh::traits::StatusMessage;

        let converged = MinimizationResult::from_ganesh(
            MinimizationSummary {
                bounds: None,
                parameter_names: Some(vec!["x".to_owned()]),
                message: StatusMessage::default().set_success_with_message("done"),
                x0: Vector::from_vec(vec![2.0]),
                x: Vector::from_vec(vec![1.0]),
                std: Vector::from_vec(vec![0.5]),
                fx: 3.0,
                evals: EvalCounts::default(),
                covariance: Matrix::from_vec(1, 1, vec![0.25]),
            },
            &[],
        );
        assert_eq!(converged.parameter_names(), ["x"]);
        assert_eq!(converged.values(), [1.0]);
        assert_eq!(converged.outcome(), FitOutcome::Converged);
        assert!(converged.converged());
        assert_eq!(converged.covariance(), Some(&[vec![0.25]][..]));
        assert_eq!(converged.standard_errors(), Some(&[0.5][..]));
        assert_eq!(converged.raw_ganesh_summary().fx, 3.0);

        let stopped = MinimizationResult::from_ganesh(
            MinimizationSummary {
                bounds: None,
                parameter_names: None,
                message: StatusMessage::default().set_custom("step limit"),
                x0: Vector::from_vec(vec![2.0]),
                x: Vector::from_vec(vec![1.5]),
                std: Vector::from_vec(vec![f64::NAN]),
                fx: 4.0,
                evals: EvalCounts::default(),
                covariance: Matrix::identity(1),
            },
            &["fallback".to_owned()],
        );
        assert_eq!(stopped.parameter_names(), ["fallback"]);
        assert_eq!(stopped.outcome(), FitOutcome::NotConverged);
        assert!(!stopped.converged());
        assert!(stopped.covariance().is_none());
        assert!(stopped.standard_errors().is_none());
    }
}
