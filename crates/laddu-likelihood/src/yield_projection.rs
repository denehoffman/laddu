//! Binned raw-yield projections and their support diagnostics.

use std::collections::{HashMap, HashSet};

use laddu_data::data::Dataset;
use laddu_expr::ExprNodeStructuralKey;
use laddu_runtime::{Execution, FinalUpperEdge, checked_bin_count};

use crate::{
    Axis, BinnedEstimate, BinnedEstimateUnit, ErrorBudget, ErrorComponent, LikelihoodError,
    LikelihoodResult, Yield,
};

fn invalid(message: impl Into<String>) -> LikelihoodError {
    LikelihoodError::InvalidCrossSection(message.into())
}

/// Named binning request for raw yields or fitted intensity.
#[derive(Clone, Debug)]
pub struct Projection {
    name: String,
    axes: Vec<Axis>,
}

impl Projection {
    /// Constructs a named projection specification.
    ///
    /// # Errors
    /// Returns an error when the name or axis group is empty.
    pub fn new(name: impl Into<String>, axes: Vec<Axis>) -> LikelihoodResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(invalid("projection names must not be empty"));
        }
        if axes.is_empty() {
            return Err(invalid("each projection must contain at least one axis"));
        }
        let bin_axes: Vec<_> = axes.iter().map(|axis| axis.binning.clone()).collect();
        if checked_bin_count(&bin_axes).is_none() {
            return Err(invalid(
                "projection axis shape exceeds addressable bin count",
            ));
        }
        Ok(Self { name, axes })
    }

    /// Public projection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Axes forming this entry's joint differential cross section.
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
}

/// Support and numerical validity of a projected bin.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum YieldBinValidity {
    /// The bin has sufficient finite model support; zero selected yield is valid.
    Valid,
    /// No generated Monte Carlo event belongs to this bin.
    MissingGeneratedSupport,
    /// No accepted Monte Carlo event belongs to this bin.
    MissingAcceptedSupport,
    /// Accepted model support is zero or negative.
    NonPositiveAcceptedSupport,
    /// Generated model support is zero or negative.
    NonPositiveGeneratedSupport,
    /// Generated integration weights do not define positive local exposure.
    InvalidExposure,
    /// A weighted sum or model intensity was nonfinite.
    NonFiniteEvaluation,
}

fn fitted_bin_validity(
    accepted: f64,
    generated: f64,
    accepted_count: usize,
    generated_count: usize,
    generated_exposure: f64,
) -> YieldBinValidity {
    if !accepted.is_finite() || !generated.is_finite() {
        YieldBinValidity::NonFiniteEvaluation
    } else if generated_count == 0 {
        YieldBinValidity::MissingGeneratedSupport
    } else if !generated_exposure.is_finite() || generated_exposure <= 0.0 {
        YieldBinValidity::InvalidExposure
    } else if accepted_count == 0 {
        YieldBinValidity::MissingAcceptedSupport
    } else if accepted <= 0.0 {
        YieldBinValidity::NonPositiveAcceptedSupport
    } else if generated <= 0.0 {
        YieldBinValidity::NonPositiveGeneratedSupport
    } else {
        YieldBinValidity::Valid
    }
}

fn absolute_rate_bins(values: &[f64], absolute: bool) -> Vec<f64> {
    if absolute {
        values.to_vec()
    } else {
        vec![f64::NAN; values.len()]
    }
}

/// Counts of coordinates excluded from a central yield projection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct YieldProjectionDiagnostics {
    /// Selected-data rows with nonfinite coordinates.
    pub selected_nonfinite: usize,
    /// Selected-data rows outside the joint axes.
    pub selected_out_of_range: usize,
    /// Accepted Monte Carlo rows with nonfinite coordinates.
    pub accepted_nonfinite: usize,
    /// Accepted Monte Carlo rows outside the joint axes.
    pub accepted_out_of_range: usize,
    /// Generated Monte Carlo rows with nonfinite coordinates.
    pub generated_nonfinite: usize,
    /// Generated Monte Carlo rows outside the joint axes.
    pub generated_out_of_range: usize,
}

/// Central row-major histogram values with their ordered geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct YieldHistogramView {
    pub(crate) axes: Vec<Vec<f64>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) errors: Vec<f64>,
    pub(crate) budget: ErrorBudget,
    pub(crate) included: Vec<ErrorComponent>,
    pub(crate) omitted: Vec<(ErrorComponent, &'static str)>,
    pub(crate) sources: Vec<(ErrorComponent, u64)>,
}

impl YieldHistogramView {
    /// Ordered bin edges.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }
    /// Bin count for each axis, with the last axis varying fastest.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Central row-major bin values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Marginal standard errors for the selected budget.
    pub fn errors(&self) -> &[f64] {
        &self.errors
    }
    /// Exact selection used for these marginal errors.
    pub fn budget(&self) -> ErrorBudget {
        self.budget
    }
    /// Sources contributing to the materialized variance.
    pub fn included(&self) -> &[ErrorComponent] {
        &self.included
    }
    /// Selected sources that were unavailable, with machine-readable reasons.
    pub fn omitted(&self) -> &[(ErrorComponent, &'static str)] {
        &self.omitted
    }
    /// Source identity of included uncertainty constituents.
    pub fn sources(&self) -> &[(ErrorComponent, u64)] {
        &self.sources
    }
}

/// Coherent selected and fitted yields with paired ensemble draws over joint axes.
#[derive(Clone, Debug)]
pub struct YieldProjection {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    bin_volumes: Vec<f64>,
    data_source_id: u64,
    accepted_source_id: u64,
    generated_source_id: u64,
    selected: Vec<f64>,
    selected_fill_variance: Vec<f64>,
    accepted: Vec<f64>,
    accepted_fill_variance: Vec<f64>,
    generated: Vec<f64>,
    generated_fill_variance: Vec<f64>,
    validity: Vec<YieldBinValidity>,
    diagnostics: YieldProjectionDiagnostics,
    has_absolute_rate: bool,
    components: HashMap<String, ComponentYieldProjection>,
    source_id: Option<u64>,
    has_replica_datasets: bool,
    selected_draws: Vec<Vec<f64>>,
    accepted_draws: Vec<Vec<f64>>,
    generated_draws: Vec<Vec<f64>>,
}

/// Differential cross sections converted from one yield projection.
#[derive(Clone, Debug)]
pub struct ComponentYieldProjection {
    tags: Vec<String>,
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    bin_volumes: Vec<f64>,
    accepted_source_id: u64,
    generated_source_id: u64,
    accepted: Vec<f64>,
    accepted_fill_variance: Vec<f64>,
    generated: Vec<f64>,
    generated_fill_variance: Vec<f64>,
    validity: Vec<YieldBinValidity>,
    accepted_draws: Vec<Vec<f64>>,
    generated_draws: Vec<Vec<f64>>,
    source_id: Option<u64>,
}

impl ComponentYieldProjection {
    pub(crate) fn bin_volumes(&self) -> &[f64] {
        &self.bin_volumes
    }
    /// Accepted-space fitted yield with paired draws.
    pub fn accepted_estimate(&self) -> BinnedEstimate {
        BinnedEstimate::projected(
            self.accepted.clone(),
            self.accepted_draws.clone(),
            self.source_id,
            self.axes.clone(),
            BinnedEstimateUnit::Yield,
        )
        .with_variance(
            ErrorComponent::AcceptedMcFill,
            self.accepted_fill_variance.clone(),
        )
        .with_variance_source(ErrorComponent::AcceptedMcFill, self.accepted_source_id)
    }
    /// Generated-space fitted yield with paired draws.
    pub fn generated_estimate(&self) -> BinnedEstimate {
        BinnedEstimate::projected(
            self.generated.clone(),
            self.generated_draws.clone(),
            self.source_id,
            self.axes.clone(),
            BinnedEstimateUnit::Yield,
        )
        .with_variance(
            ErrorComponent::GeneratedMcFill,
            self.generated_fill_variance.clone(),
        )
        .with_variance_source(ErrorComponent::GeneratedMcFill, self.generated_source_id)
    }
    /// Canonical sorted tags defining this coherent model selection.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }
    /// Ordered bin edges shared with the parent projection.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }
    /// Row-major shape shared with the parent projection.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Accepted-space fitted model yield.
    pub fn accepted(&self) -> &[f64] {
        &self.accepted
    }
    /// Generated-space fitted model yield.
    pub fn generated(&self) -> &[f64] {
        &self.generated
    }
    /// Accepted-space values in paired ensemble draw order.
    pub fn accepted_draws(&self) -> &[Vec<f64>] {
        &self.accepted_draws
    }
    /// Generated-space values in paired ensemble draw order.
    pub fn generated_draws(&self) -> &[Vec<f64>] {
        &self.generated_draws
    }
    /// Per-bin model support validity.
    pub fn validity(&self) -> &[YieldBinValidity] {
        &self.validity
    }
    /// Materialize the accepted-space central histogram.
    ///
    /// # Panics
    /// Panics only if an internally constructed projection loses its axes.
    pub fn accepted_histogram(&self) -> YieldHistogramView {
        self.accepted_estimate()
            .histogram_with_budget(ErrorBudget::default())
            .expect("projected estimate has axes")
    }
    /// Materialize the generated-space central histogram.
    ///
    /// # Panics
    /// Panics only if an internally constructed projection loses its axes.
    pub fn generated_histogram(&self) -> YieldHistogramView {
        self.generated_estimate()
            .histogram_with_budget(ErrorBudget::default())
            .expect("projected estimate has axes")
    }
}

impl YieldProjection {
    pub(crate) fn bin_volumes(&self) -> &[f64] {
        &self.bin_volumes
    }
    /// Selected-data yield with paired draws.
    pub fn selected_estimate(&self) -> BinnedEstimate {
        BinnedEstimate::projected(
            self.selected.clone(),
            self.selected_draws.clone(),
            self.source_id,
            self.axes.clone(),
            BinnedEstimateUnit::Yield,
        )
        .with_variance(
            ErrorComponent::DataFill,
            self.selected_fill_variance.clone(),
        )
        .with_variance_source(ErrorComponent::DataFill, self.data_source_id)
        .with_paired_bootstrap(self.has_replica_datasets)
    }
    /// Accepted-space fitted yield with paired draws.
    pub fn accepted_estimate(&self) -> BinnedEstimate {
        BinnedEstimate::projected(
            self.accepted.clone(),
            self.accepted_draws.clone(),
            self.source_id,
            self.axes.clone(),
            BinnedEstimateUnit::Yield,
        )
        .with_variance(
            ErrorComponent::AcceptedMcFill,
            self.accepted_fill_variance.clone(),
        )
        .with_variance_source(ErrorComponent::AcceptedMcFill, self.accepted_source_id)
        .with_paired_bootstrap(self.has_replica_datasets)
    }
    /// Generated-space fitted yield with paired draws.
    pub fn generated_estimate(&self) -> BinnedEstimate {
        BinnedEstimate::projected(
            self.generated.clone(),
            self.generated_draws.clone(),
            self.source_id,
            self.axes.clone(),
            BinnedEstimateUnit::Yield,
        )
        .with_variance(
            ErrorComponent::GeneratedMcFill,
            self.generated_fill_variance.clone(),
        )
        .with_variance_source(ErrorComponent::GeneratedMcFill, self.generated_source_id)
        .with_paired_bootstrap(self.has_replica_datasets)
    }
    /// Ensemble source shared by every draw constituent, if present.
    pub fn source_id(&self) -> Option<u64> {
        self.source_id
    }
    /// Whether each parameter draw has a matching resampled likelihood dataset.
    pub fn has_replica_datasets(&self) -> bool {
        self.has_replica_datasets
    }
    /// Selected-data values in paired ensemble draw order.
    pub fn selected_draws(&self) -> &[Vec<f64>] {
        &self.selected_draws
    }
    /// Accepted-model values in paired ensemble draw order.
    pub fn accepted_draws(&self) -> &[Vec<f64>] {
        &self.accepted_draws
    }
    /// Generated-model values in paired ensemble draw order.
    pub fn generated_draws(&self) -> &[Vec<f64>] {
        &self.generated_draws
    }
    /// Ordered bin edges.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }
    /// Row-major shape; the last axis varies fastest.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Selected data yield, D, including signed event weights.
    pub fn selected(&self) -> &[f64] {
        &self.selected
    }
    /// Accepted fitted yield, A, or NaN for a shape-only term.
    pub fn accepted(&self) -> &[f64] {
        &self.accepted
    }
    /// Generated fitted yield, G, or NaN for a shape-only term.
    pub fn generated(&self) -> &[f64] {
        &self.generated
    }
    /// Per-bin support validity.
    pub fn validity(&self) -> &[YieldBinValidity] {
        &self.validity
    }
    /// Coordinate exclusion diagnostics for each sample.
    pub fn diagnostics(&self) -> &YieldProjectionDiagnostics {
        &self.diagnostics
    }
    /// Whether the source likelihood defines absolute fitted yields.
    pub fn has_absolute_rate(&self) -> bool {
        self.has_absolute_rate
    }
    /// Named model-only component projections. Observed data belong only to
    /// the full selection and are never assigned to components.
    pub fn components(&self) -> &HashMap<String, ComponentYieldProjection> {
        &self.components
    }
    /// Materialize a central selected-data histogram view.
    ///
    /// # Panics
    /// Panics only if an internally constructed projection loses its axes.
    pub fn selected_histogram(&self) -> YieldHistogramView {
        self.selected_estimate()
            .histogram_with_budget(ErrorBudget::default())
            .expect("projected estimate has axes")
    }
    /// Materialize a central accepted-model histogram view.
    ///
    /// # Panics
    /// Panics only if an internally constructed projection loses its axes.
    pub fn accepted_histogram(&self) -> YieldHistogramView {
        self.accepted_estimate()
            .histogram_with_budget(ErrorBudget::default())
            .expect("projected estimate has axes")
    }
    /// Materialize a central generated-model histogram view.
    ///
    /// # Panics
    /// Panics only if an internally constructed projection loses its axes.
    pub fn generated_histogram(&self) -> YieldHistogramView {
        self.generated_estimate()
            .histogram_with_budget(ErrorBudget::default())
            .expect("projected estimate has axes")
    }
}

/// Named central yield projections in request order.
#[derive(Clone, Debug)]
pub struct YieldProjectionSet {
    entries: Vec<(String, YieldProjection)>,
}

impl YieldProjectionSet {
    pub(crate) fn into_entries(self) -> Vec<(String, YieldProjection)> {
        self.entries
    }

    /// Number of projections.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether the request produced no projections.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Lookup by request name.
    pub fn get(&self, name: &str) -> Option<&YieldProjection> {
        self.entries
            .iter()
            .find_map(|(candidate, result)| (candidate == name).then_some(result))
    }
    /// Iterate in request order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &YieldProjection)> {
        self.entries
            .iter()
            .map(|(name, result)| (name.as_str(), result))
    }
}

impl Yield {
    /// Evaluate one central joint yield projection on demand.
    ///
    /// # Errors
    /// Returns an error for invalid geometry, coordinates, or model evaluation.
    pub fn projection(&self, axes: &[Axis]) -> LikelihoodResult<YieldProjection> {
        let request = Projection::new("yield", axes.to_vec())?;
        Ok(self.projection_set(&[request])?.entries.remove(0).1)
    }

    /// Evaluate one central projection with named coherent model selections.
    /// Components have no observed-data constituent and need not be additive.
    ///
    /// # Errors
    /// Returns an error for invalid geometry, selections, or model evaluation.
    pub fn projection_with_components(
        &self,
        axes: &[Axis],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<YieldProjection> {
        let request = Projection::new("yield", axes.to_vec())?;
        Ok(self
            .projection_set_with_components(&[request], components)?
            .entries
            .remove(0)
            .1)
    }

    /// Evaluate independent named central yield projections in request order.
    /// Identical axis groups share bin assignments and intensity traversal.
    ///
    /// # Errors
    /// Returns an error before event traversal for an empty request, repeated
    /// names, or invalid bin volumes. Failed model evaluation is atomic.
    pub fn projection_set(
        &self,
        projections: &[Projection],
    ) -> LikelihoodResult<YieldProjectionSet> {
        self.projection_set_with_components(projections, &HashMap::new())
    }

    /// Evaluate named central projections with named coherent tag selections.
    /// Aliases with identical tag sets share one model evaluation.
    ///
    /// # Errors
    /// Returns an error for invalid requests, selections, or model evaluation.
    ///
    /// # Panics
    /// Panics only if a prepared component index becomes inconsistent internally.
    pub fn projection_set_with_components(
        &self,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<YieldProjectionSet> {
        if projections.is_empty() {
            return Err(invalid("at least one projection is required"));
        }
        let mut names = HashSet::new();
        for request in projections {
            if !names.insert(request.name()) {
                return Err(invalid(format!(
                    "duplicate projection name: {}",
                    request.name()
                )));
            }
            if !valid_bin_volumes(request.axes()) {
                return Err(invalid(format!(
                    "projection `{}` has invalid bin volume",
                    request.name()
                )));
            }
        }
        let execution = self.likelihood().execution();
        let aliases = components
            .iter()
            .map(|(name, tags)| {
                if name.trim().is_empty()
                    || tags.is_empty()
                    || tags.iter().any(|tag| tag.trim().is_empty())
                {
                    return Err(invalid(format!("invalid component name or tags: `{name}`")));
                }
                let tags = CanonicalTags::new(tags);
                for tag in tags.as_slice() {
                    if !self
                        .likelihood()
                        .intensity_model_has_tag(self.term_name(), tag)?
                    {
                        return Err(invalid(format!(
                            "component `{name}` has unknown model tag `{tag}`"
                        )));
                    }
                }
                Ok((name.clone(), tags))
            })
            .collect::<LikelihoodResult<HashMap<_, _>>>()?;
        let (unique, indexes) = deduplicate_projections(projections);
        let coordinates = ProjectionCoordinates::prepare(&unique, execution)?;
        let mut plans = unique
            .iter()
            .map(|request| PreparedProjection::new(request))
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let (_, accepted) = self.likelihood().intensity_datasets(self.term_name())?;
        let generated = self.generated_mc();
        let ensemble = self.ensemble();
        let shared_mc = !execution.is_distributed()
            && ensemble.is_none_or(|ensemble| {
                ensemble.replicas().iter().all(|replica| {
                    let other = replica.execution();
                    replica
                        .intensity_datasets(self.term_name())
                        .is_ok_and(|(_, mc)| mc.identity() == accepted.identity())
                        && other.requested_device() == execution.requested_device()
                        && other.precision() == execution.precision()
                        && other.jit_policy() == execution.jit_policy()
                        && other.autodiff_mode() == execution.autodiff_mode()
                        && other.normalization_mode() == execution.normalization_mode()
                        && other.thread_policy() == execution.thread_policy()
                        && other.partitioning() == execution.partitioning()
                        && !other.is_distributed()
                })
            });
        let draws = if shared_mc {
            ensemble.map_or(0, |ensemble| ensemble.draws().len())
        } else {
            0
        };
        let parameters = std::iter::once(self.parameters())
            .chain(
                ensemble
                    .into_iter()
                    .filter(|_| shared_mc)
                    .flat_map(|ensemble| ensemble.draws().iter().map(Vec::as_slice)),
            )
            .collect::<Vec<_>>();
        let bins = plans
            .iter()
            .try_fold(0usize, |sum, plan| sum.checked_add(plan.volumes.len()))
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let output_bins = projections
            .iter()
            .try_fold(0usize, |sum, request| {
                let count = checked_bin_count(
                    &request
                        .axes()
                        .iter()
                        .map(|axis| axis.binning.clone())
                        .collect::<Vec<_>>(),
                )?;
                sum.checked_add(count)
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        // Only bins and parameter vectors survive a source batch. Include final
        // alias copies and the temporary unique results during materialization.
        let retained_draws = ensemble.map_or(0, |ensemble| ensemble.draws().len());
        let per_bin = aliases
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(3))
            .and_then(|n| n.checked_mul(retained_draws + 1))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(256 + aliases.len() * 96))
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let workspace = bins
            .checked_add(output_bins)
            .and_then(|bins| bins.checked_mul(per_bin))
            .and_then(|bytes| {
                bytes.checked_add(
                    parameters
                        .len()
                        .checked_mul(aliases.len() + 1)?
                        .checked_mul(self.likelihood().params().len())?
                        .checked_mul(16)?,
                )
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let _workspace = execution
            .host_memory()
            .reserve(
                u64::try_from(workspace)
                    .map_err(|_| invalid("projection workspace size overflow"))?,
            )
            .map_err(laddu_runtime::RuntimeError::from)?;
        let selections = aliases.values().cloned().collect::<HashSet<_>>();
        let mut models = std::iter::once(None)
            .chain(selections.into_iter().map(Some))
            .map(|tags| {
                let evaluator = self.likelihood().intensity_evaluator(
                    self.term_name(),
                    tags.as_ref().map(CanonicalTags::as_slice),
                )?;
                let local = evaluator.parameters(&parameters)?;
                Ok(ProjectionModel {
                    tags,
                    evaluator,
                    parameters: local,
                    accepted: ModelBins::new(&plans, parameters.len()),
                    generated: ModelBins::new(&plans, parameters.len()),
                })
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        coordinates.visit(
            self.observed_data(),
            execution,
            (0, 0),
            |batch, assignments| {
                let weights = (0..batch.len())
                    .map(|row| batch.weights_at(row))
                    .collect::<Vec<_>>();
                for (plan, bins) in plans.iter_mut().zip(assignments) {
                    plan.data_bins.record(bins, &weights);
                }
                Ok(())
            },
        )?;
        let selected_draws = if draws == 0 {
            vec![Vec::new(); plans.len()]
        } else {
            let mut selected = plans
                .iter()
                .map(|_| Vec::with_capacity(draws))
                .collect::<Vec<_>>();
            let ensemble = ensemble.expect("draws require an ensemble");
            for draw in 0..draws {
                if ensemble.replicas().is_empty() {
                    for (plan, selected) in plans.iter().zip(&mut selected) {
                        selected.push(plan.data_bins.weights.clone());
                    }
                    continue;
                }
                let (data, _) = ensemble.replicas()[draw].intensity_datasets(self.term_name())?;
                let mut sums = plans
                    .iter()
                    .map(|plan| vec![0.0; plan.volumes.len()])
                    .collect::<Vec<_>>();
                coordinates.visit(data, execution, (0, 0), |batch, assignments| {
                    let weights = (0..batch.len())
                        .map(|row| batch.weights_at(row))
                        .collect::<Vec<_>>();
                    for (bins, sums) in assignments.iter().zip(&mut sums) {
                        for (row, index) in bins.indices.iter().enumerate() {
                            if let Some(index) = index {
                                sums[*index] += weights[row];
                            }
                        }
                    }
                    Ok(())
                })?;
                for (selected, sums) in selected.iter_mut().zip(sums) {
                    selected.push(sums);
                }
            }
            selected
        };
        let model_memory = models.iter().fold((0, 0), |(fixed, event), model| {
            let zero = model.evaluator.plan.batch_memory_estimate(0);
            (
                fixed.max(zero),
                event.max(
                    model
                        .evaluator
                        .plan
                        .batch_memory_estimate(1)
                        .saturating_sub(zero),
                ),
            )
        });
        for (generated_space, source) in [(false, accepted), (true, generated)] {
            coordinates.visit(source, execution, model_memory, |batch, assignments| {
                let weights = (0..batch.len())
                    .map(|row| batch.weights_at(row))
                    .collect::<Vec<_>>();
                for (plan, bins) in plans.iter_mut().zip(assignments) {
                    let summary = if generated_space {
                        &mut plan.generated_bins
                    } else {
                        &mut plan.accepted_bins
                    };
                    summary.record(bins, &weights);
                }
                for model in &mut models {
                    let bins = if generated_space {
                        &mut model.generated
                    } else {
                        &mut model.accepted
                    };
                    model.evaluator.plan.visit_batch_many(
                        execution,
                        &model.parameters,
                        batch,
                        |parameter, values| {
                            let real = values.iter().map(|value| value.re).collect::<Vec<_>>();
                            for (index, assignment) in assignments.iter().enumerate() {
                                assignment.accumulate_weighted_block(
                                    0,
                                    &weights,
                                    &real,
                                    &mut bins.values[index][parameter],
                                );
                                if parameter == 0 {
                                    assignment.accumulate_weighted_block_squared(
                                        0,
                                        &weights,
                                        &real,
                                        &mut bins.variances[index],
                                    );
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
                Ok(())
            })?;
        }
        for model in &models[1..] {
            for values in model
                .accepted
                .values
                .iter()
                .chain(&model.generated.values)
                .flatten()
            {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(invalid(format!(
                        "component {:?} has non-finite fitted yield",
                        model.tags
                    )));
                }
            }
        }
        let source_id = ensemble
            .filter(|_| draws > 0)
            .map(crate::Ensemble::source_id);
        let absolute = self.has_absolute_rate();
        let mut results = plans
            .iter()
            .enumerate()
            .map(|(index, plan)| {
                let full = &models[0];
                let a = &full.accepted.values[index][0];
                let g = &full.generated.values[index][0];
                let validity = (0..plan.volumes.len())
                    .map(|bin| {
                        if !plan.data_bins.weights[bin].is_finite() {
                            YieldBinValidity::NonFiniteEvaluation
                        } else {
                            fitted_bin_validity(
                                a[bin],
                                g[bin],
                                plan.accepted_bins.support[bin],
                                plan.generated_bins.support[bin],
                                plan.generated_bins.weights[bin],
                            )
                        }
                    })
                    .collect();
                let components = aliases
                    .iter()
                    .map(|(name, tags)| {
                        let model = models
                            .iter()
                            .find(|model| model.tags.as_ref() == Some(tags))
                            .expect("prepared component");
                        let a = &model.accepted.values[index][0];
                        let g = &model.generated.values[index][0];
                        (
                            name.clone(),
                            ComponentYieldProjection {
                                tags: tags.as_slice().to_vec(),
                                axes: plan.axes.clone(),
                                shape: plan.shape.clone(),
                                bin_volumes: plan.volumes.clone(),
                                accepted_source_id: accepted.identity(),
                                generated_source_id: generated.identity(),
                                accepted: absolute_rate_bins(a, absolute),
                                accepted_fill_variance: model.accepted.variances[index].clone(),
                                generated: absolute_rate_bins(g, absolute),
                                generated_fill_variance: model.generated.variances[index].clone(),
                                validity: (0..a.len())
                                    .map(|bin| {
                                        fitted_bin_validity(
                                            a[bin],
                                            g[bin],
                                            plan.accepted_bins.support[bin],
                                            plan.generated_bins.support[bin],
                                            plan.generated_bins.weights[bin],
                                        )
                                    })
                                    .collect(),
                                accepted_draws: model.accepted.values[index][1..]
                                    .iter()
                                    .map(|values| absolute_rate_bins(values, absolute))
                                    .collect(),
                                generated_draws: model.generated.values[index][1..]
                                    .iter()
                                    .map(|values| absolute_rate_bins(values, absolute))
                                    .collect(),
                                source_id,
                            },
                        )
                    })
                    .collect();
                YieldProjection {
                    axes: plan.axes.clone(),
                    shape: plan.shape.clone(),
                    bin_volumes: plan.volumes.clone(),
                    data_source_id: self.observed_data().identity(),
                    accepted_source_id: accepted.identity(),
                    generated_source_id: generated.identity(),
                    selected: finite_observed_bins(&plan.data_bins.weights),
                    selected_fill_variance: plan.data_bins.squared_weights.clone(),
                    accepted: absolute_rate_bins(a, absolute),
                    accepted_fill_variance: full.accepted.variances[index].clone(),
                    generated: absolute_rate_bins(g, absolute),
                    generated_fill_variance: full.generated.variances[index].clone(),
                    validity,
                    diagnostics: YieldProjectionDiagnostics {
                        selected_nonfinite: plan.data_bins.nonfinite_count,
                        selected_out_of_range: plan.data_bins.out_of_range_count,
                        accepted_nonfinite: plan.accepted_bins.nonfinite_count,
                        accepted_out_of_range: plan.accepted_bins.out_of_range_count,
                        generated_nonfinite: plan.generated_bins.nonfinite_count,
                        generated_out_of_range: plan.generated_bins.out_of_range_count,
                    },
                    has_absolute_rate: absolute,
                    components,
                    source_id,
                    has_replica_datasets: source_id.is_some()
                        && ensemble.is_some_and(|ensemble| !ensemble.replicas().is_empty()),
                    selected_draws: selected_draws[index]
                        .iter()
                        .map(|values| finite_observed_bins(values))
                        .collect(),
                    accepted_draws: full.accepted.values[index][1..]
                        .iter()
                        .map(|values| absolute_rate_bins(values, absolute))
                        .collect(),
                    generated_draws: full.generated.values[index][1..]
                        .iter()
                        .map(|values| absolute_rate_bins(values, absolute))
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        // Preserve the established behavior for replicas with different MC or
        // execution settings; they cannot use the central prepared batch plan.
        if !shared_mc && let Some(ensemble) = ensemble {
            let requests = unique
                .iter()
                .map(|request| (*request).clone())
                .collect::<Vec<_>>();
            for (draw, parameters) in ensemble.draws().iter().enumerate() {
                let likelihood = ensemble
                    .replicas()
                    .get(draw)
                    .cloned()
                    .unwrap_or_else(|| self.likelihood().clone());
                let context = Yield::with_ensemble(
                    likelihood,
                    self.term_name(),
                    generated.clone(),
                    parameters.clone(),
                    None,
                )?;
                let projected = context.projection_set_with_components(&requests, components)?;
                for (result, (_, draw)) in results.iter_mut().zip(projected.entries) {
                    result.source_id = Some(ensemble.source_id());
                    result.has_replica_datasets = !ensemble.replicas().is_empty();
                    result.selected_draws.push(draw.selected);
                    result.accepted_draws.push(draw.accepted);
                    result.generated_draws.push(draw.generated);
                    for (name, draw) in draw.components {
                        let component = result
                            .components
                            .get_mut(&name)
                            .expect("prepared component");
                        component.accepted_draws.push(draw.accepted);
                        component.generated_draws.push(draw.generated);
                        component.source_id = Some(ensemble.source_id());
                    }
                }
            }
        }
        Ok(YieldProjectionSet {
            entries: projections
                .iter()
                .zip(indexes)
                .map(|(request, index)| (request.name().to_owned(), results[index].clone()))
                .collect(),
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CanonicalTags(Vec<String>);

impl CanonicalTags {
    fn new(tags: &[String]) -> Self {
        let mut canonical = tags.to_vec();
        canonical.sort();
        canonical.dedup();
        Self(canonical)
    }

    fn as_slice(&self) -> &[String] {
        &self.0
    }
}

struct BinAssignments {
    indices: Vec<Option<usize>>,
    count: usize,
    nonfinite_count: usize,
    out_of_range_count: usize,
}

struct BinSummary {
    support: Vec<usize>,
    weights: Vec<f64>,
    squared_weights: Vec<f64>,
    nonfinite_count: usize,
    out_of_range_count: usize,
}

impl BinSummary {
    fn new(bins: usize) -> Self {
        Self {
            support: vec![0; bins],
            weights: vec![0.0; bins],
            squared_weights: vec![0.0; bins],
            nonfinite_count: 0,
            out_of_range_count: 0,
        }
    }
    fn record(&mut self, bins: &BinAssignments, weights: &[f64]) {
        self.nonfinite_count += bins.nonfinite_count;
        self.out_of_range_count += bins.out_of_range_count;
        for (index, weight) in bins.indices.iter().zip(weights) {
            if let Some(index) = index {
                self.support[*index] += 1;
                self.weights[*index] += weight;
                self.squared_weights[*index] += weight * weight;
            }
        }
    }
}

struct PreparedProjection {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    volumes: Vec<f64>,
    data_bins: BinSummary,
    accepted_bins: BinSummary,
    generated_bins: BinSummary,
}

impl PreparedProjection {
    fn new(request: &Projection) -> LikelihoodResult<Self> {
        let shape = request.axes().iter().map(Axis::bins).collect::<Vec<_>>();
        let count = shape
            .iter()
            .try_fold(1usize, |count, bins| count.checked_mul(*bins))
            .ok_or_else(|| invalid("projection axis shape exceeds addressable bin count"))?;
        Ok(Self {
            axes: request
                .axes()
                .iter()
                .map(|axis| axis.edges().to_vec())
                .collect(),
            shape,
            volumes: bin_volumes(request.axes()),
            data_bins: BinSummary::new(count),
            accepted_bins: BinSummary::new(count),
            generated_bins: BinSummary::new(count),
        })
    }
}

struct ModelBins {
    values: Vec<Vec<Vec<f64>>>,
    variances: Vec<Vec<f64>>,
}

impl ModelBins {
    fn new(plans: &[PreparedProjection], parameters: usize) -> Self {
        Self {
            values: plans
                .iter()
                .map(|plan| vec![vec![0.0; plan.volumes.len()]; parameters])
                .collect(),
            variances: plans
                .iter()
                .map(|plan| vec![0.0; plan.volumes.len()])
                .collect(),
        }
    }
}

struct ProjectionModel {
    tags: Option<CanonicalTags>,
    evaluator: crate::likelihood::IntensityEvaluator,
    parameters: Vec<laddu_expr::parameters::ParamValues>,
    accepted: ModelBins,
    generated: ModelBins,
}

struct ProjectionCoordinates<'a> {
    query: laddu_runtime::PreparedQuery,
    projections: &'a [&'a Projection],
    axes: Vec<Vec<usize>>,
}

impl<'a> ProjectionCoordinates<'a> {
    fn prepare(projections: &'a [&'a Projection], execution: &Execution) -> LikelihoodResult<Self> {
        let mut indexes = HashMap::new();
        let mut expressions = Vec::new();
        let axes = projections
            .iter()
            .map(|projection| {
                projection
                    .axes()
                    .iter()
                    .map(|axis| {
                        let graph = axis.expression.to_graph();
                        let key = (
                            graph.root().index(),
                            graph
                                .nodes()
                                .iter()
                                .map(|node| node.structural_key())
                                .collect::<Vec<_>>(),
                        );
                        *indexes.entry(key).or_insert_with(|| {
                            let index = expressions.len();
                            expressions.push(axis.expression.clone());
                            index
                        })
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            query: laddu_runtime::PreparedQuery::prepare(expressions, execution, true)?,
            projections,
            axes,
        })
    }

    fn visit(
        &self,
        source: &Dataset,
        execution: &Execution,
        model_memory: (usize, usize),
        mut consume: impl FnMut(
            &laddu_data::data::EventBatch,
            &[BinAssignments],
        ) -> LikelihoodResult<()>,
    ) -> LikelihoodResult<()> {
        let schema = source
            .schema()
            .map_err(|error| invalid(error.to_string()))?;
        let source_memory = laddu_data::BatchLayout::from_schema(&schema)
            .schema_footprint(laddu_data::schema::Precision::F64)
            .map_err(|error| invalid(error.to_string()))?;
        let query_fixed = self.query.batch_memory_estimate(0);
        let query_event = self
            .query
            .batch_memory_estimate(1)
            .saturating_sub(query_fixed);
        let fixed = source_memory
            .fixed_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(query_fixed as u64))
            .and_then(|bytes| bytes.checked_add(model_memory.0 as u64))
            .ok_or_else(|| invalid("projection batch workspace size overflow"))?;
        let event = source_memory
            .bytes_per_event
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(query_event as u64))
            .and_then(|bytes| bytes.checked_add(model_memory.1 as u64))
            .and_then(|bytes| {
                bytes.checked_add(
                    (self.projections.len() as u64)
                        .checked_mul((std::mem::size_of::<Option<usize>>() + 1) as u64)?,
                )
            })
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or_else(|| invalid("projection batch workspace size overflow"))?;
        let available = execution.host_memory().remaining();
        let maximum = source.read_plan().chunk_size.unwrap_or(8192).min(8192);
        let fit = available
            .saturating_sub(fixed)
            .checked_div(event)
            .unwrap_or(maximum as u64);
        let chunk = maximum.min(usize::try_from(fit).unwrap_or(usize::MAX));
        let bytes = fixed
            .checked_add(
                event
                    .checked_mul(chunk.max(1) as u64)
                    .ok_or_else(|| invalid("projection batch workspace size overflow"))?,
            )
            .ok_or_else(|| invalid("projection batch workspace size overflow"))?;
        let _batch_workspace = execution
            .host_memory()
            .reserve(bytes)
            .map_err(laddu_runtime::RuntimeError::from)?;
        let mut read_plan = source.read_plan();
        read_plan.chunk_size = Some(chunk.max(1));
        let mut events = 0;
        for batch in source
            .stream_with_plan(read_plan)
            .map_err(|error| invalid(error.to_string()))?
        {
            let batch = batch.map_err(|error| invalid(error.to_string()))?;
            let values = self.query.evaluate_batch(&batch)?;
            let assignments = self
                .projections
                .iter()
                .zip(&self.axes)
                .map(|(projection, axes)| {
                    let count = checked_bin_count(
                        &projection
                            .axes()
                            .iter()
                            .map(|axis| axis.binning.clone())
                            .collect::<Vec<_>>(),
                    )
                    .expect("validated bin count");
                    let mut indices = Vec::with_capacity(batch.len());
                    let mut nonfinite_count = 0;
                    let mut out_of_range_count = 0;
                    for (row, _) in values[0].iter().enumerate() {
                        let mut index = Some(0usize);
                        let mut nonfinite = false;
                        for (axis, &column) in projection.axes().iter().zip(axes) {
                            let value = values[column][row].re;
                            if !value.is_finite() {
                                nonfinite = true;
                                index = None;
                            } else if let Some(current) = index {
                                index = axis
                                    .binning
                                    .index(value, FinalUpperEdge::Exclusive)
                                    .and_then(|bin| {
                                        current.checked_mul(axis.bins())?.checked_add(bin)
                                    });
                            }
                        }
                        if nonfinite {
                            nonfinite_count += 1;
                        } else if index.is_none() {
                            out_of_range_count += 1;
                        }
                        indices.push(index);
                    }
                    BinAssignments {
                        indices,
                        count,
                        nonfinite_count,
                        out_of_range_count,
                    }
                })
                .collect::<Vec<_>>();
            consume(&batch, &assignments)?;
            events += batch.len();
        }
        if u64::try_from(events).ok() != Some(source.stats()?.events()) {
            return Err(invalid(
                "projection coordinate count does not match dataset events",
            ));
        }
        Ok(())
    }
}

fn finite_observed_bins(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| if value.is_finite() { *value } else { f64::NAN })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProjectionKey(Vec<AxisKey>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AxisKey {
    root: usize,
    nodes: Vec<ExprNodeStructuralKey>,
    edges: Vec<u64>,
}

impl ProjectionKey {
    fn new(axes: &[Axis]) -> Self {
        Self(
            axes.iter()
                .map(|axis| {
                    let graph = axis.expression.to_graph();
                    AxisKey {
                        root: graph.root().index(),
                        nodes: graph
                            .nodes()
                            .iter()
                            .map(|node| node.structural_key())
                            .collect(),
                        edges: axis
                            .binning
                            .edges()
                            .iter()
                            .map(|edge| edge.to_bits())
                            .collect(),
                    }
                })
                .collect(),
        )
    }
}

fn deduplicate_projections(projections: &[Projection]) -> (Vec<&Projection>, Vec<usize>) {
    let mut unique_indexes = HashMap::with_capacity(projections.len());
    let mut unique = Vec::with_capacity(projections.len());
    let indexes = projections
        .iter()
        .map(|projection| {
            let key = ProjectionKey::new(projection.axes());
            if let Some(index) = unique_indexes.get(&key) {
                return *index;
            }
            let index = unique.len();
            unique.push(projection);
            unique_indexes.insert(key, index);
            index
        })
        .collect();
    (unique, indexes)
}

impl BinAssignments {
    fn accumulate_weighted_block(
        &self,
        offset: usize,
        weights: &[f64],
        intensities: &[f64],
        bins: &mut [f64],
    ) {
        debug_assert!(offset + intensities.len() <= self.indices.len());
        debug_assert_eq!(self.indices.len(), weights.len());
        debug_assert_eq!(self.count, bins.len());
        for (row, &intensity) in intensities.iter().enumerate() {
            if let Some(index) = self.indices[offset + row] {
                bins[index] += weights[offset + row] * intensity;
            }
        }
    }

    fn accumulate_weighted_block_squared(
        &self,
        offset: usize,
        weights: &[f64],
        intensities: &[f64],
        bins: &mut [f64],
    ) {
        for (row, &intensity) in intensities.iter().enumerate() {
            let event = offset + row;
            if let Some(index) = self.indices[event] {
                let value = weights[event] * intensity;
                bins[index] += value * value;
            }
        }
    }
}

pub(crate) fn dataset_weights(dataset: &Dataset) -> LikelihoodResult<Vec<f64>> {
    dataset
        .try_fold_events(Vec::new(), |mut weights, event| {
            weights.push(event.weight());
            Ok(weights)
        })
        .map_err(Into::into)
}

fn bin_volumes(axes: &[Axis]) -> Vec<f64> {
    axes.iter().fold(vec![1.0], |volumes, axis| {
        volumes
            .into_iter()
            .flat_map(|volume| {
                axis.binning
                    .edges()
                    .windows(2)
                    .map(move |pair| volume * (pair[1] - pair[0]))
            })
            .collect()
    })
}

fn valid_bin_volumes(axes: &[Axis]) -> bool {
    !axes.is_empty()
        && axes
            .iter()
            .try_fold((1.0_f64, 1.0_f64), |(min_volume, max_volume), axis| {
                let (min_width, max_width) = axis.binning.edges().windows(2).fold(
                    (f64::INFINITY, 0.0_f64),
                    |(min_width, max_width), pair| {
                        let width = pair[1] - pair[0];
                        (min_width.min(width), max_width.max(width))
                    },
                );
                let min_volume = min_volume * min_width;
                let max_volume = max_volume * max_width;
                (min_volume.is_finite()
                    && min_volume > 0.0
                    && max_volume.is_finite()
                    && max_volume > 0.0)
                    .then_some((min_volume, max_volume))
            })
            .is_some()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_data::{
        data::{EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{Expr, event_scalar, parameter};

    use super::*;
    use crate::likelihood::TAGGED_PROJECTION_PREPARATIONS;
    use crate::{Ensemble, ExtendedNllTerm, Likelihood};

    fn dataset(values: &[(f64, f64)]) -> Dataset {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let batch = EventBatch::from_events(
            schema,
            values
                .iter()
                .map(|(x, weight)| OwnedEvent::weighted(vec![], vec![*x], *weight)),
        )
        .unwrap();
        Dataset::from_batches(vec![batch]).unwrap()
    }

    #[test]
    fn streaming_tagged_projections_fit_without_event_count_workspace() {
        use laddu_runtime::{ExecutionOptions, MemoryBudget, MemoryPlan};

        let model = CompiledModel::from_expr(
            &((Expr::from(parameter!("a", initial: 1.0)) * event_scalar("x")).tagged("a")
                + Expr::from(parameter!("b", initial: 1.0)).tagged("b"))
            .norm_sqr(),
        )
        .unwrap();
        let data = dataset(&[(0.25, 2.0), (0.75, -0.2)]).streaming();
        let accepted = dataset(&[(0.25, 1.0), (0.75, 1.0)]).streaming();
        let generated = dataset(&vec![(0.25, 1.0); 32_768])
            .streaming()
            .chunked(64)
            .unwrap();
        let execution = Execution::local(ExecutionOptions {
            memory: MemoryPlan::host(MemoryBudget::Bytes(256 * 1024)),
            ..ExecutionOptions::default()
        })
        .unwrap();
        let likelihood = Arc::new(
            Likelihood::with_execution(
                [ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()],
                &execution,
            )
            .unwrap(),
        );
        let ensemble = Ensemble::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.0, 1.0], vec![2.0, 1.0]],
        )
        .unwrap();
        let yields = Yield::with_ensemble(
            likelihood.clone(),
            "signal",
            generated.clone(),
            likelihood.default_params(),
            Some(ensemble),
        )
        .unwrap();
        let requests = [
            Projection::new(
                "fine",
                vec![Axis::new(event_scalar("x"), vec![0.0, 0.5, 1.0]).unwrap()],
            )
            .unwrap(),
            Projection::new(
                "wide",
                vec![Axis::new(event_scalar("x"), vec![0.0, 1.0]).unwrap()],
            )
            .unwrap(),
        ];
        let components = HashMap::from([
            ("a".into(), vec!["a".into()]),
            ("alias".into(), vec!["a".into(), "a".into()]),
            ("coherent".into(), vec!["a".into(), "b".into()]),
        ]);
        let before = execution.host_memory().report().reserved_bytes;
        let traversals = generated.source_traversals();
        TAGGED_PROJECTION_PREPARATIONS.with(|count| count.set(0));
        let result = yields
            .projection_set_with_components(&requests, &components)
            .unwrap();
        assert_eq!(TAGGED_PROJECTION_PREPARATIONS.with(|count| count.get()), 2);
        assert_eq!(generated.source_traversals() - traversals, 1);
        let fine = result.get("fine").unwrap();
        assert_eq!(fine.selected(), &[2.0, -0.2]);
        assert_eq!(fine.generated(), &[32_768.0 * 1.25_f64.powi(2), 0.0]);
        assert_eq!(
            fine.generated_draws()[1],
            vec![32_768.0 * 1.5_f64.powi(2), 0.0]
        );
        assert_eq!(
            fine.components()["a"].generated_draws(),
            fine.components()["alias"].generated_draws()
        );
        assert_eq!(execution.host_memory().report().reserved_bytes, before);
        assert!(execution.host_memory().report().high_water_bytes <= 256 * 1024);
    }

    #[test]
    fn tagged_bootstrap_projections_prepare_once_and_preserve_paired_bins() {
        let a = (Expr::from(parameter!("a", initial: 1.0)) * event_scalar("x")).tagged("a");
        let b = Expr::from(parameter!("b", initial: 1.0)).tagged("b");
        let model = CompiledModel::from_expr(&(a + b).norm_sqr()).unwrap();
        let data = dataset(&[(0.25, 2.0), (0.75, -0.2), (0.25, 1.0), (0.75, 3.0)]);
        let accepted = dataset(&[(0.25, 1.0), (0.75, 1.0)]);
        let generated = dataset(&[(0.25, 1.0), (0.75, 1.0), (0.25, 2.0), (0.75, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let components = HashMap::from([
            ("a".into(), vec!["a".into()]),
            ("a_alias".into(), vec!["a".into(), "a".into()]),
            ("b".into(), vec!["b".into()]),
            ("coherent".into(), vec!["b".into(), "a".into()]),
        ]);
        let axis = Axis::new(event_scalar("x"), vec![0.0, 0.5, 1.0]).unwrap();
        for samples in [1, 20, 200] {
            let ensemble = Ensemble::bootstrap_fit(&likelihood, samples, 42, |_, index| {
                Ok::<_, std::convert::Infallible>(vec![0.75 + index as f64 * 0.005, 1.0])
            })
            .unwrap();
            let yields = Yield::with_ensemble(
                likelihood.clone(),
                "signal",
                generated.clone(),
                likelihood.default_params(),
                Some(ensemble.clone()),
            )
            .unwrap();
            TAGGED_PROJECTION_PREPARATIONS.with(|count| count.set(0));
            let projected = yields
                .projection_with_components(std::slice::from_ref(&axis), &components)
                .unwrap();
            assert_eq!(
                TAGGED_PROJECTION_PREPARATIONS.with(|count| count.get()),
                3,
                "preparation must depend on unique components, not {samples} draws"
            );
            assert_eq!(
                projected.components()["a"].accepted_draws(),
                projected.components()["a_alias"].accepted_draws()
            );
            for (index, parameters) in ensemble.draws().iter().enumerate() {
                let (replica_data, _) = ensemble.replicas()[index]
                    .intensity_datasets("signal")
                    .unwrap();
                let weights = dataset_weights(replica_data).unwrap();
                assert_relative_eq!(
                    projected.selected_draws()[index][0],
                    weights[0] + weights[2]
                );
                assert_relative_eq!(
                    projected.selected_draws()[index][1],
                    weights[1] + weights[3]
                );
                for (bin, x) in [0.25_f64, 0.75].into_iter().enumerate() {
                    let total = (parameters[0] * x + parameters[1]).powi(2);
                    let selected = (parameters[0] * x).powi(2);
                    let exposure = if bin == 0 { 3.0 } else { 2.0 };
                    assert_relative_eq!(projected.accepted_draws()[index][bin], total);
                    assert_relative_eq!(projected.generated_draws()[index][bin], exposure * total);
                    assert_relative_eq!(
                        projected.components()["a"].accepted_draws()[index][bin],
                        selected
                    );
                    assert_relative_eq!(
                        projected.components()["b"].generated_draws()[index][bin],
                        exposure * parameters[1].powi(2)
                    );
                    assert_relative_eq!(
                        projected.components()["coherent"].generated_draws()[index][bin],
                        exposure * total
                    );
                }
            }
        }
    }

    #[test]
    fn tagged_draws_match_independent_projections_for_arbitrary_replica_sources() {
        let a = (Expr::from(parameter!("a", initial: 1.0)) * event_scalar("x")).tagged("a");
        let b = Expr::from(parameter!("b", initial: 1.0)).tagged("b");
        let model = CompiledModel::from_expr(&(a + b).norm_sqr()).unwrap();
        let data = dataset(&[(0.25, 2.0), (0.75, -0.2), (0.75, 3.0)]);
        let accepted = dataset(&[(0.25, 1.0), (0.75, 2.0)]);
        let generated = dataset(&[(0.25, 3.0), (0.75, 4.0)]);
        let likelihood = Arc::new(
            Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let components = HashMap::from([
            ("a".into(), vec!["a".into()]),
            ("coherent".into(), vec!["a".into(), "b".into()]),
        ]);
        let requests = [
            Projection::new(
                "fine",
                vec![Axis::new(event_scalar("x"), vec![0.0, 0.5, 1.0]).unwrap()],
            )
            .unwrap(),
            Projection::new(
                "wide",
                vec![Axis::new(event_scalar("x"), vec![0.0, 1.0]).unwrap()],
            )
            .unwrap(),
        ];
        for changed_mc in [false, true] {
            let replicas = [
                dataset(&[(0.75, -0.5), (0.25, 4.0)]),
                dataset(&[(0.75, 3.0), (0.25, -0.25), (0.25, 2.0), (0.75, 1.0)]),
            ]
            .into_iter()
            .enumerate()
            .map(|(index, data)| {
                let mc = if changed_mc {
                    dataset(&[(0.75, 3.0 + index as f64), (0.25, 0.5)])
                } else {
                    accepted.clone()
                };
                Arc::new(
                    Likelihood::new([ExtendedNllTerm::new("signal", &model, &data, &mc).unwrap()])
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
            let ensemble = Ensemble::with_replicas(
                vec!["a".into(), "b".into()],
                vec![vec![0.7, 1.2], vec![1.4, 0.5]],
                replicas,
            )
            .unwrap();
            let yields = Yield::with_ensemble(
                likelihood.clone(),
                "signal",
                generated.clone(),
                likelihood.default_params(),
                Some(ensemble.clone()),
            )
            .unwrap();
            TAGGED_PROJECTION_PREPARATIONS.with(|count| count.set(0));
            let actual = yields
                .projection_set_with_components(&requests, &components)
                .unwrap();
            if !changed_mc {
                assert_eq!(TAGGED_PROJECTION_PREPARATIONS.with(|count| count.get()), 2);
            }
            for (draw, parameters) in ensemble.draws().iter().enumerate() {
                let reference = Yield::with_ensemble(
                    ensemble.replicas()[draw].clone(),
                    "signal",
                    generated.clone(),
                    parameters.clone(),
                    None,
                )
                .unwrap()
                .projection_set_with_components(&requests, &components)
                .unwrap();
                for (name, expected) in reference.iter() {
                    let result = actual.get(name).unwrap();
                    assert_eq!(result.source_id, Some(ensemble.source_id()));
                    assert!(result.has_replica_datasets);
                    for (got, want) in result.selected_draws()[draw]
                        .iter()
                        .zip(expected.selected())
                    {
                        assert_relative_eq!(got, want, epsilon = 1e-12);
                    }
                    for (got, want) in result.accepted_draws()[draw]
                        .iter()
                        .zip(expected.accepted())
                    {
                        assert_relative_eq!(got, want, epsilon = 1e-12);
                    }
                    for (got, want) in result.generated_draws()[draw]
                        .iter()
                        .zip(expected.generated())
                    {
                        assert_relative_eq!(got, want, epsilon = 1e-12);
                    }
                    for (label, expected) in expected.components() {
                        let component = &result.components()[label];
                        assert_eq!(component.source_id, Some(ensemble.source_id()));
                        for (got, want) in component.accepted_draws()[draw]
                            .iter()
                            .zip(expected.accepted_estimate().values())
                        {
                            assert_relative_eq!(got, want, epsilon = 1e-12);
                        }
                        for (got, want) in component.generated_draws()[draw]
                            .iter()
                            .zip(expected.generated_estimate().values())
                        {
                            assert_relative_eq!(got, want, epsilon = 1e-12);
                        }
                    }
                }
            }
        }
    }
}
