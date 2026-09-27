//! Binned raw-yield projections and their support diagnostics.

use std::collections::{HashMap, HashSet};

use laddu_data::data::Dataset;
use laddu_expr::ExprNodeStructuralKey;
use laddu_runtime::{DatasetExprExt, Execution, FinalUpperEdge, checked_bin_count};
use rayon::prelude::*;

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
        for projection in projections {
            if !names.insert(projection.name()) {
                return Err(invalid(format!(
                    "duplicate projection name: {}",
                    projection.name()
                )));
            }
            if !valid_bin_volumes(projection.axes()) {
                return Err(invalid(format!(
                    "projection `{}` has invalid bin volume",
                    projection.name()
                )));
            }
        }
        let execution = self.likelihood().execution();
        let mut seen_expressions = HashSet::new();
        let mut expressions = Vec::new();
        for projection in projections {
            for axis in projection.axes() {
                let graph = axis.expression.to_graph();
                let key = (
                    graph.root().index(),
                    graph
                        .nodes()
                        .iter()
                        .map(|node| node.structural_key())
                        .collect::<Vec<_>>(),
                );
                if seen_expressions.insert(key) {
                    expressions.push(axis.expression.clone());
                }
            }
        }
        self.observed_data()
            .validate_real_expressions(&expressions, execution)?;
        let component_aliases = components
            .iter()
            .map(|(name, tags)| {
                if name.trim().is_empty()
                    || tags.is_empty()
                    || tags.iter().any(|tag| tag.trim().is_empty())
                {
                    return Err(invalid(format!("invalid component name or tags: `{name}`")));
                }
                Ok((name.clone(), CanonicalTags::new(tags)))
            })
            .collect::<LikelihoodResult<HashMap<_, _>>>()?;
        for (name, tags) in &component_aliases {
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
        }
        let integrals = self
            .likelihood()
            .intensity_integrals(self.term_name(), self.generated_mc())?;
        let mut component_integrals = HashMap::new();
        for tags in component_aliases.values() {
            if !component_integrals.contains_key(tags) {
                let selected = self
                    .likelihood()
                    .intensity_integrals_with_tags(
                        self.term_name(),
                        self.generated_mc(),
                        tags.as_slice().iter().map(String::as_str),
                    )
                    .map_err(|error| {
                        invalid(format!("component {:?}: {error}", tags.as_slice()))
                    })?;
                component_integrals.insert(tags.clone(), selected);
            }
        }
        let data = self.observed_data();
        let accepted = integrals.accepted_mc_source();
        let generated = integrals.generated_mc_source();
        let data_source_id = data.identity();
        let accepted_source_id = accepted.identity();
        let generated_source_id = generated.identity();
        let (unique, indexes) = deduplicate_projections(projections);
        let sample_events = [data, accepted, generated]
            .iter()
            .map(|sample| {
                usize::try_from(sample.stats()?.events())
                    .map_err(|_| invalid("projection event count exceeds addressable memory"))
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let event_total = sample_events
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let bins_total = unique
            .iter()
            .try_fold(0usize, |sum, projection| {
                let bins = checked_bin_count(
                    &projection
                        .axes()
                        .iter()
                        .map(|axis| axis.binning.clone())
                        .collect::<Vec<_>>(),
                )?;
                sum.checked_add(bins)
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let output_bins = projections
            .iter()
            .try_fold(0usize, |sum, projection| {
                let bins = checked_bin_count(
                    &projection
                        .axes()
                        .iter()
                        .map(|axis| axis.binning.clone())
                        .collect::<Vec<_>>(),
                )?;
                sum.checked_add(bins)
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let largest_sample = sample_events.iter().copied().max().unwrap_or(0);
        let distinct_axes = expressions.len();
        let workspace_bytes = event_total
            .checked_mul(unique.len())
            .and_then(|count| count.checked_mul(std::mem::size_of::<Option<usize>>() + 1))
            .and_then(|bytes| {
                bytes.checked_add(event_total.checked_mul(std::mem::size_of::<f64>())?)
            })
            .and_then(|bytes| {
                bytes.checked_add(bins_total.checked_mul(5 * std::mem::size_of::<f64>())?)
            })
            .and_then(|bytes| {
                bytes.checked_add(output_bins.checked_mul(10 * std::mem::size_of::<f64>())?)
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    bins_total
                        .checked_mul(component_integrals.len())?
                        .checked_mul(4 * std::mem::size_of::<f64>())?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    output_bins
                        .checked_mul(component_aliases.len())?
                        .checked_mul(
                            4 * std::mem::size_of::<f64>()
                                + std::mem::size_of::<YieldBinValidity>(),
                        )?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    distinct_axes
                        .checked_mul(largest_sample.min(8192))?
                        .checked_mul(32)?,
                )
            })
            .ok_or_else(|| invalid("projection workspace size overflow"))?;
        let workspace_bytes = u64::try_from(workspace_bytes)
            .map_err(|_| invalid("projection workspace size overflow"))?;
        let _workspace_lease =
            execution
                .host_memory()
                .reserve(workspace_bytes)
                .map_err(|error| {
                    LikelihoodError::Runtime(laddu_runtime::RuntimeError::Memory(error))
                })?;
        let data_weights = dataset_weights(data)?;
        let accepted_weights = dataset_weights(accepted)?;
        let generated_weights = dataset_weights(generated)?;
        let data_assignments = evaluate_bin_assignments_many(data, &unique, execution)?;
        let accepted_assignments = evaluate_bin_assignments_many(accepted, &unique, execution)?;
        let generated_assignments = evaluate_bin_assignments_many(generated, &unique, execution)?;
        let plans = unique
            .iter()
            .zip(data_assignments)
            .zip(accepted_assignments)
            .zip(generated_assignments)
            .map(
                |(((projection, data_bins), accepted_bins), generated_bins)| {
                    Ok(PreparedProjection {
                        name: projection.name().to_owned(),
                        axes: projection
                            .axes()
                            .iter()
                            .map(|axis| axis.edges().to_vec())
                            .collect(),
                        shape: projection.axes().iter().map(Axis::bins).collect(),
                        volumes: bin_volumes(projection.axes()),
                        data_bins,
                        accepted_bins,
                        generated_bins,
                    })
                },
            )
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let mut accepted_sums = plans
            .iter()
            .map(|plan| vec![0.0; plan.accepted_bins.count])
            .collect::<Vec<_>>();
        let mut generated_sums = plans
            .iter()
            .map(|plan| vec![0.0; plan.generated_bins.count])
            .collect::<Vec<_>>();
        let mut accepted_variances = accepted_sums.clone();
        let mut generated_variances = generated_sums.clone();
        let mut component_sums = component_integrals
            .keys()
            .map(|tags| {
                let accepted = plans
                    .iter()
                    .map(|plan| vec![0.0; plan.accepted_bins.count])
                    .collect::<Vec<_>>();
                let generated = plans
                    .iter()
                    .map(|plan| vec![0.0; plan.generated_bins.count])
                    .collect::<Vec<_>>();
                (tags.clone(), (accepted, generated))
            })
            .collect::<HashMap<_, _>>();
        let mut component_variances = component_sums.clone();
        if component_integrals.is_empty() {
            let parameters = [self.parameters()];
            let contexts = ["central yield projection".to_owned()];
            integrals.visit_accepted_raw_prepared_intensities_many(
                &parameters,
                &contexts,
                |offset, _, values| {
                    for ((plan, sums), variances) in plans
                        .iter()
                        .zip(&mut accepted_sums)
                        .zip(&mut accepted_variances)
                    {
                        plan.accepted_bins.accumulate_weighted_block(
                            offset,
                            &accepted_weights,
                            values,
                            sums,
                        );
                        plan.accepted_bins.accumulate_weighted_block_squared(
                            offset,
                            &accepted_weights,
                            values,
                            variances,
                        );
                    }
                },
            )?;
            integrals.visit_generated_prepared_intensities_many(
                &parameters,
                &contexts,
                |offset, _, values| {
                    for ((plan, sums), variances) in plans
                        .iter()
                        .zip(&mut generated_sums)
                        .zip(&mut generated_variances)
                    {
                        plan.generated_bins.accumulate_weighted_block(
                            offset,
                            &generated_weights,
                            values,
                            sums,
                        );
                        plan.generated_bins.accumulate_weighted_block_squared(
                            offset,
                            &generated_weights,
                            values,
                            variances,
                        );
                    }
                },
            )?;
        } else {
            let selections = component_integrals.iter().collect::<Vec<_>>();
            let models = selections
                .iter()
                .map(|(_, model)| *model)
                .collect::<Vec<_>>();
            let labels = std::iter::once("full model".to_owned())
                .chain(selections.iter().map(|(tags, _)| {
                    let aliases = component_aliases
                        .iter()
                        .filter_map(|(name, selection)| {
                            (selection == *tags).then_some(name.as_str())
                        })
                        .collect::<Vec<_>>();
                    format!("component {aliases:?} {:?}", tags.as_slice())
                }))
                .collect::<Vec<_>>();
            let projection_names = projections.iter().map(Projection::name).collect::<Vec<_>>();
            integrals
                .visit_shared_source_intensities(
                    &models,
                    &labels,
                    self.parameters(),
                    false,
                    |index, offset, values| {
                        if index == 0 {
                            for ((plan, sums), variances) in plans
                                .iter()
                                .zip(&mut accepted_sums)
                                .zip(&mut accepted_variances)
                            {
                                plan.accepted_bins.accumulate_weighted_block(
                                    offset,
                                    &accepted_weights,
                                    values,
                                    sums,
                                );
                                plan.accepted_bins.accumulate_weighted_block_squared(
                                    offset,
                                    &accepted_weights,
                                    values,
                                    variances,
                                );
                            }
                        } else {
                            let sums = &mut component_sums
                                .get_mut(selections[index - 1].0)
                                .expect("prepared component")
                                .0;
                            let variances = &mut component_variances
                                .get_mut(selections[index - 1].0)
                                .expect("prepared component")
                                .0;
                            for ((plan, bins), variance) in plans.iter().zip(sums).zip(variances) {
                                plan.accepted_bins.accumulate_weighted_block(
                                    offset,
                                    &accepted_weights,
                                    values,
                                    bins,
                                );
                                plan.accepted_bins.accumulate_weighted_block_squared(
                                    offset,
                                    &accepted_weights,
                                    values,
                                    variance,
                                );
                            }
                        }
                    },
                )
                .map_err(|error| {
                    invalid(format!(
                        "projections {projection_names:?}, accepted MC: {error}"
                    ))
                })?;
            integrals
                .visit_shared_source_intensities(
                    &models,
                    &labels,
                    self.parameters(),
                    true,
                    |index, offset, values| {
                        if index == 0 {
                            for ((plan, sums), variances) in plans
                                .iter()
                                .zip(&mut generated_sums)
                                .zip(&mut generated_variances)
                            {
                                plan.generated_bins.accumulate_weighted_block(
                                    offset,
                                    &generated_weights,
                                    values,
                                    sums,
                                );
                                plan.generated_bins.accumulate_weighted_block_squared(
                                    offset,
                                    &generated_weights,
                                    values,
                                    variances,
                                );
                            }
                        } else {
                            let sums = &mut component_sums
                                .get_mut(selections[index - 1].0)
                                .expect("prepared component")
                                .1;
                            let variances = &mut component_variances
                                .get_mut(selections[index - 1].0)
                                .expect("prepared component")
                                .1;
                            for ((plan, bins), variance) in plans.iter().zip(sums).zip(variances) {
                                plan.generated_bins.accumulate_weighted_block(
                                    offset,
                                    &generated_weights,
                                    values,
                                    bins,
                                );
                                plan.generated_bins.accumulate_weighted_block_squared(
                                    offset,
                                    &generated_weights,
                                    values,
                                    variance,
                                );
                            }
                        }
                    },
                )
                .map_err(|error| {
                    invalid(format!(
                        "projections {projection_names:?}, generated MC: {error}"
                    ))
                })?;
            for (tags, (accepted, generated)) in &component_sums {
                for (index, plan) in plans.iter().enumerate() {
                    if accepted[index]
                        .iter()
                        .chain(&generated[index])
                        .any(|value| !value.is_finite())
                    {
                        return Err(invalid(format!(
                            "projection `{}` component {:?} has non-finite fitted yield",
                            plan.name,
                            tags.as_slice()
                        )));
                    }
                }
            }
        }
        let mut unique_results = plans
            .iter()
            .enumerate()
            .map(|(plan_index, plan)| {
                let mut selected = plan.data_bins.accumulate_products(&data_weights, None);
                let accepted_raw = &accepted_sums[plan_index];
                let generated_raw = &generated_sums[plan_index];
                let accepted_counts = plan.accepted_bins.support_counts();
                let generated_counts = plan.generated_bins.support_counts();
                let generated_exposure = plan
                    .generated_bins
                    .accumulate_products(&generated_weights, None);
                let mut validity = Vec::with_capacity(selected.len());
                for bin in 0..selected.len() {
                    let d = selected[bin];
                    let a = accepted_raw[bin];
                    let g = generated_raw[bin];
                    let status = if !d.is_finite() {
                        YieldBinValidity::NonFiniteEvaluation
                    } else {
                        fitted_bin_validity(
                            a,
                            g,
                            accepted_counts[bin],
                            generated_counts[bin],
                            generated_exposure[bin],
                        )
                    };
                    validity.push(status);
                    if !d.is_finite() {
                        selected[bin] = f64::NAN;
                    }
                }
                YieldProjection {
                    axes: plan.axes.clone(),
                    shape: plan.shape.clone(),
                    bin_volumes: plan.volumes.clone(),
                    data_source_id,
                    accepted_source_id,
                    generated_source_id,
                    selected,
                    selected_fill_variance: plan
                        .data_bins
                        .accumulate_products_squared(&data_weights),
                    accepted: absolute_rate_bins(accepted_raw, self.has_absolute_rate()),
                    accepted_fill_variance: accepted_variances[plan_index].clone(),
                    generated: absolute_rate_bins(generated_raw, self.has_absolute_rate()),
                    generated_fill_variance: generated_variances[plan_index].clone(),
                    validity,
                    diagnostics: YieldProjectionDiagnostics {
                        selected_nonfinite: plan.data_bins.nonfinite_count,
                        selected_out_of_range: plan.data_bins.out_of_range_count,
                        accepted_nonfinite: plan.accepted_bins.nonfinite_count,
                        accepted_out_of_range: plan.accepted_bins.out_of_range_count,
                        generated_nonfinite: plan.generated_bins.nonfinite_count,
                        generated_out_of_range: plan.generated_bins.out_of_range_count,
                    },
                    has_absolute_rate: self.has_absolute_rate(),
                    source_id: None,
                    has_replica_datasets: false,
                    selected_draws: Vec::new(),
                    accepted_draws: Vec::new(),
                    generated_draws: Vec::new(),
                    components: component_aliases
                        .iter()
                        .map(|(name, tags)| {
                            let (accepted_sums, generated_sums) = &component_sums[tags];
                            let accepted = &accepted_sums[plan_index];
                            let generated = &generated_sums[plan_index];
                            let validity = (0..accepted.len())
                                .map(|bin| {
                                    fitted_bin_validity(
                                        accepted[bin],
                                        generated[bin],
                                        accepted_counts[bin],
                                        generated_counts[bin],
                                        generated_exposure[bin],
                                    )
                                })
                                .collect();
                            let accepted = absolute_rate_bins(accepted, self.has_absolute_rate());
                            let generated = absolute_rate_bins(generated, self.has_absolute_rate());
                            let (accepted_variance, generated_variance) =
                                &component_variances[tags];
                            (
                                name.clone(),
                                ComponentYieldProjection {
                                    tags: tags.as_slice().to_vec(),
                                    axes: plan.axes.clone(),
                                    shape: plan.shape.clone(),
                                    bin_volumes: plan.volumes.clone(),
                                    accepted_source_id,
                                    generated_source_id,
                                    accepted,
                                    accepted_fill_variance: accepted_variance[plan_index].clone(),
                                    generated,
                                    generated_fill_variance: generated_variance[plan_index].clone(),
                                    validity,
                                    accepted_draws: Vec::new(),
                                    generated_draws: Vec::new(),
                                    source_id: None,
                                },
                            )
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        if let Some(ensemble) = self.ensemble() {
            let draw_requests = unique
                .iter()
                .map(|request| (*request).clone())
                .collect::<Vec<_>>();
            for (draw_index, parameters) in ensemble.draws().iter().enumerate() {
                let likelihood = ensemble
                    .replicas()
                    .get(draw_index)
                    .cloned()
                    .unwrap_or_else(|| self.likelihood().clone());
                let draw_context = Yield::with_ensemble(
                    likelihood,
                    self.term_name(),
                    self.generated_mc().clone(),
                    parameters.clone(),
                    None,
                )?;
                let draw_results =
                    draw_context.projection_set_with_components(&draw_requests, components)?;
                for (result, (_, draw)) in unique_results.iter_mut().zip(draw_results.entries) {
                    result.source_id = Some(ensemble.source_id());
                    result.has_replica_datasets = !ensemble.replicas().is_empty();
                    result.selected_draws.push(draw.selected);
                    result.accepted_draws.push(draw.accepted);
                    result.generated_draws.push(draw.generated);
                    for (name, draw_component) in draw.components {
                        let component = result.components.get_mut(&name).ok_or_else(|| {
                            invalid(format!("missing component `{name}` in paired draw"))
                        })?;
                        component.accepted_draws.push(draw_component.accepted);
                        component.generated_draws.push(draw_component.generated);
                        component.source_id = Some(ensemble.source_id());
                    }
                }
            }
        }
        Ok(YieldProjectionSet {
            entries: projections
                .iter()
                .zip(indexes)
                .map(|(request, index)| (request.name().to_owned(), unique_results[index].clone()))
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

struct PreparedProjection {
    name: String,
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    volumes: Vec<f64>,
    data_bins: BinAssignments,
    accepted_bins: BinAssignments,
    generated_bins: BinAssignments,
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
    fn support_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.count];
        for index in self.indices.iter().flatten() {
            counts[*index] += 1;
        }
        counts
    }

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
        let worker_count = rayon::current_num_threads().min(intensities.len());
        if rayon::current_thread_index().is_none() || worker_count < 2 {
            for (row, &intensity) in intensities.iter().enumerate() {
                let event = offset + row;
                if let Some(index) = self.indices[event] {
                    bins[index] += weights[event] * intensity;
                }
            }
            return;
        }
        let chunk_size = intensities.len().div_ceil(worker_count);
        let chunk_count = intensities.len().div_ceil(chunk_size);
        let partials = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(intensities.len());
                let mut partial = vec![0.0; self.count];
                for (row, &intensity) in intensities[start..end].iter().enumerate() {
                    let event = offset + start + row;
                    if let Some(index) = self.indices[event] {
                        partial[index] += weights[event] * intensity;
                    }
                }
                partial
            })
            .collect::<Vec<_>>();
        for partial in partials {
            for (bin, value) in bins.iter_mut().zip(partial) {
                *bin += value;
            }
        }
    }

    fn accumulate_products(&self, weights: &[f64], intensities: Option<&[f64]>) -> Vec<f64> {
        debug_assert_eq!(self.indices.len(), weights.len());
        debug_assert!(intensities.is_none_or(|values| values.len() == weights.len()));
        let mut bins = vec![0.0; self.count];
        for (event, (&index, &weight)) in self.indices.iter().zip(weights).enumerate() {
            if let Some(index) = index {
                let intensity = intensities.map_or(1.0, |values| values[event]);
                bins[index] += weight * intensity;
            }
        }
        bins
    }

    fn accumulate_products_squared(&self, weights: &[f64]) -> Vec<f64> {
        debug_assert_eq!(self.indices.len(), weights.len());
        let mut bins = vec![0.0; self.count];
        for (&index, &weight) in self.indices.iter().zip(weights) {
            if let Some(index) = index {
                bins[index] += weight * weight;
            }
        }
        bins
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

fn evaluate_bin_assignments_many(
    dataset: &Dataset,
    projections: &[&Projection],
    execution: &Execution,
) -> LikelihoodResult<Vec<BinAssignments>> {
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum CoordinateStatus {
        InRange,
        OutOfRange,
        Nonfinite,
    }
    let events = usize::try_from(dataset.stats()?.events())
        .map_err(|_| invalid("projection event count exceeds addressable memory"))?;
    let mut expression_indexes = HashMap::new();
    let mut expressions = Vec::new();
    let axes_by_projection = projections
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
                    *expression_indexes.entry(key).or_insert_with(|| {
                        let index = expressions.len();
                        expressions.push(axis.expression.clone());
                        index
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut indices = projections
        .iter()
        .map(|_| vec![Some(0usize); events])
        .collect::<Vec<_>>();
    let mut statuses = projections
        .iter()
        .map(|_| vec![CoordinateStatus::InRange; events])
        .collect::<Vec<_>>();
    let mut visited = 0;
    dataset.visit_real_chunks(&expressions, execution, 8192, |offset, coordinates| {
        let chunk_len = coordinates.first().map_or(0, Vec::len);
        if coordinates.len() != expressions.len()
            || coordinates.iter().any(|values| {
                values.len() != chunk_len
                    || offset
                        .checked_add(values.len())
                        .is_none_or(|end| end > events)
            })
        {
            return Err(laddu_runtime::RuntimeError::InvalidShape {
                index: 0,
                message: "projection coordinates do not match the dataset shape".into(),
            });
        }
        visited = offset + chunk_len;
        for (projection_index, projection) in projections.iter().enumerate() {
            let assignments = &mut indices[projection_index];
            let status = &mut statuses[projection_index];
            for (axis, &coordinate_index) in projection
                .axes()
                .iter()
                .zip(&axes_by_projection[projection_index])
            {
                for (row, &value) in coordinates[coordinate_index].iter().enumerate() {
                    let event = offset + row;
                    if !value.is_finite() {
                        status[event] = CoordinateStatus::Nonfinite;
                        assignments[event] = None;
                    } else if status[event] == CoordinateStatus::InRange {
                        if let Some(bin) = axis.binning.index(value, FinalUpperEdge::Exclusive) {
                            assignments[event] = assignments[event]
                                .and_then(|index| index.checked_mul(axis.bins())?.checked_add(bin));
                        } else {
                            status[event] = CoordinateStatus::OutOfRange;
                            assignments[event] = None;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    if visited != events {
        return Err(invalid(
            "projection coordinate count does not match dataset events",
        ));
    }
    projections
        .iter()
        .zip(indices)
        .zip(statuses)
        .map(|((projection, indices), status)| {
            let count = checked_bin_count(
                &projection
                    .axes()
                    .iter()
                    .map(|axis| axis.binning.clone())
                    .collect::<Vec<_>>(),
            )
            .ok_or_else(|| invalid("projection axis shape exceeds addressable bin count"))?;
            Ok(BinAssignments {
                indices,
                count,
                nonfinite_count: status
                    .iter()
                    .filter(|&&value| value == CoordinateStatus::Nonfinite)
                    .count(),
                out_of_range_count: status
                    .iter()
                    .filter(|&&value| value == CoordinateStatus::OutOfRange)
                    .count(),
            })
        })
        .collect()
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
