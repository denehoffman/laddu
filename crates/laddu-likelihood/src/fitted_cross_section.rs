//! Cross sections from the fitted absolute intensity.

use std::{collections::HashMap, sync::Arc};

use laddu_data::data::Dataset;

use crate::measurement::{pool_fitted_binned, pool_fitted_scalar, validate_factor};
use crate::yield_projection::ProjectionKey;
use crate::{
    AreaUnit, Axis, BinnedEstimate, Ensemble, Estimate, Likelihood, LikelihoodError,
    LikelihoodResult, Luminosity, Projection, RateClosure, Yield, YieldBinValidity,
    YieldProjection, next_uncertainty_source_id,
};

fn invalid(message: impl Into<String>) -> LikelihoodError {
    LikelihoodError::InvalidCrossSection(message.into())
}

#[derive(Clone)]
enum Source {
    Single {
        yields: Box<Yield>,
        luminosity: Luminosity,
    },
    Combined {
        members: Vec<(CrossSection, Estimate)>,
        factor_covariance: Option<Vec<Vec<f64>>>,
        factor_source_id: Option<u64>,
    },
}

/// One fitted period and its physical exposure factor.
#[derive(Clone)]
pub struct CombinationMember(pub CrossSection, pub Estimate);

impl From<CrossSection> for CombinationMember {
    fn from(section: CrossSection) -> Self {
        Self(section, Estimate::central(1.0).expect("finite unity"))
    }
}
impl From<(CrossSection, f64)> for CombinationMember {
    fn from((section, factor): (CrossSection, f64)) -> Self {
        Self(
            section,
            Estimate::central(factor)
                .unwrap_or_else(|_| Estimate::central(0.0).expect("finite zero")),
        )
    }
}
impl From<(CrossSection, Estimate)> for CombinationMember {
    fn from((section, factor): (CrossSection, Estimate)) -> Self {
        Self(section, factor)
    }
}

/// An absolute-rate cross section from the fitted generated-MC intensity.
///
/// The total is `G / luminosity`; observed-data normalization is never applied.
#[derive(Clone)]
pub struct CrossSection {
    source: Arc<Source>,
    total: Estimate,
    accepted: Option<Estimate>,
    generated: Option<Estimate>,
    data: Option<Estimate>,
    unit: AreaUnit,
    total_effective_exposure: f64,
    prepared_projections: Option<Arc<PreparedProjections>>,
}

#[derive(Clone)]
struct PreparedProjections {
    requests: Vec<(String, ProjectionKey)>,
    components: HashMap<String, Vec<String>>,
    results: Vec<(String, CrossSectionProjection)>,
}

fn projection_keys(projections: &[Projection]) -> Vec<(String, ProjectionKey)> {
    projections
        .iter()
        .map(|request| {
            (
                request.name().to_owned(),
                ProjectionKey::new(request.axes()),
            )
        })
        .collect()
}

impl std::fmt::Debug for CrossSection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CrossSection")
            .field("total", &self.total)
            .field("unit", &self.unit)
            .field("total_effective_exposure", &self.total_effective_exposure)
            .finish_non_exhaustive()
    }
}

impl CrossSection {
    /// Construct from one extended-likelihood intensity term.
    ///
    /// # Errors
    /// Returns an error for a shape-only term or invalid fit, MC, or ensemble inputs.
    pub fn new(
        likelihood: Arc<Likelihood>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: Luminosity,
        parameters: Vec<f64>,
        ensemble: Option<Ensemble>,
    ) -> LikelihoodResult<Self> {
        let yields =
            Yield::with_ensemble(likelihood, term_name, generated_mc, parameters, ensemble)?;
        Self::from_yields(yields, luminosity)
    }

    fn from_yields(yields: Yield, luminosity: Luminosity) -> LikelihoodResult<Self> {
        if !yields.has_absolute_rate() {
            return Err(LikelihoodError::AbsoluteRateUnavailable(
                "cross sections require an extended intensity likelihood".to_owned(),
            ));
        }
        let accepted = yields.accepted_fitted_yield()?;
        let generated = yields.generated_fitted_yield()?;
        let data = yields.selected_yield().clone();
        let total = generated.divided_by_luminosity(&luminosity);
        Ok(Self {
            source: Arc::new(Source::Single {
                yields: Box::new(yields),
                luminosity: luminosity.clone(),
            }),
            total,
            accepted: Some(accepted),
            generated: Some(generated),
            data: Some(data),
            unit: luminosity.unit(),
            total_effective_exposure: luminosity.value(),
            prepared_projections: None,
        })
    }

    /// Construct totals and a fixed set of projections using shared MC passes.
    ///
    /// The requested projection outputs are retained for matching `project_many`
    /// calls and shared by clones. Other requests are evaluated normally without
    /// extending the retained set. Integral and event buffers are not retained.
    ///
    /// # Errors
    /// Returns an error for invalid rate, parameters, ensemble, or projections.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_projections(
        likelihood: Arc<Likelihood>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: Luminosity,
        parameters: Vec<f64>,
        ensemble: Option<Ensemble>,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<Self> {
        let (yields, projected) = Yield::with_projections(
            likelihood,
            term_name.into(),
            generated_mc,
            parameters,
            ensemble,
            projections,
            components,
        )?;
        let results = projected
            .into_entries()
            .into_iter()
            .map(|(name, value)| {
                Ok((
                    name,
                    CrossSectionProjection::from_yield_projection(value, &luminosity)?,
                ))
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let mut section = Self::from_yields(yields, luminosity)?;
        section.prepared_projections = Some(Arc::new(PreparedProjections {
            requests: projection_keys(projections),
            components: components.clone(),
            results,
        }));
        Ok(section)
    }

    /// Fitted generated-space intensity divided by integrated luminosity.
    pub fn total(&self) -> &Estimate {
        &self.total
    }

    /// Area unit of the returned cross section.
    pub fn unit(&self) -> AreaUnit {
        self.unit
    }

    /// Sum of effective exposures, including physical factors for combined periods.
    pub fn total_effective_exposure(&self) -> f64 {
        self.total_effective_exposure
    }

    /// Raw accepted-MC intensity integral for a single fitted period.
    pub fn accepted_integral(&self) -> Option<&Estimate> {
        self.accepted.as_ref()
    }

    /// Raw generated-MC intensity integral for a single fitted period.
    pub fn generated_integral(&self) -> Option<&Estimate> {
        self.generated.as_ref()
    }

    /// Observed weighted signal yield for a single fitted period.
    pub fn data_yield(&self) -> Option<&Estimate> {
        self.data.as_ref()
    }

    /// Accepted fitted yield compared with the observed yield.
    pub fn rate_closure(&self) -> Option<RateClosure> {
        match self.source.as_ref() {
            Source::Single { yields, .. } => Some(yields.rate_closure()),
            Source::Combined { .. } => None,
        }
    }

    /// Physical members of a combined result, if any.
    pub fn members(&self) -> Option<&[(CrossSection, Estimate)]> {
        match self.source.as_ref() {
            Source::Single { .. } => None,
            Source::Combined { members, .. } => Some(members),
        }
    }

    /// Project the fitted generated-MC intensity into one joint bin grid.
    ///
    /// # Errors
    /// Returns an error for invalid axes, selections, or model evaluation.
    pub fn project(
        &self,
        axes: &[Axis],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<CrossSectionProjection> {
        let request = Projection::new("cross_section", axes.to_vec())?;
        Ok(self.project_many(&[request], components)?.remove(0).1)
    }

    /// Evaluate named projections in one pass per dataset and period.
    ///
    /// # Errors
    /// Returns an error for empty requests, incompatible members, or evaluation failure.
    pub fn project_many(
        &self,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<Vec<(String, CrossSectionProjection)>> {
        if projections.is_empty() {
            return Err(invalid("at least one projection is required"));
        }
        if let Some(prepared) = &self.prepared_projections
            && prepared.components == *components
            && prepared.requests == projection_keys(projections)
        {
            return Ok(prepared.results.clone());
        }
        match self.source.as_ref() {
            Source::Single { yields, luminosity } => yields
                .projection_set_with_components(projections, components)?
                .into_entries()
                .into_iter()
                .map(|(name, value)| {
                    Ok((
                        name,
                        CrossSectionProjection::from_yield_projection(value, luminosity)?,
                    ))
                })
                .collect(),
            Source::Combined {
                members,
                factor_covariance,
                factor_source_id,
            } => {
                let mut grouped = projections
                    .iter()
                    .map(|request| (request.name().to_owned(), Vec::with_capacity(members.len())))
                    .collect::<Vec<_>>();
                for (member, factor) in members {
                    let Source::Single { luminosity, .. } = member.source.as_ref() else {
                        return Err(invalid("nested period combinations are not supported"));
                    };
                    for (slot, (_, projection)) in grouped
                        .iter_mut()
                        .zip(member.project_many(projections, components)?)
                    {
                        slot.1
                            .push((projection, luminosity.clone(), factor.clone()));
                    }
                }
                grouped
                    .into_iter()
                    .map(|(name, members)| {
                        Ok((
                            name,
                            CrossSectionProjection::combine(
                                &members,
                                factor_covariance.as_deref(),
                                *factor_source_id,
                            )?,
                        ))
                    })
                    .collect()
            }
        }
    }

    /// Pool fitted intensities using physical exposure or branching factors.
    ///
    /// # Errors
    /// Returns an error for empty or incompatible members or invalid factors.
    pub fn combine<M: Clone + Into<CombinationMember>>(members: &[M]) -> LikelihoodResult<Self> {
        Self::combine_impl(members, None)
    }

    /// Pool fitted intensities with a known covariance between physical factors.
    ///
    /// # Errors
    /// Returns an error for invalid covariance, factors, or member compatibility.
    pub fn combine_with_covariance<M: Clone + Into<CombinationMember>>(
        members: &[M],
        covariance: Vec<Vec<f64>>,
    ) -> LikelihoodResult<Self> {
        Self::combine_impl(members, Some(covariance))
    }

    fn combine_impl<M: Clone + Into<CombinationMember>>(
        inputs: &[M],
        factor_covariance: Option<Vec<Vec<f64>>>,
    ) -> LikelihoodResult<Self> {
        let members = inputs
            .iter()
            .cloned()
            .map(|input| {
                let CombinationMember(section, factor) = input.into();
                validate_factor(&factor)?;
                Ok((section, factor))
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let Some((first, _)) = members.first() else {
            return Err(invalid("at least one period is required"));
        };
        if members.iter().any(|(member, _)| member.unit != first.unit) {
            return Err(invalid("combined periods must share one area unit"));
        }
        let unit = first.unit;
        let sources = members
            .iter()
            .map(|(member, factor)| {
                let Source::Single { luminosity, .. } = member.source.as_ref() else {
                    return Err(invalid("nested period combinations are not supported"));
                };
                Ok((
                    member
                        .generated
                        .as_ref()
                        .expect("single fitted period has generated integral"),
                    luminosity,
                    factor,
                ))
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        let factor_source_id = (factor_covariance.is_some()
            || members
                .iter()
                .any(|(_, factor)| factor.standard_error().is_some()))
        .then(next_uncertainty_source_id);
        let total = pool_fitted_scalar(&sources, factor_covariance.as_deref(), factor_source_id)?;
        let total_effective_exposure = sources
            .iter()
            .map(|(_, luminosity, factor)| luminosity.value() * factor.value())
            .sum();
        Ok(Self {
            source: Arc::new(Source::Combined {
                members,
                factor_covariance,
                factor_source_id,
            }),
            total,
            accepted: None,
            generated: None,
            data: None,
            unit,
            total_effective_exposure,
            prepared_projections: None,
        })
    }
}

/// A differential projection of the fitted generated-MC intensity.
#[derive(Clone, Debug)]
pub struct CrossSectionProjection {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    total: BinnedEstimate,
    components: HashMap<String, BinnedEstimate>,
    member_validity: Vec<Vec<YieldBinValidity>>,
}

impl CrossSectionProjection {
    fn from_yield_projection(
        value: YieldProjection,
        luminosity: &Luminosity,
    ) -> LikelihoodResult<Self> {
        let total = value
            .generated_estimate()
            .divided_by_luminosity_and_measure(luminosity, value.bin_volumes())?;
        Ok(Self {
            axes: value.axes().to_vec(),
            shape: value.shape().to_vec(),
            total,
            components: value
                .components()
                .iter()
                .map(|(name, component)| {
                    Ok((
                        name.clone(),
                        component
                            .generated_estimate()
                            .divided_by_luminosity_and_measure(
                                luminosity,
                                component.bin_volumes(),
                            )?,
                    ))
                })
                .collect::<LikelihoodResult<_>>()?,
            member_validity: vec![value.validity().to_vec()],
        })
    }

    fn combine(
        members: &[(Self, Luminosity, Estimate)],
        factor_covariance: Option<&[Vec<f64>]>,
        factor_source_id: Option<u64>,
    ) -> LikelihoodResult<Self> {
        let Some((first, _, _)) = members.first() else {
            return Err(invalid("at least one projected period is required"));
        };
        if members.iter().any(|(projection, _, _)| {
            projection.axes != first.axes
                || projection.shape != first.shape
                || projection
                    .components
                    .keys()
                    .collect::<std::collections::HashSet<_>>()
                    != first.components.keys().collect()
        }) {
            return Err(invalid(
                "combined projections must share geometry and component selections",
            ));
        }
        let total_sources = members
            .iter()
            .map(|(projection, luminosity, factor)| (&projection.total, luminosity, factor))
            .collect::<Vec<_>>();
        let total = pool_fitted_binned(&total_sources, factor_covariance, factor_source_id)?;
        let components = first
            .components
            .keys()
            .map(|name| {
                let sources = members
                    .iter()
                    .map(|(projection, luminosity, factor)| {
                        (&projection.components[name], luminosity, factor)
                    })
                    .collect::<Vec<_>>();
                Ok((
                    name.clone(),
                    pool_fitted_binned(&sources, factor_covariance, factor_source_id)?,
                ))
            })
            .collect::<LikelihoodResult<HashMap<_, _>>>()?;
        Ok(Self {
            axes: first.axes.clone(),
            shape: first.shape.clone(),
            total,
            components,
            member_validity: members
                .iter()
                .flat_map(|(projection, _, _)| projection.member_validity.clone())
                .collect(),
        })
    }

    /// Ordered axis edges.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }

    /// Row-major bin shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Generated-space differential cross section.
    pub fn total(&self) -> &BinnedEstimate {
        &self.total
    }

    /// Coherent selected-model differential cross sections.
    pub fn components(&self) -> &HashMap<String, BinnedEstimate> {
        &self.components
    }

    /// Per-period bin support diagnostics.
    pub fn member_validity(&self) -> &[Vec<YieldBinValidity>] {
        &self.member_validity
    }
}

impl Likelihood {
    /// Construct a fitted cross section with shared total/projection evaluation.
    ///
    /// Requested outputs are retained as a fixed set; matching subsequent
    /// `project_many` calls reuse them. See [`CrossSection::new_with_projections`].
    ///
    /// # Errors
    /// Returns an error for invalid rate, parameters, ensemble, or projections.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_section_with_projections(
        self: &Arc<Self>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: Luminosity,
        parameters: Vec<f64>,
        ensemble: Option<Ensemble>,
        projections: &[Projection],
        components: &HashMap<String, Vec<String>>,
    ) -> LikelihoodResult<CrossSection> {
        CrossSection::new_with_projections(
            Arc::clone(self),
            term_name,
            generated_mc,
            luminosity,
            parameters,
            ensemble,
            projections,
            components,
        )
    }

    /// Prepare the sole cross-section pathway for an absolute-rate term.
    ///
    /// # Errors
    /// Returns an error for a shape-only term or invalid fit, MC, or ensemble inputs.
    pub fn cross_section(
        self: &Arc<Self>,
        term_name: impl Into<String>,
        generated_mc: Dataset,
        luminosity: Luminosity,
        parameters: Vec<f64>,
        ensemble: Option<Ensemble>,
    ) -> LikelihoodResult<CrossSection> {
        CrossSection::new(
            Arc::clone(self),
            term_name,
            generated_mc,
            luminosity,
            parameters,
            ensemble,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorBudget;
    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_data::{
        data::{EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{Expr, event_scalar, parameter};

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
    fn prepared_projections_preserve_paired_totals_outside_requested_axes() {
        let model = CompiledModel::from_expr(
            &Expr::from(parameter!("scale", initial: 1.0))
                .tagged("signal")
                .norm_sqr(),
        )
        .unwrap();
        let data = dataset(&[(0.0, 1.0), (2.0, 2.0)]);
        let accepted = dataset(&[(0.0, 1.0), (2.0, 2.0)]);
        let generated = dataset(&[(0.0, 1.0), (2.0, 2.0), (4.0, 3.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let ensemble = Ensemble::bootstrap_fit(&likelihood, 2, 13, |_replica, index| {
            Ok::<_, std::convert::Infallible>(vec![1.0 + index as f64])
        })
        .unwrap();
        let requests = [Projection::new(
            "narrow",
            vec![Axis::new(event_scalar("x"), vec![-0.5, 0.5]).unwrap()],
        )
        .unwrap()];
        let components = HashMap::from([("signal".to_owned(), vec!["signal".to_owned()])]);
        let luminosity = Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap();
        let separate = likelihood
            .cross_section(
                "signal",
                generated.clone(),
                luminosity.clone(),
                vec![1.0],
                Some(ensemble.clone()),
            )
            .unwrap();
        let prepared = likelihood
            .cross_section_with_projections(
                "signal",
                generated,
                luminosity,
                vec![1.0],
                Some(ensemble),
                &requests,
                &components,
            )
            .unwrap();
        assert_eq!(prepared.total().value(), 3.0);
        assert_eq!(prepared.total().draws(), &[3.0, 12.0]);
        assert_eq!(prepared.total().source_id(), separate.total().source_id());
        assert_eq!(
            prepared.data_yield().unwrap().draws(),
            separate.data_yield().unwrap().draws()
        );
        assert_eq!(prepared.rate_closure(), separate.rate_closure());
        assert_relative_eq!(
            prepared
                .total()
                .error_with_budget(ErrorBudget::default())
                .unwrap()
                .error(),
            separate
                .total()
                .error_with_budget(ErrorBudget::default())
                .unwrap()
                .error()
        );
        let expected = separate.project_many(&requests, &components).unwrap();
        let actual = prepared.project_many(&requests, &components).unwrap();
        assert_eq!(actual[0].1.total().values(), expected[0].1.total().values());
        assert_eq!(actual[0].1.total().draws(), expected[0].1.total().draws());
        assert_eq!(
            actual[0].1.components()["signal"].values(),
            expected[0].1.components()["signal"].values()
        );
        let factor = Estimate::central(0.5).unwrap();
        let a = CrossSection::combine_with_covariance(
            &[
                (prepared.clone(), Estimate::central(1.0).unwrap()),
                (prepared, factor.clone()),
            ],
            vec![vec![0.01, 0.002], vec![0.002, 0.01]],
        )
        .unwrap();
        let b = CrossSection::combine_with_covariance(
            &[
                (separate.clone(), Estimate::central(1.0).unwrap()),
                (separate, factor),
            ],
            vec![vec![0.01, 0.002], vec![0.002, 0.01]],
        )
        .unwrap();
        assert_eq!(a.total_effective_exposure(), 3.0);
        assert_eq!(a.total().draws(), b.total().draws());
        assert_relative_eq!(
            a.total()
                .error_with_budget(ErrorBudget::default())
                .unwrap()
                .error(),
            b.total()
                .error_with_budget(ErrorBudget::default())
                .unwrap()
                .error()
        );
        let a = a.project_many(&requests, &components).unwrap();
        let b = b.project_many(&requests, &components).unwrap();
        assert_eq!(a[0].1.total().draws(), b[0].1.total().draws());
        assert_eq!(
            a[0].1.components()["signal"].draws(),
            b[0].1.components()["signal"].draws()
        );
    }

    #[test]
    fn prepared_totals_preserve_arbitrary_replica_rows_and_errors() {
        let model = CompiledModel::from_expr(
            &(Expr::from(parameter!("scale", initial: 1.0)) * (event_scalar("x") + 1.0)),
        )
        .unwrap();
        let data = dataset(&[(0.0, 1.0)]);
        let central = Arc::new(
            Likelihood::new([crate::ExtendedNllTerm::new("signal", &model, &data, &data).unwrap()])
                .unwrap(),
        );
        let replica_data = dataset(&[(2.0, 3.0)]);
        let replica = Arc::new(
            Likelihood::new([crate::ExtendedNllTerm::new(
                "signal",
                &model,
                &replica_data,
                &replica_data,
            )
            .unwrap()])
            .unwrap(),
        );
        let ensemble =
            Ensemble::with_replicas(vec!["scale".into()], vec![vec![2.0]], vec![replica]).unwrap();
        let requests = [Projection::new(
            "x",
            vec![Axis::new(event_scalar("x"), vec![-0.5, 0.5]).unwrap()],
        )
        .unwrap()];
        let luminosity = Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap();
        let generated = dataset(&[(0.0, 1.0), (2.0, 2.0)]);
        let prepared = central
            .cross_section_with_projections(
                "signal",
                generated.clone(),
                luminosity.clone(),
                vec![1.0],
                Some(ensemble.clone()),
                &requests,
                &HashMap::new(),
            )
            .unwrap();
        let separate = central
            .cross_section(
                "signal",
                generated,
                luminosity.clone(),
                vec![1.0],
                Some(ensemble),
            )
            .unwrap();
        assert_eq!(prepared.total().value(), 3.5);
        assert_eq!(prepared.total().draws(), &[7.0]);
        assert_eq!(prepared.accepted_integral().unwrap().draws(), &[18.0]);
        assert_eq!(prepared.data_yield().unwrap().draws(), &[3.0]);
        assert_eq!(prepared.rate_closure(), separate.rate_closure());
        let a = prepared.project_many(&requests, &HashMap::new()).unwrap();
        let b = separate.project_many(&requests, &HashMap::new()).unwrap();
        assert_eq!(a[0].1.total().draws(), b[0].1.total().draws());
        for (generated, expected) in [
            (dataset(&[]), "missing"),
            (dataset(&[(0.0, 0.0)]), "zero"),
            (dataset(&[(-2.0, 1.0)]), "intensity"),
        ] {
            let error = central
                .cross_section_with_projections(
                    "signal",
                    generated,
                    luminosity.clone(),
                    vec![1.0],
                    None,
                    &requests,
                    &HashMap::new(),
                )
                .unwrap_err();
            assert!(
                match expected {
                    "missing" => matches!(error, LikelihoodError::MissingGeneratedSupport),
                    "zero" => matches!(error, LikelihoodError::ZeroGeneratedIntegral),
                    _ => matches!(
                        error,
                        LikelihoodError::NonPositiveIntensity {
                            dataset: "generated MC",
                            ..
                        }
                    ),
                },
                "{expected}: {error}"
            );
        }
    }

    #[test]
    fn prepared_totals_preserve_compensated_signed_weights() {
        let model = CompiledModel::from_expr(&Expr::from(1.0).tagged("signal")).unwrap();
        let data = dataset(&[(0.0, 1e16), (0.0, 1.0), (0.0, 1.0), (0.0, -1e16)]);
        let accepted = dataset(&[(0.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let replica_data = dataset(&[
            (0.0, 1e16),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, -1e16),
        ]);
        let replica = Arc::new(
            Likelihood::new([crate::ExtendedNllTerm::new(
                "signal",
                &model,
                &replica_data,
                &accepted,
            )
            .unwrap()])
            .unwrap(),
        );
        let ensemble = Ensemble::with_replicas(vec![], vec![vec![]], vec![replica]).unwrap();
        let requests = [Projection::new(
            "constant",
            vec![Axis::new(0.0.into(), vec![-0.5, 0.5]).unwrap()],
        )
        .unwrap()];
        let components = HashMap::from([("signal".to_owned(), vec!["signal".to_owned()])]);
        let luminosity = Luminosity::new(1.0, AreaUnit::Nanobarn).unwrap();
        let generated = dataset(&[(0.0, 1e16), (0.0, 1.0), (0.0, -1e16), (0.0, 1.0)]);
        let prepared = likelihood
            .cross_section_with_projections(
                "signal",
                generated.clone(),
                luminosity.clone(),
                vec![],
                Some(ensemble.clone()),
                &requests,
                &components,
            )
            .unwrap();
        let separate = likelihood
            .cross_section("signal", generated, luminosity, vec![], Some(ensemble))
            .unwrap();
        assert_eq!(
            prepared.data_yield().unwrap().value(),
            separate.data_yield().unwrap().value()
        );
        assert_eq!(
            prepared.data_yield().unwrap().draws(),
            separate.data_yield().unwrap().draws()
        );
        assert_eq!(prepared.rate_closure(), separate.rate_closure());
        assert_eq!(prepared.total().value(), 2.0);
        assert_eq!(prepared.total().draws(), separate.total().draws());
        let actual = prepared.project_many(&requests, &components).unwrap();
        let expected = separate.project_many(&requests, &components).unwrap();
        assert_eq!(actual[0].1.total().values(), expected[0].1.total().values());
        assert_eq!(actual[0].1.total().draws(), expected[0].1.total().draws());
        assert_eq!(
            actual[0].1.components()["signal"].values(),
            expected[0].1.components()["signal"].values()
        );
    }

    #[test]
    fn prepared_scalar_fill_budgets_match_for_native_and_general_normalization() {
        use laddu_runtime::{Execution, ExecutionOptions, NormalizationMode};
        let model = CompiledModel::from_expr(
            &(Expr::from(parameter!("scale", initial: 1.0)) * event_scalar("x")).norm_sqr(),
        )
        .unwrap();
        let data = dataset(&[(1.0, 1.0), (2.0, 2.0)]);
        let generated = dataset(&[(1.0, 3.0), (3.0, 2.0)]);
        let requests = [Projection::new(
            "x",
            vec![Axis::new(event_scalar("x"), vec![0.0, 2.0]).unwrap()],
        )
        .unwrap()];
        let budget = ErrorBudget {
            data_fill: true,
            accepted_mc_fill: true,
            generated_mc_fill: true,
            ensemble: false,
            luminosity: true,
            ..ErrorBudget::default()
        };
        for normalization in [NormalizationMode::Auto, NormalizationMode::General] {
            let execution = Execution::local(ExecutionOptions {
                normalization,
                ..ExecutionOptions::default()
            })
            .unwrap();
            let likelihood = Arc::new(
                Likelihood::with_execution(
                    [crate::ExtendedNllTerm::new("signal", &model, &data, &data).unwrap()],
                    &execution,
                )
                .unwrap(),
            );
            let luminosity = Luminosity::new(2.0, AreaUnit::Nanobarn)
                .unwrap()
                .with_relative_uncertainty(0.1, 77)
                .unwrap();
            let prepared = likelihood
                .cross_section_with_projections(
                    "signal",
                    generated.clone(),
                    luminosity.clone(),
                    vec![1.0],
                    None,
                    &requests,
                    &HashMap::new(),
                )
                .unwrap();
            let separate = likelihood
                .cross_section("signal", generated.clone(), luminosity, vec![1.0], None)
                .unwrap();
            assert_eq!(prepared.total().value(), 10.5);
            for (estimate, variance) in [
                (prepared.data_yield().unwrap(), 5.0),
                (prepared.accepted_integral().unwrap(), 65.0),
                (prepared.generated_integral().unwrap(), 333.0),
                (prepared.total(), 84.3525),
            ] {
                assert_relative_eq!(
                    estimate.error_with_budget(budget).unwrap().error().powi(2),
                    variance,
                    epsilon = 1e-12
                );
            }
            assert_relative_eq!(
                prepared.total().error_with_budget(budget).unwrap().error(),
                separate.total().error_with_budget(budget).unwrap().error()
            );
        }
    }

    #[test]
    fn coherent_generated_intensity_sets_total_and_bin_widths() {
        let a = (Expr::from(parameter!("a", initial: 1.0)) * event_scalar("x")).tagged("a");
        let b = Expr::from(parameter!("b", initial: 1.0)).tagged("b");
        let model = CompiledModel::from_expr(&(a + b).norm_sqr()).unwrap();
        let data = dataset(&[(0.0, 1.0), (1.0, 4.0)]);
        let accepted = dataset(&[(0.0, 1.0), (1.0, 1.0)]);
        let generated = dataset(&[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let section = likelihood
            .cross_section(
                "signal",
                generated,
                Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap(),
                likelihood.default_params(),
                None,
            )
            .unwrap();
        assert_relative_eq!(section.accepted_integral().unwrap().value(), 5.0);
        assert_relative_eq!(section.generated_integral().unwrap().value(), 14.0);
        assert_relative_eq!(section.data_yield().unwrap().value(), 5.0);
        assert_relative_eq!(section.total().value(), 7.0);
        assert!(section.rate_closure().unwrap().is_closed());
        let axis = Axis::new(event_scalar("x"), vec![-0.5, 0.5, 2.5]).unwrap();
        let projected = section
            .project(
                &[axis],
                &HashMap::from([
                    ("a".into(), vec!["a".into()]),
                    ("b".into(), vec!["b".into()]),
                    ("coherent".into(), vec!["a".into(), "b".into()]),
                ]),
            )
            .unwrap();
        assert_relative_eq!(projected.total().values()[0], 0.5);
        assert_relative_eq!(projected.total().values()[1], 3.25);
        assert_eq!(
            projected.components()["coherent"].values(),
            projected.total().values()
        );
        assert_relative_eq!(projected.components()["a"].values()[1], 1.25);
        assert_relative_eq!(projected.components()["b"].values()[1], 0.5);
    }

    #[test]
    fn low_acceptance_keeps_the_fitted_rate_above_flat_mc() {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0).norm_sqr()).unwrap();
        let data = dataset(&[(0.0, 1.0)]);
        let accepted = dataset(&[(0.0, 1.0)]);
        let generated = dataset(&[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let section = likelihood
            .cross_section(
                "signal",
                generated,
                Luminosity::new(1.0, AreaUnit::Nanobarn).unwrap(),
                vec![],
                None,
            )
            .unwrap();
        assert_relative_eq!(section.total().value(), 14.0);
        assert!(section.total().value() > 3.0); // D divided by flat-MC acceptance 1/3.
        assert_relative_eq!(section.accepted_integral().unwrap().value(), 1.0);
        assert!(section.rate_closure().unwrap().is_closed());
    }

    #[test]
    fn generated_projection_keeps_empty_bins_zero_and_batches_requests() {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0).norm_sqr()).unwrap();
        let data = dataset(&[(0.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::ExtendedNllTerm::new("signal", &model, &data, &data).unwrap()])
                .unwrap(),
        );
        let section = likelihood
            .cross_section(
                "signal",
                dataset(&[(0.0, 1.0), (1.0, 1.0)]),
                Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap(),
                vec![],
                None,
            )
            .unwrap();
        let fine = Axis::new(event_scalar("x"), vec![-0.5, 0.5, 1.5, 2.5]).unwrap();
        let coarse = Axis::new(event_scalar("x"), vec![-0.5, 1.5, 2.5]).unwrap();
        let results = section
            .project_many(
                &[
                    Projection::new("fine", vec![fine.clone()]).unwrap(),
                    Projection::new("coarse", vec![coarse]).unwrap(),
                ],
                &HashMap::new(),
            )
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            ["fine", "coarse"]
        );
        assert_eq!(results[0].1.total().values(), &[0.5, 2.0, 0.0]);
        assert_eq!(results[1].1.total().values(), &[1.25, 0.0]);
        assert_eq!(
            results[0].1.total().values(),
            section
                .project(&[fine], &HashMap::new())
                .unwrap()
                .total()
                .values()
        );
        assert_eq!(
            results[0].1.member_validity()[0][2],
            YieldBinValidity::MissingGeneratedSupport
        );
    }

    #[test]
    fn shape_only_term_has_no_cross_section() {
        let model = CompiledModel::from_expr(&Expr::from(1.0)).unwrap();
        let data = dataset(&[(0.0, 1.0)]);
        let accepted = dataset(&[(0.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::NllTerm::new("shape", &model, &data, &accepted).unwrap()])
                .unwrap(),
        );
        let error = likelihood
            .cross_section(
                "shape",
                dataset(&[(0.0, 1.0)]),
                Luminosity::new(1.0, AreaUnit::Nanobarn).unwrap(),
                vec![],
                None,
            )
            .unwrap_err();
        assert!(matches!(error, LikelihoodError::AbsoluteRateUnavailable(_)));
    }

    #[test]
    fn rate_mismatch_is_reported_without_rescaling_generated_intensity() {
        let model =
            CompiledModel::from_expr(&Expr::from(parameter!("scale", initial: 1.0)).norm_sqr())
                .unwrap();
        let data = dataset(&[(0.0, 4.0)]);
        let accepted = dataset(&[(0.0, 1.0)]);
        let generated = dataset(&[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        // The unconstrained extended-rate optimum is scale = sqrt(D / sum w_acc) = 2.
        let optimum = likelihood
            .cross_section(
                "signal",
                generated.clone(),
                Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap(),
                vec![2.0],
                None,
            )
            .unwrap();
        assert!(optimum.rate_closure().unwrap().is_closed());
        assert_relative_eq!(optimum.total().value(), 6.0);

        let displaced = likelihood
            .cross_section(
                "signal",
                generated,
                Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap(),
                vec![1.0],
                None,
            )
            .unwrap();
        let closure = displaced.rate_closure().unwrap();
        assert_eq!(closure.status(), crate::RateClosureStatus::Failed);
        assert_relative_eq!(closure.accepted_residual().unwrap(), -3.0);
        assert_relative_eq!(displaced.total().value(), 1.5);
    }

    #[test]
    fn paired_bootstrap_draws_follow_the_fitted_generated_integral() {
        let model =
            CompiledModel::from_expr(&Expr::from(parameter!("scale", initial: 1.0)).norm_sqr())
                .unwrap();
        let data = dataset(&[(0.0, 1.0)]);
        let accepted = dataset(&[(0.0, 1.0)]);
        let generated = dataset(&[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([
                crate::ExtendedNllTerm::new("signal", &model, &data, &accepted).unwrap(),
            ])
            .unwrap(),
        );
        let ensemble = Ensemble::bootstrap_fit(&likelihood, 2, 13, |_replica, index| {
            Ok::<_, std::convert::Infallible>(vec![1.0 + index as f64])
        })
        .unwrap();
        let source_id = ensemble.source_id();
        let section = likelihood
            .cross_section(
                "signal",
                generated,
                Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap(),
                vec![1.0],
                Some(ensemble),
            )
            .unwrap();
        assert_eq!(section.total().source_id(), Some(source_id));
        assert_eq!(section.total().draws(), &[1.5, 6.0]);
        assert_eq!(section.generated_integral().unwrap().draws(), &[3.0, 12.0]);
    }

    #[test]
    fn physical_exposure_combination_keeps_fitted_bins_and_mc_uncertainty() {
        let model = CompiledModel::from_expr(&(event_scalar("x") + 1.0).norm_sqr()).unwrap();
        let data = dataset(&[(0.0, 1.0)]);
        let generated = dataset(&[(0.0, 1.0), (1.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::ExtendedNllTerm::new("signal", &model, &data, &data).unwrap()])
                .unwrap(),
        );
        let first = likelihood
            .cross_section(
                "signal",
                generated.clone(),
                Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap(),
                vec![],
                None,
            )
            .unwrap();
        let second = likelihood
            .cross_section(
                "signal",
                generated,
                Luminosity::new(4.0, AreaUnit::Nanobarn).unwrap(),
                vec![],
                None,
            )
            .unwrap();
        let combined = CrossSection::combine(&[(first, 1.0), (second, 0.5)]).unwrap();
        assert_relative_eq!(combined.total_effective_exposure(), 4.0);
        assert_relative_eq!(combined.total().value(), 2.5);
        let error = combined
            .total()
            .error_with_budget(crate::ErrorBudget::default())
            .unwrap();
        assert!(
            error
                .included()
                .contains(&crate::ErrorComponent::GeneratedMcFill)
        );
        let axis = Axis::new(event_scalar("x"), vec![-0.5, 0.5, 1.5]).unwrap();
        let projected = combined.project(&[axis], &HashMap::new()).unwrap();
        assert_relative_eq!(projected.total().values()[0], 0.5);
        assert_relative_eq!(projected.total().values()[1], 2.0);
        assert_eq!(projected.member_validity().len(), 2);
    }

    #[test]
    fn physical_factor_errors_and_covariance_propagate_without_draws() {
        let known_a = Estimate::central(0.6)
            .unwrap()
            .with_standard_error(0.02)
            .unwrap();
        let known_b = Estimate::central(0.3)
            .unwrap()
            .with_standard_error(0.01)
            .unwrap();
        assert_relative_eq!(
            known_a
                .checked_add(&known_b)
                .unwrap()
                .std()
                .unwrap()
                .powi(2),
            0.0005
        );
        let model = CompiledModel::from_expr(&Expr::from(1.0).norm_sqr()).unwrap();
        let data = dataset(&[(0.0, 1.0)]);
        let generated = dataset(&[(0.0, 1.0), (1.0, 1.0)]);
        let likelihood = Arc::new(
            Likelihood::new([crate::ExtendedNllTerm::new("signal", &model, &data, &data).unwrap()])
                .unwrap(),
        );
        let first = likelihood
            .cross_section(
                "signal",
                generated.clone(),
                Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap(),
                vec![],
                None,
            )
            .unwrap();
        let second = likelihood
            .cross_section(
                "signal",
                generated,
                Luminosity::new(4.0, AreaUnit::Nanobarn).unwrap(),
                vec![],
                None,
            )
            .unwrap();
        let plain = CrossSection::combine(&[first.clone(), second.clone()]).unwrap();
        assert_relative_eq!(plain.total().value(), 4.0 / 6.0);
        let known = CrossSection::combine(&[
            CombinationMember::from((
                first.clone(),
                Estimate::central(1.0)
                    .unwrap()
                    .with_standard_error(0.1)
                    .unwrap(),
            )),
            CombinationMember::from((
                second.clone(),
                Estimate::central(0.5)
                    .unwrap()
                    .with_standard_error(0.1)
                    .unwrap(),
            )),
        ])
        .unwrap();
        let budget = crate::ErrorBudget {
            data_fill: false,
            accepted_mc_fill: false,
            generated_mc_fill: false,
            branching_exposure: true,
            ..crate::ErrorBudget::default()
        };
        let error = known.total().error_with_budget(budget).unwrap();
        assert!(
            error
                .included()
                .contains(&crate::ErrorComponent::BranchingExposure)
        );
        assert_relative_eq!(known.total().value(), 1.0);
        assert_relative_eq!(error.error().powi(2), 0.0125, epsilon = 1e-12);
        let axis = Axis::new(event_scalar("x"), vec![-0.5, 0.5, 1.5]).unwrap();
        let bins = known.project(&[axis], &HashMap::new()).unwrap();
        let histogram = bins.total().histogram_with_budget(budget).unwrap();
        assert_relative_eq!(histogram.values()[0], 0.5);
        assert_relative_eq!(histogram.errors()[0].powi(2), 0.003125, epsilon = 1e-12);
        assert_eq!(error.sources(), histogram.sources());
        let correlated = CrossSection::combine_with_covariance(
            &[(first, 1.0), (second, 0.5)],
            vec![vec![0.01, 0.005], vec![0.005, 0.04]],
        )
        .unwrap();
        let error = correlated.total().error_with_budget(budget).unwrap();
        assert_relative_eq!(error.error().powi(2), 0.0475, epsilon = 1e-12);
        assert!(
            CrossSection::combine_with_covariance(
                &correlated
                    .members()
                    .unwrap()
                    .iter()
                    .map(|(s, _)| (s.clone(), 1.0))
                    .collect::<Vec<_>>(),
                vec![vec![1.0, 2.0], vec![2.0, 1.0]],
            )
            .is_err()
        );
    }

    #[test]
    fn paired_factor_draws_are_not_counted_again_as_known_factor_error() {
        let generated = Estimate::with_source_id(10.0, vec![9.0, 11.0], Some(7)).unwrap();
        let factor = Estimate::with_source_id(1.0, vec![0.9, 1.1], Some(7)).unwrap();
        let luminosity = Luminosity::new(2.0, AreaUnit::Nanobarn).unwrap();
        let pooled = pool_fitted_scalar(&[(&generated, &luminosity, &factor)], None, None).unwrap();
        assert_relative_eq!(pooled.draws()[0], 5.0);
        assert_relative_eq!(pooled.draws()[1], 5.0);
        let budget = crate::ErrorBudget {
            data_fill: false,
            accepted_mc_fill: false,
            generated_mc_fill: false,
            branching_exposure: true,
            ..crate::ErrorBudget::default()
        };
        let error = pooled.error_with_budget(budget).unwrap();
        assert_eq!(
            error.omitted(),
            &[(crate::ErrorComponent::BranchingExposure, "unavailable")]
        );
    }
}
