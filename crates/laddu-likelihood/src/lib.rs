use std::{collections::HashSet, fmt::Debug, sync::Arc};

use laddu_compile::CompiledModel;
use laddu_data::data::Dataset;
use laddu_expr::parameters::{ParamError, ParamId, ParamLayout, ParamRegistry, ParamValues};
use laddu_runtime::{CpuBackend, CpuCachedDataset, CpuPlan, RuntimeError};
use num::complex::Complex64;
use thiserror::Error;

pub type LikelihoodResult<T> = Result<T, LikelihoodError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LikelihoodName(String);

impl LikelihoodName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Error)]
pub enum LikelihoodError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Params(#[from] ParamError),
    #[error("{dataset} intensity must be positive for a likelihood term, got {value}")]
    NonPositiveIntensity { dataset: &'static str, value: f64 },
    #[error("accepted MC integral must be positive for acceptance correction, got {0}")]
    NonPositiveAcceptedIntegral(f64),
    #[error("luminosity must be positive for a cross section, got {0}")]
    NonPositiveLuminosity(f64),
    #[error("duplicate likelihood term name: {0}")]
    DuplicateTermName(String),
    #[error("unknown likelihood term: {0}")]
    MissingTerm(String),
    #[error("likelihood term is not an intensity term: {0}")]
    NotIntensityTerm(String),
    #[error("likelihood term {term} references unknown parameter {parameter}")]
    MissingParameter { term: String, parameter: String },
    #[error("likelihood term {term} has invalid penalty weight {lambda}")]
    InvalidPenaltyWeight { term: String, lambda: f64 },
    #[error("parameter values were built for a different likelihood parameter layout")]
    ParameterLayoutMismatch,
}

pub trait CpuLikelihoodTerm: Debug {
    fn name(&self) -> &str;

    fn register_params(&self, _registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        Ok(())
    }

    fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()>;

    fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64>;

    fn as_intensity(&self) -> Option<&CpuNllTerm> {
        None
    }

    fn boxed(self) -> Box<dyn CpuLikelihoodTerm>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

#[derive(Debug)]
pub struct CpuLikelihood {
    params: Arc<ParamLayout>,
    terms: Vec<Box<dyn CpuLikelihoodTerm>>,
}

impl CpuLikelihood {
    pub fn new(
        terms: impl IntoIterator<Item = Box<dyn CpuLikelihoodTerm>>,
    ) -> LikelihoodResult<Self> {
        let mut terms: Vec<_> = terms.into_iter().collect();
        let mut names = HashSet::new();
        let mut registry = ParamRegistry::new();

        for term in &terms {
            if !names.insert(term.name().to_owned()) {
                return Err(LikelihoodError::DuplicateTermName(term.name().to_owned()));
            }
            term.register_params(&mut registry)?;
        }

        let params = Arc::new(registry.layout()?);
        for term in &mut terms {
            term.resolve(Arc::clone(&params))?;
        }

        Ok(Self { params, terms })
    }

    pub fn params(&self) -> &ParamLayout {
        &self.params
    }

    pub fn default_params(&self) -> ParamValues {
        self.params.default_values()
    }

    pub fn terms(&self) -> &[Box<dyn CpuLikelihoodTerm>] {
        &self.terms
    }

    pub fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        check_params(&self.params, params)?;
        self.terms
            .iter()
            .try_fold(0.0, |sum, term| Ok(sum + term.nll(params)?))
    }

    pub fn cross_section_integrals(
        &self,
        term_name: &str,
        generated_mc: &Dataset,
    ) -> LikelihoodResult<CpuCrossSectionIntegrals> {
        let Some(term) = self.terms.iter().find(|term| term.name() == term_name) else {
            return Err(LikelihoodError::MissingTerm(term_name.to_owned()));
        };
        let Some(term) = term.as_intensity() else {
            return Err(LikelihoodError::NotIntensityTerm(term_name.to_owned()));
        };
        term.cross_section_integrals(generated_mc)
    }
}

#[derive(Clone, Debug)]
pub struct CpuNllTerm {
    name: LikelihoodName,
    plan: CpuPlan,
    local_params: Arc<ParamLayout>,
    projection: Option<ParamProjection>,
    data: CpuCachedDataset,
    accepted_mc: CpuCachedDataset,
    data_weight_sum: f64,
}

impl CpuNllTerm {
    pub fn new(
        name: impl Into<String>,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> LikelihoodResult<Self> {
        let plan = CpuBackend.prepare(model);
        let data = plan.cache_dataset(data)?;
        let accepted_mc = plan.cache_dataset(accepted_mc)?;
        let data_weight_sum = data.sum_weights();
        Ok(Self {
            name: LikelihoodName::new(name),
            plan,
            local_params: Arc::new(model.params().clone()),
            projection: None,
            data,
            accepted_mc,
            data_weight_sum,
        })
    }

    pub fn data(&self) -> &CpuCachedDataset {
        &self.data
    }

    pub fn accepted_mc(&self) -> &CpuCachedDataset {
        &self.accepted_mc
    }

    pub fn data_weight_sum(&self) -> f64 {
        self.data_weight_sum
    }

    pub fn data_log_intensity_sum(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        Ok(self
            .plan
            .try_weighted_sum_cached(&local_params, &self.data, |value| {
                positive_intensity("data", value).map(f64::ln)
            })?)
    }

    pub fn accepted_normalization(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        self.weighted_intensity_sum(&local_params, &self.accepted_mc, "accepted MC")
    }

    fn cross_section_integrals(
        &self,
        generated_mc: &Dataset,
    ) -> LikelihoodResult<CpuCrossSectionIntegrals> {
        Ok(CpuCrossSectionIntegrals {
            name: self.name.clone(),
            plan: self.plan.clone(),
            projection: self.resolved_projection()?.clone(),
            accepted_mc: self.accepted_mc.clone(),
            generated_mc: self.plan.cache_dataset(generated_mc)?,
            data_weight_sum: self.data_weight_sum,
        })
    }

    fn weighted_intensity_sum(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        Ok(self
            .plan
            .try_weighted_sum_cached(params, dataset, |value| positive_intensity(name, value))?)
    }

    fn local_values(&self, params: &ParamValues) -> LikelihoodResult<ParamValues> {
        self.resolved_projection()?.project(params)
    }

    fn resolved_projection(&self) -> LikelihoodResult<&ParamProjection> {
        self.projection
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)
    }
}

impl CpuLikelihoodTerm for CpuNllTerm {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn register_params(&self, registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        for spec in self.local_params.specs() {
            registry.register(spec.clone())?;
        }
        Ok(())
    }

    fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()> {
        self.projection = Some(ParamProjection::new(
            global_params,
            &self.local_params,
            self.name(),
        )?);
        Ok(())
    }

    fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        let normalization = positive_integral(
            "accepted MC",
            self.weighted_intensity_sum(&local_params, &self.accepted_mc, "accepted MC")?,
        )?;
        let data_log_sum =
            self.plan
                .try_weighted_sum_cached(&local_params, &self.data, |value| {
                    positive_intensity("data", value).map(f64::ln)
                })?;
        Ok(self.data_weight_sum() * normalization.ln() - data_log_sum)
    }

    fn as_intensity(&self) -> Option<&CpuNllTerm> {
        Some(self)
    }
}

#[derive(Clone, Debug)]
pub struct CpuRidgePenalty {
    inner: CpuParameterPenalty,
}

impl CpuRidgePenalty {
    pub fn new(
        name: impl Into<String>,
        parameter_names: impl IntoIterator<Item = impl Into<String>>,
        lambda: f64,
    ) -> LikelihoodResult<Self> {
        Ok(Self {
            inner: CpuParameterPenalty::new(name, parameter_names, lambda, PenaltyKind::Ridge)?,
        })
    }
}

impl CpuLikelihoodTerm for CpuRidgePenalty {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()> {
        self.inner.resolve(global_params)
    }

    fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.inner.nll(params)
    }
}

#[derive(Clone, Debug)]
pub struct CpuLassoPenalty {
    inner: CpuParameterPenalty,
}

impl CpuLassoPenalty {
    pub fn new(
        name: impl Into<String>,
        parameter_names: impl IntoIterator<Item = impl Into<String>>,
        lambda: f64,
    ) -> LikelihoodResult<Self> {
        Ok(Self {
            inner: CpuParameterPenalty::new(name, parameter_names, lambda, PenaltyKind::Lasso)?,
        })
    }
}

impl CpuLikelihoodTerm for CpuLassoPenalty {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()> {
        self.inner.resolve(global_params)
    }

    fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.inner.nll(params)
    }
}

#[derive(Clone, Debug)]
struct CpuParameterPenalty {
    name: LikelihoodName,
    parameter_names: Vec<String>,
    parameter_ids: Vec<ParamId>,
    global_params: Option<Arc<ParamLayout>>,
    lambda: f64,
    kind: PenaltyKind,
}

impl CpuParameterPenalty {
    fn new(
        name: impl Into<String>,
        parameter_names: impl IntoIterator<Item = impl Into<String>>,
        lambda: f64,
        kind: PenaltyKind,
    ) -> LikelihoodResult<Self> {
        let name = LikelihoodName::new(name);
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(LikelihoodError::InvalidPenaltyWeight {
                term: name.as_str().to_owned(),
                lambda,
            });
        }
        Ok(Self {
            name,
            parameter_names: parameter_names.into_iter().map(Into::into).collect(),
            parameter_ids: Vec::new(),
            global_params: None,
            lambda,
            kind,
        })
    }

    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()> {
        self.parameter_ids = self
            .parameter_names
            .iter()
            .map(|parameter| {
                global_params
                    .id(parameter)
                    .ok_or_else(|| LikelihoodError::MissingParameter {
                        term: self.name().to_owned(),
                        parameter: parameter.clone(),
                    })
            })
            .collect::<LikelihoodResult<_>>()?;
        self.global_params = Some(global_params);
        Ok(())
    }

    fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let global_params = self
            .global_params
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)?;
        check_params(global_params, params)?;
        let mut sum = 0.0;
        for id in &self.parameter_ids {
            let value = params.get(*id)?;
            sum += match self.kind {
                PenaltyKind::Ridge => value * value,
                PenaltyKind::Lasso => value.abs(),
            };
        }
        Ok(self.lambda * sum)
    }
}

#[derive(Copy, Clone, Debug)]
enum PenaltyKind {
    Ridge,
    Lasso,
}

#[derive(Clone, Debug)]
pub struct CpuCrossSectionIntegrals {
    name: LikelihoodName,
    plan: CpuPlan,
    projection: ParamProjection,
    accepted_mc: CpuCachedDataset,
    generated_mc: CpuCachedDataset,
    data_weight_sum: f64,
}

impl CpuCrossSectionIntegrals {
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn accepted_mc(&self) -> &CpuCachedDataset {
        &self.accepted_mc
    }

    pub fn generated_mc(&self) -> &CpuCachedDataset {
        &self.generated_mc
    }

    pub fn data_weight_sum(&self) -> f64 {
        self.data_weight_sum
    }

    pub fn accepted_integral(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let local_params = self.projection.project(params)?;
        self.weighted_intensity_sum(&local_params, &self.accepted_mc, "accepted MC")
    }

    pub fn generated_integral(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let local_params = self.projection.project(params)?;
        self.weighted_intensity_sum(&local_params, &self.generated_mc, "generated MC")
    }

    pub fn acceptance(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        let generated = positive_integral("generated MC", self.generated_integral(params)?)?;
        let accepted = positive_integral("accepted MC", self.accepted_integral(params)?)?;
        Ok(accepted / generated)
    }

    pub fn acceptance_corrected_yield(
        &self,
        params: &ParamValues,
        accepted_yield: f64,
    ) -> LikelihoodResult<f64> {
        let accepted = self.accepted_integral(params)?;
        if accepted <= 0.0 {
            return Err(LikelihoodError::NonPositiveAcceptedIntegral(accepted));
        }
        Ok(accepted_yield * self.generated_integral(params)? / accepted)
    }

    pub fn cross_section(&self, params: &ParamValues, luminosity: f64) -> LikelihoodResult<f64> {
        if luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        Ok(self.acceptance_corrected_yield(params, self.data_weight_sum)? / luminosity)
    }

    fn weighted_intensity_sum(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        Ok(self
            .plan
            .try_weighted_sum_cached(params, dataset, |value| positive_intensity(name, value))?)
    }
}

#[derive(Clone, Debug)]
struct ParamProjection {
    global_layout: Arc<ParamLayout>,
    local_layout: Arc<ParamLayout>,
    global_ids: Vec<ParamId>,
}

impl ParamProjection {
    fn new(
        global_layout: Arc<ParamLayout>,
        local_layout: &ParamLayout,
        term: &str,
    ) -> LikelihoodResult<Self> {
        let global_ids = local_layout
            .specs()
            .iter()
            .map(|spec| {
                global_layout
                    .id(spec.name())
                    .ok_or_else(|| LikelihoodError::MissingParameter {
                        term: term.to_owned(),
                        parameter: spec.name().to_owned(),
                    })
            })
            .collect::<LikelihoodResult<_>>()?;
        Ok(Self {
            global_layout,
            local_layout: Arc::new(local_layout.clone()),
            global_ids,
        })
    }

    fn project(&self, params: &ParamValues) -> LikelihoodResult<ParamValues> {
        check_params(&self.global_layout, params)?;
        let values = self
            .global_ids
            .iter()
            .map(|id| params.get(*id))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ParamValues::from_full(
            Arc::clone(&self.local_layout),
            values,
        )?)
    }
}

fn check_params(layout: &ParamLayout, params: &ParamValues) -> LikelihoodResult<()> {
    if params.layout().specs() == layout.specs() {
        Ok(())
    } else {
        Err(LikelihoodError::ParameterLayoutMismatch)
    }
}

fn positive_intensity(dataset: &'static str, value: Complex64) -> LikelihoodResult<f64> {
    if value.re > 0.0 {
        Ok(value.re)
    } else {
        Err(LikelihoodError::NonPositiveIntensity {
            dataset,
            value: value.re,
        })
    }
}

fn positive_integral(dataset: &'static str, value: f64) -> LikelihoodResult<f64> {
    if value > 0.0 {
        Ok(value)
    } else {
        Err(LikelihoodError::NonPositiveIntensity { dataset, value })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_relative_eq;
    use laddu_compile::CompiledModel;
    use laddu_data::{
        data::{Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{event_scalar, parameter};

    use super::*;

    fn weighted_dataset(values: &[(f64, f64)]) -> Dataset {
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

    fn single_term_likelihood(
        name: &str,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> CpuLikelihood {
        CpuLikelihood::new([CpuNllTerm::new(name, model, data, accepted_mc)
            .unwrap()
            .boxed()])
        .unwrap()
    }

    #[test]
    fn nll_uses_data_and_accepted_mc_reductions() {
        let expr = event_scalar("x") * parameter!("scale", initial: 0.5);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_mc = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = single_term_likelihood("data", &model, &data, &accepted_mc);
        let params = likelihood.default_params();

        let expected = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), expected);
    }

    #[test]
    fn shared_parameters_are_merged_across_independent_models() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let model_b =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data_a = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_a = weighted_dataset(&[(4.0, 1.0)]);
        let data_b = weighted_dataset(&[(5.0, 2.0)]);
        let accepted_b = weighted_dataset(&[(6.0, 3.0)]);
        let likelihood = CpuLikelihood::new([
            CpuNllTerm::new("KsKs", &model_a, &data_a, &accepted_a)
                .unwrap()
                .boxed(),
            CpuNllTerm::new("eta_pi", &model_b, &data_b, &accepted_b)
                .unwrap()
                .boxed(),
        ])
        .unwrap();

        assert_eq!(likelihood.params().len(), 1);
        assert_eq!(likelihood.params().specs()[0].name(), "scale");

        let params = likelihood.default_params();
        let term_a = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        let term_b = 2.0 * 9.0_f64.ln() - 2.0 * 2.5_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), term_a + term_b);
    }

    #[test]
    fn changing_shared_parameter_affects_all_terms() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let model_b =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = CpuLikelihood::new([
            CpuNllTerm::new("a", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            CpuNllTerm::new("b", &model_b, &data, &accepted)
                .unwrap()
                .boxed(),
        ])
        .unwrap();
        let mut params = likelihood.default_params();
        params
            .set_full(likelihood.params().id("scale").unwrap(), 1.0)
            .unwrap();

        let expected_term = 1.0 * 4.0_f64.ln() - 2.0_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), 2.0 * expected_term);
    }

    #[test]
    fn incompatible_shared_parameter_specs_are_rejected() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let model_b =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 1.0)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let err = CpuLikelihood::new([
            CpuNllTerm::new("a", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            CpuNllTerm::new("b", &model_b, &data, &accepted)
                .unwrap()
                .boxed(),
        ])
        .unwrap_err();

        assert!(matches!(
            err,
            LikelihoodError::Params(ParamError::ParameterConflict { ref name, .. })
                if name == "scale"
        ));
    }

    #[test]
    fn unique_channel_parameters_remain_separate() {
        let model_a =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale_ksks", initial: 0.5)))
                .unwrap();
        let model_b = CompiledModel::from_expr(
            &(event_scalar("x") * parameter!("scale_eta_pi", initial: 0.5)),
        )
        .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = CpuLikelihood::new([
            CpuNllTerm::new("KsKs", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            CpuNllTerm::new("eta_pi", &model_b, &data, &accepted)
                .unwrap()
                .boxed(),
        ])
        .unwrap();

        assert_eq!(likelihood.params().len(), 2);
        assert!(likelihood.params().id("scale_ksks").is_some());
        assert!(likelihood.params().id("scale_eta_pi").is_some());
    }

    #[test]
    fn ridge_and_lasso_terms_add_penalties() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = CpuLikelihood::new([
            CpuNllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            CpuRidgePenalty::new("ridge", ["scale"], 2.0)
                .unwrap()
                .boxed(),
            CpuLassoPenalty::new("lasso", ["scale"], 3.0)
                .unwrap()
                .boxed(),
        ])
        .unwrap();
        let params = likelihood.default_params();

        let nll = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        let penalty = 2.0 * 0.5_f64.powi(2) + 3.0 * 0.5_f64.abs();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), nll + penalty);
    }

    #[test]
    fn penalty_terms_reject_missing_parameters() {
        let err = CpuLikelihood::new([CpuRidgePenalty::new("ridge", ["missing"], 1.0)
            .unwrap()
            .boxed()])
        .unwrap_err();

        assert!(matches!(
            err,
            LikelihoodError::MissingParameter { ref term, ref parameter }
                if term == "ridge" && parameter == "missing"
        ));
    }

    #[derive(Debug)]
    struct ConstantTerm {
        name: String,
        value: f64,
    }

    impl CpuLikelihoodTerm for ConstantTerm {
        fn name(&self) -> &str {
            &self.name
        }

        fn resolve(&mut self, _global_params: Arc<ParamLayout>) -> LikelihoodResult<()> {
            Ok(())
        }

        fn nll(&self, _params: &ParamValues) -> LikelihoodResult<f64> {
            Ok(self.value)
        }
    }

    #[test]
    fn custom_likelihood_term_can_be_user_defined() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = CpuLikelihood::new([
            CpuNllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            ConstantTerm {
                name: "constant".into(),
                value: 12.5,
            }
            .boxed(),
        ])
        .unwrap();
        let params = likelihood.default_params();

        let expected = 1.0 * 2.0_f64.ln() - 1.0_f64.ln() + 12.5;
        assert_relative_eq!(likelihood.nll(&params).unwrap(), expected);
    }

    #[test]
    fn cross_section_integrals_use_named_intensity_term_and_global_params() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 2.0)))
                .unwrap();
        let data = weighted_dataset(&[(9.0, 4.0)]);
        let accepted_mc = weighted_dataset(&[(1.0, 2.0), (2.0, 3.0)]);
        let generated_mc = weighted_dataset(&[(4.0, 5.0), (5.0, 7.0)]);
        let likelihood = single_term_likelihood("KsKs", &model, &data, &accepted_mc);
        let params = likelihood.default_params();
        let integrals = likelihood
            .cross_section_integrals("KsKs", &generated_mc)
            .unwrap();

        let accepted = 2.0 * 2.0 + 3.0 * 4.0;
        let generated = 5.0 * 8.0 + 7.0 * 10.0;
        assert_eq!(integrals.name(), "KsKs");
        assert_relative_eq!(integrals.accepted_integral(&params).unwrap(), accepted);
        assert_relative_eq!(integrals.generated_integral(&params).unwrap(), generated);
        assert_relative_eq!(integrals.acceptance(&params).unwrap(), accepted / generated);
        assert_relative_eq!(
            integrals.acceptance_corrected_yield(&params, 20.0).unwrap(),
            20.0 * generated / accepted
        );
        assert_relative_eq!(
            integrals.cross_section(&params, 5.0).unwrap(),
            data.sum_weights().unwrap() * generated / accepted / 5.0
        );
    }

    #[test]
    fn cross_section_integrals_reject_non_intensity_terms() {
        let model =
            CompiledModel::from_expr(&(event_scalar("x") * parameter!("scale", initial: 0.5)))
                .unwrap();
        let data = weighted_dataset(&[(2.0, 1.0)]);
        let accepted = weighted_dataset(&[(4.0, 1.0)]);
        let generated = weighted_dataset(&[(5.0, 1.0)]);
        let likelihood = CpuLikelihood::new([
            CpuNllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            CpuRidgePenalty::new("ridge", ["scale"], 1.0)
                .unwrap()
                .boxed(),
        ])
        .unwrap();
        let err = likelihood
            .cross_section_integrals("ridge", &generated)
            .unwrap_err();

        assert!(matches!(err, LikelihoodError::NotIntensityTerm(ref name) if name == "ridge"));
    }
}
