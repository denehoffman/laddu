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
    #[error("gradient has length {actual}, expected {expected}")]
    GradientLengthMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LikelihoodEvaluation {
    value: f64,
    gradient: Vec<f64>,
}

impl LikelihoodEvaluation {
    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    pub fn into_parts(self) -> (f64, Vec<f64>) {
        (self.value, self.gradient)
    }
}

pub trait CpuLikelihoodTerm: Debug + Send + Sync {
    fn name(&self) -> &str;

    fn register_params(&self, _registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        Ok(())
    }

    fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()>;

    fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64>;

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
    ) -> LikelihoodResult<f64> {
        let layout = params.layout();
        if gradient.len() != layout.n_free() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: layout.n_free(),
                actual: gradient.len(),
            });
        }

        let value = self.nll(params)?;
        for (free_index, id) in layout.free_params().iter().copied().enumerate() {
            let parameter = layout.spec(id)?;
            let center = params.get(id)?;
            let scale = center.abs().max(1.0);
            let base_step = f64::EPSILON.cbrt() * scale;
            let bounds = parameter.bounds_spec();
            let left_room = bounds
                .min
                .map_or(f64::INFINITY, |min| (center - min).max(0.0));
            let right_room = bounds
                .max
                .map_or(f64::INFINITY, |max| (max - center).max(0.0));

            let derivative = if left_room > 0.0 && right_room > 0.0 {
                let step = base_step.min(left_room).min(right_room);
                let mut plus = params.clone();
                let mut minus = params.clone();
                plus.set_full(id, center + step)?;
                minus.set_full(id, center - step)?;
                (self.nll(&plus)? - self.nll(&minus)?) / (2.0 * step)
            } else if right_room > 0.0 {
                let step = base_step.min(right_room);
                let mut plus = params.clone();
                plus.set_full(id, center + step)?;
                (self.nll(&plus)? - value) / step
            } else if left_room > 0.0 {
                let step = base_step.min(left_room);
                let mut minus = params.clone();
                minus.set_full(id, center - step)?;
                (value - self.nll(&minus)?) / step
            } else {
                0.0
            };
            gradient[free_index] += derivative;
        }
        Ok(value)
    }

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

    pub fn nll_with_gradient(
        &self,
        params: &ParamValues,
    ) -> LikelihoodResult<LikelihoodEvaluation> {
        check_params(&self.params, params)?;
        let mut gradient = vec![0.0; self.params.n_free()];
        let value = self.terms.iter().try_fold(0.0, |sum, term| {
            Ok::<_, LikelihoodError>(sum + term.nll_with_gradient(params, &mut gradient)?)
        })?;
        Ok(LikelihoodEvaluation { value, gradient })
    }

    pub fn gradient(&self, params: &ParamValues) -> LikelihoodResult<Vec<f64>> {
        Ok(self.nll_with_gradient(params)?.gradient)
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
        self.plan
            .par_try_weighted_sum_cached(&local_params, &self.data, |value| {
                positive_intensity("data", value).map(f64::ln)
            })
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
        self.plan
            .par_try_weighted_sum_cached(params, dataset, |value| positive_intensity(name, value))
    }

    fn weighted_intensity_sum_with_gradient(
        &self,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        name: &'static str,
    ) -> LikelihoodResult<(f64, Vec<f64>)> {
        self.plan
            .par_try_weighted_real_sum_with_gradient_cached(params, dataset, |value| {
                Ok::<_, LikelihoodError>((positive_intensity(name, value)?, 1.0))
            })
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
                .par_try_weighted_sum_cached(&local_params, &self.data, |value| {
                    positive_intensity("data", value).map(f64::ln)
                })?;
        Ok(self.data_weight_sum() * normalization.ln() - data_log_sum)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
    ) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        let (normalization, normalization_gradient) = self.weighted_intensity_sum_with_gradient(
            &local_params,
            &self.accepted_mc,
            "accepted MC",
        )?;
        let normalization = positive_integral("accepted MC", normalization)?;
        let (data_log_sum, data_log_gradient) = self
            .plan
            .par_try_weighted_real_sum_with_gradient_cached(&local_params, &self.data, |value| {
                let intensity = positive_intensity("data", value)?;
                Ok::<_, LikelihoodError>((intensity.ln(), intensity.recip()))
            })?;
        let local_gradient = normalization_gradient
            .into_iter()
            .zip(data_log_gradient)
            .map(|(normalization_derivative, data_derivative)| {
                self.data_weight_sum * normalization_derivative / normalization - data_derivative
            })
            .collect::<Vec<_>>();
        self.resolved_projection()?
            .scatter_gradient(&local_gradient, gradient)?;
        Ok(self.data_weight_sum * normalization.ln() - data_log_sum)
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

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
    ) -> LikelihoodResult<f64> {
        self.inner.nll_with_gradient(params, gradient)
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

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
    ) -> LikelihoodResult<f64> {
        self.inner.nll_with_gradient(params, gradient)
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

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
    ) -> LikelihoodResult<f64> {
        let global_params = self
            .global_params
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)?;
        check_params(global_params, params)?;
        if gradient.len() != global_params.n_free() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: global_params.n_free(),
                actual: gradient.len(),
            });
        }
        let mut sum = 0.0;
        for id in &self.parameter_ids {
            let value = params.get(*id)?;
            let (penalty, derivative) = match self.kind {
                PenaltyKind::Ridge => (value * value, 2.0 * value),
                PenaltyKind::Lasso => {
                    (value.abs(), if value == 0.0 { 0.0 } else { value.signum() })
                }
            };
            sum += penalty;
            if let Some(free) = global_params.free_id(*id)? {
                gradient[free.index()] += self.lambda * derivative;
            }
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
        self.plan
            .try_weighted_sum_cached(params, dataset, |value| positive_intensity(name, value))
    }
}

#[derive(Clone, Debug)]
struct ParamProjection {
    global_layout: Arc<ParamLayout>,
    local_layout: Arc<ParamLayout>,
    global_ids: Vec<ParamId>,
    local_free_to_global_free: Vec<usize>,
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
        let local_free_to_global_free = local_layout
            .free_params()
            .iter()
            .map(|local_id| {
                let name = local_layout.name(*local_id)?;
                let global_id =
                    global_layout
                        .id(name)
                        .ok_or_else(|| LikelihoodError::MissingParameter {
                            term: term.to_owned(),
                            parameter: name.to_owned(),
                        })?;
                global_layout
                    .free_id(global_id)?
                    .map(|id| id.index())
                    .ok_or(LikelihoodError::ParameterLayoutMismatch)
            })
            .collect::<LikelihoodResult<Vec<_>>>()?;
        Ok(Self {
            global_layout,
            local_layout: Arc::new(local_layout.clone()),
            global_ids,
            local_free_to_global_free,
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

    fn scatter_gradient(&self, local: &[f64], global: &mut [f64]) -> LikelihoodResult<()> {
        if local.len() != self.local_free_to_global_free.len() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: self.local_free_to_global_free.len(),
                actual: local.len(),
            });
        }
        if global.len() != self.global_layout.n_free() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: self.global_layout.n_free(),
                actual: global.len(),
            });
        }
        for (derivative, target) in local.iter().zip(&self.local_free_to_global_free) {
            global[*target] += derivative;
        }
        Ok(())
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
    use laddu_expr::{event_scalar, parameter, parameters::Parameter};

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

    fn finite_difference_nll(
        likelihood: &CpuLikelihood,
        params: &ParamValues,
        free_parameter: usize,
    ) -> f64 {
        let id = likelihood.params().free_params()[free_parameter];
        let center = params.get(id).unwrap();
        let h = 1.0e-6;
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus.set_full(id, center + h).unwrap();
        minus.set_full(id, center - h).unwrap();
        (likelihood.nll(&plus).unwrap() - likelihood.nll(&minus).unwrap()) / (2.0 * h)
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
    fn nll_gradient_matches_finite_difference() {
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.7));
        let expr = (event_scalar("x") + scale).powi(2);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let data = weighted_dataset(&[(0.3, 1.0), (1.1, 2.0)]);
        let accepted_mc = weighted_dataset(&[(0.5, 1.5), (1.7, 0.8)]);
        let likelihood = single_term_likelihood("data", &model, &data, &accepted_mc);
        let params = likelihood.default_params();
        let evaluation = likelihood.nll_with_gradient(&params).unwrap();

        assert_relative_eq!(evaluation.value(), likelihood.nll(&params).unwrap());
        assert_relative_eq!(
            evaluation.gradient()[0],
            finite_difference_nll(&likelihood, &params, 0),
            epsilon = 1.0e-8
        );
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
    fn shared_and_channel_specific_gradients_scatter_into_global_layout() {
        let shared = laddu_expr::Expr::from(parameter!("shared", initial: 0.4));
        let model_a = CompiledModel::from_expr(
            &(event_scalar("x")
                + shared.clone()
                + laddu_expr::Expr::from(parameter!("only_a", initial: 0.2)))
            .powi(2),
        )
        .unwrap();
        let model_b = CompiledModel::from_expr(
            &(event_scalar("x")
                + shared
                + laddu_expr::Expr::from(parameter!("only_b", initial: -0.1)))
            .powi(2),
        )
        .unwrap();
        let data = weighted_dataset(&[(0.5, 1.0), (1.2, 0.7)]);
        let accepted = weighted_dataset(&[(0.8, 1.3), (1.5, 0.9)]);
        let likelihood = CpuLikelihood::new([
            CpuNllTerm::new("a", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            CpuNllTerm::new("b", &model_b, &data, &accepted)
                .unwrap()
                .boxed(),
        ])
        .unwrap();
        let params = likelihood.default_params();
        let evaluation = likelihood.nll_with_gradient(&params).unwrap();

        for (parameter, derivative) in evaluation.gradient().iter().enumerate() {
            assert_relative_eq!(
                *derivative,
                finite_difference_nll(&likelihood, &params, parameter),
                epsilon = 1.0e-8
            );
        }
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
        assert_relative_eq!(likelihood.gradient(&params).unwrap()[0], 5.0);
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

    #[derive(Debug)]
    struct BoundedQuadraticTerm {
        parameter: Parameter,
        id: Option<ParamId>,
    }

    impl CpuLikelihoodTerm for BoundedQuadraticTerm {
        fn name(&self) -> &str {
            "bounded-quadratic"
        }

        fn register_params(&self, registry: &mut ParamRegistry) -> LikelihoodResult<()> {
            registry.register(self.parameter.clone())?;
            Ok(())
        }

        fn resolve(&mut self, global_params: Arc<ParamLayout>) -> LikelihoodResult<()> {
            self.id = global_params.id(self.parameter.name());
            Ok(())
        }

        fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
            let value = params.get(self.id.ok_or(LikelihoodError::ParameterLayoutMismatch)?)?;
            Ok((value - 2.0).powi(2))
        }
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
    fn custom_term_gradient_uses_bounded_finite_difference_fallback() {
        let likelihood = CpuLikelihood::new([BoundedQuadraticTerm {
            parameter: Parameter::free("x")
                .with_initial(0.0)
                .with_bounds(Some(0.0), None),
            id: None,
        }
        .boxed()])
        .unwrap();
        let params = likelihood.default_params();
        let evaluation = likelihood.nll_with_gradient(&params).unwrap();

        assert_relative_eq!(evaluation.value(), 4.0);
        assert_relative_eq!(evaluation.gradient()[0], -4.0, epsilon = 1.0e-5);
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
