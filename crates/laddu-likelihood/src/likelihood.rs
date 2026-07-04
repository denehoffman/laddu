use std::{collections::HashSet, fmt::Debug, sync::Arc};

use crate::{LikelihoodError, LikelihoodResult};
use laddu_compile::{CompiledModel, ReductionPlan};
use laddu_data::data::Dataset;
#[cfg(test)]
use laddu_expr::parameters::ParamError;
use laddu_expr::parameters::{ParamId, ParamLayout, ParamRegistry, ParamValues};
use laddu_runtime::{CpuBackend, CpuPlan, CpuPreparedDataset, Execution};

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

pub trait LikelihoodTerm: Debug + Send + Sync {
    fn name(&self) -> &str;

    fn register_params(&self, _registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        Ok(())
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        execution: &Execution,
    ) -> LikelihoodResult<()>;

    fn nll(&self, params: &ParamValues, execution: &Execution) -> LikelihoodResult<f64>;

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
    ) -> LikelihoodResult<f64> {
        let layout = params.layout();
        if gradient.len() != layout.n_free() {
            return Err(LikelihoodError::GradientLengthMismatch {
                expected: layout.n_free(),
                actual: gradient.len(),
            });
        }

        let value = self.nll(params, execution)?;
        for (free_index, id) in layout.free_params().iter().copied().enumerate() {
            let parameter = layout.spec(id)?;
            let free_id = layout
                .free_id(id)?
                .ok_or(LikelihoodError::ParameterLayoutMismatch)?;
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
                plus.set_free(free_id, center + step)?;
                minus.set_free(free_id, center - step)?;
                (self.nll(&plus, execution)? - self.nll(&minus, execution)?) / (2.0 * step)
            } else if right_room > 0.0 {
                let step = base_step.min(right_room);
                let mut plus = params.clone();
                plus.set_free(free_id, center + step)?;
                (self.nll(&plus, execution)? - value) / step
            } else if left_room > 0.0 {
                let step = base_step.min(left_room);
                let mut minus = params.clone();
                minus.set_free(free_id, center - step)?;
                (value - self.nll(&minus, execution)?) / step
            } else {
                0.0
            };
            gradient[free_index] += derivative;
        }
        Ok(value)
    }

    fn as_intensity(&self) -> Option<&NllTerm> {
        None
    }

    fn boxed(self) -> Box<dyn LikelihoodTerm>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

pub enum Parameters<'a> {
    Slice(&'a [f64]),
    ParamValues(&'a ParamValues),
}

impl<'a> From<&'a [f64]> for Parameters<'a> {
    fn from(val: &'a [f64]) -> Self {
        Self::Slice(val)
    }
}

impl<'a, const N: usize> From<&'a [f64; N]> for Parameters<'a> {
    fn from(val: &'a [f64; N]) -> Self {
        Self::Slice(val.as_slice())
    }
}

impl<'a> From<&'a Vec<f64>> for Parameters<'a> {
    fn from(value: &'a Vec<f64>) -> Self {
        Self::Slice(value.as_slice())
    }
}

impl<'a> From<&'a ParamValues> for Parameters<'a> {
    fn from(val: &'a ParamValues) -> Self {
        Self::ParamValues(val)
    }
}

#[derive(Debug)]
pub struct Likelihood {
    params: Arc<ParamLayout>,
    terms: Vec<Box<dyn LikelihoodTerm>>,
    execution: Execution,
}

impl Likelihood {
    pub fn new(terms: impl IntoIterator<Item = Box<dyn LikelihoodTerm>>) -> LikelihoodResult<Self> {
        Self::with_execution(terms, Execution::default())
    }

    pub fn with_execution(
        terms: impl IntoIterator<Item = Box<dyn LikelihoodTerm>>,
        execution: Execution,
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
            term.resolve(Arc::clone(&params), &execution)?;
        }

        Ok(Self {
            params,
            terms,
            execution,
        })
    }

    pub fn params(&self) -> &ParamLayout {
        &self.params
    }

    /// Return deterministic initial values in the order expected by [`Self::nll`].
    pub fn default_params(&self) -> Vec<f64> {
        self.params.initial_free_values()
    }

    /// Generate one value for each free parameter in objective-vector order.
    pub fn params_with(
        &self,
        value: impl FnMut(&laddu_expr::parameters::Parameter) -> f64,
    ) -> Vec<f64> {
        self.params.free_values_with(value)
    }

    /// Sample uniform initial ranges while preserving fixed and point-initialized parameters.
    pub fn sample_initial(&self, seed: u64) -> Vec<f64> {
        self.params.sample_initial(seed)
    }

    pub fn terms(&self) -> &[Box<dyn LikelihoodTerm>] {
        &self.terms
    }

    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    /// Evaluate the objective from free values in [`Self::params`] order.
    pub fn nll<'a>(&self, parameters: impl Into<Parameters<'a>>) -> LikelihoodResult<f64> {
        let params = match parameters.into() {
            Parameters::Slice(free) => &self.params.values(free)?,
            Parameters::ParamValues(param_values) => param_values,
        };
        self.nll_values(&params)
    }

    fn nll_values(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        check_params(&self.params, params)?;
        self.terms.iter().try_fold(
            0.0,
            |sum, term| Ok(sum + term.nll(params, &self.execution)?),
        )
    }

    pub fn nll_with_gradient<'a>(
        &self,
        parameters: impl Into<Parameters<'a>>,
    ) -> LikelihoodResult<LikelihoodEvaluation> {
        let params = match parameters.into() {
            Parameters::Slice(free) => &self.params.values(free)?,
            Parameters::ParamValues(param_values) => param_values,
        };
        self.nll_with_gradient_values(&params)
    }

    fn nll_with_gradient_values(
        &self,
        params: &ParamValues,
    ) -> LikelihoodResult<LikelihoodEvaluation> {
        check_params(&self.params, params)?;
        let mut gradient = vec![0.0; self.params.n_free()];
        let value = self.terms.iter().try_fold(0.0, |sum, term| {
            Ok::<_, LikelihoodError>(
                sum + term.nll_with_gradient(params, &mut gradient, &self.execution)?,
            )
        })?;
        Ok(LikelihoodEvaluation { value, gradient })
    }

    pub fn cross_section_integrals(
        &self,
        term_name: &str,
        generated_mc: &Dataset,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let Some(term) = self.terms.iter().find(|term| term.name() == term_name) else {
            return Err(LikelihoodError::MissingTerm(term_name.to_owned()));
        };
        let Some(term) = term.as_intensity() else {
            return Err(LikelihoodError::NotIntensityTerm(term_name.to_owned()));
        };
        term.cross_section_integrals(generated_mc, &self.execution)
    }
}

#[derive(Clone)]
pub struct NllTerm {
    name: LikelihoodName,
    model: CompiledModel,
    plan: Option<CpuPlan>,
    local_params: Arc<ParamLayout>,
    projection: Option<ParamProjection>,
    data_source: Dataset,
    accepted_mc_source: Dataset,
    data: Option<CpuPreparedDataset>,
    accepted_mc: Option<CpuPreparedDataset>,
    data_weight_sum: Option<f64>,
    execution: Option<Execution>,
}

impl std::fmt::Debug for NllTerm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NllTerm")
            .field("name", &self.name)
            .field("prepared", &self.data.is_some())
            .finish_non_exhaustive()
    }
}

impl NllTerm {
    pub fn new(
        name: impl Into<String>,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> LikelihoodResult<Self> {
        Ok(Self {
            name: LikelihoodName::new(name),
            model: model.clone(),
            plan: None,
            local_params: Arc::new(model.params().clone()),
            projection: None,
            data_source: data.clone(),
            accepted_mc_source: accepted_mc.clone(),
            data: None,
            accepted_mc: None,
            data_weight_sum: None,
            execution: None,
        })
    }

    pub fn data(&self) -> LikelihoodResult<&CpuPreparedDataset> {
        self.data
            .as_ref()
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    fn plan(&self) -> LikelihoodResult<&CpuPlan> {
        self.plan
            .as_ref()
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    pub fn accepted_mc(&self) -> LikelihoodResult<&CpuPreparedDataset> {
        self.accepted_mc
            .as_ref()
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    pub fn data_weight_sum(&self) -> LikelihoodResult<f64> {
        self.data_weight_sum
            .ok_or_else(|| LikelihoodError::UnresolvedTerm(self.name().to_owned()))
    }

    pub fn data_log_intensity_sum(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.global_values(free)?;
        let local_params = self.local_values(&params)?;
        self.reduce(
            &local_params,
            self.data()?,
            ReductionPlan::weighted_log_positive_real(),
            "data",
        )
    }

    pub fn accepted_normalization(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.global_values(free)?;
        let local_params = self.local_values(&params)?;
        self.weighted_intensity_sum(&local_params, self.accepted_mc()?, "accepted MC")
    }

    fn cross_section_integrals(
        &self,
        generated_mc: &Dataset,
        execution: &Execution,
    ) -> LikelihoodResult<CrossSectionIntegrals> {
        let plan = self.plan()?.clone();
        Ok(CrossSectionIntegrals {
            name: self.name.clone(),
            plan: plan.clone(),
            projection: self.resolved_projection()?.clone(),
            accepted_mc: self.accepted_mc()?.clone(),
            generated_mc: plan.prepare_dataset(execution, generated_mc)?,
            data_weight_sum: self.data_weight_sum()?,
            execution: execution.clone(),
        })
    }

    fn weighted_intensity_sum(
        &self,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        self.reduce(
            params,
            dataset,
            ReductionPlan::weighted_positive_real(),
            name,
        )
    }

    fn reduce(
        &self,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        reduction: ReductionPlan,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        self.plan()?
            .reduce(self.resolved_execution()?, params, dataset, reduction)
            .map_err(|error| map_reduction_error(name, error))
    }

    fn local_values(&self, params: &ParamValues) -> LikelihoodResult<ParamValues> {
        self.resolved_projection()?.project(params)
    }

    fn global_values(&self, free: &[f64]) -> LikelihoodResult<ParamValues> {
        Ok(self.resolved_projection()?.global_layout.values(free)?)
    }

    fn resolved_projection(&self) -> LikelihoodResult<&ParamProjection> {
        self.projection
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)
    }

    fn resolved_execution(&self) -> LikelihoodResult<&Execution> {
        self.execution
            .as_ref()
            .ok_or(LikelihoodError::ParameterLayoutMismatch)
    }
}

impl LikelihoodTerm for NllTerm {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn register_params(&self, registry: &mut ParamRegistry) -> LikelihoodResult<()> {
        for spec in self.local_params.specs() {
            registry.register(spec.clone())?;
        }
        Ok(())
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        execution: &Execution,
    ) -> LikelihoodResult<()> {
        self.projection = Some(ParamProjection::new(
            global_params,
            &self.local_params,
            self.name(),
        )?);
        let plan = CpuBackend.prepare_for_execution(&self.model, execution)?;
        self.data = Some(plan.prepare_dataset(execution, &self.data_source)?);
        self.accepted_mc = Some(plan.prepare_dataset(execution, &self.accepted_mc_source)?);
        self.plan = Some(plan);
        self.data_weight_sum = Some(self.data()?.stats().sum_weights());
        self.execution = Some(execution.clone());
        Ok(())
    }

    fn nll(&self, params: &ParamValues, execution: &Execution) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        let normalization = positive_integral(
            "accepted MC",
            self.plan()?
                .reduce(
                    execution,
                    &local_params,
                    self.accepted_mc()?,
                    ReductionPlan::weighted_positive_real(),
                )
                .map_err(|error| map_reduction_error("accepted MC", error))?,
        )?;
        let data_log_sum = self
            .plan()?
            .reduce(
                execution,
                &local_params,
                self.data()?,
                ReductionPlan::weighted_log_positive_real(),
            )
            .map_err(|error| map_reduction_error("data", error))?;
        Ok(self.data_weight_sum()? * normalization.ln() - data_log_sum)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        execution: &Execution,
    ) -> LikelihoodResult<f64> {
        let local_params = self.local_values(params)?;
        let normalization_evaluation = self
            .plan()?
            .reduce_with_gradient(
                execution,
                &local_params,
                self.accepted_mc()?,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error("accepted MC", error))?;
        let (normalization, normalization_gradient) = normalization_evaluation.into_parts();
        let normalization = positive_integral("accepted MC", normalization)?;
        let data_evaluation = self
            .plan()?
            .reduce_with_gradient(
                execution,
                &local_params,
                self.data()?,
                ReductionPlan::weighted_log_positive_real(),
            )
            .map_err(|error| map_reduction_error("data", error))?;
        let (data_log_sum, data_log_gradient) = data_evaluation.into_parts();
        let data_weight_sum = self.data_weight_sum()?;
        let local_gradient = normalization_gradient
            .into_iter()
            .zip(data_log_gradient)
            .map(|(normalization_derivative, data_derivative)| {
                data_weight_sum * normalization_derivative / normalization - data_derivative
            })
            .collect::<Vec<_>>();
        self.resolved_projection()?
            .scatter_gradient(&local_gradient, gradient)?;
        Ok(data_weight_sum * normalization.ln() - data_log_sum)
    }

    fn as_intensity(&self) -> Option<&NllTerm> {
        Some(self)
    }
}

#[derive(Clone, Debug)]
pub struct RidgePenalty {
    inner: CpuParameterPenalty,
}

impl RidgePenalty {
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

impl LikelihoodTerm for RidgePenalty {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        _execution: &Execution,
    ) -> LikelihoodResult<()> {
        self.inner.resolve(global_params)
    }

    fn nll(&self, params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
        self.inner.nll(params)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        _execution: &Execution,
    ) -> LikelihoodResult<f64> {
        self.inner.nll_with_gradient(params, gradient)
    }
}

#[derive(Clone, Debug)]
pub struct LassoPenalty {
    inner: CpuParameterPenalty,
}

impl LassoPenalty {
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

impl LikelihoodTerm for LassoPenalty {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn resolve(
        &mut self,
        global_params: Arc<ParamLayout>,
        _execution: &Execution,
    ) -> LikelihoodResult<()> {
        self.inner.resolve(global_params)
    }

    fn nll(&self, params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
        self.inner.nll(params)
    }

    fn nll_with_gradient(
        &self,
        params: &ParamValues,
        gradient: &mut [f64],
        _execution: &Execution,
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
pub struct CrossSectionIntegrals {
    name: LikelihoodName,
    plan: CpuPlan,
    projection: ParamProjection,
    accepted_mc: CpuPreparedDataset,
    generated_mc: CpuPreparedDataset,
    data_weight_sum: f64,
    execution: Execution,
}

impl CrossSectionIntegrals {
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn accepted_mc(&self) -> &CpuPreparedDataset {
        &self.accepted_mc
    }

    pub fn generated_mc(&self) -> &CpuPreparedDataset {
        &self.generated_mc
    }

    pub fn data_weight_sum(&self) -> f64 {
        self.data_weight_sum
    }

    pub fn accepted_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.projection.global_layout.values(free)?;
        let local_params = self.projection.project(&params)?;
        self.weighted_intensity_sum(&local_params, &self.accepted_mc, "accepted MC")
    }

    pub fn generated_integral(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let params = self.projection.global_layout.values(free)?;
        let local_params = self.projection.project(&params)?;
        self.weighted_intensity_sum(&local_params, &self.generated_mc, "generated MC")
    }

    pub fn acceptance(&self, free: &[f64]) -> LikelihoodResult<f64> {
        let generated = positive_integral("generated MC", self.generated_integral(free)?)?;
        let accepted = positive_integral("accepted MC", self.accepted_integral(free)?)?;
        Ok(accepted / generated)
    }

    pub fn acceptance_corrected_yield(
        &self,
        free: &[f64],
        accepted_yield: f64,
    ) -> LikelihoodResult<f64> {
        let accepted = self.accepted_integral(free)?;
        if accepted <= 0.0 {
            return Err(LikelihoodError::NonPositiveAcceptedIntegral(accepted));
        }
        Ok(accepted_yield * self.generated_integral(free)? / accepted)
    }

    pub fn cross_section(&self, free: &[f64], luminosity: f64) -> LikelihoodResult<f64> {
        if luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        Ok(self.acceptance_corrected_yield(free, self.data_weight_sum)? / luminosity)
    }

    fn weighted_intensity_sum(
        &self,
        params: &ParamValues,
        dataset: &CpuPreparedDataset,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        self.plan
            .reduce(
                &self.execution,
                params,
                dataset,
                ReductionPlan::weighted_positive_real(),
            )
            .map_err(|error| map_reduction_error(name, error))
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
        let free = self
            .local_layout
            .free_params()
            .iter()
            .map(|local_id| params.get(self.global_ids[local_id.index()]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.local_layout.values(&free)?)
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

fn map_reduction_error(
    dataset: &'static str,
    error: laddu_runtime::RuntimeError,
) -> LikelihoodError {
    match error {
        laddu_runtime::RuntimeError::Reduction(
            laddu_compile::ReductionError::NonPositiveValue { value, .. },
        ) => LikelihoodError::NonPositiveIntensity { dataset, value },
        error => error.into(),
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
        data::{CacheStorage, Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{
        complex, event_scalar, matrix, parameter, parameters::Parameter, solve, vector,
    };
    use laddu_runtime::{CpuOptions, Device, ExecutionOptions, Precision, ThreadPolicy};

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

    fn weighted_dataset_batches(values: &[(f64, f64)], ends: &[usize]) -> Dataset {
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x"], true).unwrap());
        let mut start = 0;
        let batches = ends
            .iter()
            .map(|&end| {
                let batch = EventBatch::from_events(
                    Arc::clone(&schema),
                    values[start..end]
                        .iter()
                        .map(|(x, weight)| OwnedEvent::weighted(vec![], vec![*x], *weight)),
                )
                .unwrap();
                start = end;
                batch
            })
            .collect::<Vec<_>>();
        assert_eq!(start, values.len());
        Dataset::from_batches(batches)
            .unwrap()
            .chunked(values.len())
            .unwrap()
    }

    fn single_term_likelihood(
        name: &str,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> Likelihood {
        Likelihood::new([NllTerm::new(name, model, data, accepted_mc)
            .unwrap()
            .boxed()])
        .unwrap()
    }

    fn single_term_likelihood_with_execution(
        name: &str,
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
        execution: Execution,
    ) -> Likelihood {
        Likelihood::with_execution(
            [NllTerm::new(name, model, data, accepted_mc)
                .unwrap()
                .boxed()],
            execution,
        )
        .unwrap()
    }

    fn finite_difference_nll(
        likelihood: &Likelihood,
        params: &[f64],
        free_parameter: usize,
    ) -> f64 {
        let center = params[free_parameter];
        let h = 1.0e-6;
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        plus[free_parameter] = center + h;
        minus[free_parameter] = center - h;
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
    fn likelihood_accepts_free_slices_and_generates_free_parameters() {
        let scale = laddu_expr::Expr::from(Parameter::free("scale").with_initial((0.25, 0.75)));
        let offset = laddu_expr::Expr::from(Parameter::fixed("offset", 1.0));
        let model = CompiledModel::from_expr(&(event_scalar("x") * scale + offset)).unwrap();
        let data = weighted_dataset(&[(1.0, 1.0), (2.0, 1.0)]);
        let accepted = weighted_dataset(&[(1.5, 1.0), (2.5, 1.0)]);
        let likelihood = single_term_likelihood("slice", &model, &data, &accepted);

        assert_eq!(likelihood.default_params(), vec![0.5]);
        assert_eq!(likelihood.sample_initial(0), vec![0.5513138035955086]);
        assert_eq!(
            likelihood.params_with(|parameter| parameter.name().len() as f64),
            vec![5.0]
        );
        assert!(likelihood.nll(&[0.5f64]).unwrap().is_finite());
        assert!(matches!(
            likelihood.nll(&[]),
            Err(LikelihoodError::Params(ParamError::FreeLengthMismatch {
                expected: 1,
                actual: 0
            }))
        ));
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
    fn likelihood_is_invariant_under_dataset_batching() {
        let x = event_scalar("x");
        let coupling = laddu_expr::Expr::from(parameter!("coupling", initial: 0.35));
        let matrix = matrix([
            [x.clone() + 2.0, complex(coupling.clone(), 0.15)],
            [complex(-0.2, coupling), 3.5.into()],
        ]);
        let amplitude = solve(matrix, vector([x.sin() + 1.0, complex(x.cos(), 0.5)])).component(1);
        let model = CompiledModel::from_expr(&(amplitude.norm_sqr() + 0.25)).unwrap();
        let values = [
            (0.15, 0.7),
            (0.35, 1.2),
            (0.65, 0.5),
            (0.95, 1.8),
            (1.25, 0.9),
            (1.55, 1.1),
            (1.85, 0.6),
        ];
        let one_batch = weighted_dataset_batches(&values, &[values.len()]);
        let two_batches = weighted_dataset_batches(&values, &[3, values.len()]);
        let uneven_batches = weighted_dataset_batches(&values, &[1, 2, 6, values.len()]);
        let streaming = weighted_dataset_batches(&values, &[2, 5, values.len()]).streaming();

        let reference = single_term_likelihood("reference", &model, &one_batch, &one_batch);
        let two = single_term_likelihood("two", &model, &two_batches, &two_batches);
        let uneven = single_term_likelihood("uneven", &model, &uneven_batches, &uneven_batches);
        let serial = single_term_likelihood_with_execution(
            "serial",
            &model,
            &streaming,
            &streaming,
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Serial,
                    ..CpuOptions::default()
                }),
                ..ExecutionOptions::default()
            })
            .unwrap(),
        );
        let fixed = single_term_likelihood_with_execution(
            "fixed",
            &model,
            &two_batches,
            &streaming,
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Fixed(2),
                    ..CpuOptions::default()
                }),
                ..ExecutionOptions::default()
            })
            .unwrap(),
        );
        let expected = reference
            .nll_with_gradient(&reference.default_params())
            .unwrap();

        for actual in [
            two.nll_with_gradient(&two.default_params()).unwrap(),
            uneven.nll_with_gradient(&uneven.default_params()).unwrap(),
            serial.nll_with_gradient(&serial.default_params()).unwrap(),
            fixed.nll_with_gradient(&fixed.default_params()).unwrap(),
        ] {
            assert_relative_eq!(actual.value(), expected.value(), epsilon = 1.0e-12);
            assert_eq!(actual.gradient().len(), expected.gradient().len());
            for (actual, expected) in actual.gradient().iter().zip(expected.gradient()) {
                assert_relative_eq!(actual, expected, epsilon = 1.0e-11);
            }
        }
        assert_eq!(
            reference.terms()[0]
                .as_intensity()
                .unwrap()
                .data()
                .unwrap()
                .stats()
                .storage(),
            CacheStorage::Resident
        );
        let streaming_stats = serial.terms()[0]
            .as_intensity()
            .unwrap()
            .data()
            .unwrap()
            .stats();
        assert_eq!(streaming_stats.storage(), CacheStorage::Streaming);
        assert_eq!(streaming_stats.resident_bytes(), 0);
        assert_eq!(streaming_stats.local_batches(), 3);
    }

    #[test]
    fn likelihood_nll_uses_configured_f32_scalar_execution() {
        let dataset = weighted_dataset(&[(1.0, 1.0), (2.0, 1.0)]);
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 1.0));
        let model = CompiledModel::from_expr(&(event_scalar("x") + scale)).unwrap();
        let likelihood = single_term_likelihood_with_execution(
            "f32",
            &model,
            &dataset,
            &dataset,
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions {
                    threads: ThreadPolicy::Serial,
                    ..CpuOptions::default()
                }),
                precision: Precision::F32,
                ..ExecutionOptions::default()
            })
            .unwrap(),
        );

        let expected = 2.0 * 5.0_f64.ln() - 2.0_f32.ln() as f64 - 3.0_f32.ln() as f64;
        assert_eq!(
            likelihood.nll(&likelihood.default_params()).unwrap(),
            expected
        );
    }

    #[cfg(feature = "mpi")]
    #[mpi_test::mpi_test(np = [2, 3, 4])]
    fn mpi_likelihood_matches_local_reference_without_multiplying_penalties() {
        use mpi::traits::Communicator;

        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let values = [(0.4, 1.5), (1.2, 0.75)];
        let resident = weighted_dataset_batches(&values, &[1, values.len()]);
        let streaming = weighted_dataset_batches(&values, &[1, values.len()]).streaming();
        let scale = laddu_expr::Expr::from(parameter!("scale", initial: 0.6));
        let model = CompiledModel::from_expr(&(event_scalar("x") + scale).powi(2)).unwrap();

        let reference = Likelihood::new([
            NllTerm::new("data", &model, &resident, &streaming)
                .unwrap()
                .boxed(),
            RidgePenalty::new("ridge", ["scale"], 0.3).unwrap().boxed(),
        ])
        .unwrap();
        let distributed = Likelihood::with_execution(
            [
                NllTerm::new("data", &model, &resident, &streaming)
                    .unwrap()
                    .boxed(),
                RidgePenalty::new("ridge", ["scale"], 0.3).unwrap().boxed(),
            ],
            Execution::distributed(
                ExecutionOptions {
                    device: Device::Cpu(CpuOptions {
                        threads: ThreadPolicy::Serial,
                        ..CpuOptions::default()
                    }),
                    partitioning: laddu_data::io::Partitioning::Contiguous,
                    ..ExecutionOptions::default()
                },
                &world,
            )
            .unwrap(),
        )
        .unwrap();

        let expected = reference
            .nll_with_gradient(&reference.default_params())
            .unwrap();
        let actual = distributed
            .nll_with_gradient(&distributed.default_params())
            .unwrap();
        assert_relative_eq!(actual.value(), expected.value(), epsilon = 1.0e-12);
        assert_relative_eq!(
            actual.gradient()[0],
            expected.gradient()[0],
            epsilon = 1.0e-11
        );
        let expected_term = reference.terms()[0].as_intensity().unwrap();
        let actual_term = distributed.terms()[0].as_intensity().unwrap();
        assert_relative_eq!(
            actual_term
                .accepted_normalization(&distributed.default_params())
                .unwrap(),
            expected_term
                .accepted_normalization(&reference.default_params())
                .unwrap(),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            actual_term
                .data_log_intensity_sum(&distributed.default_params())
                .unwrap(),
            expected_term
                .data_log_intensity_sum(&reference.default_params())
                .unwrap(),
            epsilon = 1.0e-12
        );

        let stats = distributed.terms()[0]
            .as_intensity()
            .unwrap()
            .data()
            .unwrap()
            .stats();
        assert_eq!(stats.global_events(), values.len());
        assert!(stats.local_events() <= 1 || world.size() <= 2);
    }

    #[cfg(feature = "mpi")]
    #[mpi_test::mpi_test(np = [2, 3])]
    fn mpi_likelihood_propagates_a_rank_local_error_without_deadlocking() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let data = weighted_dataset_batches(&[(-1.0, 1.0), (2.0, 1.0)], &[2]);
        let accepted_mc = weighted_dataset_batches(&[(1.0, 1.0), (2.0, 1.0)], &[2]);
        let model = CompiledModel::from_expr(&event_scalar("x")).unwrap();
        let likelihood = Likelihood::with_execution(
            [NllTerm::new("data", &model, &data, &accepted_mc)
                .unwrap()
                .boxed()],
            Execution::distributed(ExecutionOptions::default(), &world).unwrap(),
        )
        .unwrap();

        assert!(likelihood.nll(&likelihood.default_params()).is_err());
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
        let likelihood = Likelihood::new([
            NllTerm::new("KsKs", &model_a, &data_a, &accepted_a)
                .unwrap()
                .boxed(),
            NllTerm::new("eta_pi", &model_b, &data_b, &accepted_b)
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
        let likelihood = Likelihood::new([
            NllTerm::new("a", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            NllTerm::new("b", &model_b, &data, &accepted)
                .unwrap()
                .boxed(),
        ])
        .unwrap();
        let mut params = likelihood.default_params();
        params[0] = 1.0;

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
        let likelihood = Likelihood::new([
            NllTerm::new("a", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            NllTerm::new("b", &model_b, &data, &accepted)
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
        let err = Likelihood::new([
            NllTerm::new("a", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            NllTerm::new("b", &model_b, &data, &accepted)
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
        let likelihood = Likelihood::new([
            NllTerm::new("KsKs", &model_a, &data, &accepted)
                .unwrap()
                .boxed(),
            NllTerm::new("eta_pi", &model_b, &data, &accepted)
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
        let likelihood = Likelihood::new([
            NllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            RidgePenalty::new("ridge", ["scale"], 2.0).unwrap().boxed(),
            LassoPenalty::new("lasso", ["scale"], 3.0).unwrap().boxed(),
        ])
        .unwrap();
        let params = likelihood.default_params();

        let nll = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        let penalty = 2.0 * 0.5_f64.powi(2) + 3.0 * 0.5_f64.abs();
        let result = likelihood.nll_with_gradient(&params).unwrap();
        assert_relative_eq!(result.value(), nll + penalty);
        assert_relative_eq!(result.gradient()[0], 5.0);
    }

    #[test]
    fn penalty_terms_reject_missing_parameters() {
        let err = Likelihood::new([RidgePenalty::new("ridge", ["missing"], 1.0)
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

    impl LikelihoodTerm for BoundedQuadraticTerm {
        fn name(&self) -> &str {
            "bounded-quadratic"
        }

        fn register_params(&self, registry: &mut ParamRegistry) -> LikelihoodResult<()> {
            registry.register(self.parameter.clone())?;
            Ok(())
        }

        fn resolve(
            &mut self,
            global_params: Arc<ParamLayout>,
            _execution: &Execution,
        ) -> LikelihoodResult<()> {
            self.id = global_params.id(self.parameter.name());
            Ok(())
        }

        fn nll(&self, params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
            let value = params.get(self.id.ok_or(LikelihoodError::ParameterLayoutMismatch)?)?;
            Ok((value - 2.0).powi(2))
        }
    }

    impl LikelihoodTerm for ConstantTerm {
        fn name(&self) -> &str {
            &self.name
        }

        fn resolve(
            &mut self,
            _global_params: Arc<ParamLayout>,
            _execution: &Execution,
        ) -> LikelihoodResult<()> {
            Ok(())
        }

        fn nll(&self, _params: &ParamValues, _execution: &Execution) -> LikelihoodResult<f64> {
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
        let likelihood = Likelihood::new([
            NllTerm::new("data", &model, &data, &accepted)
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
        let likelihood = Likelihood::new([BoundedQuadraticTerm {
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
        let likelihood = Likelihood::new([
            NllTerm::new("data", &model, &data, &accepted)
                .unwrap()
                .boxed(),
            RidgePenalty::new("ridge", ["scale"], 1.0).unwrap().boxed(),
        ])
        .unwrap();
        let err = likelihood
            .cross_section_integrals("ridge", &generated)
            .unwrap_err();

        assert!(matches!(err, LikelihoodError::NotIntensityTerm(ref name) if name == "ridge"));
    }
}
