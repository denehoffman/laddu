use laddu_compile::CompiledModel;
use laddu_data::data::Dataset;
use laddu_expr::parameters::ParamValues;
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
    #[error("{dataset} intensity must be positive for a likelihood term, got {value}")]
    NonPositiveIntensity { dataset: &'static str, value: f64 },
    #[error("accepted MC integral must be positive for acceptance correction, got {0}")]
    NonPositiveAcceptedIntegral(f64),
    #[error("luminosity must be positive for a cross section, got {0}")]
    NonPositiveLuminosity(f64),
}

#[derive(Clone, Debug)]
pub struct CpuIntensityLikelihood {
    plan: CpuPlan,
    terms: Vec<CpuIntensityTerm>,
    data_weight_sum: f64,
}

#[derive(Clone, Debug)]
pub struct CpuIntensityTerm {
    data: CpuCachedDataset,
    accepted_mc: CpuCachedDataset,
    data_weight_sum: f64,
}

impl CpuIntensityLikelihood {
    pub fn new(
        model: &CompiledModel,
        data: &Dataset,
        accepted_mc: &Dataset,
    ) -> LikelihoodResult<Self> {
        Self::from_datasets(model, [(data, accepted_mc)])
    }

    pub fn from_datasets<'a>(
        model: &CompiledModel,
        terms: impl IntoIterator<Item = (&'a Dataset, &'a Dataset)>,
    ) -> LikelihoodResult<Self> {
        let plan = CpuBackend.prepare(model);
        let terms = terms
            .into_iter()
            .map(|(data, accepted_mc)| CpuIntensityTerm::new(&plan, data, accepted_mc))
            .collect::<LikelihoodResult<_>>()?;
        Ok(Self::from_cached(plan, terms))
    }

    pub fn from_cached(plan: CpuPlan, terms: Vec<CpuIntensityTerm>) -> Self {
        let data_weight_sum = terms.iter().map(CpuIntensityTerm::data_weight_sum).sum();
        Self {
            plan,
            terms,
            data_weight_sum,
        }
    }

    pub fn plan(&self) -> &CpuPlan {
        &self.plan
    }

    pub fn terms(&self) -> &[CpuIntensityTerm] {
        &self.terms
    }

    pub fn data_weight_sum(&self) -> f64 {
        self.data_weight_sum
    }

    pub fn data_log_intensity_sum(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.terms.iter().try_fold(0.0, |sum, term| {
            Ok(sum + term.data_log_intensity_sum(&self.plan, params)?)
        })
    }

    pub fn accepted_normalization(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.terms.iter().try_fold(0.0, |sum, term| {
            Ok(sum + term.accepted_normalization(&self.plan, params)?)
        })
    }

    pub fn nll(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.terms
            .iter()
            .try_fold(0.0, |sum, term| Ok(sum + term.nll(&self.plan, params)?))
    }
}

impl CpuIntensityTerm {
    pub fn new(plan: &CpuPlan, data: &Dataset, accepted_mc: &Dataset) -> LikelihoodResult<Self> {
        Ok(Self::from_cached(
            plan.cache_dataset(data)?,
            plan.cache_dataset(accepted_mc)?,
        ))
    }

    pub fn from_cached(data: CpuCachedDataset, accepted_mc: CpuCachedDataset) -> Self {
        let data_weight_sum = data.sum_weights();
        Self {
            data,
            accepted_mc,
            data_weight_sum,
        }
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

    pub fn data_log_intensity_sum(
        &self,
        plan: &CpuPlan,
        params: &ParamValues,
    ) -> LikelihoodResult<f64> {
        Ok(plan.try_weighted_sum_cached(params, &self.data, |value| {
            positive_intensity("data", value).map(f64::ln)
        })?)
    }

    pub fn accepted_normalization(
        &self,
        plan: &CpuPlan,
        params: &ParamValues,
    ) -> LikelihoodResult<f64> {
        self.weighted_intensity_sum(plan, params, &self.accepted_mc, "accepted MC")
    }

    pub fn nll(&self, plan: &CpuPlan, params: &ParamValues) -> LikelihoodResult<f64> {
        let normalization =
            positive_integral("accepted MC", self.accepted_normalization(plan, params)?)?;
        let data_log_sum = self.data_log_intensity_sum(plan, params)?;
        Ok(self.data_weight_sum() * normalization.ln() - data_log_sum)
    }

    fn weighted_intensity_sum(
        &self,
        plan: &CpuPlan,
        params: &ParamValues,
        dataset: &CpuCachedDataset,
        name: &'static str,
    ) -> LikelihoodResult<f64> {
        Ok(plan.try_weighted_sum_cached(params, dataset, |value| positive_intensity(name, value))?)
    }
}

#[derive(Clone, Debug)]
pub struct CpuCrossSectionIntegrals {
    plan: CpuPlan,
    accepted_mc: CpuCachedDataset,
    generated_mc: CpuCachedDataset,
}

impl CpuCrossSectionIntegrals {
    pub fn new(
        model: &CompiledModel,
        accepted_mc: &Dataset,
        generated_mc: &Dataset,
    ) -> LikelihoodResult<Self> {
        let plan = CpuBackend.prepare(model);
        let accepted_mc = plan.cache_dataset(accepted_mc)?;
        let generated_mc = plan.cache_dataset(generated_mc)?;
        Ok(Self {
            plan,
            accepted_mc,
            generated_mc,
        })
    }

    pub fn from_cached(
        plan: CpuPlan,
        accepted_mc: CpuCachedDataset,
        generated_mc: CpuCachedDataset,
    ) -> Self {
        Self {
            plan,
            accepted_mc,
            generated_mc,
        }
    }

    pub fn plan(&self) -> &CpuPlan {
        &self.plan
    }

    pub fn accepted_mc(&self) -> &CpuCachedDataset {
        &self.accepted_mc
    }

    pub fn generated_mc(&self) -> &CpuCachedDataset {
        &self.generated_mc
    }

    pub fn accepted_integral(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.weighted_intensity_sum(params, &self.accepted_mc, "accepted MC")
    }

    pub fn generated_integral(&self, params: &ParamValues) -> LikelihoodResult<f64> {
        self.weighted_intensity_sum(params, &self.generated_mc, "generated MC")
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

    pub fn cross_section(
        &self,
        params: &ParamValues,
        accepted_yield: f64,
        luminosity: f64,
    ) -> LikelihoodResult<f64> {
        if luminosity <= 0.0 {
            return Err(LikelihoodError::NonPositiveLuminosity(luminosity));
        }
        Ok(self.acceptance_corrected_yield(params, accepted_yield)? / luminosity)
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

    #[test]
    fn nll_uses_data_and_accepted_mc_reductions() {
        let expr = event_scalar("x") * parameter!("scale", initial: 0.5);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let data = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_mc = weighted_dataset(&[(4.0, 1.0)]);
        let likelihood = CpuIntensityLikelihood::new(&model, &data, &accepted_mc).unwrap();

        let expected = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), expected);
    }

    #[test]
    fn simultaneous_nll_sums_dataset_terms() {
        let expr = event_scalar("x") * parameter!("scale", initial: 0.5);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let data_a = weighted_dataset(&[(2.0, 1.0), (3.0, 1.0)]);
        let accepted_a = weighted_dataset(&[(4.0, 1.0)]);
        let data_b = weighted_dataset(&[(5.0, 2.0)]);
        let accepted_b = weighted_dataset(&[(6.0, 3.0)]);
        let likelihood = CpuIntensityLikelihood::from_datasets(
            &model,
            [(&data_a, &accepted_a), (&data_b, &accepted_b)],
        )
        .unwrap();

        let term_a = 2.0 * 2.0_f64.ln() - 1.0_f64.ln() - 1.5_f64.ln();
        let term_b = 2.0 * 9.0_f64.ln() - 2.0 * 2.5_f64.ln();
        assert_relative_eq!(likelihood.nll(&params).unwrap(), term_a + term_b);
    }

    #[test]
    fn cross_section_integrals_use_accepted_and_generated_mc() {
        let expr = event_scalar("x") * parameter!("scale", initial: 2.0);
        let model = CompiledModel::from_expr(&expr).unwrap();
        let params = Arc::new(model.params().clone()).default_values();
        let accepted_mc = weighted_dataset(&[(1.0, 2.0), (2.0, 3.0)]);
        let generated_mc = weighted_dataset(&[(4.0, 5.0), (5.0, 7.0)]);
        let integrals = CpuCrossSectionIntegrals::new(&model, &accepted_mc, &generated_mc).unwrap();

        let accepted = 2.0 * 2.0 + 3.0 * 4.0;
        let generated = 5.0 * 8.0 + 7.0 * 10.0;
        assert_relative_eq!(integrals.accepted_integral(&params).unwrap(), accepted);
        assert_relative_eq!(integrals.generated_integral(&params).unwrap(), generated);
        assert_relative_eq!(integrals.acceptance(&params).unwrap(), accepted / generated);
        assert_relative_eq!(
            integrals.acceptance_corrected_yield(&params, 20.0).unwrap(),
            20.0 * generated / accepted
        );
        assert_relative_eq!(
            integrals.cross_section(&params, 20.0, 5.0).unwrap(),
            20.0 * generated / accepted / 5.0
        );
    }
}
