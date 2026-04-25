use ganesh::traits::{CostFunction, Gradient, LogDensity};
use laddu_core::{LadduError, LadduResult};
use nalgebra::DVector;

use crate::{
    likelihood::{LikelihoodTerm, StochasticNLL},
    optimize::MaybeThreadPool,
    LikelihoodExpression, NLL,
};

impl CostFunction<MaybeThreadPool, LadduError> for NLL {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))
    }
}
impl Gradient<MaybeThreadPool, LadduError> for NLL {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        args.install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for NLL {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        Ok(-args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
    }
}

impl CostFunction<MaybeThreadPool, LadduError> for StochasticNLL {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))
    }
}
impl Gradient<MaybeThreadPool, LadduError> for StochasticNLL {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        args.install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for StochasticNLL {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        Ok(-args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
    }
}

impl CostFunction<MaybeThreadPool, LadduError> for LikelihoodExpression {
    fn evaluate(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))
    }
}
impl Gradient<MaybeThreadPool, LadduError> for LikelihoodExpression {
    fn gradient(
        &self,
        parameters: &DVector<f64>,
        args: &MaybeThreadPool,
    ) -> LadduResult<DVector<f64>> {
        args.install(|| LikelihoodTerm::evaluate_gradient(self, parameters.into()))
    }
}
impl LogDensity<MaybeThreadPool, LadduError> for LikelihoodExpression {
    fn log_density(&self, parameters: &DVector<f64>, args: &MaybeThreadPool) -> LadduResult<f64> {
        Ok(-args.install(|| LikelihoodTerm::evaluate(self, parameters.into()))?)
    }
}

#[cfg(test)]
mod tests {
    use super::MaybeThreadPool;

    #[test]
    fn maybe_thread_pool_handles_repeated_short_installs() {
        let pool = MaybeThreadPool::new(2);
        let total = (0usize..64)
            .map(|index| {
                pool.install(|| Ok(index + 1))
                    .expect("repeated install should succeed")
            })
            .sum::<usize>();
        assert_eq!(total, (1usize..=64).sum::<usize>());
    }
}
