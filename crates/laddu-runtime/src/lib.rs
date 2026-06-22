use laddu_autodiff::{ForwardGradient, forward_gradient};
use laddu_compile::CompiledModel;
use laddu_expr::ExprResult;
use laddu_params::ParamValues;
use num::complex::Complex64;

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn evaluate(&self, model: &CompiledModel, params: &ParamValues) -> ExprResult<Complex64> {
        model.graph().evaluate(model.root(), params)
    }

    pub fn evaluate_with_gradient(
        &self,
        model: &CompiledModel,
        params: &ParamValues,
    ) -> ExprResult<ForwardGradient> {
        forward_gradient(model.graph(), model.root(), params)
    }
}
