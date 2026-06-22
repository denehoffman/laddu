pub use laddu_expr::{ExprError, ExprGraph, ExprResult};
pub use num::complex::Complex64;

use laddu_expr::Expr;
use laddu_params::ParamValues;

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardGradient {
    pub value: Complex64,
    pub gradient: Vec<Complex64>,
}

pub fn forward_gradient(
    graph: &ExprGraph,
    root: Expr,
    params: &ParamValues,
) -> ExprResult<ForwardGradient> {
    let (value, gradient) = graph.evaluate_with_gradient(root, params)?;
    Ok(ForwardGradient { value, gradient })
}
