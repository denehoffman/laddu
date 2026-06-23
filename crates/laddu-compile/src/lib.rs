use laddu_expr::{
    Expr, ExprGraph, ExprNode,
    parameters::{ParamError, ParamLayout, ParamRegistry},
};
use thiserror::Error;

pub type CompileResult<T> = Result<T, CompileError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum CompileError {
    #[error(transparent)]
    Params(#[from] ParamError),
    #[error("unsupported compile feature: {0}")]
    Unsupported(&'static str),
}

#[derive(Clone, Debug)]
pub struct CompiledModel {
    graph: ExprGraph,
    params: ParamLayout,
}

impl CompiledModel {
    pub fn from_expr(expr: &Expr) -> CompileResult<Self> {
        Self::from_graph(expr.to_graph())
    }

    pub fn from_graph(graph: ExprGraph) -> CompileResult<Self> {
        let params = collect_params(&graph)?;
        Ok(Self { graph, params })
    }

    pub fn graph(&self) -> &ExprGraph {
        &self.graph
    }

    pub fn params(&self) -> &ParamLayout {
        &self.params
    }
}

pub fn collect_params(graph: &ExprGraph) -> CompileResult<ParamLayout> {
    let mut registry = ParamRegistry::new();
    for node in graph.nodes() {
        match node {
            ExprNode::ScalarParam(spec) => {
                registry.register(spec.clone())?;
            }
            ExprNode::ComplexScalarParam { re, im } => {
                registry.register(re.clone())?;
                registry.register(im.clone())?;
            }
            _ => {}
        }
    }
    Ok(registry.layout()?)
}

#[cfg(test)]
mod tests {
    use laddu_expr::{complex, parameter};
    use num::complex::Complex64;

    use super::*;

    #[test]
    fn collects_parameters_in_graph_construction_order() {
        let model = (Complex64::new(0.0, 1.0) * parameter!("y", initial: 1.0, bounds: (0.0, 2.0))
            + parameter!("x"))
        .norm_sqr();
        let compiled = CompiledModel::from_expr(&model).unwrap();

        assert_eq!(
            compiled
                .params()
                .specs()
                .iter()
                .map(|spec| spec.name())
                .collect::<Vec<_>>(),
            vec!["y", "x"]
        );
    }

    #[test]
    fn merges_reused_compatible_parameters() {
        let x = parameter!("x", initial: 1.0);
        let model = x.clone() + x;
        let compiled = CompiledModel::from_expr(&model).unwrap();

        assert_eq!(compiled.params().len(), 1);
        assert_eq!(compiled.params().specs()[0].name(), "x");
    }

    #[test]
    fn rejects_reused_incompatible_parameters() {
        let model = parameter!("x", initial: 1.0) + parameter!("x", initial: 2.0);

        assert!(matches!(
            CompiledModel::from_expr(&model),
            Err(CompileError::Params(ParamError::ParameterConflict { name, .. }))
                if name == "x"
        ));
    }

    #[test]
    fn collects_complex_scalar_parameter_components() {
        let model = complex(parameter!("a_re"), parameter!("a_im"));
        let compiled = CompiledModel::from_expr(&model).unwrap();

        assert_eq!(
            compiled
                .params()
                .specs()
                .iter()
                .map(|spec| spec.name())
                .collect::<Vec<_>>(),
            vec!["a_re", "a_im"]
        );
    }
}
