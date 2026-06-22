use laddu_expr::{Expr, ExprGraph};
use laddu_params::ParamLayout;
use thiserror::Error;

pub type CompileResult<T> = Result<T, CompileError>;

#[derive(Clone, Debug, Error)]
pub enum CompileError {
    #[error("unsupported backend feature: {0}")]
    Unsupported(&'static str),
}

#[derive(Clone, Debug)]
pub struct CompiledModel {
    graph: ExprGraph,
    root: Expr,
    params: ParamLayout,
}

impl CompiledModel {
    pub fn new(graph: ExprGraph, root: Expr, params: ParamLayout) -> Self {
        Self {
            graph,
            root,
            params,
        }
    }

    pub fn graph(&self) -> &ExprGraph {
        &self.graph
    }

    pub fn root(&self) -> Expr {
        self.root
    }

    pub fn params(&self) -> &ParamLayout {
        &self.params
    }
}
