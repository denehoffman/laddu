use laddu_expr::{ExprGraphError, parameters::ParamError};
use thiserror::Error;

pub type CompileResult<T> = Result<T, CompileError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum CompileError {
    #[error(transparent)]
    Params(#[from] ParamError),
    #[error(transparent)]
    Graph(#[from] ExprGraphError),
    #[error("unsupported compile feature: {0}")]
    Unsupported(&'static str),
}
