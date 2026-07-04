use laddu_expr::{ExprGraphError, parameters::ParamError};
use laddu_kernel::KernelIrError;
use thiserror::Error;

pub type CompileResult<T> = Result<T, CompileError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum CompileError {
    #[error(transparent)]
    Params(#[from] ParamError),
    #[error(transparent)]
    Graph(#[from] ExprGraphError),
    #[error(transparent)]
    Kernel(#[from] KernelIrError),
    #[error("invalid executable plan: {0}")]
    InvalidExecutablePlan(String),
    #[error("unsupported compile feature: {0}")]
    Unsupported(&'static str),
}
