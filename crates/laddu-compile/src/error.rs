use laddu_expr::{ExprGraphError, parameters::ParamError};
use laddu_kernel::KernelIrError;
use thiserror::Error;

/// Result type for compilation and optimization operations.
pub type CompileResult<T> = Result<T, CompileError>;

/// Errors produced while analyzing or compiling an expression graph.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CompileError {
    /// Parameter collection or validation failed.
    #[error(transparent)]
    Params(#[from] ParamError),
    /// Expression graph validation failed.
    #[error(transparent)]
    Graph(#[from] ExprGraphError),
    /// Kernel IR construction or validation failed.
    #[error(transparent)]
    Kernel(#[from] KernelIrError),
    /// The lowered executable plan was internally inconsistent.
    #[error("invalid executable plan: {0}")]
    InvalidExecutablePlan(String),
    /// The requested compilation feature is not supported.
    #[error("unsupported compile feature: {0}")]
    Unsupported(&'static str),
}
