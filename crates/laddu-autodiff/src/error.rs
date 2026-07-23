use crate::AutodiffMode;
use thiserror::Error;

/// Result type for automatic-differentiation operations.
pub type AutodiffResult<T> = Result<T, AutodiffError>;

/// Errors produced while planning or generating derivatives.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AutodiffError {
    /// The selected differentiation mode is not implemented.
    #[error("autodiff mode {0:?} is not implemented")]
    UnsupportedMode(AutodiffMode),
    /// Generated kernel IR was invalid.
    #[error(transparent)]
    Kernel(#[from] laddu_kernel::KernelError),
    /// A primal kernel instruction could not be differentiated.
    #[error("cannot differentiate kernel instruction: {0}")]
    InvalidKernel(String),
}
