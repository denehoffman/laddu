use crate::AutodiffMode;
use thiserror::Error;

pub type AutodiffResult<T> = Result<T, AutodiffError>;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AutodiffError {
    #[error("autodiff mode {0:?} is not implemented")]
    UnsupportedMode(AutodiffMode),
    #[error(transparent)]
    Kernel(#[from] laddu_kernel::KernelError),
    #[error("cannot differentiate kernel instruction: {0}")]
    InvalidKernel(String),
}
