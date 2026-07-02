use thiserror::Error;

pub type KernelResult<T> = Result<T, KernelError>;
pub type KernelIrError = KernelError;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    #[error("kernel IR contains no values")]
    Empty,
    #[error("kernel root value {root} is out of bounds for {len} values")]
    RootOutOfBounds { root: usize, len: usize },
    #[error("kernel value {value} references non-prior value {operand}")]
    InvalidOperand { value: usize, operand: usize },
    #[error("kernel value {value} has no operands for {operation}")]
    EmptyOperands {
        value: usize,
        operation: &'static str,
    },
    #[error("kernel value {value} has kind {actual:?}, but its instruction produces {expected:?}")]
    KindMismatch {
        value: usize,
        expected: crate::ir::KernelValueKind,
        actual: crate::ir::KernelValueKind,
    },
    #[error("kernel value {value} has class {actual:?}, but its dependencies require {expected:?}")]
    ClassMismatch {
        value: usize,
        expected: crate::ir::KernelValueClass,
        actual: crate::ir::KernelValueClass,
    },
    #[error("kernel value {value} has invalid operands for {operation}: {message}")]
    InvalidShape {
        value: usize,
        operation: &'static str,
        message: String,
    },
}
