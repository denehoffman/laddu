use thiserror::Error;

/// Result type for kernel construction and validation.
pub type KernelResult<T> = Result<T, KernelError>;
/// Backward-compatible name for [`KernelError`] in IR APIs.
pub type KernelIrError = KernelError;

/// Errors produced while constructing or validating kernel IR.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    /// The IR contained no values.
    #[error("kernel IR contains no values")]
    Empty,
    /// The scalar root identifier was outside the value array.
    #[error("kernel root value {root} is out of bounds for {len} values")]
    RootOutOfBounds {
        /// Invalid root index.
        root: usize,
        /// Number of IR values.
        len: usize,
    },
    /// A gradient output identifier was outside the value array.
    #[error("gradient output value {output} is out of bounds for {len} values")]
    GradientOutOfBounds {
        /// Invalid output index.
        output: usize,
        /// Number of IR values.
        len: usize,
    },
    /// A cache kernel had no outputs.
    #[error("cache kernel contains no outputs")]
    EmptyCacheOutputs,
    /// A cache output identifier was outside the value array.
    #[error("cache output value {output} is out of bounds for {len} values")]
    CacheOutputOutOfBounds {
        /// Invalid output index.
        output: usize,
        /// Number of IR values.
        len: usize,
    },
    /// A gradient output was not real-valued.
    #[error("gradient output value {output} must be real, but has kind {actual:?}")]
    GradientKindMismatch {
        /// Output index.
        output: usize,
        /// Actual output kind.
        actual: crate::ir::KernelValueKind,
    },
    /// An instruction referenced a value that did not precede it.
    #[error("kernel value {value} references non-prior value {operand}")]
    InvalidOperand {
        /// Index of the invalid instruction.
        value: usize,
        /// Invalid operand index.
        operand: usize,
    },
    /// An instruction requiring operands was empty.
    #[error("kernel value {value} has no operands for {operation}")]
    EmptyOperands {
        /// Index of the invalid instruction.
        value: usize,
        /// Operation being validated.
        operation: &'static str,
    },
    /// A value's declared kind did not match its instruction.
    #[error("kernel value {value} has kind {actual:?}, but its instruction produces {expected:?}")]
    KindMismatch {
        /// Index of the invalid value.
        value: usize,
        /// Kind produced by the instruction.
        expected: crate::ir::KernelValueKind,
        /// Declared kind.
        actual: crate::ir::KernelValueKind,
    },
    /// A value's declared evaluation class did not match its dependencies.
    #[error("kernel value {value} has class {actual:?}, but its dependencies require {expected:?}")]
    ClassMismatch {
        /// Index of the invalid value.
        value: usize,
        /// Class required by the dependencies.
        expected: crate::ir::KernelValueClass,
        /// Declared class.
        actual: crate::ir::KernelValueClass,
    },
    /// Operand shapes were incompatible with an instruction.
    #[error("kernel value {value} has invalid operands for {operation}: {message}")]
    InvalidShape {
        /// Index of the invalid value.
        value: usize,
        /// Operation being validated.
        operation: &'static str,
        /// Description of the incompatibility.
        message: String,
    },
}
