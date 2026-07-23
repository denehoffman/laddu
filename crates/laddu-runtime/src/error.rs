use thiserror::Error;

use laddu_compile::ReductionError;

/// Result type returned by runtime operations.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Error produced while preparing or evaluating a model.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RuntimeError {
    /// Execution-context configuration failed.
    #[error(transparent)]
    Execution(#[from] ExecutionError),
    /// An expression requested an event scalar without an event lookup.
    #[error("event scalar `{0}` was requested, but no event lookup was provided")]
    MissingEventScalar(String),
    /// An expression node produced an unexpected value type.
    #[error("node #{index} expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expression-node index.
        index: usize,
        /// Expected value type.
        expected: &'static str,
        /// Actual value type.
        actual: &'static str,
    },
    /// An expression node had an invalid shape.
    #[error("node #{index} has invalid shape: {message}")]
    InvalidShape {
        /// Expression-node index.
        index: usize,
        /// Description of the shape mismatch.
        message: String,
    },
    /// A matrix factorization or solve failed.
    #[error("matrix solve failed at node #{0}")]
    SingularMatrix(usize),
    /// A cache contained the wrong number of slots.
    #[error("event cache has {actual} slots, expected {expected}")]
    InvalidCache {
        /// Expected slot count.
        expected: usize,
        /// Actual slot count.
        actual: usize,
    },
    /// A cache was created for an incompatible layout.
    #[error("event cache was built for a different cache layout")]
    InvalidCacheLayout,
    /// A required event column was absent from the schema.
    #[error("event scalar `{0}` was not found in the event batch schema")]
    MissingEventColumn(String),
    /// Dataset access failed.
    #[error("data error: {0}")]
    Data(String),
    /// Parameter validation or lookup failed.
    #[error("parameter error: {0}")]
    Parameter(String),
    /// A JIT-compiled kernel returned a failure status.
    #[error("JIT kernel execution failed with status {0}")]
    JitExecution(i32),
    /// A reduction transform failed.
    #[error(transparent)]
    Reduction(#[from] ReductionError),
    /// Another distributed rank failed during evaluation.
    #[error("an MPI peer failed during distributed evaluation")]
    DistributedPeerFailure,
    /// WebGPU preparation or execution failed.
    #[error("WGPU execution failed: {0}")]
    Wgpu(String),
}

/// Error produced while resolving execution options.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ExecutionError {
    /// A fixed thread policy requested zero threads.
    #[error("fixed thread count must be nonzero")]
    ZeroThreads,
    /// A CPU worker pool could not be constructed.
    #[error("failed to create Rayon thread pool: {0}")]
    ThreadPool(String),
    /// The model cannot currently execute on the 32-bit CPU path.
    #[error("f32 CPU execution currently requires scalar arithmetic over raw event caches")]
    UnsupportedCpuF32Model,
    /// CPU gradients are not available at 32-bit precision.
    #[error("f32 CPU gradients are not implemented yet")]
    UnsupportedCpuF32Gradient,
    /// The requested GPU implementation is unavailable.
    #[error("GPU backend {0:?} is not available")]
    GpuUnavailable(crate::GpuBackend),
    /// GPU gradients are not currently available.
    #[error("GPU gradients are not implemented yet")]
    UnsupportedGpuGradient,
    /// The requested reverse-mode configuration is unsupported.
    #[error("reverse-mode autodiff is currently only supported for f64 CPU execution")]
    UnsupportedReverseAutodiff,
    /// JIT execution was requested without the `jit` feature.
    #[error("CPU JIT execution was requested but the `jit` feature is unavailable")]
    JitUnavailable,
}
