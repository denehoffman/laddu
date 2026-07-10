use thiserror::Error;

use laddu_compile::ReductionError;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error(transparent)]
    Execution(#[from] ExecutionError),
    #[error("event scalar `{0}` was requested, but no event lookup was provided")]
    MissingEventScalar(String),
    #[error("node #{index} expected {expected}, got {actual}")]
    TypeMismatch {
        index: usize,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("node #{index} has invalid shape: {message}")]
    InvalidShape { index: usize, message: String },
    #[error("matrix solve failed at node #{0}")]
    SingularMatrix(usize),
    #[error("event cache has {actual} slots, expected {expected}")]
    InvalidCache { expected: usize, actual: usize },
    #[error("event cache was built for a different cache layout")]
    InvalidCacheLayout,
    #[error("event scalar `{0}` was not found in the event batch schema")]
    MissingEventColumn(String),
    #[error("data error: {0}")]
    Data(String),
    #[error("parameter error: {0}")]
    Parameter(String),
    #[error("JIT kernel execution failed with status {0}")]
    JitExecution(i32),
    #[error(transparent)]
    Reduction(#[from] ReductionError),
    #[error("an MPI peer failed during distributed evaluation")]
    DistributedPeerFailure,
    #[error("WGPU execution failed: {0}")]
    Wgpu(String),
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ExecutionError {
    #[error("fixed thread count must be nonzero")]
    ZeroThreads,
    #[error("failed to create Rayon thread pool: {0}")]
    ThreadPool(String),
    #[error("f32 CPU execution currently requires scalar arithmetic over raw event caches")]
    UnsupportedCpuF32Model,
    #[error("f32 CPU gradients are not implemented yet")]
    UnsupportedCpuF32Gradient,
    #[error("GPU backend {0:?} is not available")]
    GpuUnavailable(crate::GpuBackend),
    #[error("GPU gradients are not implemented yet")]
    UnsupportedGpuGradient,
    #[error("reverse-mode autodiff is currently only supported for f64 CPU execution")]
    UnsupportedReverseAutodiff,
    #[error("CPU JIT execution was requested but the `jit` feature is unavailable")]
    JitUnavailable,
}
