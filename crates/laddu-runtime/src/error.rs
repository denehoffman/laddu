use thiserror::Error;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error(transparent)]
    Execution(#[from] CpuExecutionError),
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
    #[error("an MPI peer failed during distributed evaluation")]
    DistributedPeerFailure,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CpuExecutionError {
    #[error("fixed thread count must be nonzero")]
    ZeroThreads,
    #[error("failed to create Rayon thread pool: {0}")]
    ThreadPool(String),
}
