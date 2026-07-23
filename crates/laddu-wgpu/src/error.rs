use thiserror::Error;

/// Result type returned by WebGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Error produced while preparing or executing a WebGPU kernel.
#[derive(Debug, Error)]
pub enum WgpuError {
    /// No WebGPU adapters were discovered.
    #[error("no WGPU adapters are available")]
    NoAdapters,
    /// No adapter matched the requested selector.
    #[error("no WGPU adapter matches {0:?}")]
    AdapterNotFound(crate::WgpuDeviceSelector),
    /// The selected adapter cannot execute at the requested precision.
    #[error("adapter `{adapter}` does not support {precision:?} execution")]
    UnsupportedPrecision {
        /// Selected adapter name.
        adapter: String,
        /// Requested precision.
        precision: crate::WgpuPrecision,
    },
    /// Scalar-kernel lowering does not implement the resolved precision.
    #[error("WGPU scalar kernels do not yet implement {0:?} arithmetic")]
    UnsupportedKernelPrecision(crate::WgpuPrecision),
    /// WebGPU device creation failed.
    #[error("failed to create a WGPU device: {0}")]
    RequestDevice(String),
    /// A zero-byte memory budget was supplied.
    #[error("GPU memory budget must be greater than zero")]
    InvalidMemoryBudget,
    /// The memory budget cannot hold the minimum required buffers.
    #[error(
        "GPU memory budget {available} bytes is too small; at least {required} bytes are required"
    )]
    MemoryBudgetTooSmall {
        /// Minimum number of required bytes.
        required: usize,
        /// Configured number of available bytes.
        available: usize,
    },
    /// The compiled model has no scalar kernel.
    #[error("the model does not contain a scalar kernel")]
    MissingScalarKernel,
    /// The scalar kernel contains an instruction unsupported by WebGPU lowering.
    #[error("WGPU scalar lowering does not support {0}")]
    UnsupportedInstruction(String),
    /// A fused matrix solve exceeds the backend's dimension limit.
    #[error(
        "WGPU fused solves support matrices through 16x16, but the model requires {dimension}x{dimension}; use CPU execution"
    )]
    SolveDimensionTooLarge {
        /// Required square-matrix dimension.
        dimension: usize,
    },
    /// Parameter validation or lookup failed.
    #[error("parameter error: {0}")]
    Parameter(String),
    /// Mapping a GPU result buffer for host access failed.
    #[error("failed to map the WGPU result buffer: {0}")]
    BufferMap(String),
    /// Waiting for GPU work to complete failed.
    #[error("failed while waiting for WGPU execution: {0}")]
    DevicePoll(String),
    /// A required event column was absent.
    #[error("event batch is missing required column `{0}`")]
    MissingEventColumn(String),
    /// A positive-valued reduction encountered a non-positive event.
    #[error("GPU positive reduction failed at local event {0}")]
    NonPositiveEvent(usize),
    /// A matrix solve was singular for an event.
    #[error("GPU solve is singular at local event {0}")]
    SingularMatrixEvent(usize),
}
