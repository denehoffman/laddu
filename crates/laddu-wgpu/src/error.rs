use thiserror::Error;

pub type WgpuResult<T> = Result<T, WgpuError>;

#[derive(Debug, Error)]
pub enum WgpuError {
    #[error("no WGPU adapters are available")]
    NoAdapters,
    #[error("no WGPU adapter matches {0:?}")]
    AdapterNotFound(laddu_runtime::GpuDeviceSelector),
    #[error("the WGPU backend cannot satisfy a CUDA backend request")]
    CudaBackendRequested,
    #[error("adapter `{adapter}` does not support {precision:?} execution")]
    UnsupportedPrecision {
        adapter: String,
        precision: laddu_runtime::Precision,
    },
    #[error("failed to create a WGPU device: {0}")]
    RequestDevice(String),
    #[error("GPU memory budget must be greater than zero")]
    InvalidMemoryBudget,
    #[error(
        "GPU memory budget {available} bytes is too small; at least {required} bytes are required"
    )]
    MemoryBudgetTooSmall { required: usize, available: usize },
    #[error("the model does not contain a scalar kernel")]
    MissingScalarKernel,
    #[error("WGPU scalar lowering does not support {0}")]
    UnsupportedInstruction(String),
    #[error("parameter error: {0}")]
    Parameter(String),
    #[error("failed to map the WGPU result buffer: {0}")]
    BufferMap(String),
    #[error("failed while waiting for WGPU execution: {0}")]
    DevicePoll(String),
    #[error("event batch is missing required column `{0}`")]
    MissingEventColumn(String),
    #[error("GPU positive reduction failed at local event {0}")]
    NonPositiveEvent(usize),
}
