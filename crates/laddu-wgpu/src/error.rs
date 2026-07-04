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
}
