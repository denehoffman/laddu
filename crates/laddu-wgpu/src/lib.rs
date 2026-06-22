use thiserror::Error;

pub type WgpuResult<T> = Result<T, WgpuError>;

#[derive(Clone, Debug, Error)]
pub enum WgpuError {
    #[error("WGPU backend is not implemented yet")]
    NotImplemented,
}

#[derive(Clone, Debug, Default)]
pub struct WgpuBackend;
