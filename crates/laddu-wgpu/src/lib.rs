mod error;
pub use error::{WgpuError, WgpuResult};

#[derive(Clone, Debug, Default)]
pub struct WgpuBackend;
