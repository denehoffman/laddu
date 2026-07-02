mod error;
pub mod ir;
mod spec;

pub use error::{KernelError, KernelIrError, KernelResult};
pub use spec::{CacheName, KernelName, KernelSpec, kernel};
