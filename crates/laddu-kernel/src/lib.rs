//! Validated intermediate representations and specifications for executable kernels.

mod error;
/// Typed, topologically ordered intermediate representations for kernels.
pub mod ir;
mod spec;

pub use error::{KernelError, KernelIrError, KernelResult};
pub use spec::{CacheName, KernelName, KernelSpec, kernel};
