mod backend;
mod cpu;
mod error;
mod execution;
#[cfg(feature = "jit")]
mod jit;

pub use backend::{PreparedDataset, PreparedModel};
pub use cpu::*;
pub use error::{ExecutionError, RuntimeError, RuntimeResult};
pub use execution::{
    CpuOptions, Device, Execution, ExecutionOptions, GpuBackend, GpuDeviceSelector, GpuOptions,
    JitPolicy, Precision, ThreadPolicy,
};
