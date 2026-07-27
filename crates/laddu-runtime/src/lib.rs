//! Execution backends, dataset queries, and reduction support for compiled laddu models.

mod backend;
mod cpu;
mod error;
mod execution;
#[cfg(feature = "jit")]
mod jit;
mod query;

pub use backend::{PreparedDataset, PreparedModel};
pub use cpu::*;
pub use error::{ExecutionError, RuntimeError, RuntimeResult};
pub use execution::{
    CpuOptions, Device, Execution, ExecutionOptions, GpuBackend, GpuDeviceSelector, GpuOptions,
    JitPolicy, Precision, ThreadPolicy,
};
pub use laddu_memory::{
    CapacitySource, DeviceIdentity, MemoryBudget, MemoryDecision, MemoryError, MemoryLease,
    MemoryPlan, MemoryPool, MemoryPoolReport, MemoryReport, MemoryResource, MemoryResourceKind,
    MemoryState, ProcessMemoryReport,
};
pub use query::{BinSpec, Comparison, DatasetBin, DatasetExprExt, IntervalClosure, Predicate};
