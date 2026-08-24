//! Execution backends, dataset queries, and reduction support for compiled laddu models.

mod backend;
mod cpu;
mod error;
mod execution;
#[cfg(feature = "jit")]
mod jit;
mod normalization;
mod preparation;
mod query;

pub use backend::{PreparedDataset, PreparedModel};
pub use cpu::*;
pub use error::{ExecutionError, RuntimeError, RuntimeResult};
pub use execution::{
    CpuOptions, Device, Execution, ExecutionOptions, GpuBackend, GpuDeviceSelector, GpuOptions,
    JitPolicy, NormalizationMode, Precision, ThreadPolicy,
};
pub use laddu_memory::{
    CapacitySource, DeviceIdentity, MemoryBudget, MemoryDecision, MemoryError, MemoryLease,
    MemoryPlan, MemoryPool, MemoryPoolReport, MemoryReport, MemoryResource, MemoryResourceKind,
    MemoryState, ProcessMemoryReport,
};
pub use normalization::{PreparedNormalization, PreparedNormalizationDiagnostics};
pub use query::{BinSpec, Comparison, DatasetBin, DatasetExprExt, IntervalClosure, Predicate};

use laddu_compile::CompiledModel;
use laddu_expr::ExprNode;

pub(crate) fn required_event_scalars(model: &CompiledModel) -> Vec<String> {
    let mut required = Vec::new();
    for node in model.graph().nodes() {
        if let ExprNode::EventScalar(name) = node
            && !required.iter().any(|required| required == name.as_ref())
        {
            required.push(name.to_string());
        }
    }
    required
}
