//! Analysis, optimization, cache planning, and kernel lowering for expression graphs.

mod cas;
/// Static operation-cost analysis.
pub mod cost;
mod error;
mod executable;
/// Per-node value, number, and dependency analysis.
pub mod facts;
mod graph_utils;
mod model;
mod normalization;
mod reduction;

pub use cas::{OptimizationBudget, OptimizationDiagnostics};
pub use cost::{LifecycleCost, OptimizationCost};
pub use error::{CompileError, CompileResult};
pub use executable::{CacheInput, ExecutablePlan, SolveComponentPlan, SolveRowMatrixPlan};
pub use facts::{DependencyFacts, EvaluationClass, GraphFacts, NodeFacts, NumberClass};
pub use model::{
    CacheEntry, CacheLayout, CachePlan, CachePolicy, CacheStorageKind, CompileOptions,
    CompiledModel, CompiledQuery, collect_params,
};
pub use normalization::{
    NormalizationDiagnostics, NormalizationFallbackReason, NormalizationPlan, NormalizationStrategy,
};
pub use reduction::{ReductionError, ReductionOutput, ReductionPlan, ReductionTransform};
