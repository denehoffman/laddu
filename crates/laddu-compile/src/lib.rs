//! Analysis, optimization, cache planning, and kernel lowering for expression graphs.

/// Static operation-cost analysis.
pub mod cost;
mod error;
mod executable;
/// Per-node value, number, and dependency analysis.
pub mod facts;
mod model;
/// Extensible expression-graph optimization passes and rewrite rules.
pub mod optimize;
mod reduction;

pub use cost::OptimizationCost;
pub use error::{CompileError, CompileResult};
pub use executable::{ExecutablePlan, SolveComponentPlan, SolveRowMatrixPlan};
pub use facts::{DependencyFacts, EvaluationClass, GraphFacts, NodeFacts, NumberClass};
pub use model::*;
pub use optimize::{
    AlgebraicIdentityRule, CanonicalCsePass, ComplexFactRule, ConjugationRule,
    ConstantFoldScalarRule, CostGatePass, ExponentialRule, FactorCommonProductRule,
    MatrixVectorRule, NormSqrExpansionRule, OptimizationPass, OptimizationPipeline, Rewrite,
    RewriteContext, RewritePass, RewriteRule,
};
pub use reduction::{ReductionError, ReductionOutput, ReductionPlan, ReductionTransform};
