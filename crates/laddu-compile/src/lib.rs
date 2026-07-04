pub mod cost;
mod error;
mod executable;
pub mod facts;
mod model;
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
