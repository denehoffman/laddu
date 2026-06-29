pub use laddu_autodiff as autodiff;
pub use laddu_compile as compile;
pub use laddu_data as data;
pub use laddu_expr as expr;
pub use laddu_expr::parameters as params;
pub use laddu_kernel as kernel;
pub use laddu_physics as physics;
pub use laddu_runtime as runtime;

pub use laddu_expr::parameter;
pub use laddu_expr::parameters::Parameter;
pub use laddu_expr::{
    BinaryOp, Expr, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprSourceKind, P4Component,
    UnaryOp, atan2, cis, complex, dot, event_p4_component, event_scalar, matmul, matrix, matvec,
    polar_complex, solve, vector,
};

#[cfg(feature = "amplitudes")]
pub use laddu_amplitudes as amplitudes;

#[cfg(feature = "likelihood")]
pub use laddu_likelihood as likelihood;

#[cfg(feature = "wgpu")]
pub use laddu_wgpu as wgpu;

pub mod prelude {
    #[cfg(feature = "amplitudes")]
    pub use laddu_amplitudes::{
        breit_wigner_m, breit_wigner_s, relativistic_breit_wigner_custom,
        relativistic_breit_wigner_custom_s, relativistic_breit_wigner_m,
        relativistic_breit_wigner_s,
    };
    pub use laddu_autodiff::{AutodiffMode, AutodiffPlan};
    pub use laddu_compile::{
        AlgebraicIdentityRule, CacheEntry, CachePlan, CachePolicy, CanonicalCsePass, CompileError,
        CompileOptions, CompileResult, CompiledModel, ComplexFactRule, ConstantFoldScalarRule,
        DependencyFacts, EvaluationClass, ExponentialRule, FactorCommonProductRule, GraphFacts,
        MatrixVectorRule, NodeFacts, NumberClass, OptimizationPass, OptimizationPipeline, Rewrite,
        RewriteContext, RewritePass, RewriteRule,
    };
    pub use laddu_expr::parameter;
    pub use laddu_expr::parameters::{
        Bounds, FreeParamId, InitialSpec, ParamError, ParamId, ParamLayout, ParamRegistry,
        ParamState, ParamValues, Parameter,
    };
    pub use laddu_expr::{
        BinaryOp, Expr, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprSourceKind, P4Component,
        UnaryOp, atan2, cis, complex, dot, event_p4_component, event_scalar, matmul, matrix,
        matvec, polar_complex, solve, vector,
    };
    pub use laddu_kernel::{KernelSpec, kernel};
    #[cfg(feature = "likelihood")]
    pub use laddu_likelihood::{
        CpuCrossSectionIntegrals, CpuLassoPenalty, CpuLikelihood, CpuLikelihoodTerm, CpuNllTerm,
        CpuRidgePenalty, LikelihoodError, LikelihoodName, LikelihoodResult,
    };
    pub use laddu_physics::channel::{Channel, Edge, EdgeHandle, Vertex, VertexHandle, VertexView};
    pub use laddu_physics::math::{
        BarrierKind, Sheet, blatt_weisskopf, blatt_weisskopf_custom, q_m, q_s, rho_m, rho_s,
        spherical_harmonic,
    };
    pub use laddu_runtime::{CpuBackend, CpuBatchCache, CpuCachedBatch, CpuCachedDataset, CpuPlan};
}
