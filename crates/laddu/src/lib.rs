mod error;

pub use error::{LadduError, LadduResult};
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
    BinaryOp, ComponentIndex, Expr, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprShape,
    ExprShapeError, ExprSourceKind, P4Component, UnaryOp, atan2, cis, complex, dot,
    event_p4_component, event_scalar, matmul, matrix, matrix_from_flat, matvec, polar_complex,
    solve, vector,
};
pub use laddu_runtime::{BinSpec, Comparison, DatasetBin, DatasetExprExt, Predicate};

#[cfg(feature = "amplitudes")]
pub use laddu_amplitudes as amplitudes;

#[cfg(feature = "likelihood")]
pub use laddu_likelihood as likelihood;

#[cfg(feature = "wgpu")]
pub use laddu_wgpu as wgpu;

pub mod prelude {
    pub use crate::{LadduError, LadduResult};
    #[cfg(feature = "amplitudes")]
    pub use laddu_amplitudes::{
        KMatrixError, KMatrixResult, KopfA0Channel, KopfA2Channel, KopfF0Channel, KopfF2Channel,
        KopfPi1Channel, KopfRhoChannel, blatt_weisskopf_barriers, breit_wigner, f_vector, k_matrix,
        k_matrix_with_background, kopf_a0, kopf_a0_resampled, kopf_a2, kopf_a2_resampled, kopf_f0,
        kopf_f0_resampled, kopf_f2, kopf_f2_resampled, kopf_pi1, kopf_rho, p_vector,
        p_vector_with_background, relativistic_breit_wigner, relativistic_breit_wigner_custom,
    };
    pub use laddu_autodiff::{AutodiffError, AutodiffMode, AutodiffPlan, AutodiffResult};
    pub use laddu_compile::{
        AlgebraicIdentityRule, CacheEntry, CachePlan, CachePolicy, CanonicalCsePass, CompileError,
        CompileOptions, CompileResult, CompiledModel, ComplexFactRule, ConstantFoldScalarRule,
        DependencyFacts, EvaluationClass, ExponentialRule, FactorCommonProductRule, GraphFacts,
        MatrixVectorRule, NodeFacts, NumberClass, OptimizationPass, OptimizationPipeline, Rewrite,
        RewriteContext, RewritePass, RewriteRule,
    };
    pub use laddu_data::data::{CacheStorage, Dataset};
    pub use laddu_data::io::Partitioning;
    pub use laddu_expr::parameter;
    pub use laddu_expr::parameters::{
        Bounds, FreeParamId, InitialSpec, ParamError, ParamId, ParamLayout, ParamRegistry,
        ParamState, ParamValues, Parameter,
    };
    pub use laddu_expr::{
        BinaryOp, ComponentIndex, Expr, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprShape,
        ExprShapeError, ExprSourceKind, P4Component, UnaryOp, atan2, cis, complex, dot,
        event_p4_component, event_scalar, matmul, matrix, matrix_from_flat, matvec, polar_complex,
        solve, vector,
    };
    pub use laddu_kernel::{KernelSpec, kernel};
    #[cfg(feature = "likelihood")]
    pub use laddu_likelihood::{
        CrossSectionIntegrals, LassoPenalty, Likelihood, LikelihoodError, LikelihoodEvaluation,
        LikelihoodName, LikelihoodProjection, LikelihoodResult, LikelihoodTerm, NllTerm,
        RidgePenalty,
    };
    pub use laddu_physics::channel::{Channel, Edge, EdgeHandle, Vertex, VertexHandle, VertexView};
    pub use laddu_physics::math::{
        BarrierKind, Sheet, blatt_weisskopf, blatt_weisskopf_custom, chew_mandelstam, q, rho,
        spherical_harmonic,
    };
    pub use laddu_runtime::{
        BinSpec, Comparison, CpuBackend, CpuBatchCache, CpuCachedBatch, CpuCachedDataset,
        CpuExecutionMode, CpuOptions, CpuPlan, CpuPreparedDataset, DatasetBin, DatasetExprExt,
        Device, Execution, ExecutionError, ExecutionOptions, GpuBackend, GpuDeviceSelector,
        GpuOptions, JitPolicy, Precision, Predicate, PreparedDatasetStats, ThreadPolicy,
        ValueGradient,
    };
}
