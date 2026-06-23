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
    BinaryOp, Expr, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp, cis,
    complex, dot, event_scalar, matmul, matrix, matvec, polar_complex, solve, vector,
};

#[cfg(feature = "amplitudes")]
pub use laddu_amplitudes as amplitudes;

#[cfg(feature = "likelihood")]
pub use laddu_likelihood as likelihood;

#[cfg(feature = "wgpu")]
pub use laddu_wgpu as wgpu;

pub mod prelude {
    pub use laddu_autodiff::{AutodiffMode, AutodiffPlan};
    pub use laddu_compile::{CompileError, CompileResult, CompiledModel};
    pub use laddu_expr::parameter;
    pub use laddu_expr::parameters::{
        Bounds, FreeParamId, InitialSpec, ParamError, ParamId, ParamLayout, ParamRegistry,
        ParamState, ParamValues, Parameter,
    };
    pub use laddu_expr::{
        BinaryOp, Expr, ExprGraph, ExprId, ExprMetadata, ExprNode, ExprSourceKind, UnaryOp, cis,
        complex, dot, event_scalar, matmul, matrix, matvec, polar_complex, solve, vector,
    };
    pub use laddu_kernel::{KernelSpec, kernel};
    pub use laddu_runtime::{CpuBackend, CpuPlan};
}
