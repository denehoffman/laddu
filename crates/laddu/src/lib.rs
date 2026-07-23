//! Tools for constructing, compiling, and evaluating amplitude-analysis models.
//!
//! laddu exposes symbolic expressions, event datasets, particle and decay
//! topology utilities, execution backends, and optional likelihood, fitting,
//! generation, amplitude, and GPU support through one facade crate. Most
//! applications can import [`prelude`] and enable only the Cargo features they
//! need.
//!
//! # Example
//!
//! ```
//! use laddu::prelude::*;
//!
//! let mass = parameter!("mass");
//! let intensity = (mass.clone() * mass + 1.0).named("intensity");
//! let model = CompiledModel::from_expr(&intensity)?;
//!
//! assert_eq!(model.params().len(), 1);
//! # Ok::<(), CompileError>(())
//! ```

mod error;

#[cfg(feature = "python")]
/// Shared implementation of laddu's Python extension modules.
///
/// This module is public so the small Maturin distribution crates can expand
/// [`laddu_python_module!`]. Rust applications should use the crate-level Rust
/// API instead.
pub mod python;

pub use error::{LadduError, LadduResult};
pub use laddu_autodiff as autodiff;
pub use laddu_compile as compile;
pub use laddu_data as data;
pub use laddu_expr as expr;
pub use laddu_expr::parameters as params;
pub use laddu_kernel as kernel;
pub use laddu_physics as physics;
pub use laddu_runtime as runtime;

#[cfg(feature = "generation")]
pub use laddu_generation as generation;

pub use laddu_expr::parameter;
pub use laddu_expr::parameters::Parameter;
pub use laddu_expr::{
    BinaryOp, ColorPreset, ComponentIndex, DisplayColor, Expr, ExprGraph, ExprGraphDotDisplay,
    ExprGraphTreeDisplay, ExprId, ExprMetadata, ExprNode, ExprNodeKind, ExprShape, ExprShapeError,
    ExprSourceKind, NodeSelector, NodeStyle, NodeStyleRule, P4Component, RepeatedSubtrees, UnaryOp,
    acos, atan2, cis, complex, dot, event_p4_component, event_scalar, matmul, matrix,
    matrix_from_flat, matvec, polar_complex, solve, vector,
};
pub use laddu_runtime::{
    BinSpec, Comparison, DatasetBin, DatasetExprExt, IntervalClosure, Predicate,
};

#[cfg(feature = "svg")]
pub use laddu_expr::GraphRenderError;

#[cfg(feature = "amplitudes")]
pub use laddu_amplitudes as amplitudes;

#[cfg(feature = "likelihood")]
pub use laddu_likelihood as likelihood;

#[cfg(feature = "fit")]
pub use laddu_fit as fit;

#[cfg(feature = "wgpu")]
pub use laddu_wgpu as wgpu;

pub mod prelude {
    //! The experiment-neutral public API for building analyses.
    //!
    //! Collaboration-specific schemas, cuts, formats, and conventions belong
    //! in downstream crates built from these general-purpose primitives.

    pub use crate::{LadduError, LadduResult};
    #[cfg(feature = "amplitudes")]
    pub use laddu_amplitudes::{
        AmplitudeError, AmplitudeResult, KMatrixError, KMatrixResult, blatt_weisskopf_barriers,
        breit_wigner, f_vector, k_matrix, k_matrix_with_background, p_vector,
        p_vector_with_background, relativistic_breit_wigner, relativistic_breit_wigner_custom,
    };
    pub use laddu_autodiff::AutodiffMode;
    pub use laddu_compile::{CompileError, CompileOptions, CompileResult, CompiledModel};
    pub use laddu_data::{
        LadduDataError, LadduDataResult,
        data::{CacheStorage, Dataset, EventBatch, EventBatchBuilder, OwnedEvent},
        io::{
            EventSink, EventSource, Partitioning, ReadPlan, SourceCapabilities, WritePlan,
            memory::{MemorySink, MemorySource},
            parquet::{ParquetSink, ParquetSource},
            root::{RootSink, RootSource},
        },
        schema::{ColumnType, Schema},
    };
    #[cfg(feature = "svg")]
    pub use laddu_expr::GraphRenderError;
    pub use laddu_expr::parameter;
    pub use laddu_expr::parameters::{
        Bounds, FreeParamId, InitialSpec, ParamError, ParamId, ParamLayout, ParamRegistry,
        ParamState, ParamValues, Parameter,
    };
    pub use laddu_expr::{
        BinaryOp, ColorPreset, ComponentIndex, DisplayColor, Expr, ExprGraph, ExprGraphDotDisplay,
        ExprGraphTreeDisplay, ExprId, ExprMetadata, ExprNode, ExprNodeKind, ExprShape,
        ExprShapeError, ExprSourceKind, NodeSelector, NodeStyle, NodeStyleRule, P4Component,
        RepeatedSubtrees, UnaryOp, acos, atan2, cis, complex, dot, event_p4_component,
        event_scalar, matmul, matrix, matrix_from_flat, matvec, polar_complex, solve, vector,
    };
    #[cfg(feature = "fit")]
    pub use laddu_fit::{FitError, FitProblem, FitResult, StochasticFitProblem, ganesh};
    #[cfg(feature = "generation")]
    pub use laddu_generation::{
        ChannelGenerator, EnvelopeKind, EnvelopeMode, EnvelopeOverflow, GenerationError,
        GenerationReport, GenerationResult, InitialMomentum, InitialMomentumResult, MassProposal,
        ModelEvaluator, NamedMass, NamedMomentum, ProposalResult, ProposalRng,
        ScalarProposalResult, ScalarSource, TComponent, TDistribution, TwoBodyScattering,
        UnweightedConfig, VertexProposal, WeightedConfig,
    };
    #[cfg(feature = "likelihood")]
    pub use laddu_likelihood::{
        CrossSectionIntegrals, ExtendedNllTerm, LassoPenalty, Likelihood, LikelihoodError,
        LikelihoodEvaluation, LikelihoodName, LikelihoodProjection, LikelihoodResult,
        LikelihoodTerm, NllTerm, Objective, RidgePenalty, StochasticObjective,
    };
    pub use laddu_physics::quantum::builtin as particles;
    pub use laddu_physics::{
        LadduPhysicsError, LadduPhysicsResult,
        channel::{Channel, Edge, EdgeHandle, Vertex, VertexHandle, VertexView},
        clebsch_gordan,
        histogram::Histogram,
        j, l, m,
        math::*,
        quantum::*,
        s,
        vectors::{RealVec3, RealVec4, Vec3, Vec4},
    };
    pub use laddu_runtime::{
        BinSpec, Comparison, CpuOptions, DatasetBin, DatasetExprExt, Device, Execution,
        ExecutionError, ExecutionOptions, GpuBackend, GpuDeviceSelector, GpuOptions,
        IntervalClosure, JitPolicy, Precision, Predicate, RuntimeError, RuntimeResult,
        ThreadPolicy,
    };
}
