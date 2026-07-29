use thiserror::Error;

/// A result returned by the top-level laddu facade.
pub type LadduResult<T> = Result<T, LadduError>;

#[derive(Clone, Debug, Error)]
/// An error produced by one of laddu's analysis subsystems.
///
/// Each variant preserves the concrete source error, allowing callers to
/// inspect it through [`std::error::Error::source`] or match the variant when
/// subsystem-specific recovery is useful.
pub enum LadduError {
    #[error(transparent)]
    /// Automatic-differentiation planning or construction failed.
    Autodiff(#[from] laddu_autodiff::AutodiffError),
    #[error(transparent)]
    /// Expression compilation or optimization failed.
    Compile(#[from] laddu_compile::CompileError),
    #[error(transparent)]
    /// Dataset construction, transformation, or I/O failed.
    Data(#[from] laddu_data::LadduDataError),
    #[error(transparent)]
    /// A symbolic expression or parameter specification was invalid.
    Expression(#[from] laddu_expr::ExprError),
    #[error(transparent)]
    /// Kernel validation or lowering failed.
    Kernel(#[from] laddu_kernel::KernelError),
    #[error(transparent)]
    /// A particle-physics value, relation, or topology was invalid.
    Physics(#[from] laddu_physics::LadduPhysicsError),
    #[error(transparent)]
    /// Model execution failed on the selected backend.
    Runtime(#[from] laddu_runtime::RuntimeError),
    #[cfg(feature = "amplitudes")]
    #[error(transparent)]
    /// Construction of an amplitude expression failed.
    Amplitude(#[from] laddu_amplitudes::AmplitudeError),
    #[cfg(feature = "likelihood")]
    #[error(transparent)]
    /// Likelihood assembly or evaluation failed.
    Likelihood(#[from] laddu_likelihood::LikelihoodError),
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    /// GPU discovery, compilation, or execution failed.
    Wgpu(#[from] laddu_wgpu::WgpuError),
}

#[cfg(feature = "amplitudes")]
pub use laddu_amplitudes::AmplitudeError;
pub use laddu_autodiff::AutodiffError;
pub use laddu_compile::CompileError;
pub use laddu_data::LadduDataError;
pub use laddu_expr::ExprError;
pub use laddu_kernel::KernelError;
#[cfg(feature = "likelihood")]
pub use laddu_likelihood::LikelihoodError;
pub use laddu_physics::LadduPhysicsError;
pub use laddu_runtime::RuntimeError;
#[cfg(feature = "wgpu")]
pub use laddu_wgpu::WgpuError;
