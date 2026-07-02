use thiserror::Error;

pub type LadduResult<T> = Result<T, LadduError>;

#[derive(Debug, Error)]
pub enum LadduError {
    #[error(transparent)]
    Autodiff(#[from] laddu_autodiff::AutodiffError),
    #[error(transparent)]
    Compile(#[from] laddu_compile::CompileError),
    #[error(transparent)]
    Data(#[from] laddu_data::LadduDataError),
    #[error(transparent)]
    Expression(#[from] laddu_expr::ExprError),
    #[error(transparent)]
    Kernel(#[from] laddu_kernel::KernelError),
    #[error(transparent)]
    Physics(#[from] laddu_physics::LadduPhysicsError),
    #[error(transparent)]
    Runtime(#[from] laddu_runtime::RuntimeError),
    #[cfg(feature = "amplitudes")]
    #[error(transparent)]
    Amplitude(#[from] laddu_amplitudes::AmplitudeError),
    #[cfg(feature = "likelihood")]
    #[error(transparent)]
    Likelihood(#[from] laddu_likelihood::LikelihoodError),
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    Wgpu(#[from] laddu_wgpu::WgpuError),
}
