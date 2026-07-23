use laddu_expr::ExprShapeError;
use laddu_physics::LadduPhysicsError;
use thiserror::Error;

/// Result type for amplitude construction.
pub type AmplitudeResult<T> = Result<T, AmplitudeError>;
/// Result type for K-matrix amplitude construction.
pub type KMatrixResult<T> = AmplitudeResult<T>;
/// Error type for K-matrix amplitude construction.
pub type KMatrixError = AmplitudeError;

/// Errors produced while constructing symbolic amplitudes.
#[derive(Clone, Debug, Error)]
pub enum AmplitudeError {
    /// An expression had an incompatible scalar, vector, or matrix shape.
    #[error(transparent)]
    ExpressionShape(#[from] ExprShapeError),
    /// A physics helper rejected an input.
    #[error(transparent)]
    Physics(#[from] LadduPhysicsError),
    /// Coupled-channel input dimensions were incompatible.
    #[error("invalid K-matrix input shape: {0}")]
    InvalidShape(String),
}
