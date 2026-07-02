use laddu_expr::ExprShapeError;
use laddu_physics::LadduPhysicsError;
use thiserror::Error;

pub type AmplitudeResult<T> = Result<T, AmplitudeError>;
pub type KMatrixResult<T> = AmplitudeResult<T>;
pub type KMatrixError = AmplitudeError;

#[derive(Clone, Debug, Error)]
pub enum AmplitudeError {
    #[error(transparent)]
    ExpressionShape(#[from] ExprShapeError),
    #[error(transparent)]
    Physics(#[from] LadduPhysicsError),
    #[error("invalid K-matrix input shape: {0}")]
    InvalidShape(String),
}
