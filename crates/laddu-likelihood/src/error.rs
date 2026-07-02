use laddu_expr::parameters::ParamError;
use laddu_runtime::RuntimeError;
use thiserror::Error;

pub type LikelihoodResult<T> = Result<T, LikelihoodError>;

#[derive(Debug, Error)]
pub enum LikelihoodError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Params(#[from] ParamError),
    #[error("{dataset} intensity must be positive for a likelihood term, got {value}")]
    NonPositiveIntensity { dataset: &'static str, value: f64 },
    #[error("accepted MC integral must be positive for acceptance correction, got {0}")]
    NonPositiveAcceptedIntegral(f64),
    #[error("luminosity must be positive for a cross section, got {0}")]
    NonPositiveLuminosity(f64),
    #[error("duplicate likelihood term name: {0}")]
    DuplicateTermName(String),
    #[error("unknown likelihood term: {0}")]
    MissingTerm(String),
    #[error("likelihood term is not an intensity term: {0}")]
    NotIntensityTerm(String),
    #[error("likelihood term {term} references unknown parameter {parameter}")]
    MissingParameter { term: String, parameter: String },
    #[error("likelihood term {term} has invalid penalty weight {lambda}")]
    InvalidPenaltyWeight { term: String, lambda: f64 },
    #[error("parameter values were built for a different likelihood parameter layout")]
    ParameterLayoutMismatch,
    #[error("gradient has length {actual}, expected {expected}")]
    GradientLengthMismatch { expected: usize, actual: usize },
    #[error("likelihood term has not been resolved: {0}")]
    UnresolvedTerm(String),
}
