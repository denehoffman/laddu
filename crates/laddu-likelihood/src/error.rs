use laddu_data::LadduDataError;
use laddu_expr::parameters::ParamError;
use laddu_runtime::RuntimeError;
use thiserror::Error;

/// Result type returned by likelihood operations.
pub type LikelihoodResult<T> = Result<T, LikelihoodError>;

/// Error produced while constructing or evaluating a likelihood.
#[derive(Debug, Error)]
pub enum LikelihoodError {
    /// Dataset access failed.
    #[error(transparent)]
    Data(#[from] LadduDataError),
    /// Model preparation or execution failed.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// Parameter registration, validation, or lookup failed.
    #[error(transparent)]
    Params(#[from] ParamError),
    /// A likelihood intensity was non-positive.
    #[error("{dataset} intensity must be positive for a likelihood term, got {value}")]
    NonPositiveIntensity {
        /// Dataset on which the invalid intensity occurred.
        dataset: &'static str,
        /// Invalid intensity value.
        value: f64,
    },
    /// The accepted Monte Carlo integral was non-positive.
    #[error("accepted MC integral must be positive for acceptance correction, got {0}")]
    NonPositiveAcceptedIntegral(f64),
    /// A cross-section calculation received non-positive luminosity.
    #[error("luminosity must be positive for a cross section, got {0}")]
    NonPositiveLuminosity(f64),
    /// Two likelihood terms used the same name.
    #[error("duplicate likelihood term name: {0}")]
    DuplicateTermName(String),
    /// No likelihood term had the requested name.
    #[error("unknown likelihood term: {0}")]
    MissingTerm(String),
    /// The requested term was not an intensity term.
    #[error("likelihood term is not an intensity term: {0}")]
    NotIntensityTerm(String),
    /// A term referenced a parameter absent from the global layout.
    #[error("likelihood term {term} references unknown parameter {parameter}")]
    MissingParameter {
        /// Likelihood term name.
        term: String,
        /// Missing parameter name.
        parameter: String,
    },
    /// A regularization term received an invalid weight.
    #[error("likelihood term {term} has invalid penalty weight {lambda}")]
    InvalidPenaltyWeight {
        /// Penalty term name.
        term: String,
        /// Invalid regularization weight.
        lambda: f64,
    },
    /// Parameter values belong to a different layout.
    #[error("parameter values were built for a different likelihood parameter layout")]
    ParameterLayoutMismatch,
    /// A destination gradient had the wrong length.
    #[error("gradient has length {actual}, expected {expected}")]
    GradientLengthMismatch {
        /// Expected gradient length.
        expected: usize,
        /// Actual gradient length.
        actual: usize,
    },
    /// A likelihood term was used before resolution.
    #[error("likelihood term has not been resolved: {0}")]
    UnresolvedTerm(String),
    /// A stochastic batch fraction was outside `(0, 1]`.
    #[error("stochastic batch fraction must be in (0, 1], got {0}")]
    InvalidBatchFraction(f64),
}
