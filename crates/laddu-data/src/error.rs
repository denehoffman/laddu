use crate::Name;
use thiserror::Error;

/// Result type for event-data operations.
pub type LadduDataResult<T> = Result<T, LadduDataError>;

/// Errors produced by schemas, event sources, sinks, and datasets.
#[derive(Clone, Error, Debug)]
pub enum LadduDataError {
    /// The requested operation is not supported.
    #[error("Unsupported: {0}")]
    Unsupported(&'static str),
    /// An argument failed validation.
    #[error("Invalid Argument: {0}")]
    InvalidArgument(&'static str),
    /// A required physical or logical column was absent.
    #[error("Missing Column: {0}")]
    MissingColumn(Name),
    /// A schema was invalid or incompatible.
    #[error("Schema Error: {0}")]
    Schema(String),
    /// An event source failed.
    #[error("Source Error: {0}")]
    Source(String),
    /// An event sink failed.
    #[error("Sink Error: {0}")]
    Sink(String),
}
