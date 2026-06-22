use std::sync::Arc;

use thiserror::Error;

pub mod data;
pub mod io;
pub mod schema;
pub type LadduDataResult<T> = Result<T, LadduDataError>;
pub type Name = Arc<str>;

#[derive(Error, Debug)]
pub enum LadduDataError {
    #[error("Unsupported: {0}")]
    Unsupported(&'static str),
    #[error("Invalid Argument: {0}")]
    InvalidArgument(&'static str),
    #[error("Missing Column: {0}")]
    MissingColumn(Name),
    #[error("Schema Error: {0}")]
    Schema(String),
    #[error("Source Error: {0}")]
    Source(String),
    #[error("Sink Error: {0}")]
    Sink(String),
}
