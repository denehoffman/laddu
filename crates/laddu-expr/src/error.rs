use thiserror::Error;

pub type ParamResult<T> = Result<T, ParamError>;
pub type ExprResult<T> = Result<T, ExprError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ParamError {
    #[error("parameter name cannot be empty")]
    EmptyName,
    #[error("duplicate parameter name: {0}")]
    DuplicateName(String),
    #[error("parameter conflict for {name}: {reason}")]
    ParameterConflict { name: String, reason: String },
    #[error("invalid parameter id #{id} for layout of size {len}")]
    InvalidParamId { id: usize, len: usize },
    #[error("invalid free parameter id #{id} for layout with {len} free parameters")]
    InvalidFreeParamId { id: usize, len: usize },
    #[error("expected {expected} free parameters, got {actual}")]
    FreeLengthMismatch { expected: usize, actual: usize },
    #[error("invalid bounds for {name}: min {min} is greater than max {max}")]
    InvalidBounds { name: String, min: f64, max: f64 },
    #[error("invalid uniform initial range for {name}: min {min} is greater than max {max}")]
    InvalidInitialRange { name: String, min: f64, max: f64 },
    #[error("initial value {value} for {name} is outside bounds")]
    InitialOutOfBounds { name: String, value: f64 },
    #[error("initial range [{min}, {max}] for {name} is outside parameter bounds")]
    InitialRangeOutOfBounds { name: String, min: f64, max: f64 },
    #[error("fixed value {value} for {name} is outside bounds")]
    FixedValueOutOfBounds { name: String, value: f64 },
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ExprGraphError {
    #[error("graph metadata length {metadata_len} does not match node length {node_len}")]
    MetadataLength {
        node_len: usize,
        metadata_len: usize,
    },
    #[error("graph root node #{root} is out of bounds for graph with {node_len} nodes")]
    InvalidRoot { root: usize, node_len: usize },
    #[error("graph node #{node} references missing child #{child}")]
    InvalidChild { node: usize, child: usize },
    #[error(
        "graph node #{node} references child #{child}, but children must appear before parents"
    )]
    InvalidChildOrder { node: usize, child: usize },
    #[error("graph is empty")]
    Empty,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("invalid expression shape for {operation}: {message}")]
pub struct ExprShapeError {
    operation: &'static str,
    message: String,
}

impl ExprShapeError {
    pub(crate) fn new(operation: &'static str, message: impl Into<String>) -> Self {
        Self {
            operation,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ExprError {
    #[error(transparent)]
    Params(#[from] ParamError),
    #[error(transparent)]
    Graph(#[from] ExprGraphError),
    #[error(transparent)]
    Shape(#[from] ExprShapeError),
}
