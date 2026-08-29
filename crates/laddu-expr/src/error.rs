use thiserror::Error;

/// Result type for parameter construction and lookup operations.
pub type ParamResult<T> = Result<T, ParamError>;
/// Result type for expression construction and validation operations.
pub type ExprResult<T> = Result<T, ExprError>;

/// Errors produced while defining, laying out, or assigning parameters.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ParamError {
    /// A parameter name was empty.
    #[error("parameter name cannot be empty")]
    EmptyName,
    #[error("duplicate parameter name: {0}")]
    /// A parameter name was registered more than once.
    DuplicateName(String),
    #[error("unknown parameter: {0}")]
    /// A parameter name was not present in the layout.
    UnknownName(String),
    #[error("parameter conflict for {name}: {reason}")]
    /// Two definitions of the same parameter were incompatible.
    ParameterConflict {
        /// Conflicting parameter name.
        name: String,
        /// Description of the incompatibility.
        reason: String,
    },
    /// A parameter identifier was outside the layout.
    #[error("invalid parameter id #{id} for layout of size {len}")]
    InvalidParamId {
        /// Invalid identifier index.
        id: usize,
        /// Number of parameters in the layout.
        len: usize,
    },
    /// A free-parameter identifier was outside the layout.
    #[error("invalid free parameter id #{id} for layout with {len} free parameters")]
    InvalidFreeParamId {
        /// Invalid free-parameter index.
        id: usize,
        /// Number of free parameters in the layout.
        len: usize,
    },
    /// A free-parameter value vector had the wrong length.
    #[error("expected {expected} free parameters, got {actual}")]
    FreeLengthMismatch {
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// A parameter's lower bound exceeded its upper bound.
    #[error("invalid bounds for {name}: min {min} is greater than max {max}")]
    InvalidBounds {
        /// Parameter name.
        name: String,
        /// Invalid lower bound.
        min: f64,
        /// Invalid upper bound.
        max: f64,
    },
    /// A parameter bound was NaN.
    #[error("invalid NaN bound {value} for {name}")]
    InvalidBoundValue {
        /// Parameter name.
        name: String,
        /// Invalid bound value.
        value: f64,
    },
    /// A uniform initial range had its endpoints reversed.
    #[error("invalid uniform initial range for {name}: min {min} is greater than max {max}")]
    InvalidInitialRange {
        /// Parameter name.
        name: String,
        /// Invalid range minimum.
        min: f64,
        /// Invalid range maximum.
        max: f64,
    },
    /// A parameter's initial value fell outside its bounds.
    #[error("initial value {value} for {name} is outside bounds")]
    InitialOutOfBounds {
        /// Parameter name.
        name: String,
        /// Invalid initial value.
        value: f64,
    },
    /// A parameter's initial range extended outside its bounds.
    #[error("initial range [{min}, {max}] for {name} is outside parameter bounds")]
    InitialRangeOutOfBounds {
        /// Parameter name.
        name: String,
        /// Initial range minimum.
        min: f64,
        /// Initial range maximum.
        max: f64,
    },
    /// A scalar initial value was not finite.
    #[error("initial value {value} for {name} must be finite")]
    NonFiniteInitialValue {
        /// Parameter name.
        name: String,
        /// Invalid initial value.
        value: f64,
    },
    /// An initial range endpoint was not finite.
    #[error("initial range [{min}, {max}] for {name} must be finite")]
    NonFiniteInitialRange {
        /// Parameter name.
        name: String,
        /// Range minimum.
        min: f64,
        /// Range maximum.
        max: f64,
    },
    /// A fixed parameter value fell outside its bounds.
    #[error("fixed value {value} for {name} is outside bounds")]
    FixedValueOutOfBounds {
        /// Parameter name.
        name: String,
        /// Invalid fixed value.
        value: f64,
    },
    /// A fixed parameter value was not finite.
    #[error("fixed value {value} for {name} must be finite")]
    NonFiniteFixedValue {
        /// Parameter name.
        name: String,
        /// Invalid fixed value.
        value: f64,
    },
    /// An assigned parameter value fell outside its bounds.
    #[error("value {value} for {name} is outside bounds")]
    ValueOutOfBounds {
        /// Parameter name.
        name: String,
        /// Invalid assigned value.
        value: f64,
    },
    /// A periodic parameter did not have finite, ordered bounds.
    #[error("periodic parameter {name} requires finite two-sided bounds with min < max")]
    PeriodicRequiresFiniteBounds {
        /// Parameter name.
        name: String,
    },
    /// A parameter scale was not finite and positive.
    #[error("invalid scale for {name}: expected a finite positive value, got {scale}")]
    InvalidScale {
        /// Parameter name.
        name: String,
        /// Invalid scale.
        scale: f64,
    },
    /// A fixed value in a parameter update was not finite.
    #[error("parameter update fixed value must be finite, got {value}")]
    InvalidUpdateFixedValue {
        /// Invalid fixed value.
        value: f64,
    },
    /// A scalar initial value in a parameter update was not finite.
    #[error("parameter update initial value must be finite, got {value}")]
    InvalidUpdateInitialValue {
        /// Invalid initial value.
        value: f64,
    },
    /// A uniform initial range in a parameter update was not finite or ordered.
    #[error("parameter update initial range must be finite and ordered, got [{min}, {max}]")]
    InvalidUpdateInitialRange {
        /// Invalid range minimum.
        min: f64,
        /// Invalid range maximum.
        max: f64,
    },
    /// Bounds in a parameter update were unordered or contained NaN.
    #[error(
        "parameter update bounds must be ordered and cannot contain NaN, got [{min:?}, {max:?}]"
    )]
    InvalidUpdateBounds {
        /// Invalid lower bound.
        min: Option<f64>,
        /// Invalid upper bound.
        max: Option<f64>,
    },
    /// A scale in a parameter update was not finite and positive.
    #[error("parameter update scale must be finite and positive, got {scale}")]
    InvalidUpdateScale {
        /// Invalid scale.
        scale: f64,
    },
    /// A periodic value was outside its canonical half-open domain.
    #[error(
        "value {value} for periodic parameter {name} is outside canonical domain [{min}, {max})"
    )]
    ValueOutsidePeriodicDomain {
        /// Parameter name.
        name: String,
        /// Invalid value.
        value: f64,
        /// Inclusive domain minimum.
        min: f64,
        /// Exclusive domain maximum.
        max: f64,
    },
}

/// Structural validation errors for serialized expression graphs.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ExprGraphError {
    /// The node and metadata arrays had different lengths.
    #[error("graph metadata length {metadata_len} does not match node length {node_len}")]
    MetadataLength {
        /// Number of graph nodes.
        node_len: usize,
        /// Number of metadata entries.
        metadata_len: usize,
    },
    /// The root identifier was outside the node array.
    #[error("graph root node #{root} is out of bounds for graph with {node_len} nodes")]
    InvalidRoot {
        /// Invalid root index.
        root: usize,
        /// Number of graph nodes.
        node_len: usize,
    },
    /// A node referenced a child outside the node array.
    #[error("graph node #{node} references missing child #{child}")]
    InvalidChild {
        /// Parent node index.
        node: usize,
        /// Invalid child index.
        child: usize,
    },
    /// A node referenced a child stored after its parent.
    #[error(
        "graph node #{node} references child #{child}, but children must appear before parents"
    )]
    InvalidChildOrder {
        /// Parent node index.
        node: usize,
        /// Out-of-order child index.
        child: usize,
    },
    /// The graph contained no nodes.
    #[error("graph is empty")]
    Empty,
}

/// Describes an operation applied to an expression with an incompatible shape.
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

/// Errors produced while constructing or rebuilding expressions.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ExprError {
    /// A parameter operation failed.
    #[error(transparent)]
    Params(#[from] ParamError),
    /// A serialized graph was structurally invalid.
    #[error(transparent)]
    Graph(#[from] ExprGraphError),
    /// An operation received incompatible expression shapes.
    #[error(transparent)]
    Shape(#[from] ExprShapeError),
}
