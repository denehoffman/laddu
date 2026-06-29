use thiserror::Error;

pub mod channel;
pub mod histogram;
pub mod math;
pub mod quantum;
pub mod vectors;

pub type LadduPhysicsResult<T> = Result<T, LadduPhysicsError>;

#[derive(Error, Debug, Clone)]
pub enum LadduPhysicsError {
    /// An error that should be used to convert [`TryFrom`] to [`LadduPhysicsError`].
    #[error("Failed to convert value to \"{0}\"")]
    ConversionError(&'static str),
    /// An error which occurs when the user tries to parse an invalid string of text, typically
    /// into an enum variant.
    #[error("Failed to parse string: \"{name}\" does not correspond to a valid \"{object}\"!")]
    ParseError {
        /// The string which was parsed
        name: String,
        /// The name of the object it failed to parse into
        object: String,
    },
    /// A particle is missing the requested property
    #[error("Particle is missing the requested property \"{property}\"")]
    MissingParticleProperty {
        /// The name of the missing property
        property: &'static str,
    },
    /// A single value violates a domain constraint.
    #[error("Invalid value for {name}: expected {expected}, got {actual}")]
    InvalidValue {
        name: String,
        expected: String,
        actual: String,
    },
    /// A collection length or shape is invalid.
    #[error("Invalid length for {name}: expected {expected}, got {actual}")]
    InvalidLength {
        name: String,
        expected: String,
        actual: String,
    },
    /// A relation between multiple values, quantum numbers, or particle properties is invalid.
    #[error("Invalid relation: {relation}")]
    InvalidRelation { relation: String },

    /// A value is valid in principle but not implemented/supported here.
    #[error("Unsupported value for {name}: supported {supported}, got {actual}")]
    UnsupportedValue {
        name: String,
        supported: String,
        actual: String,
    },

    /// An integer operation overflowed.
    #[error("Numeric overflow while computing {operation}")]
    NumericOverflow { operation: String },
    #[error("{0}")]
    Custom(String),
}

impl LadduPhysicsError {
    pub fn invalid_value(
        name: impl Into<String>,
        expected: impl Into<String>,
        actual: impl ToString,
    ) -> Self {
        Self::InvalidValue {
            name: name.into(),
            expected: expected.into(),
            actual: actual.to_string(),
        }
    }

    pub fn invalid_length(
        name: impl Into<String>,
        expected: impl Into<String>,
        actual: impl ToString,
    ) -> Self {
        Self::InvalidLength {
            name: name.into(),
            expected: expected.into(),
            actual: actual.to_string(),
        }
    }

    pub fn invalid_relation(relation: impl Into<String>) -> Self {
        Self::InvalidRelation {
            relation: relation.into(),
        }
    }

    pub fn unsupported_value(
        name: impl Into<String>,
        supported: impl Into<String>,
        actual: impl ToString,
    ) -> Self {
        Self::UnsupportedValue {
            name: name.into(),
            supported: supported.into(),
            actual: actual.to_string(),
        }
    }

    pub fn numeric_overflow(operation: impl Into<String>) -> Self {
        Self::NumericOverflow {
            operation: operation.into(),
        }
    }

    fn custom(text: impl Into<String>) -> LadduPhysicsError {
        Self::Custom(text.into())
    }
}
