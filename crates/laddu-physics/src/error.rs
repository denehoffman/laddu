use thiserror::Error;

/// Result type used throughout the physics crate.
pub type LadduPhysicsResult<T> = Result<T, LadduPhysicsError>;

#[derive(Error, Debug, Clone)]
/// Errors raised while constructing or evaluating physics objects.
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
        /// Name of the invalid input.
        name: String,
        /// Description of the accepted domain.
        expected: String,
        /// Supplied value.
        actual: String,
    },
    /// A collection length or shape is invalid.
    #[error("Invalid length for {name}: expected {expected}, got {actual}")]
    InvalidLength {
        /// Name of the invalid collection.
        name: String,
        /// Description of the required length or shape.
        expected: String,
        /// Supplied length or shape.
        actual: String,
    },
    /// A relation between multiple values, quantum numbers, or particle properties is invalid.
    #[error("Invalid relation: {relation}")]
    InvalidRelation {
        /// Description of the violated relation.
        relation: String,
    },

    /// A value is valid in principle but not implemented/supported here.
    #[error("Unsupported value for {name}: supported {supported}, got {actual}")]
    UnsupportedValue {
        /// Name of the unsupported input.
        name: String,
        /// Description of the supported alternatives.
        supported: String,
        /// Supplied value.
        actual: String,
    },

    /// An integer operation overflowed.
    #[error("Numeric overflow while computing {operation}")]
    NumericOverflow {
        /// Operation that overflowed.
        operation: String,
    },
    /// A free-form error message for internal compatibility paths.
    #[error("{0}")]
    Custom(String),
}

impl LadduPhysicsError {
    /// Construct an invalid-value error.
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

    /// Construct an invalid-length or shape error.
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

    /// Construct an invalid-relation error.
    pub fn invalid_relation(relation: impl Into<String>) -> Self {
        Self::InvalidRelation {
            relation: relation.into(),
        }
    }

    /// Construct an unsupported-value error.
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

    /// Construct a numeric-overflow error.
    pub fn numeric_overflow(operation: impl Into<String>) -> Self {
        Self::NumericOverflow {
            operation: operation.into(),
        }
    }

    pub(crate) fn custom(text: impl Into<String>) -> LadduPhysicsError {
        Self::Custom(text.into())
    }
}
