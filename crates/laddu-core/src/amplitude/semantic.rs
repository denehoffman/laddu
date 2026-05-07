use serde::{Deserialize, Serialize};

use crate::parameters::Parameter;

/// A single named field in an [`AmplitudeSemanticKey`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmplitudeSemanticField {
    name: String,
    value: String,
}

impl AmplitudeSemanticField {
    /// Construct a semantic key field.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A semantic identity key used to opt into deduplicating equivalent amplitude computations.
///
/// The key must include enough type/configuration information to prove that two independently
/// constructed amplitudes can safely share one registered computation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmplitudeSemanticKey {
    kind: String,
    fields: Vec<AmplitudeSemanticField>,
}

impl AmplitudeSemanticKey {
    /// Construct a semantic key for the given amplitude kind.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            fields: Vec::new(),
        }
    }

    /// Add a named field to this semantic key.
    pub fn with_field(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.push(AmplitudeSemanticField::new(name, value));
        self
    }
}

/// Encode an `f64` as a stable bit-pattern string for semantic keys.
pub fn f64_key(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

/// Convert a displayable value into a semantic-key field.
pub fn display_key(value: impl std::fmt::Display) -> String {
    value.to_string()
}

/// Convert a debuggable value into a semantic-key field.
pub fn debug_key(value: impl std::fmt::Debug) -> String {
    format!("{value:?}")
}

/// Convert a parameter into a semantic-key field.
pub fn parameter_key(parameter: &Parameter) -> String {
    match parameter.fixed() {
        Some(value) => format!("{:?}:fixed:{}", parameter.name(), f64_key(value)),
        None => format!("{:?}:free", parameter.name()),
    }
}

/// Convert a parameter slice into a semantic-key field.
pub fn parameter_slice_key(parameters: &[Parameter]) -> String {
    format!(
        "[{}]",
        parameters
            .iter()
            .map(parameter_key)
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Convert a slice of parameter pairs into a semantic-key field.
pub fn parameter_pair_slice_key(parameters: &[(Parameter, Parameter)]) -> String {
    format!(
        "[{}]",
        parameters
            .iter()
            .map(|(first, second)| format!("({}, {})", parameter_key(first), parameter_key(second)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Convert a parameter array into a semantic-key field.
pub fn parameter_array_key<const N: usize>(parameters: &[Parameter; N]) -> String {
    parameter_slice_key(parameters)
}

/// Convert an optional seed into a semantic-key field.
pub fn seed_key(seed: Option<usize>) -> String {
    match seed {
        Some(seed) => format!("Some({seed})"),
        None => "None".to_string(),
    }
}
