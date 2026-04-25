//! Helpers for constructing amplitude semantic identity keys.

use crate::parameters::Parameter;

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
