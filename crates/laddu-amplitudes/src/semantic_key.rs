//! Helpers for constructing amplitude semantic identity keys.

use laddu_core::amplitudes::Parameter;

pub(crate) fn f64_key(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

pub(crate) fn display_key(value: impl std::fmt::Display) -> String {
    value.to_string()
}

pub(crate) fn debug_key(value: impl std::fmt::Debug) -> String {
    format!("{value:?}")
}

pub(crate) fn parameter_key(parameter: &Parameter) -> String {
    match parameter.fixed {
        Some(value) => format!("{:?}:fixed:{}", parameter.name, f64_key(value)),
        None => format!("{:?}:free", parameter.name),
    }
}

pub(crate) fn parameter_slice_key(parameters: &[Parameter]) -> String {
    format!(
        "[{}]",
        parameters
            .iter()
            .map(parameter_key)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn parameter_pair_slice_key(parameters: &[(Parameter, Parameter)]) -> String {
    format!(
        "[{}]",
        parameters
            .iter()
            .map(|(first, second)| format!("({}, {})", parameter_key(first), parameter_key(second)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn parameter_array_key<const N: usize>(parameters: &[Parameter; N]) -> String {
    parameter_slice_key(parameters)
}

pub(crate) fn seed_key(seed: Option<usize>) -> String {
    match seed {
        Some(seed) => format!("Some({seed})"),
        None => "None".to_string(),
    }
}
