//! Scalar-valued amplitude components.

mod components;
#[cfg(test)]
mod tests;
mod variable;

pub use components::{ComplexScalar, PolarComplexScalar, Scalar};
pub use variable::{VariableExpressionExt, VariableScalar};
