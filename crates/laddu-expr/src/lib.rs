//! Symbolic expression graphs and parameter definitions used throughout laddu.
//!
//! This crate provides shareable expression DAGs, parameter layouts and values,
//! and optional graph visualization support.

mod error;
mod expression;
/// Parameter definitions, layouts, registries, and concrete values.
pub mod parameters;
mod visualization;

pub use error::{ExprError, ExprGraphError, ExprResult, ExprShapeError, ParamError, ParamResult};
pub use expression::*;
pub use visualization::*;
