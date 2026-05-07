//! # laddu-extensions
//!
//! This is an internal crate used by `laddu`.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![warn(clippy::perf, clippy::style, missing_docs)]

pub mod experimental;

/// A module containing optimization interfaces and algorithm integrations.
pub mod optimize;
pub use optimize::LikelihoodTermObserver;

/// Extended maximum likelihood cost functions with support for additive terms.
pub mod likelihood;

/// Randomized helpers used by stochastic likelihood terms.
pub mod random;

pub use likelihood::{LikelihoodExpression, LikelihoodScalar, NLL};
pub use random::RngSubsetExtension;
