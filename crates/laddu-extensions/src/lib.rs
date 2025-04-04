//! # laddu-extensions
//!
//! This is an internal crate used by `laddu`.
#![warn(clippy::perf, clippy::style, missing_docs)]

/// Experimental extensions to the `laddu` ecosystem
///
/// <div class="warning">
///
/// This module contains experimental code which may be untested or unreliable. Use at your own
/// risk! The features contained here may eventually be moved into the standard crate modules.
///
/// </div>
pub mod experimental;

/// A module containing the `laddu` interface with the [`ganesh`] library
pub mod ganesh_ext;

/// Extended maximum likelihood cost functions with support for additive terms
pub mod likelihoods;

pub use ganesh::{Ensemble, Status};
pub use ganesh_ext::{MCMCOptions, MinimizerOptions};
pub use likelihoods::{
    LikelihoodEvaluator, LikelihoodExpression, LikelihoodID, LikelihoodManager, LikelihoodScalar,
    NLL,
};
