//! Composable likelihood objectives, penalties, projections, and cross-section utilities.

mod error;
mod likelihood;

pub use error::{LikelihoodError, LikelihoodResult};
pub use likelihood::*;
