//! Automatic differentiation plans and gradient kernel generation.

mod error;
mod gradient;
mod plan;

pub use error::{AutodiffError, AutodiffResult};
pub use gradient::gradient_ir;
pub use plan::{AutodiffMode, AutodiffPlan};
