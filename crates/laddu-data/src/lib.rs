use std::sync::Arc;

pub use laddu_physics::vectors::RealVec4;

pub mod data;
mod error;
pub mod io;
pub mod schema;
pub use error::{LadduDataError, LadduDataResult};
pub type Name = Arc<str>;
