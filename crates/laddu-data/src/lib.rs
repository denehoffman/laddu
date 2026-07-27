//! Columnar event data, schemas, streaming sources, and sinks for laddu.

use std::sync::Arc;

pub use laddu_physics::vectors::RealVec4;

/// In-memory datasets, event batches, and event views.
pub mod data;
mod error;
/// Streaming event-source and sink abstractions and file-format adapters.
pub mod io;
/// Logical event schemas and physical-column inference.
pub mod schema;
pub use error::{LadduDataError, LadduDataResult};
/// Shared immutable column or field name.
pub type Name = Arc<str>;
