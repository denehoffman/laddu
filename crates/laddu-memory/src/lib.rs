//! Memory discovery, budgeting, reservation, and reporting for laddu.

mod budget;
mod decision;
mod discovery;
mod error;
mod pool;
mod report;
mod resource;
mod state;

pub use budget::{MemoryBudget, MemoryPlan};
pub use decision::{MemoryDecision, MemoryFitRequest, MemoryFootprint};
pub use error::{FootprintOverflow, MemoryError, MemoryResult};
pub use pool::{MemoryLease, MemoryPool};
pub use report::{MemoryPoolReport, MemoryReport, MemoryResourceReport, ProcessMemoryReport};
pub use resource::{CapacitySource, DeviceIdentity, MemoryResource, MemoryResourceKind};
pub use state::MemoryState;
