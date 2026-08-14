use serde::{Deserialize, Serialize};

use crate::{budget::MemoryBudget, resource::MemoryResource};

/// Report for one resolved pool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryPoolReport {
    /// Stable resource identifier.
    pub resource_id: String,
    /// Requested budget.
    pub requested: MemoryBudget,
    /// Resolved capacity.
    pub effective_bytes: u64,
    /// Currently reserved bytes.
    pub reserved_bytes: u64,
    /// Remaining bytes.
    pub remaining_bytes: u64,
    /// Highest concurrent reservation.
    pub high_water_bytes: u64,
}

/// Report for one physical resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryResourceReport {
    /// Resource snapshot.
    pub resource: MemoryResource,
    /// Currently reserved by laddu.
    pub laddu_reserved_bytes: u64,
    /// Process-state high-water reservation.
    pub laddu_high_water_bytes: u64,
}

/// Report covering all resources in a [`crate::MemoryState`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReport {
    /// Current process memory sampled when the report was generated.
    pub process: Option<ProcessMemoryReport>,
    /// Physical resource reports.
    pub resources: Vec<MemoryResourceReport>,
}

/// Sampled operating-system memory counters for the current process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMemoryReport {
    /// Current resident-set size.
    pub resident_bytes: u64,
    /// Current virtual-memory size.
    pub virtual_bytes: u64,
    /// Largest resident-set size sampled by this state, not an OS lifetime peak.
    pub sampled_high_water_bytes: u64,
}
