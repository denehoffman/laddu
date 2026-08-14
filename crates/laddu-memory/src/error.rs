use thiserror::Error;

use crate::budget::MemoryBudget;

/// Result type for memory planning operations.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Errors produced while discovering or reserving memory.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MemoryError {
    /// A budget string or percentage is invalid.
    #[error("invalid memory budget: {0}")]
    InvalidBudget(String),
    /// A percentage cannot be resolved because capacity telemetry is unavailable.
    #[error("cannot resolve {budget} for {resource}: {basis} memory is unavailable")]
    UnknownCapacity {
        /// Resource label.
        resource: String,
        /// Requested budget.
        budget: MemoryBudget,
        /// Missing capacity basis.
        basis: &'static str,
    },
    /// A reservation exceeds the effective pool limit.
    #[error(
        "memory budget exceeded for {resource}: requested {requested} bytes, \
         {remaining} bytes remain"
    )]
    BudgetExceeded {
        /// Resource label.
        resource: String,
        /// Requested reservation.
        requested: u64,
        /// Remaining reservable bytes.
        remaining: u64,
    },
}
