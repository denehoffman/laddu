use thiserror::Error;

use crate::budget::MemoryBudget;

/// Result type for memory planning operations.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Overflow produced while composing a checked memory footprint.
///
/// This error is intentionally separate from [`MemoryError`]: footprint
/// construction is a low-level arithmetic concern, while budget exhaustion is
/// an operation-level planning result. Callers that retain the historical
/// saturating planning behavior can use the infallible footprint helpers.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum FootprintOverflow {
    /// Adding fixed or per-event components exceeded `u64`.
    #[error("memory footprint addition overflow")]
    Addition,
    /// Scaling fixed or per-event components exceeded `u64`.
    #[error("memory footprint multiplication overflow")]
    Multiplication,
    /// Converting a platform-sized byte count exceeded `u64`.
    #[error("memory footprint conversion overflow")]
    Conversion,
}

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
