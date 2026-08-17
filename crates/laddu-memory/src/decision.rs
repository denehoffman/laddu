use serde::{Deserialize, Serialize};

use crate::error::{FootprintOverflow, MemoryError, MemoryResult};

/// One memory-derived execution decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryDecision {
    /// Operation or dataset label.
    pub label: String,
    /// Fixed bytes required regardless of event count.
    pub fixed_bytes: u64,
    /// Estimated incremental bytes per event.
    pub bytes_per_event: u64,
    /// Chosen internal event count.
    pub chunk_events: usize,
    /// Estimated peak tracked bytes.
    pub estimated_peak_bytes: u64,
    /// Actual tracked high-water bytes when known.
    pub actual_high_water_bytes: Option<u64>,
    /// Selected storage/execution strategy.
    pub strategy: String,
}

/// Workspace-internal memory footprint used to construct planning decisions.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryFootprint {
    /// Fixed bytes required regardless of event count.
    pub fixed_bytes: u64,
    /// Estimated incremental bytes per event.
    pub bytes_per_event: u64,
}

impl MemoryFootprint {
    /// Creates a footprint from byte counts.
    pub const fn new(fixed_bytes: u64, bytes_per_event: u64) -> Self {
        Self {
            fixed_bytes,
            bytes_per_event,
        }
    }
    /// Creates a footprint containing only a fixed allocation.
    pub const fn fixed(fixed_bytes: u64) -> Self {
        Self::new(fixed_bytes, 0)
    }
    /// Creates a footprint containing only an event-dependent allocation.
    pub const fn per_event(bytes_per_event: u64) -> Self {
        Self::new(0, bytes_per_event)
    }
    /// Creates a footprint from platform-sized byte counts using saturation.
    pub fn from_usize(fixed_bytes: usize, bytes_per_event: usize) -> Self {
        Self::from_usize_checked(fixed_bytes, bytes_per_event)
            .unwrap_or(Self::new(u64::MAX, u64::MAX))
    }
    /// Creates a footprint from platform-sized byte counts with overflow
    /// detection.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow::Conversion`] when a platform-sized value
    /// cannot be represented by `u64`.
    pub fn from_usize_checked(
        fixed_bytes: usize,
        bytes_per_event: usize,
    ) -> Result<Self, FootprintOverflow> {
        Ok(Self::new(
            checked_u64(fixed_bytes)?,
            checked_u64(bytes_per_event)?,
        ))
    }
    /// Adds two fixed/per-event footprint components with overflow detection.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow::Addition`] when either component exceeds
    /// `u64`.
    pub const fn checked_add(self, other: Self) -> Result<Self, FootprintOverflow> {
        let fixed_bytes = match self.fixed_bytes.checked_add(other.fixed_bytes) {
            Some(value) => value,
            None => return Err(FootprintOverflow::Addition),
        };
        let bytes_per_event = match self.bytes_per_event.checked_add(other.bytes_per_event) {
            Some(value) => value,
            None => return Err(FootprintOverflow::Addition),
        };
        Ok(Self::new(fixed_bytes, bytes_per_event))
    }
    /// Scales fixed and per-event components by `factor` with overflow
    /// detection.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow::Multiplication`] when scaling exceeds
    /// `u64`.
    pub const fn checked_scale(self, factor: u64) -> Result<Self, FootprintOverflow> {
        let fixed_bytes = match self.fixed_bytes.checked_mul(factor) {
            Some(value) => value,
            None => return Err(FootprintOverflow::Multiplication),
        };
        let bytes_per_event = match self.bytes_per_event.checked_mul(factor) {
            Some(value) => value,
            None => return Err(FootprintOverflow::Multiplication),
        };
        Ok(Self::new(fixed_bytes, bytes_per_event))
    }
    /// Scales fixed and per-event components by a platform-sized factor with
    /// overflow detection.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when conversion or scaling overflows.
    pub fn checked_scale_usize(self, factor: usize) -> Result<Self, FootprintOverflow> {
        self.checked_scale(checked_u64(factor)?)
    }
    /// Calculates the peak bytes for `events` with overflow detection.
    ///
    /// # Errors
    ///
    /// Returns [`FootprintOverflow`] when conversion, multiplication, or
    /// addition overflows.
    pub fn checked_peak_bytes(self, events: usize) -> Result<u64, FootprintOverflow> {
        let events = checked_u64(events)?;
        let event_bytes = self
            .bytes_per_event
            .checked_mul(events)
            .ok_or(FootprintOverflow::Multiplication)?;
        self.fixed_bytes
            .checked_add(event_bytes)
            .ok_or(FootprintOverflow::Addition)
    }
    /// Estimates peak bytes for `events` using the shared saturation policy.
    pub fn peak_bytes(self, events: usize) -> u64 {
        self.checked_peak_bytes(events).unwrap_or(u64::MAX)
    }
    fn normalized(mut self) -> Self {
        self.bytes_per_event = self.bytes_per_event.max(1);
        self
    }
}

/// Workspace-internal named input for a memory-derived decision.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryFitRequest {
    /// Operation or dataset label.
    pub label: String,
    /// Fixed and per-event memory estimate.
    pub footprint: MemoryFootprint,
    /// Bytes available to this operation.
    pub available_bytes: u64,
    /// Maximum number of events the operation may process.
    pub event_limit: usize,
    /// Selected storage or execution strategy.
    pub strategy: String,
}

impl MemoryDecision {
    /// Derives the largest event chunk fitting within `available_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::BudgetExceeded`] when fixed cost plus one event
    /// cannot fit.
    pub fn fit(
        label: impl Into<String>,
        fixed_bytes: u64,
        bytes_per_event: u64,
        available_bytes: u64,
        event_limit: usize,
        strategy: impl Into<String>,
    ) -> MemoryResult<Self> {
        MemoryFitRequest {
            label: label.into(),
            footprint: MemoryFootprint::new(fixed_bytes, bytes_per_event),
            available_bytes,
            event_limit,
            strategy: strategy.into(),
        }
        .evaluate()
    }
}

impl MemoryFitRequest {
    /// Evaluates the largest event chunk fitting the request.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::BudgetExceeded`] when fixed cost plus one event
    /// cannot fit.
    pub fn evaluate(mut self) -> MemoryResult<MemoryDecision> {
        self.footprint = self.footprint.normalized();
        if self.event_limit == 0 {
            return Ok(self.decision(0));
        }
        let event_capacity = self
            .available_bytes
            .saturating_sub(self.footprint.fixed_bytes);
        let events =
            saturating_usize(event_capacity / self.footprint.bytes_per_event).min(self.event_limit);
        if events == 0 {
            return Err(MemoryError::BudgetExceeded {
                resource: self.label,
                requested: self
                    .footprint
                    .fixed_bytes
                    .saturating_add(self.footprint.bytes_per_event),
                remaining: self.available_bytes,
            });
        }
        Ok(self.decision(events))
    }

    /// Builds a resident decision covering the full event limit while reporting the supplied chunk.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::BudgetExceeded`] when the resident footprint
    /// exceeds the available bytes.
    pub fn evaluate_resident(self, chunk_events: usize) -> MemoryResult<MemoryDecision> {
        let peak = self.footprint.peak_bytes(self.event_limit);
        if peak > self.available_bytes {
            return Err(MemoryError::BudgetExceeded {
                resource: self.label,
                requested: peak,
                remaining: self.available_bytes,
            });
        }
        Ok(self.decision_with_peak(chunk_events, peak))
    }

    fn decision(self, events: usize) -> MemoryDecision {
        let peak = self.footprint.peak_bytes(events);
        self.decision_with_peak(events, peak)
    }

    fn decision_with_peak(self, chunk_events: usize, peak: u64) -> MemoryDecision {
        MemoryDecision {
            label: self.label,
            fixed_bytes: self.footprint.fixed_bytes,
            bytes_per_event: self.footprint.bytes_per_event,
            chunk_events,
            estimated_peak_bytes: peak,
            actual_high_water_bytes: None,
            strategy: self.strategy,
        }
    }
}

fn saturating_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}
fn checked_u64(value: usize) -> Result<u64, FootprintOverflow> {
    u64::try_from(value).map_err(|_| FootprintOverflow::Conversion)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_the_largest_safe_chunk() {
        let decision = MemoryDecision::fit("test", 100, 8, 1_000, 1_000, "streaming").unwrap();
        assert_eq!(decision.chunk_events, 112);
        assert_eq!(decision.estimated_peak_bytes, 996);
    }

    #[test]
    fn defines_zero_and_capacity_boundary_policy() {
        let zero = MemoryDecision::fit("zero", 101, 0, 100, 0, "empty").unwrap();
        assert_eq!(
            (
                zero.bytes_per_event,
                zero.chunk_events,
                zero.estimated_peak_bytes
            ),
            (1, 0, 101)
        );
        let normalized = MemoryDecision::fit("normalized", 0, 0, 100, 200, "streaming").unwrap();
        assert_eq!(
            (
                normalized.bytes_per_event,
                normalized.chunk_events,
                normalized.estimated_peak_bytes
            ),
            (1, 100, 100)
        );
        assert!(MemoryDecision::fit("full", 100, 1, 100, 1, "streaming").is_err());
        assert!(MemoryDecision::fit("full", 101, 1, 100, 1, "streaming").is_err());
    }

    #[test]
    fn saturates_at_integer_boundaries() {
        let decision =
            MemoryDecision::fit("maximum", 0, 1, u64::MAX, usize::MAX, "resident").unwrap();
        assert_eq!(
            decision.chunk_events,
            usize::try_from(u64::MAX).unwrap_or(usize::MAX)
        );
        assert_eq!(decision.estimated_peak_bytes, u64::MAX);
        assert_eq!(
            MemoryFootprint::new(u64::MAX, u64::MAX).peak_bytes(usize::MAX),
            u64::MAX
        );
        assert_eq!(
            MemoryDecision::fit("overflow", u64::MAX, 1, u64::MAX, 1, "streaming"),
            Err(MemoryError::BudgetExceeded {
                resource: "overflow".into(),
                requested: u64::MAX,
                remaining: u64::MAX,
            })
        );
    }

    #[test]
    fn named_and_resident_requests_share_policy() {
        let request = MemoryFitRequest {
            label: "named".into(),
            footprint: MemoryFootprint::from_usize(100, 8),
            available_bytes: 1_000,
            event_limit: 100,
            strategy: "resident".into(),
        };
        assert_eq!(request.clone().evaluate().unwrap().chunk_events, 100);
        let decision = request.evaluate_resident(25).unwrap();
        assert_eq!(
            (decision.chunk_events, decision.estimated_peak_bytes),
            (25, 900)
        );
    }

    #[test]
    fn resident_requests_check_the_full_footprint() {
        let request = MemoryFitRequest {
            label: "resident".into(),
            footprint: MemoryFootprint::new(100, 8),
            available_bytes: 899,
            event_limit: 100,
            strategy: "resident".into(),
        };
        assert_eq!(
            request.evaluate_resident(25),
            Err(MemoryError::BudgetExceeded {
                resource: "resident".into(),
                requested: 900,
                remaining: 899,
            })
        );

        let zero_per_event = MemoryFitRequest {
            label: "resident".into(),
            footprint: MemoryFootprint::new(100, 0),
            available_bytes: 100,
            event_limit: 100,
            strategy: "resident".into(),
        }
        .evaluate_resident(25)
        .unwrap();
        assert_eq!(
            (
                zero_per_event.bytes_per_event,
                zero_per_event.estimated_peak_bytes
            ),
            (0, 100)
        );
    }
}
