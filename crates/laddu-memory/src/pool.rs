use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
};

use crate::{
    budget::MemoryBudget,
    discovery::CapacitySnapshot,
    error::{MemoryError, MemoryResult},
    report::MemoryPoolReport,
    resource::{CapacitySource, MemoryResource},
    state::MemoryStateInner,
};

#[derive(Debug)]
pub(crate) struct ResourceLedger {
    pub(crate) snapshot: MemoryResource,
    pub(crate) reserved: u64,
    pub(crate) high_water: u64,
}

impl ResourceLedger {
    pub(crate) fn new(mut snapshot: MemoryResource) -> Self {
        snapshot.normalize_capacity();
        debug_assert_eq!(snapshot.validate(), Ok(()));
        Self {
            snapshot,
            reserved: 0,
            high_water: 0,
        }
    }

    pub(crate) fn update_snapshot(&mut self, mut snapshot: MemoryResource) {
        snapshot.normalize_capacity();
        debug_assert_eq!(snapshot.validate(), Ok(()));
        if self.snapshot.capacity_source != CapacitySource::User
            || snapshot.capacity_source == CapacitySource::User
        {
            self.snapshot = snapshot;
        }
    }

    pub(crate) fn apply_telemetry(
        &mut self,
        target: &MemoryResource,
        snapshot: CapacitySnapshot,
    ) -> bool {
        if !self.snapshot.is_refreshable()
            || self.snapshot.device_identity != target.device_identity
        {
            return false;
        }
        self.snapshot.apply_capacity_snapshot(snapshot);
        debug_assert_eq!(self.snapshot.validate(), Ok(()));
        true
    }

    fn try_reserve(&mut self, bytes: u64) -> MemoryResult<()> {
        let physical_limit = self.snapshot.effective_available();
        let Some(next) = self.reserved.checked_add(bytes) else {
            return Err(self.budget_exceeded(bytes, physical_limit));
        };
        if next > physical_limit {
            return Err(self.budget_exceeded(bytes, physical_limit));
        }
        self.reserved = next;
        self.high_water = self.high_water.max(next);
        Ok(())
    }

    fn release(&mut self, bytes: u64) {
        self.reserved = self.reserved.saturating_sub(bytes);
    }
    fn budget_exceeded(&self, requested: u64, physical_limit: u64) -> MemoryError {
        MemoryError::BudgetExceeded {
            resource: self.snapshot.name.clone(),
            requested,
            remaining: physical_limit.saturating_sub(self.reserved),
        }
    }
}

/// A resolved, reservable limit within one physical resource.
#[derive(Clone, Debug)]
pub struct MemoryPool {
    pub(crate) inner: Arc<MemoryPoolInner>,
}

#[derive(Debug)]
pub(crate) struct MemoryPoolInner {
    pub(crate) requested: MemoryBudget,
    pub(crate) accounting: Arc<ReservationAccount>,
}

#[derive(Debug)]
pub(crate) struct ReservationAccount {
    pub(crate) state: Weak<MemoryStateInner>,
    pub(crate) resource_id: String,
    pub(crate) capacity: u64,
    reserved: AtomicU64,
    high_water: AtomicU64,
}

impl ReservationAccount {
    pub(crate) fn new(state: Weak<MemoryStateInner>, resource_id: String, capacity: u64) -> Self {
        Self {
            state,
            resource_id,
            capacity,
            reserved: AtomicU64::new(0),
            high_water: AtomicU64::new(0),
        }
    }

    fn try_reserve(self: &Arc<Self>, bytes: u64) -> MemoryResult<ReservationToken> {
        let state = self.state.upgrade();
        let mut resources = state
            .as_ref()
            .map(|state| state.resources.lock().unwrap_or_else(|e| e.into_inner()));
        let next = self.try_reserve_local(bytes)?;
        if let Some(ledger) = resources
            .as_mut()
            .and_then(|resources| resources.get_mut(&self.resource_id))
            && let Err(error) = ledger.try_reserve(bytes)
        {
            self.release_local(bytes);
            return Err(error);
        }
        update_max(&self.high_water, next);
        Ok(ReservationToken {
            account: Arc::clone(self),
            bytes,
        })
    }

    fn try_reserve_local(&self, bytes: u64) -> MemoryResult<u64> {
        self.reserved
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|&next| next <= self.capacity)
            })
            .map(|previous| previous + bytes)
            .map_err(|_| self.budget_exceeded(bytes))
    }

    fn release(&self, bytes: u64) {
        if let Some(state) = self.state.upgrade() {
            let mut resources = state.resources.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(ledger) = resources.get_mut(&self.resource_id) {
                ledger.release(bytes);
            }
        }
        self.release_local(bytes);
    }

    fn release_local(&self, bytes: u64) {
        self.reserved.fetch_sub(bytes, Ordering::AcqRel);
    }
    fn budget_exceeded(&self, requested: u64) -> MemoryError {
        MemoryError::BudgetExceeded {
            resource: self.resource_id.clone(),
            requested,
            remaining: self
                .capacity
                .saturating_sub(self.reserved.load(Ordering::Acquire)),
        }
    }
}

#[derive(Debug)]
struct ReservationToken {
    account: Arc<ReservationAccount>,
    bytes: u64,
}
impl Drop for ReservationToken {
    fn drop(&mut self) {
        self.account.release(self.bytes);
    }
}

impl MemoryPool {
    /// Requested budget specification.
    pub fn requested(&self) -> MemoryBudget {
        self.inner.requested
    }
    /// Resolved pool capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.inner.accounting.capacity
    }
    /// Currently reserved bytes.
    pub fn reserved(&self) -> u64 {
        self.inner.accounting.reserved.load(Ordering::Acquire)
    }
    /// Remaining reservable bytes.
    pub fn remaining(&self) -> u64 {
        self.capacity().saturating_sub(self.reserved())
    }
    /// Highest concurrent reservation observed by this pool.
    pub fn high_water(&self) -> u64 {
        self.inner.accounting.high_water.load(Ordering::Acquire)
    }
    /// Attempts to reserve `bytes` until the returned lease is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::BudgetExceeded`] if the local or shared resource
    /// limit lacks sufficient remaining capacity.
    pub fn reserve(&self, bytes: u64) -> MemoryResult<MemoryLease> {
        Ok(MemoryLease {
            inner: Arc::new(MemoryLeaseInner {
                reservation: self.inner.accounting.try_reserve(bytes)?,
            }),
        })
    }
    /// Returns a snapshot of this pool's planning and usage.
    pub fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            resource_id: self.inner.accounting.resource_id.clone(),
            requested: self.requested(),
            effective_bytes: self.capacity(),
            reserved_bytes: self.reserved(),
            remaining_bytes: self.remaining(),
            high_water_bytes: self.high_water(),
        }
    }
}

/// A live memory reservation. Clones share one reservation.
#[derive(Clone, Debug)]
pub struct MemoryLease {
    inner: Arc<MemoryLeaseInner>,
}
#[derive(Debug)]
struct MemoryLeaseInner {
    reservation: ReservationToken,
}
impl MemoryLease {
    /// Reserved byte count.
    pub fn bytes(&self) -> u64 {
        self.inner.reservation.bytes
    }
}

fn update_max(value: &AtomicU64, candidate: u64) {
    let mut current = value.load(Ordering::Acquire);
    while candidate > current {
        match value.compare_exchange_weak(current, candidate, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryResourceKind, MemoryState};
    use std::thread;

    fn resource() -> MemoryResource {
        MemoryResource {
            id: "test".into(),
            name: "Test".into(),
            kind: MemoryResourceKind::Device,
            total_bytes: Some(1_000),
            available_bytes: Some(500),
            capacity_source: CapacitySource::User,
            device_identity: None,
        }
    }
    fn resource_reserved(state: &MemoryState) -> u64 {
        state
            .report()
            .resources
            .into_iter()
            .find(|r| r.resource.id == "test")
            .unwrap()
            .laddu_reserved_bytes
    }

    fn concurrent_reservations(pool: &MemoryPool) -> Vec<MemoryLease> {
        (0..32)
            .map(|_| {
                let pool = pool.clone();
                thread::spawn(move || pool.reserve(10))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|handle| handle.join().unwrap().ok())
            .collect()
    }

    #[test]
    fn cloned_leases_release_once_and_preserve_high_water() {
        let state = MemoryState::discover();
        state.register_device(resource());
        let pool = state.pool("test", MemoryBudget::Bytes(300)).unwrap();
        let zero = pool.reserve(0).unwrap();
        assert_eq!((pool.reserved(), resource_reserved(&state)), (0, 0));
        drop(zero);
        let lease = pool.reserve(120).unwrap();
        let clone = lease.clone();
        assert_eq!((pool.reserved(), resource_reserved(&state)), (120, 120));
        drop(lease);
        assert_eq!(pool.reserved(), 120);
        drop(clone);
        assert_eq!((pool.reserved(), pool.high_water()), (0, 120));
        assert_eq!(resource_reserved(&state), 0);
    }

    #[test]
    fn leases_enforce_release_and_align_shared_capacity() {
        let state = MemoryState::discover();
        state.register_device(resource());
        let first = state.pool("test", MemoryBudget::Bytes(400)).unwrap();
        let second = state.pool("test", MemoryBudget::Bytes(400)).unwrap();
        let first_lease = first.reserve(300).unwrap();
        assert!(second.reserve(201).is_err());
        assert_eq!((second.reserved(), resource_reserved(&state)), (0, 300));
        let second_lease = second.reserve(200).unwrap();
        assert_eq!(resource_reserved(&state), 500);
        drop(first_lease);
        drop(second_lease);
        assert_eq!(resource_reserved(&state), 0);
    }

    #[test]
    fn overflow_failure_has_no_side_effects() {
        let state = MemoryState::discover();
        let maximum = MemoryResource {
            total_bytes: Some(u64::MAX),
            available_bytes: Some(u64::MAX),
            ..resource()
        };
        state.register_device(maximum);
        let pool = state.pool("test", MemoryBudget::Bytes(u64::MAX)).unwrap();
        let lease = pool.reserve(u64::MAX).unwrap();
        assert!(pool.reserve(1).is_err());
        assert_eq!((pool.reserved(), pool.high_water()), (u64::MAX, u64::MAX));
        drop(lease);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn concurrent_reservations_are_atomic_with_and_without_state() {
        let state = MemoryState::discover();
        state.register_device(resource());
        let attached = state.pool("test", MemoryBudget::Bytes(100)).unwrap();
        let leases = concurrent_reservations(&attached);
        assert_eq!((leases.len(), attached.reserved()), (10, 100));
        drop(leases);
        assert_eq!(attached.reserved(), 0);

        let detached = state.pool("test", MemoryBudget::Bytes(100)).unwrap();
        drop(state);
        let leases = concurrent_reservations(&detached);
        assert_eq!((leases.len(), detached.reserved()), (10, 100));
        drop(leases);
        assert_eq!(detached.reserved(), 0);
    }
}
