//! Public memory planning and reservation workflow.

use laddu_memory::{MemoryBudget, MemoryState};

#[test]
fn public_workflow_reserves_and_releases_memory() {
    let state = MemoryState::discover();
    state.override_device_capacity("test", "Test", 1_000, Some(500));
    let pool = state.pool("test", MemoryBudget::bytes(300)).unwrap();
    let lease = pool.reserve(200).unwrap();
    assert_eq!(
        (lease.bytes(), pool.reserved(), pool.remaining()),
        (200, 200, 100)
    );
    drop(lease);
    assert_eq!(
        (pool.reserved(), pool.remaining(), pool.high_water()),
        (0, 300, 200)
    );
}
