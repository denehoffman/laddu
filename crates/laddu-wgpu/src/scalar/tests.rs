use std::ops::Range;

use super::WgpuScalarKernel;
use super::batch::ChunkPlan;
use super::memory::GpuMemoryLayout;
use crate::WgpuPrecision;
use laddu_memory::{FootprintOverflow, MemoryFootprint};

#[test]
fn gpu_layout_uses_checked_shared_footprint_components() {
    let layout = GpuMemoryLayout::new(8, 2, 3, 4, 2).unwrap();
    assert_eq!(layout.input_bytes.bytes_per_event, 32);
    assert_eq!(layout.cache_bytes.bytes_per_event, 48);
    assert_eq!(layout.partial_bytes.bytes_per_event, 32);
    assert_eq!(
        layout.prepared_footprint().unwrap(),
        MemoryFootprint::new(32, 113)
    );
    assert_eq!(layout.prepared_resident_bytes(65).unwrap(), 5_884);
}

#[test]
fn gpu_layout_rejects_overflow_before_buffer_geometry() {
    assert!(matches!(
        GpuMemoryLayout::new(8, usize::MAX, 0, 0, 0),
        Err(FootprintOverflow::Multiplication)
    ));
    let layout = GpuMemoryLayout::new(8, 0, 1, 1, 0).unwrap();
    assert_eq!(layout.input_buffer_bytes(usize::MAX).unwrap(), 16);
    assert_eq!(GpuMemoryLayout::groups(63), 1);
    assert_eq!(GpuMemoryLayout::groups(64), 1);
    assert_eq!(GpuMemoryLayout::groups(65), 2);
}

#[test]
fn gpu_layout_covers_empty_and_workgroup_boundaries() {
    for scalar_size in [4, 8] {
        let layout = GpuMemoryLayout::new(scalar_size, 0, 0, 0, 0).unwrap();
        let mut previous = 0;
        for (events, groups) in [(0, 0), (1, 1), (63, 1), (64, 1), (65, 2)] {
            let resident = layout.prepared_resident_bytes(events).unwrap();
            assert!(resident >= previous);
            previous = resident;
            assert_eq!(GpuMemoryLayout::groups(events), groups);
        }
        assert_eq!(
            layout.prepared_resident_bytes(0).unwrap(),
            (scalar_size * 3 + 28) as u64
        );
    }
    let layout = GpuMemoryLayout::new(8, 1, 1, 1, 1).unwrap();
    assert!(
        layout.prepared_resident_bytes(63).unwrap() < layout.prepared_resident_bytes(64).unwrap()
    );
    assert!(
        layout.prepared_resident_bytes(64).unwrap() < layout.prepared_resident_bytes(65).unwrap()
    );
}

#[test]
fn gpu_layout_binding_and_budget_arithmetic_is_checked() {
    let layout = GpuMemoryLayout::new(4, 1, 1, 1, 1).unwrap();
    assert_eq!(layout.binding_limit(120, true), 15);
    assert_eq!(layout.binding_limit(120, false), 15);
    assert_eq!(
        GpuMemoryLayout::new(4, 0, 0, 0, 0)
            .unwrap()
            .binding_limit(0, true),
        u32::MAX as usize
    );
    let footprint = layout.prepared_footprint().unwrap();
    assert_eq!(footprint.checked_peak_bytes(1).unwrap(), 41);
    assert!(
        MemoryFootprint::new(u64::MAX, 1)
            .checked_peak_bytes(1)
            .is_err()
    );
}

#[test]
fn gpu_layout_resident_chunk_estimate_tracks_workgroup_storage() {
    let layout = GpuMemoryLayout::new(4, 0, 0, 7, 2).unwrap();
    assert_eq!(
        layout.prepared_footprint().unwrap(),
        MemoryFootprint::new(24, 29)
    );
    assert_eq!(layout.prepared_resident_bytes(8).unwrap(), 132);
    assert_eq!(layout.prepared_resident_bytes(64).unwrap(), 356);
    assert_eq!(layout.prepared_resident_bytes(65).unwrap(), 416);
}

#[test]
fn refresh_preflight_rejects_a_late_chunk_without_mutating_prepared_layout() {
    let prepared_plan = ChunkPlan::for_batch(129, 64);
    let incoming_plan = ChunkPlan::for_batch(130, 64);
    let prepared_events = prepared_plan
        .ranges
        .iter()
        .map(Range::len)
        .collect::<Vec<_>>();

    assert_eq!(prepared_events, [64, 64, 1]);
    assert_eq!(
        incoming_plan
            .ranges
            .iter()
            .map(Range::len)
            .collect::<Vec<_>>(),
        [64, 64, 2]
    );
    assert!(!incoming_plan.matches_event_counts(prepared_events.iter().copied()));
    assert_eq!(prepared_events, [64, 64, 1]);
}

#[test]
fn refresh_preflight_accepts_compatible_chunks() {
    let prepared_plan = ChunkPlan::for_batch(130, 64);
    let incoming_plan = ChunkPlan::for_batch(130, 64);
    let prepared_events = prepared_plan
        .ranges
        .iter()
        .map(Range::len)
        .collect::<Vec<_>>();

    assert!(incoming_plan.matches_event_counts(prepared_events.iter().copied()));
    assert_eq!(prepared_events, [64, 64, 2]);
}

#[test]
fn chunk_plan_covers_empty_and_partial_boundaries() {
    for (events, expected) in [
        (0, vec![]),
        (1, vec![1]),
        (63, vec![63]),
        (64, vec![64]),
        (65, vec![64, 1]),
    ] {
        let plan = ChunkPlan::for_batch(events, 64);
        assert_eq!(
            plan.ranges.iter().map(Range::len).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn scalar_kernel_accepts_resolved_precisions() {
    assert!(WgpuScalarKernel::validate_precision(WgpuPrecision::F32).is_ok());
    assert!(WgpuScalarKernel::validate_precision(WgpuPrecision::F64).is_ok());
    assert!(matches!(
        WgpuScalarKernel::validate_precision(WgpuPrecision::Auto),
        Err(crate::WgpuError::UnsupportedKernelPrecision(
            WgpuPrecision::Auto
        ))
    ));
}
