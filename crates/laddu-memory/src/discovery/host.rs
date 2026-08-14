use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, get_current_pid};

use super::{CapacitySnapshot, ProbeFailure};
use crate::{CapacitySource, MemoryResource, MemoryResourceKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProcessSnapshot {
    pub(crate) resident_bytes: u64,
    pub(crate) virtual_bytes: u64,
}

pub(crate) fn discover_host() -> MemoryResource {
    let mut system = System::new();
    system.refresh_memory();
    let mut snapshot = CapacitySnapshot::new(
        system.total_memory(),
        system.available_memory(),
        CapacitySource::OperatingSystem,
    )
    .unwrap_or(CapacitySnapshot {
        total_bytes: system.total_memory(),
        available_bytes: system.available_memory(),
        source: CapacitySource::OperatingSystem,
    });
    if let Ok(cgroup) = discover_cgroup_memory()
        && cgroup.total_bytes < snapshot.total_bytes
    {
        snapshot = CapacitySnapshot {
            available_bytes: snapshot.available_bytes.min(cgroup.available_bytes),
            ..cgroup
        };
    }
    MemoryResource {
        id: "host".into(),
        name: "Host memory".into(),
        kind: MemoryResourceKind::Host,
        total_bytes: Some(snapshot.total_bytes),
        available_bytes: Some(snapshot.available_bytes),
        capacity_source: snapshot.source,
        device_identity: None,
    }
}

fn discover_cgroup_memory() -> Result<CapacitySnapshot, ProbeFailure> {
    let pid = get_current_pid().map_err(|_| ProbeFailure::Unavailable)?;
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let limits = system
        .process(pid)
        .and_then(|process| process.cgroup_limits())
        .ok_or(ProbeFailure::Unavailable)?;
    CapacitySnapshot::new(
        limits.total_memory,
        limits.free_memory,
        CapacitySource::Cgroup,
    )
}

pub(crate) fn discover_process_memory() -> Result<ProcessSnapshot, ProbeFailure> {
    let pid = get_current_pid().map_err(|_| ProbeFailure::Unavailable)?;
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let process = system.process(pid).ok_or(ProbeFailure::Unavailable)?;
    Ok(ProcessSnapshot {
        resident_bytes: process.memory(),
        virtual_bytes: process.virtual_memory(),
    })
}
