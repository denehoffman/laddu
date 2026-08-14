use super::{CapacitySnapshot, ProbeFailure, ProbeOutcome};
use crate::CapacitySource;

pub(super) fn probe(pci_bus_id: &str) -> ProbeOutcome {
    if pci_bus_id.is_empty() {
        return Err(ProbeFailure::MissingIdentity);
    }
    let nvml = nvml_wrapper::Nvml::init().map_err(|_| ProbeFailure::Unavailable)?;
    let device = nvml
        .device_by_pci_bus_id(pci_bus_id)
        .map_err(|_| ProbeFailure::AdapterMismatch)?;
    let memory = device
        .memory_info()
        .map_err(|_| ProbeFailure::Unavailable)?;
    CapacitySnapshot::new(memory.total, memory.free, CapacitySource::Nvml)
}
