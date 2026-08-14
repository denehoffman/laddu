mod host;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(feature = "nvml")]
mod nvml;
#[cfg(target_os = "windows")]
mod windows;

pub(crate) use host::{discover_host, discover_process_memory};

use crate::{CapacitySource, MemoryResource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CapacitySnapshot {
    pub(super) total_bytes: u64,
    pub(super) available_bytes: u64,
    pub(super) source: CapacitySource,
}

impl CapacitySnapshot {
    fn new(
        total_bytes: u64,
        available_bytes: u64,
        source: CapacitySource,
    ) -> Result<Self, ProbeFailure> {
        if total_bytes == 0 {
            return Err(ProbeFailure::Malformed);
        }
        Ok(Self {
            total_bytes,
            available_bytes: available_bytes.min(total_bytes),
            source,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProbeFailure {
    MissingIdentity,
    Unsupported,
    Unavailable,
    AdapterMismatch,
    Malformed,
}

pub(super) type ProbeOutcome = Result<CapacitySnapshot, ProbeFailure>;

pub(super) trait MemoryProbe: Send + Sync {
    fn probe_device(&self, resource: &MemoryResource) -> ProbeOutcome;
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SystemMemoryProbe;

impl MemoryProbe for SystemMemoryProbe {
    fn probe_device(&self, resource: &MemoryResource) -> ProbeOutcome {
        let identity = resource
            .device_identity
            .as_ref()
            .ok_or(ProbeFailure::MissingIdentity)?;
        let outcome = Err(ProbeFailure::Unsupported);
        #[cfg(feature = "nvml")]
        let outcome = fallback(outcome, || nvml::probe(&identity.pci_bus_id));
        #[cfg(target_os = "windows")]
        let outcome = fallback(outcome, || windows::probe(identity));
        #[cfg(target_os = "macos")]
        let outcome = fallback(outcome, || macos::probe(identity, &resource.name));
        #[cfg(target_os = "linux")]
        let outcome = fallback(outcome, || linux::probe(&identity.pci_bus_id));
        outcome
    }
}

fn fallback(current: ProbeOutcome, next: impl FnOnce() -> ProbeOutcome) -> ProbeOutcome {
    match current {
        Ok(snapshot) => Ok(snapshot),
        Err(_) => next(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(source: CapacitySource) -> CapacitySnapshot {
        CapacitySnapshot::new(100, 80, source).unwrap()
    }

    #[test]
    fn fallback_uses_the_first_successful_backend() {
        let first = snapshot(CapacitySource::Adaptive);
        let selected = fallback(Ok(first), || panic!("later probe must not run"));
        assert_eq!(selected, Ok(first));

        let second = snapshot(CapacitySource::OperatingSystem);
        let selected = fallback(Err(ProbeFailure::AdapterMismatch), || Ok(second));
        assert_eq!(selected, Ok(second));
    }

    #[test]
    fn capacity_snapshots_reject_zero_and_clamp_available_bytes() {
        assert_eq!(
            CapacitySnapshot::new(0, 0, CapacitySource::Adaptive),
            Err(ProbeFailure::Malformed)
        );
        assert_eq!(
            CapacitySnapshot::new(100, 120, CapacitySource::Adaptive)
                .unwrap()
                .available_bytes,
            100
        );
    }
}
