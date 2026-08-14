use super::{CapacitySnapshot, ProbeFailure, ProbeOutcome};
use crate::CapacitySource;

pub(super) fn probe(pci_bus_id: &str) -> ProbeOutcome {
    if pci_bus_id.is_empty() {
        return Err(ProbeFailure::MissingIdentity);
    }
    let entries = std::fs::read_dir("/sys/class/drm").map_err(|_| ProbeFailure::Unavailable)?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("card") || name.to_string_lossy().contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let Ok(uevent) = std::fs::read_to_string(device.join("uevent")) else {
            continue;
        };
        let matches_device = uevent.lines().any(|line| {
            line.strip_prefix("PCI_SLOT_NAME=")
                .is_some_and(|slot| slot.eq_ignore_ascii_case(pci_bus_id))
        });
        if !matches_device {
            continue;
        }
        let total = read_sysfs_u64(device.join("mem_info_vram_total"))?;
        let used = read_sysfs_u64(device.join("mem_info_vram_used")).unwrap_or(0);
        return CapacitySnapshot::new(total, total.saturating_sub(used), CapacitySource::Drm);
    }
    Err(ProbeFailure::AdapterMismatch)
}

fn read_sysfs_u64(path: impl AsRef<std::path::Path>) -> Result<u64, ProbeFailure> {
    std::fs::read_to_string(path)
        .map_err(|_| ProbeFailure::Unavailable)?
        .trim()
        .parse()
        .map_err(|_| ProbeFailure::Malformed)
}
