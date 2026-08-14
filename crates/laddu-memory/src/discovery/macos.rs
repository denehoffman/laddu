use objc2_metal::MTLDevice;

use super::{CapacitySnapshot, ProbeFailure, ProbeOutcome};
use crate::{CapacitySource, DeviceIdentity};

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

pub(super) fn probe(identity: &DeviceIdentity, expected_name: &str) -> ProbeOutcome {
    let devices = objc2_metal::MTLCopyAllDevices();
    let device = (0..devices.count())
        .map(|index| devices.objectAtIndex(index))
        .find(|device| device.name().to_string() == expected_name)
        .or_else(|| {
            (identity.adapter_index < devices.count())
                .then(|| devices.objectAtIndex(identity.adapter_index))
        })
        .ok_or(ProbeFailure::AdapterMismatch)?;
    let total = device.recommendedMaxWorkingSetSize();
    let used = device.currentAllocatedSize() as u64;
    CapacitySnapshot::new(total, total.saturating_sub(used), CapacitySource::Metal)
}
