use windows::{
    Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
        IDXGIAdapter3, IDXGIFactory1,
    },
    core::Interface,
};

use super::{CapacitySnapshot, ProbeFailure, ProbeOutcome};
use crate::{CapacitySource, DeviceIdentity};

pub(super) fn probe(identity: &DeviceIdentity) -> ProbeOutcome {
    // SAFETY: DXGI factory and adapter methods own their returned COM interfaces,
    // and all output pointers are provided by the windows crate.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().map_err(|_| ProbeFailure::Unavailable)?;
        let mut fallback = None;
        let mut matched = false;
        for index in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(index) else {
                break;
            };
            let Ok(description) = adapter.GetDesc1() else {
                continue;
            };
            if description.VendorId != identity.vendor_id
                || description.DeviceId != identity.device_id
            {
                continue;
            }
            matched = true;
            let adapter: IDXGIAdapter3 = adapter.cast().map_err(|_| ProbeFailure::Unavailable)?;
            let mut memory = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut memory)
                .map_err(|_| ProbeFailure::Unavailable)?;
            let snapshot = CapacitySnapshot::new(
                memory.Budget,
                memory.Budget.saturating_sub(memory.CurrentUsage),
                CapacitySource::Dxgi,
            )?;
            if index as usize == identity.adapter_index {
                return Ok(snapshot);
            }
            fallback.get_or_insert(snapshot);
        }
        fallback.ok_or(if matched {
            ProbeFailure::Malformed
        } else {
            ProbeFailure::AdapterMismatch
        })
    }
}
