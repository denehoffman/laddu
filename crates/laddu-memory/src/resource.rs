use serde::{Deserialize, Serialize};

use crate::discovery::{CapacitySnapshot, MemoryProbe, SystemMemoryProbe};

/// The basis used to obtain a resource's capacity information.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacitySource {
    /// Host operating-system telemetry.
    OperatingSystem,
    /// A process or container limit.
    Cgroup,
    /// NVIDIA Management Library telemetry.
    Nvml,
    /// Linux DRM/sysfs telemetry.
    Drm,
    /// Windows DXGI telemetry.
    Dxgi,
    /// Apple Metal working-set telemetry.
    Metal,
    /// Capacity supplied by the user.
    User,
    /// Capacity is not observable and planning is adaptive.
    Adaptive,
}

/// Kind of physical memory resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryResourceKind {
    /// Host RAM.
    Host,
    /// Accelerator-local or unified memory.
    Device,
}

/// Stable information used to match a runtime accelerator to platform telemetry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceIdentity {
    /// Backend adapter index.
    pub adapter_index: usize,
    /// PCI vendor identifier, or zero when unavailable.
    pub vendor_id: u32,
    /// PCI device identifier, or zero when unavailable.
    pub device_id: u32,
    /// PCI bus identifier, or an empty string when unavailable.
    pub pci_bus_id: String,
}

/// Snapshot of one physical memory resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryResource {
    /// Stable resource identifier.
    pub id: String,
    /// Human-readable resource name.
    pub name: String,
    /// Host or accelerator memory.
    pub kind: MemoryResourceKind,
    /// Total capacity when observable.
    pub total_bytes: Option<u64>,
    /// Currently available capacity when observable.
    pub available_bytes: Option<u64>,
    /// Source of capacity information.
    pub capacity_source: CapacitySource,
    /// Accelerator identity used to refresh platform telemetry.
    pub device_identity: Option<DeviceIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResourceValidationError {
    EmptyId,
    AvailableExceedsTotal,
    HostHasDeviceIdentity,
    HostHasDeviceTelemetry,
    DeviceHasHostTelemetry,
    TelemetryHasNoCapacity,
}

impl MemoryResource {
    /// Creates an accelerator resource whose physical capacity is unavailable.
    pub fn adaptive_device(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind: MemoryResourceKind::Device,
            total_bytes: None,
            available_bytes: None,
            capacity_source: CapacitySource::Adaptive,
            device_identity: None,
        }
    }

    /// Discovers accelerator capacity from platform telemetry, falling back to
    /// an adaptive backend limit when dedicated-memory telemetry is absent.
    pub fn discover_device(
        id: impl Into<String>,
        name: impl Into<String>,
        identity: DeviceIdentity,
        fallback_bytes: u64,
    ) -> Self {
        let mut resource = Self::adaptive_device(id, name);
        resource.device_identity = Some(identity);
        if let Ok(snapshot) = SystemMemoryProbe.probe_device(&resource) {
            resource.apply_capacity_snapshot(snapshot);
            return resource;
        }
        resource.total_bytes = Some(fallback_bytes);
        resource.available_bytes = Some(fallback_bytes);
        resource
    }

    /// Creates a resource with an explicit user-provided capacity.
    pub fn with_capacity(mut self, total_bytes: u64, available_bytes: Option<u64>) -> Self {
        self.total_bytes = Some(total_bytes);
        self.available_bytes = Some(available_bytes.unwrap_or(total_bytes).min(total_bytes));
        self.capacity_source = CapacitySource::User;
        self
    }

    pub(crate) fn apply_capacity_snapshot(&mut self, snapshot: CapacitySnapshot) {
        self.total_bytes = Some(snapshot.total_bytes);
        self.available_bytes = Some(snapshot.available_bytes);
        self.capacity_source = snapshot.source;
    }

    pub(crate) fn validate(&self) -> Result<(), ResourceValidationError> {
        if self.id.is_empty() {
            return Err(ResourceValidationError::EmptyId);
        }
        if let (Some(total), Some(available)) = (self.total_bytes, self.available_bytes)
            && available > total
        {
            return Err(ResourceValidationError::AvailableExceedsTotal);
        }
        match self.kind {
            MemoryResourceKind::Host => {
                if self.device_identity.is_some() {
                    return Err(ResourceValidationError::HostHasDeviceIdentity);
                }
                if matches!(
                    self.capacity_source,
                    CapacitySource::Nvml
                        | CapacitySource::Drm
                        | CapacitySource::Dxgi
                        | CapacitySource::Metal
                        | CapacitySource::Adaptive
                ) {
                    return Err(ResourceValidationError::HostHasDeviceTelemetry);
                }
            }
            MemoryResourceKind::Device => {
                if matches!(
                    self.capacity_source,
                    CapacitySource::OperatingSystem | CapacitySource::Cgroup
                ) {
                    return Err(ResourceValidationError::DeviceHasHostTelemetry);
                }
            }
        }
        if self.capacity_source != CapacitySource::Adaptive
            && (self.total_bytes.is_none() || self.available_bytes.is_none())
        {
            return Err(ResourceValidationError::TelemetryHasNoCapacity);
        }
        Ok(())
    }

    pub(crate) fn normalize_capacity(&mut self) {
        if let Some(total) = self.total_bytes
            && let Some(available) = self.available_bytes.as_mut()
        {
            *available = (*available).min(total);
        }
    }

    pub(crate) fn effective_available(&self) -> u64 {
        self.available_bytes
            .or(self.total_bytes)
            .unwrap_or(u64::MAX)
    }

    pub(crate) fn is_refreshable(&self) -> bool {
        self.kind == MemoryResourceKind::Device
            && self.capacity_source != CapacitySource::User
            && self.device_identity.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::discover_host;

    fn device() -> MemoryResource {
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

    #[test]
    fn validation_names_invalid_field_combinations() {
        let mut invalid = device();
        invalid.id.clear();
        assert_eq!(invalid.validate(), Err(ResourceValidationError::EmptyId));
        let mut invalid = device();
        invalid.available_bytes = Some(1_001);
        assert_eq!(
            invalid.validate(),
            Err(ResourceValidationError::AvailableExceedsTotal)
        );
        let mut invalid = discover_host();
        invalid.device_identity = Some(DeviceIdentity {
            adapter_index: 0,
            vendor_id: 1,
            device_id: 2,
            pci_bus_id: "bus-0".into(),
        });
        assert_eq!(
            invalid.validate(),
            Err(ResourceValidationError::HostHasDeviceIdentity)
        );
        let mut invalid = device();
        invalid.capacity_source = CapacitySource::OperatingSystem;
        assert_eq!(
            invalid.validate(),
            Err(ResourceValidationError::DeviceHasHostTelemetry)
        );

        let mut invalid = device();
        invalid.capacity_source = CapacitySource::Nvml;
        invalid.total_bytes = None;
        assert_eq!(
            invalid.validate(),
            Err(ResourceValidationError::TelemetryHasNoCapacity)
        );

        let mut invalid = discover_host();
        invalid.capacity_source = CapacitySource::Metal;
        assert_eq!(
            invalid.validate(),
            Err(ResourceValidationError::HostHasDeviceTelemetry)
        );
    }
}
