use crate::{WgpuError, WgpuResult};
use serde::{Deserialize, Serialize};

/// Floating-point precision requested for WebGPU execution.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WgpuPrecision {
    /// Select a supported precision automatically.
    #[default]
    Auto,
    /// Use 32-bit floating-point arithmetic.
    F32,
    /// Use 64-bit floating-point arithmetic.
    F64,
}

/// Rule used to select a WebGPU adapter.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WgpuDeviceSelector {
    /// Select a high-performance adapter automatically.
    #[default]
    Auto,
    /// Select the adapter at the given enumeration index.
    Index(usize),
    /// Select an adapter by PCI bus identifier.
    PciBusId(String),
    /// Select an adapter whose name matches the supplied string.
    Name(String),
}

/// Options used when opening a WebGPU execution context.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuOptions {
    /// Adapter selection rule.
    pub device: WgpuDeviceSelector,
    /// Optional upper bound, in bytes, for resident allocations.
    pub memory_budget: Option<usize>,
}

/// Capabilities and identifying information for a WebGPU adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuAdapterInfo {
    /// Zero-based adapter enumeration index.
    pub index: usize,
    /// Human-readable adapter name.
    pub name: String,
    /// PCI vendor identifier.
    pub vendor: u32,
    /// PCI device identifier.
    pub device: u32,
    /// Adapter device-class description.
    pub device_type: String,
    /// PCI bus identifier, when reported by the backend.
    pub pci_bus_id: String,
    /// Driver name.
    pub driver: String,
    /// Additional driver version or implementation information.
    pub driver_info: String,
    /// Graphics API backend name.
    pub backend: String,
    /// Whether shader `f64` arithmetic is supported.
    pub supports_f64: bool,
    /// Maximum buffer allocation size in bytes.
    pub max_buffer_size: u64,
    /// Maximum storage-buffer binding size in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum number of invocations along a workgroup's x dimension.
    pub max_compute_workgroup_size_x: u32,
}

impl WgpuAdapterInfo {
    fn from_adapter(index: usize, adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let limits = adapter.limits();
        Self {
            index,
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            pci_bus_id: info.device_pci_bus_id,
            driver: info.driver,
            driver_info: info.driver_info,
            backend: format!("{:?}", info.backend),
            supports_f64: adapter.features().contains(wgpu::Features::SHADER_F64),
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        }
    }
}

#[derive(Clone, Debug)]
struct AdapterSelectionInfo {
    index: usize,
    name: String,
    pci_bus_id: String,
    device_type: wgpu::DeviceType,
}

impl AdapterSelectionInfo {
    fn from_adapter(index: usize, adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        Self {
            index,
            name: info.name,
            pci_bus_id: info.device_pci_bus_id,
            device_type: info.device_type,
        }
    }

    fn hardware_priority(&self) -> u8 {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        }
    }
}

struct SelectedAdapter {
    adapter: wgpu::Adapter,
    info: WgpuAdapterInfo,
    backend: wgpu::Backend,
}

impl SelectedAdapter {
    fn from_adapter(index: usize, adapter: wgpu::Adapter) -> Self {
        let backend = adapter.get_info().backend;
        let info = WgpuAdapterInfo::from_adapter(index, &adapter);
        Self {
            adapter,
            info,
            backend,
        }
    }
}

/// An opened WebGPU device, queue, and resolved execution configuration.
#[derive(Debug)]
pub struct WgpuContext {
    info: WgpuAdapterInfo,
    precision: WgpuPrecision,
    memory_budget: Option<usize>,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl WgpuContext {
    /// Returns information about the selected adapter.
    pub fn info(&self) -> &WgpuAdapterInfo {
        &self.info
    }

    /// Returns the resolved shader precision.
    pub fn precision(&self) -> WgpuPrecision {
        self.precision
    }

    /// Returns the configured resident-memory budget in bytes.
    pub fn memory_budget(&self) -> Option<usize> {
        self.memory_budget
    }

    /// Replaces the tracked resident-memory budget after device discovery.
    pub fn set_memory_budget(&mut self, bytes: usize) {
        self.memory_budget = Some(bytes);
    }

    /// Returns the underlying WebGPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns the underlying WebGPU submission queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Discovers WebGPU adapters and opens execution contexts.
#[derive(Clone, Debug)]
pub struct WgpuBackend {
    instance: wgpu::Instance,
}

impl Default for WgpuBackend {
    fn default() -> Self {
        Self {
            instance: wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            ),
        }
    }
}

impl WgpuBackend {
    /// Enumerates all available adapters and their capabilities.
    pub fn adapters(&self) -> Vec<WgpuAdapterInfo> {
        pollster::block_on(self.instance.enumerate_adapters(wgpu::Backends::all()))
            .iter()
            .enumerate()
            .map(|(index, adapter)| WgpuAdapterInfo::from_adapter(index, adapter))
            .collect()
    }

    /// Selects an adapter and opens a device using the requested precision.
    ///
    /// # Errors
    ///
    /// Returns [`WgpuError`] when the memory budget is invalid, no requested
    /// adapter is available, required precision features are unsupported, or
    /// device creation fails.
    pub fn open(&self, options: &WgpuOptions, precision: WgpuPrecision) -> WgpuResult<WgpuContext> {
        validate_options(options)?;
        let selected = self.select_adapter(&options.device)?;
        let precision = resolve_precision(
            precision,
            &selected.info.name,
            selected.info.supports_f64,
            selected.backend,
        )?;
        let (device, queue) = request_device(&selected.adapter, precision)?;
        Ok(WgpuContext {
            info: selected.info,
            precision,
            memory_budget: options.memory_budget,
            device,
            queue,
        })
    }

    fn select_adapter(&self, selector: &WgpuDeviceSelector) -> WgpuResult<SelectedAdapter> {
        match selector {
            WgpuDeviceSelector::Auto => {
                let adapter = pollster::block_on(self.instance.request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        ..wgpu::RequestAdapterOptions::default()
                    },
                ))
                .map_err(|_| WgpuError::NoAdapters)?;
                let selected_info = adapter.get_info();
                let adapters =
                    pollster::block_on(self.instance.enumerate_adapters(wgpu::Backends::all()));
                let infos = adapters
                    .iter()
                    .map(wgpu::Adapter::get_info)
                    .collect::<Vec<_>>();
                let index = adapter_index_from_infos(&infos, &selected_info)
                    .ok_or(WgpuError::NoAdapters)?;
                Ok(SelectedAdapter::from_adapter(index, adapter))
            }
            selector => {
                let adapters =
                    pollster::block_on(self.instance.enumerate_adapters(wgpu::Backends::all()));
                let selection = adapters
                    .iter()
                    .enumerate()
                    .map(|(index, adapter)| AdapterSelectionInfo::from_adapter(index, adapter))
                    .collect::<Vec<_>>();
                let index = select_enumerated_adapter_index(&selection, selector)
                    .ok_or_else(|| WgpuError::AdapterNotFound(selector.clone()))?;
                let adapter = adapters
                    .into_iter()
                    .nth(index)
                    .ok_or(WgpuError::NoAdapters)?;
                Ok(SelectedAdapter::from_adapter(index, adapter))
            }
        }
    }
}

fn validate_options(options: &WgpuOptions) -> WgpuResult<()> {
    if options.memory_budget == Some(0) {
        return Err(WgpuError::InvalidMemoryBudget);
    }
    Ok(())
}

fn resolve_precision(
    precision: WgpuPrecision,
    adapter_name: &str,
    supports_f64: bool,
    backend: wgpu::Backend,
) -> WgpuResult<WgpuPrecision> {
    match precision {
        WgpuPrecision::Auto | WgpuPrecision::F32 => Ok(WgpuPrecision::F32),
        WgpuPrecision::F64 if supports_f64 && backend == wgpu::Backend::Vulkan => {
            Ok(WgpuPrecision::F64)
        }
        WgpuPrecision::F64 => Err(WgpuError::UnsupportedPrecision {
            adapter: adapter_name.to_owned(),
            precision,
        }),
    }
}

fn request_device(
    adapter: &wgpu::Adapter,
    precision: WgpuPrecision,
) -> WgpuResult<(wgpu::Device, wgpu::Queue)> {
    let required_features = if precision == WgpuPrecision::F64 {
        wgpu::Features::SHADER_F64
    } else {
        wgpu::Features::empty()
    };
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("laddu-wgpu"),
        required_features,
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|error| WgpuError::RequestDevice(error.to_string()))
}

/// Selects from an already enumerated list using the deterministic metadata policy.
///
/// `WgpuBackend::open(WgpuDeviceSelector::Auto)` intentionally delegates to
/// `wgpu::Instance::request_adapter`; this helper does not define that runtime
/// selection policy and is used for explicit selectors and characterization tests.
fn select_enumerated_adapter_index(
    adapters: &[AdapterSelectionInfo],
    selector: &WgpuDeviceSelector,
) -> Option<usize> {
    match selector {
        WgpuDeviceSelector::Auto => adapters
            .iter()
            .min_by_key(|adapter| (adapter.hardware_priority(), adapter.index))
            .map(|adapter| adapter.index),
        WgpuDeviceSelector::Index(index) => adapters.get(*index).map(|adapter| adapter.index),
        WgpuDeviceSelector::PciBusId(id) => adapters
            .iter()
            .find(|adapter| adapter.pci_bus_id.eq_ignore_ascii_case(id))
            .map(|adapter| adapter.index),
        WgpuDeviceSelector::Name(name) => adapters
            .iter()
            .find(|adapter| adapter.name.eq_ignore_ascii_case(name))
            .map(|adapter| adapter.index),
    }
}

fn adapter_index_from_infos(
    adapters: &[wgpu::AdapterInfo],
    selected: &wgpu::AdapterInfo,
) -> Option<usize> {
    adapters.iter().position(|candidate| candidate == selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_adapter_values_roundtrip_through_json() {
        let options = WgpuOptions {
            device: WgpuDeviceSelector::Name("accelerator".into()),
            memory_budget: Some(4096),
        };
        let info = adapter_info(2, "accelerator", "DiscreteGpu", "0000:01:00.0");

        let options_json = serde_json::to_string(&options).unwrap();
        assert_eq!(
            serde_json::from_str::<WgpuOptions>(&options_json).unwrap(),
            options
        );
        let info_json = serde_json::to_string(&info).unwrap();
        assert_eq!(
            serde_json::from_str::<WgpuAdapterInfo>(&info_json).unwrap(),
            info
        );
    }

    fn adapter_info(
        index: usize,
        name: &str,
        device_type: &str,
        pci_bus_id: &str,
    ) -> WgpuAdapterInfo {
        WgpuAdapterInfo {
            index,
            name: name.into(),
            vendor: 0,
            device: 0,
            device_type: device_type.into(),
            pci_bus_id: pci_bus_id.into(),
            driver: String::new(),
            driver_info: String::new(),
            backend: "Vulkan".into(),
            supports_f64: false,
            max_buffer_size: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroup_size_x: 0,
        }
    }

    fn adapter(
        index: usize,
        name: &str,
        device_type: wgpu::DeviceType,
        pci_bus_id: &str,
    ) -> AdapterSelectionInfo {
        AdapterSelectionInfo {
            index,
            name: name.into(),
            pci_bus_id: pci_bus_id.into(),
            device_type,
        }
    }

    #[test]
    fn f64_requires_both_feature_and_allowed_backend() {
        assert!(
            resolve_precision(WgpuPrecision::F64, "vulkan", false, wgpu::Backend::Vulkan).is_err()
        );
        assert!(
            resolve_precision(WgpuPrecision::F64, "metal", true, wgpu::Backend::Metal).is_err()
        );
        assert_eq!(
            resolve_precision(WgpuPrecision::F64, "vulkan", true, wgpu::Backend::Vulkan).unwrap(),
            WgpuPrecision::F64
        );
        assert_eq!(
            resolve_precision(WgpuPrecision::Auto, "metal", false, wgpu::Backend::Metal).unwrap(),
            WgpuPrecision::F32
        );
    }

    #[test]
    fn adapter_info_matching_recovers_enumeration_index() {
        let mut infos = [
            wgpu::AdapterInfo::new(wgpu::DeviceType::Cpu, wgpu::Backend::Gl),
            wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan),
            wgpu::AdapterInfo::new(wgpu::DeviceType::IntegratedGpu, wgpu::Backend::Metal),
        ];
        infos[0].name = "software".into();
        infos[1].name = "discrete".into();
        infos[2].name = "integrated".into();

        assert_eq!(adapter_index_from_infos(&infos, &infos[1]), Some(1));
        let missing = wgpu::AdapterInfo::new(wgpu::DeviceType::Other, wgpu::Backend::Dx12);
        assert_eq!(adapter_index_from_infos(&infos, &missing), None);
    }

    #[test]
    fn enumerated_selector_policy_is_deterministic_and_prefers_hardware() {
        let adapters = [
            adapter(0, "software", wgpu::DeviceType::Cpu, ""),
            adapter(
                1,
                "integrated",
                wgpu::DeviceType::IntegratedGpu,
                "0000:02:00.0",
            ),
            adapter(2, "discrete", wgpu::DeviceType::DiscreteGpu, "0000:01:00.0"),
        ];
        assert_eq!(
            select_enumerated_adapter_index(&adapters, &WgpuDeviceSelector::Auto),
            Some(2)
        );
        assert_eq!(
            select_enumerated_adapter_index(&adapters, &WgpuDeviceSelector::Index(1)),
            Some(1)
        );
        assert_eq!(
            select_enumerated_adapter_index(
                &adapters,
                &WgpuDeviceSelector::PciBusId("0000:01:00.0".into())
            ),
            Some(2)
        );
        assert_eq!(
            select_enumerated_adapter_index(
                &adapters,
                &WgpuDeviceSelector::Name("DISCRETE".into())
            ),
            Some(2)
        );
    }

    #[test]
    fn zero_memory_budget_is_rejected_before_adapter_selection() {
        let error = WgpuBackend::default()
            .open(
                &WgpuOptions {
                    memory_budget: Some(0),
                    ..WgpuOptions::default()
                },
                WgpuPrecision::F32,
            )
            .unwrap_err();

        assert!(matches!(error, WgpuError::InvalidMemoryBudget));
    }
}
