use crate::{WgpuError, WgpuResult};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum WgpuPrecision {
    #[default]
    Auto,
    F32,
    F64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum WgpuDeviceSelector {
    #[default]
    Auto,
    Index(usize),
    PciBusId(String),
    Name(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WgpuOptions {
    pub device: WgpuDeviceSelector,
    pub memory_budget: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuAdapterInfo {
    pub index: usize,
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub pci_bus_id: String,
    pub driver: String,
    pub driver_info: String,
    pub backend: String,
    pub supports_f64: bool,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
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

    fn hardware_priority(&self) -> u8 {
        match self.device_type.as_str() {
            "DiscreteGpu" => 0,
            "IntegratedGpu" => 1,
            "VirtualGpu" => 2,
            "Other" => 3,
            "Cpu" => 4,
            _ => 5,
        }
    }
}

#[derive(Debug)]
pub struct WgpuContext {
    info: WgpuAdapterInfo,
    precision: WgpuPrecision,
    memory_budget: Option<usize>,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl WgpuContext {
    pub fn info(&self) -> &WgpuAdapterInfo {
        &self.info
    }

    pub fn precision(&self) -> WgpuPrecision {
        self.precision
    }

    pub fn memory_budget(&self) -> Option<usize> {
        self.memory_budget
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

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
    pub fn adapters(&self) -> Vec<WgpuAdapterInfo> {
        pollster::block_on(self.instance.enumerate_adapters(wgpu::Backends::all()))
            .iter()
            .enumerate()
            .map(|(index, adapter)| WgpuAdapterInfo::from_adapter(index, adapter))
            .collect()
    }

    pub fn open(&self, options: &WgpuOptions, precision: WgpuPrecision) -> WgpuResult<WgpuContext> {
        if options.memory_budget == Some(0) {
            return Err(WgpuError::InvalidMemoryBudget);
        }
        let adapter = match &options.device {
            WgpuDeviceSelector::Auto => {
                pollster::block_on(self.instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..wgpu::RequestAdapterOptions::default()
                }))
                .map_err(|_| WgpuError::NoAdapters)?
            }
            selector => {
                let adapters =
                    pollster::block_on(self.instance.enumerate_adapters(wgpu::Backends::all()));
                let infos = adapters
                    .iter()
                    .enumerate()
                    .map(|(index, adapter)| WgpuAdapterInfo::from_adapter(index, adapter))
                    .collect::<Vec<_>>();
                let index = Self::select_adapter_index(&infos, selector)
                    .ok_or_else(|| WgpuError::AdapterNotFound(selector.clone()))?;
                adapters
                    .into_iter()
                    .nth(index)
                    .ok_or(WgpuError::NoAdapters)?
            }
        };
        let info = WgpuAdapterInfo::from_adapter(0, &adapter);
        let precision = match precision {
            WgpuPrecision::Auto | WgpuPrecision::F32 => WgpuPrecision::F32,
            WgpuPrecision::F64 if info.supports_f64 && info.backend == "Vulkan" => {
                WgpuPrecision::F64
            }
            WgpuPrecision::F64 => {
                return Err(WgpuError::UnsupportedPrecision {
                    adapter: info.name,
                    precision,
                });
            }
        };
        let required_features = if precision == WgpuPrecision::F64 {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("laddu-wgpu"),
            required_features,
            ..wgpu::DeviceDescriptor::default()
        }))
        .map_err(|error| WgpuError::RequestDevice(error.to_string()))?;
        Ok(WgpuContext {
            info,
            precision,
            memory_budget: options.memory_budget,
            device,
            queue,
        })
    }

    fn select_adapter_index(
        adapters: &[WgpuAdapterInfo],
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(index: usize, name: &str, device_type: &str, pci_bus_id: &str) -> WgpuAdapterInfo {
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

    #[test]
    fn selectors_are_deterministic_and_auto_prefers_hardware() {
        let adapters = [
            adapter(0, "software", "Cpu", ""),
            adapter(1, "integrated", "IntegratedGpu", "0000:02:00.0"),
            adapter(2, "discrete", "DiscreteGpu", "0000:01:00.0"),
        ];
        assert_eq!(
            WgpuBackend::select_adapter_index(&adapters, &WgpuDeviceSelector::Auto),
            Some(2)
        );
        assert_eq!(
            WgpuBackend::select_adapter_index(&adapters, &WgpuDeviceSelector::Index(1)),
            Some(1)
        );
        assert_eq!(
            WgpuBackend::select_adapter_index(
                &adapters,
                &WgpuDeviceSelector::PciBusId("0000:01:00.0".into())
            ),
            Some(2)
        );
        assert_eq!(
            WgpuBackend::select_adapter_index(
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
