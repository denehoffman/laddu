use laddu_autodiff::AutodiffMode;
use laddu_data::io::Partitioning;
use laddu_runtime::{
    CpuOptions, Device, Execution, ExecutionOptions, GpuBackend, GpuDeviceSelector, GpuOptions,
    JitPolicy, MemoryBudget, MemoryPlan, MemoryResource, MemoryState, NormalizationMode, Precision,
    ThreadPolicy,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};

use super::error::to_py_err;

fn capacity_source_name(source: &laddu_runtime::CapacitySource) -> &'static str {
    match source {
        laddu_runtime::CapacitySource::OperatingSystem => "operating_system",
        laddu_runtime::CapacitySource::Cgroup => "cgroup",
        laddu_runtime::CapacitySource::Nvml => "nvml",
        laddu_runtime::CapacitySource::Drm => "drm",
        laddu_runtime::CapacitySource::Dxgi => "dxgi",
        laddu_runtime::CapacitySource::Metal => "metal",
        laddu_runtime::CapacitySource::User => "user",
        laddu_runtime::CapacitySource::Adaptive => "adaptive",
    }
}

fn normalize(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn parse_precision(value: &str) -> PyResult<Precision> {
    match normalize(value).as_str() {
        "auto" => Ok(Precision::Auto),
        "f32" | "float32" | "single" => Ok(Precision::F32),
        "f64" | "float64" | "double" => Ok(Precision::F64),
        _ => Err(PyValueError::new_err(
            "precision must be 'auto', 'f32', or 'f64'",
        )),
    }
}

fn parse_autodiff(value: &str) -> PyResult<AutodiffMode> {
    match normalize(value).as_str() {
        "auto" => Ok(AutodiffMode::Auto),
        "forward" => Ok(AutodiffMode::Forward),
        "reverse" => Ok(AutodiffMode::Reverse),
        _ => Err(PyValueError::new_err(
            "autodiff must be 'auto', 'forward', or 'reverse'",
        )),
    }
}

fn parse_normalization(value: &str) -> PyResult<NormalizationMode> {
    match normalize(value).as_str() {
        "auto" => Ok(NormalizationMode::Auto),
        "general" => Ok(NormalizationMode::General),
        "verify" => Ok(NormalizationMode::Verify),
        _ => Err(PyValueError::new_err(
            "normalization must be 'auto', 'general', or 'verify'",
        )),
    }
}

fn parse_partitioning(value: &str) -> PyResult<Partitioning> {
    match normalize(value).as_str() {
        "auto" | "contiguous" => Ok(Partitioning::Contiguous),
        "files" | "file_groups" => Ok(Partitioning::FileGroups),
        "rows" => Ok(Partitioning::Rows),
        _ => Err(PyValueError::new_err(
            "partitioning must be 'auto', 'contiguous', 'file_groups', or 'rows'",
        )),
    }
}

fn gpu_selector(device: Option<&Bound<'_, PyAny>>) -> PyResult<GpuDeviceSelector> {
    let Some(device) = device else {
        return Ok(GpuDeviceSelector::Auto);
    };
    if let Ok(index) = device.extract::<usize>() {
        return Ok(GpuDeviceSelector::Index(index));
    }
    if let Ok(name) = device.extract::<String>() {
        if name.contains(':') && name.chars().filter(|character| *character == ':').count() >= 2 {
            return Ok(GpuDeviceSelector::PciBusId(name));
        }
        return Ok(GpuDeviceSelector::Name(name));
    }
    Err(PyValueError::new_err(
        "GPU device must be an integer index, name, PCI bus ID, or None",
    ))
}

pub(crate) fn parse_memory_budget(value: &Bound<'_, PyAny>) -> PyResult<MemoryBudget> {
    if let Ok(value) = value.extract::<PyRef<'_, PyMemoryBudget>>() {
        return Ok(value.inner);
    }
    if let Ok(bytes) = value.extract::<u64>() {
        return Ok(MemoryBudget::Bytes(bytes));
    }
    if let Ok(specification) = value.extract::<String>() {
        return specification
            .parse()
            .map_err(|error: laddu_runtime::MemoryError| PyValueError::new_err(error.to_string()));
    }
    Err(PyValueError::new_err(
        "memory budget must be a MemoryBudget, byte count, or string such as '8 GiB' or '60% available'",
    ))
}

#[pyclass(name = "MemoryBudget", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A portable byte or percentage memory request.
pub struct PyMemoryBudget {
    inner: MemoryBudget,
}

#[pymethods]
impl PyMemoryBudget {
    #[new]
    #[pyo3(signature = (value: "MemoryBudget | int | str"))]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: parse_memory_budget(value)?,
        })
    }

    #[staticmethod]
    fn bytes(bytes: u64) -> Self {
        Self {
            inner: MemoryBudget::Bytes(bytes),
        }
    }

    #[staticmethod]
    fn percent_total(percent: f64) -> PyResult<Self> {
        Ok(Self {
            inner: MemoryBudget::percent_total(percent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[staticmethod]
    fn percent_available(percent: f64) -> PyResult<Self> {
        Ok(Self {
            inner: MemoryBudget::percent_available(percent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn __repr__(&self) -> String {
        format!("MemoryBudget({:?})", self.inner.to_string())
    }
}

#[pyclass(name = "MemoryPlan", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Host and optional accelerator budgets for an execution.
pub struct PyMemoryPlan {
    pub(crate) inner: MemoryPlan,
}

fn parse_memory_plan(value: &Bound<'_, PyAny>) -> PyResult<MemoryPlan> {
    if let Ok(plan) = value.extract::<PyRef<'_, PyMemoryPlan>>() {
        return Ok(plan.inner);
    }
    let budget = parse_memory_budget(value)?;
    Ok(MemoryPlan::host_device(budget, budget))
}

#[pymethods]
impl PyMemoryPlan {
    #[new]
    #[pyo3(signature = (
        *,
        host: "MemoryBudget | int | str",
        device: "MemoryBudget | int | str | None" = None
    ))]
    fn new(host: &Bound<'_, PyAny>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: MemoryPlan {
                host: parse_memory_budget(host)?,
                device: device.map(parse_memory_budget).transpose()?,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryPlan(host={:?}, device={:?})",
            self.inner.host.to_string(),
            self.inner.device.map(|budget| budget.to_string())
        )
    }
}

#[pyclass(name = "MemoryResource", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Total and currently available memory for one physical resource.
pub struct PyMemoryResource {
    inner: MemoryResource,
}

#[pymethods]
impl PyMemoryResource {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn total_bytes(&self) -> Option<u64> {
        self.inner.total_bytes
    }

    #[getter]
    fn available_bytes(&self) -> Option<u64> {
        self.inner.available_bytes
    }

    #[getter]
    fn capacity_source(&self) -> &'static str {
        capacity_source_name(&self.inner.capacity_source)
    }

    #[getter]
    fn adapter_index(&self) -> Option<usize> {
        self.inner
            .device_identity
            .as_ref()
            .map(|identity| identity.adapter_index)
    }

    #[getter]
    fn vendor_id(&self) -> Option<u32> {
        self.inner
            .device_identity
            .as_ref()
            .map(|identity| identity.vendor_id)
    }

    #[getter]
    fn device_id(&self) -> Option<u32> {
        self.inner
            .device_identity
            .as_ref()
            .map(|identity| identity.device_id)
    }

    #[getter]
    fn pci_bus_id(&self) -> Option<&str> {
        self.inner
            .device_identity
            .as_ref()
            .map(|identity| identity.pci_bus_id.as_str())
    }

    #[pyo3(signature = (value: "MemoryBudget | int | str"))]
    fn budget(&self, value: &Bound<'_, PyAny>) -> PyResult<PyMemoryBudget> {
        Ok(PyMemoryBudget {
            inner: parse_memory_budget(value)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryResource(name={:?}, total_bytes={:?}, available_bytes={:?}, source={:?})",
            self.inner.name,
            self.inner.total_bytes,
            self.inner.available_bytes,
            self.inner.capacity_source
        )
    }
}

#[pyclass(name = "MemoryState", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Live host/device discovery and laddu reservation state.
pub struct PyMemoryState {
    inner: MemoryState,
}

#[pymethods]
impl PyMemoryState {
    #[new]
    fn new() -> Self {
        Self {
            inner: MemoryState::discover(),
        }
    }

    #[staticmethod]
    fn current() -> Self {
        Self {
            inner: MemoryState::current(),
        }
    }

    fn refresh(&self) {
        self.inner.refresh();
    }

    #[pyo3(signature = (id, total_bytes, *, available_bytes=None, name=None))]
    /// Override capacity telemetry for a device resource.
    fn set_device_capacity(
        &self,
        id: &str,
        total_bytes: u64,
        available_bytes: Option<u64>,
        name: Option<&str>,
    ) -> PyResult<()> {
        if total_bytes == 0 {
            return Err(PyValueError::new_err(
                "total_bytes must be greater than zero",
            ));
        }
        self.inner.register_device(
            MemoryResource::adaptive_device(id, name.unwrap_or(id))
                .with_capacity(total_bytes, available_bytes),
        );
        Ok(())
    }

    #[getter]
    fn host(&self) -> PyMemoryResource {
        PyMemoryResource {
            inner: self.inner.host(),
        }
    }

    #[getter]
    fn devices(&self) -> Vec<PyMemoryResource> {
        self.inner
            .devices()
            .into_iter()
            .map(|inner| PyMemoryResource { inner })
            .collect()
    }

    fn report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        memory_report_dict(py, &self.inner.report())
    }
}

fn memory_report_dict<'py>(
    py: Python<'py>,
    report: &laddu_runtime::MemoryReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let resources = report
        .resources
        .iter()
        .map(|resource| {
            let item = PyDict::new(py);
            item.set_item("id", &resource.resource.id)?;
            item.set_item("name", &resource.resource.name)?;
            item.set_item("total_bytes", resource.resource.total_bytes)?;
            item.set_item("available_bytes", resource.resource.available_bytes)?;
            item.set_item("reserved_bytes", resource.laddu_reserved_bytes)?;
            item.set_item("high_water_bytes", resource.laddu_high_water_bytes)?;
            item.set_item(
                "capacity_source",
                capacity_source_name(&resource.resource.capacity_source),
            )?;
            if let Some(identity) = &resource.resource.device_identity {
                item.set_item("adapter_index", identity.adapter_index)?;
                item.set_item("vendor_id", identity.vendor_id)?;
                item.set_item("device_id", identity.device_id)?;
                item.set_item("pci_bus_id", &identity.pci_bus_id)?;
            }
            Ok(item)
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("resources", resources)?;
    if let Some(process) = &report.process {
        let item = PyDict::new(py);
        item.set_item("resident_bytes", process.resident_bytes)?;
        item.set_item("virtual_bytes", process.virtual_bytes)?;
        item.set_item("sampled_high_water_bytes", process.sampled_high_water_bytes)?;
        out.set_item("process", item)?;
    } else {
        out.set_item("process", py.None())?;
    }
    Ok(out)
}

pub(crate) fn memory_decision_dict<'py>(
    py: Python<'py>,
    decision: &laddu_runtime::MemoryDecision,
) -> PyResult<Bound<'py, PyDict>> {
    let item = PyDict::new(py);
    item.set_item("label", &decision.label)?;
    item.set_item("fixed_bytes", decision.fixed_bytes)?;
    item.set_item("bytes_per_event", decision.bytes_per_event)?;
    item.set_item("chunk_events", decision.chunk_events)?;
    item.set_item("estimated_peak_bytes", decision.estimated_peak_bytes)?;
    item.set_item("actual_high_water_bytes", decision.actual_high_water_bytes)?;
    item.set_item("strategy", &decision.strategy)?;
    Ok(item)
}

#[pyclass(name = "Execution", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Runtime and distributed-execution configuration.
///
/// Parameters
/// ----------
/// backend : {'auto', 'cpu', 'jit', 'gpu'}, default='auto'
///     Evaluation backend. ``'auto'`` may enable JIT compilation when useful.
/// precision : {'auto', 'f32', 'f64'}, default='auto'
///     Numeric precision requested from the backend.
/// autodiff : {'auto', 'forward', 'reverse'}, default='auto'
///     Automatic-differentiation strategy.
/// normalization : {'auto', 'general', 'verify'}, default='auto'
///     Accepted-normalization strategy. ``'verify'`` compares compiler-native
///     statistics against the general event reduction.
/// threads : int, optional
///     CPU worker count. One selects serial execution.
/// device : int or str, optional
///     GPU adapter index, name, or PCI bus ID.
/// memory : MemoryPlan, MemoryBudget, int, or str, optional
///     Host and accelerator budgets. A single budget applies to both.
/// mpi : bool, optional
///     Enable MPI when the installed module has MPI support. ``False`` forces
///     local execution.
/// partitioning : {'auto', 'contiguous', 'file_groups', 'rows'}, default='auto'
///     Work distribution policy across MPI ranks.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> execution = ld.Execution("cpu", threads=1, precision="f64")
/// >>> execution.backend
/// 'cpu'
/// >>> execution.is_distributed
/// False
pub struct PyExecution {
    pub(crate) inner: Execution,
    backend: String,
}

impl PyExecution {
    /// Configure an evaluation runtime.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an option is invalid or does not apply to the selected backend.
    /// LadduError
    ///     If the requested local or distributed runtime cannot be initialized.
    pub(crate) fn default_inner() -> PyResult<Self> {
        Self::new(
            "auto", "auto", "auto", "auto", None, None, None, None, "auto",
        )
    }
}

#[pymethods]
impl PyExecution {
    #[new]
    #[pyo3(signature = (
        backend="auto",
        *,
        precision="auto",
        autodiff="auto",
        normalization="auto",
        threads=None,
        device: "int | str | None" = None,
        memory: "MemoryPlan | MemoryBudget | int | str | None" = None,
        mpi=None,
        partitioning="auto"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend: &str,
        precision: &str,
        autodiff: &str,
        normalization: &str,
        threads: Option<usize>,
        device: Option<&Bound<'_, PyAny>>,
        memory: Option<&Bound<'_, PyAny>>,
        mpi: Option<bool>,
        partitioning: &str,
    ) -> PyResult<Self> {
        let backend = normalize(backend);
        let memory = memory
            .map(parse_memory_plan)
            .transpose()?
            .unwrap_or_default();
        let thread_policy = match threads {
            None => ThreadPolicy::Auto,
            Some(0) => return Err(PyValueError::new_err("threads must be greater than zero")),
            Some(1) => ThreadPolicy::Serial,
            Some(count) => ThreadPolicy::Fixed(count),
        };
        let device_options = match backend.as_str() {
            "auto" => {
                if device.is_some() {
                    return Err(PyValueError::new_err(
                        "GPU device selection requires backend='gpu'",
                    ));
                }
                Device::Cpu(CpuOptions {
                    threads: thread_policy,
                    jit: JitPolicy::Auto,
                })
            }
            "cpu" => {
                if device.is_some() {
                    return Err(PyValueError::new_err(
                        "GPU device settings require backend='gpu'",
                    ));
                }
                Device::Cpu(CpuOptions {
                    threads: thread_policy,
                    jit: JitPolicy::Disabled,
                })
            }
            "jit" => {
                if device.is_some() {
                    return Err(PyValueError::new_err(
                        "GPU device settings require backend='gpu'",
                    ));
                }
                Device::Cpu(CpuOptions {
                    threads: thread_policy,
                    jit: JitPolicy::Enabled,
                })
            }
            "gpu" => {
                if threads.is_some() {
                    return Err(PyValueError::new_err(
                        "threads only applies to CPU and JIT execution",
                    ));
                }
                Device::Gpu(GpuOptions {
                    backend: GpuBackend::Wgpu,
                    device: gpu_selector(device)?,
                })
            }
            _ => {
                return Err(PyValueError::new_err(
                    "backend must be 'auto', 'cpu', 'jit', or 'gpu'",
                ));
            }
        };
        let options = ExecutionOptions {
            device: device_options,
            precision: parse_precision(precision)?,
            autodiff: parse_autodiff(autodiff)?,
            normalization: parse_normalization(normalization)?,
            partitioning: parse_partitioning(partitioning)?,
            memory,
        };

        #[cfg(feature = "mpi")]
        let execution = if mpi != Some(false) {
            let _ = mpi::initialize().map(|universe| Box::leak(Box::new(universe)));
            let world = mpi::topology::SimpleCommunicator::world();
            Execution::distributed(options, &world).map_err(to_py_err)?
        } else {
            Execution::local(options).map_err(to_py_err)?
        };

        #[cfg(not(feature = "mpi"))]
        let execution = {
            let _ = mpi;
            Execution::local(options).map_err(to_py_err)?
        };

        Ok(Self {
            inner: execution,
            backend,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Execution(backend={:?}, precision={:?}, autodiff={:?}, normalization={:?}, rank={}, world_size={})",
            self.backend,
            self.inner.precision(),
            self.inner.autodiff_mode(),
            self.inner.normalization_mode(),
            self.inner.rank(),
            self.inner.nranks(),
        )
    }

    #[getter]
    /// str: Requested backend name.
    fn backend(&self) -> &str {
        &self.backend
    }

    #[getter]
    /// {'auto', 'f32', 'f64'}: Effective numeric precision.
    fn precision(&self) -> &'static str {
        match self.inner.precision() {
            Precision::Auto => "auto",
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }

    #[getter]
    /// {'auto', 'forward', 'reverse'}: Automatic-differentiation mode.
    fn autodiff(&self) -> &'static str {
        match self.inner.autodiff_mode() {
            AutodiffMode::Auto => "auto",
            AutodiffMode::Forward => "forward",
            AutodiffMode::Reverse => "reverse",
        }
    }

    #[getter]
    /// {'auto', 'general', 'verify'}: Accepted-normalization mode.
    fn normalization(&self) -> &'static str {
        match self.inner.normalization_mode() {
            NormalizationMode::Auto => "auto",
            NormalizationMode::General => "general",
            NormalizationMode::Verify => "verify",
        }
    }

    #[getter]
    /// int: Zero-based rank in the execution world.
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    /// int: Number of ranks participating in execution.
    fn world_size(&self) -> usize {
        self.inner.nranks()
    }

    #[getter]
    /// bool: Whether more than one rank participates in execution.
    fn is_distributed(&self) -> bool {
        self.inner.is_distributed()
    }

    /// Return physical-resource memory information and laddu high-water usage.
    fn memory_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = memory_report_dict(py, &self.inner.memory_report())?;
        let pools = self
            .inner
            .memory_pool_reports()
            .into_iter()
            .map(|pool| {
                let item = PyDict::new(py);
                item.set_item("resource_id", pool.resource_id)?;
                item.set_item("requested", pool.requested.to_string())?;
                item.set_item("effective_bytes", pool.effective_bytes)?;
                item.set_item("reserved_bytes", pool.reserved_bytes)?;
                item.set_item("remaining_bytes", pool.remaining_bytes)?;
                item.set_item("high_water_bytes", pool.high_water_bytes)?;
                Ok(item)
            })
            .collect::<PyResult<Vec<_>>>()?;
        report.set_item("pools", pools)?;
        Ok(report)
    }

    /// Return memory-derived chunk and strategy decisions.
    fn memory_decisions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .memory_decisions()
            .into_iter()
            .map(|decision| memory_decision_dict(py, &decision))
            .collect()
    }
}

#[pyfunction]
/// Report features available in the installed module.
///
/// Returns
/// -------
/// dict[str, bool or str]
///     Backend kind and availability flags for JIT, GPU, MPI, likelihood,
///     fitting, and generation support.
pub fn capabilities(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "backend",
        if cfg!(feature = "mpi") {
            "mpi"
        } else {
            "local"
        },
    )?;
    out.set_item("jit", cfg!(feature = "jit"))?;
    out.set_item("gpu", cfg!(feature = "wgpu"))?;
    out.set_item("mpi", cfg!(feature = "mpi"))?;
    out.set_item("likelihood", cfg!(feature = "likelihood"))?;
    out.set_item("fit", cfg!(feature = "fit"))?;
    out.set_item("generation", cfg!(feature = "generation"))?;
    Ok(out)
}

#[pyclass(name = "Device", module = "laddu.gpu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Description and limits of an available GPU adapter.
///
/// Instances are returned by :func:`laddu.gpu.devices` and cannot be created
/// directly.
///
/// Attributes
/// ----------
/// index : int
///     Adapter index accepted by :class:`Execution`.
/// name : str
///     Human-readable adapter name.
/// vendor, device : int
///     PCI vendor and device identifiers.
/// device_type : str
///     Adapter category reported by the graphics API.
/// pci_bus_id : str
///     Stable PCI selector when the platform provides one.
/// driver, driver_info, backend : str
///     Driver and graphics-backend metadata.
/// supports_f64 : bool
///     Whether double-precision shader arithmetic is supported.
/// max_buffer_size : int
///     Maximum buffer allocation in bytes.
/// max_storage_buffer_binding_size : int
///     Maximum bound storage-buffer size in bytes.
/// max_compute_workgroup_size_x : int
///     Maximum workgroup width along the x axis.
pub struct PyGpuDevice {
    index: usize,
    name: String,
    vendor: u32,
    device: u32,
    device_type: String,
    pci_bus_id: String,
    driver: String,
    driver_info: String,
    backend: String,
    supports_f64: bool,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
    max_compute_workgroup_size_x: u32,
}

#[pymethods]
impl PyGpuDevice {
    fn __repr__(&self) -> String {
        format!(
            "Device(index={}, name={:?}, backend={:?})",
            self.index, self.name, self.backend
        )
    }

    #[getter]
    /// int: Adapter index accepted by ``Execution(device=...)``.
    fn index(&self) -> usize {
        self.index
    }
    #[getter]
    /// str: Human-readable adapter name.
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    /// int: PCI vendor identifier.
    fn vendor(&self) -> u32 {
        self.vendor
    }
    #[getter]
    /// int: PCI device identifier.
    fn device(&self) -> u32 {
        self.device
    }
    #[getter]
    /// str: Hardware category reported by the graphics API.
    fn device_type(&self) -> &str {
        &self.device_type
    }
    #[getter]
    /// str: PCI bus identifier, if available.
    fn pci_bus_id(&self) -> &str {
        &self.pci_bus_id
    }
    #[getter]
    /// str: Driver name.
    fn driver(&self) -> &str {
        &self.driver
    }
    #[getter]
    /// str: Additional driver version or description.
    fn driver_info(&self) -> &str {
        &self.driver_info
    }
    #[getter]
    /// str: Graphics API used by the adapter.
    fn backend(&self) -> &str {
        &self.backend
    }
    #[getter]
    /// bool: Whether the adapter supports 64-bit floating-point shaders.
    fn supports_f64(&self) -> bool {
        self.supports_f64
    }
    #[getter]
    /// int: Maximum buffer allocation in bytes.
    fn max_buffer_size(&self) -> u64 {
        self.max_buffer_size
    }
    #[getter]
    /// int: Maximum storage-buffer binding size in bytes.
    fn max_storage_buffer_binding_size(&self) -> u64 {
        self.max_storage_buffer_binding_size
    }
    #[getter]
    /// int: Maximum compute workgroup width along the x axis.
    fn max_compute_workgroup_size_x(&self) -> u32 {
        self.max_compute_workgroup_size_x
    }
}

#[pymodule(submodule)]
/// GPU discovery and adapter metadata.
pub mod gpu {
    use super::*;

    #[pymodule_export]
    use super::PyGpuDevice as Device;

    #[pyfunction]
    /// List GPU adapters visible to the installed graphics backend.
    ///
    /// Returns
    /// -------
    /// list of Device
    ///     Available adapters, or an empty list when GPU support is unavailable.
    fn devices() -> Vec<PyGpuDevice> {
        #[cfg(feature = "wgpu")]
        {
            laddu_wgpu::WgpuBackend::default()
                .adapters()
                .into_iter()
                .map(|info| PyGpuDevice {
                    index: info.index,
                    name: info.name,
                    vendor: info.vendor,
                    device: info.device,
                    device_type: info.device_type,
                    pci_bus_id: info.pci_bus_id,
                    driver: info.driver,
                    driver_info: info.driver_info,
                    backend: info.backend,
                    supports_f64: info.supports_f64,
                    max_buffer_size: info.max_buffer_size,
                    max_storage_buffer_binding_size: info.max_storage_buffer_binding_size,
                    max_compute_workgroup_size_x: info.max_compute_workgroup_size_x,
                })
                .collect()
        }
        #[cfg(not(feature = "wgpu"))]
        Vec::new()
    }

    #[pyfunction]
    /// Return whether at least one compatible GPU adapter is available.
    ///
    /// Returns
    /// -------
    /// bool
    ///     ``True`` when :func:`devices` is non-empty.
    fn is_available() -> bool {
        !devices().is_empty()
    }
}

impl_json_methods!(PyMemoryBudget);
impl_json_methods!(PyMemoryPlan);
impl_json_methods!(PyMemoryResource);
