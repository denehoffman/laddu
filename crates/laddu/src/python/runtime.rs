use laddu_autodiff::AutodiffMode;
use laddu_data::io::Partitioning;
use laddu_runtime::{
    CpuOptions, Device, Execution, ExecutionOptions, GpuBackend, GpuDeviceSelector, GpuOptions,
    JitPolicy, Precision, ThreadPolicy,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};

use super::error::to_py_err;

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
        "forward" => Ok(AutodiffMode::Forward),
        "reverse" => Ok(AutodiffMode::Reverse),
        _ => Err(PyValueError::new_err(
            "autodiff must be 'forward' or 'reverse'",
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
/// autodiff : {'forward', 'reverse'}, default='forward'
///     Automatic-differentiation strategy.
/// threads : int, optional
///     CPU worker count. One selects serial execution.
/// device : int or str, optional
///     GPU adapter index, name, or PCI bus ID.
/// memory_budget : int, optional
///     GPU allocation budget in bytes.
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
        Self::new("auto", "auto", "forward", None, None, None, None, "auto")
    }
}

#[pymethods]
impl PyExecution {
    #[new]
    #[pyo3(signature = (backend="auto", *, precision="auto", autodiff="forward", threads=None, device=None, memory_budget=None, mpi=None, partitioning="auto"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend: &str,
        precision: &str,
        autodiff: &str,
        threads: Option<usize>,
        device: Option<&Bound<'_, PyAny>>,
        memory_budget: Option<usize>,
        mpi: Option<bool>,
        partitioning: &str,
    ) -> PyResult<Self> {
        let backend = normalize(backend);
        let thread_policy = match threads {
            None => ThreadPolicy::Auto,
            Some(0) => return Err(PyValueError::new_err("threads must be greater than zero")),
            Some(1) => ThreadPolicy::Serial,
            Some(count) => ThreadPolicy::Fixed(count),
        };
        let device_options = match backend.as_str() {
            "auto" => {
                if device.is_some() || memory_budget.is_some() {
                    return Err(PyValueError::new_err(
                        "GPU device settings require backend='gpu'",
                    ));
                }
                Device::Cpu(CpuOptions {
                    threads: thread_policy,
                    jit: JitPolicy::Auto,
                })
            }
            "cpu" => {
                if device.is_some() || memory_budget.is_some() {
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
                if device.is_some() || memory_budget.is_some() {
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
                    memory_budget,
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
            partitioning: parse_partitioning(partitioning)?,
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
            "Execution(backend={:?}, precision={:?}, autodiff={:?}, rank={}, world_size={})",
            self.backend,
            self.inner.precision(),
            self.inner.autodiff_mode(),
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
    /// {'forward', 'reverse'}: Automatic-differentiation mode.
    fn autodiff(&self) -> &'static str {
        match self.inner.autodiff_mode() {
            AutodiffMode::Forward => "forward",
            AutodiffMode::Reverse => "reverse",
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
