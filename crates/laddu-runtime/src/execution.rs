use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

use laddu_autodiff::AutodiffMode;
use laddu_data::io::{Partitioning, ReadPlan};
#[cfg(feature = "wgpu")]
use laddu_memory::DeviceIdentity;
use laddu_memory::{
    MemoryBudget, MemoryDecision, MemoryPlan, MemoryPool, MemoryPoolReport, MemoryReport,
    MemoryState,
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};

#[cfg(feature = "wgpu")]
use crate::RuntimeError;
use crate::{ExecutionError, RuntimeResult};

pub(crate) type NormalizationCache =
    HashMap<(u64, u64, NormalizationMode), Weak<crate::PreparedNormalization>>;

#[cfg(feature = "mpi")]
use mpi::{
    collective::SystemOperation,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives},
};

/// Numeric precision used to execute a model.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    /// Select a precision appropriate for the resolved device.
    #[default]
    Auto,
    /// Use 32-bit floating-point arithmetic.
    F32,
    /// Use 64-bit floating-point arithmetic.
    F64,
}

/// Policy controlling CPU worker-thread use.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadPolicy {
    /// Use the runtime's default parallelism.
    #[default]
    Auto,
    /// Execute serially on the calling thread.
    Serial,
    /// Use exactly the given number of worker threads.
    Fixed(usize),
}

/// Policy controlling CPU just-in-time compilation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum JitPolicy {
    /// Use JIT compilation when it is available and applicable.
    #[default]
    Auto,
    /// Require the JIT-capable execution path.
    Enabled,
    /// Always use the interpreter.
    Disabled,
}

/// Policy controlling compiler-native accepted normalization.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationMode {
    /// Select exact sufficient statistics when cost and memory estimates are favorable.
    #[default]
    Auto,
    /// Always use the ordinary accepted-event reduction path.
    General,
    /// Evaluate both selected and general paths and reject disagreements.
    Verify,
}

/// CPU-specific execution options.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuOptions {
    /// Worker-thread policy.
    pub threads: ThreadPolicy,
    /// JIT compilation policy.
    pub jit: JitPolicy,
}

/// GPU implementation to use.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// Select an available GPU backend automatically.
    #[default]
    Auto,
    /// Use the WebGPU backend.
    Wgpu,
    /// Use the CUDA backend.
    Cuda,
}

/// Rule used to select a GPU device.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDeviceSelector {
    /// Select a suitable device automatically.
    #[default]
    Auto,
    /// Select the adapter at the given enumeration index.
    Index(usize),
    /// Select the adapter with the given PCI bus identifier.
    PciBusId(String),
    /// Select an adapter whose name matches the given string.
    Name(String),
}

/// GPU-specific execution options.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuOptions {
    /// GPU backend to use.
    pub backend: GpuBackend,
    /// GPU device selection rule.
    pub device: GpuDeviceSelector,
}

/// Device on which a model should execute.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// Select a suitable device automatically.
    #[default]
    Auto,
    /// Execute on a CPU with the supplied options.
    Cpu(CpuOptions),
    /// Execute on a GPU with the supplied options.
    Gpu(GpuOptions),
}

/// Options used to construct an [`Execution`] context.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionOptions {
    /// Requested execution device.
    pub device: Device,
    /// Requested numeric precision.
    pub precision: Precision,
    /// Automatic-differentiation strategy.
    pub autodiff: AutodiffMode,
    /// Accepted-normalization selection and verification policy.
    #[serde(default)]
    pub normalization: NormalizationMode,
    /// Dataset partitioning strategy for distributed execution.
    pub partitioning: Partitioning,
    /// Host and accelerator memory budgets.
    pub memory: MemoryPlan,
}

/// Resolved resources and policies used to execute models.
#[derive(Clone)]
pub struct Execution {
    requested_device: Device,
    precision: Precision,
    autodiff: AutodiffMode,
    normalization: NormalizationMode,
    threads: ThreadPolicy,
    jit: JitPolicy,
    pool: Option<Arc<ThreadPool>>,
    partitioning: Partitioning,
    memory_state: MemoryState,
    host_memory: MemoryPool,
    device_memory: Option<MemoryPool>,
    memory_decisions: Arc<Mutex<Vec<MemoryDecision>>>,
    normalization_cache: Arc<Mutex<NormalizationCache>>,
    #[cfg(feature = "wgpu")]
    wgpu: Option<Arc<laddu_wgpu::WgpuContext>>,
    #[cfg(feature = "mpi")]
    communicator: Option<Arc<SimpleCommunicator>>,
}

impl std::fmt::Debug for Execution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(feature = "wgpu")]
        let resolved_device = if self.wgpu.is_some() { "wgpu" } else { "cpu" };
        #[cfg(not(feature = "wgpu"))]
        let resolved_device = "cpu";
        formatter
            .debug_struct("Execution")
            .field("requested_device", &self.requested_device)
            .field("resolved_device", &resolved_device)
            .field("precision", &self.precision)
            .field("autodiff", &self.autodiff)
            .field("normalization", &self.normalization)
            .field("threads", &self.threads)
            .field("jit", &self.jit)
            .field("partitioning", &self.partitioning)
            .field("host_memory", &self.host_memory.report())
            .field(
                "device_memory",
                &self.device_memory.as_ref().map(MemoryPool::report),
            )
            .field("ranks", &self.nranks())
            .finish_non_exhaustive()
    }
}

struct ResolvedCpu {
    threads: ThreadPolicy,
    jit: JitPolicy,
    pool: Option<Arc<ThreadPool>>,
}

struct ResolvedHost {
    state: MemoryState,
    pool: MemoryPool,
}

fn resolve_host_memory(budget: MemoryBudget) -> RuntimeResult<ResolvedHost> {
    let state = MemoryState::current();
    state.refresh();
    let pool = state.pool("host", budget)?;
    Ok(ResolvedHost { state, pool })
}

fn resolve_precision(requested: Precision, gpu_requested: bool) -> Precision {
    match requested {
        Precision::Auto if gpu_requested => Precision::F32,
        Precision::Auto => Precision::F64,
        precision => precision,
    }
}

fn resolve_cpu(options: CpuOptions) -> RuntimeResult<ResolvedCpu> {
    #[cfg(not(feature = "jit"))]
    if options.jit == JitPolicy::Enabled {
        return Err(ExecutionError::JitUnavailable.into());
    }
    let pool = match options.threads {
        ThreadPolicy::Fixed(0) => return Err(ExecutionError::ZeroThreads.into()),
        ThreadPolicy::Fixed(threads) => Some(Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| ExecutionError::ThreadPool(error.to_string()))?,
        )),
        ThreadPolicy::Auto | ThreadPolicy::Serial => None,
    };
    Ok(ResolvedCpu {
        threads: options.threads,
        jit: options.jit,
        pool,
    })
}

#[cfg(feature = "wgpu")]
struct ResolvedGpu {
    context: Arc<laddu_wgpu::WgpuContext>,
    memory: MemoryPool,
}

#[cfg(feature = "wgpu")]
fn resolve_gpu(
    memory_state: &MemoryState,
    options: &GpuOptions,
    requested_precision: Precision,
    requested_memory: Option<MemoryBudget>,
) -> RuntimeResult<ResolvedGpu> {
    if options.backend == GpuBackend::Cuda {
        return Err(ExecutionError::GpuUnavailable(options.backend).into());
    }
    let selector = match &options.device {
        GpuDeviceSelector::Auto => laddu_wgpu::WgpuDeviceSelector::Auto,
        GpuDeviceSelector::Index(index) => laddu_wgpu::WgpuDeviceSelector::Index(*index),
        GpuDeviceSelector::PciBusId(id) => laddu_wgpu::WgpuDeviceSelector::PciBusId(id.clone()),
        GpuDeviceSelector::Name(name) => laddu_wgpu::WgpuDeviceSelector::Name(name.clone()),
    };
    let precision = match requested_precision {
        Precision::Auto => laddu_wgpu::WgpuPrecision::Auto,
        Precision::F32 => laddu_wgpu::WgpuPrecision::F32,
        Precision::F64 => laddu_wgpu::WgpuPrecision::F64,
    };
    let mut context = laddu_wgpu::WgpuBackend::default()
        .open(
            &laddu_wgpu::WgpuOptions {
                device: selector,
                memory_budget: None,
            },
            precision,
        )
        .map_err(|error| RuntimeError::Wgpu(error.to_string()))?;
    let resource_id = if context.info().pci_bus_id.is_empty() {
        format!("wgpu:{}", context.info().index)
    } else {
        format!("pci:{}", context.info().pci_bus_id)
    };
    let fallback = context
        .info()
        .max_buffer_size
        .min(512 * 1024 * 1024)
        .max(context.info().max_storage_buffer_binding_size);
    memory_state.register_discovered_device(
        resource_id.clone(),
        context.info().name.clone(),
        DeviceIdentity {
            adapter_index: context.info().index,
            vendor_id: context.info().vendor,
            device_id: context.info().device,
            pci_bus_id: context.info().pci_bus_id.clone(),
        },
        fallback,
    );
    let memory = memory_state.pool(&resource_id, requested_memory.unwrap_or(MemoryBudget::Auto))?;
    context.set_memory_budget(usize::try_from(memory.capacity()).unwrap_or(usize::MAX));
    Ok(ResolvedGpu {
        context: Arc::new(context),
        memory,
    })
}

impl Default for Execution {
    fn default() -> Self {
        let resolved_host = resolve_host_memory(MemoryBudget::Auto)
            .expect("host memory discovery must resolve an automatic budget");
        Self {
            requested_device: Device::Auto,
            precision: Precision::F64,
            autodiff: AutodiffMode::Auto,
            normalization: NormalizationMode::Auto,
            threads: ThreadPolicy::Auto,
            jit: JitPolicy::Auto,
            pool: None,
            partitioning: Partitioning::default(),
            memory_state: resolved_host.state,
            host_memory: resolved_host.pool,
            device_memory: None,
            memory_decisions: Default::default(),
            normalization_cache: Default::default(),
            #[cfg(feature = "wgpu")]
            wgpu: None,
            #[cfg(feature = "mpi")]
            communicator: None,
        }
    }
}

impl Execution {
    /// Creates a non-distributed execution context from `options`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the requested backend or precision is
    /// unavailable, GPU initialization fails, or the CPU thread pool cannot be
    /// created.
    pub fn local(options: ExecutionOptions) -> RuntimeResult<Self> {
        let resolved_host = resolve_host_memory(options.memory.host)?;
        let memory_state = resolved_host.state;
        let host_memory = resolved_host.pool;
        #[cfg(feature = "wgpu")]
        let resolved_gpu = match &options.device {
            Device::Gpu(gpu_options) => Some(resolve_gpu(
                &memory_state,
                gpu_options,
                options.precision,
                options.memory.device,
            )?),
            _ => None,
        };
        #[cfg(feature = "wgpu")]
        let wgpu = resolved_gpu.as_ref().map(|gpu| Arc::clone(&gpu.context));
        #[cfg(feature = "wgpu")]
        let device_memory = resolved_gpu.map(|gpu| gpu.memory);
        #[cfg(not(feature = "wgpu"))]
        let device_memory = None;
        #[cfg(not(feature = "wgpu"))]
        if let Device::Gpu(gpu_options) = &options.device {
            return Err(ExecutionError::GpuUnavailable(gpu_options.backend).into());
        }
        let cpu_options = match &options.device {
            Device::Cpu(options) => options.clone(),
            Device::Auto | Device::Gpu(_) => CpuOptions::default(),
        };
        let cpu = resolve_cpu(cpu_options)?;
        let precision =
            resolve_precision(options.precision, matches!(options.device, Device::Gpu(_)));
        Ok(Self {
            requested_device: options.device,
            precision,
            autodiff: options.autodiff,
            normalization: options.normalization,
            threads: cpu.threads,
            jit: cpu.jit,
            pool: cpu.pool,
            partitioning: options.partitioning,
            memory_state,
            host_memory,
            device_memory,
            memory_decisions: Default::default(),
            normalization_cache: Default::default(),
            #[cfg(feature = "wgpu")]
            wgpu,
            #[cfg(feature = "mpi")]
            communicator: None,
        })
    }

    #[cfg(feature = "mpi")]
    /// Creates a distributed execution context over the supplied MPI communicator.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when local execution initialization fails.
    pub fn distributed<C>(options: ExecutionOptions, world: &C) -> RuntimeResult<Self>
    where
        C: Communicator,
    {
        let local_processes = mpi_local_process_count(world.size());
        let mut options = options;
        options.memory.host = shared_mpi_budget(options.memory.host, local_processes);
        options.memory.device = options
            .memory
            .device
            .map(|budget| shared_mpi_budget(budget, local_processes));
        let mut execution = Self::local(options)?;
        execution.record_memory_decision(MemoryDecision {
            label: "mpi-memory-share".into(),
            fixed_bytes: 0,
            bytes_per_event: 0,
            chunk_events: 0,
            estimated_peak_bytes: 0,
            actual_high_water_bytes: None,
            strategy: format!("equal-share-across-{local_processes}-local-ranks"),
        });
        execution.communicator = Some(Arc::new(world.duplicate()));
        Ok(execution)
    }

    /// Returns the device requested when this context was created.
    pub fn requested_device(&self) -> &Device {
        &self.requested_device
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn wgpu_context(&self) -> Option<&Arc<laddu_wgpu::WgpuContext>> {
        self.wgpu.as_ref()
    }

    /// Returns the resolved numeric precision.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Returns the automatic-differentiation strategy.
    pub fn autodiff_mode(&self) -> AutodiffMode {
        self.autodiff
    }

    /// Returns the accepted-normalization policy.
    pub fn normalization_mode(&self) -> NormalizationMode {
        self.normalization
    }

    pub(crate) fn normalization_cache(&self) -> &Mutex<NormalizationCache> {
        &self.normalization_cache
    }

    /// Returns the CPU worker-thread policy.
    pub fn thread_policy(&self) -> ThreadPolicy {
        self.threads
    }

    /// Returns the CPU JIT policy.
    pub fn jit_policy(&self) -> JitPolicy {
        self.jit
    }

    /// Returns the distributed dataset-partitioning strategy.
    pub fn partitioning(&self) -> Partitioning {
        self.partitioning
    }

    /// Returns the live memory state shared by this execution.
    pub fn memory_state(&self) -> &MemoryState {
        &self.memory_state
    }

    /// Returns the resolved host-memory pool.
    pub fn host_memory(&self) -> &MemoryPool {
        &self.host_memory
    }

    /// Returns the resolved accelerator-memory pool, if any.
    pub fn device_memory(&self) -> Option<&MemoryPool> {
        self.device_memory.as_ref()
    }

    /// Returns current physical-resource memory information.
    pub fn memory_report(&self) -> MemoryReport {
        self.memory_state.report()
    }

    /// Returns the resolved execution-pool reports.
    pub fn memory_pool_reports(&self) -> Vec<MemoryPoolReport> {
        std::iter::once(self.host_memory.report())
            .chain(self.device_memory.as_ref().map(MemoryPool::report))
            .collect()
    }

    /// Returns memory-derived decisions recorded by this execution.
    pub fn memory_decisions(&self) -> Vec<MemoryDecision> {
        self.memory_decisions
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Records a memory-planning decision for later diagnostics.
    pub fn record_memory_decision(&self, decision: MemoryDecision) {
        self.memory_decisions
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(decision);
    }

    /// Returns the zero-based rank of this process.
    pub fn rank(&self) -> usize {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            return communicator.rank() as usize;
        }
        0
    }

    /// Returns the number of participating ranks.
    pub fn nranks(&self) -> usize {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            return communicator.size() as usize;
        }
        1
    }

    /// Returns whether this context spans more than one rank.
    pub fn is_distributed(&self) -> bool {
        self.nranks() > 1
    }

    #[allow(unused_mut)]
    pub(crate) fn read_plan(&self, mut plan: ReadPlan) -> ReadPlan {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            plan.distribution = laddu_data::io::Distribution::from_world(communicator.as_ref())
                .with_partitioning(self.partitioning);
        }
        plan
    }

    pub(crate) fn sum_f64(&self, local: f64) -> f64 {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            let mut global = 0.0;
            communicator.all_reduce_into(&local, &mut global, SystemOperation::sum());
            return global;
        }
        local
    }

    pub(crate) fn sum_usize(&self, local: usize) -> usize {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            let local = local as u64;
            let mut global = 0_u64;
            communicator.all_reduce_into(&local, &mut global, SystemOperation::sum());
            return global as usize;
        }
        local
    }

    pub(crate) fn sum_slice(&self, local: &[f64]) -> Vec<f64> {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            let mut global = vec![0.0; local.len()];
            communicator.all_reduce_into(local, &mut global, SystemOperation::sum());
            return global;
        }
        local.to_vec()
    }

    pub(crate) fn all_succeeded(&self, local_success: bool) -> bool {
        self.sum_usize(usize::from(local_success)) == self.nranks()
    }

    pub(crate) fn is_parallel(&self) -> bool {
        self.threads != ThreadPolicy::Serial
    }

    pub(crate) fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(operation),
            None => operation(),
        }
    }
}

#[cfg(feature = "mpi")]
fn shared_mpi_budget(budget: MemoryBudget, local_processes: u64) -> MemoryBudget {
    let divisor = local_processes.max(1);
    match budget {
        MemoryBudget::Auto => MemoryBudget::PercentAvailable(0.80 / divisor as f64),
        MemoryBudget::Bytes(bytes) => MemoryBudget::Bytes((bytes / divisor).max(1)),
        MemoryBudget::PercentTotal(fraction) => {
            MemoryBudget::PercentTotal(fraction / divisor as f64)
        }
        MemoryBudget::PercentAvailable(fraction) => {
            MemoryBudget::PercentAvailable(fraction / divisor as f64)
        }
    }
}

#[cfg(feature = "mpi")]
fn mpi_local_process_count(world_size: i32) -> u64 {
    // Common launchers expose node-local process counts. Falling back to the
    // world size is conservative on multi-node jobs and prevents accidental
    // host/device overcommit when launcher metadata is unavailable.
    const VARIABLES: [&str; 4] = [
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
        "MV2_COMM_WORLD_LOCAL_SIZE",
        "SLURM_NTASKS_PER_NODE",
    ];
    VARIABLES
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .find_map(|value| {
            value
                .split(|character: char| !character.is_ascii_digit())
                .find(|part| !part.is_empty())
                .and_then(|part| part.parse::<u64>().ok())
                .filter(|count| *count > 0)
        })
        .unwrap_or_else(|| u64::try_from(world_size).unwrap_or(1).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeError;
    use crate::execution::GpuBackend;

    #[test]
    fn execution_options_roundtrip_through_json() {
        let options = ExecutionOptions {
            device: Device::Gpu(GpuOptions {
                backend: GpuBackend::Wgpu,
                device: GpuDeviceSelector::PciBusId("0000:01:00.0".into()),
            }),
            precision: Precision::F64,
            autodiff: AutodiffMode::Reverse,
            normalization: NormalizationMode::Verify,
            partitioning: Partitioning::FileGroups,
            memory: MemoryPlan::host_device(
                MemoryBudget::PercentAvailable(0.5),
                MemoryBudget::Bytes(1 << 30),
            ),
        };

        let json = serde_json::to_string(&options).unwrap();
        assert_eq!(
            serde_json::from_str::<ExecutionOptions>(&json).unwrap(),
            options
        );
    }

    #[test]
    fn execution_selects_nested_cpu_options() {
        let serial = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions {
                threads: ThreadPolicy::Serial,
                jit: JitPolicy::Disabled,
            }),
            ..ExecutionOptions::default()
        })
        .unwrap();
        assert!(!serial.is_parallel());
        assert_eq!(serial.jit_policy(), JitPolicy::Disabled);
        assert_eq!(serial.precision(), Precision::F64);

        let fixed = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions {
                threads: ThreadPolicy::Fixed(2),
                ..CpuOptions::default()
            }),
            ..ExecutionOptions::default()
        })
        .unwrap();
        assert_eq!(fixed.install(rayon::current_num_threads), 2);
    }

    #[test]
    fn unavailable_execution_modes_return_capability_errors() {
        #[cfg(not(feature = "wgpu"))]
        assert!(matches!(
            Execution::local(ExecutionOptions {
                device: Device::Gpu(GpuOptions {
                    backend: GpuBackend::Wgpu,
                    ..GpuOptions::default()
                }),
                ..ExecutionOptions::default()
            }),
            Err(RuntimeError::Execution(ExecutionError::GpuUnavailable(
                GpuBackend::Wgpu
            )))
        ));
        #[cfg(feature = "wgpu")]
        assert!(
            Execution::local(ExecutionOptions {
                device: Device::Gpu(GpuOptions {
                    backend: GpuBackend::Wgpu,
                    ..GpuOptions::default()
                }),
                ..ExecutionOptions::default()
            })
            .is_ok()
        );
        let f32 = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap();
        assert_eq!(f32.precision(), Precision::F32);

        let reverse = Execution::local(ExecutionOptions {
            autodiff: AutodiffMode::Reverse,
            ..ExecutionOptions::default()
        })
        .unwrap();
        assert_eq!(reverse.autodiff_mode(), AutodiffMode::Reverse);

        let reverse_f32 = Execution::local(ExecutionOptions {
            precision: Precision::F32,
            autodiff: AutodiffMode::Reverse,
            ..ExecutionOptions::default()
        })
        .unwrap();
        assert_eq!(reverse_f32.precision(), Precision::F32);
        assert_eq!(reverse_f32.autodiff_mode(), AutodiffMode::Reverse);
    }

    #[test]
    fn focused_resource_resolution_covers_cpu_policy_matrix() {
        let cases = [
            (ThreadPolicy::Auto, JitPolicy::Disabled, false),
            (ThreadPolicy::Serial, JitPolicy::Auto, false),
            (ThreadPolicy::Fixed(2), JitPolicy::Disabled, true),
        ];
        for (threads, jit, has_pool) in cases {
            let resolved = resolve_cpu(CpuOptions { threads, jit }).unwrap();
            assert_eq!(resolved.threads, threads);
            assert_eq!(resolved.jit, jit);
            assert_eq!(resolved.pool.is_some(), has_pool);
        }

        assert!(matches!(
            resolve_cpu(CpuOptions {
                threads: ThreadPolicy::Fixed(0),
                jit: JitPolicy::Disabled,
            }),
            Err(RuntimeError::Execution(ExecutionError::ZeroThreads))
        ));

        #[cfg(not(feature = "jit"))]
        assert!(matches!(
            resolve_cpu(CpuOptions {
                threads: ThreadPolicy::Auto,
                jit: JitPolicy::Enabled,
            }),
            Err(RuntimeError::Execution(ExecutionError::JitUnavailable))
        ));
        #[cfg(feature = "jit")]
        assert!(
            resolve_cpu(CpuOptions {
                threads: ThreadPolicy::Auto,
                jit: JitPolicy::Enabled,
            })
            .is_ok()
        );
    }

    #[test]
    fn focused_precision_resolution_uses_device_defaults() {
        let cases = [
            (Precision::Auto, false, Precision::F64),
            (Precision::Auto, true, Precision::F32),
            (Precision::F32, false, Precision::F32),
            (Precision::F64, true, Precision::F64),
        ];
        for (requested, gpu_requested, expected) in cases {
            assert_eq!(resolve_precision(requested, gpu_requested), expected);
        }
    }
}
