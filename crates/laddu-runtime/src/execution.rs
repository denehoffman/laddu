use std::sync::Arc;

use laddu_autodiff::AutodiffMode;
use laddu_data::io::{Partitioning, ReadPlan};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};

#[cfg(feature = "wgpu")]
use crate::RuntimeError;
use crate::{ExecutionError, RuntimeResult};

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
    /// Optional upper bound, in bytes, for resident GPU allocations.
    pub memory_budget: Option<usize>,
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionOptions {
    /// Requested execution device.
    pub device: Device,
    /// Requested numeric precision.
    pub precision: Precision,
    /// Automatic-differentiation strategy.
    pub autodiff: AutodiffMode,
    /// Dataset partitioning strategy for distributed execution.
    pub partitioning: Partitioning,
}

/// Resolved resources and policies used to execute models.
#[derive(Clone)]
pub struct Execution {
    requested_device: Device,
    precision: Precision,
    autodiff: AutodiffMode,
    threads: ThreadPolicy,
    jit: JitPolicy,
    pool: Option<Arc<ThreadPool>>,
    partitioning: Partitioning,
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
            .field("threads", &self.threads)
            .field("jit", &self.jit)
            .field("partitioning", &self.partitioning)
            .field("ranks", &self.nranks())
            .finish_non_exhaustive()
    }
}

impl Default for Execution {
    fn default() -> Self {
        Self {
            requested_device: Device::Auto,
            precision: Precision::F64,
            autodiff: AutodiffMode::Forward,
            threads: ThreadPolicy::Auto,
            jit: JitPolicy::Auto,
            pool: None,
            partitioning: Partitioning::default(),
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
        #[cfg(feature = "wgpu")]
        let mut wgpu = None;
        let cpu = match &options.device {
            Device::Auto => CpuOptions::default(),
            Device::Cpu(options) => options.clone(),
            Device::Gpu(gpu_options) => {
                #[cfg(feature = "wgpu")]
                {
                    if gpu_options.backend == GpuBackend::Cuda {
                        return Err(ExecutionError::GpuUnavailable(gpu_options.backend).into());
                    }
                    let selector = match &gpu_options.device {
                        GpuDeviceSelector::Auto => laddu_wgpu::WgpuDeviceSelector::Auto,
                        GpuDeviceSelector::Index(index) => {
                            laddu_wgpu::WgpuDeviceSelector::Index(*index)
                        }
                        GpuDeviceSelector::PciBusId(id) => {
                            laddu_wgpu::WgpuDeviceSelector::PciBusId(id.clone())
                        }
                        GpuDeviceSelector::Name(name) => {
                            laddu_wgpu::WgpuDeviceSelector::Name(name.clone())
                        }
                    };
                    let precision = match options.precision {
                        Precision::Auto => laddu_wgpu::WgpuPrecision::Auto,
                        Precision::F32 => laddu_wgpu::WgpuPrecision::F32,
                        Precision::F64 => laddu_wgpu::WgpuPrecision::F64,
                    };
                    let context = laddu_wgpu::WgpuBackend::default()
                        .open(
                            &laddu_wgpu::WgpuOptions {
                                device: selector,
                                memory_budget: gpu_options.memory_budget,
                            },
                            precision,
                        )
                        .map_err(|error| RuntimeError::Wgpu(error.to_string()))?;
                    wgpu = Some(Arc::new(context));
                    CpuOptions::default()
                }
                #[cfg(not(feature = "wgpu"))]
                return Err(ExecutionError::GpuUnavailable(gpu_options.backend).into());
            }
        };
        let precision = match options.precision {
            Precision::Auto if matches!(options.device, Device::Gpu(_)) => Precision::F32,
            Precision::Auto => Precision::F64,
            precision => precision,
        };
        #[cfg(not(feature = "jit"))]
        if cpu.jit == JitPolicy::Enabled {
            return Err(ExecutionError::JitUnavailable.into());
        }
        let pool = match cpu.threads {
            ThreadPolicy::Fixed(0) => return Err(ExecutionError::ZeroThreads.into()),
            ThreadPolicy::Fixed(threads) => Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|error| ExecutionError::ThreadPool(error.to_string()))?,
            )),
            ThreadPolicy::Auto | ThreadPolicy::Serial => None,
        };
        Ok(Self {
            requested_device: options.device,
            precision,
            autodiff: options.autodiff,
            threads: cpu.threads,
            jit: cpu.jit,
            pool,
            partitioning: options.partitioning,
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
        let mut execution = Self::local(options)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "wgpu"))]
    use crate::RuntimeError;
    use crate::execution::GpuBackend;

    #[test]
    fn execution_options_roundtrip_through_json() {
        let options = ExecutionOptions {
            device: Device::Gpu(GpuOptions {
                backend: GpuBackend::Wgpu,
                device: GpuDeviceSelector::PciBusId("0000:01:00.0".into()),
                memory_budget: Some(1 << 30),
            }),
            precision: Precision::F64,
            autodiff: AutodiffMode::Reverse,
            partitioning: Partitioning::FileGroups,
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
}
