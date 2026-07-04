use std::sync::Arc;

use laddu_data::io::{Partitioning, ReadPlan};
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::{ExecutionError, RuntimeResult};

#[cfg(feature = "mpi")]
use mpi::{
    collective::SystemOperation,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives},
};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    Auto,
    F32,
    F64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ThreadPolicy {
    #[default]
    Auto,
    Serial,
    Fixed(usize),
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum JitPolicy {
    #[default]
    Auto,
    Enabled,
    Disabled,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuOptions {
    pub threads: ThreadPolicy,
    pub jit: JitPolicy,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum GpuBackend {
    #[default]
    Auto,
    Wgpu,
    Cuda,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum GpuDeviceSelector {
    #[default]
    Auto,
    Index(usize),
    PciBusId(String),
    Name(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuOptions {
    pub backend: GpuBackend,
    pub device: GpuDeviceSelector,
    pub memory_budget: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Auto,
    Cpu(CpuOptions),
    Gpu(GpuOptions),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionOptions {
    pub device: Device,
    pub precision: Precision,
    pub partitioning: Partitioning,
}

#[derive(Clone)]
pub struct Execution {
    requested_device: Device,
    precision: Precision,
    threads: ThreadPolicy,
    jit: JitPolicy,
    pool: Option<Arc<ThreadPool>>,
    partitioning: Partitioning,
    #[cfg(feature = "mpi")]
    communicator: Option<Arc<SimpleCommunicator>>,
}

impl std::fmt::Debug for Execution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Execution")
            .field("requested_device", &self.requested_device)
            .field("resolved_device", &"cpu")
            .field("precision", &self.precision)
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
            threads: ThreadPolicy::Auto,
            jit: JitPolicy::Auto,
            pool: None,
            partitioning: Partitioning::default(),
            #[cfg(feature = "mpi")]
            communicator: None,
        }
    }
}

impl Execution {
    pub fn local(options: ExecutionOptions) -> RuntimeResult<Self> {
        let cpu = match &options.device {
            Device::Auto => CpuOptions::default(),
            Device::Cpu(options) => options.clone(),
            Device::Gpu(options) => {
                return Err(ExecutionError::GpuUnavailable(options.backend).into());
            }
        };
        let precision = match options.precision {
            Precision::Auto | Precision::F64 => Precision::F64,
            Precision::F32 => return Err(ExecutionError::UnsupportedCpuPrecision.into()),
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
            threads: cpu.threads,
            jit: cpu.jit,
            pool,
            partitioning: options.partitioning,
            #[cfg(feature = "mpi")]
            communicator: None,
        })
    }

    #[cfg(feature = "mpi")]
    pub fn distributed<C>(options: ExecutionOptions, world: &C) -> RuntimeResult<Self>
    where
        C: Communicator,
    {
        let mut execution = Self::local(options)?;
        execution.communicator = Some(Arc::new(world.duplicate()));
        Ok(execution)
    }

    pub fn requested_device(&self) -> &Device {
        &self.requested_device
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn thread_policy(&self) -> ThreadPolicy {
        self.threads
    }

    pub fn jit_policy(&self) -> JitPolicy {
        self.jit
    }

    pub fn partitioning(&self) -> Partitioning {
        self.partitioning
    }

    pub fn rank(&self) -> usize {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            return communicator.rank() as usize;
        }
        0
    }

    pub fn nranks(&self) -> usize {
        #[cfg(feature = "mpi")]
        if let Some(communicator) = &self.communicator {
            return communicator.size() as usize;
        }
        1
    }

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
    use crate::{RuntimeError, execution::GpuBackend};

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
        assert!(matches!(
            Execution::local(ExecutionOptions {
                device: Device::Cpu(CpuOptions::default()),
                precision: Precision::F32,
                ..ExecutionOptions::default()
            }),
            Err(RuntimeError::Execution(
                ExecutionError::UnsupportedCpuPrecision
            ))
        ));
    }
}
