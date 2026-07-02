use std::sync::Arc;

use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

use laddu_data::io::{Partitioning, ReadPlan};

#[cfg(feature = "mpi")]
use mpi::{
    collective::SystemOperation,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives},
};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ThreadPolicy {
    /// Use Rayon's process-wide thread pool.
    #[default]
    Auto,
    /// Evaluate on the calling thread.
    Serial,
    /// Use a private Rayon pool with exactly this many threads.
    Fixed(usize),
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuExecutionOptions {
    /// Local CPU concurrency. `Auto` uses Rayon's global pool.
    pub threads: ThreadPolicy,
    /// MPI row assignment when a distributed communicator is attached.
    pub partitioning: Partitioning,
}

#[derive(Clone, Debug, Error)]
pub enum CpuExecutionError {
    #[error("fixed thread count must be nonzero")]
    ZeroThreads,
    #[error("failed to create Rayon thread pool: {0}")]
    ThreadPool(String),
}

#[derive(Clone)]
pub struct CpuExecution {
    threads: ThreadPolicy,
    pool: Option<Arc<ThreadPool>>,
    partitioning: Partitioning,
    #[cfg(feature = "mpi")]
    communicator: Option<Arc<SimpleCommunicator>>,
}

impl std::fmt::Debug for CpuExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CpuExecution")
            .field("threads", &self.threads)
            .field("partitioning", &self.partitioning)
            .field("ranks", &self.nranks())
            .finish_non_exhaustive()
    }
}

impl Default for CpuExecution {
    fn default() -> Self {
        Self::local(CpuExecutionOptions::default())
            .expect("the automatic execution policy does not construct a private pool")
    }
}

impl CpuExecution {
    /// Create local execution with runtime-selectable Rayon behavior.
    pub fn local(options: CpuExecutionOptions) -> Result<Self, CpuExecutionError> {
        let pool = match options.threads {
            ThreadPolicy::Fixed(0) => return Err(CpuExecutionError::ZeroThreads),
            ThreadPolicy::Fixed(threads) => Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|error| CpuExecutionError::ThreadPool(error.to_string()))?,
            )),
            ThreadPolicy::Auto | ThreadPolicy::Serial => None,
        };
        Ok(Self {
            threads: options.threads,
            pool,
            partitioning: options.partitioning,
            #[cfg(feature = "mpi")]
            communicator: None,
        })
    }

    #[cfg(feature = "mpi")]
    /// Create distributed execution by duplicating the supplied MPI communicator.
    ///
    /// Dataset partitioning and global likelihood reductions then happen transparently. The caller
    /// remains responsible for initializing MPI before constructing this value.
    pub fn distributed<C>(
        options: CpuExecutionOptions,
        world: &C,
    ) -> Result<Self, CpuExecutionError>
    where
        C: Communicator,
    {
        let mut execution = Self::local(options)?;
        execution.communicator = Some(Arc::new(world.duplicate()));
        Ok(execution)
    }

    pub fn thread_policy(&self) -> ThreadPolicy {
        self.threads
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

    #[test]
    fn execution_selects_serial_auto_and_fixed_thread_policies() {
        let serial = CpuExecution::local(CpuExecutionOptions {
            threads: ThreadPolicy::Serial,
            ..CpuExecutionOptions::default()
        })
        .unwrap();
        assert!(!serial.is_parallel());

        let automatic = CpuExecution::default();
        assert!(automatic.is_parallel());

        let fixed = CpuExecution::local(CpuExecutionOptions {
            threads: ThreadPolicy::Fixed(2),
            ..CpuExecutionOptions::default()
        })
        .unwrap();
        assert!(fixed.is_parallel());
        assert_eq!(fixed.install(rayon::current_num_threads), 2);

        assert!(matches!(
            CpuExecution::local(CpuExecutionOptions {
                threads: ThreadPolicy::Fixed(0),
                ..CpuExecutionOptions::default()
            }),
            Err(CpuExecutionError::ZeroThreads)
        ));
    }
}
