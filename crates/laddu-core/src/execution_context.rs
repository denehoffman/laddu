//! Execution context prototype for reusing thread policy and scratch allocations.

use parking_lot::Mutex;

use crate::{LadduError, LadduResult};

/// Thread-policy options for [`ExecutionContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadPolicy {
    /// Run work on the current thread.
    Single,
    /// Use the global Rayon pool.
    GlobalPool,
    /// Use a dedicated Rayon pool with `n_threads`.
    Dedicated(usize),
}

/// Reusable scratch buffers owned by an [`ExecutionContext`].
#[derive(Debug, Default)]
pub struct ScratchAllocator {
    byte_scratch: Vec<u8>,
    scalar_scratch: Vec<f64>,
}

impl ScratchAllocator {
    /// Ensure and return a byte scratch slice with `len` elements.
    pub fn reserve_bytes(&mut self, len: usize) -> &mut [u8] {
        if self.byte_scratch.len() < len {
            self.byte_scratch.resize(len, 0);
        }
        &mut self.byte_scratch[..len]
    }

    /// Ensure and return an `f64` scratch slice with `len` elements.
    pub fn reserve_scalars(&mut self, len: usize) -> &mut [f64] {
        if self.scalar_scratch.len() < len {
            self.scalar_scratch.resize(len, 0.0);
        }
        &mut self.scalar_scratch[..len]
    }

    /// Clear scratch values while retaining allocated capacity for reuse.
    pub fn clear(&mut self) {
        self.byte_scratch.clear();
        self.scalar_scratch.clear();
    }

    /// Return `(byte_capacity, scalar_capacity)` for current buffers.
    pub fn capacities(&self) -> (usize, usize) {
        (self.byte_scratch.capacity(), self.scalar_scratch.capacity())
    }
}

/// Prototype execution context owning thread policy and scratch allocators.
///
/// This type is intended for repeated evaluator calls where setup overhead should
/// be amortized by reusing a single context instance.
#[derive(Debug)]
pub struct ExecutionContext {
    thread_policy: ThreadPolicy,
    #[cfg(feature = "rayon")]
    dedicated_pool: Option<rayon::ThreadPool>,
    scratch: Mutex<ScratchAllocator>,
}

impl ExecutionContext {
    /// Create a new context with the requested thread policy.
    pub fn new(thread_policy: ThreadPolicy) -> LadduResult<Self> {
        #[cfg(not(feature = "rayon"))]
        {
            if thread_policy != ThreadPolicy::Single {
                return Err(LadduError::ExecutionContextError {
                    reason: "Rayon feature is required for non-single thread policies".into(),
                });
            }
        }

        #[cfg(feature = "rayon")]
        let dedicated_pool = match thread_policy {
            ThreadPolicy::Dedicated(n_threads) => {
                if n_threads == 0 {
                    return Err(LadduError::ExecutionContextError {
                        reason: "Dedicated thread pool size must be >= 1".into(),
                    });
                }
                Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(n_threads)
                        .build()?,
                )
            }
            ThreadPolicy::Single | ThreadPolicy::GlobalPool => None,
        };

        Ok(Self {
            thread_policy,
            #[cfg(feature = "rayon")]
            dedicated_pool,
            scratch: Mutex::new(ScratchAllocator::default()),
        })
    }

    /// Return the configured thread policy.
    pub fn thread_policy(&self) -> ThreadPolicy {
        self.thread_policy
    }

    /// Execute work under this context's thread policy.
    #[cfg(feature = "rayon")]
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match &self.dedicated_pool {
            Some(pool) => pool.install(op),
            None => op(),
        }
    }

    /// Execute work under this context's thread policy.
    #[cfg(not(feature = "rayon"))]
    pub fn install<R>(&self, op: impl FnOnce() -> R) -> R {
        op()
    }

    /// Access reusable scratch buffers.
    pub fn with_scratch<R>(&self, op: impl FnOnce(&mut ScratchAllocator) -> R) -> R {
        let mut scratch = self.scratch.lock();
        op(&mut scratch)
    }
}
