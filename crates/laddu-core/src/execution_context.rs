//! Execution context prototype for reusing thread policy and scratch allocations.
//!
//! `ExecutionContext` is intended to be created once and reused across many evaluator calls.
//! Reusing a context avoids repeated setup and enables scratch-buffer reuse for
//! `ThreadPolicy::Single`.
//!
//! Lifecycle:
//! - Create once with [`ExecutionContext::new`].
//! - Reuse in repeated calls to
//!   [`Evaluator::evaluate_with_ctx`](crate::amplitudes::Evaluator::evaluate_with_ctx)
//!   and
//!   [`Evaluator::evaluate_gradient_with_ctx`](crate::amplitudes::Evaluator::evaluate_gradient_with_ctx).
//! - Drop when analysis work is complete; thread-pool and scratch resources are released.
//!
//! Thread policy guidance:
//! - [`ThreadPolicy::Single`]: runs on the caller thread and reuses shared scratch buffers.
//! - [`ThreadPolicy::GlobalPool`]: uses Rayon global parallelism when available.
//! - [`ThreadPolicy::Dedicated`]: creates a private Rayon pool; setup is higher-cost, so it
//!   should be reused across many calls.

#[cfg(feature = "rayon")]
use std::sync::Arc;
use std::sync::OnceLock;

use nalgebra::DVector;
use num::complex::Complex64;
use parking_lot::Mutex;
#[cfg(feature = "rayon")]
use parking_lot::RwLock;

use crate::{LadduError, LadduResult};

/// Thread-policy options for [`ExecutionContext`].
///
/// These control where evaluator work executes when using context-aware evaluator methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadPolicy {
    /// Run work on the current thread.
    Single,
    /// Use the global Rayon pool.
    GlobalPool,
    /// Use a dedicated Rayon pool with `n_threads`.
    Dedicated(usize),
}

/// Shared manager for per-call Rayon thread-pool reuse.
///
/// This manager is intended for APIs that accept an optional thread count on each call.
/// Requests with `None` or `Some(0)` use the ambient/global Rayon behavior. Positive thread
/// counts reuse one cached dedicated pool for the most recently requested size.
#[derive(Debug, Default)]
pub struct ThreadPoolManager {
    #[cfg(feature = "rayon")]
    dedicated_pool: RwLock<Option<(usize, Arc<rayon::ThreadPool>)>>,
}

impl ThreadPoolManager {
    /// Return the process-wide shared pool manager.
    pub fn shared() -> &'static Self {
        static THREAD_POOL_MANAGER: OnceLock<ThreadPoolManager> = OnceLock::new();
        THREAD_POOL_MANAGER.get_or_init(Self::default)
    }

    /// Execute work using the requested thread-count policy.
    ///
    /// `None` or `Some(0)` uses the ambient/global Rayon behavior. Positive thread counts reuse
    /// a cached dedicated pool of that size.
    #[cfg(feature = "rayon")]
    pub fn install<R: Send>(
        &self,
        requested_threads: Option<usize>,
        op: impl FnOnce() -> R + Send,
    ) -> LadduResult<R> {
        match Self::normalize_thread_request(requested_threads) {
            Some(n_threads) => Ok(self.pool_for_threads(n_threads)?.install(op)),
            None => Ok(op()),
        }
    }

    /// Execute work using the requested thread-count policy.
    ///
    /// Without Rayon, all work runs on the caller thread and the requested thread count is
    /// ignored.
    #[cfg(not(feature = "rayon"))]
    pub fn install<R>(
        &self,
        _requested_threads: Option<usize>,
        op: impl FnOnce() -> R,
    ) -> LadduResult<R> {
        Ok(op())
    }

    #[cfg(feature = "rayon")]
    fn normalize_thread_request(requested_threads: Option<usize>) -> Option<usize> {
        requested_threads.filter(|&n_threads| n_threads > 0)
    }

    #[cfg(feature = "rayon")]
    fn pool_for_threads(&self, n_threads: usize) -> LadduResult<Arc<rayon::ThreadPool>> {
        if let Some((cached_threads, pool)) = &*self.dedicated_pool.read() {
            if *cached_threads == n_threads {
                return Ok(pool.clone());
            }
        }

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()?,
        );

        let mut dedicated_pool = self.dedicated_pool.write();
        *dedicated_pool = Some((n_threads, pool.clone()));
        Ok(pool)
    }
}

/// Reusable scratch buffers owned by an [`ExecutionContext`].
#[derive(Debug, Default)]
pub struct ScratchAllocator {
    byte_scratch: Vec<u8>,
    scalar_scratch: Vec<f64>,
    complex_scratch: Vec<Complex64>,
    gradient_event_scratch: Vec<DVector<Complex64>>,
    gradient_expr_scratch: Vec<DVector<Complex64>>,
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

    /// Ensure and return reusable complex workspaces for value evaluation.
    pub fn reserve_value_workspaces(
        &mut self,
        amplitude_len: usize,
        slot_count: usize,
    ) -> (&mut [Complex64], &mut [Complex64]) {
        let total = amplitude_len + slot_count;
        if self.complex_scratch.len() < total {
            self.complex_scratch.resize(total, Complex64::ZERO);
        }
        let (amplitudes, slots) = self.complex_scratch[..total].split_at_mut(amplitude_len);
        (amplitudes, slots)
    }

    /// Ensure and return reusable workspaces for gradient evaluation.
    #[allow(clippy::type_complexity)]
    pub fn reserve_gradient_workspaces(
        &mut self,
        amplitude_len: usize,
        slot_count: usize,
        grad_dim: usize,
    ) -> (
        &mut [Complex64],
        &mut [Complex64],
        &mut [DVector<Complex64>],
        &mut [DVector<Complex64>],
    ) {
        Self::ensure_gradient_shape(&mut self.gradient_event_scratch, amplitude_len, grad_dim);
        Self::ensure_gradient_shape(&mut self.gradient_expr_scratch, slot_count, grad_dim);
        let total = amplitude_len + slot_count;
        if self.complex_scratch.len() < total {
            self.complex_scratch.resize(total, Complex64::ZERO);
        }
        let (amplitudes, slots) = self.complex_scratch[..total].split_at_mut(amplitude_len);
        (
            amplitudes,
            slots,
            &mut self.gradient_event_scratch[..amplitude_len],
            &mut self.gradient_expr_scratch[..slot_count],
        )
    }

    /// Clear scratch values while retaining allocated capacity for reuse.
    pub fn clear(&mut self) {
        self.byte_scratch.clear();
        self.scalar_scratch.clear();
        self.complex_scratch.clear();
        self.gradient_event_scratch.clear();
        self.gradient_expr_scratch.clear();
    }

    /// Return `(byte_capacity, scalar_capacity)` for current buffers.
    pub fn capacities(&self) -> (usize, usize) {
        (self.byte_scratch.capacity(), self.scalar_scratch.capacity())
    }

    fn ensure_gradient_shape(
        buffer: &mut Vec<DVector<Complex64>>,
        outer_len: usize,
        grad_dim: usize,
    ) {
        if buffer.len() < outer_len {
            buffer.extend((buffer.len()..outer_len).map(|_| DVector::zeros(grad_dim)));
        } else if buffer.len() > outer_len {
            buffer.truncate(outer_len);
        }
        for gradient in buffer.iter_mut() {
            if gradient.len() != grad_dim {
                *gradient = DVector::zeros(grad_dim);
            }
        }
    }
}

/// Prototype execution context owning thread policy and scratch allocators.
///
/// This type should usually be long-lived relative to evaluator calls.
/// Creating one context and reusing it across repeated calls provides the intended behavior.
#[derive(Debug)]
pub struct ExecutionContext {
    thread_policy: ThreadPolicy,
    #[cfg(feature = "rayon")]
    dedicated_pool: Option<rayon::ThreadPool>,
    scratch: Mutex<ScratchAllocator>,
}

impl ExecutionContext {
    /// Create a new context with the requested thread policy.
    ///
    /// Returns an error when the requested policy is incompatible with the current feature set
    /// (for example, non-single policy without `rayon`) or when a dedicated pool size is invalid.
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
    ///
    /// `Dedicated` runs inside the dedicated pool. Other policies run the closure directly.
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
    ///
    /// Scratch memory capacity is retained across calls to support repeated evaluations.
    pub fn with_scratch<R>(&self, op: impl FnOnce(&mut ScratchAllocator) -> R) -> R {
        let mut scratch = self.scratch.lock();
        op(&mut scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::ThreadPoolManager;

    #[cfg(feature = "rayon")]
    use std::sync::Arc;

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_reuses_cached_pool_for_same_thread_count() {
        let manager = ThreadPoolManager::default();
        let first_pool = manager
            .pool_for_threads(2)
            .expect("pool for two threads should build");
        let second_pool = manager
            .pool_for_threads(2)
            .expect("pool for two threads should be cached");
        assert!(Arc::ptr_eq(&first_pool, &second_pool));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_separates_distinct_thread_counts() {
        let manager = ThreadPoolManager::default();
        let two_thread_pool = manager
            .pool_for_threads(2)
            .expect("pool for two threads should build");
        let three_thread_pool = manager
            .pool_for_threads(3)
            .expect("pool for three threads should build");
        assert!(!Arc::ptr_eq(&two_thread_pool, &three_thread_pool));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_replaces_cached_pool_when_thread_count_changes() {
        let manager = ThreadPoolManager::default();
        let first_two_thread_pool = manager
            .pool_for_threads(2)
            .expect("pool for two threads should build");
        manager
            .pool_for_threads(3)
            .expect("pool for three threads should replace the cache");
        let second_two_thread_pool = manager
            .pool_for_threads(2)
            .expect("pool for two threads should rebuild after cache replacement");
        assert!(!Arc::ptr_eq(
            &first_two_thread_pool,
            &second_two_thread_pool
        ));
    }

    #[test]
    fn thread_pool_manager_treats_zero_threads_as_global_fallback() {
        let manager = ThreadPoolManager::default();
        let value = manager
            .install(Some(0), || 17usize)
            .expect("global fallback install should succeed");
        assert_eq!(value, 17);
        #[cfg(feature = "rayon")]
        assert!(manager.dedicated_pool.read().is_none());
    }
}
