//! Shared thread-pool manager for APIs that accept a per-call thread count.

#[cfg(feature = "rayon")]
use std::sync::Arc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

#[cfg(feature = "rayon")]
use parking_lot::RwLock;

#[cfg(feature = "rayon")]
use crate::LadduError;
use crate::LadduResult;

static GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Shared thread-execution mode used by both [`ThreadPoolManager`] and
/// [`ExecutionContext`](crate::execution::ExecutionContext).
#[derive(Debug, Clone, Default)]
pub(crate) enum ThreadExecutor {
    /// Run work on the caller thread / ambient global Rayon context.
    #[default]
    Ambient,
    /// Run work on a dedicated Rayon pool.
    #[cfg(feature = "rayon")]
    Dedicated(Arc<rayon::ThreadPool>),
}

impl ThreadExecutor {
    /// Create a dedicated executor with `n_threads`.
    #[cfg(feature = "rayon")]
    pub(crate) fn dedicated(n_threads: usize) -> LadduResult<Self> {
        if n_threads == 0 {
            return Err(LadduError::ExecutionContextError {
                reason: "Dedicated thread pool size must be >= 1".into(),
            });
        }

        Ok(Self::Dedicated(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()?,
        )))
    }

    /// Execute work using this executor.
    #[cfg(feature = "rayon")]
    pub(crate) fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match self {
            Self::Ambient => op(),
            Self::Dedicated(pool) => pool.install(op),
        }
    }

    /// Execute work using this executor.
    #[allow(dead_code)]
    #[cfg(not(feature = "rayon"))]
    pub(crate) fn install<R>(&self, op: impl FnOnce() -> R) -> R {
        op()
    }
}

/// Shared manager for per-call Rayon thread-pool reuse.
///
/// This manager is intended for APIs that accept an optional thread count on each call.
/// Requests with `None` or `Some(0)` use the configured global default. When that default is `0`,
/// work falls back to the ambient/global Rayon behavior. Positive thread counts reuse one cached
/// dedicated pool for the most recently requested size.
#[derive(Debug, Default)]
pub struct ThreadPoolManager {
    #[cfg(feature = "rayon")]
    pub(crate) dedicated_pool: RwLock<Option<(usize, ThreadExecutor)>>,
}

impl ThreadPoolManager {
    /// Return the process-wide shared pool manager.
    pub fn shared() -> &'static Self {
        static THREAD_POOL_MANAGER: OnceLock<ThreadPoolManager> = OnceLock::new();
        THREAD_POOL_MANAGER.get_or_init(Self::default)
    }

    /// Set the process-global default thread count used by omitted or zero-valued requests.
    ///
    /// A value of `0` resets the default to the ambient/global Rayon behavior.
    pub fn set_global_thread_count(n_threads: usize) {
        GLOBAL_THREAD_COUNT.store(n_threads, Ordering::Relaxed);
    }

    /// Return the process-global default thread count used by omitted or zero-valued requests.
    ///
    /// Returns `None` when the default is the ambient/global Rayon behavior.
    pub fn global_thread_count() -> Option<usize> {
        Self::normalize_thread_request(Some(GLOBAL_THREAD_COUNT.load(Ordering::Relaxed)))
    }

    /// Resolve an optional thread request against the process-global default.
    ///
    /// `None` and `Some(0)` both use the configured global default. Positive thread counts bypass
    /// the global default and are returned directly.
    pub fn resolve_thread_request(requested_threads: Option<usize>) -> Option<usize> {
        match requested_threads {
            None | Some(0) => Self::global_thread_count(),
            Some(n_threads) => Some(n_threads),
        }
    }

    /// Execute work using the requested thread-count policy.
    ///
    /// `None` and `Some(0)` both use the configured global default. Positive thread counts reuse a
    /// cached dedicated pool of that size.
    #[cfg(feature = "rayon")]
    pub fn install<R: Send>(
        &self,
        requested_threads: Option<usize>,
        op: impl FnOnce() -> R + Send,
    ) -> LadduResult<R> {
        match Self::resolve_thread_request(requested_threads) {
            Some(n_threads) => Ok(self.executor_for_threads(n_threads)?.install(op)),
            None => Ok(ThreadExecutor::default().install(op)),
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

    fn normalize_thread_request(requested_threads: Option<usize>) -> Option<usize> {
        requested_threads.filter(|&n_threads| n_threads > 0)
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn executor_for_threads(&self, n_threads: usize) -> LadduResult<ThreadExecutor> {
        if let Some((cached_threads, executor)) = &*self.dedicated_pool.read() {
            if *cached_threads == n_threads {
                return Ok(executor.clone());
            }
        }

        let executor = ThreadExecutor::dedicated(n_threads)?;
        let mut dedicated_pool = self.dedicated_pool.write();
        *dedicated_pool = Some((n_threads, executor.clone()));
        Ok(executor)
    }
}
