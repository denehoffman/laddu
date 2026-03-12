//! Shared thread-pool manager for APIs that accept a per-call thread count.

#[cfg(feature = "rayon")]
use std::sync::Arc;
use std::sync::OnceLock;

#[cfg(feature = "rayon")]
use parking_lot::RwLock;

use crate::{LadduError, LadduResult};

/// Shared thread-execution mode used by both [`ThreadPoolManager`] and
/// [`ExecutionContext`](crate::execution_context::ExecutionContext).
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
    #[cfg(not(feature = "rayon"))]
    pub(crate) fn install<R>(&self, op: impl FnOnce() -> R) -> R {
        op()
    }
}

/// Shared manager for per-call Rayon thread-pool reuse.
///
/// This manager is intended for APIs that accept an optional thread count on each call.
/// Requests with `None` or `Some(0)` use the ambient/global Rayon behavior. Positive thread
/// counts reuse one cached dedicated pool for the most recently requested size.
#[derive(Debug, Default)]
pub struct ThreadPoolManager {
    #[cfg(feature = "rayon")]
    dedicated_pool: RwLock<Option<(usize, ThreadExecutor)>>,
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

    #[cfg(feature = "rayon")]
    fn normalize_thread_request(requested_threads: Option<usize>) -> Option<usize> {
        requested_threads.filter(|&n_threads| n_threads > 0)
    }

    #[cfg(feature = "rayon")]
    fn executor_for_threads(&self, n_threads: usize) -> LadduResult<ThreadExecutor> {
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

#[cfg(test)]
mod tests {
    use super::{ThreadExecutor, ThreadPoolManager};

    #[cfg(feature = "rayon")]
    use std::sync::Arc;

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_reuses_cached_pool_for_same_thread_count() {
        let manager = ThreadPoolManager::default();
        let first_pool = match manager
            .executor_for_threads(2)
            .expect("pool for two threads should build")
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
        let second_pool = match manager
            .executor_for_threads(2)
            .expect("pool for two threads should be cached")
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
        assert!(Arc::ptr_eq(&first_pool, &second_pool));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_separates_distinct_thread_counts() {
        let manager = ThreadPoolManager::default();
        let two_thread_pool = match manager
            .executor_for_threads(2)
            .expect("pool for two threads should build")
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
        let three_thread_pool = match manager
            .executor_for_threads(3)
            .expect("pool for three threads should build")
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
        assert!(!Arc::ptr_eq(&two_thread_pool, &three_thread_pool));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_replaces_cached_pool_when_thread_count_changes() {
        let manager = ThreadPoolManager::default();
        let first_two_thread_pool = match manager
            .executor_for_threads(2)
            .expect("pool for two threads should build")
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
        manager
            .executor_for_threads(3)
            .expect("pool for three threads should replace the cache");
        let second_two_thread_pool = match manager
            .executor_for_threads(2)
            .expect("pool for two threads should rebuild after cache replacement")
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
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

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_reuses_cached_pool_across_many_short_installs() {
        let manager = ThreadPoolManager::default();
        let first_pool = match manager
            .executor_for_threads(2)
            .expect("pool for two threads should build")
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };

        let total = (0usize..128)
            .map(|index| {
                let value = manager
                    .install(Some(2), || index + 1)
                    .expect("repeated short install should succeed");
                let cached_pool = match manager
                    .executor_for_threads(2)
                    .expect("pool for two threads should remain cached")
                {
                    ThreadExecutor::Dedicated(pool) => pool,
                    ThreadExecutor::Ambient => panic!("executor should be dedicated"),
                };
                assert!(Arc::ptr_eq(&first_pool, &cached_pool));
                value
            })
            .sum::<usize>();

        assert_eq!(total, (1usize..=128).sum::<usize>());
    }
}
