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
    dedicated_pool: RwLock<Option<(usize, ThreadExecutor)>>,
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
    #[cfg(feature = "rayon")]
    use super::ThreadExecutor;
    use super::ThreadPoolManager;

    use std::sync::{Mutex, MutexGuard};

    #[cfg(feature = "rayon")]
    use std::sync::Arc;

    static GLOBAL_THREAD_COUNT_TEST_GUARD: Mutex<()> = Mutex::new(());

    struct GlobalThreadCountReset {
        previous: Option<usize>,
        _guard: MutexGuard<'static, ()>,
    }

    impl GlobalThreadCountReset {
        fn new(n_threads: usize) -> Self {
            let guard = GLOBAL_THREAD_COUNT_TEST_GUARD
                .lock()
                .expect("global thread-count test guard should not be poisoned");
            let previous = ThreadPoolManager::global_thread_count();
            ThreadPoolManager::set_global_thread_count(n_threads);
            Self {
                previous,
                _guard: guard,
            }
        }
    }

    impl Drop for GlobalThreadCountReset {
        fn drop(&mut self) {
            ThreadPoolManager::set_global_thread_count(self.previous.unwrap_or(0));
        }
    }

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
        let _reset = GlobalThreadCountReset::new(0);
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

    #[cfg(feature = "rayon")]
    #[test]
    fn thread_pool_manager_uses_global_default_for_omitted_and_zero_requests() {
        let _reset = GlobalThreadCountReset::new(2);
        let manager = ThreadPoolManager::default();

        manager
            .install(None, || 17usize)
            .expect("omitted thread request should succeed");
        let omitted_pool = match manager
            .dedicated_pool
            .read()
            .as_ref()
            .expect("global default should create a dedicated pool")
            .1
            .clone()
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
        assert_eq!(
            manager
                .dedicated_pool
                .read()
                .as_ref()
                .map(|(n_threads, _)| *n_threads),
            Some(2)
        );

        manager
            .install(Some(0), || 23usize)
            .expect("zero thread request should use the global default");
        let zero_pool = match manager
            .dedicated_pool
            .read()
            .as_ref()
            .expect("global default should remain cached")
            .1
            .clone()
        {
            ThreadExecutor::Dedicated(pool) => pool,
            ThreadExecutor::Ambient => panic!("executor should be dedicated"),
        };
        assert!(Arc::ptr_eq(&omitted_pool, &zero_pool));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn explicit_thread_request_overrides_global_default() {
        let _reset = GlobalThreadCountReset::new(2);
        let manager = ThreadPoolManager::default();

        manager
            .install(Some(3), || 19usize)
            .expect("explicit thread request should succeed");
        assert_eq!(
            manager
                .dedicated_pool
                .read()
                .as_ref()
                .map(|(n_threads, _)| *n_threads),
            Some(3)
        );

        manager
            .install(None, || 29usize)
            .expect("omitted thread request should fall back to the global default");
        assert_eq!(
            manager
                .dedicated_pool
                .read()
                .as_ref()
                .map(|(n_threads, _)| *n_threads),
            Some(2)
        );
    }

    #[test]
    fn resolve_thread_request_prioritizes_explicit_positive_values() {
        let _reset = GlobalThreadCountReset::new(2);

        assert_eq!(ThreadPoolManager::resolve_thread_request(None), Some(2));
        assert_eq!(ThreadPoolManager::resolve_thread_request(Some(0)), Some(2));
        assert_eq!(ThreadPoolManager::resolve_thread_request(Some(3)), Some(3));
    }

    #[test]
    fn set_global_thread_count_zero_resets_to_ambient_behavior() {
        let _reset = GlobalThreadCountReset::new(3);
        assert_eq!(ThreadPoolManager::global_thread_count(), Some(3));

        ThreadPoolManager::set_global_thread_count(0);
        assert_eq!(ThreadPoolManager::global_thread_count(), None);
        assert_eq!(ThreadPoolManager::resolve_thread_request(None), None);
        assert_eq!(ThreadPoolManager::resolve_thread_request(Some(0)), None);
    }
}
