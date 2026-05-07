#[cfg(feature = "rayon")]
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};

use super::ThreadPoolManager;
#[cfg(feature = "rayon")]
use crate::execution::thread_pool::ThreadExecutor;

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
