//! Optimization helpers and algorithm integrations for likelihood terms.

/// Callback and observer integration used by optimization algorithms.
pub mod callbacks;
/// Bindings and adapters for optimization algorithms from `ganesh`.
pub mod ganesh;

pub use callbacks::LikelihoodTermObserver;
use laddu_core::{LadduResult, ThreadPoolManager};

/// A wrapper for the requested thread-count policy used by optimization callbacks.
#[derive(Clone, Copy, Debug)]
pub struct MaybeThreadPool {
    requested_threads: Option<usize>,
}

impl MaybeThreadPool {
    /// Crate a new thread pool with the given number of threads. This is typically used as
    /// user-data for [`ganesh`] optimizations.
    pub fn new(num_threads: usize) -> Self {
        Self {
            requested_threads: Some(num_threads),
        }
    }

    /// Run the given operation on the current thread pool.
    pub fn install<R: Send>(&self, op: impl FnOnce() -> LadduResult<R> + Send) -> LadduResult<R> {
        ThreadPoolManager::shared().install(self.requested_threads, op)?
    }
}
