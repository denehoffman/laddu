//! Execution-policy and thread-pool coordination helpers.

#[cfg(feature = "execution-context-prototype")]
mod context;
mod thread_pool;

#[cfg(feature = "execution-context-prototype")]
pub use context::{ExecutionContext, ScratchAllocator, ThreadPolicy};
pub use thread_pool::ThreadPoolManager;

#[cfg(test)]
mod tests;
