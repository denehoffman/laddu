use laddu_core::{LadduResult, ThreadPoolManager};

pub mod callbacks;
pub mod experimental;
pub mod likelihood;
pub mod optimize;

#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) fn install_laddu_with_threads<R: Send>(
    threads: Option<usize>,
    op: impl FnOnce() -> LadduResult<R> + Send,
) -> LadduResult<R> {
    ThreadPoolManager::shared().install(threads, op)?
}
