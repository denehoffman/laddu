/// Dataset transformations, traversal, and reductions.
pub mod dataset;
/// Columnar batches and borrowed or owned event views.
pub mod event;

/// Numerically accurate reduction accumulators.
#[cfg(feature = "parallel")]
pub use dataset::accurate;
pub use dataset::{CacheStorage, Dataset, MemoryPolicy};
pub(crate) use event::BatchAssembler;
pub use event::{BatchEvent, Event, EventBatch, EventBatchBuilder, OwnedEvent};
