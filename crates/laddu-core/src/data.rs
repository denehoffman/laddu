//! Methods for loading and manipulating [`EventData`](crate::data::EventData)-based data.

mod dataset;
mod event;
pub mod io;
mod metadata;
#[cfg(test)]
mod tests;

pub use dataset::*;
pub use event::*;
pub use metadata::*;
