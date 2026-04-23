use std::{
    ops::{Deref, DerefMut, Index, IndexMut},
    sync::Arc,
};

use accurate::{sum::Klein, traits::*};
use auto_ops::impl_op_ex;
#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::{
    event::{
        test_event, ColumnarP4Column, DatasetStorage, Event, EventData, NamedEventView,
        TEST_AUX_NAMES, TEST_P4_NAMES,
    },
    metadata::DatasetMetadata,
};
#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;

#[cfg(feature = "mpi")]
pub(crate) type WorldHandle = SimpleCommunicator;
#[cfg(not(feature = "mpi"))]
pub(crate) type WorldHandle = ();

#[cfg(feature = "mpi")]
// Chosen from local two-rank probes: 512 matched or beat smaller chunks
// while keeping the fetched-event cache modest.
const DEFAULT_MPI_EVENT_FETCH_CHUNK_SIZE: usize = 512;
#[cfg(feature = "mpi")]
const MPI_EVENT_FETCH_CHUNK_SIZE_ENV: &str = "LADDU_MPI_EVENT_FETCH_CHUNK_SIZE";

use indexmap::IndexMap;

use crate::{
    math::get_bin_edges,
    variables::{IntoP4Selection, P4Selection, Variable, VariableExpression},
    vectors::Vec4,
    LadduError, LadduResult,
};

/// A dataset that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitudes::Amplitude). This particular dataset contains a single
/// [`EventData`] generated from [`test_event`].
pub fn test_dataset() -> Dataset {
    let metadata = Arc::new(
        DatasetMetadata::new(
            TEST_P4_NAMES.iter().map(|s| (*s).to_string()).collect(),
            TEST_AUX_NAMES.iter().map(|s| (*s).to_string()).collect(),
        )
        .expect("Test metadata should be valid"),
    );
    Dataset::new_with_metadata(vec![Arc::new(test_event())], metadata)
}

/// A collection of events with optional metadata for name-based lookups.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// The [`EventData`] contained in the [`Dataset`]
    events: Vec<Event>,
    pub(crate) columnar: DatasetStorage,
    pub(crate) metadata: Arc<DatasetMetadata>,
    pub(crate) cached_local_weighted_sum: f64,
    #[cfg(feature = "mpi")]
    pub(crate) cached_global_event_count: usize,
    #[cfg(feature = "mpi")]
    pub(crate) cached_global_weighted_sum: f64,
}

impl IntoIterator for Dataset {
    type Item = Event;

    type IntoIter = DatasetIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                // Cache total before moving fields out of self for MPI iteration.
                let total = self.n_events();
                return DatasetIntoIter::Mpi(DatasetMpiIntoIter {
                    events: self.events,
                    metadata: self.metadata,
                    world,
                    index: 0,
                    total,
                    cursor: MpiEventChunkCursor::for_iteration(total),
                });
            }
        }
        DatasetIntoIter::Local(self.events.into_iter())
    }
}

fn shared_dataset_iter(dataset: Arc<Dataset>) -> DatasetArcIter {
    #[cfg(feature = "mpi")]
    {
        if let Some(world) = crate::mpi::get_world() {
            let total = dataset.n_events();
            return DatasetArcIter::Mpi(DatasetArcMpiIter {
                dataset,
                world,
                index: 0,
                total,
                cursor: MpiEventChunkCursor::for_iteration(total),
            });
        }
    }
    DatasetArcIter::Local { dataset, index: 0 }
}

/// Extension methods for shared [`Arc<Dataset>`] handles.
pub trait SharedDatasetIterExt {
    /// Build an iterator over a shared [`Arc<Dataset>`] without cloning the dataset contents.
    fn shared_iter(&self) -> DatasetArcIter;

    /// Alias for [`SharedDatasetIterExt::shared_iter`].
    fn shared_iter_global(&self) -> DatasetArcIter;
}

impl SharedDatasetIterExt for Arc<Dataset> {
    fn shared_iter(&self) -> DatasetArcIter {
        shared_dataset_iter(self.clone())
    }

    fn shared_iter_global(&self) -> DatasetArcIter {
        self.shared_iter()
    }
}

impl Dataset {
    /// Borrow locally stored events.
    ///
    /// When MPI is enabled, this slice contains only the current rank's event ownership.
    pub fn events_local(&self) -> &[Event] {
        &self.events
    }

    /// Collect all events into a [`Vec`] using the default global iteration semantics.
    ///
    /// When MPI is enabled, the returned vector is ordered like [`Dataset::iter`] and
    /// may include remotely owned events fetched on demand.
    pub fn events_global(&self) -> Vec<Event> {
        self.iter_global().collect()
    }

    #[cfg(test)]
    pub(crate) fn clear_events_local(&mut self) {
        self.events.clear();
    }

    /// Iterate over all events in the dataset. When MPI is enabled, this will visit
    /// every event across all ranks, fetching remote events on demand.
    pub fn iter(&self) -> DatasetIter<'_> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                let total = self.n_events();
                return DatasetIter::Mpi(DatasetMpiIter {
                    dataset: self,
                    world,
                    index: 0,
                    total,
                    cursor: MpiEventChunkCursor::for_iteration(total),
                });
            }
        }
        DatasetIter::Local(self.events.iter())
    }

    /// Alias for [`Dataset::iter`].
    ///
    /// This preserves dataset-wide ordering under MPI.
    pub fn iter_global(&self) -> DatasetIter<'_> {
        self.iter()
    }

    /// Borrow the dataset metadata used for name lookups.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Clone the internal metadata handle for external consumers (e.g., language bindings).
    pub fn metadata_arc(&self) -> Arc<DatasetMetadata> {
        self.metadata.clone()
    }

    /// Names corresponding to stored four-momenta.
    pub fn p4_names(&self) -> &[String] {
        &self.metadata.p4_names
    }

    /// Names corresponding to stored auxiliary scalars.
    pub fn aux_names(&self) -> &[String] {
        &self.metadata.aux_names
    }

    /// Resolve the index of a four-momentum by name.
    pub fn p4_index(&self, name: &str) -> Option<usize> {
        self.metadata.p4_index(name)
    }

    /// Resolve the index of an auxiliary scalar by name.
    pub fn aux_index(&self, name: &str) -> Option<usize> {
        self.metadata.aux_index(name)
    }

    /// Borrow event data together with metadata-based helpers as an [`Event`] view.
    pub fn named_event(&self, index: usize) -> LadduResult<Event> {
        self.event(index)
    }

    /// Alias for [`Dataset::named_event`].
    pub fn named_event_global(&self, index: usize) -> LadduResult<Event> {
        self.named_event(index)
    }

    /// Retrieve a single event by index, returning `None` when out of range.
    pub fn get_event(&self, index: usize) -> Option<Event> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                let total = self.n_events();
                if index >= total {
                    return None;
                }
                return Some(fetch_event_mpi(self, index, &world, total));
            }
        }

        self.events.get(index).cloned()
    }

    /// Alias for [`Dataset::get_event`].
    ///
    /// This preserves the default global indexing semantics under MPI.
    pub fn get_event_global(&self, index: usize) -> Option<Event> {
        self.get_event(index)
    }

    /// Retrieve a single event by index.
    pub fn event(&self, index: usize) -> LadduResult<Event> {
        self.get_event(index).ok_or_else(|| {
            LadduError::Custom(format!(
                "Dataset index out of bounds: index {index}, length {}",
                self.n_events()
            ))
        })
    }

    /// Alias for [`Dataset::event`].
    ///
    /// This preserves the default global indexing semantics under MPI.
    pub fn event_global(&self, index: usize) -> LadduResult<Event> {
        self.event(index)
    }

    /// Retrieve a four-momentum by name for the event at `event_index`.
    pub fn p4_by_name(&self, event_index: usize, name: &str) -> Option<Vec4> {
        self.get_event(event_index).and_then(|event| event.p4(name))
    }

    /// Retrieve an auxiliary scalar by name for the event at `event_index`.
    pub fn aux_by_name(&self, event_index: usize, name: &str) -> Option<f64> {
        let idx = self.aux_index(name)?;
        self.get_event(event_index)
            .and_then(|event| event.aux.get(idx).copied())
    }

    /// Iterate over all local events as metadata-aware columnar views.
    pub fn for_each_named_event_local<F>(&self, op: F)
    where
        F: FnMut(usize, NamedEventView<'_>),
    {
        self.columnar.for_each_named_event_local(op);
    }

    /// Retrieve a metadata-aware columnar event view by local index.
    pub fn event_view(&self, event_index: usize) -> NamedEventView<'_> {
        self.columnar.event_view(event_index)
    }

    /// Get a reference to the [`EventData`] at the given index in the [`Dataset`] (non-MPI
    /// version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should use [`Dataset::event`] instead:
    ///
    /// ```ignore
    /// let ds: Dataset = Dataset::new(events);
    /// let event_0 = ds.event(0)?;
    /// ```
    pub fn index_local(&self, index: usize) -> &Event {
        &self.events[index]
    }

    #[cfg(feature = "mpi")]
    fn partition(
        events: Vec<Arc<EventData>>,
        world: &SimpleCommunicator,
    ) -> Vec<Vec<Arc<EventData>>> {
        let partition = world.partition(events.len());
        (0..partition.n_ranks())
            .map(|rank| {
                let range = partition.range_for_rank(rank);
                events[range.clone()].to_vec()
            })
            .collect()
    }
}

/// Iterator over a [`Dataset`].
pub enum DatasetIter<'a> {
    /// Iterator over locally available events.
    Local(std::slice::Iter<'a, Event>),
    #[cfg(feature = "mpi")]
    /// Iterator that fetches events across MPI ranks.
    Mpi(DatasetMpiIter<'a>),
}

impl<'a> Iterator for DatasetIter<'a> {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DatasetIter::Local(iter) => iter.next().cloned(),
            #[cfg(feature = "mpi")]
            DatasetIter::Mpi(iter) => iter.next(),
        }
    }
}

/// Owning iterator over a [`Dataset`].
pub enum DatasetIntoIter {
    /// Iterator over locally available events, consuming the dataset.
    Local(std::vec::IntoIter<Event>),
    #[cfg(feature = "mpi")]
    /// Iterator that fetches events across MPI ranks, consuming the dataset.
    Mpi(DatasetMpiIntoIter),
}

impl Iterator for DatasetIntoIter {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DatasetIntoIter::Local(iter) => iter.next(),
            #[cfg(feature = "mpi")]
            DatasetIntoIter::Mpi(iter) => iter.next(),
        }
    }
}

/// Iterator over a shared [`Arc<Dataset>`].
pub enum DatasetArcIter {
    /// Iterator over locally available events from a shared dataset handle.
    Local {
        /// Shared dataset handle.
        dataset: Arc<Dataset>,
        /// Next local event index to read.
        index: usize,
    },
    #[cfg(feature = "mpi")]
    /// Iterator that fetches events across MPI ranks from a shared dataset handle.
    Mpi(DatasetArcMpiIter),
}

impl Iterator for DatasetArcIter {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DatasetArcIter::Local { dataset, index } => {
                let event = dataset.events.get(*index).cloned();
                *index += 1;
                event
            }
            #[cfg(feature = "mpi")]
            DatasetArcIter::Mpi(iter) => iter.next(),
        }
    }
}

#[cfg(feature = "mpi")]
/// Iterator over a [`Dataset`] that fetches events across MPI ranks.
pub struct DatasetMpiIter<'a> {
    dataset: &'a Dataset,
    world: SimpleCommunicator,
    index: usize,
    total: usize,
    cursor: MpiEventChunkCursor,
}

#[cfg(feature = "mpi")]
#[derive(Debug, Clone)]
pub(crate) struct MpiEventChunkCursor {
    chunk_start: usize,
    chunk_size: usize,
    events: Vec<Event>,
}

#[cfg(feature = "mpi")]
pub(crate) fn resolve_mpi_event_fetch_chunk_size(total: usize) -> usize {
    let clamped_total = total.max(1);
    if let Some(raw) = std::env::var_os(MPI_EVENT_FETCH_CHUNK_SIZE_ENV) {
        if let Some(parsed) = raw.to_str().and_then(|value| value.parse::<usize>().ok()) {
            return parsed.max(1).min(clamped_total);
        }
    }
    DEFAULT_MPI_EVENT_FETCH_CHUNK_SIZE.min(clamped_total)
}

#[cfg(feature = "mpi")]
impl MpiEventChunkCursor {
    pub(crate) fn for_iteration(total: usize) -> Self {
        Self::new(resolve_mpi_event_fetch_chunk_size(total))
    }
}

#[cfg(feature = "mpi")]
impl MpiEventChunkCursor {
    pub(crate) fn new(chunk_size: usize) -> Self {
        Self {
            chunk_start: 0,
            chunk_size: chunk_size.max(1),
            events: Vec::new(),
        }
    }

    fn chunk_end(&self) -> usize {
        self.chunk_start + self.events.len()
    }

    fn contains(&self, global_index: usize) -> bool {
        global_index >= self.chunk_start && global_index < self.chunk_end()
    }

    pub(crate) fn event_for_dataset(
        &mut self,
        dataset: &Dataset,
        global_index: usize,
        world: &SimpleCommunicator,
        total: usize,
    ) -> Option<Event> {
        if global_index >= total {
            return None;
        }
        if !self.contains(global_index) {
            self.chunk_start = global_index;
            self.events =
                fetch_event_chunk_mpi(dataset, global_index, self.chunk_size, world, total);
        }
        self.events.get(global_index - self.chunk_start).cloned()
    }

    pub(crate) fn event_for_events(
        &mut self,
        events: &[Event],
        metadata: &Arc<DatasetMetadata>,
        global_index: usize,
        world: &SimpleCommunicator,
        total: usize,
    ) -> Option<Event> {
        if global_index >= total {
            return None;
        }
        if !self.contains(global_index) {
            self.chunk_start = global_index;
            self.events = fetch_event_chunk_mpi_from_events(
                events,
                metadata,
                global_index,
                self.chunk_size,
                world,
                total,
            );
        }
        self.events.get(global_index - self.chunk_start).cloned()
    }
}

#[cfg(feature = "mpi")]
impl<'a> Iterator for DatasetMpiIter<'a> {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        let event =
            self.cursor
                .event_for_dataset(self.dataset, self.index, &self.world, self.total);
        self.index += 1;
        event
    }
}

#[cfg(feature = "mpi")]
/// Iterator over a shared [`Arc<Dataset>`] that fetches events across MPI ranks.
pub struct DatasetArcMpiIter {
    dataset: Arc<Dataset>,
    world: SimpleCommunicator,
    index: usize,
    total: usize,
    cursor: MpiEventChunkCursor,
}

#[cfg(feature = "mpi")]
impl Iterator for DatasetArcMpiIter {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        let event =
            self.cursor
                .event_for_dataset(&self.dataset, self.index, &self.world, self.total);
        self.index += 1;
        event
    }
}

#[cfg(feature = "mpi")]
/// Owning iterator over a [`Dataset`] that fetches events across MPI ranks.
pub struct DatasetMpiIntoIter {
    events: Vec<Event>,
    metadata: Arc<DatasetMetadata>,
    world: SimpleCommunicator,
    index: usize,
    total: usize,
    cursor: MpiEventChunkCursor,
}

#[cfg(feature = "mpi")]
impl Iterator for DatasetMpiIntoIter {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.cursor.event_for_events(
            &self.events,
            &self.metadata,
            self.index,
            &self.world,
            self.total,
        );
        self.index += 1;
        event
    }
}

#[cfg(feature = "mpi")]
fn fetch_event_mpi(
    dataset: &Dataset,
    global_index: usize,
    world: &SimpleCommunicator,
    total: usize,
) -> Event {
    fetch_event_mpi_generic(
        global_index,
        total,
        world,
        &dataset.metadata,
        |local_index| dataset.index_local(local_index),
    )
}

#[cfg(feature = "mpi")]
pub(crate) fn fetch_event_chunk_mpi(
    dataset: &Dataset,
    start: usize,
    len: usize,
    world: &SimpleCommunicator,
    total: usize,
) -> Vec<Event> {
    fetch_event_chunk_mpi_generic(start, len, total, world, &dataset.metadata, |local_index| {
        dataset.index_local(local_index)
    })
}

#[cfg(feature = "mpi")]
fn fetch_event_chunk_mpi_from_events(
    events: &[Event],
    metadata: &Arc<DatasetMetadata>,
    start: usize,
    len: usize,
    world: &SimpleCommunicator,
    total: usize,
) -> Vec<Event> {
    fetch_event_chunk_mpi_generic(start, len, total, world, metadata, |local_index| {
        &events[local_index]
    })
}

#[cfg(feature = "mpi")]
fn fetch_event_mpi_generic<'a, F>(
    global_index: usize,
    total: usize,
    world: &SimpleCommunicator,
    metadata: &Arc<DatasetMetadata>,
    local_event: F,
) -> Event
where
    F: Fn(usize) -> &'a Event,
{
    let (owning_rank, local_index) = world.owner_of_global_index(global_index, total);
    let mut serialized_event_buffer_len: usize = 0;
    let mut serialized_event_buffer: Vec<u8> = Vec::default();
    if world.rank() == owning_rank {
        let event = local_event(local_index);
        serialized_event_buffer = bitcode::serialize(event.data()).unwrap();
        serialized_event_buffer_len = serialized_event_buffer.len();
    }
    world
        .process_at_rank(owning_rank)
        .broadcast_into(&mut serialized_event_buffer_len);
    if world.rank() != owning_rank {
        serialized_event_buffer = vec![0; serialized_event_buffer_len];
    }
    world
        .process_at_rank(owning_rank)
        .broadcast_into(&mut serialized_event_buffer);

    if world.rank() == owning_rank {
        local_event(local_index).clone()
    } else {
        let event: EventData = bitcode::deserialize(&serialized_event_buffer[..]).unwrap();
        Event::new(Arc::new(event), metadata.clone())
    }
}

#[cfg(feature = "mpi")]
#[allow(dead_code)]
fn fetch_event_chunk_mpi_generic<'a, F>(
    start: usize,
    len: usize,
    total: usize,
    world: &SimpleCommunicator,
    metadata: &Arc<DatasetMetadata>,
    local_event: F,
) -> Vec<Event>
where
    F: Fn(usize) -> &'a Event,
{
    if len == 0 || start >= total {
        return Vec::new();
    }

    let end = (start + len).min(total);
    let partition = world.partition(total);
    let local_range = partition.range_for_rank(world.rank() as usize);
    let owned_start = start.max(local_range.start);
    let owned_end = end.min(local_range.end);
    let local_indices = if owned_start < owned_end {
        (owned_start - local_range.start)..(owned_end - local_range.start)
    } else {
        0..0
    };

    let local_events: Vec<EventData> = local_indices
        .map(|local_index| local_event(local_index).data().clone())
        .collect();
    let local_event_count = local_events.len() as i32;

    let serialized_local = if local_events.is_empty() {
        Vec::new()
    } else {
        bitcode::serialize(&local_events).unwrap()
    };
    let local_byte_count = serialized_local.len() as i32;

    let mut gathered_event_counts = vec![0_i32; world.size() as usize];
    let mut gathered_byte_counts = vec![0_i32; world.size() as usize];
    world.all_gather_into(&local_event_count, &mut gathered_event_counts);
    world.all_gather_into(&local_byte_count, &mut gathered_byte_counts);

    let mut gathered_byte_displs = vec![0_i32; gathered_byte_counts.len()];
    for index in 1..gathered_byte_displs.len() {
        gathered_byte_displs[index] =
            gathered_byte_displs[index - 1] + gathered_byte_counts[index - 1];
    }
    let gathered_bytes = world.all_gather_with_counts(
        &serialized_local,
        &gathered_byte_counts,
        &gathered_byte_displs,
    );

    let mut events = Vec::with_capacity(end - start);
    for rank in 0..world.size() as usize {
        if gathered_event_counts[rank] == 0 {
            continue;
        }
        let byte_start = gathered_byte_displs[rank] as usize;
        let byte_end = byte_start + gathered_byte_counts[rank] as usize;
        let decoded: Vec<EventData> =
            bitcode::deserialize(&gathered_bytes[byte_start..byte_end]).unwrap();
        debug_assert_eq!(decoded.len(), gathered_event_counts[rank] as usize);
        events.extend(
            decoded
                .into_iter()
                .map(|event| Event::new(Arc::new(event), metadata.clone())),
        );
    }

    events
}

impl Dataset {
    #[cfg(feature = "mpi")]
    pub(crate) fn set_cached_global_event_count_from_world(&mut self, world: &SimpleCommunicator) {
        let local_count = self.n_events_local();
        let mut global_count = 0usize;
        world.all_reduce_into(
            &local_count,
            &mut global_count,
            mpi::collective::SystemOperation::sum(),
        );
        self.cached_global_event_count = global_count;
    }

    #[cfg(feature = "mpi")]
    pub(crate) fn set_cached_global_weighted_sum_from_world(&mut self, world: &SimpleCommunicator) {
        let mut weighted_sums = vec![0.0_f64; world.size() as usize];
        world.all_gather_into(&self.cached_local_weighted_sum, &mut weighted_sums);
        #[cfg(feature = "rayon")]
        {
            self.cached_global_weighted_sum = weighted_sums
                .into_par_iter()
                .parallel_sum_with_accumulator::<Klein<f64>>();
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.cached_global_weighted_sum = weighted_sums
                .into_iter()
                .sum_with_accumulator::<Klein<f64>>();
        }
    }

    fn columnar_from_wrapped_events(
        events: &[Event],
        metadata: Arc<DatasetMetadata>,
    ) -> LadduResult<DatasetStorage> {
        let n_events = events.len();
        let (n_p4, n_aux) = match events.first() {
            Some(first) => (first.p4s.len(), first.aux.len()),
            None => (metadata.p4_names.len(), metadata.aux_names.len()),
        };
        let mut p4 = (0..n_p4)
            .map(|_| ColumnarP4Column::with_capacity(n_events))
            .collect::<Vec<_>>();
        let mut aux = (0..n_aux)
            .map(|_| Vec::with_capacity(n_events))
            .collect::<Vec<_>>();
        let mut weights = Vec::with_capacity(n_events);
        for (event_index, event) in events.iter().enumerate() {
            if event.p4s.len() != n_p4 || event.aux.len() != n_aux {
                return Err(LadduError::Custom(format!(
                    "Ragged dataset shape at event {event_index}: expected ({n_p4} p4, {n_aux} aux), got ({} p4, {} aux)",
                    event.p4s.len(),
                    event.aux.len()
                )));
            }
            for (column, value) in p4.iter_mut().zip(event.p4s.iter()) {
                column.push(*value);
            }
            for (column, value) in aux.iter_mut().zip(event.aux.iter()) {
                column.push(*value);
            }
            weights.push(event.weight);
        }
        Ok(DatasetStorage {
            metadata,
            p4,
            aux,
            weights,
        })
    }

    /// Create a new [`Dataset`] from a list of [`EventData`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::new`] instead.
    pub fn new_local(events: Vec<Arc<EventData>>, metadata: Arc<DatasetMetadata>) -> Self {
        let wrapped_events = events
            .into_iter()
            .map(|event| Event::new(event, metadata.clone()))
            .collect::<Vec<_>>();
        #[cfg(feature = "mpi")]
        let local_count = wrapped_events.len();
        let columnar = Self::columnar_from_wrapped_events(&wrapped_events, metadata.clone())
            .expect("Dataset requires rectangular p4/aux columns for canonical columnar storage");
        #[cfg(feature = "rayon")]
        let local_weighted_sum = columnar
            .weights
            .par_iter()
            .copied()
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let local_weighted_sum = columnar
            .weights
            .iter()
            .copied()
            .sum_with_accumulator::<Klein<f64>>();
        Dataset {
            events: wrapped_events,
            columnar,
            metadata,
            cached_local_weighted_sum: local_weighted_sum,
            #[cfg(feature = "mpi")]
            cached_global_event_count: local_count,
            #[cfg(feature = "mpi")]
            cached_global_weighted_sum: local_weighted_sum,
        }
    }

    /// Create a new [`Dataset`] from a list of [`EventData`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::new`] instead.
    #[cfg(feature = "mpi")]
    pub fn new_mpi(
        events: Vec<Arc<EventData>>,
        metadata: Arc<DatasetMetadata>,
        world: &SimpleCommunicator,
    ) -> Self {
        let partitions = Dataset::partition(events, world);
        let local: Vec<Event> = partitions[world.rank() as usize]
            .iter()
            .cloned()
            .map(|event| Event::new(event, metadata.clone()))
            .collect();
        let columnar = Self::columnar_from_wrapped_events(&local, metadata.clone())
            .expect("Dataset requires rectangular p4/aux columns for canonical columnar storage");
        #[cfg(feature = "rayon")]
        let local_weighted_sum = columnar
            .weights
            .par_iter()
            .copied()
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        let local_weighted_sum = columnar
            .weights
            .iter()
            .copied()
            .sum_with_accumulator::<Klein<f64>>();
        let mut dataset = Dataset {
            events: local,
            columnar,
            metadata,
            cached_local_weighted_sum: local_weighted_sum,
            cached_global_event_count: 0,
            cached_global_weighted_sum: local_weighted_sum,
        };
        dataset.set_cached_global_event_count_from_world(world);
        dataset.set_cached_global_weighted_sum_from_world(world);
        dataset
    }

    /// Create a new [`Dataset`] from a list of [`EventData`].
    ///
    /// This method is prefered for external use because it contains proper MPI construction
    /// methods. Constructing a [`Dataset`] manually is possible, but may cause issues when
    /// interfacing with MPI and should be avoided unless you know what you are doing.
    pub fn new(events: Vec<Arc<EventData>>) -> Self {
        Dataset::new_with_metadata(events, Arc::new(DatasetMetadata::default()))
    }

    /// Create a dataset with explicit metadata for name-based lookups.
    /// Create a dataset with explicit metadata for name-based lookups.
    pub fn new_with_metadata(events: Vec<Arc<EventData>>, metadata: Arc<DatasetMetadata>) -> Self {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return Dataset::new_mpi(events, metadata, &world);
            }
        }
        Dataset::new_local(events, metadata)
    }

    /// The number of [`EventData`]s in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events`] instead.
    pub fn n_events_local(&self) -> usize {
        self.columnar.n_events()
    }

    /// The number of [`EventData`]s in the [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events`] instead.
    #[cfg(feature = "mpi")]
    pub fn n_events_mpi(&self, _world: &SimpleCommunicator) -> usize {
        self.cached_global_event_count
    }

    /// The number of [`EventData`]s in the [`Dataset`].
    pub fn n_events(&self) -> usize {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.n_events_mpi(&world);
            }
        }
        self.n_events_local()
    }

    /// Alias for [`Dataset::n_events`].
    ///
    /// This returns the global event count under MPI.
    pub fn n_events_global(&self) -> usize {
        self.n_events()
    }
}

impl Dataset {
    /// Extract a list of weights over each [`EventData`] in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::weights`] instead.
    pub fn weights_local(&self) -> Vec<f64> {
        self.columnar.weights.clone()
    }

    /// Extract a list of weights over each [`EventData`] in the [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::weights`] instead.
    #[cfg(feature = "mpi")]
    pub fn weights_mpi(&self, world: &SimpleCommunicator) -> Vec<f64> {
        let local_weights = self.weights_local();
        let n_events = self.n_events();
        let mut buffer: Vec<f64> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            // NOTE: gather is required because this API returns full global event weights.
            // Use all-reduce only for scalar/vector aggregate values.
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_weights, &mut partitioned_buffer);
        }
        buffer
    }

    /// Extract a list of weights over each [`EventData`] in the [`Dataset`].
    pub fn weights(&self) -> Vec<f64> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.weights_mpi(&world);
            }
        }
        self.weights_local()
    }

    /// Alias for [`Dataset::weights`].
    ///
    /// This returns the global weight vector in dataset order under MPI.
    pub fn weights_global(&self) -> Vec<f64> {
        self.weights()
    }

    /// Returns the sum of the weights for each [`EventData`] in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events_weighted`] instead.
    pub fn n_events_weighted_local(&self) -> f64 {
        self.cached_local_weighted_sum
    }
    /// Returns the sum of the weights for each [`EventData`] in the [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events_weighted`] instead.
    #[cfg(feature = "mpi")]
    pub fn n_events_weighted_mpi(&self, _world: &SimpleCommunicator) -> f64 {
        self.cached_global_weighted_sum
    }

    /// Returns the sum of the weights for each [`EventData`] in the [`Dataset`].
    pub fn n_events_weighted(&self) -> f64 {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.n_events_weighted_mpi(&world);
            }
        }
        self.n_events_weighted_local()
    }

    /// Alias for [`Dataset::n_events_weighted`].
    ///
    /// This returns the global weighted event count under MPI.
    pub fn n_events_weighted_global(&self) -> f64 {
        self.n_events_weighted()
    }

    /// Generate a new dataset with the same length by resampling the events in the original datset
    /// with replacement. This can be used to perform error analysis via the bootstrap method. (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::bootstrap`] instead.
    pub fn bootstrap_local(&self, seed: usize) -> Arc<Dataset> {
        let mut rng = fastrand::Rng::with_seed(seed as u64);
        let mut indices: Vec<usize> = (0..self.n_events())
            .map(|_| rng.usize(0..self.n_events()))
            .collect::<Vec<usize>>();
        indices.sort();
        #[cfg(feature = "rayon")]
        let bootstrapped_events: Vec<Arc<EventData>> = indices
            .into_par_iter()
            .map(|idx| self.events[idx].data_arc())
            .collect();
        #[cfg(not(feature = "rayon"))]
        let bootstrapped_events: Vec<Arc<EventData>> = indices
            .into_iter()
            .map(|idx| self.events[idx].data_arc())
            .collect();
        Arc::new(Dataset::new_with_metadata(
            bootstrapped_events,
            self.metadata.clone(),
        ))
    }

    /// Generate a new dataset with the same length by resampling the events in the original datset
    /// with replacement. This can be used to perform error analysis via the bootstrap method. (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::bootstrap`] instead.
    #[cfg(feature = "mpi")]
    pub fn bootstrap_mpi(&self, seed: usize, world: &SimpleCommunicator) -> Arc<Dataset> {
        let n_events = self.n_events();
        let mut indices: Vec<usize> = vec![0; n_events];
        if world.is_root() {
            let mut rng = fastrand::Rng::with_seed(seed as u64);
            indices = (0..n_events)
                .map(|_| rng.usize(0..n_events))
                .collect::<Vec<usize>>();
            indices.sort();
        }
        world.process_at_root().broadcast_into(&mut indices);
        let local_indices: Vec<usize> = indices
            .into_iter()
            .filter_map(|idx| {
                let (owning_rank, local_index) = world.owner_of_global_index(idx, n_events);
                if world.rank() == owning_rank {
                    Some(local_index)
                } else {
                    None
                }
            })
            .collect();
        // `local_indices` only contains indices owned by the current rank, translating them into
        // local indices on the events vector.
        #[cfg(feature = "rayon")]
        let bootstrapped_events: Vec<Arc<EventData>> = local_indices
            .into_par_iter()
            .map(|idx| self.events[idx].data_arc())
            .collect();
        #[cfg(not(feature = "rayon"))]
        let bootstrapped_events: Vec<Arc<EventData>> = local_indices
            .into_iter()
            .map(|idx| self.events[idx].data_arc())
            .collect();
        Arc::new(Dataset::new_with_metadata(
            bootstrapped_events,
            self.metadata.clone(),
        ))
    }

    /// Generate a new dataset with the same length by resampling the events in the original datset
    /// with replacement. This can be used to perform error analysis via the bootstrap method.
    pub fn bootstrap(&self, seed: usize) -> Arc<Dataset> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.bootstrap_mpi(seed, &world);
            }
        }
        self.bootstrap_local(seed)
    }

    /// Filter the [`Dataset`] by a given [`VariableExpression`], selecting events for which
    /// the expression returns `true`.
    pub fn filter(&self, expression: &VariableExpression) -> LadduResult<Arc<Dataset>> {
        let compiled = expression.compile(&self.metadata)?;
        #[cfg(feature = "rayon")]
        let filtered_events: Vec<Arc<EventData>> = (0..self.n_events_local())
            .into_par_iter()
            .filter_map(|event_index| {
                let event = self.event_view(event_index);
                compiled
                    .evaluate(&event)
                    .then(|| self.events[event_index].data_arc())
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let filtered_events: Vec<Arc<EventData>> = (0..self.n_events_local())
            .into_iter()
            .filter_map(|event_index| {
                let event = self.event_view(event_index);
                compiled
                    .evaluate(&event)
                    .then(|| self.events[event_index].data_arc())
            })
            .collect();
        Ok(Arc::new(Dataset::new_with_metadata(
            filtered_events,
            self.metadata.clone(),
        )))
    }

    /// Bin a [`Dataset`] by the value of the given [`Variable`] into a number of `bins` within the
    /// given `range`.
    pub fn bin_by<V>(
        &self,
        mut variable: V,
        bins: usize,
        range: (f64, f64),
    ) -> LadduResult<BinnedDataset>
    where
        V: Variable,
    {
        variable.bind(self.metadata())?;
        let bin_width = (range.1 - range.0) / bins as f64;
        let bin_edges = get_bin_edges(bins, range);
        let variable = variable;
        #[cfg(feature = "rayon")]
        let evaluated: Vec<(usize, Arc<EventData>)> = (0..self.n_events_local())
            .into_par_iter()
            .filter_map(|event| {
                let value = variable.value(&self.event_view(event));
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, self.events[event].data_arc()))
                } else {
                    None
                }
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let evaluated: Vec<(usize, Arc<EventData>)> = (0..self.n_events_local())
            .into_iter()
            .filter_map(|event| {
                let value = variable.value(&self.event_view(event));
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, self.events[event].data_arc()))
                } else {
                    None
                }
            })
            .collect();
        let mut binned_events: Vec<Vec<Arc<EventData>>> = vec![Vec::default(); bins];
        for (bin_index, event) in evaluated {
            binned_events[bin_index].push(event.clone());
        }
        #[cfg(feature = "rayon")]
        let datasets: Vec<Arc<Dataset>> = binned_events
            .into_par_iter()
            .map(|events| Arc::new(Dataset::new_with_metadata(events, self.metadata.clone())))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let datasets: Vec<Arc<Dataset>> = binned_events
            .into_iter()
            .map(|events| Arc::new(Dataset::new_with_metadata(events, self.metadata.clone())))
            .collect();
        Ok(BinnedDataset {
            datasets,
            edges: bin_edges,
        })
    }

    /// Boost all the four-momenta in all [`EventData`]s to the rest frame of the given set of
    /// four-momenta identified by name.
    pub fn boost_to_rest_frame_of<S>(&self, names: &[S]) -> Arc<Dataset>
    where
        S: AsRef<str>,
    {
        let mut indices: Vec<usize> = Vec::new();
        for name in names {
            let name_ref = name.as_ref();
            if let Some(selection) = self.metadata.p4_selection(name_ref) {
                indices.extend_from_slice(selection.indices());
            } else {
                panic!("Unknown particle name '{name}'", name = name_ref);
            }
        }
        #[cfg(feature = "rayon")]
        let boosted_events: Vec<Arc<EventData>> = self
            .events
            .par_iter()
            .map(|event| Arc::new(event.data().boost_to_rest_frame_of(&indices)))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let boosted_events: Vec<Arc<EventData>> = self
            .events
            .iter()
            .map(|event| Arc::new(event.data().boost_to_rest_frame_of(&indices)))
            .collect();
        Arc::new(Dataset::new_with_metadata(
            boosted_events,
            self.metadata.clone(),
        ))
    }
    /// Evaluate a [`Variable`] on every event in the [`Dataset`].
    pub fn evaluate<V: Variable>(&self, variable: &V) -> LadduResult<Vec<f64>> {
        variable.value_on(self)
    }
}

#[cfg(test)]
pub(crate) use super::io::write_parquet_storage;
pub use super::io::{
    read_parquet, read_parquet_chunks, read_parquet_chunks_with_options, read_root, write_parquet,
    write_root,
};
#[cfg(test)]
pub(crate) use super::io::{read_parquet_storage, read_root_storage};

impl_op_ex!(+ |a: &Dataset, b: &Dataset| -> Dataset {
    debug_assert_eq!(a.metadata.p4_names, b.metadata.p4_names);
    debug_assert_eq!(a.metadata.aux_names, b.metadata.aux_names);
    let events = a
        .events
        .iter()
        .chain(b.events.iter())
        .map(Event::data_arc)
        .collect::<Vec<_>>();
    Dataset::new_with_metadata(events, a.metadata.clone())
});

/// Incrementally builds a [`Dataset`] from chunked dataset reads.
#[derive(Default)]
pub struct DatasetChunkBuilder {
    metadata: Option<Arc<DatasetMetadata>>,
    events: Vec<Arc<EventData>>,
}

impl DatasetChunkBuilder {
    /// Create an empty chunk builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a dataset chunk.
    pub fn push_chunk(&mut self, chunk: &Dataset) -> LadduResult<()> {
        if let Some(existing) = &self.metadata {
            if existing.p4_names != chunk.metadata.p4_names
                || existing.aux_names != chunk.metadata.aux_names
            {
                return Err(LadduError::Custom(
                    "Dataset chunk metadata does not match previous chunks".to_string(),
                ));
            }
        } else {
            self.metadata = Some(chunk.metadata.clone());
        }
        self.events
            .extend(chunk.events_local().iter().map(Event::data_arc));
        Ok(())
    }

    /// Finish building a dataset from all received chunks.
    pub fn finish(self) -> Arc<Dataset> {
        let metadata = self
            .metadata
            .unwrap_or_else(|| Arc::new(DatasetMetadata::empty()));
        Arc::new(Dataset::new_with_metadata(self.events, metadata))
    }
}

/// Fold over chunked datasets without materializing a full dataset.
pub fn try_fold_dataset_chunks<I, T, F>(chunks: I, init: T, mut op: F) -> LadduResult<T>
where
    I: IntoIterator<Item = LadduResult<Arc<Dataset>>>,
    F: FnMut(T, &Dataset) -> LadduResult<T>,
{
    let mut acc = init;
    for chunk in chunks {
        let chunk = chunk?;
        acc = op(acc, &chunk)?;
    }
    Ok(acc)
}

/// Options for reading a [`Dataset`] from a file.
///
/// # See Also
/// [`read_parquet`], [`read_root`]
#[derive(Default, Clone)]
pub struct DatasetReadOptions {
    /// Particle names to read from the data file.
    pub p4_names: Option<Vec<String>>,
    /// Auxiliary scalar names to read from the data file.
    pub aux_names: Option<Vec<String>>,
    /// Name of the tree to read when loading ROOT files. When absent and the file contains a
    /// single tree, it will be selected automatically.
    pub tree: Option<String>,
    /// Optional aliases mapping logical names to selections of four-momenta.
    pub aliases: IndexMap<String, P4Selection>,
    /// Preferred chunk size for chunked read APIs.
    pub chunk_size: Option<usize>,
}

/// Precision for writing floating-point columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FloatPrecision {
    /// 32-bit floats.
    F32,
    /// 64-bit floats.
    #[default]
    F64,
}

/// Options for writing a [`Dataset`] to disk.
#[derive(Clone, Debug)]
pub struct DatasetWriteOptions {
    /// Number of events to include in each batch when writing.
    pub batch_size: usize,
    /// Floating-point precision to use for persisted columns.
    pub precision: FloatPrecision,
    /// Tree name to use when writing ROOT files.
    pub tree: Option<String>,
}

impl Default for DatasetWriteOptions {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_WRITE_BATCH_SIZE,
            precision: FloatPrecision::default(),
            tree: None,
        }
    }
}

impl DatasetWriteOptions {
    /// Override the batch size used for writing; defaults to 10_000.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Select the floating-point precision for persisted columns.
    pub fn precision(mut self, precision: FloatPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set the ROOT tree name (defaults to \"events\").
    pub fn tree<S: Into<String>>(mut self, name: S) -> Self {
        self.tree = Some(name.into());
        self
    }
}
impl DatasetReadOptions {
    /// Create a new [`Default`] set of [`DatasetReadOptions`].
    pub fn new() -> Self {
        Self::default()
    }

    /// If provided, the specified particles will be read from the data file (assuming columns with
    /// required suffixes are present, i.e. `<particle>_px`, `<particle>_py`, `<particle>_pz`, and `<particle>_e`). Otherwise, all valid columns with these suffixes will be read.
    pub fn p4_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.p4_names = Some(names.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// If provided, the specified columns will be read as auxiliary scalars. Otherwise, all valid
    /// columns which do not satisfy the conditions required to be read as four-momenta will be
    /// used.
    pub fn aux_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.aux_names = Some(names.into_iter().map(|s| s.as_ref().to_string()).collect());
        self
    }

    /// Select the tree to read when opening ROOT files.
    pub fn tree<S>(mut self, name: S) -> Self
    where
        S: AsRef<str>,
    {
        self.tree = Some(name.as_ref().to_string());
        self
    }

    /// Register an alias for one or more existing four-momenta.
    pub fn alias<N, S>(mut self, name: N, selection: S) -> Self
    where
        N: Into<String>,
        S: IntoP4Selection,
    {
        self.aliases.insert(name.into(), selection.into_selection());
        self
    }

    /// Register multiple aliases for four-momenta selections.
    pub fn aliases<I, N, S>(mut self, aliases: I) -> Self
    where
        I: IntoIterator<Item = (N, S)>,
        N: Into<String>,
        S: IntoP4Selection,
    {
        for (name, selection) in aliases {
            self = self.alias(name, selection);
        }
        self
    }

    /// Set the chunk size used by chunked read APIs; values below 1 are clamped to 1.
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size.max(1));
        self
    }

    pub(crate) fn resolve_metadata(
        &self,
        detected_p4_names: Vec<String>,
        detected_aux_names: Vec<String>,
    ) -> LadduResult<Arc<DatasetMetadata>> {
        let p4_names_vec = self.p4_names.clone().unwrap_or(detected_p4_names);
        let aux_names_vec = self.aux_names.clone().unwrap_or(detected_aux_names);

        let mut metadata = DatasetMetadata::new(p4_names_vec, aux_names_vec)?;
        if !self.aliases.is_empty() {
            metadata.add_p4_aliases(self.aliases.clone())?;
        }
        Ok(Arc::new(metadata))
    }
}

const DEFAULT_WRITE_BATCH_SIZE: usize = 10_000;
pub(crate) const DEFAULT_READ_CHUNK_SIZE: usize = 10_000;

/// A list of [`Dataset`]s formed by binning [`EventData`] by some [`Variable`].
pub struct BinnedDataset {
    datasets: Vec<Arc<Dataset>>,
    edges: Vec<f64>,
}

impl Index<usize> for BinnedDataset {
    type Output = Arc<Dataset>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.datasets[index]
    }
}

impl IndexMut<usize> for BinnedDataset {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.datasets[index]
    }
}

impl Deref for BinnedDataset {
    type Target = Vec<Arc<Dataset>>;

    fn deref(&self) -> &Self::Target {
        &self.datasets
    }
}

impl DerefMut for BinnedDataset {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.datasets
    }
}

impl BinnedDataset {
    /// The number of bins in the [`BinnedDataset`].
    pub fn n_bins(&self) -> usize {
        self.datasets.len()
    }

    /// Returns a list of the bin edges that were used to form the [`BinnedDataset`].
    pub fn edges(&self) -> Vec<f64> {
        self.edges.clone()
    }

    /// Returns the range that was used to form the [`BinnedDataset`].
    pub fn range(&self) -> (f64, f64) {
        (self.edges[0], self.edges[self.n_bins()])
    }
}
