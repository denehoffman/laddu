use std::{
    borrow::Cow,
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
    event::{test_event, ColumnarP4Column, DatasetStorage, Event, EventData, OwnedEvent},
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

const TEST_P4_NAMES: &[&str] = &["beam", "proton", "kshort1", "kshort2"];
const TEST_AUX_NAMES: &[&str] = &["pol_magnitude", "pol_angle"];

fn local_weighted_sum(weights: &[f64]) -> f64 {
    #[cfg(feature = "rayon")]
    {
        weights
            .par_iter()
            .copied()
            .parallel_sum_with_accumulator::<Klein<f64>>()
    }
    #[cfg(not(feature = "rayon"))]
    {
        weights.iter().copied().sum_with_accumulator::<Klein<f64>>()
    }
}

/// A dataset that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitude::Amplitude). This particular dataset contains a single
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
    pub(crate) columnar: Arc<DatasetStorage>,
    rows: RowSelection,
    pub(crate) metadata: Arc<DatasetMetadata>,
    pub(crate) cached_local_weighted_sum: f64,
    #[cfg(feature = "mpi")]
    pub(crate) cached_global_event_count: usize,
    #[cfg(feature = "mpi")]
    pub(crate) cached_global_weighted_sum: f64,
    #[cfg(feature = "mpi")]
    pub(crate) mpi_layout: Option<MpiDatasetLayout>,
}

#[derive(Debug, Clone)]
enum RowSelection {
    Identity,
    Indices(Arc<[usize]>),
}

impl RowSelection {
    fn len(&self, storage_len: usize) -> usize {
        match self {
            Self::Identity => storage_len,
            Self::Indices(indices) => indices.len(),
        }
    }

    const fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }

    fn physical_index(&self, logical_index: usize) -> usize {
        match self {
            Self::Identity => logical_index,
            Self::Indices(indices) => indices[logical_index],
        }
    }
}

#[cfg(feature = "mpi")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MpiDatasetLayout {
    Canonical,
    RoundRobin,
    Derived,
}

#[cfg(feature = "mpi")]
impl MpiDatasetLayout {
    fn owner_of(
        self,
        global_index: usize,
        total: usize,
        local_len: usize,
        world: &SimpleCommunicator,
    ) -> (i32, usize) {
        match self {
            Self::Canonical => world.owner_of_global_index(global_index, total),
            Self::RoundRobin => {
                let size = world.size() as usize;
                ((global_index % size) as i32, global_index / size)
            }
            Self::Derived => {
                let counts = gather_local_event_counts(local_len, world);
                let mut start = 0usize;
                for (rank, count) in counts.into_iter().enumerate() {
                    let end = start + count;
                    if global_index < end {
                        return (rank as i32, global_index - start);
                    }
                    start = end;
                }
                debug_assert!(
                    global_index < total,
                    "validated derived global event index should be in range"
                );
                (world.rank(), 0)
            }
        }
    }

    fn local_range(
        self,
        total: usize,
        local_len: usize,
        world: &SimpleCommunicator,
    ) -> std::ops::Range<usize> {
        match self {
            Self::Canonical => world.partition(total).range_for_rank(world.rank() as usize),
            Self::RoundRobin => 0..local_len_for_round_robin(total, world),
            Self::Derived => {
                let counts = gather_local_event_counts(local_len, world);
                let start = counts
                    .iter()
                    .take(world.rank() as usize)
                    .copied()
                    .sum::<usize>();
                start..start + counts[world.rank() as usize]
            }
        }
    }

    fn local_indices_for_range(
        self,
        start: usize,
        end: usize,
        total: usize,
        local_len: usize,
        world: &SimpleCommunicator,
    ) -> Vec<usize> {
        match self {
            Self::Canonical => {
                let local_range = self.local_range(total, local_len, world);
                let owned_start = start.max(local_range.start);
                let owned_end = end.min(local_range.end);
                if owned_start < owned_end {
                    (owned_start - local_range.start..owned_end - local_range.start).collect()
                } else {
                    Vec::new()
                }
            }
            Self::RoundRobin => {
                let rank = world.rank() as usize;
                let size = world.size() as usize;
                (start..end)
                    .filter_map(|global_index| {
                        if global_index % size == rank {
                            Some(global_index / size)
                        } else {
                            None
                        }
                    })
                    .filter(|local_index| *local_index < local_len)
                    .collect()
            }
            Self::Derived => {
                let counts = gather_local_event_counts(local_len, world);
                let local_start = counts
                    .iter()
                    .take(world.rank() as usize)
                    .copied()
                    .sum::<usize>();
                let local_end = local_start + local_len;
                let owned_start = start.max(local_start);
                let owned_end = end.min(local_end);
                if owned_start < owned_end {
                    (owned_start - local_start..owned_end - local_start).collect()
                } else {
                    Vec::new()
                }
            }
        }
    }
}

#[cfg(feature = "mpi")]
fn gather_local_event_counts(local_len: usize, world: &SimpleCommunicator) -> Vec<usize> {
    let mut counts = vec![0usize; world.size() as usize];
    world.all_gather_into(&local_len, &mut counts);
    counts
}

#[cfg(feature = "mpi")]
fn local_len_for_round_robin(total: usize, world: &SimpleCommunicator) -> usize {
    let rank = world.rank() as usize;
    let size = world.size() as usize;
    if total <= rank {
        0
    } else {
        (total - 1 - rank) / size + 1
    }
}

fn shared_dataset_iter(dataset: Arc<Dataset>) -> DatasetArcIter {
    #[cfg(feature = "mpi")]
    {
        if let Some(world) = crate::mpi::get_world() {
            if let Some(layout) = dataset.mpi_layout {
                let total = dataset.n_events();
                return DatasetArcIter::Mpi(DatasetArcMpiIter {
                    dataset,
                    world,
                    index: 0,
                    total,
                    cursor: MpiEventChunkCursor::for_iteration(total),
                    layout,
                });
            }
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
    fn from_columnar_storage(
        columnar: DatasetStorage,
        metadata: Arc<DatasetMetadata>,
        rows: RowSelection,
    ) -> Self {
        #[cfg(feature = "mpi")]
        let local_count = rows.len(columnar.n_events());
        let local_weighted_sum = Self::weighted_sum_for_rows(&columnar, &rows);
        Dataset {
            columnar: Arc::new(columnar),
            rows,
            metadata,
            cached_local_weighted_sum: local_weighted_sum,
            #[cfg(feature = "mpi")]
            cached_global_event_count: local_count,
            #[cfg(feature = "mpi")]
            cached_global_weighted_sum: local_weighted_sum,
            #[cfg(feature = "mpi")]
            mpi_layout: None,
        }
    }

    fn weighted_sum_for_rows(columnar: &DatasetStorage, rows: &RowSelection) -> f64 {
        match rows {
            RowSelection::Identity => local_weighted_sum(&columnar.weights),
            RowSelection::Indices(indices) => {
                #[cfg(feature = "rayon")]
                {
                    indices
                        .par_iter()
                        .map(|index| columnar.weight(*index))
                        .parallel_sum_with_accumulator::<Klein<f64>>()
                }
                #[cfg(not(feature = "rayon"))]
                {
                    indices
                        .iter()
                        .map(|index| columnar.weight(*index))
                        .sum_with_accumulator::<Klein<f64>>()
                }
            }
        }
    }

    fn indexed_local_view<I>(&self, indices: I) -> Arc<Dataset>
    where
        I: IntoIterator<Item = usize>,
    {
        let rows = RowSelection::Indices(indices.into_iter().collect::<Vec<_>>().into());
        let local_weighted_sum = Self::weighted_sum_for_rows(&self.columnar, &rows);
        let dataset = Dataset {
            columnar: self.columnar.clone(),
            rows,
            metadata: self.metadata.clone(),
            cached_local_weighted_sum: local_weighted_sum,
            #[cfg(feature = "mpi")]
            cached_global_event_count: 0,
            #[cfg(feature = "mpi")]
            cached_global_weighted_sum: local_weighted_sum,
            #[cfg(feature = "mpi")]
            mpi_layout: self.mpi_layout,
        };
        #[cfg(feature = "mpi")]
        {
            let mut dataset = dataset;
            if dataset.mpi_layout.is_some() {
                dataset.mpi_layout = Some(MpiDatasetLayout::Derived);
                if let Some(world) = crate::mpi::get_world() {
                    dataset.set_cached_global_event_count_from_world(&world);
                    dataset.set_cached_global_weighted_sum_from_world(&world);
                }
            }
            Arc::new(dataset)
        }
        #[cfg(not(feature = "mpi"))]
        {
            Arc::new(dataset)
        }
    }

    fn ensure_mutable_storage(&self, operation: &str) -> LadduResult<()> {
        if self.rows.is_identity() {
            Ok(())
        } else {
            Err(LadduError::Custom(format!(
                "Cannot {operation} on a filtered or bootstrapped dataset view; materialize it first"
            )))
        }
    }

    /// Iterate over locally stored events as borrowed [`Event`] views.
    pub fn events_local(&self) -> impl Iterator<Item = Event<'_>> {
        DatasetViewIter {
            dataset: self,
            index: 0,
        }
    }

    /// Iterate over all events using the default global iteration semantics.
    ///
    /// When MPI is enabled, the iterator is ordered like [`Dataset::event_global`] and may
    /// fetch remotely owned events in chunks.
    pub fn events_global(&self) -> DatasetGlobalIter<'_> {
        let total = self.n_events();
        #[cfg(feature = "mpi")]
        {
            if let (Some(world), Some(layout)) = (crate::mpi::get_world(), self.mpi_layout) {
                return DatasetGlobalIter {
                    dataset: self,
                    index: 0,
                    total,
                    world: Some(world),
                    cursor: Some(MpiEventChunkCursor::for_iteration(total)),
                    layout: Some(layout),
                };
            }
        }
        DatasetGlobalIter {
            dataset: self,
            index: 0,
            total,
            #[cfg(feature = "mpi")]
            world: None,
            #[cfg(feature = "mpi")]
            cursor: None,
            #[cfg(feature = "mpi")]
            layout: None,
        }
    }

    fn refresh_local_weight_cache(&mut self) {
        self.cached_local_weighted_sum = Self::weighted_sum_for_rows(&self.columnar, &self.rows);
        #[cfg(feature = "mpi")]
        {
            self.cached_global_weighted_sum = self.cached_local_weighted_sum;
            self.cached_global_event_count = self.n_events_local();
            if self.mpi_layout.is_some() {
                if let Some(world) = crate::mpi::get_world() {
                    self.set_cached_global_event_count_from_world(&world);
                    self.set_cached_global_weighted_sum_from_world(&world);
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn clear_events_local(&mut self) {
        self.ensure_mutable_storage("clear local events")
            .expect("test datasets should be materialized");
        let columnar = Arc::make_mut(&mut self.columnar);
        for column in &mut columnar.p4 {
            column.px.clear();
            column.py.clear();
            column.pz.clear();
            column.e.clear();
        }
        for column in &mut columnar.aux {
            column.clear();
        }
        columnar.weights.clear();
        self.refresh_local_weight_cache();
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

    fn event_global_opt(&self, index: usize) -> LadduResult<Option<OwnedEvent>> {
        #[cfg(feature = "mpi")]
        {
            if let (Some(world), Some(_)) = (crate::mpi::get_world(), self.mpi_layout) {
                let total = self.n_events();
                if index >= total {
                    return Ok(None);
                }
                return self.fetch_event_mpi(index, &world, total).map(Some);
            }
        }

        Ok((index < self.n_events_local())
            .then(|| OwnedEvent::new(self.event_data_arc_local(index), self.metadata.clone())))
    }

    /// Retrieve a single owned event by global index.
    pub fn event_global(&self, index: usize) -> LadduResult<OwnedEvent> {
        self.event_global_opt(index)?.ok_or_else(|| {
            LadduError::Custom(format!(
                "Dataset index out of bounds: index {index}, length {}",
                self.n_events()
            ))
        })
    }

    /// Borrow a locally stored event by local rank index.
    pub fn event_local(&self, event_index: usize) -> LadduResult<Event<'_>> {
        if event_index >= self.n_events_local() {
            return Err(LadduError::Custom(format!(
                "Dataset local index out of bounds: index {event_index}, length {}",
                self.n_events_local()
            )));
        }
        Ok(self.event_view(event_index))
    }

    /// Retrieve a four-momentum by name for the event at `event_index`.
    pub fn p4_by_name(&self, event_index: usize, name: &str) -> Option<Vec4> {
        self.event_global_opt(event_index)
            .ok()
            .flatten()
            .and_then(|event| event.p4(name))
    }

    /// Retrieve an auxiliary scalar by name for the event at `event_index`.
    pub fn aux_by_name(&self, event_index: usize, name: &str) -> Option<f64> {
        let idx = self.aux_index(name)?;
        self.event_global_opt(event_index)
            .ok()
            .flatten()
            .and_then(|event| event.aux.get(idx).copied())
    }

    pub(crate) fn event_view(&self, event_index: usize) -> Event<'_> {
        self.columnar
            .event_view(self.rows.physical_index(event_index))
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
    /// let event_0 = ds.event_global(0)?;
    /// ```
    pub(crate) fn event_data_arc_local(&self, index: usize) -> Arc<EventData> {
        Arc::new(self.columnar.event_data(self.rows.physical_index(index)))
    }

    pub(crate) fn local_event_data_arcs(&self) -> Vec<Arc<EventData>> {
        (0..self.n_events_local())
            .map(|index| self.event_data_arc_local(index))
            .collect()
    }

    pub(crate) fn local_storage_for_export(&self) -> LadduResult<Cow<'_, DatasetStorage>> {
        if self.rows.is_identity() {
            Ok(Cow::Borrowed(self.columnar.as_ref()))
        } else {
            Ok(Cow::Owned(Self::columnar_from_events(
                &self.local_event_data_arcs(),
                self.metadata.clone(),
            )?))
        }
    }

    pub(crate) fn local_weight_cache_key(&self) -> (usize, usize) {
        match &self.rows {
            RowSelection::Identity => (
                self.columnar.weights.as_ptr() as usize,
                self.n_events_local(),
            ),
            RowSelection::Indices(indices) => (indices.as_ptr() as usize, indices.len()),
        }
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

/// Iterator over local borrowed event views in a [`Dataset`].
pub(crate) struct DatasetViewIter<'a> {
    dataset: &'a Dataset,
    index: usize,
}

impl<'a> Iterator for DatasetViewIter<'a> {
    type Item = Event<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.n_events_local() {
            return None;
        }
        let event = self.dataset.event_view(self.index);
        self.index += 1;
        Some(event)
    }
}

/// Iterator over global owned events in a [`Dataset`].
pub struct DatasetGlobalIter<'a> {
    dataset: &'a Dataset,
    index: usize,
    total: usize,
    #[cfg(feature = "mpi")]
    world: Option<SimpleCommunicator>,
    #[cfg(feature = "mpi")]
    cursor: Option<MpiEventChunkCursor>,
    #[cfg(feature = "mpi")]
    layout: Option<MpiDatasetLayout>,
}

impl Iterator for DatasetGlobalIter<'_> {
    type Item = OwnedEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let index = self.index;
        self.index += 1;

        #[cfg(feature = "mpi")]
        {
            if let (Some(world), Some(cursor), Some(layout)) =
                (&self.world, &mut self.cursor, self.layout)
            {
                return cursor
                    .event_for_dataset(self.dataset, index, world, self.total, layout)
                    .ok()
                    .flatten();
            }
        }

        self.dataset.event_global_opt(index).ok().flatten()
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
    type Item = OwnedEvent;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DatasetArcIter::Local { dataset, index } => {
                let event = (*index < dataset.n_events_local()).then(|| {
                    OwnedEvent::new(
                        dataset.event_data_arc_local(*index),
                        dataset.metadata.clone(),
                    )
                });
                *index += 1;
                event
            }
            #[cfg(feature = "mpi")]
            DatasetArcIter::Mpi(iter) => iter.next(),
        }
    }
}

#[cfg(feature = "mpi")]
#[derive(Debug, Clone)]
pub(crate) struct MpiEventChunkCursor {
    chunk_start: usize,
    chunk_size: usize,
    cached_events: Vec<OwnedEvent>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum ColumnMutationKind {
    P4,
    Aux,
}

#[cfg(feature = "mpi")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ColumnMutationStatus {
    kind: ColumnMutationKind,
    name: String,
    len_ok: bool,
    duplicate: bool,
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
            cached_events: Vec::new(),
        }
    }

    fn chunk_end(&self) -> usize {
        self.chunk_start + self.cached_events.len()
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
        layout: MpiDatasetLayout,
    ) -> LadduResult<Option<OwnedEvent>> {
        if global_index >= total {
            return Ok(None);
        }
        if !self.contains(global_index) {
            self.chunk_start = global_index;
            self.cached_events = dataset.fetch_event_chunk_mpi(
                global_index,
                self.chunk_size,
                world,
                total,
                layout,
            )?;
        }
        Ok(self
            .cached_events
            .get(global_index - self.chunk_start)
            .cloned())
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
    layout: MpiDatasetLayout,
}

#[cfg(feature = "mpi")]
impl Iterator for DatasetArcMpiIter {
    type Item = OwnedEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self
            .cursor
            .event_for_dataset(
                &self.dataset,
                self.index,
                &self.world,
                self.total,
                self.layout,
            )
            .ok()
            .flatten();
        self.index += 1;
        event
    }
}

impl Dataset {
    #[cfg(feature = "mpi")]
    fn validate_global_column_add(
        &self,
        kind: ColumnMutationKind,
        name: &str,
        len_ok: bool,
    ) -> LadduResult<()> {
        let Some(world) = crate::mpi::get_world() else {
            return Ok(());
        };
        let duplicate = match kind {
            ColumnMutationKind::P4 => self.metadata.ensure_new_p4_name(name).is_err(),
            ColumnMutationKind::Aux => self.metadata.ensure_new_aux_name(name).is_err(),
        };
        let local_status = ColumnMutationStatus {
            kind,
            name: name.to_string(),
            len_ok,
            duplicate,
        };
        let serialized = bitcode::serialize(&local_status)?;
        let local_byte_count = serialized.len() as i32;
        let mut byte_counts = vec![0_i32; world.size() as usize];
        world.all_gather_into(&local_byte_count, &mut byte_counts);
        let mut byte_displs = vec![0_i32; byte_counts.len()];
        for index in 1..byte_displs.len() {
            byte_displs[index] = byte_displs[index - 1] + byte_counts[index - 1];
        }
        let gathered_bytes = world.all_gather_with_counts(&serialized, &byte_counts, &byte_displs);
        let mut statuses = Vec::with_capacity(world.size() as usize);
        for rank in 0..world.size() as usize {
            let start = byte_displs[rank] as usize;
            let end = start + byte_counts[rank] as usize;
            statuses.push(bitcode::deserialize::<ColumnMutationStatus>(
                &gathered_bytes[start..end],
            )?);
        }
        for (rank, status) in statuses.iter().enumerate() {
            if status.kind != kind {
                return Err(LadduError::Custom(format!(
                    "Collective dataset column add mismatch: rank {rank} used {:?}, expected {:?}",
                    status.kind, kind
                )));
            }
            if status.name != name {
                return Err(LadduError::Custom(format!(
                    "Collective dataset column add mismatch: rank {rank} used name '{}', expected '{name}'",
                    status.name
                )));
            }
            if !status.len_ok {
                return Err(LadduError::Custom(format!(
                    "Collective dataset column add mismatch: rank {rank} provided a column with the wrong local length"
                )));
            }
            if status.duplicate {
                let category = match kind {
                    ColumnMutationKind::P4 => "p4",
                    ColumnMutationKind::Aux => "aux",
                };
                return Err(LadduError::DuplicateName {
                    category,
                    name: name.to_string(),
                });
            }
        }
        Ok(())
    }

    #[cfg(feature = "mpi")]
    fn fetch_event_mpi(
        &self,
        global_index: usize,
        world: &SimpleCommunicator,
        total: usize,
    ) -> LadduResult<OwnedEvent> {
        let layout = self.mpi_layout.ok_or_else(|| {
            LadduError::Custom(
                "global MPI event fetch requires a global dataset layout".to_string(),
            )
        })?;
        let (owning_rank, local_index) =
            layout.owner_of(global_index, total, self.n_events_local(), world);
        let mut serialized_event_buffer_len: usize = 0;
        let mut serialized_event_buffer: Vec<u8> = Vec::default();
        if world.rank() == owning_rank {
            let event = self
                .columnar
                .event_data(self.rows.physical_index(local_index));
            serialized_event_buffer = bitcode::serialize(&event)?;
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
            Ok(OwnedEvent::new(
                Arc::new(
                    self.columnar
                        .event_data(self.rows.physical_index(local_index)),
                ),
                self.metadata.clone(),
            ))
        } else {
            let event: EventData = bitcode::deserialize(&serialized_event_buffer[..])?;
            Ok(OwnedEvent::new(Arc::new(event), self.metadata.clone()))
        }
    }

    #[cfg(feature = "mpi")]
    pub(crate) fn fetch_event_chunk_mpi(
        &self,
        start: usize,
        len: usize,
        world: &SimpleCommunicator,
        total: usize,
        layout: MpiDatasetLayout,
    ) -> LadduResult<Vec<OwnedEvent>> {
        if len == 0 || start >= total {
            return Ok(Vec::new());
        }

        let end = (start + len).min(total);
        let local_indices =
            layout.local_indices_for_range(start, end, total, self.n_events_local(), world);

        let local_events: Vec<EventData> = local_indices
            .into_iter()
            .map(|local_index| {
                self.columnar
                    .event_data(self.rows.physical_index(local_index))
            })
            .collect();
        let local_event_count = local_events.len() as i32;

        let serialized_local = if local_events.is_empty() {
            Vec::new()
        } else {
            bitcode::serialize(&local_events)?
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

        let mut events_by_rank = vec![Vec::new(); world.size() as usize];
        for rank in 0..world.size() as usize {
            if gathered_event_counts[rank] == 0 {
                continue;
            }
            let byte_start = gathered_byte_displs[rank] as usize;
            let byte_end = byte_start + gathered_byte_counts[rank] as usize;
            let decoded: Vec<EventData> =
                bitcode::deserialize(&gathered_bytes[byte_start..byte_end])?;
            debug_assert_eq!(decoded.len(), gathered_event_counts[rank] as usize);
            events_by_rank[rank] = decoded
                .into_iter()
                .map(|event| OwnedEvent::new(Arc::new(event), self.metadata.clone()))
                .collect();
        }

        let mut offsets = vec![0usize; world.size() as usize];
        let mut events = Vec::with_capacity(end - start);
        for global_index in start..end {
            let (owning_rank, _) =
                layout.owner_of(global_index, total, self.n_events_local(), world);
            let rank = owning_rank as usize;
            let offset = offsets[rank];
            events.push(events_by_rank[rank][offset].clone());
            offsets[rank] += 1;
        }
        Ok(events)
    }

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

    fn columnar_from_events(
        events: &[Arc<EventData>],
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
        let columnar = Self::columnar_from_events(&events, metadata.clone())
            .expect("Dataset requires rectangular p4/aux columns for canonical columnar storage");
        Self::from_columnar_storage(columnar, metadata, RowSelection::Identity)
    }

    /// Create an empty local dataset with explicit metadata.
    ///
    /// The returned dataset is valid immediately and can be extended with
    /// [`Dataset::push_event_local`] or [`Dataset::push_event_named_local`].
    ///
    /// Under MPI, pushed rows are stored only on the rank that performs the
    /// push. Use [`Dataset::push_event_global`] for collective single-copy
    /// appends.
    pub fn empty_local(metadata: DatasetMetadata) -> Self {
        let metadata = Arc::new(metadata);
        #[cfg(feature = "mpi")]
        {
            if crate::mpi::get_world().is_some() {
                let dataset = Dataset {
                    columnar: Arc::new(DatasetStorage::empty_with_capacity(metadata.clone(), 0)),
                    rows: RowSelection::Identity,
                    metadata,
                    cached_local_weighted_sum: 0.0,
                    cached_global_event_count: 0,
                    cached_global_weighted_sum: 0.0,
                    mpi_layout: None,
                };
                return dataset;
            }
        }
        Dataset {
            columnar: Arc::new(DatasetStorage::empty_with_capacity(metadata.clone(), 0)),
            rows: RowSelection::Identity,
            metadata,
            cached_local_weighted_sum: 0.0,
            #[cfg(feature = "mpi")]
            cached_global_event_count: 0,
            #[cfg(feature = "mpi")]
            cached_global_weighted_sum: 0.0,
            #[cfg(feature = "mpi")]
            mpi_layout: None,
        }
    }

    /// Create a local dataset from ordered four-momentum columns, auxiliary columns, and weights.
    ///
    /// `p4_columns` and `aux_columns` must be ordered to match the supplied metadata. Each
    /// column must have the same length as `weights`.
    pub fn from_columns_local(
        metadata: DatasetMetadata,
        p4_columns: Vec<Vec<Vec4>>,
        aux_columns: Vec<Vec<f64>>,
        weights: Vec<f64>,
    ) -> LadduResult<Self> {
        let n_events = weights.len();
        if p4_columns.len() != metadata.p4_names().len() {
            return Err(LadduError::Custom(format!(
                "Expected {} p4 columns, got {}",
                metadata.p4_names().len(),
                p4_columns.len()
            )));
        }
        if aux_columns.len() != metadata.aux_names().len() {
            return Err(LadduError::Custom(format!(
                "Expected {} aux columns, got {}",
                metadata.aux_names().len(),
                aux_columns.len()
            )));
        }
        for (index, column) in p4_columns.iter().enumerate() {
            if column.len() != n_events {
                return Err(LadduError::Custom(format!(
                    "P4 column {index} length {} does not match weight length {n_events}",
                    column.len()
                )));
            }
        }
        for (index, column) in aux_columns.iter().enumerate() {
            if column.len() != n_events {
                return Err(LadduError::Custom(format!(
                    "Aux column {index} length {} does not match weight length {n_events}",
                    column.len()
                )));
            }
        }

        let events = (0..n_events)
            .map(|event_index| {
                Arc::new(EventData {
                    p4s: p4_columns
                        .iter()
                        .map(|column| column[event_index])
                        .collect(),
                    aux: aux_columns
                        .iter()
                        .map(|column| column[event_index])
                        .collect(),
                    weight: weights[event_index],
                })
            })
            .collect();
        Ok(Dataset::new_local(events, Arc::new(metadata)))
    }

    /// Create a global dataset from ordered columns.
    ///
    /// Under MPI, every rank must pass the same global columns. The rows are
    /// partitioned across ranks using laddu's canonical contiguous partition.
    pub fn from_columns_global(
        metadata: DatasetMetadata,
        p4_columns: Vec<Vec<Vec4>>,
        aux_columns: Vec<Vec<f64>>,
        weights: Vec<f64>,
    ) -> LadduResult<Self> {
        let dataset = Self::from_columns_local(metadata, p4_columns, aux_columns, weights)?;
        let events = dataset.local_event_data_arcs();
        Ok(Dataset::new_with_metadata(events, dataset.metadata))
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
        let local = &partitions[world.rank() as usize];
        let columnar = Self::columnar_from_events(local, metadata.clone())
            .expect("Dataset requires rectangular p4/aux columns for canonical columnar storage");
        let local_weighted_sum = local_weighted_sum(&columnar.weights);
        let mut dataset = Dataset {
            columnar: Arc::new(columnar),
            rows: RowSelection::Identity,
            metadata,
            cached_local_weighted_sum: local_weighted_sum,
            cached_global_event_count: 0,
            cached_global_weighted_sum: local_weighted_sum,
            mpi_layout: Some(MpiDatasetLayout::Canonical),
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

    fn push_event_data_local(&mut self, event_data: Arc<EventData>) -> LadduResult<()> {
        self.ensure_mutable_storage("push events")?;
        Arc::make_mut(&mut self.columnar).push_event_data(&event_data);
        self.refresh_local_weight_cache();
        Ok(())
    }

    fn replace_metadata(&mut self, metadata: DatasetMetadata) {
        let metadata = Arc::new(metadata);
        self.metadata = metadata.clone();
        Arc::make_mut(&mut self.columnar).set_metadata(metadata);
    }

    fn validate_p4_column_len(&self, name: &str, len: usize) -> LadduResult<()> {
        if len != self.n_events_local() {
            return Err(LadduError::LengthMismatch {
                context: format!("P4 column '{name}'"),
                expected: self.n_events_local(),
                actual: len,
            });
        }
        Ok(())
    }

    fn validate_aux_column_len(&self, name: &str, len: usize) -> LadduResult<()> {
        if len != self.n_events_local() {
            return Err(LadduError::LengthMismatch {
                context: format!("Aux column '{name}'"),
                expected: self.n_events_local(),
                actual: len,
            });
        }
        Ok(())
    }

    fn add_p4_column_unchecked(&mut self, name: String, values: Vec<Vec4>) -> LadduResult<()> {
        let mut metadata = (*self.metadata).clone();
        metadata.add_p4_name(name)?;
        Arc::make_mut(&mut self.columnar).push_p4_column(values);
        self.replace_metadata(metadata);
        Ok(())
    }

    fn add_aux_column_unchecked(&mut self, name: String, values: Vec<f64>) -> LadduResult<()> {
        let mut metadata = (*self.metadata).clone();
        metadata.add_aux_name(name)?;
        Arc::make_mut(&mut self.columnar).push_aux_column(values);
        self.replace_metadata(metadata);
        Ok(())
    }

    /// Add a four-momentum column to the current rank only.
    ///
    /// This method is non-collective. Under MPI it is only valid for datasets
    /// without an MPI layout; use [`Dataset::add_p4_column_global`] for shared
    /// MPI datasets.
    pub fn add_p4_column_local<N, V>(&mut self, name: N, values: V) -> LadduResult<()>
    where
        N: Into<String>,
        V: IntoIterator<Item = Vec4>,
    {
        self.ensure_mutable_storage("add a p4 column")?;
        #[cfg(feature = "mpi")]
        {
            if self.mpi_layout.is_some() {
                return Err(LadduError::Custom(
                    "Cannot add a local p4 column to an MPI dataset; use add_p4_column_global"
                        .to_string(),
                ));
            }
        }
        let name = name.into();
        let values = values.into_iter().collect::<Vec<_>>();
        self.metadata.ensure_new_p4_name(&name)?;
        self.validate_p4_column_len(&name, values.len())?;
        self.add_p4_column_unchecked(name, values)
    }

    /// Add an auxiliary scalar column to the current rank only.
    ///
    /// This method is non-collective. Under MPI it is only valid for datasets
    /// without an MPI layout; use [`Dataset::add_aux_column_global`] for shared
    /// MPI datasets.
    pub fn add_aux_column_local<N, V>(&mut self, name: N, values: V) -> LadduResult<()>
    where
        N: Into<String>,
        V: IntoIterator<Item = f64>,
    {
        self.ensure_mutable_storage("add an aux column")?;
        #[cfg(feature = "mpi")]
        {
            if self.mpi_layout.is_some() {
                return Err(LadduError::Custom(
                    "Cannot add a local aux column to an MPI dataset; use add_aux_column_global"
                        .to_string(),
                ));
            }
        }
        let name = name.into();
        let values = values.into_iter().collect::<Vec<_>>();
        self.metadata.ensure_new_aux_name(&name)?;
        self.validate_aux_column_len(&name, values.len())?;
        self.add_aux_column_unchecked(name, values)
    }

    /// Add a four-momentum column collectively across all MPI ranks.
    ///
    /// Under MPI, every rank must call this method in the same order with the
    /// same column name. Each rank supplies values for its local events only.
    pub fn add_p4_column_global<N, V>(&mut self, name: N, values: V) -> LadduResult<()>
    where
        N: Into<String>,
        V: IntoIterator<Item = Vec4>,
    {
        self.ensure_mutable_storage("add a p4 column")?;
        let name = name.into();
        let values = values.into_iter().collect::<Vec<_>>();
        #[cfg(feature = "mpi")]
        {
            if crate::mpi::get_world().is_some() {
                self.validate_global_column_add(
                    ColumnMutationKind::P4,
                    &name,
                    values.len() == self.n_events_local(),
                )?;
                self.metadata.ensure_new_p4_name(&name)?;
                self.validate_p4_column_len(&name, values.len())?;
                return self.add_p4_column_unchecked(name, values);
            }
        }
        self.add_p4_column_local(name, values)
    }

    /// Add an auxiliary scalar column collectively across all MPI ranks.
    ///
    /// Under MPI, every rank must call this method in the same order with the
    /// same column name. Each rank supplies values for its local events only.
    pub fn add_aux_column_global<N, V>(&mut self, name: N, values: V) -> LadduResult<()>
    where
        N: Into<String>,
        V: IntoIterator<Item = f64>,
    {
        self.ensure_mutable_storage("add an aux column")?;
        let name = name.into();
        let values = values.into_iter().collect::<Vec<_>>();
        #[cfg(feature = "mpi")]
        {
            if crate::mpi::get_world().is_some() {
                self.validate_global_column_add(
                    ColumnMutationKind::Aux,
                    &name,
                    values.len() == self.n_events_local(),
                )?;
                self.metadata.ensure_new_aux_name(&name)?;
                self.validate_aux_column_len(&name, values.len())?;
                return self.add_aux_column_unchecked(name, values);
            }
        }
        self.add_aux_column_local(name, values)
    }

    /// Append one ordered event row to the current rank.
    ///
    /// `p4s` and `aux` must be ordered to match [`Dataset::p4_names`] and
    /// [`Dataset::aux_names`].
    ///
    /// Under MPI, this method performs no communication beyond refreshing
    /// cached global counts. Calling it on every rank appends one row per rank.
    pub fn push_event_local<P, A>(&mut self, p4s: P, aux: A, weight: f64) -> LadduResult<()>
    where
        P: IntoIterator<Item = Vec4>,
        A: IntoIterator<Item = f64>,
    {
        self.ensure_mutable_storage("push events")?;
        #[cfg(feature = "mpi")]
        {
            if self.mpi_layout == Some(MpiDatasetLayout::RoundRobin) && self.n_events() > 0 {
                return Err(LadduError::Custom(
                    "Cannot push local events into a round-robin global dataset".to_string(),
                ));
            }
            self.mpi_layout = None;
        }
        let p4s = p4s.into_iter().collect::<Vec<_>>();
        let aux = aux.into_iter().collect::<Vec<_>>();
        if p4s.len() != self.metadata.p4_names().len() {
            return Err(LadduError::Custom(format!(
                "Expected {} p4 values, got {}",
                self.metadata.p4_names().len(),
                p4s.len()
            )));
        }
        if aux.len() != self.metadata.aux_names().len() {
            return Err(LadduError::Custom(format!(
                "Expected {} aux values, got {}",
                self.metadata.aux_names().len(),
                aux.len()
            )));
        }

        let event_data = Arc::new(EventData { p4s, aux, weight });
        self.push_event_data_local(event_data)
    }

    /// Append one ordered event row collectively as a single global event.
    ///
    /// Under MPI, this method is collective. Exactly one rank stores the event,
    /// selected by `next_global_index % n_ranks`; non-owning ranks ignore their
    /// supplied row values. All ranks must call this method in the same order.
    pub fn push_event_global<P, A>(&mut self, p4s: P, aux: A, weight: f64) -> LadduResult<()>
    where
        P: IntoIterator<Item = Vec4>,
        A: IntoIterator<Item = f64>,
    {
        self.ensure_mutable_storage("push events")?;
        let p4s = p4s.into_iter().collect::<Vec<_>>();
        let aux = aux.into_iter().collect::<Vec<_>>();
        if p4s.len() != self.metadata.p4_names().len() {
            return Err(LadduError::Custom(format!(
                "Expected {} p4 values, got {}",
                self.metadata.p4_names().len(),
                p4s.len()
            )));
        }
        if aux.len() != self.metadata.aux_names().len() {
            return Err(LadduError::Custom(format!(
                "Expected {} aux values, got {}",
                self.metadata.aux_names().len(),
                aux.len()
            )));
        }

        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                if self.mpi_layout != Some(MpiDatasetLayout::RoundRobin) && self.n_events() > 0 {
                    return Err(LadduError::Custom(
                        "Cannot push round-robin global events into a non-empty local/canonical dataset"
                            .to_string(),
                    ));
                }
                self.mpi_layout = Some(MpiDatasetLayout::RoundRobin);
                let global_index = self.n_events();
                if global_index % world.size() as usize == world.rank() as usize {
                    self.push_event_data_local(Arc::new(EventData { p4s, aux, weight }))?;
                } else {
                    self.refresh_local_weight_cache();
                }
                return Ok(());
            }
        }

        self.push_event_data_local(Arc::new(EventData { p4s, aux, weight }))
    }

    /// Append one named event row to the current rank.
    ///
    /// The supplied p4 and aux names must exactly match this dataset's metadata, regardless of
    /// order. Duplicate, missing, and unknown names are rejected.
    pub fn push_event_named_local<P, PN, A, AN>(
        &mut self,
        p4s: P,
        aux: A,
        weight: f64,
    ) -> LadduResult<()>
    where
        P: IntoIterator<Item = (PN, Vec4)>,
        PN: AsRef<str>,
        A: IntoIterator<Item = (AN, f64)>,
        AN: AsRef<str>,
    {
        let mut ordered_p4s = vec![None; self.metadata.p4_names().len()];
        for (name, p4) in p4s {
            let name = name.as_ref();
            let index = self
                .metadata
                .p4_index(name)
                .ok_or_else(|| LadduError::UnknownName {
                    category: "p4",
                    name: name.to_string(),
                })?;
            if ordered_p4s[index].replace(p4).is_some() {
                return Err(LadduError::DuplicateName {
                    category: "p4",
                    name: name.to_string(),
                });
            }
        }
        let mut ordered_aux = vec![None; self.metadata.aux_names().len()];
        for (name, value) in aux {
            let name = name.as_ref();
            let index = self
                .metadata
                .aux_index(name)
                .ok_or_else(|| LadduError::UnknownName {
                    category: "aux",
                    name: name.to_string(),
                })?;
            if ordered_aux[index].replace(value).is_some() {
                return Err(LadduError::DuplicateName {
                    category: "aux",
                    name: name.to_string(),
                });
            }
        }

        let p4s = ordered_p4s
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                value.ok_or_else(|| {
                    LadduError::Custom(format!(
                        "Missing p4 value for '{}'",
                        self.metadata.p4_names()[index]
                    ))
                })
            })
            .collect::<LadduResult<Vec<_>>>()?;
        let aux = ordered_aux
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                value.ok_or_else(|| {
                    LadduError::Custom(format!(
                        "Missing aux value for '{}'",
                        self.metadata.aux_names()[index]
                    ))
                })
            })
            .collect::<LadduResult<Vec<_>>>()?;

        self.push_event_local(p4s, aux, weight)
    }

    /// Append one named event row collectively as a single global event.
    ///
    /// Under MPI, this method is collective. Exactly one rank stores the event,
    /// selected by `next_global_index % n_ranks`; non-owning ranks ignore their
    /// supplied row values. All ranks must call this method in the same order.
    pub fn push_event_named_global<P, PN, A, AN>(
        &mut self,
        p4s: P,
        aux: A,
        weight: f64,
    ) -> LadduResult<()>
    where
        P: IntoIterator<Item = (PN, Vec4)>,
        PN: AsRef<str>,
        A: IntoIterator<Item = (AN, f64)>,
        AN: AsRef<str>,
    {
        let mut ordered_p4s = vec![None; self.metadata.p4_names().len()];
        for (name, p4) in p4s {
            let name = name.as_ref();
            let index = self
                .metadata
                .p4_index(name)
                .ok_or_else(|| LadduError::UnknownName {
                    category: "p4",
                    name: name.to_string(),
                })?;
            if ordered_p4s[index].replace(p4).is_some() {
                return Err(LadduError::DuplicateName {
                    category: "p4",
                    name: name.to_string(),
                });
            }
        }
        let mut ordered_aux = vec![None; self.metadata.aux_names().len()];
        for (name, value) in aux {
            let name = name.as_ref();
            let index = self
                .metadata
                .aux_index(name)
                .ok_or_else(|| LadduError::UnknownName {
                    category: "aux",
                    name: name.to_string(),
                })?;
            if ordered_aux[index].replace(value).is_some() {
                return Err(LadduError::DuplicateName {
                    category: "aux",
                    name: name.to_string(),
                });
            }
        }

        let p4s = ordered_p4s
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                value.ok_or_else(|| {
                    LadduError::Custom(format!(
                        "Missing p4 value for '{}'",
                        self.metadata.p4_names()[index]
                    ))
                })
            })
            .collect::<LadduResult<Vec<_>>>()?;
        let aux = ordered_aux
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                value.ok_or_else(|| {
                    LadduError::Custom(format!(
                        "Missing aux value for '{}'",
                        self.metadata.aux_names()[index]
                    ))
                })
            })
            .collect::<LadduResult<Vec<_>>>()?;

        self.push_event_global(p4s, aux, weight)
    }

    /// The number of [`EventData`]s in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events`] instead.
    pub fn n_events_local(&self) -> usize {
        self.rows.len(self.columnar.n_events())
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
            if self.mpi_layout.is_some() {
                if let Some(world) = crate::mpi::get_world() {
                    return self.n_events_mpi(&world);
                }
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
        match &self.rows {
            RowSelection::Identity => self.columnar.weights.clone(),
            RowSelection::Indices(indices) => indices
                .iter()
                .map(|index| self.columnar.weight(*index))
                .collect(),
        }
    }

    /// Extract a list of weights over each [`EventData`] in the [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::weights`] instead.
    #[cfg(feature = "mpi")]
    pub fn weights_mpi(&self, world: &SimpleCommunicator) -> Vec<f64> {
        if matches!(
            self.mpi_layout,
            Some(MpiDatasetLayout::RoundRobin | MpiDatasetLayout::Derived)
        ) {
            return self.events_global().map(|event| event.weight()).collect();
        }
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
            if self.mpi_layout.is_some() {
                if let Some(world) = crate::mpi::get_world() {
                    return self.weights_mpi(&world);
                }
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
            if self.mpi_layout.is_some() {
                if let Some(world) = crate::mpi::get_world() {
                    return self.n_events_weighted_mpi(&world);
                }
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
        let n_events = self.n_events_local();
        let mut indices: Vec<usize> = (0..n_events)
            .map(|_| rng.usize(0..n_events))
            .collect::<Vec<usize>>();
        indices.sort();
        self.indexed_local_view(
            indices
                .into_iter()
                .map(|index| self.rows.physical_index(index)),
        )
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
        self.indexed_local_view(
            local_indices
                .into_iter()
                .map(|index| self.rows.physical_index(index)),
        )
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
        let filtered_indices: Vec<usize> = (0..self.n_events_local())
            .into_par_iter()
            .filter_map(|event_index| {
                let event = self.event_view(event_index);
                compiled
                    .evaluate(&event)
                    .then(|| self.rows.physical_index(event_index))
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let filtered_indices: Vec<usize> = (0..self.n_events_local())
            .into_iter()
            .filter_map(|event_index| {
                let event = self.event_view(event_index);
                compiled
                    .evaluate(&event)
                    .then(|| self.rows.physical_index(event_index))
            })
            .collect();
        Ok(self.indexed_local_view(filtered_indices))
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
        let evaluated: Vec<(usize, usize)> = (0..self.n_events_local())
            .into_par_iter()
            .filter_map(|event| {
                let value = variable.value(&self.event_view(event));
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, self.rows.physical_index(event)))
                } else {
                    None
                }
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let evaluated: Vec<(usize, usize)> = (0..self.n_events_local())
            .into_iter()
            .filter_map(|event| {
                let value = variable.value(&self.event_view(event));
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, self.rows.physical_index(event)))
                } else {
                    None
                }
            })
            .collect();
        let mut binned_indices: Vec<Vec<usize>> = vec![Vec::default(); bins];
        for (bin_index, index) in evaluated {
            binned_indices[bin_index].push(index);
        }
        #[cfg(feature = "rayon")]
        let datasets: Vec<Arc<Dataset>> = binned_indices
            .into_par_iter()
            .map(|indices| self.indexed_local_view(indices))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let datasets: Vec<Arc<Dataset>> = binned_indices
            .into_iter()
            .map(|indices| self.indexed_local_view(indices))
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
            .local_event_data_arcs()
            .into_par_iter()
            .map(|event| Arc::new(event.boost_to_rest_frame_of(&indices)))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let boosted_events: Vec<Arc<EventData>> = self
            .local_event_data_arcs()
            .into_iter()
            .map(|event| Arc::new(event.boost_to_rest_frame_of(&indices)))
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
        .local_event_data_arcs()
        .into_iter()
        .chain(b.local_event_data_arcs())
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
        self.events.extend(chunk.local_event_data_arcs());
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
